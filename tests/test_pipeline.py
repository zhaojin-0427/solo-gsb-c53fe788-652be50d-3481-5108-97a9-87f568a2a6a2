"""合成场景端到端测试。

场景（辐射度以参照曝光为单位）：
- 竖直渐变天空：顶部 1.4（参照中过曝）到底部 0.05
- 太阳圆盘 E=1.5（参照过曝、短曝光可恢复），核心 E=8（全组都裁切）
- 暗部条纹 E=0.0006 / 0.0009（参照暗部裁切、长曝光可恢复）
- 运动物体 E=0.45：帧 0、帧 2 都在旧位置 A，参照帧在新位置 B
  —— 正是"参照在旧位置过曝、两张非参照帧拍到同一物体"的陷阱场景，
     旧物体绝不能当作高光细节留下。
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from pipeline import (  # noqa: E402
    PipelineError,
    classify_and_merge,
    run_pipeline,
)

H, W = 240, 320
TIMES = [0.25, 1.0, 2.0]   # 参照帧为索引 1（t_ref = 1.0）
REF = 1
OBJ_A = (30, 30)           # 旧位置（x, y）
OBJ_B = (130, 30)          # 新位置
OBJ_SIZE = 36
SUN = (220, 50)            # 太阳中心 (x, y)


def build_scene():
    e = (np.linspace(1.4, 0.05, H, dtype=np.float32)[:, None]
         * np.ones((1, W), np.float32))
    # 确定性纹理，模拟真实照片中的表面细节（便于特征配准）
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
    fine = cv2.GaussianBlur(noise, (0, 0), 1.0)
    coarse = cv2.GaussianBlur(noise, (0, 0), 4.0)
    tex = 0.06 * fine / max(fine.std(), 1e-6) + 0.04 * coarse / max(coarse.std(), 1e-6)
    e = e * (1.0 + tex)
    cv2.circle(e, SUN, 18, 1.5, -1)      # 太阳：参照过曝，短曝光可恢复
    cv2.circle(e, SUN, 7, 8.0, -1)       # 核心：全组裁切 -> 不确定
    for r in range(180, 220, 8):         # 暗部条纹：参照裁切，长曝光可恢复
        e[r:r + 4, 240:300] = 0.0006
        e[r + 4:r + 8, 240:300] = 0.0009
    return e


def render(e, ratio, obj_pos):
    img = e.copy()
    if obj_pos is not None:
        x, y = obj_pos
        img[y:y + OBJ_SIZE, x:x + OBJ_SIZE] = 0.45
    lin = np.clip(img * ratio, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    v = (srgb * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)


def make_frames():
    """返回 3 帧 uint8 BGR：物体在帧0/帧2 位于 A，参照帧位于 B。"""
    e = build_scene()
    positions = [OBJ_A, OBJ_B, OBJ_A]
    return [render(e, TIMES[i] / TIMES[REF], positions[i]) for i in range(3)]


def make_jpegs(shift_frame0=None):
    frames = make_frames()
    if shift_frame0 is not None:
        dx, dy = shift_frame0
        m = np.float32([[1, 0, dx], [0, 1, dy]])
        frames[0] = cv2.warpAffine(frames[0], m, (W, H), borderValue=0)
    return [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            for f in frames]


def center(pos):
    return pos[1] + OBJ_SIZE // 2, pos[0] + OBJ_SIZE // 2   # (row, col)


# ---------------------------------------------------------------------------
# 单元级：直接对无噪声合成帧做分类与融合
# ---------------------------------------------------------------------------

def test_classify_and_merge_static_and_dynamic():
    frames = make_frames()
    masks = [np.full((H, W), 255, np.uint8) for _ in frames]
    common = np.full((H, W), 255, np.uint8)
    display, dynamic, uncertain, _ = classify_and_merge(
        frames, TIMES, REF, masks, common)

    ra, ca = center(OBJ_A)
    rb, cb = center(OBJ_B)

    # 旧位置 A：参照过曝，两张非参照帧都拍到物体 —— 仍判动态、
    # 保留参照（过曝白）内容、标记不确定，物体不得留下
    assert dynamic[ra, ca]
    assert uncertain[ra, ca]
    assert display[ra, ca].mean() > 0.99, "旧物体被当作高光细节留下了"

    # 新位置 B：动态，输出参照中的物体（0.45），且确定
    assert dynamic[rb, cb]
    assert not uncertain[rb, cb]
    assert abs(display[rb, cb].mean() - 0.45) < 0.05

    # 静态过曝太阳（环带）：从短曝光恢复，确定
    sr, sc = SUN[1] + 13, SUN[0]
    assert not uncertain[sr, sc]
    assert abs(display[sr, sc].mean() - 1.5) < 0.15

    # 全组裁切的太阳核心：不确定，保留参照
    assert uncertain[SUN[1], SUN[0]]

    # 暗部条纹：从长曝光恢复，确定，条纹对比保持
    assert not uncertain[182, 270]
    assert not uncertain[186, 270]
    s1, s2 = display[182, 270].mean(), display[186, 270].mean()
    assert 1.2 < s2 / max(s1, 1e-9) < 1.9

    # 静态中间调：融合结果接近真实辐射度
    expected = build_scene()[150, 30]
    assert abs(display[150, 30].mean() - expected) < 0.03


# ---------------------------------------------------------------------------
# 集成级：走完整流水线（JPEG 编解码 + 配准 + 融合 + 色调映射）
# ---------------------------------------------------------------------------

def test_end_to_end_ghost_free():
    result = run_pipeline(make_jpegs(), TIMES, REF)
    meta = result["meta"]
    assert meta["overlap"] > 0.95

    img = cv2.imdecode(np.frombuffer(result["result_png"], np.uint8),
                       cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(result["mask_png"], np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    ra, ca = center(OBJ_A)
    rb, cb = center(OBJ_B)
    bg = img[ra, 100].mean()          # 同一行的静态背景

    # 旧位置 A 与背景一致（无重影），新位置 B 可见物体
    assert abs(img[ra, ca].mean() - bg) < 25
    assert img[rb, cb].mean() < bg - 20
    # 遮罩：A 不确定，B 确定
    assert mask[ra, ca] == 255
    assert mask[rb, cb] == 0
    # 太阳环带被恢复（不再是纯白）且确定；核心不确定
    sr, sc = SUN[1] + 13, SUN[0]
    assert mask[sr, sc] == 0
    assert 100 < img[sr, sc].mean() < 245
    assert mask[SUN[1], SUN[0]] == 255


def test_end_to_end_with_small_misalignment():
    """帧 0 平移 (3, -2) 像素：应成功配准，且不产生重影/误报。"""
    result = run_pipeline(make_jpegs(shift_frame0=(3, -2)), TIMES, REF)
    assert result["meta"]["overlap"] > 0.85
    img = cv2.imdecode(np.frombuffer(result["result_png"], np.uint8),
                       cv2.IMREAD_COLOR)
    ra, ca = center(OBJ_A)
    bg = img[ra, 100].mean()
    assert abs(img[ra, ca].mean() - bg) < 30   # 旧位置无重影
    sr, sc = SUN[1] + 13, SUN[0]
    assert 100 < img[sr, sc].mean() < 245      # 太阳仍正常恢复


def test_overlap_insufficient_raises():
    """帧 0 平移 200px（重叠 < 50%）：不输出成片。"""
    with pytest.raises(PipelineError, match="重叠|配准"):
        run_pipeline(make_jpegs(shift_frame0=(200, 0)), TIMES, REF)


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

def test_rejects_wrong_count():
    jpegs = make_jpegs()
    with pytest.raises(PipelineError, match="3～7"):
        run_pipeline(jpegs[:2], TIMES[:2], 0)
    with pytest.raises(PipelineError, match="3～7"):
        run_pipeline(jpegs * 3, TIMES * 3, 0)


def test_rejects_invalid_shutter_times():
    jpegs = make_jpegs()
    for bad in ([0.0, 1.0, 2.0], [-1.0, 1.0, 2.0],
                [float("nan"), 1.0, 2.0], [float("inf"), 1.0, 2.0],
                ["abc", 1.0, 2.0]):
        with pytest.raises(PipelineError, match="快门"):
            run_pipeline(jpegs, bad, REF)
    with pytest.raises(PipelineError, match="数量"):
        run_pipeline(jpegs, [1.0, 2.0], REF)


def test_rejects_corrupt_and_mismatched_images():
    jpegs = make_jpegs()
    with pytest.raises(PipelineError, match="解码|损坏"):
        run_pipeline([b"not-a-jpeg", jpegs[1], jpegs[2]], TIMES, REF)
    small = cv2.imencode(".jpg", np.zeros((100, 100, 3), np.uint8))[1].tobytes()
    with pytest.raises(PipelineError, match="尺寸"):
        run_pipeline([jpegs[0], jpegs[1], small], TIMES, REF)


def test_rejects_bad_reference_index():
    with pytest.raises(PipelineError, match="参照"):
        run_pipeline(make_jpegs(), TIMES, 5)
