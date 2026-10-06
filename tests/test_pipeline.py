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


# ---------------------------------------------------------------------------
# 逐通道裁切：某一通道在全部帧过曝，其它通道仍有有效曝光
# 旧逻辑按整像素裁决（任一通道有权重即融合），缺失通道被写成 0 -> 偏色，
# 且部分缺失的像素漏标遮罩。修复后：缺失通道保留参照值、可恢复通道保留
# 融合细节，部分缺失像素标为不确定，全通道可恢复才确定。
# ---------------------------------------------------------------------------

# 静态彩色区域（BGR，辐射度，参照曝光为单位）。
# 用大块平滑色块（边缘羽化、中心平核）避免硬边在 JPEG + 模糊后触发运动
# 分类；背景带随机纹理保证 ORB 配准。下面坐标为平核区域。
PATCH_R1 = (35, 65, 80, 150)    # R=5.0 全组裁切，B/G 正常
PATCH_R2 = (35, 65, 175, 245)   # R=1.5 参照过曝、短曝光可恢复
PATCH_R3 = (165, 195, 110, 180) # B=G=R=8.0 全组三通道裁切


def _soft_rect(shape, patch, band=16.0):
    """边缘羽化、中心恒为 1 的权重（核心区域颜色完全平坦）。"""
    y0, y1, x0, x1 = patch
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    d = np.minimum(np.minimum(xx - x0, x1 - xx),
                   np.minimum(yy - y0, y1 - y0))
    return np.clip(d / band, 0.0, 1.0)


def build_color_scene():
    """中性灰背景（带纹理便于配准）+ 三个不同裁切情况的平滑彩色块。"""
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
    tex = 0.05 * cv2.GaussianBlur(noise, (0, 0), 1.2)
    base = 0.4
    b = np.full((H, W), base, np.float32) * (1.0 + tex)
    g = b.copy()
    r = b.copy()

    def blend(patch, vb, vg, vr):
        nonlocal b, g, r
        m = _soft_rect((H, W), patch)
        b = b * (1.0 - m) + vb * m
        g = g * (1.0 - m) + vg * m
        r = r * (1.0 - m) + vr * m

    blend(PATCH_R1, 0.4, 0.25, 5.0)
    blend(PATCH_R2, 0.4, 0.25, 1.5)
    blend(PATCH_R3, 8.0, 8.0, 8.0)
    return cv2.merge([b, g, r])


def render_color(scene_bgr, ratio):
    lin = np.clip(scene_bgr * ratio, 0.0, 1.0)
    srgb = np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def make_color_frames():
    scene = build_color_scene()
    return [render_color(scene, TIMES[i] / TIMES[REF]) for i in range(3)]


def make_color_jpegs():
    return [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            for f in make_color_frames()]


def patch_center(patch):
    y0, y1, x0, x1 = patch
    return (y0 + y1) // 2, (x0 + x1) // 2


def test_per_channel_clipped_keeps_reference_color():
    frames = make_color_frames()
    masks = [np.full((H, W), 255, np.uint8) for _ in frames]
    common = np.full((H, W), 255, np.uint8)
    display, dynamic, uncertain, _ = classify_and_merge(
        frames, TIMES, REF, masks, common)

    r1 = patch_center(PATCH_R1)
    r2 = patch_center(PATCH_R2)
    r3 = patch_center(PATCH_R3)

    # 场景静态，无运动
    assert not dynamic[r1] and not dynamic[r2] and not dynamic[r3]

    # R1：R 通道全组裁切 -> 保留参照值（=1，而非旧逻辑的 0），不偏色；
    # B/G 仍为融合的真实辐射度；该像素部分缺失 -> 不确定
    b1, g1, r1v = display[r1]
    assert r1v > 0.99, f"缺失的 R 通道被写成 {r1v}（旧的零值偏色 bug）"
    assert abs(b1 - 0.4) < 0.04 and abs(g1 - 0.25) < 0.04
    assert uncertain[r1], "部分颜色通道缺失的像素必须标为不确定"

    # R2：R 通道可从短曝光恢复 -> 三通道融合细节保留，像素确定
    b2, g2, r2v = display[r2]
    assert abs(r2v - 1.5) < 0.12, "可恢复的 R 通道应保留融合细节"
    assert abs(b2 - 0.4) < 0.04 and abs(g2 - 0.25) < 0.04
    assert not uncertain[r2], "三通道均可恢复时不应标为不确定"

    # R3：三通道全组裁切 -> 保留参照（白）且不确定
    assert abs(display[r3].mean() - 1.0) < 0.02
    assert uncertain[r3]


def test_per_channel_clipped_end_to_end():
    """JPEG + 配准 + 色调映射全链路：成片不偏色、遮罩标对。"""
    result = run_pipeline(make_color_jpegs(), TIMES, REF)
    assert result["meta"]["overlap"] > 0.9
    img = cv2.imdecode(np.frombuffer(result["result_png"], np.uint8),
                       cv2.IMREAD_COLOR)
    mask = cv2.imdecode(np.frombuffer(result["mask_png"], np.uint8),
                        cv2.IMREAD_GRAYSCALE)

    r1 = patch_center(PATCH_R1)
    r2 = patch_center(PATCH_R2)
    r3 = patch_center(PATCH_R3)
    b1, g1, rc1 = (int(v) for v in img[r1])
    # 旧 bug：R 写成 0 -> 蓝/绿压倒红（R 通道成片接近 0）；
    # 修复后 R 通道保留参照高光值，经全局色调映射仍明显高于 B/G。
    assert rc1 > b1 > g1, f"R 通道缺失导致成片偏色，得到 BGR=({b1},{g1},{rc1})"
    assert rc1 > 120, f"缺失 R 通道被压成 {rc1}（旧的零值 bug）"
    assert mask[r1] == 255, "部分通道缺失像素的遮罩漏标"
    # 全通道可恢复 -> 确定；全通道裁切 -> 不确定
    assert mask[r2] == 0
    assert mask[r3] == 255
