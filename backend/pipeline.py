"""曝光包围 HDR 合成与去重影核心流水线。

流程：解码校验 -> 特征配准（ORB + 部分仿射）-> 重叠率检查 ->
运动区域分类（参照帧为唯一动态内容来源）-> 静态区域加权融合 ->
不确定区域标注 -> 色调映射输出 LDR 成片。

设计要点：
- 运动物体只保留参照帧位置的内容：任何与参照帧（经曝光补偿后）
  不一致的像素都判为动态，动态区域一律回退到参照帧内容。
- 参照帧在某像素过曝/欠曝裁切时，用"一致性检验"而非直接比较：
  若非参照帧补偿后的亮度与裁切方向矛盾（如参照高光裁切但其它帧
  看到明显更暗的内容），说明场景发生了变化 -> 动态，不能把其它帧
  的内容当作高光细节留下（即使多张非参照帧内容一致）。
- 饱和裁切、曝光差异、配准残差不直接当作运动：曝光按快门时间补偿、
  裁切像素不参与比较、比较在模糊 + 邻域包络下进行。
- 全组曝光都缺失或无法可靠判断的区域：保留参照内容并标为不确定。
"""

from __future__ import annotations

import numpy as np
import cv2


class PipelineError(Exception):
    """用户可读的流水线错误（由 API 层映射为 HTTP 400）。"""


# ---- 输入约束 ----
MIN_IMAGES = 3
MAX_IMAGES = 7

# ---- 裁切判定（8-bit 域）----
CLIP_LO = 3
CLIP_HI = 252

# ---- 配准 ----
MIN_OVERLAP = 0.5      # 公共重叠区域最低占比，不足则不输出成片
MAX_DIM = 3000         # 处理分辨率上限（超出则整体降采样）
ALIGN_DIM = 1600       # 特征配准用的降采样分辨率
MIN_MATCHES = 10       # 有效特征匹配/内点下限

# ---- 运动判定阈值（线性辐射度域，已补偿到参照曝光）----
REL_DIFF_TH = 0.25     # 双帧均可信时的相对差异阈值
ABS_DIFF_TH = 0.05     # 绝对差异下限，抑制暗部噪声
HI_CONSIST_TH = 0.75   # 参照高光裁切时，补偿帧亮度低于该值视为不一致（动态）
LO_CONSIST_TH = 0.02   # 参照暗部裁切时，补偿帧亮度高于该值视为不一致（动态）
BLUR_SIGMA = 1.5       # 比较前的高斯模糊，吸收亚像素配准残差


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def srgb_to_linear(u: np.ndarray) -> np.ndarray:
    """sRGB 电光转换函数（EOTF），输入 [0,1]，输出线性 [0,1]。"""
    return np.where(u <= 0.04045, u / 12.92, ((u + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """线性 -> sRGB，输入裁剪到 [0,1]。"""
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92,
                    1.055 * np.power(lin, 1.0 / 2.4) - 0.055)


def _to_linear(img8: np.ndarray) -> np.ndarray:
    return srgb_to_linear(img8.astype(np.float32) / 255.0)


# ---------------------------------------------------------------------------
# 输入校验与解码
# ---------------------------------------------------------------------------

def validate_times(times, n_images: int) -> list[float]:
    if len(times) != n_images:
        raise PipelineError(
            f"快门时间数量（{len(times)}）与图片数量（{n_images}）不一致")
    out = []
    for t in times:
        try:
            v = float(t)
        except (TypeError, ValueError):
            raise PipelineError(f"非法快门时间：{t!r}，必须为有限正数（秒）")
        if not np.isfinite(v) or v <= 0.0:
            raise PipelineError(f"非法快门时间：{t!r}，必须为有限正数（秒）")
        out.append(v)
    return out


def decode_images(raw_list: list[bytes]) -> list[np.ndarray]:
    if not (MIN_IMAGES <= len(raw_list) <= MAX_IMAGES):
        raise PipelineError(
            f"需要 {MIN_IMAGES}～{MAX_IMAGES} 张 JPEG 图片，收到 {len(raw_list)} 张")
    images = []
    shape = None
    for idx, data in enumerate(raw_list):
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise PipelineError(
                f"第 {idx + 1} 张图片无法解码（文件损坏或不是有效的 JPEG）")
        if shape is None:
            shape = img.shape
        elif img.shape != shape:
            raise PipelineError(
                f"图片尺寸不一致：第 1 张为 {shape[1]}×{shape[0]}，"
                f"第 {idx + 1} 张为 {img.shape[1]}×{img.shape[0]}")
        images.append(img)
    return images


# ---------------------------------------------------------------------------
# 配准
# ---------------------------------------------------------------------------

def _estimate_transform_orb(gray_src: np.ndarray, gray_ref: np.ndarray):
    """估计 src -> ref 的部分仿射变换（旋转+平移+统一缩放），失败返回 None。"""
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=12)
    k1, d1 = orb.detectAndCompute(gray_src, None)
    k2, d2 = orb.detectAndCompute(gray_ref, None)
    if d1 is None or d2 is None or len(k1) < MIN_MATCHES or len(k2) < MIN_MATCHES:
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = []
    for pair in bf.knnMatch(d1, d2, k=2):
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    if len(good) < MIN_MATCHES:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None or inliers is None or int(inliers.sum()) < MIN_MATCHES:
        return None
    return M


def _estimate_transform_ecc(gray_src: np.ndarray, gray_ref: np.ndarray):
    """ECC 兜底配准：在梯度幅值图上做仿射配准。

    梯度幅值对曝光差异近似只差一个增益因子，而 ECC 的相关性准则
    对增益/偏置变化不变，适合低纹理或特征稀少的画面。
    """
    def grad_mag(g):
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
        m = cv2.magnitude(gx, gy)
        return cv2.normalize(m, None, 0.0, 1.0, cv2.NORM_MINMAX)

    a, b = grad_mag(gray_src), grad_mag(gray_ref)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(b, a, warp, cv2.MOTION_AFFINE,
                                        criteria, None, 5)
    except cv2.error:
        return None
    if not np.isfinite(cc) or cc < 0.5:
        return None
    return warp


def _estimate_transform(gray_src, gray_ref):
    """先 ORB 特征配准，失败则 ECC 兜底，都失败返回 None。"""
    M = _estimate_transform_orb(gray_src, gray_ref)
    if M is not None:
        return M
    return _estimate_transform_ecc(gray_src, gray_ref)


def align(images: list[np.ndarray], ref_idx: int):
    """把所有帧配准到参照帧坐标系。

    返回 (aligned, valid_masks, common_mask, overlap_ratio)。
    重叠不足时抛出 PipelineError，不输出成片。
    """
    n = len(images)
    h, w = images[0].shape[:2]
    scale = min(1.0, ALIGN_DIM / max(h, w))
    if scale < 1.0:
        smalls = [cv2.resize(im, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA) for im in images]
    else:
        smalls = images
    grays = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for im in smalls:
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        grays.append(clahe.apply(g))   # 均衡局部对比度，削弱帧间曝光差异
    ref_gray = grays[ref_idx]

    aligned, masks = [], []
    full = np.full((h, w), 255, np.uint8)
    for i in range(n):
        if i == ref_idx:
            aligned.append(images[i])
            masks.append(full.copy())
            continue
        M = _estimate_transform(grays[i], ref_gray)
        if M is None:
            raise PipelineError(
                f"第 {i + 1} 张图片无法与参照帧配准（特征匹配不足）")
        # 小图坐标系的变换换算回原图坐标系：平移量除以缩放比
        M_full = M.astype(np.float64).copy()
        M_full[0, 2] /= scale
        M_full[1, 2] /= scale
        warped = cv2.warpAffine(images[i], M_full, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(full, M_full, (w, h),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        aligned.append(warped)
        masks.append(mask)

    common = np.full((h, w), 255, np.uint8)
    for m in masks:
        common = cv2.bitwise_and(common, m)
    # 侵蚀边界，去掉 warp 插值边缘
    common = cv2.erode(common, np.ones((5, 5), np.uint8))
    overlap = float(np.count_nonzero(common)) / float(h * w)
    if overlap < MIN_OVERLAP:
        raise PipelineError(
            f"配准重叠不足：有效重叠仅 {overlap * 100:.1f}%"
            f"（要求 ≥ {MIN_OVERLAP * 100:.0f}%），未输出成片")
    return aligned, masks, common, overlap


# ---------------------------------------------------------------------------
# 运动分类 + HDR 融合
# ---------------------------------------------------------------------------

def classify_and_merge(aligned, times, ref_idx, masks, common):
    """核心：逐像素分类（静态/动态）并融合。

    返回 (display, dynamic, uncertain, common_b)：
    - display   : 参照曝光域的线性 HDR 结果（float32, HxWx3）
    - dynamic   : 动态区域布尔掩码
    - uncertain : 不确定区域布尔掩码（保留参照内容）
    - common_b  : 公共重叠区域布尔掩码
    """
    n = len(aligned)
    h, w = aligned[0].shape[:2]
    t = [float(x) for x in times]
    t_ref = t[ref_idx]

    ref8 = aligned[ref_idx]
    ref_lin = _to_linear(ref8)
    ref_blur = cv2.GaussianBlur(ref_lin, (0, 0), BLUR_SIGMA)
    ref_max = ref8.max(axis=2)
    ref_min = ref8.min(axis=2)
    ref_hi = ref_max >= CLIP_HI          # 参照高光裁切
    ref_lo = ref_min <= CLIP_LO          # 参照暗部裁切
    ref_ok = ~(ref_hi | ref_lo)          # 参照完全未裁切
    common_b = common > 0

    k3 = np.ones((3, 3), np.uint8)
    # 参照的 ±1px 邻域包络：补偿帧落在包络内即视为一致，
    # 避免把轻微配准残差误判为运动。
    ref_env_hi = cv2.dilate(ref_blur, k3)
    ref_env_lo = cv2.erode(ref_blur, k3)

    dynamic = np.zeros((h, w), bool)
    for i in range(n):
        if i == ref_idx:
            continue
        img8 = aligned[i]
        lin = _to_linear(img8)
        comp = lin * (t_ref / t[i])            # 曝光补偿到参照曝光域
        comp_blur = cv2.GaussianBlur(comp, (0, 0), BLUR_SIGMA)
        img_max = img8.max(axis=2)
        img_min = img8.min(axis=2)
        img_ok = (img_max < CLIP_HI) & (img_min > CLIP_LO)  # 该帧未裁切
        in_region = common_b & (masks[i] > 0)

        # 情形一：参照与该帧均未裁切 —— 直接比较（带邻域包络容差）
        both_ok = in_region & img_ok & ref_ok
        exceed = np.maximum(comp_blur - ref_env_hi,
                            ref_env_lo - comp_blur).max(axis=2)
        bright = np.maximum(comp_blur.max(axis=2), ref_blur.max(axis=2))
        rel = exceed / np.maximum(bright, 0.02)
        dynamic |= both_ok & (exceed > ABS_DIFF_TH) & (rel > REL_DIFF_TH)

        # 情形二：参照高光裁切 —— 静态场景应在该帧也呈现高亮度；
        # 若补偿后明显偏暗，说明画面内容变了（如物体已移走），判动态。
        case_hi = in_region & img_ok & ref_hi
        dynamic |= case_hi & (comp_blur.max(axis=2) < HI_CONSIST_TH)

        # 情形三：参照暗部裁切 —— 静态场景应同样接近黑；
        # 若补偿后明显偏亮，说明内容变了，判动态。
        case_lo = in_region & img_ok & ref_lo
        dynamic |= case_lo & (comp_blur.max(axis=2) > LO_CONSIST_TH)

    # 形态学清理：去散点、闭合空洞、适度膨胀覆盖运动边缘光晕
    dynamic &= common_b
    dyn_u = dynamic.astype(np.uint8) * 255
    dyn_u = cv2.morphologyEx(dyn_u, cv2.MORPH_OPEN, k3)
    dyn_u = cv2.dilate(dyn_u, np.ones((5, 5), np.uint8), iterations=2)
    dynamic = (dyn_u > 0) & common_b

    # ---- 静态区域加权融合（Debevec 帽式权重，裁切处权重为 0）----
    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w, 3), np.float32)
    for i in range(n):
        z = aligned[i].astype(np.float32)
        wgt = np.minimum(z, 255.0 - z) / 128.0
        wgt[(z <= CLIP_LO) | (z >= CLIP_HI)] = 0.0
        wgt *= (masks[i] > 0)[..., None].astype(np.float32)
        acc += wgt * (_to_linear(aligned[i]) / t[i])
        wsum += wgt
    has_data = wsum.max(axis=2) > 1e-6
    radiance = np.where(wsum > 1e-6, acc / np.maximum(wsum, 1e-6), 0.0)

    ref_radiance = ref_lin / t_ref
    out = np.where(has_data[..., None], radiance, ref_radiance)
    out = np.where(dynamic[..., None], ref_radiance, out)   # 动态：只用参照
    out = np.where(common_b[..., None], out, ref_radiance)  # 重叠区外：参照
    display = out * t_ref                                    # 回到参照曝光域

    # ---- 不确定区域：保留参照内容并标注 ----
    ref_clipped = ref_hi | ref_lo
    uncertain = ((~common_b)                       # 非公共重叠区，只有参照可用
                 | (dynamic & ref_clipped)         # 动态且参照已裁切，无法恢复
                 | (~has_data & ~dynamic))         # 全组曝光都缺失细节
    return display.astype(np.float32), dynamic, uncertain, common_b


# ---------------------------------------------------------------------------
# 色调映射
# ---------------------------------------------------------------------------

def tonemap(display: np.ndarray, region: np.ndarray) -> np.ndarray:
    """全局 Reinhard 色调映射 + 自动关键值，输出 8-bit BGR。"""
    lum = (display[..., 2] * 0.2126 + display[..., 1] * 0.7152
           + display[..., 0] * 0.0722)
    vals = lum[region]
    med = float(np.median(vals)) if vals.size else 0.18
    key = float(np.clip(0.18 / max(med, 1e-3), 0.2, 5.0))
    L = lum * key
    Lm = L / (1.0 + L)
    scale = Lm / np.maximum(L, 1e-6)
    out = np.clip(display * key * scale[..., None], 0.0, 1.0)
    return (linear_to_srgb(out) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------

def run_pipeline(raw_list: list[bytes], times, ref_idx: int) -> dict:
    """完整流水线，返回 PNG 字节与统计信息。错误抛 PipelineError。"""
    if not (MIN_IMAGES <= len(raw_list) <= MAX_IMAGES):
        raise PipelineError(
            f"需要 {MIN_IMAGES}～{MAX_IMAGES} 张 JPEG 图片，收到 {len(raw_list)} 张")
    times = validate_times(times, len(raw_list))
    if not (0 <= ref_idx < len(raw_list)):
        raise PipelineError(f"参照帧索引越界：{ref_idx}")

    images = decode_images(raw_list)

    # 超限分辨率整体降采样，控制内存与耗时
    h, w = images[0].shape[:2]
    scale = min(1.0, MAX_DIM / max(h, w))
    if scale < 1.0:
        images = [cv2.resize(im, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA) for im in images]

    aligned, masks, common, overlap = align(images, ref_idx)
    display, dynamic, uncertain, common_b = classify_and_merge(
        aligned, times, ref_idx, masks, common)
    ldr = tonemap(display, common_b)

    result_png = cv2.imencode(".png", ldr)[1].tobytes()
    mask_png = cv2.imencode(".png", uncertain.astype(np.uint8) * 255)[1].tobytes()
    ref_png = cv2.imencode(".png", aligned[ref_idx])[1].tobytes()

    meta = {
        "width": int(ldr.shape[1]),
        "height": int(ldr.shape[0]),
        "n_images": len(images),
        "reference_index": int(ref_idx),
        "overlap": round(overlap, 4),
        "dynamic": round(float(np.count_nonzero(dynamic)) / dynamic.size, 4),
        "uncertain": round(float(np.count_nonzero(uncertain)) / uncertain.size, 4),
    }
    return {
        "result_png": result_png,
        "mask_png": mask_png,
        "reference_png": ref_png,
        "meta": meta,
    }
