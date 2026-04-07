
import numpy as np
from numba import njit
 
# =========================
# Lookup Table
# =========================
 
LUT_SIZE = 4096
 
_lut_cos = np.empty(0, dtype=np.float32)
_lut_sin = np.empty(0, dtype=np.float32)
 
def rebuild_lut(size: int = LUT_SIZE):
    global LUT_SIZE, _lut_cos, _lut_sin
    LUT_SIZE = size
    theta      = np.linspace(0.0, np.pi, size, dtype=np.float64)
    _lut_cos   = np.cos(theta).astype(np.float32)
    _lut_sin   = np.sqrt(np.maximum(0.0, 1.0 - _lut_cos.astype(np.float64) ** 2)).astype(np.float32)
 
rebuild_lut(LUT_SIZE)
 
 
# =========================
# Numba Kernels
# =========================
 
@njit(fastmath=True, cache=True)
def getMappingIdxWeight_lut(phase, direction, max_idx, shift, lut_cos, lut_sin):
    """
    LUT mapping：use phase（0~1） to look up cos and sin values, then compute the final pixel index and weight

    phase: relative position within the half-cycle, range [0, 1]
    """
    lut_size = len(lut_cos)
 
    # phase → LUT index
    if direction == -1:
        phase = 1.0 - phase
    lut_idx = int(phase * (lut_size - 1) + 0.5)   # round
    if lut_idx >= lut_size: lut_idx = lut_size - 1
    if lut_idx < 0:         lut_idx = 0
 
    x_phys = lut_cos[lut_idx]
    weight = lut_sin[lut_idx]
 
    x_norm = (x_phys + 1.0) * 0.5
    x_pix  = x_norm * max_idx
 
    if direction == +1:
        x_pix -= shift
    else:
        x_pix += shift
 
    idx = int(x_pix + 0.5)   # round
    if idx >= max_idx: idx = max_idx - 1
    if idx < 0:        idx = 0
 
    return idx, weight
 
 
@njit(fastmath=True, cache=True)
def accumulate_xhalfcycle(
    volume, count,
    signal, signal_offset,
    xs, xe, x_dir,
    W_out,
    z_idx_map, zw_map,
    lut_cos, lut_sin,
):
    """
    Accumulate one x half-cycle into a 2D (Z × W_out) buffer.
 
    Parameters
    ----------
    volume, count : float32 (Z, W_out)  — accumulators（caller 負責 zero-init）
    signal        : int16               — PMT sliding window
    signal_offset : int                 — signal[0] 對應的絕對 sample index
    xs, xe        : int                 — x half-cycle 的絕對 [start, end)
    x_dir         : int                 — +1 or -1
    shift         : float               — x 方向 shift（pixels）
    W_out         : int                 — x 輸出解析度
    z_idx_map     : int16 (xe-xs,)      — 每個 sample 的 z index，-1 = 無 z coverage
    zw_map        : float32 (xe-xs,)    — 對應的 z weight
    lut_cos/sin   : LUT arrays
    """
    x_len = xe - xs

    for i in range(x_len):
        z_idx = z_idx_map[i]
        if z_idx < 0:
            continue
        z_w = zw_map[i]

        val = max(0.0, -float(signal[xs + i - signal_offset]))

        x_phase = i / x_len
        x_idx, x_w = getMappingIdxWeight_lut(x_phase, x_dir, W_out, 0.0, lut_cos, lut_sin)

        weight = x_w * z_w
        volume[z_idx, x_idx] += val * weight
        count[z_idx, x_idx]  += weight

def precompute_z_halfcycle(zs: int, ze: int, z_dir: int, Z_out: int):
    """
    一次算完一個 z half-cycle 內所有 sample 的 z_idx 和 zw。
    在每個 z half-cycle 完成時呼叫一次。
 
    Returns
    -------
    z_idx_arr : int16   (ze - zs,)
    zw_arr    : float32 (ze - zs,)
    """
    z_len = ze - zs
    if z_len == 0:
        return np.empty(0, dtype=np.int16), np.empty(0, dtype=np.float32)
 
    j     = np.arange(z_len, dtype=np.float64)
    phase = j / z_len
    if z_dir == -1:
        phase = 1.0 - phase
 
    lut_size = len(_lut_cos)
    lut_idx  = np.clip(
        np.round(phase * (lut_size - 1)).astype(np.int32), 0, lut_size - 1)
    z_phys   = _lut_cos[lut_idx].astype(np.float64)
    zw       = _lut_sin[lut_idx].copy()
    z_norm   = (z_phys + 1.0) * 0.5
    z_idx    = np.clip(
        np.round(z_norm * Z_out).astype(np.int16), 0, Z_out - 1)
 
    return z_idx, zw

 
def build_z_map_for_xcycle(xs: int, xe: int, z_cache: list):
    """
    從 z_cache 建立一個覆蓋 [xs, xe) 的 z_idx_map 和 zw_map。
 
    z_cache : list of (zs, ze, z_dir, z_idx_arr, zw_arr)
              z_dir 只是為了向後相容保留，計算已在 precompute_z_halfcycle 完成
 
    Returns
    -------
    z_idx_map : int16   (xe-xs,)   -1 = 無 z coverage
    zw_map    : float32 (xe-xs,)
    """
    x_len     = xe - xs
    z_idx_map = np.full(x_len, -1, dtype=np.int16)
    zw_map    = np.zeros(x_len, dtype=np.float32)
 
    for (zs, ze, _z_dir, z_idx_arr, zw_arr) in z_cache:
        lo   = max(zs, xs) - xs   # local index in x half-cycle
        hi   = min(ze, xe) - xs
        z_lo = max(zs, xs) - zs   # local index in z half-cycle
        z_hi = min(ze, xe) - zs
        if lo >= hi:
            continue
        z_idx_map[lo:hi] = z_idx_arr[z_lo:z_hi]
        zw_map[lo:hi]    = zw_arr[z_lo:z_hi]
 
    return z_idx_map, zw_map


def finalize_COO(volume, count, y_idx: int):
    """
    finalize the COO format from the accumulated volume and count
    """
    mask = count > 1e-6
    volume[mask] /= count[mask]
    z_c, x_c = np.where(mask)
    y_c = np.full(len(z_c), y_idx, dtype=np.uint16)
    return (z_c.astype(np.uint16), y_c.astype(np.uint16), x_c.astype(np.uint16), volume[mask].copy())