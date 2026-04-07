"""
mapping.py
==========
Half-cycle → pixel index mapping。
"""

import numpy as np
from lut import lut_lookup


# =============================================================
# Z mapping
# =============================================================

def precompute_z_halfcycle(
    zs: int, ze: int, z_dir: int, Z_out: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute z_idx and zw for a z half-cycle
    """
    z_len = ze - zs
    if z_len == 0:
        return np.empty(0, dtype=np.int16), np.empty(0, dtype=np.float32)

    phase    = np.arange(z_len, dtype=np.float64) / z_len
    cos_v, sin_v = lut_lookup(phase, z_dir)

    z_norm   = (cos_v.astype(np.float64) + 1.0) * 0.5
    z_idx    = np.clip(
        np.round(z_norm * Z_out).astype(np.int16), 0, Z_out - 1)

    return z_idx, sin_v   # sin_v = zw


def build_z_map_for_xcycle(
    xs: int, xe: int,
    z_cache: list,
) -> tuple[np.ndarray, np.ndarray]:
    """
    use z_cache to build z_idx_map and zw_map for a x half-cycle
    """
    x_len     = xe - xs
    z_idx_map = np.full(x_len, -1, dtype=np.int16)
    zw_map    = np.zeros(x_len, dtype=np.float32)

    for (zs, ze, _z_dir, z_idx_arr, zw_arr) in z_cache:
        lo   = max(zs, xs) - xs   # local offset in x half-cycle
        hi   = min(ze, xe) - xs
        z_lo = max(zs, xs) - zs   # local offset in z half-cycle
        z_hi = min(ze, xe) - zs
        if lo >= hi:
            continue
        z_idx_map[lo:hi] = z_idx_arr[z_lo:z_hi]
        zw_map[lo:hi]    = zw_arr[z_lo:z_hi]

    return z_idx_map, zw_map


# =============================================================
# X mapping
# =============================================================

def precompute_x_halfcycle(
    xs: int, xe: int, x_dir: int, shift: float, W_out: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute x_idx and xw for a x half-cycle
    """
    x_len = xe - xs
    if x_len == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)

    phase    = np.arange(x_len, dtype=np.float64) / x_len
    cos_v, sin_v = lut_lookup(phase, x_dir)

    x_pix    = (cos_v.astype(np.float64) + 1.0) * 0.5 * W_out
    x_pix   -= shift if x_dir == +1 else -shift
    x_idx    = np.clip(np.round(x_pix).astype(np.int32), 0, W_out - 1)

    return x_idx, sin_v   # sin_v = xw
