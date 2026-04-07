"""
accumulate.py
=============
accumulate 一個 x half-cycle 內的 sample，輸出 COO。
"""

import numpy as np


def accumulate_xcycle_COO(
    z_idx_map: np.ndarray,   # int16  (x_len,)  -1 = no z coverage
    zw_map: np.ndarray,      # float32 (x_len,)
    x_idx_arr: np.ndarray,   # int32  (x_len,)
    xw_arr: np.ndarray,      # float32 (x_len,)
    signal: np.ndarray,      # int16  PMT sliding window
    signal_offset: int,      # signal[0] absolute sample index
    xs: int,                 # x half-cycle start absolute sample index
    W_out: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    accumulate a x half-cycle into COO format
    """
    # ── filter valid sample（with z coverage）─────────────────────────────
    valid = z_idx_map >= 0
    if not valid.any():
        return (np.empty(0, dtype=np.uint16),
                np.empty(0, dtype=np.uint16),
                np.empty(0, dtype=np.float32))

    j = np.where(valid)[0]

    z_v    = z_idx_map[j].astype(np.int32)
    zw_v   = zw_map[j]
    x_v    = x_idx_arr[j]
    xw_v   = xw_arr[j]

    # ── PMT signal ────────────────────────────────────────────────────────
    val_v  = np.maximum(0.0, -signal[xs + j - signal_offset].astype(np.float32))

    # ── weighted values ───────────────────────────────────────────────────
    w_v    = xw_v * zw_v
    wval_v = val_v * w_v

    # ── linear index & weighted sum ───────────────────────────────────────
    lin    = z_v * W_out + x_v

    order  = np.argsort(lin, kind='stable')
    lin_s  = lin[order]
    wval_s = wval_v[order]
    w_s    = w_v[order]

    uniq, first = np.unique(lin_s, return_index=True)
    val_sum = np.add.reduceat(wval_s, first)
    w_sum   = np.add.reduceat(w_s,   first)

    # ── weighted average ──────────────────────────────────────────────────
    good    = w_sum > 1e-6
    z_c     = (uniq[good] // W_out).astype(np.uint16)
    x_c     = (uniq[good] %  W_out).astype(np.uint16)
    vals    = (val_sum[good] / w_sum[good]).astype(np.float32)

    return z_c, x_c, vals
