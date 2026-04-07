
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
def accumulateVolume(
    volume, count,
    signal, signal_offset,
    x_starts, x_ends, x_dirs,
    z_starts, z_ends, z_dirs,
    W_out, H_out, Z,
    shifts, W_scan, H_scan,
    yi_offset,
    lut_cos, lut_sin,
):
    n_z_cycles = len(z_starts)
    n_x_cycles = len(x_starts)
    scale_x    = W_out / W_scan
 
    if n_z_cycles > 0:
        zi, zs, ze, zdir = 0, z_starts[0], z_ends[0], z_dirs[0]
    else:
        return

    for yi in range(n_x_cycles):
        xs    = x_starts[yi]
        xe    = x_ends[yi]
        x_dir = x_dirs[yi]
        x_len = xe - xs
 
 
        abs_yi = yi + yi_offset
        shift  = shifts[abs_yi] * scale_x if abs_yi < len(shifts) else 0.0
 
        y_norm = (abs_yi + 0.5) / H_scan
        y_idx  = int(y_norm * H_out)
        if y_idx >= H_out: y_idx = H_out - 1
        if y_idx < 0:      y_idx = 0
 
 
        for t_abs in range(xs, xe):
 
            while zi < n_z_cycles and t_abs >= z_ends[zi]:
                zi += 1
                if zi < n_z_cycles:
                    zs, ze, zdir = z_starts[zi], z_ends[zi], z_dirs[zi]
 
            if zi >= n_z_cycles:
                break
 
            val = max(0.0, -float(signal[t_abs - signal_offset]))
 
            z_len   = ze - zs
            z_phase = (t_abs - zs) / z_len if z_len > 0 else 0.0
            z_idx, zw = getMappingIdxWeight_lut(z_phase, zdir, Z,    0.0,   lut_cos, lut_sin)


            x_phase = (t_abs - xs) / x_len if x_len > 0 else 0.0
            x_idx, xw = getMappingIdxWeight_lut(x_phase, x_dir, W_out, shift, lut_cos, lut_sin)
 
            weight = xw * zw
            volume[z_idx, y_idx, x_idx] += val * weight
            count [z_idx, y_idx, x_idx] += weight


def make_accumulators(Z, H_out, W_out):
    """
    initialize accumulators for volume and count
    """
    volume = np.zeros((Z, H_out, W_out), dtype=np.float32)
    count  = np.zeros((Z, H_out, W_out), dtype=np.float32)
    return volume, count

def feed_chunk(volume, count, signal,
               signal_offset,
               x_chunk, z_chunk, shifts,
               H_out, W_out, Z, H_scan, W_scan,
               yi_offset):
    """
    feed in multiple cycles chunk by chunk, 
    volume and count will be accumulated across chunks
    """
    if len(x_chunk) == 0 or len(z_chunk) == 0:
        return

    x_starts = x_chunk[:, 0].astype(np.int64)
    x_ends   = x_chunk[:, 1].astype(np.int64)
    x_dirs   = x_chunk[:, 2].astype(np.int32)
    z_starts = z_chunk[:, 0]
    z_ends   = z_chunk[:, 1]
    z_dirs   = z_chunk[:, 2].astype(np.int32)

    accumulateVolume(
        volume, count, signal,
        signal_offset,
        x_starts, x_ends, x_dirs,
        z_starts, z_ends, z_dirs,
        W_out, H_out, Z, shifts, W_scan, H_scan,
        yi_offset,
        _lut_cos, _lut_sin,
    )

def finalize_COO(volume, count):
    """
    finalize the COO format from the accumulated volume and count
    """
    mask = count > 1e-6
    volume[mask] /= count[mask]
    z_c, y_c, x_c = np.where(mask)
    return z_c.astype(np.uint16), y_c.astype(np.uint16), \
           x_c.astype(np.uint16), volume[mask]