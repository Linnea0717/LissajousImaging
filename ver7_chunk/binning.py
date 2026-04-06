import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True)
def getMappingIdxWeight(t, t_start, t_end, direction, max_idx, shift=0.0):
    """
    Get the index and weight for a time point within a half-cycle.
    Using cosine mapping.
    """
    total_len = t_end - t_start
    if total_len == 0: return 0, 0.0
    
    phase = (t - t_start) / total_len
    theta = np.pi * phase
    if direction == -1:
        theta = np.pi - theta
        
    x_phys = np.cos(theta)

    xws = np.sqrt(max(0.0, 1.0 - x_phys*x_phys))
    
    x_norm = (x_phys + 1.0) * 0.5
    x_pix = x_norm * max_idx

    if direction == +1:
        x_pix -= shift
    else:
        x_pix += shift

    xis = int(round(x_pix))
    if xis >= max_idx: xis = max_idx - 1
    if xis < 0: xis = 0
    
    return xis, xws


@njit(fastmath=True, cache=True, parallel=True)
def accumulateVolume(
    volume, count, 
    signal,
    x_starts, x_ends, x_dirs,
    z_starts, z_ends, z_dirs,
    W_out, H_out, Z,
    shifts,
    W_scan, H_scan,
    yi_offset,
):
    """
    One volume accumulation kernel.
    """

    n_z_cycles = len(z_starts)
    n_x_cycles = len(x_starts)

    scale_x = W_out / W_scan
    
    for yi in prange(n_x_cycles):
        xs = x_starts[yi]
        xe = x_ends[yi]
        x_dir = x_dirs[yi]
        
        zi, zs, ze, zdir = -1, 0, 0, 0

        shift = 0.0
        abs_yi = yi + yi_offset
        if abs_yi < len(shifts):
            shift = shifts[abs_yi]

        shift = shift * scale_x

        y_norm = (abs_yi + 0.5) / H_scan
        y_idx = int(y_norm * H_out)

        if y_idx >= H_out: y_idx = H_out - 1
        if y_idx < 0: y_idx = 0

        nzi = np.searchsorted(z_starts, xs, side='right') - 1
        if 0 <= nzi < n_z_cycles:
            zi = nzi
            zs = z_starts[zi]
            ze = z_ends[zi]
            zdir = z_dirs[zi]

        
        for t_abs in range(xs, xe):
            if t_abs >= ze:
                nzi = zi + 1
                if nzi < n_z_cycles and z_starts[nzi] <= t_abs < z_ends[nzi]:
                    zi = nzi
                    zs = z_starts[zi]
                    ze = z_ends[zi]
                    zdir = z_dirs[zi]

            if zi == -1:
                continue

            # get z index and weight
            z_idx, zw = getMappingIdxWeight(t_abs, zs, ze, zdir, Z, 0.0)
            
            # get x index and weight
            x_idx, xw = getMappingIdxWeight(t_abs, xs, xe, x_dir, W_out, shift)
            
            val = signal[t_abs]
            val = max(0.0, -val)
            
            weight = xw * zw
            
            volume[z_idx, y_idx, x_idx] += val * weight
            count[z_idx, y_idx, x_idx] += weight


def make_accumulators(Z, H_out, W_out):
    """
    initialize accumulators for volume and count
    """
    volume = np.zeros((Z, H_out, W_out), dtype=np.float32)
    count  = np.zeros((Z, H_out, W_out), dtype=np.float32)
    return volume, count

def feed_chunk(volume, count, signal,
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
        x_starts, x_ends, x_dirs,
        z_starts, z_ends, z_dirs,
        W_out, H_out, Z, shifts, W_scan, H_scan,
        yi_offset
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