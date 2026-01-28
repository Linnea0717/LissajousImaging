import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def getMappingIdxWeight(t, t_start, t_end, direction, max_idx):
    """
    Get the index and weight for time points within a half-cycle.
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
    xis = int(x_norm * max_idx)
    if xis >= max_idx: xis = max_idx - 1
    if xis < 0: xis = 0
    
    return xis, xws

@jit(nopython=True, parallel=False, cache=True)
def accumulateFrame(
    volume, count, 
    signal,
    x_starts, x_ends, x_dirs,
    z_starts, z_ends, z_dirs,
    frame_t_start, frame_t_end,
    W, H, Z
):
    """
    One frame accumulation kernel.
    """
    t_len = frame_t_end - frame_t_start
    z_idx_table = np.full(t_len, -1, dtype=np.int32)
    z_weight_table = np.zeros(t_len, dtype=np.float32)

    n_z_cycles = len(z_starts)
    for i in range(n_z_cycles):
        zs = max(z_starts[i], frame_t_start)
        ze = min(z_ends[i], frame_t_end)
        zdir = z_dirs[i]

        for t in range(zs, ze):
            zi, zw = getMappingIdxWeight(t, zs, ze, zdir, Z)
            t_rel = t - frame_t_start
            z_idx_table[t_rel] = zi
            z_weight_table[t_rel] = zw

    n_x_cycles = len(x_starts)
    
    for yi in range(n_x_cycles):
        xs = x_starts[yi]
        xe = x_ends[yi]
        x_dir = x_dirs[yi]
        
        for t_abs in range(xs, xe):
            # get z index and weight
            t_rel = t_abs - frame_t_start
            if t_rel < 0 or t_rel >= len(z_idx_table):
                continue
            
            zi = z_idx_table[t_rel]
            if zi < 0:
                continue
            zw = z_weight_table[t_rel]
            
            # get x index and weight
            xi, xw = getMappingIdxWeight(t_abs, xs, xe, x_dir, W)
            
            val = signal[t_abs]
            val = max(0.0, -val)
            
            weight = xw * zw
            
            volume[zi, yi, xi] += val * weight
            count[zi, yi, xi] += weight


def XYZbinning_numba(
    frame_x_half_cycles, 
    frame_z_half_cycles,
    signal,
    H, W, Z
):

    x_starts = np.array([x[0] for x in frame_x_half_cycles], dtype=np.int64)
    x_ends   = np.array([x[1] for x in frame_x_half_cycles], dtype=np.int64)
    x_dirs   = np.array([x[2] for x in frame_x_half_cycles], dtype=np.int32)
    
    z_starts = np.array([z[0] for z in frame_z_half_cycles], dtype=np.int64)
    z_ends   = np.array([z[1] for z in frame_z_half_cycles], dtype=np.int64)
    z_dirs   = np.array([z[2] for z in frame_z_half_cycles], dtype=np.int32)

    volume = np.zeros((Z, H, W), dtype=np.float32)
    count = np.zeros((Z, H, W), dtype=np.float32)

    if len(x_starts) == 0 or len(z_starts) == 0:
        return count, volume

    frame_t_start = x_starts[0]
    frame_t_end = x_ends[-1]
    
    accumulateFrame(
        volume, count,
        signal,
        x_starts, x_ends, x_dirs,
        z_starts, z_ends, z_dirs,
        frame_t_start, frame_t_end,
        W, H, Z
    )
    
    mask = count > 1e-6
    volume[mask] /= count[mask]
    
    return count, volume