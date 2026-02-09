import numpy as np
from numba import njit, prange

@njit(fastmath=True, cache=True)
def getMappingIdxWeight(t, t_start, t_end, direction, max_idx):
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
    xis = int(x_norm * max_idx)
    if xis >= max_idx: xis = max_idx - 1
    if xis < 0: xis = 0
    
    return xis, xws


@njit(fastmath=True, cache=True, parallel=True)
def accumulateVolume(
    volume, count, 
    signal,
    x_starts, x_ends, x_dirs,
    z_starts, z_ends, z_dirs,
    frame_t_start, frame_t_end,
    W, H, Z
):
    """
    One volume accumulation kernel.
    """
    t_len = frame_t_end - frame_t_start

    n_z_cycles = len(z_starts)
    n_x_cycles = len(x_starts)
    
    for yi in prange(n_x_cycles):
        xs = x_starts[yi]
        xe = x_ends[yi]
        x_dir = x_dirs[yi]
        
        zi, zs, ze, zdir = -1, 0, 0, 0

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
            zi, zw = getMappingIdxWeight(t_abs, zs, ze, zdir, Z)
            
            # get x index and weight
            xi, xw = getMappingIdxWeight(t_abs, xs, xe, x_dir, W)
            
            val = signal[t_abs]
            val = max(0.0, -val)
            
            weight = xw * zw
            
            volume[zi, yi, xi] += val * weight
            count[zi, yi, xi] += weight


def XYZbinning_numba(
    x_starts, x_ends, x_dirs,
    z_starts, z_ends, z_dirs,
    signal,
    H, W, Z
):

    volume = np.zeros((Z, H, W), dtype=np.float32)
    count = np.zeros((Z, H, W), dtype=np.float32)

    if len(x_starts) == 0 or len(z_starts) == 0:
        return count, volume

    frame_t_start = x_starts[0]
    frame_t_end = x_ends[-1]
    
    accumulateVolume(
        volume, count,
        signal,
        x_starts, x_ends, x_dirs,
        z_starts, z_ends, z_dirs,
        frame_t_start, frame_t_end,
        W, H, Z
    )
    
    mask = count > 1e-6
    volume[mask] /= count[mask]

    print(f"[INFO] Binning completed: non-zero voxels = {np.sum(mask)}")
    
    return count, volume