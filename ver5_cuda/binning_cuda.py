import math
import numpy as np
from numba import cuda

# ==========================================
# Device Functions
# ==========================================

@cuda.jit(device=True)
def get_mapping_gpu(t, t_start, t_end, direction, max_idx):

    total_len = t_end - t_start
    if total_len <= 0:
        return 0, 0.0
    
    phase = (t - t_start) / total_len
    theta = math.pi * phase
    
    if direction == -1:
        theta = math.pi - theta
        
    x_phys = math.cos(theta)

    val_sq = 1.0 - x_phys * x_phys
    if val_sq < 0.0:
        val_sq = 0.0
    xws = math.sqrt(val_sq)
    
    x_norm = (x_phys + 1.0) * 0.5
    xis = int(x_norm * max_idx)
    
    if xis >= max_idx:
        xis = max_idx - 1
    elif xis < 0:
        xis = 0
    
    return xis, xws


# ==========================================
# Kernel Functions
# ==========================================

@cuda.jit(fastmath=True)
def accumulate_frame_kernel(
    volume, count,
    signal_slice,
    x_starts, x_ends, x_dirs,
    z_starts, z_ends, z_dirs,
    signal_offset,
    W, H, Z
):
    yi = cuda.blockIdx.x
    
    if yi >= x_starts.shape[0]:
        return

    xs = x_starts[yi]
    xe = x_ends[yi]
    x_dir = x_dirs[yi]
    
    n_z_cycles = z_starts.shape[0]
    zi = -1
    zs = 0
    ze = 0
    zdir = 0
    
    for i in range(n_z_cycles):
        if z_starts[i] <= xs < z_ends[i]:
            zi = i
            zs = z_starts[i]
            ze = z_ends[i]
            zdir = z_dirs[i]
            break
            
    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    
    for t_abs in range(xs + tid, xe, stride):
        
        if zi != -1 and t_abs >= ze:
            next_zi = zi + 1
            if next_zi < n_z_cycles:
                if z_starts[next_zi] <= t_abs < z_ends[next_zi]:
                    zi = next_zi
                    zs = z_starts[zi]
                    ze = z_ends[zi]
                    zdir = z_dirs[zi]
                else:
                    zi = -1
            else:
                zi = -1

        if zi == -1:
            for i in range(n_z_cycles):
                if z_starts[i] <= t_abs < z_ends[i]:
                    zi = i
                    zs = z_starts[i]
                    ze = z_ends[i]
                    zdir = z_dirs[i]
                    break
        
        if zi == -1:
            continue

        idx_z, w_z = get_mapping_gpu(t_abs, zs, ze, zdir, Z)
        
        idx_x, w_x = get_mapping_gpu(t_abs, xs, xe, x_dir, W)
        
        sig_idx = t_abs - signal_offset
        val = signal_slice[sig_idx]
        
        
        val = -val
        if val < 0.0: val = 0.0
            
        final_weight = w_x * w_z
        weighted_val = val * final_weight
        
        cuda.atomic.add(volume, (idx_z, yi, idx_x), weighted_val)
        cuda.atomic.add(count, (idx_z, yi, idx_x), final_weight)

# ==========================================
# Host Function (CPU 端接口)
# ==========================================

def XYZbinning_cuda(
    x_starts, x_ends, x_dirs,
    z_starts, z_ends, z_dirs,
    signal,
    H, W, Z
):
    volume_device = cuda.device_array((Z, H, W), dtype=np.float32)
    count_device = cuda.device_array((Z, H, W), dtype=np.float32)
    
    volume_device[:] = 0
    count_device[:] = 0

    if len(x_starts) == 0:
        return count_device.copy_to_host(), volume_device.copy_to_host()

    frame_t_start = x_starts[0]
    frame_t_end = x_ends[-1]
    
    frame_t_start = max(0, frame_t_start)
    frame_t_end = min(len(signal), frame_t_end)
    
    signal_slice = signal[frame_t_start:frame_t_end]
    signal_slice = signal_slice.astype(np.float32) 
    d_signal_slice = cuda.to_device(signal_slice)

    d_x_starts = cuda.to_device(np.ascontiguousarray(x_starts))
    d_x_ends   = cuda.to_device(np.ascontiguousarray(x_ends))
    d_x_dirs   = cuda.to_device(np.ascontiguousarray(x_dirs))
    
    d_z_starts = cuda.to_device(np.ascontiguousarray(z_starts))
    d_z_ends   = cuda.to_device(np.ascontiguousarray(z_ends))
    d_z_dirs   = cuda.to_device(np.ascontiguousarray(z_dirs))


    threads_per_block = 256
    blocks_per_grid = len(x_starts)
    

    accumulate_frame_kernel[blocks_per_grid, threads_per_block](
        volume_device, count_device,
        d_signal_slice,
        d_x_starts, d_x_ends, d_x_dirs,
        d_z_starts, d_z_ends, d_z_dirs,
        frame_t_start,
        W, H, Z
    )
    
    
    volume_host = volume_device.copy_to_host()
    count_host = count_device.copy_to_host()
    
    del d_signal_slice, d_x_starts, d_x_ends, d_x_dirs
    del d_z_starts, d_z_ends, d_z_dirs, volume_device, count_device
    
    mask = count_host > 1e-6
    volume_host[mask] /= count_host[mask]
    
    print(f"[INFO-GPU] Binning completed: non-zero voxels = {np.sum(mask)}")
    
    return count_host, volume_host