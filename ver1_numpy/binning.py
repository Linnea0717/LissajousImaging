import numpy as np
from numba import jit

from mapping import (
    Xmapping_linear, 
    Zmapping_linear,
    Xmapping_cos_per_halfcycle,
    Zmapping_cos_per_halfcycle,
)

def XYZbinning(
    frame_x_half_cycles,
    frame_z_half_cycles,
    signal: np.ndarray,
    H: int = 1024,
    W: int = 1024,
    Z: int = 31,
):
    print(f"[INFO] Performing XYZ binning: H={H}, W={W}, Z={Z}")
    print(f"[INFO] Number of x half-cycles: {len(frame_x_half_cycles)}")
    print(f"[INFO] Number of z half-cycles: {len(frame_z_half_cycles)}")

    # time range of the frame
    frame_t_start = frame_x_half_cycles[0][0]
    frame_t_end   = frame_x_half_cycles[-1][1]
    T = frame_t_end - frame_t_start

    # z_time_index to z_position and weight
    z_idx_table = np.full(T, -1, dtype=np.int32)
    z_weight_table = np.zeros(T, dtype=np.float32)

    for (zs, ze, z_dir) in frame_z_half_cycles:
        zs = max(zs, frame_t_start)
        ze = min(ze, frame_t_end)
        if ze <= zs: continue

        t = np.arange(zs, ze)

        zi, zw = Zmapping_cos_per_halfcycle(
            t=t,
            z_start=zs,
            z_end=ze,
            dir=z_dir,
            Z=Z,
        )

        table_indices = t - frame_t_start
        z_idx_table[table_indices] = zi
        z_weight_table[table_indices] = zw


    # x, y, z positions, weights and values (PMT) of each time sample
    all_zi = []
    all_yi = []
    all_xi = []
    all_wvals = []
    all_weights = []

    volume = np.zeros((Z, H, W), dtype=np.float32)
    count  = np.zeros((Z, H, W), dtype=np.float32)

    # process each x half-cycle (i.e., each y-line)
    for yi, (xs, xe, x_dir) in enumerate(frame_x_half_cycles):

        # time indices within this half-cycle
        abs_t = np.arange(xs, xe)

        # map time indices to x positions and weights
        xis, xws = Xmapping_cos_per_halfcycle(
            t=abs_t,
            x_start=xs,
            x_end=xe,
            dir=x_dir,
            W=W,
        )

        # time indices relative to frame start for z-table lookup
        rel_t = abs_t - frame_t_start

        # === filter valid time indices ===
        valid_t = (rel_t >= 0) & (rel_t < T)
        if not np.any(valid_t): continue

        rel_t = rel_t[valid_t]
        xis = xis[valid_t]
        xws = xws[valid_t]
        abs_t = abs_t[valid_t]
        # ================================

        zis = z_idx_table[rel_t]
        zws = z_weight_table[rel_t]
        # === filter valid z indices ===
        valid_z = (zis >= 0) & (zis < Z)
        if not np.any(valid_z): continue

        zis = zis[valid_z]
        zws = zws[valid_z]
        xis = xis[valid_z]
        xws = xws[valid_z]
        final_t = abs_t[valid_z]
        # ================================

        raw_sig = signal[final_t]
        val_line = np.maximum(0.0, -raw_sig)

        w_line = xws * zws

        yis = np.full_like(zis, yi, dtype=np.int32)

        all_zi.append(zis)
        all_yi.append(yis)
        all_xi.append(xis)
        all_wvals.append(val_line * w_line)
        all_weights.append(w_line)
    
    if not all_zi:
        return np.zeros((Z, H, W), dtype=np.float32), np.zeros((Z, H, W), dtype=np.float32)

    ZI = np.concatenate(all_zi)
    YI = np.concatenate(all_yi)
    XI = np.concatenate(all_xi)
    W_VALS = np.concatenate(all_wvals)
    WEIGHTS = np.concatenate(all_weights)

    volume = np.zeros((Z, H, W), dtype=np.float32)
    count  = np.zeros((Z, H, W), dtype=np.float32)

    print(f"[INFO] Accumulating {len(ZI)} samples into volume...")
    np.add.at(volume, (ZI, YI, XI), W_VALS)
    np.add.at(count,  (ZI, YI, XI), WEIGHTS)

    mask = count > 0  # avoid division by zero
    volume[mask] /= count[mask]

    print(
        "volume min/max:",
        volume.min(),
        volume.max(),
        "total voxels:",
        volume.size,
        "nonzero voxels:",
        np.count_nonzero(mask),
    )

    return count, volume

def XYbinning(
    x_half_cycles,
    signal: np.ndarray,
    H: int = 1024,
    W: int = 1024,
):
    """
    Build a (H, W) image from one frame using x binning only.

    Parameters
    ----------
    x_half_cycles : list[(s, e, dir)]
        Half-cycles belonging to one frame.
    signal : np.ndarray
        PMT signal.
    W : int
        Number of x bins.

    Returns
    -------
    image : np.ndarray, shape (H, W)
    """

    image = np.zeros((H, W), dtype=np.float32)
    count = np.zeros((H, W), dtype=np.int32)

    for y, (s, e, dir) in enumerate(x_half_cycles):

        x_idx = Xmapping_linear(
            indices=np.arange(s, e),
            x_start=s,
            x_end=e,
            dir=dir,
            W=W,
        )

        vals = signal[s:e]

        for xi, v in zip(x_idx, vals):
            image[y, xi] += v
            count[y, xi] += 1

    mask = count > 0  # avoid division by zero
    image[mask] /= count[mask]

    return image