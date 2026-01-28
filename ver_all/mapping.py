import numpy as np

# ======== unused ========
def Xmapping_linear(
    indices: np.ndarray,
    x_start: int, 
    x_end: int,
    dir: int,
    W: int = 1024,
):
    """
    Map samples within ONE resonant half-cycle to x indices.

    Assumes:
    - [s, e) corresponds to a single monotonic scan segment.
    - direction is constant within this segment.
    """
    
    x_norm = (indices - x_start) / (x_end - x_start)    # [0, 1)
    if dir == -1:
        x_norm = 1.0 - x_norm

    x_idx = np.floor(x_norm * W).astype(np.int32)       # [0, W)
    x_idx = np.clip(x_idx, 0, W - 1)                    # [0, W-1]

    return x_idx

def Zmapping_linear(
    indices: np.ndarray,
    z_start: int,
    z_end: int,
    dir: int,
    Z: int,
):
    """
    Map samples within ONE TAG lens monotonic half-cycle to z indices.

    Assumes:
    - [s, e) corresponds to a single monotonic z scan.
    - direction is constant within this segment.
    """
    
    z_norm = (indices - z_start) / (z_end - z_start)    # [0, 1)
    if dir == -1:
        z_norm = 1.0 - z_norm

    z_idx = np.floor(z_norm * Z).astype(np.int32)       # [0, Z)
    z_idx = np.clip(z_idx, 0, Z - 1)                    # [0, Z-1]

    return z_idx
# =========================

def Xmapping_cos_per_halfcycle(
    t: np.ndarray,
    x_start: int,
    x_end: int,
    dir: int,
    W: int,
):
    """
    Sinusoidal x-mapping using per-half-cycle phase.
    Input indices are assumed to be within one half-cycle.

    Returns:
    - x_idx: mapped x indices
    - weights: per-sample weights for volume weighting
    """

    # per-half-cycle phase: 0 → π
    theta = np.pi * (t - x_start) / (x_end - x_start)

    # reverse direction if backward scan
    if dir == -1:
        theta = np.pi - theta

    # physical coordinate in [-1, 1]
    x_phys = np.cos(theta)

    # weight = sin(theta) = sqrt(1 - x_phys^2)
    x_weights = np.sqrt(np.maximum(0.0, 1.0 - x_phys**2))

    # map to uniform grid
    x_norm = (x_phys + 1.0) * 0.5
    x_idx = np.floor(x_norm * W).astype(np.int32)
    x_idx = np.clip(x_idx, 0, W - 1)

    return x_idx, x_weights

def Zmapping_cos_per_halfcycle(
    t: np.ndarray,
    z_start: int,
    z_end: int,
    dir: int,
    Z: int,
):
    """
    Sinusoidal z-mapping using per-half-cycle phase.
    """

    theta = np.pi * (t - z_start) / (z_end - z_start)

    if dir == -1:
        theta = np.pi - theta

    z_phys = np.cos(theta)

    z_weights = np.sqrt(np.maximum(0.0, 1.0 - z_phys**2))

    z_norm = (z_phys + 1.0) * 0.5
    z_idx = np.floor(z_norm * Z).astype(np.int32)
    z_idx = np.clip(z_idx, 0, Z - 1)

    return z_idx, z_weights