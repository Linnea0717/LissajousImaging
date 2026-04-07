# test_coord_collision.py
import numpy as np
from collections import defaultdict

FS         = 1_000_000_000
X_FREQ     = 8_000
Z_FREQ     = 190_000
W_OUT      = 1024
Z_OUT      = 31

X_HALF_LEN = FS // (2 * X_FREQ)
Z_HALF_LEN = FS // (2 * Z_FREQ)

def cos_idx(j, length, n_out, direction):
    phase = j / length
    if direction == -1:
        phase = 1.0 - phase
    x_phys = np.cos(np.pi * phase)
    return np.clip(np.round((x_phys + 1.0) / 2.0 * n_out).astype(int), 0, n_out - 1)

# 一個 x 半週期內的所有 sample
x_samples = np.arange(X_HALF_LEN)
x_idx_all = cos_idx(x_samples, X_HALF_LEN, W_OUT, direction=+1)

# 切成 z 半週期
coord_to_z_ids = defaultdict(set)

zi = 0
t = 0
while t < X_HALF_LEN:
    z_lo  = t
    z_hi  = min(t + Z_HALF_LEN, X_HALF_LEN)
    z_dir = +1 if zi % 2 == 0 else -1

    local_j = np.arange(z_hi - z_lo)
    z_idx_all = cos_idx(local_j, Z_HALF_LEN, Z_OUT, direction=z_dir)

    for j in range(z_hi - z_lo):
        coord = (int(x_idx_all[z_lo + j]), int(z_idx_all[j]))
        coord_to_z_ids[coord].add(zi)

    t  += Z_HALF_LEN
    zi += 1

# 統計
multi_z = {c: ids for c, ids in coord_to_z_ids.items() if len(ids) > 1}
print(f"x 半週期長度     : {X_HALF_LEN} samples")
print(f"z 半週期長度     : {Z_HALF_LEN} samples")
print(f"z 半週期數 / x   : {zi}")
print(f"總座標數         : {len(coord_to_z_ids)}")
print(f"有跨 z 半週期碰撞: {len(multi_z)}")
if multi_z:
    for coord, ids in list(multi_z.items()):
        print(f"  x={coord[0]:4d}  z={coord[1]:3d}  被 z-cycles {sorted(ids)} 打到")