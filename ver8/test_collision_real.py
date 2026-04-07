"""
visualize_collision.py
======================
用真實資料，視覺化跨 z half-cycle 碰撞的位置分布。

用法：
  python visualize_collision.py --dataset 1 --n_xcycles 50
"""

import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import read_raw_u16_mmap, extract_trigs_and_data14_signed
from parser import XHalfCycleParser, ZHalfCycleParser

PROJECT_ROOT      = Path(__file__).resolve().parent.parent
DATA_ROOT         = PROJECT_ROOT / "20251229data"
Z_START_THRESHOLD = 1000
W_OUT             = 1024
Z_OUT             = 31


def cos_idx(j_arr, length, n_out, direction):
    phase  = j_arr / length
    if direction == -1:
        phase = 1.0 - phase
    x_phys = np.cos(np.pi * phase)
    return np.clip(np.round((x_phys + 1.0) / 2.0 * n_out).astype(np.int32), 0, n_out - 1)


def analyze_xcycle(x_hc, overlap_z_arr):
    """
    回傳：
      all_coords   : list of (x_idx, z_idx, zi)   — 所有有效 sample 的座標和所屬 zi
      collision_coords : set of (x_idx, z_idx)     — 跨 z 碰撞的座標
    """
    xs, xe, x_dir = int(x_hc[0]), int(x_hc[1]), int(x_hc[2])
    x_len = xe - xs
    if x_len == 0 or len(overlap_z_arr) == 0:
        return [], set()

    j_arr     = np.arange(x_len, dtype=np.float64)
    x_idx_all = cos_idx(j_arr, x_len, W_OUT, x_dir)

    all_coords = []
    coord_to_zis = defaultdict(set)

    for zi, z_hc in enumerate(overlap_z_arr):
        zs, ze, z_dir = int(z_hc[0]), int(z_hc[1]), int(z_hc[2])
        lo = max(zs, xs) - xs
        hi = min(ze, xe) - xs
        if lo >= hi:
            continue
        z_len       = ze - zs
        local_j_in_z = np.arange(max(zs, xs) - zs, min(ze, xe) - zs, dtype=np.float64)
        z_idx_seg   = cos_idx(local_j_in_z, z_len, Z_OUT, z_dir)
        x_idx_seg   = x_idx_all[lo:hi]

        for k in range(len(x_idx_seg)):
            coord = (int(x_idx_seg[k]), int(z_idx_seg[k]))
            all_coords.append((coord[0], coord[1], zi))
            coord_to_zis[coord].add(zi)

    collision_coords = {c for c, zis in coord_to_zis.items() if len(zis) > 1}
    return all_coords, collision_coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, default="1")
    parser.add_argument("--n_xcycles", type=int, default=50)
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT))
    args = parser.parse_args()

    data_dir  = Path(args.data_root) / args.dataset
    raw0      = read_raw_u16_mmap(data_dir / "raw_data_0.bin")
    raw1      = read_raw_u16_mmap(data_dir / "raw_data_1.bin")

    need   = args.n_xcycles * 3000 + 200_000
    chunk  = min(need, len(raw0))
    trig0, _ = extract_trigs_and_data14_signed(raw0[:chunk])
    _, tag      = extract_trigs_and_data14_signed(raw1[:chunk])

    x_hcs = XHalfCycleParser().feed(trig0, 0)
    z_hcs = ZHalfCycleParser(threshold=Z_START_THRESHOLD).feed(tag, 0)
    x_hcs = x_hcs[:args.n_xcycles]
    # print(f"[INFO] {len(x_hcs)} x half-cycles, {len(z_hcs)} z half-cycles")

    # ── 蒐集所有 x half-cycle 的碰撞資訊 ─────────────────────────────
    # density map: 每個 (x_idx, z_idx) 被幾個 x half-cycle 碰到過
    collision_density = np.zeros((Z_OUT + 1, W_OUT + 1), dtype=np.int32)
    all_density       = np.zeros((Z_OUT + 1, W_OUT + 1), dtype=np.int32)

    per_xcycle_stats = []
    z_cursor = 0

    for x_hc in x_hcs:
        xs, xe = int(x_hc[0]), int(x_hc[1])

        while z_cursor < len(z_hcs) and z_hcs[z_cursor][1] < xs:
            z_cursor += 1
        tmp = z_cursor
        overlap_z = []
        while tmp < len(z_hcs) and z_hcs[tmp][0] < xe:
            overlap_z.append(z_hcs[tmp])
            tmp += 1
        if not overlap_z:
            continue

        all_coords, col_coords = analyze_xcycle(x_hc, np.array(overlap_z, dtype=np.int64))
        per_xcycle_stats.append(len(col_coords))

        for xi, zi, _ in all_coords:
            all_density[zi, xi] += 1
        for xi, zi in col_coords:
            collision_density[zi, xi] += 1

    # print(f"[INFO] 跨 z 碰撞座標數（平均）: {np.mean(per_xcycle_stats):.1f}")
    # print(f"[INFO] 跨 z 碰撞座標數（最大）: {np.max(per_xcycle_stats)}")

    # ── 視覺化 ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 5))

    # 圖：碰撞位置疊加在落點上（單一 x half-cycle 範例）
    ax = fig.add_subplot(1, 1, 1)
    example_hc = x_hcs[len(x_hcs) // 2]
    xs, xe = int(example_hc[0]), int(example_hc[1])
    z_cursor2 = 0
    while z_cursor2 < len(z_hcs) and z_hcs[z_cursor2][1] < xs:
        z_cursor2 += 1
    tmp2 = z_cursor2
    ov2 = []
    while tmp2 < len(z_hcs) and z_hcs[tmp2][0] < xe:
        ov2.append(z_hcs[tmp2]); tmp2 += 1

    coords2, col2 = analyze_xcycle(example_hc, np.array(ov2, dtype=np.int64))
    normal_x = [c[0] for c in coords2 if (c[0], c[1]) not in col2]
    normal_z = [c[1] for c in coords2 if (c[0], c[1]) not in col2]
    col_x    = [c[0] for c in coords2 if (c[0], c[1]) in col2]
    col_z    = [c[1] for c in coords2 if (c[0], c[1]) in col2]

    ax.scatter(normal_x, normal_z, s=1,   c='steelblue', alpha=0.3, label='normal')
    ax.scatter(col_x,    col_z,    s=10,  c='red',       alpha=0.9, label=f'collision ({len(col2)})')
    ax.set_xlim(0, W_OUT); ax.set_ylim(0, Z_OUT)
    ax.set_title("single x half-cycle collision example")
    ax.set_xlabel("x_idx"); ax.set_ylabel("z_idx")
    ax.legend(markerscale=4, fontsize=9)

    plt.tight_layout()
    out_path = Path("output") / "images" / "collision_visualization.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] image saved to {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()