import argparse
import numpy as np
import time
import pandas as pd  # 用於漂亮的表格輸出
from pathlib import Path

# 引用你原本的模組
from utils import (
    read_raw_u16_mmap,
    extract_trigs_and_data14_signed,
    locateResonantTransitions,
    transitions2HalfCycles,
    locateTagRisingEdges,
    risingEdges2HalfCycles,
    compute_shift_array
)
from binning_numba import XYZbinning_numba

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "20251229data"

DEFAULT_COEFFS = [1.7, -0.3, 7, 9, -76, 0.38]
Z_START_THRESHOLD = 1000

# (Z, H, W)
TEST_RESOLUTIONS = [
    (31, 512, 512),
    (31, 1024, 1024),
    (10, 512, 512),
    (10, 1024, 1024),
]

def load_data_once(dataset_name, H_scan, W_scan):

    data_dir = DATA_ROOT / dataset_name
    raw0_path = data_dir / "raw_data_0.bin"
    raw1_path = data_dir / "raw_data_1.bin"

    if not raw0_path.exists():
        raise FileNotFoundError(f"Cannot find file: {raw0_path}")

    print(f"[INFO] Loading raw data from {dataset_name}...")
    t0 = time.time()
    
    raw_u16_0 = read_raw_u16_mmap(raw0_path)
    raw_u16_1 = read_raw_u16_mmap(raw1_path)
    
    trig0, _, PMT = extract_trigs_and_data14_signed(raw_u16_0)
    _, _, TAG = extract_trigs_and_data14_signed(raw_u16_1)
    
    print(f"[INFO] Parsing cycles...")
    transitions = locateResonantTransitions(trig0)
    x_half_cycles = transitions2HalfCycles(transitions, trig0)

    rising_edges = locateTagRisingEdges(TAG, Z_START_THRESHOLD)
    z_half_cycles = risingEdges2HalfCycles(rising_edges)
    
    shifts = compute_shift_array(H_scan, W_scan, DEFAULT_COEFFS)
    
    print(f"[INFO] Data prepared in {time.time()-t0:.2f}s")
    
    return {
        'x_half_cycles': x_half_cycles,
        'z_half_cycles': z_half_cycles,
        'pmt': PMT,
        'shifts': shifts
    }

def get_one_volume_cycles(data_pack, H_scan, x_skip=0):

    x_half_cycles = data_pack['x_half_cycles']
    z_half_cycles = data_pack['z_half_cycles']

    lines_per_vol = H_scan
    
    if len(x_half_cycles) < x_skip + lines_per_vol:
        print(f"[ERROR] Not enough X half-cycles for one volume after skipping {x_skip} lines.")
        return None

    x_subset = x_half_cycles[x_skip : x_skip + lines_per_vol]
    t_start = x_subset[0][0]
    t_end = x_subset[-1][1]
    
    z_subset = []
    for zc in z_half_cycles:
        zs, ze, zdir = zc
        if ze < t_start: continue
        if zs > t_end: break
        z_subset.append(zc)
    
    return x_subset, np.array(z_subset, dtype=np.int64)

def run_benchmark(dataset_name, scan_h, scan_w, N_volumes=1):
    data = load_data_once(dataset_name, scan_h, scan_w)
    
    x_sub, z_sub = get_one_volume_cycles(data, scan_h)
    
    if x_sub is None:
        print("[ERROR] Data insufficient for one volume.")
        return

    results = []

    print(f"\n[INFO] Starting Benchmark on Dataset {dataset_name}...")
    print(f"[INFO] Base Scan Resolution: {scan_h} x {scan_w}")
    print("-" * 60)

    for (z_res, h_res, w_res) in TEST_RESOLUTIONS:

        res_results = []

        for vol_idx in range(N_volumes):

            count, _ = XYZbinning_numba(
                x_starts=x_sub[:, 0].astype(np.int64),
                x_ends=x_sub[:, 1].astype(np.int64),
                x_dirs=x_sub[:, 2].astype(np.int32),
                z_starts=z_sub[:, 0],
                z_ends=z_sub[:, 1],
                z_dirs=z_sub[:, 2].astype(np.int32),
                signal=data['pmt'], 
                H_out=h_res, W_out=w_res, Z=z_res,
                H_scan=scan_h, W_scan=scan_w,
                shifts=data['shifts']
            )
            
            total_voxels = z_res * h_res * w_res
            sampled_voxels = np.sum(count > 1e-6)
            unsampled_voxels = total_voxels - sampled_voxels
            fill_rate = (sampled_voxels / total_voxels) * 100.0

            res_results.append({
                'Volume': vol_idx,
                'Sampled Voxels': sampled_voxels,
                'Unsampled Voxels': unsampled_voxels,
                'Fill Rate (%)': fill_rate
            })

        total_voxels = z_res * h_res * w_res
        avg_sampled = np.mean([r['Sampled Voxels'] for r in res_results])
        avg_unsampled = np.mean([r['Unsampled Voxels'] for r in res_results])
        avg_fill_rate = np.mean([r['Fill Rate (%)'] for r in res_results])
        results.append({
            'H_res': h_res,
            'W_res': w_res,
            'Z_res': z_res,
            'Sampled Voxels': int(avg_sampled),
            'Unsampled Voxels': int(avg_unsampled),
            'Fill Rate (%)': f"{avg_fill_rate:.2f}"
        })
        
        print(f"  -> Tested {z_res}x{h_res}x{w_res}: {avg_fill_rate:.2f}%")


    df = pd.DataFrame(results)
    
    print("\n" + "="*35 + " BENCHMARK REPORT " + "="*36)
    print(f"Dataset: {dataset_name}")
    print(df.to_markdown(index=False)) 
    # print(df.to_string(index=False))
    print("="*89)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="1", help="Dataset name")
    parser.add_argument("--scan_h", type=int, default=1024)
    parser.add_argument("--scan_w", type=int, default=1024)
    args = parser.parse_args()

    run_benchmark(args.dataset, args.scan_h, args.scan_w, 1)