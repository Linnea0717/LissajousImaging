import argparse
import gc
from pathlib import Path
import numpy as np
import time

from utils import (
    read_raw_u16_mmap,
    extract_trigs_and_data14_signed,
    compute_shift_array,
)

from parser import (
    XHalfCycleParser,
    ZHalfCycleParser,
    StreamingVolumeProcessor,
)

# =========================
# Parameters
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "20251229data"
SAVE_ROOT = PROJECT_ROOT / 'output' / 'ver7'

SCAN_W = 1024
SCAN_H = 1024

EXPECTED_HALFCYCLE_LEN = None
LEN_TOL = 0.5

DROP_BEFORE_FIRST_FRAME = 20
DROP_AFTER_EACH_FRAME   = 20
DROP_TAIL = True

Z_START_THRESHOLD = 1000       # channel1 > 1000 is considered the start of a z oscillation cycle (trigger)

# COEFFS = [1.7, -0.3, 7, 9, -76, 0.38]
COEFFS = [0, 0, 0, 0, 0, 0]

CHUNK_SAMPLES = 100_000


def processDataset(
    dataset_name: str, 
    z_slices: int,
    scan_w: int,
    scan_h: int, 
    out_w: int, 
    out_h: int,
    data_root: Path,
    save_root: Path,
    chunk_samples: int = CHUNK_SAMPLES,
):
    data_dir = data_root / dataset_name
    save_dir = save_root / dataset_name

    raw0_path = data_dir / "raw_data_0.bin"
    raw1_path = data_dir / "raw_data_1.bin"
    if not raw0_path.exists() or not raw1_path.exists():
        raise FileNotFoundError(f"Cannot find {raw0_path} or {raw1_path}")
    
    save_base = save_dir / f'x{out_w}y{out_h}z{z_slices}_coo'
    save_base.mkdir(parents=True, exist_ok=True)

    time_record = {}

    # ==== mmap .bin ====
    load_start = time.time()
    raw_u16_0 = read_raw_u16_mmap(raw0_path, endian="<u2")
    raw_u16_1 = read_raw_u16_mmap(raw1_path, endian="<u2")
    total_samples = len(raw_u16_0)
    time_record['mmap'] = time.time() - load_start
    print(f"[INFO] Dataset {dataset_name}: {total_samples} samples")

    # ==== extrace trigs ====
    extract_start = time.time()
    _, _, PMT = extract_trigs_and_data14_signed(raw_u16_0)

    time_record['extract_pmt'] = time.time() - extract_start
 
    shifts = compute_shift_array(scan_h, scan_w, COEFFS)


    x_parser = XHalfCycleParser()
    z_parser = ZHalfCycleParser(threshold=Z_START_THRESHOLD)
    vol_proc = StreamingVolumeProcessor(
        lines_per_volume=scan_h,
        z_slices=z_slices,
        out_h=out_h, out_w=out_w,
        H_scan=scan_h, W_scan=scan_w,
        shifts=shifts,
    )


    process_start = time.time()
    n_volumes_saved = 0
    offset = 0
 
    while offset < total_samples:
        end = min(offset + chunk_samples, total_samples)
 
        chunk0 = raw_u16_0[offset:end]
        chunk1 = raw_u16_1[offset:end]
 
        trig0, _, _   = extract_trigs_and_data14_signed(chunk0)
        _,     _, tag = extract_trigs_and_data14_signed(chunk1)
 
        x_hc = x_parser.feed(trig0, offset)
        z_hc = z_parser.feed(tag,   offset)
 
        completed = vol_proc.feed(x_hc, z_hc, PMT)
 
        for vol in completed:
            np.savez_compressed(
                save_base / f"vol_{vol['index']:04d}.npz",
                x=vol['x'], y=vol['y'], z=vol['z'],
                signal=vol['signal'],
                shape=vol['shape'],
            )
            n_volumes_saved += 1
 
        offset = end
 
    time_record['streaming'] = time.time() - process_start
    print(f"[INFO] Done. {n_volumes_saved} volumes saved to {save_base}")

    return time_record

def __main__():

    parser = argparse.ArgumentParser(
        description="Lissajous 3D reconstruction — streaming COO output"
    )
    parser.add_argument("--dataset", nargs="+", type=str,
                        default=["1", "2", "3"], metavar="ID")
    parser.add_argument("--z_slices", type=int, default=31)
    parser.add_argument("--scan_h",   type=int, default=SCAN_W)
    parser.add_argument("--scan_w",   type=int, default=SCAN_H)
    parser.add_argument("--out_h",    type=int, default=None)
    parser.add_argument("--out_w",    type=int, default=None)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--save_root", type=str, default=SAVE_ROOT)
    parser.add_argument("--chunk_samples", type=int, default=CHUNK_SAMPLES)

    args = parser.parse_args()

    DATASET_DIR_NAMES = args.dataset
    OUT_H = args.out_h if args.out_h is not None else args.scan_h
    OUT_W = args.out_w if args.out_w is not None else args.scan_w

    time_records = {}
    
    for ds in DATASET_DIR_NAMES:
        print(f"[INFO] Processing dataset {ds}")
        time_records[ds] = processDataset(
            dataset_name=ds,
            z_slices=args.z_slices,
            scan_h=args.scan_h,
            scan_w=args.scan_w,
            out_h=OUT_H,
            out_w=OUT_W,
            data_root=Path(args.data_root),
            save_root=Path(args.save_root),
            chunk_samples=args.chunk_samples,
        )


    print("\n[SUMMARY] Average execution times:")
    for key in time_records[DATASET_DIR_NAMES[0]].keys():
        avg_time = np.mean([time_records[ds][key] for ds in DATASET_DIR_NAMES])
        print(f"  {key}: {avg_time:.2f} seconds")

if __name__ == "__main__":
    __main__()
