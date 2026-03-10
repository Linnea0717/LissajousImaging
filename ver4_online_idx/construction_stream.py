
import argparse
import gc
from pathlib import Path
import numpy as np
import time

from utils import (
    read_raw_u16_mmap,
    extract_trigs_and_data14_signed,
    locateResonantTransitions,
    transitions2HalfCycles,
    locateTagRisingEdges,
    risingEdges2HalfCycles,
    compute_shift_array,
    saveXYFrame_u16,
    saveXYZVolume_u16
)

from binning_numba import (
    XYZbinning_numba,
)


# =========================
# Parameters
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_HALFCYCLE_LEN = None
LEN_TOL = 0.5

DROP_BEFORE_FIRST_FRAME = 20
DROP_AFTER_EACH_FRAME = 20
DROP_TAIL = True

Z_START_THRESHOLD = 1000  # channel1 > 1000 is considered the start of a z oscillation cycle (trigger)

# COEFFS = [1.7, -0.3, 7, 9, -76, 0.38]
COEFFS = [0, 0, 0, 0, 0, 0]


def streamVolumesGenerator(
    x_half_cycles,
    z_half_cycles,
    lines_per_volume: int,
    drop_before: int,
    drop_after: int,
    drop_tail: bool
):
    total_x_cycles = len(x_half_cycles)
    cur_x_idx = drop_before
    frame_count = 0

    cur_z_idx = 0
    num_z_half_cycles = len(z_half_cycles)

    while cur_x_idx + lines_per_volume <= total_x_cycles:

        frame_x_half_cycles = x_half_cycles[cur_x_idx: cur_x_idx + lines_per_volume]

        t_start = frame_x_half_cycles[0][0]
        t_end = frame_x_half_cycles[-1][1]

        frame_z_half_cycles = []

        while cur_z_idx < num_z_half_cycles and z_half_cycles[cur_z_idx][1] < t_start:
            cur_z_idx += 1

        temp_idx = cur_z_idx
        while temp_idx < num_z_half_cycles:
            zs, ze, zdir = z_half_cycles[temp_idx]
            if zs > t_end:
                break

            if not (ze <= t_start or zs >= t_end):
                frame_z_half_cycles.append((zs, ze, zdir))

            temp_idx += 1

        yield frame_count, frame_x_half_cycles, frame_z_half_cycles

        frame_count += 1
        cur_x_idx += lines_per_volume + drop_after


def processDataset(
    dataset_name: str,
    z_slices: int,
    scan_h: int,
    scan_w: int,
    out_h: int,
    out_w: int,
    data_root: Path,
    save_root: Path,
):
    data_dir = data_root / dataset_name
    save_dir = save_root / dataset_name

    raw0_path = data_dir / "raw_data_0.bin"
    raw1_path = data_dir / "raw_data_1.bin"
    if not raw0_path.exists() or not raw1_path.exists():
        raise FileNotFoundError(f"Cannot find {raw0_path} or {raw1_path}")

    time_record = {}

    # ========================= Data Loading =========================
    load_start_time = time.time()

    raw_u16_0 = read_raw_u16_mmap(raw0_path, endian="<u2")
    raw_u16_1 = read_raw_u16_mmap(raw1_path, endian="<u2")

    load_end_time = time.time()
    load_elapsed = load_end_time - load_start_time
    time_record['data_loading'] = load_elapsed
    # ================================================================

    # ========================= Preprocessing =========================
    extract_trid_start_time = time.time()

    trig0, _, PMT = extract_trigs_and_data14_signed(raw_u16_0)
    _, _, TAG = extract_trigs_and_data14_signed(raw_u16_1)

    extract_trid_end_time = time.time()
    extract_trid_elapsed = extract_trid_end_time - extract_trid_start_time
    time_record['extract_trigs'] = extract_trid_elapsed
    print(f"[INFO] Loaded {len(trig0)} samples from dataset {dataset_name}")

    locate_cycles_start_time = time.time()

    transitions = locateResonantTransitions(trig0)
    x_half_cycles = transitions2HalfCycles(transitions, trig0)

    rising_edges = locateTagRisingEdges(TAG, Z_START_THRESHOLD)
    z_half_cycles = risingEdges2HalfCycles(rising_edges)

    shifts = compute_shift_array(scan_h, scan_w, COEFFS)

    locate_cycles_end_time = time.time()
    locate_cycles_elapsed = locate_cycles_end_time - locate_cycles_start_time
    time_record['locate_cycles'] = locate_cycles_elapsed

    print(f"[INFO] Extracted {len(x_half_cycles)} x half-cycles from data")
    print(f"[INFO] Extracted {len(z_half_cycles)} z half-cycles from data")
    print(f"[INFO] Mean x half-cycle length: {np.mean(x_half_cycles[:,1] - x_half_cycles[:,0]):.2f} samples")
    print(f"[INFO] Mean z half-cycle length: {np.mean(z_half_cycles[:,1] - z_half_cycles[:,0]):.2f} samples")
    # ================================================================

    # ========================== Gridding ===========================
    print(f"[INFO] Start streaming volumes...")
    gridding_start_time = time.time()

    volume_gen = streamVolumesGenerator(
        x_half_cycles=x_half_cycles,
        z_half_cycles=z_half_cycles,
        lines_per_volume=scan_h,
        drop_before=DROP_BEFORE_FIRST_FRAME,
        drop_after=DROP_AFTER_EACH_FRAME,
        drop_tail=DROP_TAIL
    )

    volumes = []

    for i, volume_x, volume_z in volume_gen:
        x_starts = volume_x[:, 0].astype(np.int64)
        x_ends = volume_x[:, 1].astype(np.int64)
        x_dirs = volume_x[:, 2].astype(np.int32)

        volume_z = np.array(volume_z, dtype=np.int64)
        z_starts = volume_z[:, 0]
        z_ends = volume_z[:, 1]
        z_dirs = volume_z[:, 2].astype(np.int32)

        count, volume = XYZbinning_numba(
            x_starts=x_starts, x_ends=x_ends, x_dirs=x_dirs,
            z_starts=z_starts, z_ends=z_ends, z_dirs=z_dirs,
            signal=PMT,
            H_out=out_h, W_out=out_w, Z=z_slices,
            H_scan=scan_h, W_scan=scan_w,
            shifts=shifts,
        )

        volumes.append(volume)

        if i % 5 == 0:
            gc.collect()

    gridding_end_time = time.time()
    gridding_elapsed = gridding_end_time - gridding_start_time
    time_record['gridding'] = gridding_elapsed

    saving_start_time = time.time()
    # ================================================================

    # =========================== Saving ============================
    save_base = save_dir / f'x{out_w}y{out_h}z{z_slices}'

    for i, vol in enumerate(volumes):
        saveXYZVolume_u16(
            volume=vol,
            index=i,
            save_dir=save_base / 'xyzt_volumes',
        )

        for z in range(z_slices):
            saveXYFrame_u16(
                image=vol[z, :, :],
                index=i,
                save_dir=save_base / 'xyt_frames' / f'z-{z:03d}',
            )

    saving_end_time = time.time()
    saving_elapsed = saving_end_time - saving_start_time
    time_record['saving'] = saving_elapsed
    # ================================================================

    return time_record


def __main__():
    parser = argparse.ArgumentParser(
        description=(
            "Lissajous Scanning 3D Image Reconstruction Module.\n"
            "Processes raw photomultiplier tube (PMT) signals and scanning trajectory\n"
            "data to reconstruct volumetric images using spatial binning algorithms."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "EXAMPLES:\n"
            "  1. Process datasets under a specified data root:\n"
            "     python construction_stream.py --dataset trial1 trial2 --data_root C:/data/group --save_root C:/out/group --scan_h 128 --scan_w 128\n\n"
            "  2. Process a dataset with high-resolution Z-axis (64 slices):\n"
            "     python construction_stream.py --dataset trial3 --z_slices 64 --data_root C:/data/group --save_root C:/out/group --scan_h 512 --scan_w 512 --out_h 512 --out_w 512"
        )
    )

    parser.add_argument(
        "--dataset",
        nargs="+",
        type=str,
        default=["1", "2", "3"],
        metavar="ID",
        help=(
            "List of dataset directory names to process.\n"
            "These directories must exist within data_root.\n"
            "(Default: 1, 2, 3)"
        )
    )

    parser.add_argument(
        "--z_slices",
        type=int,
        default=31,
        metavar="N",
        help=(
            "Target resolution for the Z-axis (depth).\n"
            "Determines the number of bins used during the Z-mapping process.\n"
            "(Default: 31)"
        )
    )

    parser.add_argument(
        "--scan_h",
        type=int,
        required=True,
        metavar="H",
        help="Scan-space height."
    )

    parser.add_argument(
        "--scan_w",
        type=int,
        required=True,
        metavar="W",
        help="Scan-space width."
    )

    parser.add_argument(
        "--out_h",
        type=int,
        default=None,
        metavar="H",
        help="Output image height. Default = scan_h."
    )

    parser.add_argument(
        "--out_w",
        type=int,
        default=None,
        metavar="W",
        help="Output image width. Default = scan_w."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing dataset subfolders."
    )

    parser.add_argument(
        "--save_root",
        type=str,
        required=True,
        help="Root directory for saving outputs."
    )

    args = parser.parse_args()

    DATASET_DIR_NAMES = args.dataset
    Z_SLICES = args.z_slices
    SCAN_H = args.scan_h
    SCAN_W = args.scan_w
    OUT_H = args.out_h if args.out_h is not None else SCAN_H
    OUT_W = args.out_w if args.out_w is not None else SCAN_W
    DATA_ROOT = Path(args.data_root)
    SAVE_ROOT = Path(args.save_root)

    time_records = {}

    for dataset_name in DATASET_DIR_NAMES:
        print(f"[INFO] Processing dataset {dataset_name}")
        time_record = processDataset(
            dataset_name=dataset_name,
            z_slices=Z_SLICES,
            scan_h=SCAN_H,
            scan_w=SCAN_W,
            out_h=OUT_H,
            out_w=OUT_W,
            data_root=DATA_ROOT,
            save_root=SAVE_ROOT,
        )
        time_records[dataset_name] = time_record

    print("\n[SUMMARY] Average execution times:")
    for key in time_records[DATASET_DIR_NAMES[0]].keys():
        avg_time = np.mean([time_records[ds][key] for ds in DATASET_DIR_NAMES])
        print(f"  {key}: {avg_time:.2f} seconds")


if __name__ == "__main__":
    __main__()