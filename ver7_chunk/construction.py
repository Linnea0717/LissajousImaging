import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import time

from utils import (
    file_chunk_generator,
    extract_trigs_and_data14_signed,
    compute_shift_array,
    print_timing_summary,
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

Z_START_THRESHOLD = 1000
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

    # ── timer ──────────────────────────────────────
    t = defaultdict(float)
    n_chunks = 0
    n_volumes_saved = 0
    pmt_prev    = np.zeros(0, dtype=np.int16)
    prev_offset = 0

    print(f"[INFO] Dataset {dataset_name}: streaming from {raw0_path.name}")

    for offset, chunk0, chunk1 in file_chunk_generator(raw0_path, raw1_path, chunk_samples):

        # ── Step 1: extract trig0 + PMT from channel 0 ──────────────────
        t0 = time.perf_counter()
        trig0, pmt_cur = extract_trigs_and_data14_signed(chunk0)
        t['1_extract_ch0'] += time.perf_counter() - t0

        # ── Step 2: extract TAG signal from channel 1 ────────────────────
        t0 = time.perf_counter()
        _, tag = extract_trigs_and_data14_signed(chunk1)
        t['2_extract_ch1'] += time.perf_counter() - t0

        # ── Step 3: build 2-chunk PMT sliding window ─────────────────────
        t0 = time.perf_counter()
        pmt_window    = np.concatenate([pmt_prev, pmt_cur])
        window_offset = prev_offset
        t['3_pmt_window'] += time.perf_counter() - t0

        # ── Step 4: parse x half-cycles ──────────────────────────────────
        t0 = time.perf_counter()
        x_hc = x_parser.feed(trig0, offset)
        t['4_x_parser'] += time.perf_counter() - t0

        # ── Step 5: parse z half-cycles ──────────────────────────────────
        t0 = time.perf_counter()
        z_hc = z_parser.feed(tag, offset)
        t['5_z_parser'] += time.perf_counter() - t0

        # ── Step 6~9: volume accumulation (in StreamingVolumeProcessor) ──
        t0 = time.perf_counter()
        completed, inner_t = vol_proc.feed_timed(x_hc, z_hc, pmt_window, window_offset)
        t['6_vol_proc (total)'] += time.perf_counter() - t0
        for k, v in inner_t.items():
            t[f'  6{k}'] += v

        # ── Step 10: save completed volumes ──────────────────────────────
        t0 = time.perf_counter()
        for vol in completed:
            np.savez_compressed(
                save_base / f"vol_{vol['index']:04d}.npz",
                x=vol['x'], y=vol['y'], z=vol['z'],
                signal=vol['signal'],
                shape=vol['shape'],
            )
            n_volumes_saved += 1
        t['10_save_npz'] += time.perf_counter() - t0

        pmt_prev    = pmt_cur
        prev_offset = offset
        n_chunks   += 1

        if n_chunks % 500 == 0:
            print(f"  [chunk {n_chunks}]  offset={offset:,}  volumes={n_volumes_saved}")

    print(f"[INFO] Done. {n_volumes_saved} volumes saved to {save_base}")
    print_timing_summary(t, n_chunks, n_volumes_saved)

    return dict(t)


def __main__():
    parser = argparse.ArgumentParser(
        description="Lissajous 3D reconstruction — streaming COO output (with timing)"
    )
    parser.add_argument("--dataset",       nargs="+", type=str, default=["1", "2", "3"], metavar="ID")
    parser.add_argument("--z_slices",      type=int,  default=31)
    parser.add_argument("--scan_h",        type=int,  default=SCAN_H)
    parser.add_argument("--scan_w",        type=int,  default=SCAN_W)
    parser.add_argument("--out_h",         type=int,  default=None)
    parser.add_argument("--out_w",         type=int,  default=None)
    parser.add_argument("--data_root",     type=str,  default=str(DATA_ROOT))
    parser.add_argument("--save_root",     type=str,  default=str(SAVE_ROOT))
    parser.add_argument("--chunk_samples", type=int,  default=CHUNK_SAMPLES)
    args = parser.parse_args()

    OUT_H = args.out_h if args.out_h is not None else args.scan_h
    OUT_W = args.out_w if args.out_w is not None else args.scan_w

    for ds in args.dataset:
        print(f"\n[INFO] Processing dataset {ds}")
        processDataset(
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


if __name__ == "__main__":
    __main__()