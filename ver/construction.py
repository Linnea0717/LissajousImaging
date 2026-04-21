"""
construction.py
===============

1. Read full raw data into memory once.
2. Iterate sample by sample, detecting trigger edges and feed sample one by one,
    or feed in batches between events (faster, for reference only).
"""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import time

from utils import (
    read_raw_u16_mmap,
    extract_trigs_and_data14_signed,
    compute_shift_array,
    print_timing_summary,
)
from processor import StreamingVolumeProcessor

# =========================
# Parameters
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "20251229data"
SAVE_ROOT    = PROJECT_ROOT / "output" / "cur"

SCAN_W        = 1024
SCAN_H        = 1024
Z_SLICES      = 31
Z_THRESHOLD   = 1000
COEFFS        = [0, 0, 0, 0, 0, 0]
DROP_BEFORE   = 20
DROP_AFTER    = 20
DROP_TAIL     = True

X_FREQ        = 8_000
Z_FREQ        = 192_000
SAMPLING_RATE    = 1_000_000_000
NOMINAL_X_LEN    = SAMPLING_RATE / (X_FREQ * 2)  # theoretical x halfcycle length
NOMINAL_Z_LEN    = SAMPLING_RATE / Z_FREQ        # theoretical z cycle length
ESTIMATOR_WINDOW = 6


# =========================
# Main processing
# =========================

def processDataset(
    dataset_name:  str,
    z_slices:      int,
    scan_w:        int,
    scan_h:        int,
    out_w:         int,
    out_h:         int,
    data_root:     Path,
    save_root:     Path,
    mode:          str,
    nominal_x_len: float = NOMINAL_X_LEN,
    nominal_z_period: float = NOMINAL_Z_LEN,
) -> dict:
    data_dir  = data_root / dataset_name
    raw0_path = data_dir / "raw_data_0.bin"
    raw1_path = data_dir / "raw_data_1.bin"
    if not raw0_path.exists() or not raw1_path.exists():
        raise FileNotFoundError(f"Cannot find raw data in {data_dir}")

    save_base = save_root / dataset_name / f"x{out_w}y{out_h}z{z_slices}_coo"
    save_base.mkdir(parents=True, exist_ok=True)

    t = defaultdict(float)

    # ── Step 1: read raw data ────────────────────────────────────────
    t0 = time.perf_counter()
    raw0      = read_raw_u16_mmap(raw0_path)
    raw1      = read_raw_u16_mmap(raw1_path)
    n_samples = len(raw0)
    print(f"[INFO] Dataset {dataset_name}: {n_samples:,} samples")
    t['1_read'] += time.perf_counter() - t0

    # ── Step 2: extract trig0, PMT, TAG ──────────────────────────────
    t0 = time.perf_counter()
    trig0, pmt = extract_trigs_and_data14_signed(raw0)
    _,     tag = extract_trigs_and_data14_signed(raw1)
    t['2_unpack'] += time.perf_counter() - t0

    # ── Step 3: initialise processor ──────────────────────────────────────
    shifts = compute_shift_array(scan_h, scan_w, COEFFS)
    proc = StreamingVolumeProcessor(
        z_slices         = z_slices,
        out_h            = out_h,
        out_w            = out_w,
        H_scan           = scan_h,
        W_scan           = scan_w,
        shifts           = shifts,
        nominal_x_len    = nominal_x_len,
        nominal_z_period    = nominal_z_period,
        estimator_window = ESTIMATOR_WINDOW,
        lines_per_volume = scan_h,
        drop_before      = DROP_BEFORE,
        drop_after       = DROP_AFTER,
    )

    vol_buf: dict[int, list] = defaultdict(list)
    n_xcycles = 0
    n_coo_out = 0

    # ── Step 4: sample-by-sample edge detection ──────────────────────
    #   prev_trig0   : last seen x trigger level (0 or 1)
    #   prev_z_above : whether last sample was above Z_THRESHOLD
    #   seg_start    : first sample of the current unprocessed segment; for batch feeding
    #   z_midpoint   : sample index where the -1 halfcycle should start

    prev_trig0   = int(trig0[0])
    prev_z_above = int(tag[0]) > Z_THRESHOLD
    z_midpoint   = None   # int | None
    seg_start    = 0

    t0 = time.perf_counter()
    # # Batch version
    for i in range(n_samples):
        cur_trig0   = int(trig0[i])
        cur_z_above = int(tag[i]) > Z_THRESHOLD

        # Detect events at sample i (z before x at same position).
        z_mid_fires  = z_midpoint is not None and i >= z_midpoint
        z_rise_fires = cur_z_above and not prev_z_above
        x_fires      = cur_trig0 != prev_trig0

        if mode == "batch":

            if z_mid_fires or z_rise_fires or x_fires:
                # Feed all samples accumulated since the last event as one batch.
                if i > seg_start:
                    proc.feed_samples(seg_start, pmt[seg_start:i])
                seg_start = i

                # ── z events ────────────────────────
                if z_mid_fires:  # start -1 halfcycle
                    proc.notify_z_hc_end(z_midpoint, -1)
                    z_midpoint = None

                if z_rise_fires: # start +1 halfcycle
                    proc.notify_z_hc_end(i, +1)
                    z_midpoint = i + int(proc.z_period_est / 2)

                # ── x event ──────────────────────────────────────────
                if x_fires:
                    n_xcycles += 1
                    x_dir_next = +1 if cur_trig0 == 1 else -1
                    for result in proc.notify_x_hc_end(i, x_dir_next):
                        n_coo_out += 1
                        _collect(result, vol_buf, save_base, scan_h, t)

            prev_trig0   = cur_trig0
            prev_z_above = cur_z_above

    # Sample-by-sample version (veeeeeery slow on Python; for FPGA implementation)
        elif mode == "one-by-one":    
            cur_trig0   = int(trig0[i])
            cur_z_above = int(tag[i]) > Z_THRESHOLD
    
            # ── z events ───────────
            if z_mid_fires:
                proc.notify_z_hc_end(z_midpoint, -1)
                z_midpoint = None
    
            if z_rise_fires:
                proc.notify_z_hc_end(i, +1)
                z_midpoint = i + int(proc.z_period_est / 2)
    
            # ── x event ────────────
            if x_fires:
                n_xcycles += 1
                x_dir_next = +1 if cur_trig0 == 1 else -1
                for result in proc.notify_x_hc_end(i, x_dir_next):
                    n_coo_out += 1
                    _collect(result, vol_buf, save_base, scan_h, t)
    
            # ── feed this sample ──────────────────────────────────────
            proc.feed_sample(i, int(pmt[i]))
    
            prev_trig0   = cur_trig0
            prev_z_above = cur_z_above


    # Feed remaining samples after the last event; not needed for one-by-one version
    if mode == "batch" and seg_start < n_samples:
        proc.feed_samples(seg_start, pmt[seg_start:])

    t['4_loop'] += time.perf_counter() - t0

    if not DROP_TAIL:
        # ── flush last partial halfcycle ─────────────────────────────────
        for result in proc.flush():
            _collect(result, vol_buf, save_base, scan_h, t)

        # ── save incomplete volumes ──────────────────────────────────────
        for lines in vol_buf.values():
            if lines:
                _save_volume(lines, save_base, t)

    print(f"[INFO] Done.  x_hcs={n_xcycles}  coo_out={n_coo_out}")
    print(f"COO output saved to: {save_base}")
    print_timing_summary(t, n_xcycles, n_coo_out)
    return dict(t)


# =========================
# Helpers
# =========================

def _collect(result, vol_buf, save_base, scan_h, t):
    '''
    buffering coo data in result into vol_buf
    if a full volume is collected, save it and clear the buffer
    '''
    vol_idx = result['volume_index']
    vol_buf[vol_idx].append(result)
    if len(vol_buf[vol_idx]) >= scan_h:
        _save_volume(vol_buf.pop(vol_idx), save_base, t)


def _save_volume(lines: list, save_base: Path, t: dict) -> None:
    t0      = time.perf_counter()
    vol_idx = lines[0]['volume_index']
    shape   = lines[0]['shape']
    vol_z   = np.concatenate([l['z'] for l in lines])
    vol_y   = np.concatenate([l['y'] for l in lines])
    vol_x   = np.concatenate([l['x'] for l in lines])
    vol_s   = np.concatenate([l['signal'] for l in lines])
    np.savez_compressed(
        save_base / f"vol_{vol_idx:04d}.npz",
        x=vol_x, y=vol_y, z=vol_z, signal=vol_s, shape=shape,
    )
    fill = 100 * len(vol_s) / np.prod(shape)
    print(f"[VOL {vol_idx}]  {len(vol_s)} non-zero voxels  ({fill:.1f}% fill)")
    t['5_save_npz'] += time.perf_counter() - t0


# =========================
# CLI
# =========================

def __main__():
    parser = argparse.ArgumentParser(
        description="Lissajous 3D reconstruction — ver9 streaming"
    )
    parser.add_argument("--dataset", "-d", nargs="+", type=str,
                        default=["1", "2", "3"], metavar="ID")
    parser.add_argument("--scan_h",        type=int,   default=SCAN_H)
    parser.add_argument("--scan_w",        type=int,   default=SCAN_W)
    parser.add_argument("--out_h",         type=int,   default=None)
    parser.add_argument("--out_w",         type=int,   default=None)
    parser.add_argument("--out_z",         type=int,   default=Z_SLICES)
    parser.add_argument("--nominal_x_len", type=float, default=NOMINAL_X_LEN)
    parser.add_argument("--nominal_z_period", type=float, default=NOMINAL_Z_LEN)
    parser.add_argument("--data_root",     type=str,   default=str(DATA_ROOT))
    parser.add_argument("--save_root",     type=str,   default=str(SAVE_ROOT))
    parser.add_argument("--mode",          type=str,   default="batch", choices=["one-by-one", "batch"])
    args = parser.parse_args()

    OUT_H = args.out_h if args.out_h is not None else args.scan_h
    OUT_W = args.out_w if args.out_w is not None else args.scan_w

    for ds in args.dataset:
        print(f"\n[INFO] Processing dataset {ds}")
        processDataset(
            dataset_name  = ds,
            z_slices      = args.out_z,
            scan_h        = args.scan_h,
            scan_w        = args.scan_w,
            out_h         = OUT_H,
            out_w         = OUT_W,
            nominal_x_len = args.nominal_x_len,
            nominal_z_period = args.nominal_z_period,
            data_root     = Path(args.data_root),
            save_root     = Path(args.save_root),
            mode          = args.mode,
        )


if __name__ == "__main__":
    __main__()