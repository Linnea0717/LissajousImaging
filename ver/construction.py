"""
construction.py
===============
模擬：把 raw .bin 資料切成 half-cycle，一個一個餵進 VolumeProcessor。

FPGA 廠商不需要參考這個檔案。
FPGA 廠商處理 raw data → half-cycle，再呼叫：
    proc.feed_z_halfcycle(zs, ze, z_dir)
    result = proc.feed_x_halfcycle(xs, xe, x_dir, signal, signal_offset)

輸出在 Step 5b 的 for 迴圈內：每個 x half-cycle 完成後立刻拿到該段 COO（coordinate list），
FPGA 整合時在此處改成 stream 到 DGX SPARK。
目前模擬：收集到 vol_buf，volume 填滿後存成 .npz。
"""

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
from parser import XHalfCycleParser, ZHalfCycleParser
from processor import VolumeProcessor

# =========================
# Parameters
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT    = PROJECT_ROOT / "20251229data"
SAVE_ROOT    = PROJECT_ROOT / "output" / "ver8"

SCAN_W        = 1024
SCAN_H        = 1024
Z_SLICES      = 31
Z_THRESHOLD   = 1000
COEFFS        = [0, 0, 0, 0, 0, 0]
CHUNK_SAMPLES = 100_000
DROP_BEFORE   = 20
DROP_AFTER    = 20


# =========================
# Main processing
# =========================

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
    data_dir  = data_root / dataset_name
    raw0_path = data_dir / "raw_data_0.bin"
    raw1_path = data_dir / "raw_data_1.bin"
    if not raw0_path.exists() or not raw1_path.exists():
        raise FileNotFoundError(f"Cannot find raw data in {data_dir}")

    save_base = save_root / dataset_name / f"x{out_w}y{out_h}z{z_slices}_coo"
    save_base.mkdir(parents=True, exist_ok=True)

    shifts = compute_shift_array(scan_h, scan_w, COEFFS)

    # ── raw data → half-cycle parsers ───────────────────────────
    x_parser = XHalfCycleParser()
    z_parser = ZHalfCycleParser(threshold=Z_THRESHOLD)

    # ── main ─────────────────────────────────────────────────────────
    proc = VolumeProcessor(
        z_slices         = z_slices,
        out_h            = out_h,
        out_w            = out_w,
        H_scan           = scan_h,
        W_scan           = scan_w,
        shifts           = shifts,
        lines_per_volume = scan_h,
        drop_before      = DROP_BEFORE,
        drop_after       = DROP_AFTER,
    )

    t            = defaultdict(float)
    n_chunks     = 0
    n_xcycles    = 0
    n_coo_out    = 0
    vol_buf: dict[int, list] = defaultdict(list)

    pmt_prev    = np.zeros(0, dtype=np.int16)
    prev_offset = 0

    print(f"[INFO] Dataset {dataset_name}: "
          f"{raw0_path.stat().st_size // 2:,} samples")

    for offset, chunk0, chunk1 in file_chunk_generator(
            raw0_path, raw1_path, chunk_samples):

        # ── Step 1: extract trig0 + PMT ──────────────────────────────────
        t0 = time.perf_counter()
        trig0, pmt_cur = extract_trigs_and_data14_signed(chunk0)
        t['1_extract_ch0'] += time.perf_counter() - t0

        # ── Step 2: extract TAG ───────────────────────────────────────────
        t0 = time.perf_counter()
        _, tag = extract_trigs_and_data14_signed(chunk1)
        t['2_extract_ch1'] += time.perf_counter() - t0

        # ── Step 3: 2-chunk PMT sliding window ───────────────────────────
        # x_half_len < chunk_size → 2 chunks enough to cover any half-cycle
        t0 = time.perf_counter()
        pmt_window    = np.concatenate([pmt_prev, pmt_cur])
        window_offset = prev_offset
        t['3_pmt_window'] += time.perf_counter() - t0

        # ── Step 4: parse half-cycles ─────────────────────────────────────
        t0 = time.perf_counter()
        x_hcs = x_parser.feed(trig0, offset)
        z_hcs = z_parser.feed(tag,   offset)
        t['4_parse'] += time.perf_counter() - t0

        # ── Step 5a: z half-cycles → VolumeProcessor ─────────────────────
        # z first, to ensure z_idx/zw are ready when x comes
        t0 = time.perf_counter()
        for z in z_hcs:
            proc.feed_z_halfcycle(int(z[0]), int(z[1]), int(z[2]))
        t['5a_feed_z'] += time.perf_counter() - t0

        # ── Step 5b: x half-cycles → VolumeProcessor → COO ───────────────
        t0 = time.perf_counter()
        for x in x_hcs:
            n_xcycles += 1
            result = proc.feed_x_halfcycle(
                int(x[0]), int(x[1]), int(x[2]),
                pmt_window, window_offset,
            )

            if result is not None:
                n_coo_out += 1

                # ── OUTPUT: get COO as soon as a x half-cycle is processed ────────────
                vol_idx = result['volume_index']
                vol_buf[vol_idx].append(result)
                if len(vol_buf[vol_idx]) >= scan_h:
                    _save_volume(vol_buf.pop(vol_idx), save_base, t)

        t['5b_feed_x'] += time.perf_counter() - t0

        pmt_prev    = pmt_cur
        prev_offset = offset
        n_chunks   += 1

        if n_chunks % 500 == 0:
            print(f"  [chunk {n_chunks}]  offset={offset:,}  "
                  f"x_hcs={n_xcycles}  coo_out={n_coo_out}")

    # 尾巴：未滿的 volume
    for lines in vol_buf.values():
        if lines:
            _save_volume(lines, save_base, t)

    print(f"[INFO] Done.  x_hcs={n_xcycles}  coo_out={n_coo_out}")
    print_timing_summary(t, n_chunks, n_coo_out)
    return dict(t)


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
    print(f"[VOL {vol_idx}]  {len(vol_s)} non-zero voxels  "
          f"({100 * len(vol_s) / np.prod(shape):.1f}% fill)")
    t['6_save_npz'] += time.perf_counter() - t0


# =========================
# CLI
# =========================

def __main__():
    parser = argparse.ArgumentParser(
        description="Lissajous 3D reconstruction — ver8 simulation"
    )
    parser.add_argument("--dataset",       nargs="+", type=str,
                        default=["1", "2", "3"], metavar="ID")
    parser.add_argument("--z_slices",      type=int, default=Z_SLICES)
    parser.add_argument("--scan_h",        type=int, default=SCAN_H)
    parser.add_argument("--scan_w",        type=int, default=SCAN_W)
    parser.add_argument("--out_h",         type=int, default=None)
    parser.add_argument("--out_w",         type=int, default=None)
    parser.add_argument("--data_root",     type=str, default=str(DATA_ROOT))
    parser.add_argument("--save_root",     type=str, default=str(SAVE_ROOT))
    parser.add_argument("--chunk_samples", type=int, default=CHUNK_SAMPLES)
    args = parser.parse_args()

    OUT_H = args.out_h if args.out_h is not None else args.scan_h
    OUT_W = args.out_w if args.out_w is not None else args.scan_w

    for ds in args.dataset:
        print(f"\n[INFO] Processing dataset {ds}")
        processDataset(
            dataset_name  = ds,
            z_slices      = args.z_slices,
            scan_h        = args.scan_h,
            scan_w        = args.scan_w,
            out_h         = OUT_H,
            out_w         = OUT_W,
            data_root     = Path(args.data_root),
            save_root     = Path(args.save_root),
            chunk_samples = args.chunk_samples,
        )


if __name__ == "__main__":
    __main__()
