"""
utils.py
========
Shared utilities for ver9.
Only functions actually used in this version are kept.
"""

import numpy as np
from pathlib import Path


def read_raw_u16_mmap(path: Path, endian: str = "<u2") -> np.ndarray:
    """Memory-map a raw uint16 binary file (read-only)."""
    return np.memmap(path, dtype=np.dtype(endian), mode='r')


def extract_trigs_and_data14_signed(
    raw_u16: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unpack a uint16 raw channel:
      trig0  : uint8  — bit 15 (x trigger or TAG gate)
      data14 : int16  — bits 13:0, sign-extended from 14-bit two's complement
    """
    trig0    = ((raw_u16 >> 15) & 0x1).astype(np.uint8)
    data14_u = raw_u16 & 0x3FFF
    data14   = data14_u.astype(np.int16)
    data14   = np.where((data14_u & 0x2000) != 0, data14 - 0x4000, data14)
    return trig0, data14


def compute_shift_array(
    scan_h: int,
    scan_w: int,
    coeffs: list,
) -> np.ndarray:
    """
    Per-scan-line x shift in input pixel units.
    coeffs : polynomial coefficients [c0, c1, ...] applied to scan-line index.
    All zeros → no shift.
    """
    y     = np.arange(scan_h, dtype=np.float64)
    shift = np.zeros(scan_h, dtype=np.float64)
    for power, c in enumerate(coeffs):
        shift += c * (y ** power)
    return shift


def print_timing_summary(t: dict, n_xcycles: int, n_coo_out: int) -> None:
    total = sum(t.values())
    print("\n─── Timing summary ───────────────────────────────────────")
    for key in sorted(t):
        frac = 100 * t[key] / total if total > 0 else 0
        print(f"  {key:<30s}  {t[key]:8.3f} s  ({frac:5.1f}%)")
    print(f"  {'TOTAL':<30s}  {total:8.3f} s")
    print(f"  {n_xcycles} x halfcycles,  {n_coo_out} COO outputs")
    print("──────────────────────────────────────────────────────────\n")
