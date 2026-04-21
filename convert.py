"""
coo_to_volume.py
================
Convert COO-format .npz files (output of construction.py) into dense 3D volumes
and save them as ImageJ-compatible TIFF stacks (ZYX axis order).

Each vol_*.npz contains:
    x       : uint16[K]   x coordinates of non-zero voxels
    y       : uint16[K]   y coordinates
    z       : uint16[K]   z coordinates
    signal  : float32[K]  weighted-average PMT signal
    shape   : int32[3]    [z_slices, out_h, out_w]

Usage
-----
    # Convert all volumes in a COO directory:
    python coo_to_volume.py --coo_dir output/ver9/1/x1024y1024z31_coo
                            --out_dir output/ver9/1/tiff

    # Keep float32 .npy files as well (useful for denoising pipeline):
    python coo_to_volume.py --coo_dir ... --out_dir ... --save_npy

    # Adjust percentile normalization:
    python coo_to_volume.py --coo_dir ... --out_dir ... --pmin 0 --pmax 99.5

Output
------
    out_dir/vol_0000.tiff   — uint16, shape (Z, Y, X), ImageJ metadata
    out_dir/vol_0000.npy    — float32, shape (Z, Y, X)  [only with --save_npy]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import tifffile


# =========================
# Core conversion
# =========================

def coo_to_dense(npz_path: Path) -> np.ndarray:
    """
    Load one vol_*.npz and scatter its COO entries into a dense float32 volume.

    When out_h < H_scan, multiple scan lines share the same y_idx, so the
    same (z, y, x) voxel may appear more than once. Those contributions are
    summed and then divided by their count to give a proper average.

    Returns
    -------
    vol : float32 array, shape (z_slices, out_h, out_w)
    """
    d      = np.load(npz_path)
    shape  = tuple(int(v) for v in d['shape'])   # (z_slices, out_h, out_w)

    z_c = d['z'].astype(np.intp)
    y_c = d['y'].astype(np.intp)
    x_c = d['x'].astype(np.intp)
    sig = d['signal'].astype(np.float32)

    vol   = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.int32)

    np.add.at(vol,   (z_c, y_c, x_c), sig)
    np.add.at(count, (z_c, y_c, x_c), 1)

    mask       = count > 0
    vol[mask] /= count[mask]

    return vol.astype(np.float32)


def normalize_uint16(
    vol: np.ndarray,
    pmin: float = 0.0,
    pmax: float = 99.9,
) -> np.ndarray:
    """
    Percentile-stretch a float32 volume to uint16.

    Only non-zero voxels are used to compute the percentile range so that
    empty (background) regions do not drag down the stretch.
    If the volume is entirely zero, returns an all-zero uint16 array.
    """
    nonzero = vol[vol > 0]
    if nonzero.size == 0:
        return np.zeros(vol.shape, dtype=np.uint16)

    lo = float(np.percentile(nonzero, pmin))
    hi = float(np.percentile(nonzero, pmax))

    if hi <= lo:
        return np.zeros(vol.shape, dtype=np.uint16)

    stretched = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    return (stretched * 65535).astype(np.uint16)


# =========================
# I/O helpers
# =========================

def save_tiff(vol_u16: np.ndarray, path: Path) -> None:
    """Save (Z, Y, X) uint16 array as an ImageJ-compatible TIFF stack."""
    tifffile.imwrite(
        str(path),
        vol_u16,
        imagej=True,
        metadata={'axes': 'ZYX'},
    )


def save_npy(vol_f32: np.ndarray, path: Path) -> None:
    """Save (Z, Y, X) float32 array as .npy."""
    np.save(str(path), vol_f32)


# =========================
# Main
# =========================

def convert_directory(
    coo_dir: Path,
    out_dir: Path,
    pmin:     float = 0.0,
    pmax:     float = 99.9,
    save_npy: bool  = False,
) -> None:
    npz_files = sorted(coo_dir.glob("vol_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No vol_*.npz files found in {coo_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] {len(npz_files)} volumes found in {coo_dir}")
    print(f"[INFO] Output → {out_dir}")

    t_total = 0.0
    for npz_path in npz_files:
        t0 = time.perf_counter()

        vol_f32  = coo_to_dense(npz_path)
        vol_u16  = normalize_uint16(vol_f32, pmin=pmin, pmax=pmax)

        stem     = npz_path.stem                     # e.g. "vol_0003"
        tiff_path = out_dir / f"{stem}.tiff"
        save_tiff(vol_u16, tiff_path)

        if save_npy:
            npy_path = out_dir / f"{stem}.npy"
            save_npy(vol_f32, npy_path)

        elapsed  = time.perf_counter() - t0
        t_total += elapsed
        fill_pct = 100.0 * np.count_nonzero(vol_f32) / vol_f32.size
        print(f"  {stem}  shape={vol_f32.shape}  "
              f"fill={fill_pct:.1f}%  "
              f"signal=[{vol_f32[vol_f32 > 0].min():.3f}, {vol_f32.max():.3f}]  "
              f"({elapsed:.2f} s)")

    print(f"\n[INFO] Done. {len(npz_files)} volumes in {t_total:.1f} s "
          f"({t_total / len(npz_files):.2f} s/vol)")


def __main__():
    parser = argparse.ArgumentParser(
        description="Convert COO .npz volumes to dense TIFF stacks"
    )
    parser.add_argument(
        "--coo_dir", required=True, type=Path,
        help="Directory containing vol_*.npz files (output of construction.py)",
    )
    parser.add_argument(
        "--out_dir", required=True, type=Path,
        help="Directory to write TIFF (and optionally .npy) files",
    )
    parser.add_argument(
        "--pmin", type=float, default=0.0,
        help="Lower percentile for uint16 normalization (default: 0.0)",
    )
    parser.add_argument(
        "--pmax", type=float, default=99.9,
        help="Upper percentile for uint16 normalization (default: 99.9)",
    )
    parser.add_argument(
        "--save_npy", action="store_true",
        help="Also save float32 .npy files alongside the TIFFs",
    )
    args = parser.parse_args()

    convert_directory(
        coo_dir  = args.coo_dir,
        out_dir  = args.out_dir,
        pmin     = args.pmin,
        pmax     = args.pmax,
        save_npy = args.save_npy,
    )


if __name__ == "__main__":
    __main__()