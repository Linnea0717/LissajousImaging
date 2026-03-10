import argparse
import subprocess
import sys
import time
from pathlib import Path

VERSION_MAP = {
    "v1": "ver1_numpy/construction_numpy.py",
    "v2": "ver2_numba/construction_numba.py",
    "v3": "ver3_numba_streaming/construction_stream.py",
    "v4": "ver4_online_idx/construction_stream.py",
    "v5": "ver5_cuda/construction_cuda.py",
}

import argparse
import subprocess
import sys
import time
from pathlib import Path

VERSION_MAP = {
    "v1": "ver1_numpy/construction_numpy.py",
    "v2": "ver2_numba/construction_numba.py",
    "v3": "ver3_numba_streaming/construction_stream.py",
    "v4": "ver4_online_idx/construction_stream.py",
    "v5": "ver5_cuda/construction_cuda.py",
}

def run_experiment(
    versions,
    datasets,
    scan_h,
    scan_w,
    out_h,
    out_w,
    z_slices,
    data_root,
    save_root,
    quiet=False
):
    base_dir = Path(__file__).parent.resolve()

    if out_h is None:
        out_h = scan_h
    if out_w is None:
        out_w = scan_w

    print("=" * 60)
    print(f"Versions: {versions}")
    print(f"Datasets: {datasets}")
    print(f"scan_h x scan_w = {scan_h} x {scan_w}")
    print(f"out_h  x out_w  = {out_h} x {out_w}")
    print(f"z_slices = {z_slices}")
    print(f"data_root = {data_root}")
    print(f"save_root = {save_root}")
    print(f'Quiet Mode: {"ON" if quiet else "OFF"}')
    print("=" * 60)

    for ver_key in versions:
        if ver_key not in VERSION_MAP:
            print(f"[-] Cannot find version key '{ver_key}', skipping.")
            continue

        script_path = base_dir / VERSION_MAP[ver_key]
        if not script_path.exists():
            print(f"[-] Cannot find file: {script_path}, please check the path configuration.")
            continue

        cmd = [
            sys.executable,
            str(script_path),
            "--dataset", *datasets,
            "--z_slices", str(z_slices),
            "--scan_h", str(scan_h),
            "--scan_w", str(scan_w),
            "--out_h", str(out_h),
            "--out_w", str(out_w),
            "--data_root", str(data_root),
            "--save_root", str(save_root),
        ]

        start_time = time.time()
        try:
            if quiet:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            else:
                subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[-] Version {ver_key} failed (Exit Code: {e.returncode})")
        except KeyboardInterrupt:
            print("\n[-] User interrupted execution")
            sys.exit(0)
        else:
            elapsed = time.time() - start_time
            print(f"[+] Version {ver_key} completed. Time elapsed: {elapsed:.2f} seconds")
            print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--versions", nargs="+", choices=VERSION_MAP.keys(), required=True)
    parser.add_argument("-d", "--dataset", nargs="+", default=["1"])
    parser.add_argument("-z", "--z_slices", type=int, default=31)
    parser.add_argument("-q", "--quiet", action="store_true")

    parser.add_argument("--scan_h", type=int, required=True, help="Scan-space height")
    parser.add_argument("--scan_w", type=int, required=True, help="Scan-space width")
    parser.add_argument("--out_h", type=int, default=None, help="Output height; default = scan_h")
    parser.add_argument("--out_w", type=int, default=None, help="Output width; default = scan_w")

    parser.add_argument("--data_root", type=str, required=True, help="Root folder containing dataset subfolders")
    parser.add_argument("--save_root", type=str, required=True, help="Root folder to save outputs")

    args = parser.parse_args()

    run_experiment(
        versions=args.versions,
        datasets=args.dataset,
        scan_h=args.scan_h,
        scan_w=args.scan_w,
        out_h=args.out_h,
        out_w=args.out_w,
        z_slices=args.z_slices,
        data_root=args.data_root,
        save_root=args.save_root,
        quiet=args.quiet,
    )
