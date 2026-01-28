import argparse
import subprocess
import sys
import time
from pathlib import Path

VERSION_MAP = {
    "v1": "ver1_numpy/construction_numpy.py",
    "v2": "ver2_numba/construction_numba.py",
    "v3": "ver3_numba_streaming/construction_stream.py",
}

def run_experiment(versions, datasets, z_slices, quiet=False):
    base_dir = Path(__file__).parent.resolve()

    print("="*60)
    print(f"Versions: {versions}")
    print(f"Datasets: {datasets}")
    print(f"Z-Slices: {z_slices}")
    print(f'Quiet Mode: {"ON" if quiet else "OFF"}')
    print("="*60)

    for ver_key in versions:
        if ver_key not in VERSION_MAP:
            print(f"[-] Cannot find version key '{ver_key}', skipping.")
            continue

        script_path = base_dir / VERSION_MAP[ver_key]
        
        if not script_path.exists():
            print(f"[-] Cannot find file: {script_path}, please check the path configuration.")
            continue

        print(f"\n[+] Running version: [{ver_key}]")
        print(f"      Path: {script_path.relative_to(base_dir)}")
        
        cmd = [
            sys.executable,
            str(script_path),
            "--dataset", *datasets,
            "--z_slices", str(z_slices)
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

    parser = argparse.ArgumentParser(
        description=(
            "Automated Benchmark Runner for Bio-optics Image Reconstruction.\n"
            "Executes specified algorithm versions against selected datasets and\n"
            "measures execution performance."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "EXAMPLES:\n"
            "  1. Compare v2 and v3 on dataset 1:\n"
            "     python run_benchmark.py -v v2 v3 -d 1\n\n"
            "  2. Run v3 on multiple datasets with custom Z-slices:\n"
            "     python run_benchmark.py -v v3 -d 1 2 3 --z_slices 64\n\n"
            "  3. Run in quiet mode (output only execution time):\n"
            "     python run_benchmark.py -v v3 -d 1 -q\n"
        )
    )

    parser.add_argument(
        "-v", "--versions", 
        nargs="+", 
        choices=VERSION_MAP.keys(),
        required=True,
        metavar="VER",
        help="Specify one or more algorithm versions.\nAvailable options: " + ", ".join(VERSION_MAP.keys())
    )

    parser.add_argument(
        "-d", "--dataset", 
        nargs="+", 
        default=["1"],
        metavar="ID",
        help="Specify one or more dataset IDs to process. (Default: 1)"
    )

    parser.add_argument(
        "-z", "--z_slices", 
        type=int, 
        default=31,
        metavar="N",
        help="Set the number of Z-axis slices for volume reconstruction. (Default: 31)"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Enable quiet mode. Suppresses standard output logs from\nthe subprocesses and displays only the final execution time."
    )

    args = parser.parse_args()

    run_experiment(args.versions, args.dataset, args.z_slices, args.quiet)