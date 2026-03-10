from pathlib import Path
import subprocess
import sys
import re
import shutil

# =========================
# 路徑設定
# =========================
REPO_DIR = Path(r"C:\Users\HP\LissajousImaging")
SRC_ROOT = Path(r"C:\Users\HP\Desktop\Ian_lissajous\20260308\roi_flb")
DST_ROOT = Path(r"C:\Users\HP\Desktop\Ian_lissajous\20260308_processed\roi_flb")

# 你的 z-slices，若你平常不是 31，改這裡
Z_SLICES = 31

GROUPS = [
    "ztag5d_xy128",
    "ztag5d_xy256",
    "ztag5d_xy512",
    "ztag5d_xy1024",
]

TRIALS = [f"trial{i}" for i in range(1, 21)]


def parse_xy(group_name: str) -> int:
    m = re.search(r"xy(\d+)", group_name)
    if not m:
        raise ValueError(f"Cannot parse xy size from folder name: {group_name}")
    return int(m.group(1))


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_one_trial(group_name: str, trial_name: str):
    xy = parse_xy(group_name)

    src_group_dir = SRC_ROOT / group_name
    dst_group_dir = DST_ROOT / group_name
    ensure_dir(dst_group_dir)

    src_trial_dir = src_group_dir / trial_name
    if not (src_trial_dir / "raw_data_0.bin").exists():
        raise FileNotFoundError(f"Missing: {src_trial_dir / 'raw_data_0.bin'}")
    if not (src_trial_dir / "raw_data_1.bin").exists():
        raise FileNotFoundError(f"Missing: {src_trial_dir / 'raw_data_1.bin'}")

    cmd = [
        sys.executable,
        str(REPO_DIR / "run.py"),
        "-v", "v4",
        "-d", trial_name,
        "--z_slices", str(Z_SLICES),

        "--scan_h", str(xy),
        "--scan_w", str(xy),
        "--out_h", str(xy),
        "--out_w", str(xy),

        "--data_root", str(src_group_dir),
        "--save_root", str(dst_group_dir),
    ]

    print("\n==================================================")
    print(f"Running {group_name} / {trial_name}")
    print(" ".join([f'"{c}"' if " " in c else c for c in cmd]))
    print("==================================================")

    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def sorted_files(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file()])


def copy_concat_files(src_dirs, dst_dir: Path):
    """
    把多個來源資料夾中的檔案，依照 trial 順序 + 原始檔名排序，
    連續複製到 dst_dir，並重新編成 000000.ext, 000001.ext, ...
    """
    ensure_dir(dst_dir)
    counter = 0

    for src_dir in src_dirs:
        files = sorted_files(src_dir)
        for f in files:
            ext = f.suffix
            dst = dst_dir / f"{counter:06d}{ext}"
            shutil.copy2(f, dst)
            counter += 1

    return counter


def build_sum_for_group(group_name: str):
    xy = parse_xy(group_name)
    group_out_dir = DST_ROOT / group_name
    base_rel = Path(f"x{xy}y{xy}z{Z_SLICES}")

    sum_root = group_out_dir / "sum" / base_rel
    sum_xyzt_dir = sum_root / "xyzt_volumes"
    sum_xyt_root = sum_root / "xyt_frames"

    # 1) 合併 xyzt_volumes
    src_volume_dirs = []
    for trial_name in TRIALS:
        p = group_out_dir / trial_name / base_rel / "xyzt_volumes"
        if p.exists():
            src_volume_dirs.append(p)

    copied_vols = copy_concat_files(src_volume_dirs, sum_xyzt_dir)
    print(f"[SUM] {group_name}: copied {copied_vols} files into {sum_xyzt_dir}")

    # 2) 合併 xyt_frames
    #    先找所有 z-xxx 子資料夾名稱
    z_dir_names = set()
    for trial_name in TRIALS:
        xyt_root = group_out_dir / trial_name / base_rel / "xyt_frames"
        if xyt_root.exists():
            for sub in xyt_root.iterdir():
                if sub.is_dir():
                    z_dir_names.add(sub.name)

    for z_name in sorted(z_dir_names):
        src_frame_dirs = []
        for trial_name in TRIALS:
            p = group_out_dir / trial_name / base_rel / "xyt_frames" / z_name
            if p.exists():
                src_frame_dirs.append(p)

        dst_z_dir = sum_xyt_root / z_name
        copied_frames = copy_concat_files(src_frame_dirs, dst_z_dir)
        print(f"[SUM] {group_name}: {z_name} copied {copied_frames} files into {dst_z_dir}")


def main():
    ensure_dir(DST_ROOT)

    # 先逐 trial 跑
    for group_name in GROUPS:
        for trial_name in TRIALS:
            run_one_trial(group_name, trial_name)

    # 再建立 sum
    for group_name in GROUPS:
        build_sum_for_group(group_name)

    print("\nAll done.")


if __name__ == "__main__":
    main()