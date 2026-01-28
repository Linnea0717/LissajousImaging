import argparse
import gc
from pathlib import Path
import numpy as np

from utils import (
    # read_raw_u16,
    read_raw_u16_mmap,
    extract_trigs_and_data14_signed,
    locateResonantTransitions,
    transitions2HalfCycles,
    locateTagRisingEdges,
    risingEdges2HalfCycles,
    saveXYFrames,
    saveXYZVolume_u16,
)

# from binning import XYZbinning, XYbinning

from binning_numba import (
    XYZbinning_numba,
)


# =========================
# Parameters
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "20251229data"

W = 1024
H = 1024

EXPECTED_HALFCYCLE_LEN = None
LEN_TOL = 0.5

DROP_BEFORE_FIRST_FRAME = 20
DROP_AFTER_EACH_FRAME   = 20
DROP_TAIL = True

Z_START_THRESHOLD = 1000       # channel1 > 1000 is considered the start of a z oscillation cycle (trigger)


def buildFrames(
    half_cycles, 
    LINES_PER_FRAME: int,
    DROP_BEFORE_FIRST_FRAME: int, 
    DROP_AFTER_EACH_FRAME: int, 
    drop_tail: bool
):
    frames = []
    cur = DROP_BEFORE_FIRST_FRAME
    N = len(half_cycles)

    while cur + LINES_PER_FRAME <= N:
        frame = half_cycles[cur:cur+LINES_PER_FRAME]
        frames.append(frame)

        cur += LINES_PER_FRAME + DROP_AFTER_EACH_FRAME

    if drop_tail is False and cur < N:
        frame = half_cycles[cur:N]
        frames.append(frame)
    
    return frames


def processDataset(
    dataset_name: str,
    z_slices: int,
):
    data_dir = DATA_ROOT / dataset_name
    save_dir = PROJECT_ROOT / "output" / "numba-batch" / dataset_name

    raw0_path = data_dir / "raw_data_0.bin"
    raw1_path = data_dir / "raw_data_1.bin"
    if not raw0_path.exists() or not raw1_path.exists():
        raise FileNotFoundError(f"Cannot find {raw0_path} or {raw1_path}")

    raw_u16_0 = read_raw_u16_mmap(raw0_path, endian="<u2")
    raw_u16_1 = read_raw_u16_mmap(raw1_path, endian="<u2")

    trig0, _, PMT = extract_trigs_and_data14_signed(raw_u16_0)
    _, _, TAG = extract_trigs_and_data14_signed(raw_u16_1)
    print(f"[INFO] Loaded {len(trig0)} samples from dataset {dataset_name}")


    transitions = locateResonantTransitions(trig0)
    x_half_cycles = transitions2HalfCycles(transitions, trig0)

    print(f"[INFO] Extracted {len(x_half_cycles)} half-cycles from data")

    frames_x_half_cycles = buildFrames(
        half_cycles=x_half_cycles,
        LINES_PER_FRAME=H,
        DROP_BEFORE_FIRST_FRAME=DROP_BEFORE_FIRST_FRAME,
        DROP_AFTER_EACH_FRAME=DROP_AFTER_EACH_FRAME,
        drop_tail=DROP_TAIL,
    )

    N_frames = len(frames_x_half_cycles)

    print(f"[INFO] Built {N_frames} frames from half-cycles")
    
    frames_t = [
        (frame[0][0], frame[-1][1]) for frame in frames_x_half_cycles
    ]

    z_edges = locateTagRisingEdges(signal=TAG, threshold=Z_START_THRESHOLD)
    z_half_cycles = risingEdges2HalfCycles(z_edges)

    print(f"[INFO] Extracted {len(z_half_cycles)} z half-cycles from TAG signal")

    frames_z_half_cycles = []
    for (t_start, t_end) in frames_t:
        frame_z_half_cycles = []
        for (zs, ze, z_dir) in z_half_cycles:
            if zs >= t_start and ze <= t_end:
                frame_z_half_cycles.append((zs, ze, z_dir))
        frames_z_half_cycles.append(frame_z_half_cycles)

    for i in range(N_frames):
        print(f"[INFO] Binning frame {i+1} / {len(frames_x_half_cycles)}")

        x_starts = frames_x_half_cycles[i][:, 0]
        x_ends = frames_x_half_cycles[i][:, 1]
        x_dirs = frames_x_half_cycles[i][:, 2]

        z_starts = np.array(frames_z_half_cycles[i])[:, 0]
        z_ends = np.array(frames_z_half_cycles[i])[:, 1]
        z_dirs = np.array(frames_z_half_cycles[i])[:, 2]

        count, volume = XYZbinning_numba(
            x_starts=x_starts, x_ends=x_ends, x_dirs=x_dirs,
            z_starts=z_starts, z_ends=z_ends, z_dirs=z_dirs,
            signal=PMT,
            H=H,
            W=W,
            Z=z_slices,
        )

        saveXYZVolume_u16(
            volume=volume,
            index=i,
            save_dir=save_dir / f"z{z_slices}",
        )

        del volume
        del count

        gc.collect()

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
            "  1. Process specific datasets with default settings:\n"
            "     python construction_stream.py --dataset 1 2\n\n"
            "  2. Process a dataset with high-resolution Z-axis (64 slices):\n"
            "     python construction_stream.py --dataset 3 --z_slices 64"
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
            "These directories must exist within the defined data root.\n"
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

    args = parser.parse_args()

    DATASET_DIR_NAMES = args.dataset
    Z_SLICES = args.z_slices

    for dataset_name in DATASET_DIR_NAMES:
        print(f"[INFO] Processing dataset {dataset_name}")
        processDataset(dataset_name=dataset_name, z_slices=Z_SLICES)

if __name__ == "__main__":
    __main__()
