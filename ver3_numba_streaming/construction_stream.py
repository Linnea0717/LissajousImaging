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


def streamFramesGenerator(
    x_half_cycles,
    z_half_cycles,
    lines_per_frame: int,
    drop_before: int,
    drop_after: int,
    drop_tail: bool
):
    total_x_cycles = len(x_half_cycles)
    cur_x_idx = drop_before
    frame_count = 0
    
    cur_z_idx = 0  # index for z half-cycles
    num_z_half_cycles = len(z_half_cycles)

    while cur_x_idx + lines_per_frame <= total_x_cycles:

        # x half cycles for this frame
        frame_x_half_cycles = x_half_cycles[cur_x_idx: cur_x_idx + lines_per_frame]
        
        t_start = frame_x_half_cycles[0][0]
        t_end = frame_x_half_cycles[-1][1]
        
        # z half cycles within this frame
        frame_z_half_cycles = []
        
        # move cursor to the first possible overlapping z half-cycle
        while cur_z_idx < num_z_half_cycles and z_half_cycles[cur_z_idx][1] < t_start:
            cur_z_idx += 1
            
        # find all overlapping z half-cycles
        temp_idx = cur_z_idx
        while temp_idx < num_z_half_cycles:
            zs, ze, zdir = z_half_cycles[temp_idx]
            if zs > t_end: # beyond current frame
                break
            
            # check overlap
            if not (ze <= t_start or zs >= t_end):
                frame_z_half_cycles.append((zs, ze, zdir))
            
            temp_idx += 1
            
        # Yield
        yield frame_count, frame_x_half_cycles, frame_z_half_cycles
        
        # update counters
        frame_count += 1
        cur_x_idx += lines_per_frame + drop_after


def processDataset(dataset_name: str, z_slices: int):
    data_dir = DATA_ROOT / dataset_name
    save_dir = PROJECT_ROOT / "output" / "numba-streaming" / dataset_name

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

    print(f"[INFO] Extracted {len(x_half_cycles)} x half-cycles from data")

    rising_edges = locateTagRisingEdges(TAG, Z_START_THRESHOLD)
    z_half_cycles = risingEdges2HalfCycles(rising_edges)
    print(f"[INFO] Extracted {len(z_half_cycles)} z half-cycles from data")


    print(f"[INFO] Start streaming frames...")
    
    frame_gen = streamFramesGenerator(
        x_half_cycles=x_half_cycles,
        z_half_cycles=z_half_cycles,
        lines_per_frame=H,
        drop_before=DROP_BEFORE_FIRST_FRAME,
        drop_after=DROP_AFTER_EACH_FRAME,
        drop_tail=DROP_TAIL
    )
    
    for i, frame_x, frame_z in frame_gen:
        print(f"[INFO] Processing Frame {i} | Z-cycles: {len(frame_z)}")

        x_starts = frame_x[:, 0].astype(np.int64)
        x_ends = frame_x[:, 1].astype(np.int64)
        x_dirs = frame_x[:, 2].astype(np.int32)
        
        frame_z = np.array(frame_z, dtype=np.int64)
        z_starts = frame_z[:, 0]
        z_ends = frame_z[:, 1]
        z_dirs = frame_z[:, 2].astype(np.int32)
        
        count, volume = XYZbinning_numba(
            x_starts=x_starts, x_ends=x_ends, x_dirs=x_dirs,
            z_starts=z_starts, z_ends=z_ends, z_dirs=z_dirs,
            signal=PMT,
            H=H, W=W, Z=z_slices
        )
        
        saveXYZVolume_u16(
            volume=volume,
            index=i,
            save_dir=save_dir / f'z{z_slices}',
        )
        
        del volume, count
        
        if i % 5 == 0:
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
