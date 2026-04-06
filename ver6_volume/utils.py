import numpy as np
from pathlib import Path
import tifffile as tiff

# =========================
# Data extraction
# =========================
def extract_trigs_and_data14_signed(raw_u16: np.ndarray):
    trig0 = (raw_u16 >> 15) & 0x1
    trig1 = (raw_u16 >> 14) & 0x1
    data14_u = raw_u16 & 0x3FFF
    data14   = data14_u.astype(np.int16)
    data14   = np.where((data14_u & 0x2000) != 0, data14 - 0x4000, data14)
    return trig0.astype(np.uint8), trig1.astype(np.uint8), data14


def read_raw_u16_mmap(path: Path, endian="<u2") -> np.ndarray:
    return np.memmap(path, dtype=np.dtype(endian), mode='r')

def read_raw_u16(path: Path, endian="<u2") -> np.ndarray:
    return np.fromfile(path.as_posix(), dtype=np.dtype(endian))


# =========================
# Locate Transitions
# =========================
def locateResonantTransitions(trig0: np.ndarray) -> np.ndarray:
    '''
    return the indices where the signal changes value
    '''
    s = trig0.astype(np.uint8)
    return np.flatnonzero(s[1:] != s[:-1]) + 1

def transitions2HalfCycles(
    transitions: np.ndarray,
    trig0: np.ndarray,
) -> np.ndarray:
    '''
    Convert transitions to half-cycles.
    Each pair of transitions defines a half-cycle.
    '''
    half_cycles = []
    N = len(transitions)
    for i in range(N - 1):
        s = transitions[i]
        e = transitions[i + 1]
        dir = +1 if int(trig0[s]) == 1 else -1
        half_cycles.append((s, e, dir))

    return np.array(half_cycles, dtype=np.int64)

def locateTagRisingEdges(signal: np.ndarray, threshold: int) -> np.ndarray:
    '''
    return the indices where the signal rises above the threshold
    '''
    s = signal.astype(np.int32)
    above_thresh = s > threshold

    starts = []
    if bool(above_thresh[0]): starts.append(0)

    rising_edges = np.flatnonzero(above_thresh[1:] & (~above_thresh[:-1])) + 1
    starts.extend(rising_edges.tolist())

    return np.array(starts, dtype=np.int64)

def risingEdges2HalfCycles(
    rising_edges: np.ndarray,
) -> np.ndarray:
    '''
    Convert rising edges to half-cycles.
    2 half-cycles between each pair of rising edges.
    '''
    half_cycles = []
    N = len(rising_edges)
    for i in range(N - 1):
        s = rising_edges[i]
        e = rising_edges[i + 1]
        m = (s + e) // 2
        half_cycles.append((s, m, +1))
        half_cycles.append((m, e, -1))
    return np.array(half_cycles, dtype=np.int64)


def compute_shift_array(H, W, coefficients):
    """
    coefficients: [offset, shift, edge1, edge2, start, gap]
    """
    offset = coefficients[0]
    shift_gain = coefficients[1]
    edge1 = coefficients[2]
    edge2 = coefficients[3]

    f = 1 / (H * 2) 
    t = np.arange(0, H)
    
    val_sin = np.sin(2 * np.pi * f * t - np.pi / 2)

    y_sin4 = np.zeros(H)
    mid = H // 2
    
    y_sin4[:mid] = (val_sin[:mid] ** 4) * edge1
    y_sin4[mid:] = (val_sin[mid:] ** 4) * edge2
    
    final_shift = shift_gain * y_sin4 + offset
    
    scale_factor = W / 1024.0
    final_shift_scaled = final_shift * scale_factor
    
    return final_shift_scaled.astype(np.float32)


# =========================
# Image Saving
# =========================

def rescale(image, min_per=0.0, max_per=99.9, new_min=0.0, new_max=32767.0):
    old_min = np.percentile(image, min_per)
    old_max = np.percentile(image, max_per)
    
    image_norm = (image - old_min) / (old_max - old_min)
    image_norm = np.clip(image_norm, 0.0, 1.0)
    image_rescaled = image_norm * (new_max - new_min) + new_min

    return image_rescaled

def saveXYFrame_u16(
    image,
    save_dir: Path,
    index: int,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    image_u16 = rescale(image).astype(np.uint16)

    save_path = save_dir / f"frame_{index:05d}.tiff"
    tiff.imwrite(save_path.as_posix(), image_u16, imagej=True, metadata={'axes': 'YX'})

    # print(f"[INFO] Saved frame {index} to {save_path}")

def saveXYZVolume_u16(
    volume,
    index: int,
    save_dir: Path,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    vol_u16 = rescale(volume).astype(np.uint16)

    save_path = save_dir / f"volume_{index:05d}.tiff"
    tiff.imwrite(save_path.as_posix(), vol_u16, imagej=True, metadata={'axes': 'ZYX'})

    print(f"[SAVE] {save_path} {index}  shape={vol_u16.shape}")