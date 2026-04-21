"""
lut.py
======
Cosine/sine lookup table.

Pre-compute cos/sin for phases in [0, 1].
Resolution is controlled by LUT_SIZE.
"""

import numpy as np

LUT_SIZE = 4096

_lut_cos: np.ndarray = np.empty(0, dtype=np.float32)
_lut_sin: np.ndarray = np.empty(0, dtype=np.float32)


def rebuild_lut(size: int = LUT_SIZE) -> None:
    """Rebuild LUT. Call once when LUT_SIZE changes."""
    global LUT_SIZE, _lut_cos, _lut_sin
    LUT_SIZE = size
    theta    = np.linspace(0.0, np.pi, size, dtype=np.float64)
    _lut_cos = np.cos(theta).astype(np.float32)
    _lut_sin = np.sqrt(
        np.maximum(0.0, 1.0 - _lut_cos.astype(np.float64) ** 2)
    ).astype(np.float32)


def lut_lookup(phase: np.ndarray, direction: int) -> tuple[np.ndarray, np.ndarray]:
    p   = phase if direction == +1 else 1.0 - phase
    idx = np.clip(
        np.round(p * (LUT_SIZE - 1)).astype(np.int32), 0, LUT_SIZE - 1)
    return _lut_cos[idx], _lut_sin[idx]


# Initialise LUT on import
rebuild_lut(LUT_SIZE)
