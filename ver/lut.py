"""
lut.py
======
Cosine/sine lookup table。

預先算好相位 [0, 1] 對應的 cos/sin 值
相位切分多細可用 LUT_SIZE 調整
"""

import numpy as np

LUT_SIZE = 4096

_lut_cos: np.ndarray = np.empty(0, dtype=np.float32)
_lut_sin: np.ndarray = np.empty(0, dtype=np.float32)


def rebuild_lut(size: int = LUT_SIZE) -> None:
    """建立 LUT。size 改變時呼叫一次即可。"""
    global LUT_SIZE, _lut_cos, _lut_sin
    LUT_SIZE = size
    theta    = np.linspace(0.0, np.pi, size, dtype=np.float64)
    _lut_cos = np.cos(theta).astype(np.float32)
    _lut_sin = np.sqrt(
        np.maximum(0.0, 1.0 - _lut_cos.astype(np.float64) ** 2)
    ).astype(np.float32)


def lut_lookup(phase: np.ndarray, direction: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized LUT lookup。

    Parameters
    ----------
    phase     : float64 array, values in [0, 1]
    direction : +1 or -1

    Returns
    -------
    cos_vals : float32 array — cos(θ), 即 physical position in [-1, 1]
    sin_vals : float32 array — sin(θ), 即 weighting factor
    """
    p = phase if direction == +1 else 1.0 - phase
    idx = np.clip(
        np.round(p * (LUT_SIZE - 1)).astype(np.int32), 0, LUT_SIZE - 1)
    return _lut_cos[idx], _lut_sin[idx]


# intialize LUT when module is imported
rebuild_lut(LUT_SIZE)
