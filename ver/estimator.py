"""
estimator.py
============
Sliding-window halfcycle length estimator.

The window is pre-filled with a default value. 
Each completed halfcycle pushes out the oldest entry and pulls in the measured length.
"""

from collections import deque


class HalfCycleLengthEstimator:
    """
    Parameters
    ----------
    default_length : float
        Nominal halfcycle length (in samples) used to pre-fill the window.
    window : int
        Width of the sliding average.
    """

    def __init__(self, default_length: float, window: int = 6):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._hist = deque([float(default_length)] * window, maxlen=window)
        self._sum  = float(default_length) * window

    def update(self, true_length: int) -> None:
        """
        Updates the sliding window with the new measurement.
        """
        self._sum -= self._hist[0]          # evict oldest
        self._hist.append(float(true_length))
        self._sum += float(true_length)

    @property
    def estimate(self) -> float:
        """Current sliding-window average."""
        return self._sum / len(self._hist)
