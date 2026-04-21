"""
estimator.py
============
Sliding-window halfcycle length estimator.

The window is pre-filled with a default value so the estimator is valid
from sample zero — no bootstrap hold-off needed. Each completed halfcycle
pushes out the oldest entry and pulls in the true measured length.

Convergence example (window=6, default=900, true length=1000):

  before hc 0 : estimate = 900.0   (all slots = default)
  after  hc 0 : estimate = 916.7   (5×900 + 1×1000) / 6
  after  hc 1 : estimate = 933.3
  ...
  after  hc 5 : estimate = 1000.0  (fully converged, all defaults evicted)
"""

from collections import deque


class HalfCycleLengthEstimator:
    """
    Parameters
    ----------
    default_length : float
        Nominal halfcycle length (in samples) used to pre-fill the window.
        Derive from: sample_rate / (2 * scan_frequency).
    window : int
        Number of past halfcycles kept in the sliding average.
    """

    def __init__(self, default_length: float, window: int = 6):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._hist = deque([float(default_length)] * window, maxlen=window)
        self._sum  = float(default_length) * window

    def update(self, true_length: int) -> None:
        """
        Call once per completed halfcycle with its measured length.
        O(1): evict oldest entry and add new one without re-summing.
        """
        self._sum -= self._hist[0]          # evict oldest
        self._hist.append(float(true_length))
        self._sum += float(true_length)

    @property
    def estimate(self) -> float:
        """Current sliding-window average. Always valid (never zero)."""
        return self._sum / len(self._hist)
