import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from utils import locateResonantTransitions


def resonantGlobalCosFit(trig0: np.ndarray, N: int = 100, T_MAX: int = 100_000):
    '''
    Fit trig0 to a sine wave: y(t) = sin(ω t + φ)
    Return ω, φ, and phase_fit (array of phase values for each t)

    Assuming the resonant galvo follows a sinusoidal motion.

    Procedure:
    1. Estimate an initial resonant frequency from the average half-cycle length
       derived from trig0 transitions.
    2. Treat trig0 as a sign constraint (+1 / -1) rather than a full waveform.
    3. Fit the phase offset (phi) by minimizing the squared error between
       the sign signal and a sinusoidal model with fixed frequency.
    4. Return the estimated angular frequency, phase offset, and per-sample phase.
    '''

    edges = locateResonantTransitions(trig0)
    # print(f"[DEBUG] number of resonant edges = {len(edges)}")

    half_cycles = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]

    lengths = [e - s for (s, e) in half_cycles]
    mean_halfcycle = np.mean(lengths)

    period = mean_halfcycle * 2.0
    omega0 = 2.0 * np.pi / period

    # print(f"[DEBUG] mean_halfcycle = {mean_halfcycle}")
    # print(f"[DEBUG] omega0 = {omega0}")

    # y(t) ≈ cos(ω t + φ)
    y = np.where(trig0 == 0, +1.0, -1.0)
    t = np.arange(len(trig0), dtype=np.float64)

    y_sub = y[::T_MAX]
    t_sub = t[::T_MAX]

    candidate_phis = np.linspace(0, 2*np.pi, num=N, endpoint=False)
    best_phi = 0.0
    best_err = float('inf')

    # for phi in candidate_phis:
    #     y_fit = np.cos(omega0 * t_sub + phi)
    #     err = np.mean((y_sub - y_fit) ** 2)
    #     if err < best_err:
    #         best_err = err
    #         best_phi = phi
    
    phase_fit = (omega0 * t + best_phi) % (2 * np.pi)

    return omega0, best_phi, phase_fit


def splitHalfCyclesFromPhase(phase: np.ndarray, min_len: int = 10):
    """
    Split continuous phase (0 ~ 2π) into half-cycles.

    A half-cycle is defined as a continuous region where:
        floor(phase / π) is constant.

    Parameters
    ----------
    phase : np.ndarray
        Phase array in [0, 2π), shape (N,).
    min_len : int
        Minimum allowed length (in samples) of a half-cycle.

    Returns
    -------
    half_cycles : list of (start, end, direction)
        start : inclusive index
        end   : exclusive index
        direction : +1 for forward (0 → π), -1 for backward (π → 2π)
    """

    if phase.ndim != 1:
        raise ValueError("phase must be a 1D array")

    N = len(phase)
    if N == 0:
        return []

    half_idx = np.floor(phase / np.pi).astype(np.int32)

    change_points = np.flatnonzero(half_idx[1:] != half_idx[:-1]) + 1

    boundaries = np.concatenate(([0], change_points, [N]))

    half_cycles = []

    for i in range(len(boundaries) - 1):
        s = int(boundaries[i])
        e = int(boundaries[i + 1])
        if e - s < min_len:
            continue

        mid = (s + e) // 2
        direction = +1 if phase[mid] < np.pi else -1

        half_cycles.append((s, e, direction))

    return half_cycles


class ResonantPhaseTracker:
    """
    Incremental (causal) sliding-window phase correction
    for resonant galvo phase tracking.

    Parameters
    ----------
    window_size_lines : int
        Number of half-cycles to include in a sliding window.
    N_phi : int
        Number of candidate phase offsets Δφ to evaluate.
    sub_stride : int
        Subsampling stride for efficiency.
    """

    def __init__(
        self,
        window_size_lines: int = 5,
        N_phi: int = 50,
        sub_stride: int = 10,
    ):
        # configuration
        self.window_size_lines = window_size_lines
        self.N_phi = N_phi
        self.sub_stride = sub_stride

        # buffers (state)
        self.phase_buf = deque()
        self.y_buf = deque()

        # candidate phase offsets
        self.candidate_dphi = np.linspace(
            -np.pi, np.pi, N_phi, endpoint=False
        )
        self.cos_dphi = np.cos(self.candidate_dphi)
        self.sin_dphi = np.sin(self.candidate_dphi)

    def reset(self):
        """Reset internal state (e.g. when starting a new dataset)."""
        self.phase_buf.clear()
        self.y_buf.clear()

    def update(
        self,
        phase_line: np.ndarray,
        trig0_line: np.ndarray,
    ) -> float:
        """
        Add one half-cycle and estimate phase correction Δφ
        for the current line.

        Parameters
        ----------
        phase_line : np.ndarray
            Phase values for the current half-cycle.
        trig0_line : np.ndarray
            Trig0 values for the current half-cycle.

        Returns
        -------
        delta_phi : float
            Estimated phase correction (rad).
        """

        # convert trig0 to ±1
        y_line = np.where(trig0_line == 0, +1.0, -1.0)

        # add new half-cycle to buffer
        self.phase_buf.append(phase_line)
        self.y_buf.append(y_line)

        # maintain window size
        if len(self.phase_buf) > self.window_size_lines:
            self.phase_buf.popleft()
            self.y_buf.popleft()

        # concatenate window samples
        phase_w = np.concatenate(self.phase_buf)[:: self.sub_stride]
        y_w = np.concatenate(self.y_buf)[:: self.sub_stride]

        if phase_w.size == 0:
            return 0.0

        # precompute sin / cos once
        sin_phase = np.sin(phase_w)
        cos_phase = np.cos(phase_w)

        # brute-force Δφ search
        best_err = np.inf
        best_dphi = 0.0

        for k in range(self.N_phi):
            y_fit = (
                sin_phase * self.cos_dphi[k]
                + cos_phase * self.sin_dphi[k]
            )
            err = np.mean((y_w - y_fit) ** 2)
            if err < best_err:
                best_err = err
                best_dphi = self.candidate_dphi[k]

        return best_dphi
    
def refinePhaseSlidingWindow(
    phase_global: np.ndarray,
    trig0: np.ndarray,
    half_cycles,
    window_size_line: int = 5,
    N_phi: int = 50,
) -> np.ndarray:
    """
    Refine global phase estimates using sliding-window phase tracker.

    Parameters
    ----------
    phase_global : np.ndarray
        Global phase estimates, shape (N,).
    trig0 : np.ndarray
        Trig0 signal, shape (N,).
    half_cycles : list of (start, end, direction)
        Half-cycle boundaries.
    window_size_line : int
        Number of half-cycles to include in a sliding window.
    N_phi : int
        Number of candidate phase offsets Δφ to evaluate.

    Returns
    -------
    delta_phi : np.ndarray
        Estimated phase corrections for each half-cycle, shape (num_half_cycles,).
    """

    tracker = ResonantPhaseTracker(
        window_size_lines=window_size_line,
        N_phi=N_phi,
        sub_stride=10,
    )

    num_half_cycles = len(half_cycles)
    delta_phi = np.zeros(num_half_cycles, dtype=np.float64)

    for i in range(num_half_cycles):
        s, e, _ = half_cycles[i]
        phase_line = phase_global[s:e]
        trig0_line = trig0[s:e]

        dphi = tracker.update(phase_line, trig0_line)
        delta_phi[i] = dphi

    return delta_phi


def plotResonantFit(trig0: np.ndarray, omega: float, phi: float, T_SHOW: int = 1000, PLOT_STRIDE: int = 1):
    t = np.arange(T_SHOW, dtype=np.float64)
    y = np.where(trig0 == 0, +1.0, -1.0)[:T_SHOW]
    y_fit = np.sin(omega * t + phi)

    edges = locateResonantTransitions(trig0)

    plt.figure(figsize=(12, 6))
    plt.plot(t[::PLOT_STRIDE], y[::PLOT_STRIDE], label='Trig0 Signal', alpha=0.5)
    plt.plot(t[::PLOT_STRIDE], y_fit[::PLOT_STRIDE], label='Fitted Sine Wave', alpha=0.7)
    for edge in edges:
        if edge < T_SHOW:
            plt.axvline(x=edge, color='red', linestyle='--', alpha=0.3)
    plt.xlabel('Sample Index')
    plt.ylabel('Signal Value')
    plt.title('Resonant Galvo Sine Wave Fit')
    plt.legend()
    plt.grid()
    plt.show()

def plotHalfCycles(
    phase: np.ndarray,
    half_cycles,
    T_SHOW: int = 200_000
):
    t = np.arange(min(T_SHOW, len(phase)))
    phase_show = phase[:T_SHOW]

    plt.figure(figsize=(12, 4))
    plt.plot(t, phase_show, label="phase", linewidth=1.0)

    for (s, e, _) in half_cycles:
        if s < T_SHOW:
            plt.axvline(s, color="red", linestyle="--", alpha=0.3)
        if e < T_SHOW:
            plt.axvline(e, color="red", linestyle="--", alpha=0.3)

    plt.xlabel("sample index")
    plt.ylabel("phase (rad)")
    plt.title("Half-cycle split from global phase")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plotDeltaPhi(
    delta_phi: np.ndarray,
    T_SHOW: int = None,
    PLOT_STRIDE: int = 1
):
    plt.figure(figsize=(10, 4))
    if T_SHOW is None:
        T_SHOW = len(delta_phi)
    t = np.arange(min(T_SHOW, len(delta_phi)))
    plt.plot(t[::PLOT_STRIDE], delta_phi[:T_SHOW][::PLOT_STRIDE], label="Δφ", linewidth=1.0)
    plt.axhline(0, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("half-cycle index")
    plt.ylabel("Δφ (rad)")
    plt.title("Sliding-window phase correction")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

