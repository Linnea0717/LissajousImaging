"""
processor.py
============
FPGA 廠商需要對接的介面。

Interface
---------
    proc = StreamingVolumeProcessor(...)

    # On z trigger edge (call BEFORE x if both fire at same sample):
    proc.notify_z_hc_end(ze, z_dir_next)

    # On each sample batch (all within the same x and z halfcycle):
    proc.feed_samples(abs_start, pmt_chunk)

    # On x trigger edge — returns list of COO dicts (0 or 1 element):
    results = proc.notify_x_hc_end(xe, x_dir_next)

    # After all data consumed, emit the last partial halfcycle:
    results = proc.flush()

Accumulation and emission
--------------------------
_acc accumulates samples during each x halfcycle.
When notify_x_hc_end fires, ALL of _acc is emitted as one scan line and
cleared. _acc therefore always holds exactly the current halfcycle's voxels.

_acc entry : [wval_sum: float, w_sum: float]
y_idx       : self._x_in_vol (one scan line per halfcycle, no resampling)

Output dict (unchanged from ver8)
-----------------------------------
    'volume_index' : int
    'y_idx'        : int         — x_in_vol of this halfcycle
    'z'            : uint16[K]
    'y'            : uint16[K]   — all equal to y_idx
    'x'            : uint16[K]
    'signal'       : float32[K]  — weighted-average PMT
    'shape'        : int32[3]    — [z_slices, out_h, out_w]
"""

import numpy as np
from estimator import HalfCycleLengthEstimator
from lut import lut_lookup


class StreamingVolumeProcessor:
    """
    Parameters
    ----------
    z_slices          : z resolution (output depth)
    out_h, out_w      : y, x output resolution (should equal H_scan, W_scan)
    H_scan, W_scan    : y, x input scan resolution
    shifts            : per-scan-line x shift (input pixels), length >= H_scan
    nominal_x_len     : default x halfcycle length (samples), pre-fills estimator
    nominal_z_len     : default z halfcycle length (samples), pre-fills estimator
    estimator_window  : sliding window size for both estimators
    lines_per_volume  : x halfcycles per volume (typically = H_scan)
    drop_before       : x halfcycles to discard at dataset start
    drop_after        : x halfcycles to discard at each volume boundary
    """

    def __init__(
        self,
        z_slices:         int,
        out_h:            int,
        out_w:            int,
        H_scan:           int,
        W_scan:           int,
        shifts:           np.ndarray,
        nominal_x_len:    float,
        nominal_z_len:    float,
        estimator_window: int = 6,
        lines_per_volume: int = 1024,
        drop_before:      int = 20,
        drop_after:       int = 20,
    ):
        # ── geometry ─────────────────────────────────────────────────────
        self.z_slices         = z_slices
        self.out_h            = out_h
        self.out_w            = out_w
        self.H_scan           = H_scan
        self.scale_x          = out_w / W_scan
        self.shifts           = shifts
        self.lines_per_volume = lines_per_volume

        # ── halfcycle length estimators ───────────────────────────────────
        self._x_est = HalfCycleLengthEstimator(nominal_x_len, estimator_window)
        self._z_est = HalfCycleLengthEstimator(nominal_z_len, estimator_window)

        # ── current halfcycle state ───────────────────────────────────────
        # Both initialized to 0: the first halfcycle is assumed to start
        # at sample 0, so feed_samples works correctly from the very first sample.
        self._x_hc_start = 0
        self._z_hc_start = 0
        self._x_dir      = +1
        self._z_dir      = +1

        # ── accumulation buffer ───────────────────────────────────────────
        # key  : (z_idx: int, x_idx: int)
        # value: [wval_sum: float, w_sum: float]
        # Cleared at every halfcycle boundary — always holds exactly the
        # current halfcycle's voxels.
        self._acc: dict[tuple[int, int], list] = {}

        # ── volume / scan-line tracking ───────────────────────────────────
        self._x_in_vol       = 0
        self._volume_index   = 0
        self._drop_after     = drop_after
        self._drop_countdown = drop_before

    # =========================================================
    # Sample-level input
    # =========================================================

    def feed_sample(self, abs_idx: int, pmt_val: int) -> None:
        """
        Process a single PMT sample.
        Primary interface for FPGA / sample-by-sample use.

        Parameters
        ----------
        abs_idx : absolute sample index
        pmt_val : raw PMT value (int16)
        """
        self.feed_samples(abs_idx, np.array([pmt_val], dtype=np.int16))

    def feed_samples(self, abs_start: int, pmt_chunk: np.ndarray) -> None:
        """
        Accumulate a batch of consecutive PMT samples into _acc.
        All samples must lie within the same x halfcycle and z halfcycle
        (call between trigger events, not across them).

        Parameters
        ----------
        abs_start : absolute sample index of pmt_chunk[0]
        pmt_chunk : int16 array of PMT values
        """
        if self._drop_countdown > 0:
            return
        n = len(pmt_chunk)
        if n == 0:
            return

        L_x = self._x_est.estimate
        L_z = self._z_est.estimate

        # ── x coordinate mapping ──────────────────────────────────────
        i_x     = np.arange(abs_start - self._x_hc_start,
                             abs_start - self._x_hc_start + n,
                             dtype=np.float64)
        phase_x = np.clip(i_x / L_x, 0.0, 1.0)
        x_cos, xw = lut_lookup(phase_x, self._x_dir)

        shift  = (float(self.shifts[self._x_in_vol]) * self.scale_x
                  if self._x_in_vol < len(self.shifts) else 0.0)
        x_pix  = (x_cos.astype(np.float64) + 1.0) * 0.5 * self.out_w
        x_pix -= shift if self._x_dir == +1 else -shift
        x_idx  = np.clip(np.round(x_pix).astype(np.int32), 0, self.out_w - 1)

        # ── z coordinate mapping ──────────────────────────────────────
        i_z     = np.arange(abs_start - self._z_hc_start,
                             abs_start - self._z_hc_start + n,
                             dtype=np.float64)
        phase_z = np.clip(i_z / L_z, 0.0, 1.0)
        z_cos, zw = lut_lookup(phase_z, self._z_dir)
        z_norm  = (z_cos.astype(np.float64) + 1.0) * 0.5
        z_idx   = np.clip(
            np.round(z_norm * self.z_slices).astype(np.int32),
            0, self.z_slices - 1,
        ).astype(np.int16)

        # ── signal (PMT is negative-going) ────────────────────────────
        sig  = np.maximum(0.0, -pmt_chunk.astype(np.float32))
        w    = xw * zw
        wval = sig * w

        # ── scatter-add into _acc ─────────────────────────────────────
        # Group samples mapping to the same (z, x) pixel and sum their
        # weighted values. The Python loop below is over unique pixels
        # (bounded by z_slices × out_w), not over raw samples.
        lin   = z_idx.astype(np.int32) * self.out_w + x_idx
        order = np.argsort(lin, kind='stable')
        lin_s, wval_s, w_s = lin[order], wval[order], w[order]

        uniq, first_u = np.unique(lin_s, return_index=True)
        val_sums = np.add.reduceat(wval_s, first_u)
        w_sums   = np.add.reduceat(w_s,   first_u)

        for lk, vs, ws in zip(uniq.tolist(), val_sums.tolist(), w_sums.tolist()):
            key = (int(lk) // self.out_w, int(lk) % self.out_w)
            if key in self._acc:
                self._acc[key][0] += vs
                self._acc[key][1] += ws
            else:
                self._acc[key] = [vs, ws]

    # =========================================================
    # Trigger event: x halfcycle ended
    # =========================================================

    def notify_x_hc_end(self, xe: int, x_dir_next: int) -> list[dict]:
        """
        Call when the x trigger fires at absolute sample xe.
        Emits ALL voxels in _acc as one scan line, clears _acc,
        updates estimator, advances counters.

        Parameters
        ----------
        xe          : absolute sample index of the trigger edge
        x_dir_next  : direction (+1/-1) of the halfcycle starting at xe

        Returns
        -------
        List of COO dicts — empty in drop zone, one element otherwise.
        """
        # ── update estimator ─────────────────────────────────────────
        true_len = xe - self._x_hc_start
        if true_len > 0:
            self._x_est.update(true_len)

        # ── advance halfcycle state ───────────────────────────────────
        self._x_hc_start = xe
        self._x_dir      = x_dir_next

        # ── drop zone ────────────────────────────────────────────────
        if self._drop_countdown > 0:
            self._drop_countdown -= 1
            self._acc.clear()
            self._x_in_vol += 1
            if self._x_in_vol >= self.lines_per_volume:
                self._volume_index  += 1
                self._x_in_vol       = 0
                self._drop_countdown = self._drop_after
            return []

        # ── emit all accumulated voxels for this halfcycle ────────────
        result = self._emit_acc(self._x_in_vol)

        # ── advance scan-line / volume ────────────────────────────────
        self._x_in_vol += 1
        if self._x_in_vol >= self.lines_per_volume:
            self._volume_index  += 1
            self._x_in_vol       = 0
            self._drop_countdown = self._drop_after

        return [result] if result is not None else []

    @property
    def z_hc_length_est(self) -> float:
        """Current sliding-window estimate of z halfcycle length (samples)."""
        return self._z_est.estimate

    # =========================================================
    # Trigger event: z halfcycle ended
    # =========================================================

    def notify_z_hc_end(self, ze: int, z_dir_next: int) -> None:
        """
        Call when the z trigger fires at absolute sample ze.
        Updates z length estimator and advances z halfcycle state.

        Parameters
        ----------
        ze          : absolute sample index of the z trigger edge
        z_dir_next  : direction (+1/-1) of the z halfcycle starting at ze
        """
        true_len = ze - self._z_hc_start
        if true_len > 0:
            self._z_est.update(true_len)
        self._z_hc_start = ze
        self._z_dir      = z_dir_next

    # =========================================================
    # End-of-data flush
    # =========================================================

    def flush(self) -> list[dict]:
        """
        Emit any voxels remaining in _acc after the last trigger.
        Call once after all data has been fed.
        """
        if not self._acc or self._drop_countdown > 0:
            self._acc.clear()
            return []
        result = self._emit_acc(self._x_in_vol)
        return [result] if result is not None else []

    # =========================================================
    # Internal
    # =========================================================

    def _emit_acc(self, y_scan: int) -> dict | None:
        """Bundle _acc into a COO dict, clear _acc. Returns None if empty."""
        if not self._acc:
            return None

        keys    = list(self._acc.keys())
        z_arr   = np.array([k[0] for k in keys], dtype=np.uint16)
        x_arr   = np.array([k[1] for k in keys], dtype=np.uint16)
        sig_arr = np.array(
            [self._acc[k][0] / self._acc[k][1]
             if self._acc[k][1] > 1e-9 else 0.0
             for k in keys],
            dtype=np.float32,
        )
        self._acc.clear()

        y_idx = min(max(int((y_scan + 0.5) / self.H_scan * self.out_h), 0), self.out_h - 1)
        y_arr = np.full(len(z_arr), y_idx, dtype=np.uint16)

        return {
            'volume_index': self._volume_index,
            'y_idx':        y_idx,
            'z':            z_arr,
            'y':            y_arr,
            'x':            x_arr,
            'signal':       sig_arr,
            'shape':        np.array(
                [self.z_slices, self.out_h, self.out_w], dtype=np.int32),
        }