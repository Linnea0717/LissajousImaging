"""
processor.py
============
FPGA 廠商需要對接的介面。

Interface
---------
    proc = StreamingVolumeProcessor(...)

    # On z trigger edge:
    proc.notify_z_hc_end(ze, z_dir_next)

    # On each sample (or batch of samples):
    proc.feed_samples(abs_start, pmt_chunk)

    # On x trigger edge — returns list of COO dicts:
    results = proc.notify_x_hc_end(xe, x_dir_next)

    # After all data consumed, emit the rest data:
    results = proc.flush()

Accumulation and emission
--------------------------
As it is guaranteed that the coordinates of any two samples are identical if they are not in the same x half-cycle,
    we can safely emit COO at the end of each x half-cycle.
_acc is the accumulation buffer storing samples during each x halfcycle.
When notify_x_hc_end fires, all of _acc is emitted andcleared.
    _acc therefore always holds exactly the current halfcycle's voxels.


Output dict
-----------------------------------
    'volume_index' : int
    'y_idx'        : int         — y index, derived from the current x halfcycle number and out_h
    'z'            : uint16[K]
    'y'            : uint16[K]   — same as y_idx; same for all samples in the same x halfcycle
    'x'            : uint16[K]
    'signal'       : float32[K]  — weighted-average PMT
    'shape'        : int32[3]    — [z_slices, out_h, out_w]
"""

import numpy as np
from estimator import HalfCycleLengthEstimator
from lut import lut_lookup


class StreamingVolumeProcessor:
    def __init__(
        self,
        z_slices:         int,
        out_h:            int,
        out_w:            int,
        H_scan:           int,
        W_scan:           int,
        shifts:           np.ndarray,
        nominal_x_len:    float,
        nominal_z_period: float,
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

        # ── length estimators ─────────────────────────────────────────────
        # x: tracks halfcycle length (trigger transition to transition)
        # z: tracks full period length (rising edge to rising edge),
        self._x_est = HalfCycleLengthEstimator(nominal_x_len,    estimator_window)
        self._z_est = HalfCycleLengthEstimator(nominal_z_period, estimator_window)

        # ── current halfcycle state ───────────────────────────────────────
        self._x_hc_start    = 0   # x halfcycle start sample index
        self._z_hc_start    = 0   # z halfcycle start sample index
        self._z_last_rising = 0   # last z rising edge
        self._x_dir         = +1
        self._z_dir         = +1

        # ── accumulation buffer ───────────────────────────────────────────
        # key  : (z_idx: int, x_idx: int)
        # value: [wval_sum: float, w_sum: float]
        self._acc: dict[tuple[int, int], list] = {}

        # ── volume / scan-line tracking ───────────────────────────────────
        self._x_in_vol       = 0
        self._volume_index   = 0
        self._drop_after     = drop_after
        self._drop_countdown = drop_before

    def feed_sample(self, abs_idx: int, pmt_val: int) -> None:
        """
        Process a single PMT sample.
        """
        self.feed_samples(abs_idx, np.array([pmt_val], dtype=np.int16))

    def feed_samples(self, abs_start: int, pmt_chunk: np.ndarray) -> None:
        """
        Accumulate a batch of consecutive PMT samples into _acc.
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
        phase_z = np.clip(i_z / (L_z / 2), 0.0, 1.0)
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
    # Trigger events
    # =========================================================

    def notify_x_hc_end(self, xe: int, x_dir_next: int) -> list[dict]:
        """
        Updates x halfcycle states and estimates and emits all voxels in _acc.
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
    def z_period_est(self) -> float:
        """Current sliding-window estimate of the full z period"""
        return self._z_est.estimate


    def notify_z_hc_end(self, ze: int, z_dir_next: int) -> None:
        """
        Updates z halfcycle state and updates z period estimates on rising edges.
        """
        if z_dir_next == +1:
            period = ze - self._z_last_rising
            if period > 0:
                self._z_est.update(period)
            self._z_last_rising = ze
        self._z_hc_start = ze
        self._z_dir      = z_dir_next


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