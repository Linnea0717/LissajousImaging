"""
processor.py
============
FPGA 廠商需要對接的介面。
輸入：一個一個 half-cycle（z 或 x，各自獨立）
輸出：每個 x half-cycle 完成後立刻輸出該 scan line 的 COO

使用方式：
    proc = VolumeProcessor(...)

    # z 和 x 各自獨立輸入，z 要在對應的 x 之前到
    proc.feed_z_halfcycle(zs, ze, z_dir)
    ...
    result = proc.feed_x_halfcycle(xs, xe, x_dir, signal, signal_offset)

    # result 為 COO dict 或 None（drop 區間）
"""

import numpy as np
from mapping import (
    precompute_z_halfcycle,
    build_z_map_for_xcycle,
    precompute_x_halfcycle,
)
from accumulate import accumulate_xcycle_COO


class VolumeProcessor:
    """
    Parameters
    ----------
    z_slices        : z resolution
    out_h, out_w    : y, x resolution of output
    H_scan, W_scan  : y, x resolution of input
    shifts          : shift on each scan line (length = H_scan, in input pixel unit)
    lines_per_volume: number of scan lines in one volume (= H_scan)
    drop_before     : number of scan lines to drop at the beginning
    drop_after      : number of scan lines to drop at the end
    """

    def __init__(
        self,
        z_slices: int,
        out_h: int,
        out_w: int,
        H_scan: int,
        W_scan: int,
        shifts: np.ndarray,
        lines_per_volume: int = 1024,
        drop_before: int = 20,
        drop_after:  int = 20,
    ):
        self.z_slices         = z_slices
        self.out_h            = out_h
        self.out_w            = out_w
        self.H_scan           = H_scan
        self.W_scan           = W_scan
        self.shifts           = shifts
        self.scale_x          = out_w / W_scan
        self.lines_per_volume = lines_per_volume
        self._drop_after      = drop_after

        # z cache：list of (zs, ze, z_dir, z_idx_arr, zw_arr)
        self._z_cache: list = []

        self._drop_countdown = drop_before
        self._x_in_vol       = 0     # number of lines processed in current volume
        self._volume_index   = 0

    # =========================================================
    # z half-cycle input
    # =========================================================

    def feed_z_halfcycle(self, zs: int, ze: int, z_dir: int) -> None:
        """
        input: a z half-cycle
        compute z index and weight for this half-cycle and store in cache for later x half-cycle to query
        """
        z_idx_arr, zw_arr = precompute_z_halfcycle(zs, ze, z_dir, self.z_slices)
        self._z_cache.append((zs, ze, z_dir, z_idx_arr, zw_arr))

    # =========================================================
    # x half-cycle input -> output COO
    # =========================================================

    def feed_x_halfcycle(
        self,
        xs: int,
        xe: int,
        x_dir: int,
        signal: np.ndarray,
        signal_offset: int,
    ) -> dict | None:
        """
        input: an x half-cycle + corresponding PMT signal
        output: COO dict for this x half-cycle, or None in drop region

        signal         : PMT signal array for this halfcycle
        signal_offset  : signal[0] absolute sample index

        Returns
        -------
        dict with keys:
            'volume_index' : int      — volume index this line belongs to
            'y_idx'        : int      — y coordinate of this line
            'z'            : uint16   — z coordinate
            'y'            : uint16   — y coordinate (all are y_idx)
            'x'            : uint16   — x coordinate
            'signal'       : float32  — normalized signal value
            'shape'        : int32[3] — [z_slices, out_h, out_w]
        """
        # ── drop and clean z cache ───────────────────────────────────────
        if self._drop_countdown > 0:
            self._drop_countdown -= 1
            self._prune_z_cache(xe)
            return None

        # ── y_idx ────────────────────────────────────────────────────────
        abs_yi = self._x_in_vol
        y_norm = (abs_yi + 0.5) / self.H_scan
        y_idx  = min(max(int(y_norm * self.out_h), 0), self.out_h - 1)
        shift  = (float(self.shifts[abs_yi]) * self.scale_x
                  if abs_yi < len(self.shifts) else 0.0)

        # ── z mapping for this x half-cycle ──────────────────────────────
        relevant_z = [(zs, ze, d, zi, zw)
                      for (zs, ze, d, zi, zw) in self._z_cache
                      if zs < xe and ze > xs]
        z_idx_map, zw_map = build_z_map_for_xcycle(xs, xe, relevant_z)

        # ── x mapping ────────────────────────────────────────────────────
        x_idx_arr, xw_arr = precompute_x_halfcycle(xs, xe, x_dir, shift, self.out_w)

        # ── COO accumulation ─────────────────────────────────────────────
        z_c, x_c, vals = accumulate_xcycle_COO(
            z_idx_map, zw_map,
            x_idx_arr, xw_arr,
            signal, signal_offset, xs,
            self.out_w,
        )

        # ── fixed y_idx ───────────────────────────────────────────────────
        y_c = np.full(len(z_c), y_idx, dtype=np.uint16)

        # ── if volume finished ────────────────────────────────────────────
        self._x_in_vol += 1
        self._prune_z_cache(xe)

        if self._x_in_vol >= self.lines_per_volume:
            self._volume_index  += 1
            self._x_in_vol       = 0
            self._drop_countdown = self._drop_after

        return {
            'volume_index': self._volume_index,
            'y_idx':        y_idx,
            'z':            z_c,
            'y':            y_c,
            'x':            x_c,
            'signal':       vals,
            'shape':        np.array(
                [self.z_slices, self.out_h, self.out_w], dtype=np.int32),
        }


    def _prune_z_cache(self, xe: int) -> None:
        """remove z cache entry ended before xe"""
        self._z_cache = [(zs, ze, d, zi, zw)
                         for (zs, ze, d, zi, zw) in self._z_cache
                         if ze > xe]
