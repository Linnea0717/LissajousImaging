import time
import numpy as np
from collections import defaultdict

from binning import (
    precompute_z_halfcycle,
    build_z_map_for_xcycle,
    accumulate_xhalfcycle,
    finalize_COO,
    _lut_cos, _lut_sin,
)


class XHalfCycleParser:
    def __init__(self):
        self.last_trig0_val = None
        self.pending_start  = None
        self.pending_dir    = None

    def feed(self, trig0: np.ndarray, sample_offset: int):
        if self.last_trig0_val is not None:
            extended = np.concatenate([[self.last_trig0_val], trig0])
        else:
            extended = trig0

        transitions_local = np.flatnonzero(extended[1:] != extended[:-1])
        transitions_abs   = transitions_local + sample_offset
        self.last_trig0_val = int(trig0[-1])

        if self.pending_start is not None:
            all_trans = np.concatenate([[self.pending_start], transitions_abs])
        else:
            all_trans = transitions_abs

        half_cycles = []
        for i in range(len(all_trans) - 1):
            s = int(all_trans[i])
            e = int(all_trans[i + 1])
            s_local = s - sample_offset
            if 0 <= s_local < len(trig0):
                direction = +1 if trig0[s_local] == 1 else -1
            else:
                direction = self.pending_dir
            half_cycles.append((s, e, direction))

        if len(all_trans) > 0:
            last       = int(all_trans[-1])
            last_local = last - sample_offset
            if 0 <= last_local < len(trig0):
                self.pending_dir = +1 if trig0[last_local] == 1 else -1
            self.pending_start = last

        if len(half_cycles) == 0:
            return np.empty((0, 3), dtype=np.int64)
        return np.array(half_cycles, dtype=np.int64)


class ZHalfCycleParser:
    def __init__(self, threshold: int = 1000):
        self.threshold    = threshold
        self.last_above   = None
        self.pending_edge = None

    def feed(self, tag: np.ndarray, sample_offset: int):
        above = tag.astype(np.int32) > self.threshold

        if self.last_above is not None:
            extended = np.concatenate([[self.last_above], above])
        else:
            extended = above

        rising_local = np.flatnonzero(~extended[:-1] & extended[1:])
        rising_abs   = rising_local + sample_offset
        self.last_above = bool(above[-1])

        if self.pending_edge is not None:
            all_edges = np.concatenate([[self.pending_edge], rising_abs])
        else:
            all_edges = rising_abs

        half_cycles = []
        for i in range(len(all_edges) - 1):
            s = int(all_edges[i])
            e = int(all_edges[i + 1])
            m = (s + e) // 2
            half_cycles.append((s, m, +1))
            half_cycles.append((m, e, -1))

        if len(all_edges) > 0:
            self.pending_edge = int(all_edges[-1])

        if len(half_cycles) == 0:
            return np.empty((0, 3), dtype=np.int64)
        return np.array(half_cycles, dtype=np.int64)


class StreamingVolumeProcessor:
    """
    設計原則
    --------
    - z half-cycle 完成時：立刻 vectorized 算好整條 z_idx/zw，存入 z_cache
    - x half-cycle 完成時：
        1. 從 z_cache 查表，建立這條 x line 的 z_idx_map
        2. Numba kernel 做 scatter-add → 2D (Z × W) accumulator
        3. finalize → 這條 x line 的 COO（(z, y, x, signal)，y 固定）
        4. 收集成完整 volume 後輸出
    """

    def __init__(self, lines_per_volume, z_slices, out_h, out_w,
                 H_scan, W_scan, shifts, drop_before=20, drop_after=20):
        self.lines_per_volume = lines_per_volume
        self.z_slices  = z_slices
        self.out_h     = out_h
        self.out_w     = out_w
        self.H_scan    = H_scan
        self.W_scan    = W_scan
        self.shifts    = shifts
        self.scale_x   = out_w / W_scan
        self._drop_before = drop_before
        self._drop_after = drop_after

        self._z_cache: list = []

        self._x_in_vol = 0
        self._volume_index = 0
        self._vol_z: list = []
        self._vol_y: list = []
        self._vol_x: list = []
        self._vol_s: list = []


    # ── original feed ────────────────────────────
    def feed(self, x_hc, z_hc, pmt_chunk, signal_offset):
        completed, _ = self.feed_timed(x_hc, z_hc, pmt_chunk, signal_offset)
        return completed

    # ── timed feed ───────────────
    def feed_timed(self, x_hc: np.ndarray, z_hc: np.ndarray,
                   pmt_chunk: np.ndarray, signal_offset: int):
        t = defaultdict(float)
        completed = []

        # ── Step 6a: update z_pool ─────────────────────────────────────────
        t0 = time.perf_counter()
        
        for z in z_hc:
            zs, ze, z_dir = int(z[0]), int(z[1]), int(z[2])
            z_idx_map, zw_map = precompute_z_halfcycle(zs, ze, z_dir, self.z_slices)
            self._z_cache.append((zs, ze, z_dir, z_idx_map, zw_map))
        t['a_z_cache_update'] += time.perf_counter() - t0


        # ── Step 6: process x half-cycles ─────────────────────────────────
        for x in x_hc:
            xs, xe, x_dir = int(x[0]), int(x[1]), int(x[2])

            if self._drop_before > 0:
                self._drop_before -= 1
                continue

            abs_yi = self._x_in_vol
            y_norm = (abs_yi + 0.5) / self.H_scan
            y_idx = min(max(int(y_norm * self.out_h), 0), self.out_h - 1)
            shift = (float(self.shifts[abs_yi]) * self.scale_x if abs_yi < len(self.shifts) else 0.0)


            # ── b1: look up from z_cache to build z_idx_map for this x line ────────────
            t0 = time.perf_counter()
            relevant_zc = [zc for zc in self._z_cache if zc[0] < xe and zc[1] > xs]
            z_idx_map, zw_map = build_z_map_for_xcycle(xs, xe, relevant_zc)
            t['b1_z_cache_lookup'] += time.perf_counter() - t0


            # ── b2: accumulate this x half-cycle into the volume ───────────────────────
            t0 = time.perf_counter()
            volume = np.zeros((self.z_slices, self.out_w), dtype=np.float32)
            count  = np.zeros((self.z_slices, self.out_w), dtype=np.float32)
            accumulate_xhalfcycle(
                volume, count,
                pmt_chunk, signal_offset,
                xs, xe, x_dir,
                self.out_w,
                z_idx_map, zw_map,
                _lut_cos, _lut_sin,
            )
            t['b2_accumulate_xhalfcycle'] += time.perf_counter() - t0

            # ── b3: 2D -> COO ───────────────────────
            t0 = time.perf_counter()
            z_c, y_c, x_c, vals = finalize_COO(volume, count, y_idx)
            t['b3_finalize_COO'] += time.perf_counter() - t0

            if len(vals) > 0:
                self._vol_z.append(z_c)
                self._vol_y.append(y_c)
                self._vol_x.append(x_c)
                self._vol_s.append(vals)

            self._x_in_vol += 1


            # ── b4: trim z_cache ───────────────────────
            t0 = time.perf_counter()
            self._z_cache = [zc for zc in self._z_cache if zc[1] > xe]
            t['b4_trim_z_cache'] += time.perf_counter() - t0

            # ── b5: check if volume is completed ───────────────────────
            if self._x_in_vol >= self.lines_per_volume + self._drop_after:
                t0 = time.perf_counter()
                vol = self._assemble_and_reset()
                completed.append(vol)
                t['b5_assemble_and_reset'] += time.perf_counter() - t0

        return completed, dict(t)
    
    def _assemble_and_reset(self):
        """Concatenate all x-line COOs → one volume dict, then reset state."""
        if self._vol_z:
            vol_z = np.concatenate(self._vol_z)
            vol_y = np.concatenate(self._vol_y)
            vol_x = np.concatenate(self._vol_x)
            vol_s = np.concatenate(self._vol_s)
        else:
            vol_z = vol_y = vol_x = np.empty(0, dtype=np.uint16)
            vol_s = np.empty(0, dtype=np.float32)
 
        result = {
            'index':  self._volume_index,
            'z':      vol_z,
            'y':      vol_y,
            'x':      vol_x,
            'signal': vol_s,
            'shape':  np.array(
                [self.z_slices, self.out_h, self.out_w], dtype=np.int32),
        }
 
        print(f"[VOL {self._volume_index}]  "
              f"{len(vol_s)} non-zero voxels  "
              f"({100 * len(vol_s) / (self.z_slices * self.out_h * self.out_w):.1f}% fill)")
 
        self._volume_index += 1
        self._x_in_vol   = 0
        self._vol_z      = []
        self._vol_y      = []
        self._vol_x      = []
        self._vol_s      = []
        self._drop_before = self._drop_after   # skip after each volume
 
        return result