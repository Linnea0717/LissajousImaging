import time
import numpy as np
from collections import defaultdict

from binning import (
    make_accumulators,
    feed_chunk,
    finalize_COO,
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
    def __init__(self, lines_per_volume, z_slices, out_h, out_w,
                 H_scan, W_scan, shifts):
        self.lines_per_volume = lines_per_volume
        self.z_slices  = z_slices
        self.out_h     = out_h
        self.out_w     = out_w
        self.H_scan    = H_scan
        self.W_scan    = W_scan
        self.shifts    = shifts

        self.volume, self.count = make_accumulators(z_slices, out_h, out_w)
        self.x_lines_accumulated = 0
        self.volume_index = 0
        self._z_pool = []

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
        if len(z_hc) > 0:
            self._z_pool.extend(z_hc.tolist())
        t['a_z_pool_update'] += time.perf_counter() - t0

        if len(x_hc) == 0:
            return completed, dict(t)

        i = 0
        while i < len(x_hc):
            space_left = self.lines_per_volume - self.x_lines_accumulated
            batch_x    = x_hc[i : i + space_left]

            # ── Step 6b: z_pool filter ──
            t0 = time.perf_counter()
            t_start = int(batch_x[0,  0])
            t_end   = int(batch_x[-1, 1])
            batch_z = np.array(
                [zc for zc in self._z_pool if zc[1] >= t_start and zc[0] <= t_end],
                dtype=np.int64
            )
            t['b_z_pool_filter'] += time.perf_counter() - t0

            # ── Step 6c: accumulateVolume ────────────────────
            t0 = time.perf_counter()
            if len(batch_z) > 0:
                feed_chunk(
                    self.volume, self.count,
                    pmt_chunk, signal_offset,
                    batch_x, batch_z, self.shifts,
                    self.out_h, self.out_w, self.z_slices,
                    self.H_scan, self.W_scan,
                    yi_offset=self.x_lines_accumulated,
                )
            t['c_accumulate'] += time.perf_counter() - t0

            self.x_lines_accumulated += len(batch_x)
            i += len(batch_x)

            # ── fill volume → finalize + reset ───────────────────────────
            if self.x_lines_accumulated >= self.lines_per_volume:

                # ── Step 6d: finalize COO ────────────────────────────────
                t0 = time.perf_counter()
                z_c, y_c, x_c, vals = finalize_COO(self.volume, self.count)
                t['d_finalize_COO'] += time.perf_counter() - t0

                completed.append({
                    'index':  self.volume_index,
                    'z':      z_c,
                    'y':      y_c,
                    'x':      x_c,
                    'signal': vals,
                    'shape':  np.array([self.z_slices, self.out_h, self.out_w],
                                       dtype=np.int32),
                })
                print(f"[VOL {self.volume_index}] "
                      f"{len(vals)} non-zero voxels "
                      f"({100*len(vals)/(self.z_slices*self.out_h*self.out_w):.1f}% fill)")

                # ── Step 6e: reset accumulators ──────────────────────────
                t0 = time.perf_counter()
                self.volume_index += 1
                self.volume, self.count = make_accumulators(
                    self.z_slices, self.out_h, self.out_w)
                self.x_lines_accumulated = 0
                self._z_pool = [zc for zc in self._z_pool if zc[1] > t_end]
                t['e_reset'] += time.perf_counter() - t0

        return completed, dict(t)