import numpy as np

from binning import (
    make_accumulators,
    feed_chunk,
    finalize_COO
)

class XHalfCycleParser:
    """
    input: a segment of trig0 data
    output: x half-cyclers in this segment, storing fragmented half-cycles for later input
    """
 
    def __init__(self):
        self.last_trig0_val = None
        self.pending_start  = None
        self.pending_dir    = None
 
    def feed(self, trig0: np.ndarray, sample_offset: int):
        """
        sample_offset: the absolute index of the first sample in this chunk

        return full half-cycles, in shape (N, 3), columns = [start, end, dir]
        """

        # concatenate with last value to find transitions, if exists
        if self.last_trig0_val is not None:
            extended = np.concatenate([[self.last_trig0_val], trig0])
        else:
            extended = trig0
 
        # find all transitions in this chunk (local index)
        transitions_local = np.flatnonzero(extended[1:] != extended[:-1])
        transitions_abs   = transitions_local + sample_offset

        self.last_trig0_val = int(trig0[-1])

 
        # ---- add pending transition from last chunk if exists ----
        if self.pending_start is not None:
            all_trans = np.concatenate([[self.pending_start], transitions_abs])
        else:
            all_trans = transitions_abs
 
        # ---- match transitions in pairs ----
        half_cycles = []
        for i in range(len(all_trans) - 1):
            s = int(all_trans[i])
            e = int(all_trans[i + 1])
            s_local = s - sample_offset
            if 0 <= s_local < len(trig0):
                # start position within this chunk → determine direction by trig0 value at start
                direction = +1 if trig0[s_local] == 1 else -1
            else:
                # start position outside this chunk → use pending direction from last chunk
                direction = self.pending_dir
            half_cycles.append((s, e, direction))

 
        if len(all_trans) > 0:
            last = int(all_trans[-1])
            last_local = last - sample_offset
            if 0 <= last_local < len(trig0):
                self.pending_dir = +1 if trig0[last_local] == 1 else -1
            self.pending_start = last

        if len(half_cycles) == 0:
            return np.empty((0, 3), dtype=np.int64)
        return np.array(half_cycles, dtype=np.int64)
 
 
class ZHalfCycleParser:
    """
    input: a segment of trig1 data
    output: z half-cyclers in this segment, storing fragmented half-cycles for later input
    """
 
    def __init__(self, threshold: int = 1000):
        self.threshold       = threshold
        self.last_above      = None
        self.pending_edge    = None 
 
    def feed(self, tag: np.ndarray, sample_offset: int):
        """
        sample_offset: the absolute index of the first sample in this chunk 
        
        return full half-cycles, in shape (N, 3), columns = [start, end, dir]
        """

        above = tag.astype(np.int32) > self.threshold
 
        # ---- extend with last value (above or not) ----
        if self.last_above is not None:
            extended = np.concatenate([[self.last_above], above])
        else:
            extended = above

        rising_local  = np.flatnonzero(~extended[:-1] & extended[1:])
        rising_abs    = rising_local + sample_offset

        self.last_above = bool(above[-1])
 
        if self.pending_edge is not None:
            all_edges = np.concatenate([[self.pending_edge], rising_abs])
        else:
            all_edges = rising_abs
 
        # ---- match edges in pairs ----
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
    streaming processing of x and z half-cycles to produce volumes in COO format
    """
 
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
        self.x_lines_accumulated = 0   # number of x lines accumulated in current volume
        self.volume_index = 0
 
        # z half-cycle pool: store all z half-cycles that may overlap with future x half-cycles
        self._z_pool = []
 
    def feed(self, x_hc: np.ndarray, z_hc: np.ndarray, PMT: np.ndarray):
        """
        x_hc : shape (Nx, 3)  full x half-cycle, absolute index, columns = [start, end, dir]
        z_hc : shape (Nz, 3)  full z half-cycle, absolute index, columns = [start, end, dir]
        PMT  : shape (total_samples,)  signal array for the whole scan
        
        return a list of completed volumes in COO format, each is a dict with keys
              {'index', 'z', 'y', 'x', 'signal', 'shape'}
        """
        completed = []

        # add new z half-cycles to the pool
        if len(z_hc) > 0:
            self._z_pool.extend(z_hc.tolist())
 
        if len(x_hc) == 0:
            return completed
 
        # process x half-cycles line by line; finalise when a volume is filled
        i = 0
        while i < len(x_hc):
            space_left = self.lines_per_volume - self.x_lines_accumulated
            batch_x = x_hc[i : i + space_left]
 
            # find overlapping z half-cycles for this batch of x half-cycles
            t_start = int(batch_x[0,  0])
            t_end   = int(batch_x[-1, 1])
            batch_z = np.array(
                [zc for zc in self._z_pool if zc[1] >= t_start and zc[0] <= t_end],
                dtype=np.int64
            )
 
            # accumulate volume for this batch
            if len(batch_z) > 0:
                feed_chunk(
                    self.volume, self.count, PMT,
                    batch_x, batch_z, self.shifts,
                    self.out_h, self.out_w, self.z_slices,
                    self.H_scan, self.W_scan,
                    yi_offset=self.x_lines_accumulated
                )
 
            self.x_lines_accumulated += len(batch_x)
            i += len(batch_x)
 
            # volume is filled, finalize COO and reset for next volume
            if self.x_lines_accumulated >= self.lines_per_volume:
                z_c, y_c, x_c, vals = finalize_COO(self.volume, self.count)
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
 
                self.volume_index += 1
                self.volume, self.count = make_accumulators(
                    self.z_slices, self.out_h, self.out_w)
                self.x_lines_accumulated = 0
 
                # remove z half-cycles that have ended before t_end
                self._z_pool = [zc for zc in self._z_pool if zc[1] > t_end]
 
        return completed