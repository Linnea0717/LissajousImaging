"""
parser.py
=========
模擬：把一段（chunk） raw data 中的 trig/TAG 訊號切成 half-cycle。

這個檔案是模擬的一部分，FPGA 廠商不需要參考。
"""

import numpy as np


class XHalfCycleParser:
    """
    tirg0 -> x half-cycle
    maintain state across chunks to handle half-cycles that span multiple chunks
    """

    def __init__(self):
        self.last_trig0_val = None
        self.pending_start  = None
        self.pending_dir    = None

    def feed(self, trig0: np.ndarray, sample_offset: int) -> np.ndarray:
        """
        Parameters
        ----------
        trig0         : uint8 chunk
        sample_offset : trig0[0] absolute sample index

        Returns
        -------
        half_cycles : int64 (N, 3)  columns = [start, end, dir]
        """
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
            s       = int(all_trans[i])
            e       = int(all_trans[i + 1])
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
    """
    TAG -> z half-cycle
    maintain state across chunks to handle half-cycles that span multiple chunks
    """

    def __init__(self, threshold: int = 1000):
        self.threshold    = threshold
        self.last_above   = None
        self.pending_edge = None

    def feed(self, tag: np.ndarray, sample_offset: int) -> np.ndarray:
        """
        Parameters
        ----------
        tag           : int16 chunk（channel-1 signal）
        sample_offset : tag[0] absolute sample index

        Returns
        -------
        half_cycles : int64 (N, 3)  columns = [start, end, dir]
                      dir = +1（first half）or -1（second half）
        """
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
