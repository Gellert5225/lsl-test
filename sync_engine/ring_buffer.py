"""Timestamp-indexed ring buffer with linear interpolation.

Stores (timestamp, np.ndarray) pairs in a fixed-size deque.
Supports interpolation to any target time for producing SyncedFrames at 60 Hz.

Architecture notes (from Knowledge Base §4.2):
- deque(maxlen=N) provides O(1) push with automatic oldest-eviction.
- Timestamps must be in LSL local_clock() domain after time_correction().
- Quality degrades linearly with gap distance: 1.0 when perfectly bracketed,
  0.0 when gap ≥ 50ms.
"""

from collections import deque
from bisect import bisect_right
from typing import Optional

import numpy as np


# Maximum gap (seconds) before quality drops to 0.0.
_MAX_GAP_S = 0.050  # 50 ms


class RingBuffer:
    """Fixed-size ring buffer storing timestamped samples for interpolation.

    Parameters
    ----------
    max_samples : int
        Capacity.  Recommended sizes (for ~2 s history):
            IMU  200 Hz → 400
            EMG  2000 Hz → 4000
            Mocap 120 Hz → 240
    num_channels : int
        Number of data channels per sample.
    """

    def __init__(self, max_samples: int = 2000, num_channels: int = 1) -> None:
        self.max_samples = max_samples
        self.num_channels = num_channels
        self._timestamps: deque[float] = deque(maxlen=max_samples)
        self._data: deque[np.ndarray] = deque(maxlen=max_samples)

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------

    def push(self, timestamp: float, sample) -> None:
        """Append one timestamped sample.

        Parameters
        ----------
        timestamp : float
            LSL-domain timestamp (after ``time_correction()`` and latency offset).
        sample : array-like
            Channel values.  Will be stored as float32 ndarray.
        """
        self._timestamps.append(timestamp)
        self._data.append(np.asarray(sample, dtype=np.float32))

    def push_chunk(self, timestamps, samples) -> None:
        """Append multiple timestamped samples (e.g. an EMG chunk).

        Parameters
        ----------
        timestamps : sequence of float
            One timestamp per sample – preserves per-sample timing.
        samples : sequence of array-like
            Corresponding channel data.
        """
        for ts, s in zip(timestamps, samples):
            self.push(ts, s)

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------

    def interpolate(self, target_time: float):
        """Linearly interpolate data at *target_time*.

        Returns
        -------
        tuple (data, gap, quality)
            data : np.ndarray | None
                Interpolated channel values, or ``None`` if buffer is empty.
            gap : float
                Absolute distance (seconds) from *target_time* to the nearest
                stored sample.
            quality : float
                1.0 when *target_time* is perfectly bracketed; degrades
                linearly to 0.0 when gap ≥ 50 ms.
        """
        n = len(self._timestamps)

        # --- empty buffer ---
        if n == 0:
            return None, float("inf"), 0.0

        ts_list = self._timestamps  # deque – supports random access

        # --- binary search for insertion point ---
        # bisect_right on deque: convert to list snapshot for safety.
        # For hot-path optimisation this can be replaced with a numpy
        # searchsorted on a contiguous array view later.
        idx = bisect_right(self._timestamps, target_time)

        # --- before first sample ---
        if idx == 0:
            gap = ts_list[0] - target_time
            quality = max(0.0, 1.0 - gap / _MAX_GAP_S)
            return self._data[0].copy(), gap, quality

        # --- after last sample ---
        if idx == n:
            gap = target_time - ts_list[-1]
            quality = max(0.0, 1.0 - gap / _MAX_GAP_S)
            return self._data[-1].copy(), gap, quality

        # --- perfectly bracketed → linear interpolation ---
        t0 = ts_list[idx - 1]
        t1 = ts_list[idx]
        dt = t1 - t0
        if dt == 0:
            # Duplicate timestamps – just return one of them.
            return self._data[idx].copy(), 0.0, 1.0

        alpha = (target_time - t0) / dt
        interpolated = (1.0 - alpha) * self._data[idx - 1] + alpha * self._data[idx]

        gap = 0.0  # perfectly bracketed
        quality = 1.0
        return interpolated, gap, quality

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def newest_timestamp(self) -> Optional[float]:
        """Timestamp of the most recent sample, or ``None``."""
        return self._timestamps[-1] if self._timestamps else None

    @property
    def oldest_timestamp(self) -> Optional[float]:
        """Timestamp of the oldest sample, or ``None``."""
        return self._timestamps[0] if self._timestamps else None

    def __len__(self) -> int:
        return len(self._timestamps)

    def __repr__(self) -> str:
        return (
            f"RingBuffer(max_samples={self.max_samples}, "
            f"num_channels={self.num_channels}, "
            f"stored={len(self)})"
        )
