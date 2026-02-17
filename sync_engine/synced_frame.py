"""SyncedFrame — output structure produced by the sync engine at 60 Hz.

Each frame contains interpolated data from every stream at a single target
timestamp, plus per-stream quality scores so the renderer can flag staleness.

Architecture notes (from Knowledge Base §5.2):
- target_time is in LSL local_clock() domain.
- Per-stream quality allows the renderer to show staleness indicators.
- overall_quality = min(all stream qualities).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class StreamSnapshot:
    """Interpolated data for one stream at a given target time.

    Attributes
    ----------
    data : np.ndarray
        Interpolated channel values at target_time.
    gap : float
        Distance (seconds) from target_time to nearest actual sample.
        0.0 when perfectly bracketed.
    quality : float
        0.0–1.0.  1.0 = perfectly bracketed; 0.0 = gap ≥ 50 ms.
    """

    data: np.ndarray
    gap: float
    quality: float


@dataclass
class SyncedFrame:
    """Container for one synchronisation cycle's output.

    Produced by the sync engine at 60 Hz and consumed by the renderer via
    a double buffer.

    Attributes
    ----------
    target_time : float
        The common LSL timestamp all streams were interpolated to.
    streams : dict[str, StreamSnapshot]
        Per-stream interpolated results keyed by stream name
        (e.g. ``"IMU-Knee"``, ``"EMG-Quad"``, ``"Mocap"``).
    overall_quality : float
        ``min(quality)`` across all streams. Used for quick go/no-go checks.
    """

    target_time: float
    streams: Dict[str, StreamSnapshot] = field(default_factory=dict)
    overall_quality: float = 1.0

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def add(
        self,
        stream_name: str,
        data: Optional[np.ndarray],
        gap: float,
        quality: float,
    ) -> None:
        """Add interpolated data for *stream_name* and update overall quality.

        Parameters
        ----------
        stream_name : str
            Unique identifier for the stream (e.g. ``"IMU-Knee"``).
        data : np.ndarray or None
            Interpolated channel values; ``None`` if the stream had no data.
        gap : float
            Seconds from target_time to nearest actual sample.
        quality : float
            Per-stream quality 0.0–1.0.
        """
        if data is None:
            data = np.array([], dtype=np.float32)
            quality = 0.0

        self.streams[stream_name] = StreamSnapshot(
            data=data, gap=gap, quality=quality
        )
        self._update_overall_quality()

    def _update_overall_quality(self) -> None:
        if self.streams:
            self.overall_quality = min(s.quality for s in self.streams.values())
        else:
            self.overall_quality = 0.0

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def is_stale(self, threshold: float = 0.3) -> bool:
        """Return True if *any* stream quality is below *threshold*."""
        return any(s.quality < threshold for s in self.streams.values())

    def stale_streams(self, threshold: float = 0.3) -> list[str]:
        """Return names of streams whose quality is below *threshold*."""
        return [
            name
            for name, snap in self.streams.items()
            if snap.quality < threshold
        ]

    def __repr__(self) -> str:
        streams_repr = ", ".join(
            f"{k}: q={v.quality:.2f}" for k, v in self.streams.items()
        )
        return (
            f"SyncedFrame(t={self.target_time:.4f}, "
            f"overall_q={self.overall_quality:.2f}, "
            f"streams={{ {streams_repr} }})"
        )
