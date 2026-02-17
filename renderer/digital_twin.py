"""Digital twin renderer stub — consumes SyncedFrames for 3D visualisation.

This is a placeholder.  The real renderer will use a 3D engine (e.g. Open3D,
PyVista, or a game engine) to display the digital twin skeleton.
"""

from __future__ import annotations

from typing import Optional

from sync_engine.synced_frame import SyncedFrame


class DigitalTwinRenderer:
    """Consumes SyncedFrames and updates a 3D visualisation.

    This stub logs frame info to console.  Replace with actual rendering.
    """

    def __init__(self, stale_threshold: float = 0.3) -> None:
        self.stale_threshold = stale_threshold
        self._frame_count = 0
        self._last_frame: Optional[SyncedFrame] = None

    def update(self, frame: SyncedFrame) -> None:
        """Process one SyncedFrame.

        Called at ~60 Hz by the main loop.
        """
        self._last_frame = frame
        self._frame_count += 1

        # Flag stale streams
        stale = frame.stale_streams(self.stale_threshold)

        if self._frame_count % 60 == 0:  # Log once per second
            print(
                f"[Renderer] Frame {self._frame_count}: "
                f"t={frame.target_time:.3f}, "
                f"overall_q={frame.overall_quality:.2f}, "
                f"stale={stale or 'none'}"
            )

    @property
    def frame_count(self) -> int:
        return self._frame_count
