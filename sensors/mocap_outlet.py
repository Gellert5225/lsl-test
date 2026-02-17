"""Mocap outlet — OptiTrack / optical mocap SDK → LSL outlet.

Architecture notes (from Knowledge Base §3.2):
- Ethernet transport: ~1 ms, very consistent, minimal jitter.
- Hardware-triggered cameras ensure even frame spacing at 120 Hz.
- Total latency ~12 ms (exposure + triangulation + solving + transport).
- Most stable stream → used as reference for sync-event calibration.
- N markers × 3 coordinates (x, y, z) per frame.
"""

from __future__ import annotations

import time
from typing import Optional, Protocol

import numpy as np

try:
    import pylsl
except ImportError:
    pylsl = None


# ---------------------------------------------------------------------------
# Abstract SDK interface
# ---------------------------------------------------------------------------

class MocapSdk(Protocol):
    """Protocol for a mocap system SDK (e.g. OptiTrack NatNet)."""

    def connect(self) -> bool: ...
    def read_frame(self) -> tuple[np.ndarray, float]:
        """Return (marker_positions, timestamp).
        marker_positions: flat array [x0,y0,z0, x1,y1,z1, ...]
        timestamp: SDK timestamp for this frame.
        """
        ...
    def disconnect(self) -> None: ...
    @property
    def num_markers(self) -> int: ...


# ---------------------------------------------------------------------------
# Mocap → LSL outlet
# ---------------------------------------------------------------------------

class MocapOutlet:
    """Reads frames from a mocap SDK and pushes to LSL.

    Parameters
    ----------
    sdk : MocapSdk
        Sensor SDK implementing the MocapSdk protocol.
    device_id : str
        Unique identifier for the LSL stream.
    sample_rate : float
        Nominal frame rate (default 120 Hz).
    num_markers : int
        Number of markers tracked (determines channel count = markers × 3).
    """

    def __init__(
        self,
        sdk: MocapSdk,
        device_id: str = "optitrack_1",
        sample_rate: float = 120.0,
        num_markers: int = 23,
    ) -> None:
        if pylsl is None:
            raise RuntimeError("pylsl is not installed")

        self.sdk = sdk
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.num_markers = num_markers
        self.num_channels = num_markers * 3

        # LSL stream info
        self._info = pylsl.StreamInfo(
            name="Mocap-Skeleton",
            type="Mocap",
            channel_count=self.num_channels,
            nominal_srate=sample_rate,
            channel_format=pylsl.cf_float32,
            source_id=device_id,
        )
        self._add_channel_metadata()
        self._outlet: Optional[pylsl.StreamOutlet] = None
        self._running = False

    def _add_channel_metadata(self) -> None:
        channels = self._info.desc().append_child("channels")
        axes = ["x", "y", "z"]
        for m in range(self.num_markers):
            for axis in axes:
                ch = channels.append_child("channel")
                ch.append_child_value("label", f"marker{m}_{axis}")
                ch.append_child_value("unit", "meters")

    def start(self) -> None:
        """Connect to SDK and create LSL outlet."""
        self.sdk.connect()
        self._outlet = pylsl.StreamOutlet(self._info)
        self._running = True
        print(f"[MocapOutlet] Started: {self.device_id} "
              f"({self.num_markers} markers @ {self.sample_rate} Hz)")

    def stop(self) -> None:
        """Disconnect SDK and destroy outlet."""
        self._running = False
        self.sdk.disconnect()
        self._outlet = None
        print(f"[MocapOutlet] Stopped: {self.device_id}")

    def run(self) -> None:
        """Main loop — read frames from SDK, push to LSL.

        Call from a dedicated thread or as the main loop.
        """
        if self._outlet is None:
            raise RuntimeError("Call start() before run()")

        interval = 1.0 / self.sample_rate

        while self._running:
            t_start = time.perf_counter()
            try:
                positions, ts = self.sdk.read_frame()
            except Exception:
                time.sleep(0.001)
                continue

            self._outlet.push_sample(positions.tolist(), ts)

            # Maintain frame pacing (SDK may or may not block)
            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
