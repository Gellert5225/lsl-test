"""EMG outlet — Delsys Trigno SDK → LSL outlet.

Architecture notes (from Knowledge Base §3.2, §7.2):
- WiFi + chunk buffer: 48 samples at 2 kHz accumulate (~24 ms) before send.
- Delsys SDK provides PER-SAMPLE timestamps within the chunk.
- Must use push_chunk(data, timestamps) to preserve per-sample timing.
  Never push_chunk(data) alone — that stamps the entire chunk at arrival.
- 8 channels: bilateral quads (e.g. VastusLat_L, VastusMed_L, ...).
- EMG envelope processing happens AFTER sync alignment, not here.
"""

from __future__ import annotations

import time
from typing import List, Optional, Protocol, Tuple

import numpy as np

try:
    import pylsl
except ImportError:
    pylsl = None


# ---------------------------------------------------------------------------
# Abstract SDK interface
# ---------------------------------------------------------------------------

class EmgSdk(Protocol):
    """Protocol for an EMG sensor SDK (e.g. Delsys Trigno)."""

    def connect(self) -> bool: ...
    def read_chunk(self) -> Tuple[List[np.ndarray], List[float]]:
        """Return (samples, timestamps) — per-sample timestamps from SDK."""
        ...
    def disconnect(self) -> None: ...


# ---------------------------------------------------------------------------
# EMG → LSL outlet
# ---------------------------------------------------------------------------

class EmgOutlet:
    """Reads EMG chunks from SDK and pushes to LSL with per-sample timestamps.

    Parameters
    ----------
    sdk : EmgSdk
        Sensor SDK implementing the EmgSdk protocol.
    device_id : str
        Unique identifier for the LSL stream.
    channel_labels : list[str] or None
        Custom channel names.  Default: bilateral quad muscles.
    sample_rate : float
        Nominal sample rate (default 2000 Hz).
    """

    DEFAULT_CHANNEL_LABELS = [
        "VastusLat_L",
        "VastusMed_L",
        "RectusFem_L",
        "BicepsFem_L",
        "VastusLat_R",
        "VastusMed_R",
        "RectusFem_R",
        "BicepsFem_R",
    ]

    def __init__(
        self,
        sdk: EmgSdk,
        device_id: str = "delsys_unit_7",
        channel_labels: Optional[List[str]] = None,
        sample_rate: float = 2000.0,
    ) -> None:
        if pylsl is None:
            raise RuntimeError("pylsl is not installed")

        self.sdk = sdk
        self.device_id = device_id
        self.channel_labels = channel_labels or self.DEFAULT_CHANNEL_LABELS
        self.num_channels = len(self.channel_labels)
        self.sample_rate = sample_rate

        # LSL stream info
        self._info = pylsl.StreamInfo(
            name="EMG-Quad",
            type="EMG",
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
        for label in self.channel_labels:
            ch = channels.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "microvolts")

    def start(self) -> None:
        """Connect to SDK and create LSL outlet."""
        self.sdk.connect()
        self._outlet = pylsl.StreamOutlet(self._info, chunk_size=48)
        self._running = True
        print(f"[EmgOutlet] Started: {self.device_id} "
              f"({self.num_channels}ch @ {self.sample_rate} Hz)")

    def stop(self) -> None:
        """Disconnect SDK and destroy outlet."""
        self._running = False
        self.sdk.disconnect()
        self._outlet = None
        print(f"[EmgOutlet] Stopped: {self.device_id}")

    def run(self) -> None:
        """Main loop — read chunks from SDK, push to LSL with per-sample timestamps.

        Call from a dedicated thread or as the main loop.
        """
        if self._outlet is None:
            raise RuntimeError("Call start() before run()")

        while self._running:
            try:
                samples, timestamps = self.sdk.read_chunk()
            except Exception:
                time.sleep(0.001)
                continue

            if not samples:
                time.sleep(0.001)
                continue

            # push_chunk with per-sample timestamps — preserves EMG timing
            chunk_data = [s.tolist() for s in samples]
            self._outlet.push_chunk(chunk_data, timestamps)
