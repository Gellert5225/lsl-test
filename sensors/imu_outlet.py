"""IMU outlet — Xsens DOT SDK → LSL outlet.

Architecture notes (from Knowledge Base §3.2, §7.1):
- BLE transport creates bursts of 2–3 samples every ~15 ms.
- NEVER stamp burst samples with local_clock() at arrival — they'd all get
  the same timestamp.
- Use SDK hardware timestamps OR reconstruct from (sample_counter × 1/rate).
- 7 channels: qw, qx, qy, qz, ax, ay, az.
- Nominal rate 200 Hz.
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
# Abstract SDK interface (swap for real Xsens SDK or simulator)
# ---------------------------------------------------------------------------

class ImuSdk(Protocol):
    """Protocol for an IMU sensor SDK."""

    def connect(self) -> bool: ...
    def read_sample(self) -> tuple[np.ndarray, float, int]:
        """Return (sample, hardware_timestamp, sequence_number)."""
        ...
    def disconnect(self) -> None: ...


# ---------------------------------------------------------------------------
# IMU → LSL outlet
# ---------------------------------------------------------------------------

class ImuOutlet:
    """Reads from an IMU SDK and pushes samples to an LSL outlet with
    correct hardware timestamps.

    Parameters
    ----------
    sdk : ImuSdk
        Sensor SDK implementing the ImuSdk protocol.
    device_id : str
        Unique identifier for the LSL stream (e.g. serial number).
    sample_rate : float
        Nominal rate in Hz (default 200).
    use_hardware_timestamps : bool
        If True (default), use timestamps from the SDK.
        If False, reconstruct from sequence counter.
    """

    CHANNEL_LABELS = ["qw", "qx", "qy", "qz", "ax", "ay", "az"]
    NUM_CHANNELS = 7

    def __init__(
        self,
        sdk: ImuSdk,
        device_id: str = "imu_unit_1",
        sample_rate: float = 200.0,
        use_hardware_timestamps: bool = True,
    ) -> None:
        if pylsl is None:
            raise RuntimeError("pylsl is not installed")

        self.sdk = sdk
        self.device_id = device_id
        self.sample_rate = sample_rate
        self.use_hardware_timestamps = use_hardware_timestamps

        # LSL stream info
        self._info = pylsl.StreamInfo(
            name="IMU-Orientation",
            type="IMU",
            channel_count=self.NUM_CHANNELS,
            nominal_srate=sample_rate,
            channel_format=pylsl.cf_float32,
            source_id=device_id,
        )
        self._add_channel_metadata()
        self._outlet: Optional[pylsl.StreamOutlet] = None

        # For timestamp reconstruction from sequence counter
        self._first_sample_time: Optional[float] = None
        self._running = False

    def _add_channel_metadata(self) -> None:
        channels = self._info.desc().append_child("channels")
        for label in self.CHANNEL_LABELS:
            ch = channels.append_child("channel")
            ch.append_child_value("label", label)
            unit = "quaternion" if label.startswith("q") else "m/s^2"
            ch.append_child_value("unit", unit)

    def start(self) -> None:
        """Connect to SDK and create LSL outlet."""
        self.sdk.connect()
        self._outlet = pylsl.StreamOutlet(self._info)
        self._running = True
        print(f"[ImuOutlet] Started: {self.device_id}")

    def stop(self) -> None:
        """Disconnect SDK and destroy outlet."""
        self._running = False
        self.sdk.disconnect()
        self._outlet = None
        print(f"[ImuOutlet] Stopped: {self.device_id}")

    def run(self) -> None:
        """Main loop — read from SDK, push to LSL.

        Call from a dedicated thread or as the main loop.
        """
        if self._outlet is None:
            raise RuntimeError("Call start() before run()")

        while self._running:
            try:
                sample, hw_ts, seq = self.sdk.read_sample()
            except Exception:
                time.sleep(0.001)
                continue

            ts = self._resolve_timestamp(hw_ts, seq)
            self._outlet.push_sample(sample.tolist(), ts)

    def _resolve_timestamp(self, hw_ts: float, seq: int) -> float:
        """Choose the best timestamp for this sample.
        """
        if self.use_hardware_timestamps and hw_ts > 0:
            return hw_ts

        # Reconstruct from sample counter
        if self._first_sample_time is None:
            self._first_sample_time = pylsl.local_clock()
        return self._first_sample_time + seq * (1.0 / self.sample_rate)
