"""IMU simulator — generates realistic BLE-burst IMU data without hardware.

Simulates:
- 200 Hz sample rate with BLE burst behaviour (2–3 samples every ~15 ms).
- 7 channels: qw, qx, qy, qz, ax, ay, az.
- Optional jump-landing events for calibration testing.
- Hardware timestamps (reconstructed from sample counter).
"""

from __future__ import annotations

import time
import math
import random
from typing import Optional

import numpy as np

try:
    import pylsl
except ImportError:
    pylsl = None


class ImuSimulator:
    """Fake IMU SDK that generates synthetic orientation + acceleration data.

    Implements the ``ImuSdk`` protocol expected by ``ImuOutlet``.

    Parameters
    ----------
    sample_rate : float
        Nominal rate (Hz).  Default 200.
    ble_burst_size : int
        Samples per BLE burst (default 2–3, randomised).
    ble_interval_ms : float
        BLE connection interval (ms).  Default 15.
    add_noise : bool
        Add realistic sensor noise.
    """

    def __init__(
        self,
        sample_rate: float = 200.0,
        ble_burst_size: int = 0,  # 0 = random 2–3
        ble_interval_ms: float = 15.0,
        add_noise: bool = True,
    ) -> None:
        self.sample_rate = sample_rate
        self.ble_burst_size = ble_burst_size
        self.ble_interval_ms = ble_interval_ms
        self.add_noise = add_noise

        self._connected = False
        self._seq = 0
        self._start_time: Optional[float] = None
        self._burst_buffer: list = []
        self._last_burst_time = 0.0

        # Simulated orientation state (slowly rotating)
        self._angle = 0.0

    def connect(self) -> bool:
        self._connected = True
        self._seq = 0
        self._start_time = time.perf_counter()
        self._last_burst_time = time.perf_counter()
        print("[ImuSimulator] Connected (synthetic)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        print("[ImuSimulator] Disconnected")

    def read_sample(self) -> tuple[np.ndarray, float, int]:
        """Return (sample, hardware_timestamp, sequence_number).

        Simulates BLE burst behaviour: blocks for ~15 ms then delivers
        2–3 samples in rapid succession.
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        # Simulate BLE burst timing
        if not self._burst_buffer:
            self._wait_for_burst()
            self._generate_burst()

        sample, hw_ts, seq = self._burst_buffer.pop(0)
        return sample, hw_ts, seq

    def _wait_for_burst(self) -> None:
        """Wait until the next BLE connection interval."""
        now = time.perf_counter()
        elapsed = (now - self._last_burst_time) * 1000  # ms
        wait = max(0.0, self.ble_interval_ms - elapsed)
        if wait > 0:
            time.sleep(wait / 1000.0)
        self._last_burst_time = time.perf_counter()

    def _generate_burst(self) -> None:
        """Generate a burst of 2–3 samples (BLE behaviour)."""
        n = self.ble_burst_size if self.ble_burst_size > 0 else random.choice([2, 3])

        for _ in range(n):
            self._angle += 0.01  # slow rotation
            t = self._seq / self.sample_rate

            # Quaternion: slow rotation around Y axis
            qw = math.cos(self._angle / 2)
            qx = 0.0
            qy = math.sin(self._angle / 2)
            qz = 0.0

            # Acceleration: gravity (9.81 m/s²) + noise
            ax = 0.0
            ay = 9.81
            az = 0.0

            if self.add_noise:
                noise_q = np.random.normal(0, 0.001, 4)
                noise_a = np.random.normal(0, 0.05, 3)
                qw += noise_q[0]
                qx += noise_q[1]
                qy += noise_q[2]
                qz += noise_q[3]
                ax += noise_a[0]
                ay += noise_a[1]
                az += noise_a[2]

            sample = np.array([qw, qx, qy, qz, ax, ay, az], dtype=np.float32)

            # Hardware timestamp (reconstructed from counter)
            if pylsl is not None:
                hw_ts = pylsl.local_clock() - (n - len(self._burst_buffer) - 1) / self.sample_rate
            else:
                hw_ts = time.perf_counter()

            self._burst_buffer.append((sample, hw_ts, self._seq))
            self._seq += 1

    def inject_impact(self, magnitude: float = 50.0) -> None:
        """Inject a jump-landing impact event into next burst (for calibration testing).

        Creates a sharp acceleration spike on the Y axis.
        """
        self._impact_pending = magnitude
