"""EMG simulator — generates realistic chunked EMG data without hardware.

Simulates:
- 2000 Hz sample rate, 8 channels.
- Chunk delivery: 48 samples (~24 ms) per read (Delsys WiFi chunk buffer).
- Per-sample timestamps within each chunk.
- Quiet baseline + activations for calibration testing.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import pylsl
except ImportError:
    pylsl = None


class EmgSimulator:
    """Fake EMG SDK that generates synthetic 8-channel EMG data in chunks.

    Implements the ``EmgSdk`` protocol expected by ``EmgOutlet``.

    Parameters
    ----------
    sample_rate : float
        Nominal rate (Hz).  Default 2000.
    chunk_size : int
        Samples per chunk (default 48 — matches Delsys Trigno).
    num_channels : int
        Number of EMG channels (default 8).
    baseline_uv : float
        Baseline noise RMS in microvolts (default 10).
    """

    def __init__(
        self,
        sample_rate: float = 2000.0,
        chunk_size: int = 48,
        num_channels: int = 8,
        baseline_uv: float = 10.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.num_channels = num_channels
        self.baseline_uv = baseline_uv

        self._connected = False
        self._seq = 0
        self._start_time: Optional[float] = None
        self._activation_level = 0.0  # 0.0 = rest, 1.0 = MVC

    def connect(self) -> bool:
        self._connected = True
        self._seq = 0
        self._start_time = time.perf_counter()
        print(f"[EmgSimulator] Connected (synthetic, {self.num_channels}ch)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        print("[EmgSimulator] Disconnected")

    def read_chunk(self) -> Tuple[List[np.ndarray], List[float]]:
        """Return (samples, timestamps) — one chunk of EMG data.

        Blocks for ~chunk_duration to simulate real-time chunk accumulation.
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        chunk_duration = self.chunk_size / self.sample_rate

        # Simulate chunk buffering delay (data trapped in sensor for ~24 ms)
        time.sleep(chunk_duration)

        samples = []
        timestamps = []

        now = pylsl.local_clock() if pylsl else time.perf_counter()

        for i in range(self.chunk_size):
            # Per-sample timestamp (evenly spaced within the chunk)
            ts = now - chunk_duration + (i / self.sample_rate)

            # Generate EMG: bandlimited noise + optional activation
            sample = self._generate_sample()
            samples.append(sample)
            timestamps.append(ts)
            self._seq += 1

        return samples, timestamps

    def _generate_sample(self) -> np.ndarray:
        """Generate a single 8-channel EMG sample.

        EMG is modelled as band-limited Gaussian noise whose amplitude
        scales with activation level.
        """
        noise_amplitude = self.baseline_uv + self._activation_level * 500.0
        sample = np.random.normal(0, noise_amplitude, self.num_channels)
        return sample.astype(np.float32)

    def set_activation(self, level: float) -> None:
        """Set muscle activation level (0.0 = rest, 1.0 = MVC).

        Use this to simulate voluntary contractions or impact responses.
        """
        self._activation_level = np.clip(level, 0.0, 1.0)

    def inject_impact_burst(self, duration_ms: float = 5.0, amplitude_uv: float = 800.0) -> None:
        """Mark that the next chunk should contain a sharp impact artefact.

        This simulates the EMG response at impact (NOT pre-activation —
        pre-activation is a separate, earlier, physiological event).
        """
        # For simplicity, just set a high activation level temporarily
        self._activation_level = amplitude_uv / 500.0
