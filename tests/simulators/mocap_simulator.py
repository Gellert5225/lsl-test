"""Mocap simulator — generates synthetic motion capture frames without hardware.

Simulates:
- 120 Hz frame rate with minimal jitter (~0.5 ms).
- N markers × 3 coordinates per frame.
- Hardware-triggered camera timing (very consistent).
- Optional jump-landing events for calibration testing.
"""

from __future__ import annotations

import time
import math
from typing import Optional

import numpy as np

try:
    import pylsl
except ImportError:
    pylsl = None


class MocapSimulator:
    """Fake mocap SDK that generates synthetic 3-D marker positions.

    Implements the ``MocapSdk`` protocol expected by ``MocapOutlet``.

    Parameters
    ----------
    sample_rate : float
        Nominal frame rate (Hz).  Default 120.
    num_markers : int
        Number of markers tracked.  Default 23 (full-body skeleton).
    jitter_ms : float
        Frame-to-frame timing jitter (ms).  Default 0.5 (very stable).
    """

    def __init__(
        self,
        sample_rate: float = 120.0,
        num_markers: int = 23,
        jitter_ms: float = 0.5,
    ) -> None:
        self.sample_rate = sample_rate
        self._num_markers = num_markers
        self.num_channels = num_markers * 3
        self.jitter_ms = jitter_ms

        self._connected = False
        self._frame = 0
        self._start_time: Optional[float] = None

        # Simulated skeleton state
        self._base_positions: Optional[np.ndarray] = None
        self._vertical_offset = 0.0  # for jump simulation

    @property
    def num_markers(self) -> int:
        return self._num_markers

    def connect(self) -> bool:
        self._connected = True
        self._frame = 0
        self._start_time = time.perf_counter()
        self._base_positions = self._generate_t_pose()
        print(f"[MocapSimulator] Connected (synthetic, {self._num_markers} markers "
              f"@ {self.sample_rate} Hz)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        print("[MocapSimulator] Disconnected")

    def read_frame(self) -> tuple[np.ndarray, float]:
        """Return (marker_positions, timestamp) for one frame.

        Blocks for ~1/rate to simulate real-time frame delivery.
        """
        if not self._connected:
            raise RuntimeError("Not connected")

        # Frame pacing (hardware-triggered, very consistent)
        interval = 1.0 / self.sample_rate
        jitter = np.random.normal(0, self.jitter_ms / 1000.0)
        time.sleep(max(0.0, interval + jitter))

        # Timestamp (minimal latency)
        ts = pylsl.local_clock() if pylsl else time.perf_counter()

        # Generate frame
        positions = self._generate_frame()
        self._frame += 1

        return positions, ts

    def _generate_t_pose(self) -> np.ndarray:
        """Generate base T-pose marker positions (N markers × 3).

        Simple humanoid skeleton:
        - Markers 0–4:   spine (pelvis → head)
        - Markers 5–9:   left arm (shoulder → fingers)
        - Markers 10–14: right arm
        - Markers 15–18: left leg (hip → foot)
        - Markers 19–22: right leg
        """
        positions = np.zeros((self._num_markers, 3), dtype=np.float32)

        # Spine (vertical)
        for i in range(min(5, self._num_markers)):
            positions[i] = [0.0, 0.8 + i * 0.15, 0.0]

        # Left arm (horizontal)
        for i in range(5, min(10, self._num_markers)):
            j = i - 5
            positions[i] = [-0.2 - j * 0.15, 1.4, 0.0]

        # Right arm
        for i in range(10, min(15, self._num_markers)):
            j = i - 10
            positions[i] = [0.2 + j * 0.15, 1.4, 0.0]

        # Left leg
        for i in range(15, min(19, self._num_markers)):
            j = i - 15
            positions[i] = [-0.1, 0.8 - j * 0.25, 0.0]

        # Right leg
        for i in range(19, min(23, self._num_markers)):
            j = i - 19
            positions[i] = [0.1, 0.8 - j * 0.25, 0.0]

        return positions

    def _generate_frame(self) -> np.ndarray:
        """Generate marker positions for the current frame.

        Applies small breathing motion + optional vertical offset for jumps.
        """
        t = self._frame / self.sample_rate

        # Start from base pose
        positions = self._base_positions.copy()

        # Breathing motion (subtle vertical oscillation)
        breath = 0.005 * math.sin(2 * math.pi * 0.25 * t)
        positions[:, 1] += breath

        # Jump / vertical offset
        positions[:, 1] += self._vertical_offset

        # Small noise (marker tracking noise ~0.5 mm)
        positions += np.random.normal(0, 0.0005, positions.shape).astype(np.float32)

        return positions.flatten()

    def inject_jump_landing(self, drop_height: float = 0.3, duration_s: float = 0.5) -> None:
        """Simulate a jump landing by dropping vertical offset.

        For calibration testing.  Creates a sharp downward position change
        that onset detection can pick up.
        """
        self._vertical_offset = -drop_height
        # Reset after duration (simple approach — in practice use a decay)
        import threading
        def _reset():
            time.sleep(duration_s)
            self._vertical_offset = 0.0
        threading.Thread(target=_reset, daemon=True).start()
