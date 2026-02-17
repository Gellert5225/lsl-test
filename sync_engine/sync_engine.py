"""Sync engine — discovers LSL streams, pulls data, produces SyncedFrames at 60 Hz.

Architecture notes (from Knowledge Base §5.1):
- Background pull thread runs at ~1 kHz, pulling from all inlets with timeout=0.
- time_correction() applied at pull time.
- Per-stream latency_offset from calibration applied at pull time.
- Render/output thread calls ``get_frame()`` at 60 Hz → interpolates all
  ring buffers to a common target_time = local_clock().
- Double buffer prevents the renderer from reading half-updated data.

Data flow (§4.3):
  LSL Inlet → Queue → Pull Thread → Ring Buffer → interpolate → SyncedFrame
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import pylsl
except ImportError:
    pylsl = None  # allow import for testing without pylsl installed

from .ring_buffer import RingBuffer
from .synced_frame import SyncedFrame


# ---------------------------------------------------------------------------
# Stream descriptor
# ---------------------------------------------------------------------------

@dataclass
class StreamConfig:
    """Configures one expected LSL stream.

    Parameters
    ----------
    name : str
        Friendly name (e.g. ``"IMU-Knee"``).
    lsl_type : str
        LSL stream type to match during resolve (e.g. ``"IMU"``, ``"EMG"``).
    lsl_name : str | None
        LSL stream name to match (optional; narrows discovery).
    num_channels : int
        Expected channel count.
    sample_rate : float
        Nominal sample rate (Hz).  Used to size the ring buffer.
    buffer_seconds : float
        How many seconds of history to keep.  Default 2 s.
    latency_offset : float
        Fixed latency correction from sync-event calibration (seconds).
        Negative value means this stream timestamps *later* than reference.
    """

    name: str
    lsl_type: str
    lsl_name: Optional[str] = None
    num_channels: int = 1
    sample_rate: float = 100.0
    buffer_seconds: float = 2.0
    latency_offset: float = 0.0

    @property
    def buffer_capacity(self) -> int:
        return int(self.sample_rate * self.buffer_seconds)


# ---------------------------------------------------------------------------
# Internal per-stream state
# ---------------------------------------------------------------------------

@dataclass
class _StreamState:
    config: StreamConfig
    inlet: object = None  # pylsl.StreamInlet
    ring_buffer: RingBuffer = field(default=None)
    connected: bool = False

    def __post_init__(self):
        if self.ring_buffer is None:
            self.ring_buffer = RingBuffer(
                max_samples=self.config.buffer_capacity,
                num_channels=self.config.num_channels,
            )


# ---------------------------------------------------------------------------
# DoubleBuffer (thread-safe frame hand-off to renderer)
# ---------------------------------------------------------------------------

class DoubleBuffer:
    """Two-slot swap buffer for passing SyncedFrames to the renderer.

    The sync thread writes to the *back* slot, then swaps.  The renderer
    reads from the *front* slot.  A lock guards the swap only — the renderer
    never blocks on a long write.
    """

    def __init__(self) -> None:
        self._front: Optional[SyncedFrame] = None
        self._back: Optional[SyncedFrame] = None
        self._lock = threading.Lock()

    def write(self, frame: SyncedFrame) -> None:
        """Write a new frame (called by sync thread)."""
        self._back = frame
        with self._lock:
            self._front, self._back = self._back, self._front

    def read(self) -> Optional[SyncedFrame]:
        """Read the latest frame (called by renderer)."""
        with self._lock:
            return self._front


# ---------------------------------------------------------------------------
# SyncEngine
# ---------------------------------------------------------------------------

class SyncEngine:
    """Multi-stream synchronisation engine.

    Usage::

        engine = SyncEngine([
            StreamConfig("IMU-Knee",  "IMU", num_channels=7,  sample_rate=200),
            StreamConfig("EMG-Quad",  "EMG", num_channels=8,  sample_rate=2000),
            StreamConfig("Mocap",     "Mocap", num_channels=69, sample_rate=120),
        ])
        engine.start()
        ...
        frame = engine.get_frame()   # call at 60 Hz from renderer
        ...
        engine.stop()

    Parameters
    ----------
    stream_configs : list[StreamConfig]
        One config per expected stream.
    output_rate : float
        Target output rate in Hz (default 60).
    resolve_timeout : float
        Seconds to wait when discovering each stream (default 5).
    frame_callback : callable or None
        Optional callback invoked with each new SyncedFrame.
    """

    def __init__(
        self,
        stream_configs: List[StreamConfig],
        output_rate: float = 60.0,
        resolve_timeout: float = 5.0,
        frame_callback: Optional[Callable[[SyncedFrame], None]] = None,
    ) -> None:
        self.output_rate = output_rate
        self.resolve_timeout = resolve_timeout
        self.frame_callback = frame_callback

        self._streams: Dict[str, _StreamState] = {
            cfg.name: _StreamState(config=cfg) for cfg in stream_configs
        }
        self._double_buffer = DoubleBuffer()
        self._pull_thread: Optional[threading.Thread] = None
        self._output_thread: Optional[threading.Thread] = None
        self._running = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Resolve streams, open inlets, start pull + output threads."""
        if pylsl is None:
            raise RuntimeError(
                "pylsl is not installed.  Install with: pip install pylsl"
            )

        self._resolve_streams()
        self._running.set()

        self._pull_thread = threading.Thread(
            target=self._pull_loop, name="sync-pull", daemon=True
        )
        self._output_thread = threading.Thread(
            target=self._output_loop, name="sync-output", daemon=True
        )
        self._pull_thread.start()
        self._output_thread.start()

    def stop(self) -> None:
        """Signal threads to stop and wait for them to finish."""
        self._running.clear()
        if self._pull_thread:
            self._pull_thread.join(timeout=2.0)
        if self._output_thread:
            self._output_thread.join(timeout=2.0)
        self._close_inlets()

    def get_frame(self) -> Optional[SyncedFrame]:
        """Return the latest SyncedFrame (non-blocking).

        Called by the renderer at its own cadence.
        """
        return self._double_buffer.read()

    def update_latency_offset(self, stream_name: str, offset: float) -> None:
        """Apply a calibrated latency offset for *stream_name*.

        Parameters
        ----------
        stream_name : str
            Must match a configured stream name.
        offset : float
            Seconds.  Typically negative (stream timestamps later than
            reference).
        """
        if stream_name not in self._streams:
            raise KeyError(f"Unknown stream: {stream_name}")
        self._streams[stream_name].config.latency_offset = offset

    @property
    def stream_names(self) -> List[str]:
        return list(self._streams.keys())

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # ------------------------------------------------------------------
    # Stream discovery
    # ------------------------------------------------------------------

    def _resolve_streams(self) -> None:
        """Resolve each configured stream via LSL multicast discovery."""
        for state in self._streams.values():
            cfg = state.config
            pred = f"type='{cfg.lsl_type}'"
            if cfg.lsl_name:
                pred += f" and name='{cfg.lsl_name}'"

            results = pylsl.resolve_bypred(pred, minimum=1, timeout=self.resolve_timeout)
            if not results:
                print(f"[SyncEngine] WARNING: stream '{cfg.name}' not found "
                      f"(pred={pred})")
                continue

            inlet = pylsl.StreamInlet(
                results[0],
                max_buflen=int(cfg.buffer_seconds),
                max_chunklen=0,  # smallest possible chunks
            )
            inlet.open_stream(timeout=self.resolve_timeout)
            state.inlet = inlet
            state.connected = True
            print(f"[SyncEngine] Connected: {cfg.name} "
                  f"({cfg.lsl_type}, {cfg.num_channels}ch @ {cfg.sample_rate} Hz)")

    def _close_inlets(self) -> None:
        for state in self._streams.values():
            if state.inlet is not None:
                try:
                    state.inlet.close_stream()
                except Exception:
                    pass
                state.inlet = None
                state.connected = False

    # ------------------------------------------------------------------
    # Background pull thread (~1 kHz)
    # ------------------------------------------------------------------

    def _pull_loop(self) -> None:
        """Pull samples from all inlets and push into ring buffers.

        Runs in a background thread at ~1 kHz.  Applies:
        1. ``time_correction()`` — LSL clock offset (0.0 on same machine).
        2. ``latency_offset`` — differential latency from calibration.
        """
        while self._running.is_set():
            for state in self._streams.values():
                if not state.connected or state.inlet is None:
                    continue
                self._pull_stream(state)
            time.sleep(0.001)  # ~1 kHz

    def _pull_stream(self, state: _StreamState) -> None:
        """Pull all available samples from one stream's inlet."""
        try:
            samples, timestamps = state.inlet.pull_chunk(timeout=0.0)
        except Exception:
            return

        if not timestamps:
            return

        try:
            correction = state.inlet.time_correction(timeout=0.0)
        except Exception:
            correction = 0.0

        offset = state.config.latency_offset

        for sample, ts in zip(samples, timestamps):
            corrected_ts = ts + correction + offset
            state.ring_buffer.push(corrected_ts, sample)

    # ------------------------------------------------------------------
    # Output thread (60 Hz)
    # ------------------------------------------------------------------

    def _output_loop(self) -> None:
        """Produce SyncedFrames at the target output rate."""
        interval = 1.0 / self.output_rate
        while self._running.is_set():
            t_start = time.perf_counter()

            frame = self._build_frame()
            self._double_buffer.write(frame)

            if self.frame_callback is not None:
                try:
                    self.frame_callback(frame)
                except Exception:
                    pass

            elapsed = time.perf_counter() - t_start
            sleep_time = max(0.0, interval - elapsed)
            time.sleep(sleep_time)

    def _build_frame(self) -> SyncedFrame:
        """Interpolate all ring buffers at the current local_clock()."""
        target_time = pylsl.local_clock()
        frame = SyncedFrame(target_time=target_time)

        for name, state in self._streams.items():
            data, gap, quality = state.ring_buffer.interpolate(target_time)
            frame.add(name, data, gap, quality)

        return frame

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stream_status(self) -> Dict[str, dict]:
        """Return per-stream diagnostic info."""
        status = {}
        for name, state in self._streams.items():
            rb = state.ring_buffer
            status[name] = {
                "connected": state.connected,
                "buffer_fill": len(rb),
                "buffer_capacity": rb.max_samples,
                "newest_ts": rb.newest_timestamp,
                "oldest_ts": rb.oldest_timestamp,
                "latency_offset": state.config.latency_offset,
            }
        return status
