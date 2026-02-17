"""Interactive real-time sync visualizer.

Three continuously-streaming synthetic sensors (IMU 200Hz, EMG 2000Hz, Mocap 120Hz)
with realistic medical signals, simulated clock drift, and a toggle to switch
between UNSYNCED and SYNCED views in real time.

Controls:
  - [Sync ON/OFF] button — toggles synchronization
  - Streams run continuously; watch clock drift accumulate over ~30s+
  - Each subplot is labeled with its native frequency
  - Close the window to exit

Usage:
    python examples/interactive_sync.py
"""

from __future__ import annotations

import sys
import os
import time
import threading
import math
from collections import deque

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sync_engine.ring_buffer import RingBuffer


# =====================================================================
# Physiologically-plausible signal generators
# =====================================================================

class PhysiologicalSignalGen:
    """Generates continuous medically-realistic synthetic signals."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self._emg_activation = 0.0
        self._emg_target = 0.0
        self._next_emg_event = 2.0  # seconds until first activation
        self._gait_phase = 0.0

    def imu_sample(self, t: float) -> np.ndarray:
        """7-ch IMU: qw,qx,qy,qz,ax,ay,az — simulates standing with sway + breathing."""
        # Postural sway (slow, ~0.3 Hz) + breathing (~0.25 Hz)
        sway = 0.02 * math.sin(2 * math.pi * 0.3 * t + 0.5)
        breath = 0.01 * math.sin(2 * math.pi * 0.25 * t)

        # Occasional weight shift (~0.1 Hz, larger amplitude)
        shift = 0.04 * math.sin(2 * math.pi * 0.08 * t)

        tilt = sway + breath + shift
        qw = math.cos(tilt / 2)
        qx = math.sin(tilt / 2) * 0.7
        qy = math.sin(tilt / 2) * 0.7
        qz = 0.0

        # Acceleration: gravity + small tremor
        ax = 0.1 * math.sin(2 * math.pi * 8 * t)  # physiological tremor ~8 Hz
        ay = 9.81 + breath * 2
        az = sway * 5

        noise = self.rng.normal(0, 0.02, 7)
        return np.array([qw, qx, qy, qz, ax, ay, az], dtype=np.float32) + noise.astype(np.float32)

    def imu_accel_magnitude(self, sample: np.ndarray) -> float:
        """Extract scalar acceleration magnitude for plotting."""
        return float(np.sqrt(sample[4]**2 + sample[5]**2 + sample[6]**2))

    def emg_sample(self, t: float, dt: float) -> np.ndarray:
        """8-ch EMG: bilateral quads — realistic activation patterns.

        Generates periodic muscle activations that mimic clinical scenarios:
        - Quiet baseline (resting muscle tone)
        - Periodic contractions (simulating repeated exercises)
        - Co-contraction patterns (bilateral)
        """
        # Update activation state machine
        self._next_emg_event -= dt
        if self._next_emg_event <= 0:
            # Toggle between contraction and rest
            if self._emg_target < 0.1:
                # Start contraction — vary intensity
                self._emg_target = self.rng.uniform(0.3, 0.9)
                self._next_emg_event = self.rng.uniform(0.8, 2.0)  # hold duration
            else:
                # Relax
                self._emg_target = 0.0
                self._next_emg_event = self.rng.uniform(1.5, 4.0)  # rest duration

        # Smooth activation transition (muscle can't switch instantly)
        rate = 5.0  # activation rate constant
        self._emg_activation += (self._emg_target - self._emg_activation) * rate * dt

        # Raw EMG = band-limited noise, amplitude scales with activation
        baseline_uv = 8.0  # resting noise floor ~8 µV
        active_uv = 500.0  # max contraction amplitude

        amplitude = baseline_uv + self._emg_activation * active_uv

        # 8 channels with slight asymmetry (bilateral)
        sample = np.zeros(8, dtype=np.float32)
        for ch in range(8):
            # Bilateral asymmetry: left (0-3) vs right (4-7)
            side_factor = 1.0 if ch < 4 else 0.85 + 0.15 * math.sin(0.5 * t)
            sample[ch] = self.rng.normal(0, amplitude * side_factor)

        return sample

    def emg_rectified_sum(self, sample: np.ndarray) -> float:
        """Scalar rectified sum for plotting."""
        return float(np.sum(np.abs(sample)))

    def mocap_sample(self, t: float, n_markers: int = 23) -> np.ndarray:
        """N×3 mocap positions — standing with breathing + sway."""
        positions = np.zeros(n_markers * 3, dtype=np.float32)

        # Base skeleton (simplified)
        for m in range(n_markers):
            base_y = 0.2 + (m / n_markers) * 1.5  # spread vertically
            base_x = 0.1 * math.sin(m * 1.3)
            base_z = 0.05 * math.cos(m * 0.7)

            # Breathing (thorax markers move more)
            breath = 0.008 * math.sin(2 * math.pi * 0.25 * t) * (1.0 if m > 10 else 0.3)

            # Sway
            sway = 0.005 * math.sin(2 * math.pi * 0.3 * t)

            positions[m * 3 + 0] = base_x + sway
            positions[m * 3 + 1] = base_y + breath
            positions[m * 3 + 2] = base_z

        # Marker noise (~0.5mm)
        positions += self.rng.normal(0, 0.0005, len(positions)).astype(np.float32)
        return positions

    def mocap_vertical_center(self, positions: np.ndarray) -> float:
        """Scalar: average vertical (Y) position of all markers."""
        return float(np.mean(positions[1::3]))


# =====================================================================
# Simulated sensor clocks with drift
# =====================================================================

class DriftingClock:
    """Simulates a sensor clock that drifts relative to the reference.

    Real crystals drift at ~20 ppm.  We exaggerate to ~200 ppm (10×)
    so drift is visible within ~30 seconds instead of 30 minutes.
    """

    def __init__(self, drift_ppm: float = 0.0):
        """drift_ppm: positive = this clock runs fast."""
        self.drift_rate = drift_ppm * 1e-6  # fractional
        self.start_wall = time.perf_counter()

    def now(self) -> float:
        """Return this clock's current time (drifted)."""
        elapsed = time.perf_counter() - self.start_wall
        return elapsed * (1.0 + self.drift_rate)


# =====================================================================
# Stream simulator (runs in background thread)
# =====================================================================

class StreamSimulator:
    """Continuously generates samples and pushes to a ring buffer.

    Runs in a background thread at the specified sample rate.
    """

    def __init__(
        self,
        name: str,
        sample_rate: float,
        clock: DriftingClock,
        buffer: RingBuffer,
        generate_fn,
        scalar_fn,
        latency_s: float = 0.0,
        jitter_std_s: float = 0.0,
    ):
        self.name = name
        self.sample_rate = sample_rate
        self.clock = clock
        self.buffer = buffer
        self.generate_fn = generate_fn
        self.scalar_fn = scalar_fn
        self.latency_s = latency_s
        self.jitter_std_s = jitter_std_s

        # Plot buffer (scalar values for display, using observer timestamps)
        self.plot_times: deque = deque(maxlen=int(sample_rate * 8))
        self.plot_values: deque = deque(maxlen=int(sample_rate * 8))
        self.true_times: deque = deque(maxlen=int(sample_rate * 8))

        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._sample_count = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self):
        rng = np.random.default_rng(hash(self.name) & 0xFFFFFFFF)
        interval = 1.0 / self.sample_rate
        next_time = time.perf_counter()

        while self._running:
            now_wall = time.perf_counter()
            if now_wall < next_time:
                time.sleep(max(0, next_time - now_wall))

            sensor_time = self.clock.now()
            true_wall = now_wall - self.clock.start_wall

            # Generate sample
            sample = self.generate_fn(sensor_time)
            scalar = self.scalar_fn(sample)

            # Observed timestamp = sensor_time + latency + jitter
            jitter = rng.normal(0, self.jitter_std_s) if self.jitter_std_s > 0 else 0.0
            observed_ts = sensor_time + self.latency_s + jitter

            with self._lock:
                self.buffer.push(observed_ts, sample)
                self.plot_times.append(observed_ts)
                self.plot_values.append(scalar)
                self.true_times.append(true_wall)

            self._sample_count += 1
            next_time += interval

    def get_plot_data(self):
        """Return snapshot of (times, values, true_times) for plotting."""
        with self._lock:
            return (
                np.array(self.plot_times, dtype=np.float64),
                np.array(self.plot_values, dtype=np.float64),
                np.array(self.true_times, dtype=np.float64),
            )


# =====================================================================
# Interactive application
# =====================================================================

class InteractiveSyncApp:
    """Real-time interactive sync visualization with toggle."""

    WINDOW_S = 4.0      # visible time window (seconds)
    UPDATE_MS = 50       # plot refresh interval (ms) — 20 FPS

    # Clock drift (exaggerated 10× so it's visible in ~30s)
    DRIFT_PPM = {
        "IMU":   +200,   # IMU clock runs fast
        "EMG":   -150,   # EMG clock runs slow
        "Mocap":    0,   # Mocap = reference (no drift)
    }

    LATENCY = {"Mocap": 0.013, "IMU": 0.020, "EMG": 0.028}
    JITTER  = {"Mocap": 0.0005, "IMU": 0.005, "EMG": 0.001}
    COLORS  = {"Mocap": "#2196F3", "IMU": "#FF9800", "EMG": "#4CAF50"}

    def __init__(self):
        self.sync_enabled = False
        self.sig_gen = PhysiologicalSignalGen(np.random.default_rng(42))

        # --- Clocks ---
        self.ref_clock = DriftingClock(drift_ppm=0)
        self.clocks = {
            name: DriftingClock(drift_ppm=ppm)
            for name, ppm in self.DRIFT_PPM.items()
        }

        # --- Ring buffers (for sync interpolation) ---
        self.buffers = {
            "IMU":   RingBuffer(max_samples=200 * 10, num_channels=7),
            "EMG":   RingBuffer(max_samples=2000 * 10, num_channels=8),
            "Mocap": RingBuffer(max_samples=120 * 10, num_channels=69),
        }

        # --- Stream simulators ---
        self.streams = {
            "IMU": StreamSimulator(
                "IMU", 200, self.clocks["IMU"], self.buffers["IMU"],
                self.sig_gen.imu_sample, self.sig_gen.imu_accel_magnitude,
                latency_s=self.LATENCY["IMU"], jitter_std_s=self.JITTER["IMU"],
            ),
            "EMG": StreamSimulator(
                "EMG", 2000, self.clocks["EMG"], self.buffers["EMG"],
                lambda t: self.sig_gen.emg_sample(t, 1/2000),
                self.sig_gen.emg_rectified_sum,
                latency_s=self.LATENCY["EMG"], jitter_std_s=self.JITTER["EMG"],
            ),
            "Mocap": StreamSimulator(
                "Mocap", 120, self.clocks["Mocap"], self.buffers["Mocap"],
                self.sig_gen.mocap_sample, self.sig_gen.mocap_vertical_center,
                latency_s=self.LATENCY["Mocap"], jitter_std_s=self.JITTER["Mocap"],
            ),
        }

        # --- Sync buffers (corrected) ---
        self.sync_buffers = {
            "IMU":   RingBuffer(max_samples=200 * 10, num_channels=1),
            "EMG":   RingBuffer(max_samples=2000 * 10, num_channels=1),
            "Mocap": RingBuffer(max_samples=120 * 10, num_channels=1),
        }

    def run(self):
        """Launch the interactive window."""
        # Start all streams
        for s in self.streams.values():
            s.start()

        time.sleep(0.3)  # let buffers fill a bit

        # --- Setup figure ---
        self.fig, self.axes = plt.subplots(
            3, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={"hspace": 0.35, "top": 0.88, "bottom": 0.10,
                         "left": 0.08, "right": 0.92},
        )
        self.fig.patch.set_facecolor("#1a1a2e")

        self.fig.suptitle(
            "UNSYNCED — Watch clock drift accumulate",
            fontsize=14, fontweight="bold", color="white", y=0.95,
        )

        # Per-stream plot setup
        self.lines = {}
        stream_order = ["IMU", "EMG", "Mocap"]
        freq_labels = {"IMU": "200 Hz", "EMG": "2000 Hz", "Mocap": "120 Hz"}
        y_labels = {
            "IMU": "Accel |a| (m/s²)",
            "EMG": "Rect. EMG (µV)",
            "Mocap": "Vert. Pos (m)",
        }
        drift_labels = {
            "IMU": f"drift: +{self.DRIFT_PPM['IMU']} ppm (fast)",
            "EMG": f"drift: {self.DRIFT_PPM['EMG']} ppm (slow)",
            "Mocap": "drift: 0 ppm (reference)",
        }

        for i, name in enumerate(stream_order):
            ax = self.axes[i]
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white", labelsize=9)
            for spine in ax.spines.values():
                spine.set_color("#333")

            line, = ax.plot([], [], color=self.COLORS[name], linewidth=1.0, alpha=0.9)
            self.lines[name] = line

            # Labels
            ax.set_ylabel(y_labels[name], color="white", fontsize=10)
            ax.set_title(
                f"{name}  [{freq_labels[name]}]   —   "
                f"latency: {self.LATENCY[name]*1000:.0f}ms  |  "
                f"jitter: {self.JITTER[name]*1000:.1f}ms  |  "
                f"{drift_labels[name]}",
                color=self.COLORS[name], fontsize=10, fontweight="bold",
                loc="left", pad=8,
            )

        self.axes[-1].set_xlabel("Time (s)", color="white", fontsize=11)

        # --- Drift indicator text ---
        self.drift_text = self.fig.text(
            0.5, 0.02,
            "", ha="center", fontsize=10, color="#ffab40",
            fontstyle="italic",
        )

        # --- Sync toggle button ---
        ax_btn = self.fig.add_axes([0.82, 0.94, 0.12, 0.04])
        self.btn = Button(
            ax_btn, "Sync: OFF",
            color="#D32F2F", hovercolor="#F44336",
        )
        self.btn.label.set_color("white")
        self.btn.label.set_fontweight("bold")
        self.btn.on_clicked(self._toggle_sync)

        # --- Animation ---
        self.anim = animation.FuncAnimation(
            self.fig, self._update_plot,
            interval=self.UPDATE_MS, blit=False, cache_frame_data=False,
        )

        plt.show()

        # Cleanup
        for s in self.streams.values():
            s.stop()

    def _toggle_sync(self, event):
        self.sync_enabled = not self.sync_enabled
        if self.sync_enabled:
            self.btn.label.set_text("Sync: ON")
            self.btn.color = "#1B5E20"
            self.btn.hovercolor = "#2E7D32"
            self.fig.suptitle(
                "SYNCED — All streams aligned (drift corrected, latency calibrated)",
                fontsize=14, fontweight="bold", color="#4CAF50", y=0.95,
            )
        else:
            self.btn.label.set_text("Sync: OFF")
            self.btn.color = "#D32F2F"
            self.btn.hovercolor = "#F44336"
            self.fig.suptitle(
                "UNSYNCED — Watch clock drift accumulate",
                fontsize=14, fontweight="bold", color="white", y=0.95,
            )
        self.fig.canvas.draw_idle()

    def _update_plot(self, frame_num):
        """Called every UPDATE_MS to refresh the plots."""
        ref_now = self.ref_clock.now()

        # Drift info
        drifts = {}
        for name in ["IMU", "EMG", "Mocap"]:
            sensor_now = self.clocks[name].now()
            drifts[name] = (sensor_now - ref_now) * 1000  # ms

        self.drift_text.set_text(
            f"Clock drift from reference:   "
            f"IMU: {drifts['IMU']:+.1f}ms   |   "
            f"EMG: {drifts['EMG']:+.1f}ms   |   "
            f"Mocap: {drifts['Mocap']:+.1f}ms   "
            f"(elapsed: {ref_now:.1f}s)"
        )

        for i, name in enumerate(["IMU", "EMG", "Mocap"]):
            ax = self.axes[i]

            if self.sync_enabled:
                times, values = self._get_synced_data(name, ref_now)
            else:
                plot_ts, plot_vals, _ = self.streams[name].get_plot_data()
                times, values = plot_ts, plot_vals

            if len(times) == 0:
                continue

            # For EMG, downsample for plotting performance
            if name == "EMG" and len(times) > 2000:
                step = len(times) // 2000
                times = times[::step]
                values = values[::step]

            self.lines[name].set_data(times, values)

            # Scroll the window
            if self.sync_enabled:
                t_right = ref_now
            else:
                t_right = times[-1] if len(times) > 0 else ref_now

            ax.set_xlim(t_right - self.WINDOW_S, t_right + 0.1)

            # Auto-scale Y
            if len(values) > 10:
                visible = values[-min(len(values), int(self.streams[name].sample_rate * self.WINDOW_S)):]
                if len(visible) > 0:
                    ymin, ymax = np.min(visible), np.max(visible)
                    margin = max(0.1, (ymax - ymin) * 0.15)
                    ax.set_ylim(ymin - margin, ymax + margin)

        return list(self.lines.values())

    def _get_synced_data(self, name: str, ref_now: float):
        """Build synced (corrected) plot data for one stream.

        Applies:
        1. Clock drift correction (sensor_time → reference_time)
        2. Latency offset calibration (differential latency removal)
        """
        plot_ts, plot_vals, true_times = self.streams[name].get_plot_data()

        if len(plot_ts) == 0:
            return np.array([]), np.array([])

        # Correction 1: undo clock drift
        #   sensor_ts was stamped by drifted clock. true_times is wall clock.
        #   The difference = what time_correction() would return.
        drift_correction = true_times - plot_ts  # how much the sensor clock is off

        # Correction 2: undo differential latency (calibration offset)
        latency_offset = -(self.LATENCY[name] - self.LATENCY["Mocap"])

        corrected_ts = plot_ts + drift_correction + latency_offset

        return corrected_ts, plot_vals


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Interactive Sync Visualizer")
    print("  - Watch 3 streams with different frequencies")
    print("  - Clock drift accumulates in real time (10× exaggerated)")
    print("  - Click [Sync: OFF] button to toggle synchronization")
    print("  - Close window to exit")
    print("=" * 60)
    app = InteractiveSyncApp()
    app.run()
