# Digital Twin Sync Engine — Knowledge Base

> **Purpose:** This document captures all architecture decisions, technical constraints,
> and implementation details for the medical digital twin synchronization system.
> It serves as persistent context for AI coding assistants (Copilot, Claude, etc.).
>
> **Last updated:** 2026-02-16
> **Source:** Extended educational sessions covering synchronization theory, sensor
> latency, EMG signal processing, buffer management, LSL internals, and calibration.

---

## 1. Project Overview

**Goal:** In-house software for physical therapists to observe a patient's
musculoskeletal system in real-time via a digital twin, combining data from
multiple sensor types.

**Sensor streams:**
| Sensor | Type | Rate | Channels | Transport | Connection |
|--------|------|------|----------|-----------|------------|
| IMU (e.g., Xsens DOT) | Orientation + acceleration | 200 Hz | 7 (qw,qx,qy,qz,ax,ay,az) | BLE | Wireless to host PC |
| EMG (e.g., Delsys Trigno) | Muscle activation | 2000 Hz | 8 (bilateral quads) | WiFi → USB base station | Base station to host PC |
| Mocap (e.g., OptiTrack) | 3D joint positions | 120 Hz | N markers × 3 | Ethernet (GigE) | Dedicated server or direct |

**Output:** SyncedFrame at 60 Hz containing interpolated data from all streams,
with per-stream quality scores, fed to a 3D renderer for the digital twin.

---

## 2. Synchronization Architecture

### 2.1 The Three Problems

Synchronization requires solving three distinct problems. Each has a different owner:

| Problem | Definition | Solved By | Method |
|---------|-----------|-----------|--------|
| **Clock drift** | Two clocks tick at slightly different rates (±20 ppm). Over 30 min, this creates ~36ms misalignment. | **LSL** | Continuous NTP-like offset probing between outlet and inlet |
| **Differential latency** | Each sensor has a different fixed delay from physical event to LSL timestamp. Mocap ~13ms, IMU ~20ms, EMG ~28ms. | **Your code** | Sync event calibration (jump landing) to measure per-sensor offsets |
| **Jitter** | Irregular arrival timing (BLE bursts, OS scheduling). IMU samples arrive in clumps of 2–3 every ~15ms. | **Your code** | Ring buffer + interpolation using LSL's accurate timestamps |

### 2.2 Lab Streaming Layer (LSL)

**What LSL is:** A C++ library (with Python/C#/Java wrappers) for real-time streaming of time-series data between applications on a network. Runs as threads inside your process, not as a separate service.

**Three layers:**

1. **Discovery (UDP multicast):** Outlets broadcast their existence on 224.0.0.183:16571. Inlets listen and auto-discover. No IP configuration needed.
2. **Transport (TCP):** Reliable, ordered delivery of timestamped samples. Supports `push_sample()` (one at a time) and `push_chunk()` (batched, efficient for high-rate streams like EMG).
3. **Clock sync (UDP probing):** Background thread measures round-trip time between inlet and outlet, estimates clock offset. Uses shortest-RTT filter for best accuracy. Achieves <1ms on wired LAN.

**Key function — `local_clock()`:** Monotonic, high-resolution, per-machine timestamp. Uses `QueryPerformanceCounter` (Windows) or `clock_gettime(CLOCK_MONOTONIC)` (Linux). All LSL timestamps are in this domain.

**Same-machine optimization:** When outlet and inlet are on the same PC, they share the same `local_clock()`. Offset is exactly 0.0 — no probing needed, no estimation error. **This is the recommended starting configuration.**

**What LSL does NOT do:**
- Does not reduce sensor-internal latency
- Does not smooth jitter
- Does not resample, filter, or interpolate
- Does not record to disk (use LabRecorder for that)
- Does not know about sensor-internal processing delays

### 2.3 Recommended Setup

**Phase 1 (start here):** All sensors connected to one PC. All LSL outlets share one `local_clock()`. Zero clock offset, zero network latency. Focus on sync engine correctness.

**Phase 2 (when hardware forces it):** Mocap server on separate machine. LSL clock sync handles the offset automatically. Your sync engine code doesn't change — `time_correction()` simply returns nonzero.

**Reasons to go multi-machine:** Mocap vendor requires dedicated PC (Vicon/OptiTrack), GPU contention between AI mocap and renderer, USB/peripheral limits, physical distance to cameras.

---

## 3. Sensor Latency Profiles

### 3.1 End-to-End Latency Pipeline

Every sensor sample passes through five stages:

```
Physical Event → Transduction → On-board DSP → Transport → Host SDK → LSL Outlet
                  (physics)     (filtering)    (wireless)   (driver)   (timestamp)
```

LSL timestamps are applied at the **last stage**. Everything before that is invisible to LSL and baked into the timestamp.

### 3.2 Per-Sensor Breakdown

**IMU (BLE):**
- Transduction: ~0.1ms (MEMS accelerometer/gyro)
- On-board DSP: ~2ms (Kalman filter → quaternion)
- Transport: **7–20ms** (BLE connection interval wait — THIS IS THE DOMINANT DELAY)
- Host SDK: ~1ms
- **Total: 10–23ms, variable (BLE jitter)**
- Key issue: BLE connection interval creates bursts. 2–3 samples arrive together, then silence for ~15ms. If outlet uses `local_clock()` at arrival, all burst samples get the same timestamp — WRONG. Use SDK hardware timestamps instead.

**EMG (WiFi + chunk buffer):**
- Transduction: ~0.1ms (surface electrodes + ADC)
- On-board DSP: ~1ms (bandpass filter)
- Transport: **24ms** (chunk buffer: 48 samples at 2kHz must accumulate before WiFi transmission)
- Host SDK: ~1ms
- **Total: ~26ms, consistent (low jitter)**
- Key issue: Chunk buffering dominates. Data is trapped inside the sensor for up to 24ms. But Delsys SDK provides per-sample timestamps within the chunk, so original 2kHz timing is recoverable. Use `push_chunk()` with individual timestamps.

**EMG Envelope (additional processing delay):**
- Raw EMG → rectification → low-pass filter (4–6 Hz) = activation envelope
- Low-pass filter adds **150–250ms** group delay depending on cutoff frequency
- This is **mathematical, not fixable** — the filter needs to see a window of samples
- Envelope should be computed AFTER sync alignment, not before
- Cutoff tradeoff: 4 Hz = smoother, more latency; 10 Hz = responsive, noisier; clinical standard: 4–6 Hz

**Mocap (Ethernet, optical):**
- Transduction: ~4ms (camera exposure + 2D centroid extraction)
- Reconstruction: ~4ms (triangulation → 3D positions)
- Labeling/solving: ~3ms (skeleton solver)
- Transport: ~1ms (Ethernet, very fast and consistent)
- **Total: ~12ms, consistent (minimal jitter)**
- Steadiest stream. Hardware-triggered cameras ensure even frame spacing.

**Mocap (AI/depth camera):**
- Neural net inference: **30–150ms** (GPU-dependent, variable)
- High jitter from GPU scheduling variance
- Much less reliable for real-time sync than optical mocap

### 3.3 Differential Latency

The same physical event produces different LSL timestamps from each sensor:
- Mocap stamps at ~13ms
- IMU stamps at ~20ms
- EMG stamps at ~28ms

These must be aligned via sync event calibration (Section 6).

---

## 4. Buffer Architecture

### 4.1 Buffer Types Used

| Buffer | Type | Where | Purpose |
|--------|------|-------|---------|
| **Ring buffer** | Fixed-size, overwrites oldest | Per-stream in sync engine | Store ~2s of timestamped samples for interpolation |
| **Queue** | FIFO, grows unbounded | LSL inlet → processing thread | Thread-safe handoff |
| **Double buffer** | Two buffers swap roles | Sync engine → renderer | Prevent renderer from reading half-updated frame |
| **Chunk buffer** | Batches before send | Inside EMG sensor, inside LSL | Efficient transport for high-rate streams |
| **Jitter buffer** | Fixed delay, absorbs variance | Optional, for BLE IMU input | Smooth bursty BLE arrivals (adds ~20–30ms latency) |

### 4.2 Ring Buffer Implementation

```python
from collections import deque
import numpy as np

class RingBuffer:
    def __init__(self, max_samples=2000):
        self.timestamps = deque(maxlen=max_samples)
        self.data = deque(maxlen=max_samples)

    def push(self, timestamp, sample):
        self.timestamps.append(timestamp)
        self.data.append(np.array(sample, dtype=np.float32))

    def interpolate(self, target_time):
        """Linear interpolation at target_time. Returns (data, gap, quality)."""
        # Binary search for bracketing samples
        # Interpolate between them
        # Quality = 1.0 if bracketed, degrades with distance (0.0 at 50ms gap)
        ...
```

Buffer sizes:
- IMU (200 Hz, 2s): 400 samples × 7 channels × 4 bytes = ~11 KB
- EMG (2000 Hz, 2s): 4000 samples × 8 channels × 4 bytes = ~128 KB
- Mocap (120 Hz, 2s): 240 samples × N channels × 4 bytes = ~few KB

### 4.3 Data Flow

```
Sensors → [BLE/WiFi/Ethernet] → Vendor SDK → LSL Outlet
    → [TCP/loopback] → LSL Inlet → Queue → Pull Thread
    → Ring Buffer (per stream, with corrected timestamps)
    → Sync Engine (interpolate all to target time)
    → Double Buffer → Renderer (60 Hz)
```

---

## 5. Sync Engine Design

### 5.1 Core Loop

```python
# Background pull thread (runs at ~1kHz)
for stream in streams:
    samples, timestamps = stream.inlet.pull_chunk(timeout=0.0)
    correction = stream.inlet.time_correction()
    for sample, ts in zip(samples, timestamps):
        corrected_ts = ts + correction + stream.latency_offset
        stream.ring_buffer.push(corrected_ts, sample)

# Render thread (runs at 60 Hz)
target_time = pylsl.local_clock()
frame = SyncedFrame(target_time)
for stream in streams:
    data, gap, quality = stream.ring_buffer.interpolate(target_time)
    frame.add(stream.name, data, gap, quality)
# Push frame to double buffer → renderer picks up
```

### 5.2 SyncedFrame

```python
class SyncedFrame:
    target_time: float           # The common timestamp all streams are interpolated to
    streams: dict[str, {
        data: np.ndarray,        # Interpolated channel values
        gap: float,              # Distance from target to nearest sample (seconds)
        quality: float,          # 0.0–1.0 (1.0 = perfectly bracketed)
    }]
    overall_quality: float       # min(all stream qualities)
```

### 5.3 Quality Scoring

- Quality 1.0: target_time is between two samples (perfect interpolation)
- Quality degrades linearly with gap distance (0.0 at 50ms gap)
- Per-stream quality allows the renderer to show staleness indicators
- If a stream's quality drops below threshold (e.g., 0.3), flag it visually for the PT

---

## 6. Sync Event Calibration

### 6.1 Purpose

Measure and correct the fixed differential latency between sensors. LSL aligns clocks but can't see sensor-internal delays.

### 6.2 Procedure

1. **Perform sync event:** Jump landing (best), heel strike, or clap. Must produce sharp, unambiguous signals on all sensors simultaneously.
2. **Detect onset per stream:**
   - IMU: accelerometer magnitude exceeds baseline + 3 SD
   - EMG: rectified EMG exceeds baseline + 3 SD (use IMPACT onset, not pre-activation)
   - Mocap: vertical velocity exceeds threshold
3. **Compute offsets:** Pick mocap as reference. Offset = stream_onset − reference_onset.
4. **Apply:** `corrected_ts = lsl_timestamp + latency_offset` (offset is negative for late streams)
5. **Validate:** Second sync event should show all onsets within ±2ms.

### 6.3 The EMG Pre-Activation Trap

**CRITICAL:** During jump landings, muscles activate 50–80ms BEFORE impact (preparatory activation). This is real physiology, NOT sensor latency. Always calibrate on the **impact event** (acceleration spike / position change), never on EMG onset. Getting this wrong introduces ~70ms error that makes EMG appear to fire after movement — physiologically backwards.

### 6.4 Cross-Correlation Method (Alternative)

```python
def find_latency_offset(ref_signal, test_signal, sample_rate):
    ref = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
    test = (test_signal - np.mean(test_signal)) / np.std(test_signal)
    correlation = np.correlate(ref, test, mode='full')
    lags = np.arange(-len(test) + 1, len(ref))
    best_lag = lags[np.argmax(correlation)]
    return best_lag / sample_rate  # seconds
```

### 6.5 Session Workflow

- **Start:** 3–5 sync events (~30 seconds). Average offsets.
- **During:** Apply fixed offsets. Stable within session if hardware unchanged.
- **End:** One more sync event to verify. Flag if drift >5ms (indicates BLE clock drift).

---

## 7. LSL Outlet Best Practices

### 7.1 Timestamping

```python
# ❌ BAD: stamps all burst samples with same arrival time
sample = imu_sdk.read()
outlet.push_sample(sample)  # default: local_clock() at call time

# ✅ GOOD: use SDK hardware timestamps
sample, hw_ts = imu_sdk.read_with_timestamp()
outlet.push_sample(sample, hw_ts + clock_domain_offset)

# ✅ GOOD: reconstruct from sample counter
sample, seq = imu_sdk.read()
reconstructed_ts = first_sample_time + seq * (1.0 / sample_rate)
outlet.push_sample(sample, reconstructed_ts)
```

### 7.2 EMG Chunk Delivery

```python
# Use push_chunk with per-sample timestamps from Delsys SDK
chunk_data = []      # list of [ch1, ch2, ..., ch8]
chunk_timestamps = [] # list of float (per-sample acquisition times)
for sample, ts in delsys_sdk.read_chunk():
    chunk_data.append(sample)
    chunk_timestamps.append(ts + clock_domain_offset)
outlet.push_chunk(chunk_data, chunk_timestamps)
```

### 7.3 Stream Metadata

```python
info = pylsl.StreamInfo("EMG-Quad", "EMG", 8, 2000, pylsl.cf_float32, "delsys_unit_7")
channels = info.desc().append_child("channels")
for name in ["VastusLat_L", "VastusMed_L", ...]:
    ch = channels.append_child("channel")
    ch.append_child_value("label", name)
    ch.append_child_value("unit", "microvolts")
```

---

## 8. EMG Signal Processing

### 8.1 Raw EMG → Activation Envelope Pipeline

```
Raw EMG (±μV, zero-mean noise)
  → Bandpass filter (20–450 Hz) — removes motion artifact + high-freq noise
  → Full-wave rectification (absolute value) — makes all values positive
  → Low-pass filter (4–6 Hz) — smooths into activation envelope
  → Activation envelope (smooth curve, 0 = rest, 1 = max voluntary contraction)
```

### 8.2 Implementation Notes

- Compute envelope AFTER sync alignment (on the raw data in the ring buffer)
- Envelope has inherent ~150–250ms delay from low-pass filter group delay
- Use causal filters only (IIR Butterworth) — cannot use future samples in real-time
- Cutoff frequency: 4 Hz (clinical standard) to 10 Hz (responsive but noisy)
- Normalize to MVC (Maximum Voluntary Contraction) for clinical interpretation

---

## 9. Clinical Quality Requirements

- **Sync accuracy:** <15ms between any two streams after calibration
- **Display latency:** <50ms from physical event to visible on digital twin
- **Per-stream staleness:** Flag data older than 50ms (stale)
- **Session duration:** 30-minute PT sessions typical
- **Drift budget:** <36ms over 30 min (20 ppm crystal). LSL corrects this.

---

## 10. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Sync protocol | LSL | Purpose-built for multi-sensor, handles discovery + transport + clock sync |
| Starting topology | Single machine | Eliminates clock sync complexity. Move to multi-machine only when hardware forces it |
| Buffer type | Ring buffer per stream | Fixed memory, O(1) push, supports timestamp-indexed interpolation |
| Output rate | 60 Hz | Matches typical display refresh. Interpolate all streams to this rate |
| Calibration method | Sync event (jump landing) + threshold detection | Simple, fast (~30s), robust for clinical use |
| EMG processing | Envelope computed after sync | Avoids adding envelope delay to sync pipeline |
| Reference stream | Mocap | Most stable timing, lowest jitter, best baseline |
| Clock correction | `inlet.time_correction()` applied at pull time | Standard LSL practice |

---

## 11. File / Module Structure (Planned)

```
project/
├── sensors/
│   ├── imu_outlet.py         # Xsens SDK → LSL outlet (BLE)
│   ├── emg_outlet.py         # Delsys SDK → LSL outlet (WiFi/USB)
│   └── mocap_outlet.py       # OptiTrack SDK → LSL outlet (Ethernet)
├── sync_engine/
│   ├── ring_buffer.py        # Timestamp-indexed ring buffer
│   ├── sync_engine.py        # Inlet management, pull loop, interpolation
│   ├── synced_frame.py       # Output frame structure
│   └── calibration.py        # Sync event detection + offset computation
├── processing/
│   ├── emg_envelope.py       # Raw → rectified → low-pass → envelope
│   └── imu_orientation.py    # Quaternion → Euler / rotation matrix
├── renderer/
│   └── digital_twin.py       # 3D visualization (consumes SyncedFrames)
├── config/
│   ├── sensor_config.yaml    # Per-sensor settings (rates, channels, offsets)
│   └── calibration.yaml      # Stored latency offsets from last calibration
└── tests/
    ├── test_ring_buffer.py
    ├── test_sync_engine.py
    └── simulators/            # Fake sensors for testing without hardware
        ├── imu_simulator.py
        ├── emg_simulator.py
        └── mocap_simulator.py
```

---

## 12. Glossary

| Term | Definition |
|------|-----------|
| **Drift** | Slow divergence of two clocks over time due to oscillator frequency differences (~20 ppm) |
| **Jitter** | Random variation in sample arrival timing (caused by BLE intervals, OS scheduling, WiFi) |
| **Latency** | Fixed delay from physical event to data availability in code |
| **Differential latency** | Different sensors have different fixed latencies — causes temporal misalignment |
| **PPM** | Parts per million — clock accuracy measure. 1 ppm = 86ms/day drift |
| **Ring buffer** | Fixed-size circular buffer that overwrites oldest data. O(1) operations |
| **Double buffer** | Two buffers that swap reader/writer roles to prevent torn reads |
| **Chunk buffer** | Accumulates N samples before transmitting as a batch |
| **Jitter buffer** | Adds fixed delay to absorb arrival timing variance |
| **Sync event** | Known physical event (jump landing) used to calibrate differential latency |
| **Onset detection** | Finding the exact moment a signal first responds to an event (threshold or cross-correlation) |
| **Activation envelope** | Smoothed representation of muscle activity level derived from raw EMG |
| **SyncedFrame** | Output structure containing interpolated data from all streams at one target timestamp |
| **Quality score** | 0.0–1.0 measure of how well a stream's buffer brackets the target time |
| **local_clock()** | LSL's monotonic high-resolution timestamp function (per-machine) |
| **time_correction()** | LSL's estimated clock offset between outlet and inlet machines |
| **MVC** | Maximum Voluntary Contraction — normalization reference for EMG amplitude |
