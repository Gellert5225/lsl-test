# Copilot Instructions — Digital Twin Sync Engine

## Project Context

This is a medical digital twin system for physical therapists. It synchronizes three
sensor streams (IMU, EMG, Motion Capture) in real-time using Lab Streaming Layer (LSL).

## Architecture Rules

- **All timestamps must be in LSL's `local_clock()` domain** after applying `time_correction()`.
- **Never stamp BLE burst samples with `local_clock()` at arrival.** Use SDK hardware timestamps or reconstruct from sample counter + known rate.
- **EMG chunks must preserve per-sample timestamps.** Use `push_chunk(data, timestamps)` not `push_chunk(data)`.
- **EMG envelope processing happens AFTER sync alignment**, not before. The envelope's 150–250ms group delay must not contaminate the sync pipeline.
- **Ring buffers are the primary data structure.** Fixed-size (`deque(maxlen=N)`), timestamp-indexed, support interpolation.
- **All streams are interpolated to a common 60 Hz target time.** The SyncedFrame contains interpolated values + per-stream quality scores.
- **Sync event calibration** corrects differential latency. Offsets are measured via jump landing onset detection. Mocap is the reference stream.

## Sensor Characteristics

- **IMU (BLE, 200 Hz):** Arrives in bursts of 2–3 samples every ~15ms due to BLE connection intervals. High jitter. Use SDK timestamps.
- **EMG (WiFi, 2000 Hz, 8ch):** Arrives in chunks of 48 samples (~24ms). Low jitter but high chunk latency. Delsys SDK provides per-sample timestamps.
- **Mocap (Ethernet, 120 Hz):** Arrives frame-by-frame with minimal jitter (~0.5ms). Most stable stream. Reference for calibration.

## Key Types

```python
class RingBuffer:
    """deque-based, stores (timestamp, np.ndarray) pairs. Supports interpolate(target_time) → (data, gap, quality)."""

class SyncedFrame:
    """Contains target_time + dict of {stream_name: {data, gap, quality}}. quality is 0.0–1.0."""

class SyncEngine:
    """Discovers LSL streams, pulls in background thread, stores in per-stream RingBuffers, produces SyncedFrames at 60Hz."""
```

## Common Pitfalls

1. **EMG pre-activation is physiology, not latency.** Muscles fire 50–80ms before impact. Don't "correct" this — it's the signal the PT needs.
2. **Same-machine LSL = zero clock offset.** Don't overcomplicate. `time_correction()` returns 0.0 when outlet and inlet are on the same PC.
3. **Quality scoring must be per-stream.** A stale IMU (BLE gap) shouldn't block fresh mocap data from rendering.
4. **Jitter buffer is optional.** Ring buffer + interpolation handles most jitter. Only add a jitter buffer if BLE variance exceeds 20ms consistently.

## Dependencies

- `pylsl` (≥1.16) — LSL Python bindings
- `numpy` — array operations, interpolation
- Sensor vendor SDKs (Xsens, Delsys, OptiTrack) — for real hardware
