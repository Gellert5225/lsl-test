"""Generate example datasets with realistic timing characteristics.

Creates synthetic sensor data for a jump-landing scenario with:
- Known differential latencies (Mocap ~13ms, IMU ~20ms, EMG ~28ms)
- Realistic jitter per transport (BLE bursts, WiFi chunks, Ethernet stable)
- A clear impact event at t=2.0s for visual comparison

The datasets can be used to demonstrate unsynced vs synced alignment.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict


@dataclass
class SensorDataset:
    """One stream's worth of synthetic timestamped data."""
    name: str
    timestamps: np.ndarray      # shape (N,) — "true" event times + latency + jitter
    true_timestamps: np.ndarray # shape (N,) — ground-truth event times (no latency)
    data: np.ndarray            # shape (N,) — 1-D signal for visualization
    sample_rate: float
    latency_s: float            # fixed differential latency applied
    jitter_std_s: float         # jitter SD applied


def generate_datasets(
    duration_s: float = 4.0,
    impact_time: float = 2.0,
    seed: int = 42,
) -> Dict[str, SensorDataset]:
    """Generate IMU, EMG, and Mocap datasets for a jump-landing event.

    Parameters
    ----------
    duration_s : float
        Total recording duration (seconds).
    impact_time : float
        Time of the jump-landing impact (seconds).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict[str, SensorDataset]
        Keyed by stream name.
    """
    rng = np.random.default_rng(seed)

    datasets = {}

    # ------------------------------------------------------------------
    # Mocap (120 Hz, ~13ms latency, ~0.5ms jitter) — REFERENCE
    # ------------------------------------------------------------------
    rate_mocap = 120.0
    n_mocap = int(duration_s * rate_mocap)
    true_ts_mocap = np.arange(n_mocap) / rate_mocap

    latency_mocap = 0.013  # 13 ms
    jitter_mocap = 0.0005  # 0.5 ms SD

    observed_ts_mocap = (
        true_ts_mocap
        + latency_mocap
        + rng.normal(0, jitter_mocap, n_mocap)
    )

    # Signal: vertical position — drops sharply at impact
    sig_mocap = np.ones(n_mocap) * 1.0  # standing height
    # Pre-jump rise
    pre_jump = (true_ts_mocap >= 1.5) & (true_ts_mocap < impact_time)
    sig_mocap[pre_jump] = 1.0 + 0.15 * np.sin(
        np.pi * (true_ts_mocap[pre_jump] - 1.5) / 0.5
    )
    # Impact: sharp drop
    post_impact = true_ts_mocap >= impact_time
    decay = np.exp(-15 * (true_ts_mocap[post_impact] - impact_time))
    sig_mocap[post_impact] = 1.0 - 0.3 * decay
    # Add small noise
    sig_mocap += rng.normal(0, 0.002, n_mocap)

    datasets["Mocap"] = SensorDataset(
        name="Mocap (120 Hz)",
        timestamps=observed_ts_mocap,
        true_timestamps=true_ts_mocap,
        data=sig_mocap,
        sample_rate=rate_mocap,
        latency_s=latency_mocap,
        jitter_std_s=jitter_mocap,
    )

    # ------------------------------------------------------------------
    # IMU (200 Hz, ~20ms latency, ~5ms jitter from BLE bursts)
    # ------------------------------------------------------------------
    rate_imu = 200.0
    n_imu = int(duration_s * rate_imu)
    true_ts_imu = np.arange(n_imu) / rate_imu

    latency_imu = 0.020  # 20 ms
    jitter_imu = 0.005   # 5 ms SD (BLE connection interval variance)

    # BLE burst pattern: groups of 2-3 samples get similar jitter
    burst_jitter = np.zeros(n_imu)
    i = 0
    while i < n_imu:
        burst_size = rng.choice([2, 3])
        burst_offset = rng.normal(0, jitter_imu)
        for j in range(burst_size):
            if i + j < n_imu:
                burst_jitter[i + j] = burst_offset + rng.normal(0, 0.0005)
        i += burst_size

    observed_ts_imu = true_ts_imu + latency_imu + burst_jitter

    # Signal: accelerometer magnitude — baseline ~9.81, spike at impact
    sig_imu = np.ones(n_imu) * 9.81
    sig_imu += rng.normal(0, 0.1, n_imu)  # sensor noise
    # Impact spike
    impact_mask = (true_ts_imu >= impact_time) & (true_ts_imu < impact_time + 0.05)
    impact_envelope = np.exp(-80 * (true_ts_imu[impact_mask] - impact_time))
    sig_imu[impact_mask] += 40 * impact_envelope
    # Post-impact settling
    settle_mask = (true_ts_imu >= impact_time + 0.05) & (true_ts_imu < impact_time + 0.3)
    sig_imu[settle_mask] += 3 * np.sin(
        30 * (true_ts_imu[settle_mask] - impact_time)
    ) * np.exp(-10 * (true_ts_imu[settle_mask] - impact_time))

    datasets["IMU"] = SensorDataset(
        name="IMU Accel (200 Hz)",
        timestamps=observed_ts_imu,
        true_timestamps=true_ts_imu,
        data=sig_imu,
        sample_rate=rate_imu,
        latency_s=latency_imu,
        jitter_std_s=jitter_imu,
    )

    # ------------------------------------------------------------------
    # EMG (2000 Hz, ~28ms latency, ~1ms jitter but chunk-delivered)
    # ------------------------------------------------------------------
    rate_emg = 2000.0
    n_emg = int(duration_s * rate_emg)
    true_ts_emg = np.arange(n_emg) / rate_emg

    latency_emg = 0.028  # 28 ms
    jitter_emg = 0.001   # 1 ms SD (low, WiFi is stable)

    # Chunk pattern: 48 samples per chunk get uniform jitter
    chunk_jitter = np.zeros(n_emg)
    chunk_size = 48
    for start in range(0, n_emg, chunk_size):
        end = min(start + chunk_size, n_emg)
        chunk_offset = rng.normal(0, jitter_emg)
        chunk_jitter[start:end] = chunk_offset

    observed_ts_emg = true_ts_emg + latency_emg + chunk_jitter

    # Signal: rectified EMG sum — noise at rest, burst at impact
    # Pre-activation 50-80ms BEFORE impact (physiology, not latency!)
    sig_emg = np.abs(rng.normal(0, 10, n_emg))  # baseline noise ~10 µV

    # Pre-activation (muscles fire BEFORE impact — this is real physiology)
    preact_start = impact_time - 0.070  # 70ms before impact
    preact_mask = (true_ts_emg >= preact_start) & (true_ts_emg < impact_time)
    preact_ramp = (true_ts_emg[preact_mask] - preact_start) / 0.070
    sig_emg[preact_mask] += 200 * preact_ramp + np.abs(rng.normal(0, 50, preact_mask.sum()))

    # Impact burst (mechanical artefact + reflex)
    impact_emg_mask = (true_ts_emg >= impact_time) & (true_ts_emg < impact_time + 0.15)
    impact_env_emg = np.exp(-10 * (true_ts_emg[impact_emg_mask] - impact_time))
    sig_emg[impact_emg_mask] += 500 * impact_env_emg + np.abs(
        rng.normal(0, 80, impact_emg_mask.sum())
    )

    # Post-impact sustained activation
    post_emg = (true_ts_emg >= impact_time + 0.15) & (true_ts_emg < impact_time + 0.8)
    sig_emg[post_emg] += 100 * np.exp(
        -3 * (true_ts_emg[post_emg] - impact_time - 0.15)
    )

    datasets["EMG"] = SensorDataset(
        name="EMG Rectified (2000 Hz)",
        timestamps=observed_ts_emg,
        true_timestamps=true_ts_emg,
        data=sig_emg,
        sample_rate=rate_emg,
        latency_s=latency_emg,
        jitter_std_s=jitter_emg,
    )

    return datasets


if __name__ == "__main__":
    ds = generate_datasets()
    for name, d in ds.items():
        print(f"{name}: {len(d.timestamps)} samples, "
              f"latency={d.latency_s*1000:.0f}ms, "
              f"jitter={d.jitter_std_s*1000:.1f}ms")
