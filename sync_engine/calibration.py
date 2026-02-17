"""Sync event calibration — measure and correct differential sensor latency.

Architecture notes (from Knowledge Base §6):
- Sync event = jump landing (preferred), heel strike, or clap.
- Detect onset per stream via threshold (baseline + 3 SD).
- Pick mocap as reference. offset = stream_onset − reference_onset.
- Apply offset via corrected_ts = lsl_timestamp + latency_offset.
- Validate with a second sync event (all onsets within ±2 ms).

CRITICAL (§6.3 — The EMG Pre-Activation Trap):
  During jump landings, muscles fire 50–80 ms BEFORE impact.  This is
  physiology (preparatory activation), NOT latency.  Always calibrate on
  the impact onset (accel spike / position change), never on EMG onset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OnsetResult:
    """Result of onset detection for one stream."""

    stream_name: str
    onset_time: float          # LSL timestamp of detected onset
    onset_index: int           # Index into the provided signal array
    baseline_mean: float       # Mean of baseline window
    baseline_std: float        # SD of baseline window
    threshold: float           # Threshold used (mean + n_sd * std)
    confidence: float          # 0.0–1.0 — sharpness of onset


@dataclass
class CalibrationResult:
    """Outcome of a full calibration run (3–5 sync events averaged)."""

    reference_stream: str
    offsets: Dict[str, float]  # stream_name → offset in seconds
    per_event_offsets: Dict[str, List[float]]  # for diagnostics
    validation_error: Optional[float] = None   # max residual from validation event

    def is_valid(self, tolerance: float = 0.005) -> bool:
        """True if validation residual is within *tolerance* (default ±5 ms)."""
        if self.validation_error is None:
            return True  # no validation event provided
        return self.validation_error <= tolerance


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def detect_onset(
    timestamps: np.ndarray,
    signal: np.ndarray,
    baseline_end: float,
    n_sd: float = 3.0,
    min_duration_samples: int = 5,
) -> Optional[OnsetResult]:
    """Threshold-based onset detection.

    Parameters
    ----------
    timestamps : ndarray, shape (N,)
        LSL timestamps for each sample.
    signal : ndarray, shape (N,)
        1-D signal to threshold (e.g. accel magnitude, rectified EMG,
        vertical velocity).
    baseline_end : float
        LSL timestamp marking the end of the quiet baseline window.
        Everything before this is used to compute mean & SD.
    n_sd : float
        Number of standard deviations above baseline mean for threshold.
    min_duration_samples : int
        How many consecutive above-threshold samples are required to
        confirm a real onset (avoids spurious spikes).

    Returns
    -------
    OnsetResult or None
        ``None`` if no onset was detected.
    """
    # --- baseline statistics ---
    baseline_mask = timestamps <= baseline_end
    if baseline_mask.sum() < 10:
        return None

    bl_mean = float(np.mean(signal[baseline_mask]))
    bl_std = float(np.std(signal[baseline_mask]))
    if bl_std == 0:
        bl_std = 1e-9  # avoid /0

    threshold = bl_mean + n_sd * bl_std

    # --- scan for onset (first sustained crossing) ---
    above = signal > threshold
    run = 0
    for i, is_above in enumerate(above):
        if is_above:
            run += 1
            if run >= min_duration_samples:
                onset_idx = i - min_duration_samples + 1
                # Confidence: how sharply the signal crosses the threshold
                peak_in_window = float(np.max(signal[onset_idx : i + 1]))
                confidence = min(1.0, (peak_in_window - threshold) / (bl_std * n_sd))
                return OnsetResult(
                    stream_name="",  # caller fills in
                    onset_time=float(timestamps[onset_idx]),
                    onset_index=int(onset_idx),
                    baseline_mean=bl_mean,
                    baseline_std=bl_std,
                    threshold=threshold,
                    confidence=max(0.0, confidence),
                )
        else:
            run = 0

    return None


# ---------------------------------------------------------------------------
# Signal extractors (prepare 1-D signal from multi-channel data)
# ---------------------------------------------------------------------------

def imu_accel_magnitude(data: np.ndarray) -> np.ndarray:
    """Extract accelerometer magnitude from 7-ch IMU data (qw,qx,qy,qz,ax,ay,az).

    Parameters
    ----------
    data : ndarray, shape (N, 7)

    Returns
    -------
    ndarray, shape (N,) — sqrt(ax² + ay² + az²)
    """
    accel = data[:, 4:7]  # ax, ay, az
    return np.sqrt(np.sum(accel ** 2, axis=1))


def emg_rectified_sum(data: np.ndarray) -> np.ndarray:
    """Sum of rectified EMG across all channels.

    Used for impact onset detection ONLY — not for pre-activation detection.

    Parameters
    ----------
    data : ndarray, shape (N, C) — C channels of raw EMG.

    Returns
    -------
    ndarray, shape (N,) — sum of |each channel|.
    """
    return np.sum(np.abs(data), axis=1)


def mocap_vertical_velocity(
    timestamps: np.ndarray,
    positions: np.ndarray,
    vertical_axis: int = 1,
) -> np.ndarray:
    """Compute vertical velocity from 3-D marker positions.

    Parameters
    ----------
    timestamps : ndarray, shape (N,)
    positions : ndarray, shape (N, M*3)
        Flattened marker positions; we take the mean vertical coordinate.
    vertical_axis : int
        Which axis is vertical (0=X, 1=Y, 2=Z).

    Returns
    -------
    ndarray, shape (N,) — absolute vertical velocity (differentiated).
    """
    n_markers = positions.shape[1] // 3
    # Extract vertical coordinate for each marker, then average
    vert = np.mean(
        positions[:, vertical_axis::3].reshape(-1, n_markers), axis=1
    )
    dt = np.diff(timestamps)
    dt[dt == 0] = 1e-9
    vel = np.abs(np.diff(vert) / dt)
    # Pad to match original length
    return np.concatenate([[0.0], vel])


# ---------------------------------------------------------------------------
# Cross-correlation method (alternative to threshold)
# ---------------------------------------------------------------------------

def cross_correlation_offset(
    ref_signal: np.ndarray,
    test_signal: np.ndarray,
    sample_rate: float,
) -> float:
    """Find latency offset between *ref_signal* and *test_signal* via
    normalised cross-correlation.

    Parameters
    ----------
    ref_signal, test_signal : ndarray, shape (N,)
        Signals should be the same length and aligned to the same time window.
    sample_rate : float
        Sample rate (Hz) of both signals (resample first if different).

    Returns
    -------
    float — offset in seconds.  Positive = test lags ref.
    """
    ref = (ref_signal - np.mean(ref_signal)) / (np.std(ref_signal) + 1e-12)
    test = (test_signal - np.mean(test_signal)) / (np.std(test_signal) + 1e-12)
    correlation = np.correlate(ref, test, mode="full")
    lags = np.arange(-len(test) + 1, len(ref))
    best_lag = lags[np.argmax(correlation)]
    return best_lag / sample_rate


# ---------------------------------------------------------------------------
# Full calibration run
# ---------------------------------------------------------------------------

def calibrate(
    stream_onsets: Dict[str, List[float]],
    reference_stream: str = "Mocap",
    validation_onsets: Optional[Dict[str, float]] = None,
) -> CalibrationResult:
    """Compute per-stream latency offsets from multiple sync-event onsets.

    Parameters
    ----------
    stream_onsets : dict[str, list[float]]
        ``{stream_name: [onset_time_event1, onset_time_event2, ...]}``
        Should contain 3–5 onset times per stream.
    reference_stream : str
        Stream to use as the reference (default ``"Mocap"``).
    validation_onsets : dict[str, float] or None
        Optional single validation event's onset times per stream.

    Returns
    -------
    CalibrationResult
    """
    if reference_stream not in stream_onsets:
        raise KeyError(f"Reference stream '{reference_stream}' not in onsets")

    ref_times = stream_onsets[reference_stream]
    offsets: Dict[str, float] = {}
    per_event: Dict[str, List[float]] = {}

    for name, times in stream_onsets.items():
        if name == reference_stream:
            offsets[name] = 0.0
            per_event[name] = [0.0] * len(times)
            continue

        # Per-event offset (stream − reference)
        event_offsets = [
            t - r for t, r in zip(times, ref_times)
        ]
        per_event[name] = event_offsets
        # Average offset — applied as negative to correct
        offsets[name] = -float(np.mean(event_offsets))

    # Validation
    val_error = None
    if validation_onsets is not None and reference_stream in validation_onsets:
        ref_val = validation_onsets[reference_stream]
        residuals = []
        for name, val_t in validation_onsets.items():
            if name == reference_stream:
                continue
            corrected = val_t + offsets.get(name, 0.0)
            residuals.append(abs(corrected - ref_val))
        val_error = max(residuals) if residuals else 0.0

    return CalibrationResult(
        reference_stream=reference_stream,
        offsets=offsets,
        per_event_offsets=per_event,
        validation_error=val_error,
    )
