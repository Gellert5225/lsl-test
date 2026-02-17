"""Tests for the sync engine — SyncedFrame, DoubleBuffer, calibration.

These tests use the ring buffer and synced frame directly (no LSL required).
"""

import numpy as np
import pytest

from sync_engine.ring_buffer import RingBuffer
from sync_engine.synced_frame import SyncedFrame, StreamSnapshot
from sync_engine.sync_engine import DoubleBuffer, StreamConfig
from sync_engine.calibration import (
    detect_onset,
    calibrate,
    imu_accel_magnitude,
    emg_rectified_sum,
    cross_correlation_offset,
)


# ======================================================================
# SyncedFrame
# ======================================================================

class TestSyncedFrame:

    def test_add_stream(self):
        frame = SyncedFrame(target_time=100.0)
        frame.add("IMU", np.array([1.0, 2.0, 3.0]), gap=0.0, quality=1.0)
        assert "IMU" in frame.streams
        assert frame.overall_quality == 1.0

    def test_overall_quality_is_min(self):
        frame = SyncedFrame(target_time=100.0)
        frame.add("IMU", np.array([1.0]), gap=0.0, quality=0.9)
        frame.add("EMG", np.array([2.0]), gap=0.01, quality=0.5)
        frame.add("Mocap", np.array([3.0]), gap=0.0, quality=1.0)
        assert frame.overall_quality == pytest.approx(0.5)

    def test_none_data_gets_zero_quality(self):
        frame = SyncedFrame(target_time=100.0)
        frame.add("Dead", None, gap=1.0, quality=0.8)
        assert frame.streams["Dead"].quality == 0.0

    def test_is_stale(self):
        frame = SyncedFrame(target_time=100.0)
        frame.add("IMU", np.array([1.0]), gap=0.04, quality=0.2)
        assert frame.is_stale(threshold=0.3)

    def test_stale_streams(self):
        frame = SyncedFrame(target_time=100.0)
        frame.add("IMU", np.array([1.0]), gap=0.0, quality=0.9)
        frame.add("EMG", np.array([2.0]), gap=0.04, quality=0.1)
        assert frame.stale_streams(threshold=0.3) == ["EMG"]

    def test_repr(self):
        frame = SyncedFrame(target_time=100.0)
        frame.add("IMU", np.array([1.0]), gap=0.0, quality=1.0)
        assert "SyncedFrame" in repr(frame)


# ======================================================================
# DoubleBuffer
# ======================================================================

class TestDoubleBuffer:

    def test_read_empty(self):
        db = DoubleBuffer()
        assert db.read() is None

    def test_write_and_read(self):
        db = DoubleBuffer()
        frame = SyncedFrame(target_time=1.0)
        frame.add("IMU", np.array([1.0]), gap=0.0, quality=1.0)
        db.write(frame)
        result = db.read()
        assert result is not None
        assert result.target_time == 1.0

    def test_latest_frame_wins(self):
        db = DoubleBuffer()
        db.write(SyncedFrame(target_time=1.0))
        db.write(SyncedFrame(target_time=2.0))
        result = db.read()
        assert result.target_time == 2.0


# ======================================================================
# StreamConfig
# ======================================================================

class TestStreamConfig:

    def test_buffer_capacity(self):
        cfg = StreamConfig(
            name="IMU-Knee",
            lsl_type="IMU",
            num_channels=7,
            sample_rate=200.0,
            buffer_seconds=2.0,
        )
        assert cfg.buffer_capacity == 400

    def test_emg_capacity(self):
        cfg = StreamConfig(
            name="EMG-Quad",
            lsl_type="EMG",
            num_channels=8,
            sample_rate=2000.0,
            buffer_seconds=2.0,
        )
        assert cfg.buffer_capacity == 4000


# ======================================================================
# Onset Detection
# ======================================================================

class TestOnsetDetection:

    def test_detect_onset_accel(self):
        """Detect impact onset from accelerometer magnitude."""
        np.random.seed(42)
        n = 2000
        rate = 200.0
        timestamps = np.arange(n) / rate

        # Baseline: ~9.81 m/s² with noise
        signal = 9.81 + np.random.normal(0, 0.1, n)

        # Impact at t=5.0 (index 1000): sharp spike
        impact_idx = 1000
        signal[impact_idx : impact_idx + 20] = 50.0  # big spike

        result = detect_onset(
            timestamps, signal,
            baseline_end=4.0,  # first 4 seconds are baseline
            n_sd=3.0,
        )
        assert result is not None
        assert abs(result.onset_time - 5.0) < 0.02  # within 20ms

    def test_no_onset_quiet_signal(self):
        """No onset detected in quiet signal."""
        n = 1000
        timestamps = np.arange(n) / 200.0
        signal = np.random.normal(0, 1, n)  # just noise
        result = detect_onset(timestamps, signal, baseline_end=2.0)
        assert result is None

    def test_imu_accel_magnitude(self):
        """Extract acceleration magnitude from 7-channel IMU data."""
        data = np.array([[1, 0, 0, 0, 3, 4, 0]], dtype=np.float32)  # |a| = 5
        mag = imu_accel_magnitude(data)
        assert mag[0] == pytest.approx(5.0, abs=1e-5)

    def test_emg_rectified_sum(self):
        """Sum of rectified EMG across channels."""
        data = np.array([[-1, 2, -3, 4, -5, 6, -7, 8]], dtype=np.float32)
        total = emg_rectified_sum(data)
        assert total[0] == pytest.approx(36.0)


# ======================================================================
# Calibration
# ======================================================================

class TestCalibration:

    def test_calibrate_basic(self):
        """Compute offsets from known onset times."""
        # Mocap sees onset at t=1.000 (reference)
        # IMU sees onset at t=1.007 (7ms late)
        # EMG sees onset at t=1.015 (15ms late)
        stream_onsets = {
            "Mocap": [1.000, 2.000, 3.000],
            "IMU":   [1.007, 2.008, 3.006],
            "EMG":   [1.015, 2.016, 3.014],
        }
        result = calibrate(stream_onsets, reference_stream="Mocap")

        assert result.offsets["Mocap"] == 0.0
        assert result.offsets["IMU"] == pytest.approx(-0.007, abs=0.001)
        assert result.offsets["EMG"] == pytest.approx(-0.015, abs=0.001)

    def test_calibrate_with_validation(self):
        """Validation event should show small residual after correction."""
        stream_onsets = {
            "Mocap": [1.000],
            "IMU":   [1.010],
        }
        validation = {"Mocap": 5.000, "IMU": 5.010}
        result = calibrate(
            stream_onsets,
            reference_stream="Mocap",
            validation_onsets=validation,
        )
        assert result.is_valid(tolerance=0.005)

    def test_missing_reference_raises(self):
        with pytest.raises(KeyError):
            calibrate({"IMU": [1.0]}, reference_stream="Mocap")


# ======================================================================
# Cross-Correlation
# ======================================================================

class TestCrossCorrelation:

    def test_known_lag(self):
        """Detect a known 10-sample lag at 100 Hz = 100ms."""
        np.random.seed(42)
        n = 1000
        rate = 100.0

        ref = np.zeros(n)
        ref[500:510] = 1.0  # pulse at sample 500

        test = np.zeros(n)
        test[510:520] = 1.0  # same pulse but 10 samples later

        offset = cross_correlation_offset(ref, test, rate)
        assert offset == pytest.approx(-0.1, abs=0.02)  # test lags by 100ms


# ======================================================================
# Integration: RingBuffer → SyncedFrame
# ======================================================================

class TestIntegration:

    def test_multi_stream_sync(self):
        """Simulate building a SyncedFrame from three ring buffers."""
        # Setup buffers
        imu_rb = RingBuffer(max_samples=400, num_channels=7)
        emg_rb = RingBuffer(max_samples=4000, num_channels=8)
        mocap_rb = RingBuffer(max_samples=240, num_channels=69)

        # Fill with synthetic data (1 second)
        base_time = 1000.0
        for i in range(200):
            imu_rb.push(base_time + i / 200.0, np.zeros(7))
        for i in range(2000):
            emg_rb.push(base_time + i / 2000.0, np.zeros(8))
        for i in range(120):
            mocap_rb.push(base_time + i / 120.0, np.zeros(69))

        # Build frame at midpoint
        target = base_time + 0.5
        frame = SyncedFrame(target_time=target)

        for name, rb in [("IMU", imu_rb), ("EMG", emg_rb), ("Mocap", mocap_rb)]:
            data, gap, quality = rb.interpolate(target)
            frame.add(name, data, gap, quality)

        assert frame.overall_quality == 1.0
        assert len(frame.streams) == 3
        assert not frame.is_stale()
