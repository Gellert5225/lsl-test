"""Tests for RingBuffer — timestamp-indexed ring buffer with interpolation."""

import numpy as np
import pytest

from sync_engine.ring_buffer import RingBuffer


class TestRingBufferBasics:
    """Push, length, capacity, overwrite."""

    def test_empty_buffer(self):
        rb = RingBuffer(max_samples=10, num_channels=3)
        assert len(rb) == 0
        assert rb.newest_timestamp is None
        assert rb.oldest_timestamp is None

    def test_push_and_length(self):
        rb = RingBuffer(max_samples=100, num_channels=2)
        for i in range(10):
            rb.push(float(i), [i, i * 2])
        assert len(rb) == 10

    def test_overwrites_oldest(self):
        rb = RingBuffer(max_samples=5, num_channels=1)
        for i in range(10):
            rb.push(float(i), [i])
        assert len(rb) == 5
        assert rb.oldest_timestamp == 5.0
        assert rb.newest_timestamp == 9.0

    def test_push_chunk(self):
        rb = RingBuffer(max_samples=100, num_channels=2)
        timestamps = [1.0, 1.001, 1.002]
        samples = [[1, 2], [3, 4], [5, 6]]
        rb.push_chunk(timestamps, samples)
        assert len(rb) == 3

    def test_repr(self):
        rb = RingBuffer(max_samples=50, num_channels=3)
        assert "RingBuffer" in repr(rb)


class TestRingBufferInterpolation:
    """Linear interpolation and quality scoring."""

    def _filled_buffer(self) -> RingBuffer:
        """Create a buffer with 10 samples at 100 Hz (t = 0.00 .. 0.09)."""
        rb = RingBuffer(max_samples=100, num_channels=2)
        for i in range(10):
            t = i * 0.01  # 100 Hz
            rb.push(t, [float(i), float(i * 10)])
        return rb

    def test_interpolate_empty(self):
        rb = RingBuffer(max_samples=10, num_channels=1)
        data, gap, quality = rb.interpolate(1.0)
        assert data is None
        assert gap == float("inf")
        assert quality == 0.0

    def test_interpolate_exact_sample(self):
        rb = self._filled_buffer()
        data, gap, quality = rb.interpolate(0.05)  # exact match at i=5
        assert quality == 1.0
        assert gap == 0.0
        np.testing.assert_allclose(data, [5.0, 50.0], atol=1e-5)

    def test_interpolate_between_samples(self):
        rb = self._filled_buffer()
        # Midpoint between sample 3 (t=0.03) and sample 4 (t=0.04)
        data, gap, quality = rb.interpolate(0.035)
        assert quality == 1.0
        assert gap == 0.0
        np.testing.assert_allclose(data, [3.5, 35.0], atol=1e-5)

    def test_interpolate_before_first(self):
        rb = self._filled_buffer()
        data, gap, quality = rb.interpolate(-0.01)
        assert gap == 0.01
        assert quality == pytest.approx(1.0 - 0.01 / 0.05, abs=1e-6)  # 0.8
        np.testing.assert_allclose(data, [0.0, 0.0])  # returns first sample

    def test_interpolate_after_last(self):
        rb = self._filled_buffer()
        data, gap, quality = rb.interpolate(0.12)
        assert gap == pytest.approx(0.12 - 0.09, abs=1e-6)
        # gap = 0.03 → quality = 1 - 0.03/0.05 = 0.4
        assert quality == pytest.approx(0.4, abs=1e-6)

    def test_quality_zero_at_50ms_gap(self):
        rb = self._filled_buffer()
        data, gap, quality = rb.interpolate(0.14)  # 50ms past last sample
        assert quality == pytest.approx(0.0, abs=1e-6)

    def test_quality_zero_beyond_50ms(self):
        rb = self._filled_buffer()
        data, gap, quality = rb.interpolate(1.0)  # way past
        assert quality == 0.0

    def test_linear_interpolation_multi_channel(self):
        rb = RingBuffer(max_samples=10, num_channels=3)
        rb.push(1.0, [0.0, 10.0, 100.0])
        rb.push(2.0, [1.0, 20.0, 200.0])
        data, gap, quality = rb.interpolate(1.5)
        np.testing.assert_allclose(data, [0.5, 15.0, 150.0], atol=1e-5)


class TestRingBufferSizing:
    """Buffer sizes match Knowledge Base §4.2 recommendations."""

    def test_imu_buffer_size(self):
        """IMU: 200 Hz × 2s = 400 samples × 7 channels."""
        rb = RingBuffer(max_samples=400, num_channels=7)
        assert rb.max_samples == 400

    def test_emg_buffer_size(self):
        """EMG: 2000 Hz × 2s = 4000 samples × 8 channels."""
        rb = RingBuffer(max_samples=4000, num_channels=8)
        assert rb.max_samples == 4000

    def test_mocap_buffer_size(self):
        """Mocap: 120 Hz × 2s = 240 samples."""
        rb = RingBuffer(max_samples=240, num_channels=69)
        assert rb.max_samples == 240
