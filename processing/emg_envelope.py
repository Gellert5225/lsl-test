"""EMG envelope processing — raw EMG → activation envelope.

Architecture notes (from Knowledge Base §8):
- Pipeline: Raw EMG → Bandpass (20–450 Hz) → Rectification → Low-pass (4–6 Hz) → Envelope
- Compute AFTER sync alignment (on raw data from the ring buffer), not before.
- The low-pass filter adds 150–250 ms group delay — this is inherent to the
  filter and NOT a bug to fix.
- Use causal IIR filters only (Butterworth) — cannot use future samples in real-time.
- Cutoff tradeoff: 4 Hz = smoother, more latency; 10 Hz = responsive, noisier.
- Normalize to MVC (Maximum Voluntary Contraction) for clinical interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal as sig


@dataclass
class EnvelopeConfig:
    """Configuration for EMG envelope processing.

    Parameters
    ----------
    sample_rate : float
        EMG sample rate (Hz).  Default 2000.
    bandpass_low : float
        High-pass cutoff for bandpass filter (Hz).  Default 20.
        Removes motion artefact.
    bandpass_high : float
        Low-pass cutoff for bandpass filter (Hz).  Default 450.
        Removes high-frequency noise.
    bandpass_order : int
        Butterworth filter order for bandpass.  Default 4.
    envelope_cutoff : float
        Low-pass cutoff (Hz) for activation envelope.  Default 6.
        Clinical standard: 4–6 Hz.
    envelope_order : int
        Butterworth filter order for envelope low-pass.  Default 4.
    mvc_values : np.ndarray or None
        Maximum Voluntary Contraction per channel for normalisation.
        Shape (n_channels,).  If None, envelope is in raw units.
    """

    sample_rate: float = 2000.0
    bandpass_low: float = 20.0
    bandpass_high: float = 450.0
    bandpass_order: int = 4
    envelope_cutoff: float = 6.0
    envelope_order: int = 4
    mvc_values: Optional[np.ndarray] = None


class EmgEnvelopeProcessor:
    """Real-time EMG envelope processor using causal IIR filters.

    Maintains filter state (zi) across calls so it can process streaming
    chunks without discontinuities.

    Usage::

        proc = EmgEnvelopeProcessor(config, n_channels=8)
        # In sync loop, after alignment:
        envelope = proc.process(raw_chunk)  # shape (N, 8)
    """

    def __init__(self, config: EnvelopeConfig, n_channels: int = 8) -> None:
        self.config = config
        self.n_channels = n_channels

        # --- Design filters ---
        nyq = config.sample_rate / 2.0

        # Bandpass (20–450 Hz) — removes motion artefact + HF noise
        self._bp_b, self._bp_a = sig.butter(
            config.bandpass_order,
            [config.bandpass_low / nyq, config.bandpass_high / nyq],
            btype="bandpass",
        )

        # Low-pass (4–6 Hz) — activation envelope
        self._lp_b, self._lp_a = sig.butter(
            config.envelope_order,
            config.envelope_cutoff / nyq,
            btype="lowpass",
        )

        # --- Initialise filter states (per-channel) ---
        bp_zi_single = sig.lfilter_zi(self._bp_b, self._bp_a)
        lp_zi_single = sig.lfilter_zi(self._lp_b, self._lp_a)

        # Shape: (filter_order, n_channels) for multi-channel lfilter
        self._bp_zi = np.tile(bp_zi_single[:, np.newaxis], (1, n_channels))
        self._lp_zi = np.tile(lp_zi_single[:, np.newaxis], (1, n_channels))

    def process(self, raw: np.ndarray) -> np.ndarray:
        """Process a chunk of raw EMG data into an activation envelope.

        Parameters
        ----------
        raw : ndarray, shape (N, n_channels)
            Raw EMG data in microvolts.

        Returns
        -------
        ndarray, shape (N, n_channels)
            Activation envelope.  If MVC is configured, values are 0.0–1.0
            (fraction of MVC).
        """
        # 1. Bandpass filter (causal)
        filtered, self._bp_zi = sig.lfilter(
            self._bp_b, self._bp_a, raw, axis=0, zi=self._bp_zi
        )

        # 2. Full-wave rectification
        rectified = np.abs(filtered)

        # 3. Low-pass filter → activation envelope (causal)
        envelope, self._lp_zi = sig.lfilter(
            self._lp_b, self._lp_a, rectified, axis=0, zi=self._lp_zi
        )

        # 4. Normalise to MVC if available
        if self.config.mvc_values is not None:
            envelope = envelope / (self.config.mvc_values + 1e-12)

        return envelope.astype(np.float32)

    def reset(self) -> None:
        """Reset filter states (e.g. at start of new recording)."""
        bp_zi_single = sig.lfilter_zi(self._bp_b, self._bp_a)
        lp_zi_single = sig.lfilter_zi(self._lp_b, self._lp_a)
        self._bp_zi = np.tile(bp_zi_single[:, np.newaxis], (1, self.n_channels))
        self._lp_zi = np.tile(lp_zi_single[:, np.newaxis], (1, self.n_channels))

    @property
    def group_delay_ms(self) -> float:
        """Approximate group delay of the envelope low-pass filter in ms.

        This is inherent to the filter — NOT a bug.  The PT should be aware
        that the displayed envelope lags behind raw EMG by this amount.
        """
        # For Butterworth: group delay ≈ order / (2π × cutoff)
        return (
            self.config.envelope_order
            / (2.0 * np.pi * self.config.envelope_cutoff)
            * 1000.0
        )
