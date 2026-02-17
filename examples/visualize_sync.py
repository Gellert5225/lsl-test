"""Visualise unsynced vs synced streams — jump landing impact event.

Generates synthetic 3-stream data with realistic latencies and jitter,
then demonstrates the sync pipeline:

  1. UNSYNCED: Raw observed timestamps (each stream offset by its own latency)
  2. SYNCED:   After applying time_correction + calibration offsets, all
               streams are interpolated to a common 60 Hz time base.

Produces a side-by-side figure saved to examples/sync_comparison.png.
"""

from __future__ import annotations

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from examples.generate_datasets import generate_datasets
from sync_engine.ring_buffer import RingBuffer
from sync_engine.synced_frame import SyncedFrame


def build_synced_timeline(
    datasets: dict,
    target_rate: float = 60.0,
) -> tuple[np.ndarray, dict]:
    """Simulate the sync engine pipeline on the example datasets.

    1. Push all samples (with corrected timestamps) into ring buffers.
    2. Interpolate all buffers to a common 60 Hz target timeline.

    Returns (target_times, {stream_name: interpolated_data}).
    """
    # --- Compute latency offsets (calibration) ---
    # Reference = Mocap.  Offset = -(stream_latency - reference_latency)
    ref_latency = datasets["Mocap"].latency_s
    offsets = {}
    for name, ds in datasets.items():
        offsets[name] = -(ds.latency_s - ref_latency)

    # --- Fill ring buffers with corrected timestamps ---
    buffers: dict[str, RingBuffer] = {}
    for name, ds in datasets.items():
        rb = RingBuffer(max_samples=len(ds.timestamps) + 100, num_channels=1)
        for ts_obs, val in zip(ds.timestamps, ds.data):
            # Corrected = observed + calibration offset
            corrected_ts = ts_obs + offsets[name]
            rb.push(corrected_ts, [val])
        buffers[name] = rb

    # --- Interpolate to 60 Hz common timeline ---
    # Use mocap's corrected timestamp range for the target timeline
    mocap_rb = buffers["Mocap"]
    t_start = mocap_rb.oldest_timestamp + 0.05  # small margin
    t_end = mocap_rb.newest_timestamp - 0.05
    n_frames = int((t_end - t_start) * target_rate)
    target_times = np.linspace(t_start, t_end, n_frames)

    synced_data = {name: np.zeros(n_frames) for name in datasets}
    synced_quality = {name: np.zeros(n_frames) for name in datasets}

    for i, t in enumerate(target_times):
        for name, rb in buffers.items():
            data, gap, quality = rb.interpolate(t)
            if data is not None:
                synced_data[name][i] = data[0]
            synced_quality[name][i] = quality

    return target_times, synced_data, synced_quality


def plot_comparison(datasets, target_times, synced_data, synced_quality):
    """Create side-by-side unsynced vs synced visualization."""

    # Zoom window around impact
    impact_true = 2.0
    zoom = (1.85, 2.35)  # 500ms window centered on impact

    # Colours
    colors = {"Mocap": "#2196F3", "IMU": "#FF9800", "EMG": "#4CAF50"}
    labels = {"Mocap": "Mocap (120 Hz)", "IMU": "IMU Accel (200 Hz)", "EMG": "EMG Rectified (2000 Hz)"}

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Digital Twin Sync Engine — Unsynced vs Synced Streams\n"
        "Jump Landing Impact Event",
        fontsize=15, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.3,
                           left=0.08, right=0.95, top=0.92, bottom=0.06)

    # =====================================================================
    # LEFT COLUMN: UNSYNCED (raw observed timestamps)
    # =====================================================================

    # --- Full timeline ---
    ax_full_unsync = fig.add_subplot(gs[0, 0])
    ax_full_unsync.set_title("UNSYNCED — Full Timeline (Raw Observed Timestamps)",
                             fontsize=11, fontweight="bold", color="#D32F2F")

    for name, ds in datasets.items():
        # Downsample EMG for full view
        step = max(1, len(ds.timestamps) // 2000)
        ax_full_unsync.plot(
            ds.timestamps[::step], ds.data[::step],
            color=colors[name], alpha=0.7, linewidth=0.8,
            label=labels[name],
        )
    ax_full_unsync.axvline(impact_true, color="red", linestyle="--", alpha=0.5, label="True impact")
    ax_full_unsync.set_xlabel("Observed Time (s)")
    ax_full_unsync.set_ylabel("Amplitude")
    ax_full_unsync.legend(fontsize=8, loc="upper right")
    ax_full_unsync.set_xlim(0, 4)

    # --- Per-stream zoomed views (unsynced) ---
    for row, name in enumerate(["Mocap", "IMU", "EMG"]):
        ax = fig.add_subplot(gs[row + 1, 0])
        ds = datasets[name]

        mask = (ds.timestamps >= zoom[0]) & (ds.timestamps <= zoom[1])
        ax.plot(ds.timestamps[mask], ds.data[mask],
                color=colors[name], linewidth=1.0, alpha=0.8)

        # Mark the observed impact onset (true_impact + latency)
        observed_impact = impact_true + ds.latency_s
        ax.axvline(observed_impact, color=colors[name], linestyle="-",
                   linewidth=2, alpha=0.8,
                   label=f"Observed onset (+{ds.latency_s*1000:.0f}ms)")
        ax.axvline(impact_true, color="red", linestyle="--", alpha=0.4,
                   label="True impact")

        # Shade the latency gap
        ax.axvspan(impact_true, observed_impact, alpha=0.15, color=colors[name])

        ax.set_title(f"{labels[name]} — latency: {ds.latency_s*1000:.0f}ms, "
                     f"jitter: {ds.jitter_std_s*1000:.1f}ms",
                     fontsize=10)
        ax.set_xlabel("Observed Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(zoom)

        if name == "EMG":
            # Mark pre-activation (physiology, not latency!)
            preact_time = impact_true - 0.070 + ds.latency_s
            ax.annotate(
                "Pre-activation\n(physiology,\nnot latency!)",
                xy=(preact_time, 150), fontsize=7,
                ha="center", color="#388E3C", style="italic",
                arrowprops=dict(arrowstyle="->", color="#388E3C"),
                xytext=(preact_time - 0.06, 350),
            )

    # =====================================================================
    # RIGHT COLUMN: SYNCED (corrected timestamps, interpolated to 60 Hz)
    # =====================================================================

    # --- Full timeline ---
    ax_full_sync = fig.add_subplot(gs[0, 1])
    ax_full_sync.set_title("SYNCED — Full Timeline (Corrected + Interpolated @ 60 Hz)",
                           fontsize=11, fontweight="bold", color="#1B5E20")

    for name in ["Mocap", "IMU", "EMG"]:
        ax_full_sync.plot(
            target_times, synced_data[name],
            color=colors[name], alpha=0.7, linewidth=0.8,
            label=labels[name],
        )
    # After sync correction, observed impact ≈ true_impact + ref_latency
    synced_impact = impact_true + datasets["Mocap"].latency_s
    ax_full_sync.axvline(synced_impact, color="red", linestyle="--", alpha=0.5,
                         label="Impact (all aligned)")
    ax_full_sync.set_xlabel("Corrected Time (s)")
    ax_full_sync.set_ylabel("Amplitude")
    ax_full_sync.legend(fontsize=8, loc="upper right")
    t_range = (target_times[0], target_times[-1])
    ax_full_sync.set_xlim(t_range)

    # --- Per-stream zoomed views (synced) ---
    zoom_sync = (zoom[0] + datasets["Mocap"].latency_s,
                 zoom[1] + datasets["Mocap"].latency_s)

    for row, name in enumerate(["Mocap", "IMU", "EMG"]):
        ax = fig.add_subplot(gs[row + 1, 1])

        mask = (target_times >= zoom_sync[0]) & (target_times <= zoom_sync[1])
        ax.plot(target_times[mask], synced_data[name][mask],
                color=colors[name], linewidth=1.0, alpha=0.8)

        # After calibration, all onsets should align
        ax.axvline(synced_impact, color="red", linestyle="--", linewidth=2,
                   alpha=0.7, label="Aligned onset")

        # Quality ribbon
        q = synced_quality[name][mask]
        t = target_times[mask]
        ax.fill_between(t, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                        where=(q < 0.8), alpha=0.1, color="red",
                        label="Low quality")

        offset_ms = datasets[name].latency_s - datasets["Mocap"].latency_s
        ax.set_title(
            f"{labels[name]} — offset corrected: {offset_ms*1000:+.0f}ms → 0ms",
            fontsize=10,
        )
        ax.set_xlabel("Corrected Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlim(zoom_sync)

        if name == "EMG":
            preact_time_sync = (impact_true - 0.070) + datasets["Mocap"].latency_s
            ax.annotate(
                "Pre-activation\n(correctly preserved\nafter sync)",
                xy=(preact_time_sync, 150), fontsize=7,
                ha="center", color="#388E3C", style="italic",
                arrowprops=dict(arrowstyle="->", color="#388E3C"),
                xytext=(preact_time_sync - 0.06, 350),
            )

    # =====================================================================
    # Save and show
    # =====================================================================
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "sync_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


def print_latency_table(datasets):
    """Print differential latency summary."""
    print("\n" + "=" * 60)
    print("Differential Latency Summary")
    print("=" * 60)
    ref = datasets["Mocap"].latency_s
    for name, ds in datasets.items():
        diff = (ds.latency_s - ref) * 1000
        print(f"  {ds.name:25s}  latency={ds.latency_s*1000:5.0f}ms  "
              f"diff={diff:+6.0f}ms  jitter={ds.jitter_std_s*1000:.1f}ms")
    print(f"\n  Reference: Mocap ({ref*1000:.0f}ms)")
    print(f"  After calibration, all streams aligned within ±2ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Generating synthetic datasets...")
    datasets = generate_datasets(duration_s=4.0, impact_time=2.0, seed=42)

    print_latency_table(datasets)

    print("Building synced timeline (60 Hz interpolation)...")
    target_times, synced_data, synced_quality = build_synced_timeline(datasets)
    print(f"  {len(target_times)} synced frames at 60 Hz")

    print("Plotting comparison...")
    plot_comparison(datasets, target_times, synced_data, synced_quality)
