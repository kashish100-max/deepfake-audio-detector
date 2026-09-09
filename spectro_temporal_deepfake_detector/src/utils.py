"""
Plotting utilities for spectrograms, phase, confidence, and attention.

All functions return a ``matplotlib.figure.Figure`` so they can be
embedded in Streamlit via ``st.pyplot(fig)`` or saved to disk.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# ---------------------------------------------------------------------------
# Color scheme — professional blue/teal palette (no purple)
# ---------------------------------------------------------------------------
_REAL_COLOR = "#2ecc71"
_FAKE_COLOR = "#e74c3c"
_SPECTRO_CMAP = "magma"
_PHASE_CMAP = "coolwarm"


# ---------------------------------------------------------------------------
# Spectrogram plots
# ---------------------------------------------------------------------------
def plot_mel_spectrogram(mel: np.ndarray, sr: int = 16000, title: str = "Mel-Spectrogram") -> plt.Figure:
    """Plot a log-Mel spectrogram (N_MELS, T)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        cmap=_SPECTRO_CMAP,
        extent=[0, mel.shape[1] / sr * 1000, 0, mel.shape[0]],
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, format="%+2.0f dB", label="Log-power (dB)")
    fig.tight_layout()
    return fig


def plot_stft_magnitude(mag: np.ndarray, sr: int = 16000, title: str = "STFT Magnitude") -> plt.Figure:
    """Plot a log-STFT magnitude spectrogram (F, T)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(
        mag,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        extent=[0, mag.shape[1] / sr * 1000, 0, sr / 2],
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, format="%+2.0f dB", label="Log-magnitude (dB)")
    fig.tight_layout()
    return fig


def plot_phase(phase: np.ndarray, sr: int = 16000, title: str = "Instantaneous Frequency") -> plt.Figure:
    """Plot the instantaneous-frequency / phase representation (F, T)."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(
        phase,
        aspect="auto",
        origin="lower",
        cmap=_PHASE_CMAP,
        extent=[0, phase.shape[1] / sr * 1000, 0, sr / 2],
    )
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Normalized IF")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Confidence visualization
# ---------------------------------------------------------------------------
def plot_confidence(prob_fake: float, prob_real: float | None = None) -> plt.Figure:
    """Horizontal bar chart showing REAL vs DEEPFAKE confidence."""
    if prob_real is None:
        prob_real = 1.0 - prob_fake

    labels = ["REAL", "DEEPFAKE"]
    values = [prob_real, prob_fake]
    colors = [_REAL_COLOR, _FAKE_COLOR]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="white", linewidth=1.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val*100:.1f}%",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="#333",
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence")
    ax.set_title("Detection Confidence", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_confidence_gauge(prob_fake: float) -> plt.Figure:
    """Semi-circular gauge chart for deepfake probability."""
    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={"aspect": "equal"})

    # Background semicircle
    theta = np.linspace(np.pi, 0, 180)
    ax.plot(np.cos(theta), np.sin(theta), color="#ecf0f1", linewidth=20, solid_capstyle="round")

    # Filled arc proportional to prob_fake
    fill_angle = np.pi * prob_fake
    fill_theta = np.linspace(np.pi, np.pi - fill_angle, max(int(180 * prob_fake), 1))
    arc_color = _FAKE_COLOR if prob_fake > 0.5 else _REAL_COLOR
    if len(fill_theta) > 1:
        ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=arc_color, linewidth=20, solid_capstyle="round")

    # Needle
    needle_angle = np.pi - fill_angle
    ax.annotate(
        "",
        xy=(0.75 * np.cos(needle_angle), 0.75 * np.sin(needle_angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=2.5),
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.3, 1.3)
    ax.axis("off")
    label = "DEEPFAKE" if prob_fake > 0.5 else "REAL"
    ax.text(0, -0.15, f"{label}\n{prob_fake*100:.1f}%", ha="center", va="top",
            fontsize=16, fontweight="bold", color=arc_color)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Attention heatmap
# ---------------------------------------------------------------------------
def plot_attention_heatmap(attention_weights: np.ndarray, title: str = "Channel Attention Weights") -> plt.Figure:
    """Bar chart of channel-attention weights.

    Parameters
    ----------
    attention_weights : np.ndarray
        1-D array of per-channel weights in [0, 1].
    """
    n = len(attention_weights)
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = plt.cm.YlOrRd(Normalize(vmin=0, vmax=1)(attention_weights))
    bars = ax.bar(range(n), attention_weights, color=colors, edgecolor="#333", linewidth=0.5)
    ax.set_xlabel("Fused Channel Index")
    ax.set_ylabel("Weight")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(1.0, attention_weights.max() * 1.15))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Combined fingerprint visualization
# ---------------------------------------------------------------------------
def plot_fingerprint(fingerprint: np.ndarray, sr: int = 16000) -> list[plt.Figure]:
    """Return a list of figures for all 3 channels of a fingerprint tensor.

    Parameters
    ----------
    fingerprint : np.ndarray
        Shape (3, F, T) — channels: Mel, STFT-mag, IF.
    """
    figs = []
    titles = ["Mel-Spectrogram", "STFT Magnitude", "Instantaneous Frequency"]
    cmaps = [_SPECTRO_CMAP, "viridis", _PHASE_CMAP]
    for i in range(3):
        fig, ax = plt.subplots(figsize=(8, 3))
        im = ax.imshow(
            fingerprint[i],
            aspect="auto",
            origin="lower",
            cmap=cmaps[i],
            extent=[0, fingerprint[i].shape[1] / sr * 1000, 0, fingerprint[i].shape[0]],
        )
        ax.set_title(titles[i], fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (ms)")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------
def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
) -> plt.Figure:
    """Plot loss and accuracy curves over epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, "-o", color="#3498db", label="Train Loss", markersize=3)
    ax1.plot(epochs, val_losses, "-o", color="#e67e22", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves", fontweight="bold")
    ax1.legend()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(epochs, train_accs, "-o", color="#2ecc71", label="Train Acc", markersize=3)
    ax2.plot(epochs, val_accs, "-o", color="#e74c3c", label="Val Acc", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves", fontweight="bold")
    ax2.legend()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig
