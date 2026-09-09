"""Spectro-Temporal Deepfake Detector package."""

from .audio_processing import (
    load_audio,
    normalize_audio,
    segment_audio,
    extract_fingerprint,
    SAMPLE_RATE,
    SEGMENT_DURATION,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
)
from .model import SpectroTemporalDetector, MODEL_CONFIG
from .utils import (
    plot_mel_spectrogram,
    plot_stft_magnitude,
    plot_phase,
    plot_confidence,
    plot_attention_heatmap,
)

__all__ = [
    "load_audio",
    "normalize_audio",
    "segment_audio",
    "extract_fingerprint",
    "SAMPLE_RATE",
    "SEGMENT_DURATION",
    "N_FFT",
    "HOP_LENGTH",
    "N_MELS",
    "SpectroTemporalDetector",
    "MODEL_CONFIG",
    "plot_mel_spectrogram",
    "plot_stft_magnitude",
    "plot_phase",
    "plot_confidence",
    "plot_attention_heatmap",
]
