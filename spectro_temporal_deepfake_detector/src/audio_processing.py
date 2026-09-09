"""
Audio processing module for Spectro-Temporal Deepfake Detection.

Handles loading, normalization, segmentation, and multi-channel
spectro-temporal fingerprint extraction (Mel-spectrogram, STFT magnitude,
and instantaneous-frequency / phase representation).
"""

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import hilbert

# ---------------------------------------------------------------------------
# Global constants — tuned for speech at 16 kHz mono
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
SEGMENT_DURATION = 3.0  # seconds per analysis window
SEGMENT_LENGTH = int(SAMPLE_RATE * SEGMENT_DURATION)  # 48 000 samples
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
TARGET_RMS_DB = -20.0  # normalization target loudness


# ---------------------------------------------------------------------------
# 1. Audio Input & Quality Normalization
# ---------------------------------------------------------------------------
def load_audio(
    path: str, sr: int = SAMPLE_RATE, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Load an audio file, resample to *sr*, and convert to mono if requested.

    Returns (samples, sample_rate).
    """
    audio, sample_rate = librosa.load(path, sr=sr, mono=mono)
    return audio.astype(np.float32), sample_rate


def load_audio_from_bytes(
    raw: bytes, sr: int = SAMPLE_RATE, mono: bool = True
) -> tuple[np.ndarray, int]:
    """Load audio from an in-memory byte buffer (uploaded file)."""
    import io
    import os
    import tempfile

    try:
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32")
        if mono and audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != sr:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sr)
        return audio.astype(np.float32), sr
    except (RuntimeError, ValueError):
        # SoundFile may not decode compressed formats such as MP3 on every platform.
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as handle:
            handle.write(raw)
            temp_path = handle.name
        try:
            audio, _ = librosa.load(temp_path, sr=sr, mono=mono)
            return audio.astype(np.float32), sr
        finally:
            os.unlink(temp_path)


def trim_silence(audio: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading/trailing silence below *top_db* from the signal."""
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed


def normalize_audio(audio: np.ndarray, target_db: float = TARGET_RMS_DB) -> np.ndarray:
    """Peak-normalize then RMS-normalize to a fixed target loudness (dBFS).

    This ensures consistent energy across recordings regardless of
    microphone gain or distance.
    """
    # Guard against all-zero audio
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio
    audio = audio / peak  # peak-normalize to [-1, 1]

    rms = np.sqrt(np.mean(audio**2) + 1e-10)
    if rms < 1e-8:
        return audio
    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain = 10.0 ** (gain_db / 20.0)
    audio = audio * gain

    # Final clip to prevent overflow
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def segment_audio(
    audio: np.ndarray,
    segment_length: int = SEGMENT_LENGTH,
    hop_length: int | None = None,
) -> np.ndarray:
    """Split *audio* into fixed-length overlapping or non-overlapping segments.

    Each segment is exactly *segment_length* samples; shorter tails are
    zero-padded.  Returns an array of shape (num_segments, segment_length).
    """
    if hop_length is None:
        hop_length = segment_length  # non-overlapping by default

    if len(audio) < segment_length:
        padded = np.zeros(segment_length, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded[np.newaxis, :]

    segments = []
    start = 0
    while start < len(audio):
        end = start + segment_length
        chunk = audio[start:end]
        if len(chunk) < segment_length:
            padded = np.zeros(segment_length, dtype=np.float32)
            padded[: len(chunk)] = chunk
            segments.append(padded)
        else:
            segments.append(chunk)
        start += hop_length

    return np.stack(segments, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Spectro-Temporal Fingerprint Extraction
# ---------------------------------------------------------------------------
def _mel_spectrogram(segment: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute log-scaled Mel-spectrogram of shape (N_MELS, T)."""
    mel = librosa.feature.melspectrogram(
        y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # Normalize to zero-mean / unit-variance for stable training
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)
    return log_mel.astype(np.float32)


def _stft_magnitude(segment: np.ndarray) -> np.ndarray:
    """Compute log-scaled STFT magnitude spectrogram of shape (F, T)."""
    stft = librosa.stft(segment, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft)
    log_mag = librosa.amplitude_to_db(mag, ref=np.max)
    log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-8)
    return log_mag.astype(np.float32)


def _instantaneous_frequency(segment: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Compute instantaneous-frequency spectrogram from analytic-signal phase.

    Phase inconsistencies are a strong tell for vocoder/GAN-based deepfakes
    (HiFi-GAN, BigVGAN, MelGAN) which often produce unwrapped phase patterns
    that differ from natural speech.  We derive IF as the temporal derivative
    of the phase of the analytic STFT, yielding a (F, T) tensor in Hz.
    """
    # STFT of the real signal
    stft = librosa.stft(segment, n_fft=N_FFT, hop_length=HOP_LENGTH)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # Analytic signal per frame via Hilbert transform along time axis
    # We approximate analytic STFT by using the complex STFT directly.
    phase = np.angle(stft)  # (F, T)

    # Unwrap phase along time axis, then differentiate to get IF
    unwrapped = np.unwrap(phase, axis=1)
    if_raw = np.diff(unwrapped, axis=1, prepend=unwrapped[:, :1])
    # Convert radians/sample to Hz:  IF = dphi/dt * sr / (2*pi)
    if_hz = if_raw * sr / (2.0 * np.pi)

    # Wrap to [0, sr/2] by folding — but for normalization we just clip
    if_hz = np.clip(if_hz, 0, sr / 2)

    # Normalize
    if_norm = (if_hz - if_hz.mean()) / (if_hz.std() + 1e-8)
    return if_norm.astype(np.float32)


def extract_fingerprint(segment: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract a 3-channel spectro-temporal fingerprint tensor.

    Channels:
        0 — Log-Mel spectrogram       (N_MELS, T)
        1 — Log-STFT magnitude         (F, T)
        2 — Instantaneous frequency    (F, T)

    All channels are resized to a common (N_MELS, T_target) grid so they can
    be stacked into a single (3, N_MELS, T_target) tensor.
    """
    mel = _mel_spectrogram(segment, sr)  # (N_MELS, T_mel)
    mag = _stft_magnitude(segment)  # (F, T_mag)
    ifreq = _instantaneous_frequency(segment, sr)  # (F, T_if)

    # Resize magnitude and IF to match Mel's (N_MELS, T) grid
    T_target = mel.shape[1]
    F_target = N_MELS

    mag_resized = _resize_2d(mag, (F_target, T_target))
    if_resized = _resize_2d(ifreq, (F_target, T_target))

    fingerprint = np.stack([mel, mag_resized, if_resized], axis=0)  # (3, F, T)
    return fingerprint.astype(np.float32)


def _resize_2d(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear-ish resize of a 2D array via repeated interpolation."""
    h_src, w_src = arr.shape
    h_tgt, w_tgt = target_shape

    # Resize width (time axis)
    if w_src != w_tgt:
        arr = np.array(
            [
                np.interp(
                    np.linspace(0, w_src - 1, w_tgt),
                    np.arange(w_src),
                    arr[row],
                )
                for row in range(h_src)
            ]
        )

    # Resize height (frequency axis)
    if h_src != h_tgt:
        arr = np.array(
            [
                np.interp(
                    np.linspace(0, h_src - 1, h_tgt),
                    np.arange(h_src),
                    arr[:, col],
                )
                for col in range(w_tgt)
            ]
        ).T

    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience: full pipeline for a single file
# ---------------------------------------------------------------------------
def process_file(
    path: str, sr: int = SAMPLE_RATE
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load, normalize, segment, and extract fingerprints from an audio file.

    Returns (raw_audio, list_of_fingerprint_tensors).
    """
    audio, _ = load_audio(path, sr=sr)
    audio = trim_silence(audio)
    audio = normalize_audio(audio)
    segments = segment_audio(audio)
    fingerprints = [extract_fingerprint(seg, sr) for seg in segments]
    return audio, fingerprints


def process_audio_array(
    audio: np.ndarray, sr: int = SAMPLE_RATE
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Same as process_file but accepts a pre-loaded audio array."""
    audio = trim_silence(audio)
    audio = normalize_audio(audio)
    segments = segment_audio(audio)
    fingerprints = [extract_fingerprint(seg, sr) for seg in segments]
    return audio, fingerprints
