"""
Synthetic dummy dataset generator for Spectro-Temporal Deepfake Detection.

Generates a small set of "real" and "fake" audio samples so the entire
pipeline (training + inference + UI) can be tested without external data.

Real samples  — synthetic speech-like signals with natural pitch variation,
                 jitter, and realistic harmonic structure.
Fake samples  — vocoder-like signals with phase inconsistencies, unnaturally
                 stable pitch, and periodic artifacts mimicking GAN/vocoder
                 fingerprints.

Output structure:
    data/
    ├── real/   *.wav
    └── fake/   *.wav
"""

import os
import numpy as np
import soundfile as sf

from src.audio_processing import SAMPLE_RATE, SEGMENT_DURATION

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_REAL = 60
NUM_FAKE = 60
DURATION_RANGE = (2.0, 6.0)  # seconds
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42


def _generate_real_sample(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a speech-like signal with natural characteristics.

    Simulates voiced segments with:
    - Fundamental frequency (F0) drift over time
    - Natural jitter (micro pitch perturbations)
    - Formant-like harmonic structure
    - Breath noise and soft onsets
    """
    n = int(sr * duration)
    t = np.arange(n) / sr

    # Slowly varying F0
    base_f0 = rng.uniform(90, 220)
    f0_drift = base_f0 + 15 * np.sin(2 * np.pi * 0.5 * t) + rng.normal(0, 3, n)

    # Accumulate phase with jitter
    jitter = rng.normal(0, 0.5, n)
    phase = 2 * np.pi * np.cumsum(f0_drift + jitter) / sr

    # Harmonic structure (formants)
    signal = np.zeros(n, dtype=np.float32)
    n_harmonics = rng.integers(4, 8)
    for h in range(1, n_harmonics + 1):
        amp = 1.0 / h * rng.uniform(0.6, 1.0)
        signal += amp * np.sin(h * phase)

    # Amplitude modulation (syllable rhythm ~3-5 Hz)
    syllable_rate = rng.uniform(2.5, 5.0)
    am = 0.5 + 0.5 * np.sin(2 * np.pi * syllable_rate * t + rng.uniform(0, 2 * np.pi))
    signal *= am

    # Soft onset/offset
    fade = int(0.1 * sr)
    if fade * 2 < n:
        signal[:fade] *= np.linspace(0, 1, fade)
        signal[-fade:] *= np.linspace(1, 0, fade)

    # Breath noise
    noise = rng.normal(0, 0.02, n).astype(np.float32)
    signal = 0.8 * signal / (np.max(np.abs(signal)) + 1e-8) + noise

    return signal.astype(np.float32)


def _generate_fake_sample(sr: int, duration: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a vocoder/GAN-like deepfake signal with telltale artifacts.

    Simulates artifacts from HiFi-GAN / BigVGAN / MelGAN:
    - Overly stable F0 (minimal jitter)
    - Phase discontinuities / periodic phase resets
    - Spectral artifacts (ringing, buzziness)
    - Unnatural amplitude envelope
    """
    n = int(sr * duration)
    t = np.arange(n) / sr

    # Stable F0 with minimal drift
    base_f0 = rng.uniform(100, 250)
    f0 = base_f0 + 2 * np.sin(2 * np.pi * 0.3 * t)  # very small drift

    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Phase discontinuities (vocoder artifact)
    num_discontinuities = rng.integers(3, 8)
    for _ in range(num_discontinuities):
        idx = rng.integers(0, n)
        jump = rng.uniform(-np.pi, np.pi)
        phase[idx:] += jump

    # Fewer harmonics, unnaturally strong
    signal = np.zeros(n, dtype=np.float32)
    n_harmonics = rng.integers(3, 5)
    for h in range(1, n_harmonics + 1):
        amp = 0.8 / np.sqrt(h)
        signal += amp * np.sin(h * phase)

    # Buzziness: add a periodic buzz component
    buzz_freq = base_f0 * rng.integers(2, 4)
    signal += 0.15 * np.sign(np.sin(2 * np.pi * buzz_freq * t))

    # Unnatural amplitude envelope (sharper transitions)
    env_rate = rng.uniform(4.0, 8.0)
    env = 0.5 + 0.5 * np.square(np.sin(2 * np.pi * env_rate * t))
    signal *= env

    # Abrupt onsets (no natural fade)
    cut = int(0.02 * sr)
    if cut * 2 < n:
        signal[:cut] *= np.linspace(0, 1, cut) ** 0.3

    # Spectral ringing artifact
    ring_freq = rng.uniform(2000, 6000)
    signal += 0.05 * np.sin(2 * np.pi * ring_freq * t) * np.sin(2 * np.pi * 50 * t)

    # Low noise (vocoders often have lower noise floor)
    noise = rng.normal(0, 0.005, n).astype(np.float32)
    signal = 0.9 * signal / (np.max(np.abs(signal)) + 1e-8) + noise

    return signal.astype(np.float32)


def generate_dataset(
    output_dir: str = OUTPUT_DIR,
    num_real: int = NUM_REAL,
    num_fake: int = NUM_FAKE,
    sr: int = SAMPLE_RATE,
    seed: int = SEED,
) -> None:
    """Generate the full synthetic dataset."""
    rng = np.random.default_rng(seed)

    real_dir = os.path.join(output_dir, "real")
    fake_dir = os.path.join(output_dir, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    print("=" * 60)
    print("Generating synthetic dummy dataset")
    print(f"  Real samples: {num_real}")
    print(f"  Fake samples: {num_fake}")
    print(f"  Sample rate:  {sr} Hz")
    print(f"  Output dir:   {output_dir}")
    print("=" * 60)

    for i in range(num_real):
        duration = rng.uniform(*DURATION_RANGE)
        audio = _generate_real_sample(sr, duration, rng)
        path = os.path.join(real_dir, f"real_{i:04d}.wav")
        sf.write(path, audio, sr)
        if (i + 1) % 10 == 0:
            print(f"  [REAL] {i+1}/{num_real} generated")

    for i in range(num_fake):
        duration = rng.uniform(*DURATION_RANGE)
        audio = _generate_fake_sample(sr, duration, rng)
        path = os.path.join(fake_dir, f"fake_{i:04d}.wav")
        sf.write(path, audio, sr)
        if (i + 1) % 10 == 0:
            print(f"  [FAKE] {i+1}/{num_fake} generated")

    print(f"\nDone! Dataset saved to: {output_dir}")
    print(f"  {num_real} real samples in: {real_dir}")
    print(f"  {num_fake} fake samples in: {fake_dir}")


if __name__ == "__main__":
    generate_dataset()
