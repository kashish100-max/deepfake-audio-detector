# Spectro-Temporal Deepfake Detector

A complete PyTorch research prototype for detecting voice and audio deepfakes through novel spectro-temporal fingerprinting.

## What it does

The detector processes each recording through four stages:

1. **Audio quality normalization** — converts audio to 16 kHz mono, trims silence, normalizes peak and RMS loudness, and creates fixed 3-second windows with zero-padding for short clips.
2. **Fingerprint extraction** — creates a three-channel tensor containing log-Mel energy, log-STFT magnitude, and instantaneous-frequency / phase features.
3. **Dual-branch neural model** — combines a spectral 2D CNN with a temporal convolution + bidirectional GRU branch, then fuses them with channel attention.
4. **Screening result** — returns REAL or DEEPFAKE probabilities, per-window predictions, spectrogram plots, and an artifact explanation.

The included dataset generator creates realistic test signals for an immediate end-to-end smoke test. These generated examples are for pipeline validation only; train on representative real data before making claims about production accuracy.

## Project layout

```text
spectro_temporal_deepfake_detector/
├── requirements.txt
├── README.md
├── dataset_prep.py
├── train.py
├── inference.py
├── app.py
└── src/
    ├── __init__.py
    ├── audio_processing.py
    ├── model.py
    └── utils.py
```

## Requirements

- Python 3.10 or 3.11
- CPU is supported; CUDA is used automatically when available
- WAV, MP3, FLAC, OGG, and other formats supported by the local audio decoder

## How to run

From the `spectro_temporal_deepfake_detector` directory:

### 1. Create an environment and install dependencies

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell, use:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Generate sample data

```bash
python dataset_prep.py
```

This creates 60 synthetic real recordings and 60 synthetic deepfake-like recordings in `data/real` and `data/fake`.

### 3. Train the detector

For a quick test run:

```bash
python train.py --epochs 5 --batch_size 8
```

For a fuller local run:

```bash
python train.py --epochs 30 --batch_size 16 --lr 0.001
```

The best checkpoint is written to `checkpoints/best_model.pth`. The latest checkpoint, metrics, and training curves are also saved there.

### 4. Run command-line inference

Use one of the generated test files:

```bash
python inference.py --input data/fake/fake_0000.wav --visualize
```

The report prints the overall verdict, per-segment probabilities, and artifact summary. With `--visualize`, plots are saved under `inference_output/`.

### 5. Launch the interactive web app

```bash
streamlit run app.py
```

Open the local address shown by Streamlit, upload an audio recording, and review the player, verdict, confidence gauge, per-window table, Mel-spectrogram, STFT magnitude, instantaneous-frequency plot, and attention weights.

## Using your own dataset

Keep the same directory convention:

```text
data/
├── real/
│   ├── recording_001.wav
│   └── recording_002.mp3
└── fake/
    ├── generated_001.wav
    └── generated_002.mp3
```

Then train with:

```bash
python train.py --data_dir ./data --epochs 30
```

For credible evaluation, keep speakers, source recordings, and generation systems separated between training and validation/test splits. Include varied microphones, codecs, languages, speech styles, and generator families.

## Notes on interpretation

This is a screening system, not a forensic certificate. Confidence can be affected by background noise, aggressive compression, clipping, very short recordings, language, microphone response, and synthetic generators not represented in training data. A high-confidence result should be reviewed with provenance and human expertise.
