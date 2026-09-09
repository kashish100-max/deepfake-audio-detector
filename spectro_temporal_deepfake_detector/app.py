"""
Interactive Streamlit UI for Spectro-Temporal Deepfake Detection.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from src.audio_processing import (
    load_audio_from_bytes,
    trim_silence,
    normalize_audio,
    segment_audio,
    extract_fingerprint,
    SAMPLE_RATE,
    SEGMENT_LENGTH,
)
from src.model import build_model, MODEL_CONFIG
from src.utils import (
    plot_confidence,
    plot_confidence_gauge,
    plot_fingerprint,
    plot_attention_heatmap,
)

# ---------------------------------------------------------------------------
# Page setup and theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spectro-Temporal Deepfake Detector",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --navy: #0b1f33; --blue: #1d6fa5; --teal: #18a999; --red: #d94f4f; --muted: #64748b; }
    .main { background: #f5f8fb; }
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    .hero { background: linear-gradient(135deg, #0b1f33 0%, #123b5d 55%, #146b73 100%); border-radius: 20px; padding: 2.25rem 2.5rem; color: white; margin-bottom: 1.5rem; box-shadow: 0 12px 35px rgba(11,31,51,.16); }
    .hero h1 { font-size: 2.15rem; margin: 0 0 .55rem 0; letter-spacing: -.03em; color: #ffffff; }
    .hero p { color: #dcecf5; margin: 0; font-size: 1.05rem; max-width: 840px; line-height: 1.55; }
    .eyebrow { color: #7fe4d4; text-transform: uppercase; letter-spacing: .15em; font-size: .72rem; font-weight: 800; margin-bottom: .7rem; }
    .metric-card { background: #ffffff; border: 1px solid #e4edf3; border-radius: 14px; padding: 1rem 1.15rem; box-shadow: 0 4px 14px rgba(24,55,80,.05); }
    .metric-label { color: #64748b; text-transform: uppercase; letter-spacing: .08em; font-size: .7rem; font-weight: 800; }
    .metric-value { color: #0b1f33; font-size: 1.4rem; font-weight: 800; margin-top: .25rem; }
    .verdict-real { background: #eaf8f1; color: #11734b; border: 1px solid #bdebd3; border-radius: 16px; padding: 1.5rem; text-align: center; }
    .verdict-fake { background: #fff0ef; color: #b83232; border: 1px solid #f5c7c4; border-radius: 16px; padding: 1.5rem; text-align: center; }
    .verdict-title { font-size: 2rem; font-weight: 900; letter-spacing: .04em; }
    .verdict-subtitle { margin-top: .35rem; color: inherit; opacity: .85; }
    .section-label { color: #0b1f33; font-size: 1.25rem; font-weight: 800; margin: 1.6rem 0 .8rem; }
    .explanation { background: #ffffff; border-radius: 14px; padding: 1.25rem 1.4rem; border: 1px solid #e4edf3; color: #334155; line-height: 1.65; }
    .explanation strong { color: #0b1f33; }
    .stButton > button { border-radius: 10px; font-weight: 700; }
    [data-testid="stSidebar"] { background: #0b1f33; }
    [data-testid="stSidebar"] * { color: #e7f1f6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached model loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_detector(checkpoint_path: str):
    """Load checkpoint once per Streamlit session."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(MODEL_CONFIG, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device, checkpoint


def analyse_audio(raw_bytes: bytes, model: torch.nn.Module, device: str) -> dict:
    """Run the complete normalization, fingerprint, and prediction pipeline."""
    audio, sample_rate = load_audio_from_bytes(raw_bytes, sr=SAMPLE_RATE)
    audio = trim_silence(audio)
    audio = normalize_audio(audio)
    segments = segment_audio(audio, SEGMENT_LENGTH)
    fingerprints = [extract_fingerprint(seg, SAMPLE_RATE) for seg in segments]
    batch = torch.from_numpy(np.stack(fingerprints)).to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).cpu().numpy().flatten()
        attention = model.extract_attention_weights(batch[:1]).cpu().numpy().flatten()

    mean_fake = float(np.mean(probs))
    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "segments": segments,
        "fingerprints": fingerprints,
        "segment_probs": probs,
        "prob_fake": mean_fake,
        "prob_real": 1.0 - mean_fake,
        "attention": attention,
    }


def show_explanation(prob_fake: float) -> None:
    """Display an accessible explanation of model evidence."""
    if prob_fake > 0.5:
        st.markdown(
            """
            <div class="explanation">
            <strong>Why this was flagged:</strong> The fingerprint contains patterns that are
            more consistent with generated audio than natural speech. The strongest evidence
            comes from phase/instantaneous-frequency irregularities, unnaturally regular
            spectral harmonics, and timing patterns across the analysed segment.
            <br><br><strong>Important:</strong> This is an AI screening result, not proof of
            fraud. Use it alongside source verification and human review.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="explanation">
            <strong>Why this was classified as real:</strong> The fingerprint shows coherent
            phase behaviour, natural variation in harmonic energy, and timing patterns that
            are more consistent with an organic recording than a vocoder-generated sample.
            <br><br><strong>Important:</strong> A real result means no strong synthetic signal
            was found; it cannot guarantee that an audio file is authentic.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar: setup and upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Detector setup")
    st.caption("Novel spectro-temporal fingerprinting")
    st.markdown("---")

    default_checkpoint = os.path.join(os.path.dirname(__file__), "checkpoints", "best_model.pth")
    checkpoint_path = st.text_input("Model checkpoint", value=default_checkpoint)
    checkpoint_exists = os.path.exists(checkpoint_path)

    if checkpoint_exists:
        st.success("Trained model ready")
    else:
        st.warning("No checkpoint found")
        st.caption("Train a model with train.py, then refresh this page.")

    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown(
        "1. **Normalize** — 16 kHz mono, silence trim, loudness normalization\n"
        "2. **Fingerprint** — Mel, STFT magnitude, instantaneous frequency\n"
        "3. **Fuse** — spectral CNN + temporal Conv/GRU + attention\n"
        "4. **Decide** — calibrated real/deepfake confidence"
    )
    st.markdown("---")
    st.caption("Supported formats: WAV, MP3, FLAC")
    st.caption("Analysis window: 3 seconds")
    st.caption("Designed for research screening")

# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Audio authenticity lab</div>
        <h1>Voice / Audio Deepfake Detector</h1>
        <p>Upload a recording to inspect its spectro-temporal fingerprint and estimate whether its acoustic signature is natural or synthetically generated.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not checkpoint_exists:
    st.info("Upload is available after a trained checkpoint is created. Start with the sample-data and training commands in the project guide.")
    st.stop()

# Load model
try:
    detector, device, checkpoint_info = load_detector(checkpoint_path)
except Exception as exc:
    st.error(f"The model could not be loaded. Check that the checkpoint matches this project. Details: {exc}")
    st.stop()

# Upload section
st.markdown('<div class="section-label">1. Choose an audio sample</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a WAV, MP3, or FLAC file here",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="The file is processed locally by this app and split into 3-second analysis windows.",
)

if uploaded is None:
    st.markdown("Upload a recording to see the prediction, confidence breakdown, and fingerprint views.")
    st.stop()

# Audio playback
raw_bytes = uploaded.getvalue()
st.audio(raw_bytes, format=uploaded.type or "audio/wav")

with st.spinner("Extracting fingerprint and evaluating the recording..."):
    try:
        result = analyse_audio(raw_bytes, detector, device)
    except Exception as exc:
        st.error(f"This audio file could not be analysed: {exc}")
        st.stop()

# Summary metrics
st.markdown('<div class="section-label">2. Detection result</div>', unsafe_allow_html=True)
prob_fake = result["prob_fake"]
label = "DEEPFAKE" if prob_fake > 0.5 else "REAL"
verdict_class = "verdict-fake" if label == "DEEPFAKE" else "verdict-real"

verdict_col, gauge_col = st.columns([1, 1.15])
with verdict_col:
    st.markdown(
        f'<div class="{verdict_class}"><div class="metric-label">Model verdict</div><div class="verdict-title">{label}</div><div class="verdict-subtitle">Based on {len(result["segments"])} three-second analysis window(s)</div></div>',
        unsafe_allow_html=True,
    )
    st.pyplot(plot_confidence(prob_fake, 1.0 - prob_fake), use_container_width=True)
with gauge_col:
    st.pyplot(plot_confidence_gauge(prob_fake), use_container_width=True)

metric_cols = st.columns(4)
metrics = [
    ("Deepfake confidence", f"{prob_fake * 100:.1f}%"),
    ("Real confidence", f"{(1 - prob_fake) * 100:.1f}%"),
    ("Audio duration", f"{len(result['audio']) / SAMPLE_RATE:.2f} s"),
    ("Window count", str(len(result["segments"]))),
]
for col, (name, value) in zip(metric_cols, metrics):
    with col:
        st.markdown(f'<div class="metric-card"><div class="metric-label">{name}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

show_explanation(prob_fake)

# Segment-level detail
st.markdown('<div class="section-label">3. Segment-level confidence</div>', unsafe_allow_html=True)
segment_probs = result["segment_probs"]
segment_rows = []
for index, probability in enumerate(segment_probs):
    segment_rows.append(
        {
            "Window": f"{index + 1}  ({index * 3:.0f}s–{min((index + 1) * 3, len(result['audio']) / SAMPLE_RATE):.1f}s)",
            "Prediction": "DEEPFAKE" if probability > 0.5 else "REAL",
            "Deepfake confidence": f"{probability * 100:.1f}%",
        }
    )
st.dataframe(segment_rows, use_container_width=True, hide_index=True)

# Fingerprints
st.markdown('<div class="section-label">4. Spectro-temporal fingerprint</div>', unsafe_allow_html=True)
st.caption("The first analysis window is shown below. Mel energy, STFT magnitude, and instantaneous-frequency patterns expose different classes of acoustic artifacts.")
fp_tabs = st.tabs(["Mel-spectrogram", "STFT magnitude", "Phase / instantaneous frequency"])
figures = plot_fingerprint(result["fingerprints"][0], SAMPLE_RATE)
for tab, figure in zip(fp_tabs, figures):
    with tab:
        st.pyplot(figure, use_container_width=True)

# Attention
st.markdown('<div class="section-label">5. Artifact attention</div>', unsafe_allow_html=True)
st.caption("Attention weights indicate which fused feature channels contributed most strongly to the screening decision. They are explanatory signals, not independent probabilities.")
st.pyplot(plot_attention_heatmap(result["attention"]), use_container_width=True)

# Footer
st.markdown("---")
st.caption("Research prototype — predictions can be affected by recording quality, compression, language, microphone characteristics, and unseen generators.")
