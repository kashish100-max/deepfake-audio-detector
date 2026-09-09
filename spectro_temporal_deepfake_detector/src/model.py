"""
Dual-Branch Deep Architecture for Spectro-Temporal Deepfake Detection.

Branch 1 — Spectral CNN:       2D convolutions over Mel/STFT frequency patterns.
Branch 2 — Temporal Conv/GRU:  1D temporal convolutions + bidirectional GRU.
Fusion — Channel Attention:    Squeeze-and-excitation style channel attention.
Classifier — Linear + Dropout + Sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model configuration — exported for use by train.py / inference.py / app.py
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "num_channels": 3,        # Mel, STFT-mag, Instantaneous-Frequency
    "num_mels": 80,            # frequency bins
    "num_time_frames": 188,    # time frames in a 3-second segment (48000/256 - 1 + pad)
    "spectral_channels": [32, 64, 128, 128],
    "temporal_channels": [64, 128, 128],
    "gru_hidden": 128,
    "gru_layers": 2,
    "attention_reduction": 16,
    "fusion_dim": 256,
    "dropout": 0.3,
    "num_classes": 1,           # binary: 0 = real, 1 = deepfake
}


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class ConvBlock2d(nn.Module):
    """Conv2d → BatchNorm → ReLU → (optional) MaxPool2d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        pool: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d((2, 2)))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBlock1d(nn.Module):
    """Conv1d → BatchNorm → ReLU → (optional) MaxPool1d."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        pool: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool1d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation channel attention.

    Learns per-channel weights from a global summary of the fused feature
    map, amplifying channels that carry the most discriminative
    spectro-temporal artifacts.
    """

    def __init__(self, num_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(num_channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.gap(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w.expand_as(x)


# ---------------------------------------------------------------------------
# Dual-branch model
# ---------------------------------------------------------------------------
class SpectroTemporalDetector(nn.Module):
    """Dual-branch spectro-temporal deepfake detector.

    Parameters
    ----------
    config : dict
        Configuration dictionary (see ``MODEL_CONFIG``).
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = {**MODEL_CONFIG, **(config or {})}
        self.cfg = cfg

        num_in_ch = cfg["num_channels"]
        spec_chs = cfg["spectral_channels"]
        temp_chs = cfg["temporal_channels"]
        gru_hidden = cfg["gru_hidden"]
        gru_layers = cfg["gru_layers"]
        attn_reduction = cfg["attention_reduction"]
        fusion_dim = cfg["fusion_dim"]
        dropout = cfg["dropout"]
        num_classes = cfg["num_classes"]

        # ---- Branch 1: Spectral CNN (operates on 2D fingerprint) ----
        spectral_layers: list[nn.Module] = []
        in_ch = num_in_ch
        for out_ch in spec_chs:
            spectral_layers.append(ConvBlock2d(in_ch, out_ch))
            in_ch = out_ch
        self.spectral_cnn = nn.Sequential(*spectral_layers)

        # After 4 max-pools: (C, F/16, T/16)
        spec_flat_channels = spec_chs[-1]
        spec_flat_freq = cfg["num_mels"] // (2 ** len(spec_chs))
        spec_flat_time = cfg["num_time_frames"] // (2 ** len(spec_chs))
        spec_flat_dim = spec_flat_channels * max(spec_flat_freq, 1) * max(spec_flat_time, 1)

        # ---- Branch 2: Temporal Conv + BiGRU ----
        temporal_layers: list[nn.Module] = []
        in_ch = num_in_ch
        for i, out_ch in enumerate(temp_chs):
            temporal_layers.append(ConvBlock1d(in_ch, out_ch, pool=(i < len(temp_chs) - 1)))
            in_ch = out_ch
        self.temporal_cnn = nn.Sequential(*temporal_layers)

        temp_out_ch = temp_chs[-1]
        self.bigru = nn.GRU(
            input_size=temp_out_ch,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        temporal_embed_dim = gru_hidden * 2  # bidirectional

        # ---- Fusion + Attention ----
        fused_channels = spec_flat_channels + temporal_embed_dim
        # Reshape spectral features to (B, C_spec, H, W) for attention
        self.spec_reshape_channels = spec_flat_channels
        self.spec_reshape_h = max(spec_flat_freq, 1)
        self.spec_reshape_w = max(spec_flat_time, 1)

        # Project temporal embedding to a 2D-compatible shape for fusion
        self.temporal_proj = nn.Linear(temporal_embed_dim, spec_flat_channels * self.spec_reshape_h * self.spec_reshape_w)

        # After temporal projection, we'll have a (B, C_spec, H, W) temporal map
        # Concatenate along channel dim → (B, 2*C_spec, H, W)
        fused_attn_channels = spec_flat_channels * 2
        self.attention = ChannelAttention(fused_attn_channels, reduction=attn_reduction)

        # Global pooling after attention → (B, fused_attn_channels)
        self.gap_fused = nn.AdaptiveAvgPool2d(1)

        # ---- Classifier head ----
        self.classifier = nn.Sequential(
            nn.Linear(fused_attn_channels, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, num_classes),
        )

    # -------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input fingerprint of shape (B, C, F, T).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).  Apply sigmoid for probability.
        """
        b = x.size(0)

        # --- Branch 1: Spectral CNN ---
        spec_feat = self.spectral_cnn(x)  # (B, C_spec, H, W)
        spec_map = spec_feat  # keep 2D map for attention

        # --- Branch 2: Temporal Conv + BiGRU ---
        # Reshape (B, C, F, T) → (B, C, T) by pooling over frequency
        temp_input = x.mean(dim=2)  # (B, C, T)
        temp_feat = self.temporal_cnn(temp_input)  # (B, C_temp, T')
        temp_feat = temp_feat.permute(0, 2, 1)  # (B, T', C_temp) for GRU
        gru_out, _ = self.bigru(temp_feat)  # (B, T', 2*hidden)
        # Take the mean over time steps as the temporal embedding
        temporal_embed = gru_out.mean(dim=1)  # (B, 2*hidden)

        # --- Fusion ---
        # Project temporal embedding to match spectral map spatial size
        temp_map = self.temporal_proj(temporal_embed)  # (B, C_spec * H * W)
        temp_map = temp_map.view(
            b,
            self.spec_reshape_channels,
            self.spec_reshape_h,
            self.spec_reshape_w,
        )

        # Concatenate along channel dimension
        fused = torch.cat([spec_map, temp_map], dim=1)  # (B, 2*C_spec, H, W)

        # Attention
        attended = self.attention(fused)  # (B, 2*C_spec, H, W)

        # Global average pool → (B, 2*C_spec)
        pooled = self.gap_fused(attended).view(b, -1)

        # Classifier
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits

    # -------------------------------------------------------------------
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))

    # -------------------------------------------------------------------
    @torch.no_grad()
    def extract_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return channel-attention weights for visualization.

        Returns a tensor of shape (B, C_fused).
        """
        b = x.size(0)
        spec_feat = self.spectral_cnn(x)
        spec_map = spec_feat

        temp_input = x.mean(dim=2)
        temp_feat = self.temporal_cnn(temp_input)
        temp_feat = temp_feat.permute(0, 2, 1)
        gru_out, _ = self.bigru(temp_feat)
        temporal_embed = gru_out.mean(dim=1)

        temp_map = self.temporal_proj(temporal_embed)
        temp_map = temp_map.view(
            b,
            self.spec_reshape_channels,
            self.spec_reshape_h,
            self.spec_reshape_w,
        )

        fused = torch.cat([spec_map, temp_map], dim=1)
        weights = self.attention.fc(self.attention.gap(fused).view(b, -1))
        return weights


# ---------------------------------------------------------------------------
# Helper: build model from config
# ---------------------------------------------------------------------------
def build_model(config: dict | None = None, device: str = "cpu") -> SpectroTemporalDetector:
    """Instantiate and move model to *device*."""
    model = SpectroTemporalDetector(config)
    model.to(device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
