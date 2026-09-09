"""
Training script for the Spectro-Temporal Deepfake Detector.

Usage:
    python train.py --epochs 30 --batch_size 16 --lr 1e-3
    python train.py --data_dir ./data --output_dir ./checkpoints
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm

from src.audio_processing import (
    load_audio,
    trim_silence,
    normalize_audio,
    segment_audio,
    extract_fingerprint,
    SAMPLE_RATE,
    SEGMENT_LENGTH,
)
from src.model import build_model, count_parameters, MODEL_CONFIG
from src.utils import plot_training_curves

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AudioDeepfakeDataset(Dataset):
    """Loads audio files from real/ and fake/ subdirectories.

    Each audio file is segmented into 3-second windows; each segment is
    treated as one training sample.  Labels: 0 = real, 1 = fake.
    """

    def __init__(self, data_dir: str, sr: int = SAMPLE_RATE, segment_length: int = SEGMENT_LENGTH):
        self.samples: list[tuple[str, int]] = []  # (file_path, label)
        self.sr = sr
        self.segment_length = segment_length

        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")

        if os.path.isdir(real_dir):
            for fname in sorted(os.listdir(real_dir)):
                if fname.endswith((".wav", ".mp3", ".flac")):
                    self.samples.append((os.path.join(real_dir, fname), 0))
        if os.path.isdir(fake_dir):
            for fname in sorted(os.listdir(fake_dir)):
                if fname.endswith((".wav", ".mp3", ".flac")):
                    self.samples.append((os.path.join(fake_dir, fname), 1))

        if not self.samples:
            raise RuntimeError(
                f"No audio files found in {data_dir}. "
                "Run `python dataset_prep.py` first to generate a dummy dataset."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        audio, _ = load_audio(path, sr=self.sr)
        audio = trim_silence(audio)
        audio = normalize_audio(audio)
        segments = segment_audio(audio, self.segment_length)

        # Pick a random segment for training augmentation
        seg_idx = np.random.randint(len(segments))
        segment = segments[seg_idx]
        fingerprint = extract_fingerprint(segment, self.sr)

        fp_tensor = torch.from_numpy(fingerprint)  # (3, F, T)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return fp_tensor, label_tensor


# ---------------------------------------------------------------------------
# Split dataset
# ---------------------------------------------------------------------------
def split_dataset(
    dataset: AudioDeepfakeDataset, val_ratio: float = 0.2, seed: int = 42
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """Random train/val split preserving label balance approximately."""
    from torch.utils.data import random_split

    n_val = max(1, int(len(dataset) * val_ratio))
    n_train = len(dataset) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [n_train, n_val], generator=gen)
    return train_subset, val_subset


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, float]:
    """Evaluate model on a data loader. Returns (loss, accuracy, roc_auc)."""
    model.eval()
    all_probs: list[float] = []
    all_labels: list[int] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().flatten().astype(int).tolist())

    avg_loss = total_loss / max(n_batches, 1)
    preds = [1 if p > 0.5 else 0 for p in all_probs]
    acc = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return avg_loss, acc, auc


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(
    data_dir: str = "./data",
    output_dir: str = "./checkpoints",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_ratio: float = 0.2,
    device: str | None = None,
) -> None:
    """Full training pipeline with metrics and checkpointing."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Spectro-Temporal Deepfake Detector — Training")
    print("=" * 70)
    print(f"Device:      {device}")
    print(f"Data dir:    {data_dir}")
    print(f"Output dir:  {output_dir}")
    print(f"Epochs:      {epochs}")
    print(f"Batch size:  {batch_size}")
    print(f"LR:          {lr}")
    print("-" * 70)

    # Dataset
    full_dataset = AudioDeepfakeDataset(data_dir)
    train_subset, val_subset = split_dataset(full_dataset, val_ratio)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples:   {len(val_subset)}")
    print("-" * 70)

    # Model
    model = build_model(MODEL_CONFIG, device)
    print(f"Model parameters: {count_parameters(model):,}")
    print("-" * 70)

    # Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    best_val_auc = 0.0
    best_path = os.path.join(output_dir, "best_model.pth")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / len(train_loader)
        train_acc = correct / max(total, 1)

        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"  Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  | "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  AUC: {val_auc:.4f}"
        )

        # Checkpoint best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                    "config": MODEL_CONFIG,
                },
                best_path,
            )
            print(f"  ** Best model saved (AUC={val_auc:.4f})")

        # Always save latest
        latest_path = os.path.join(output_dir, "latest_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "config": MODEL_CONFIG,
            },
            latest_path,
        )

    print("-" * 70)
    print(f"Training complete. Best validation AUC: {best_val_auc:.4f}")
    print(f"Best model saved to: {best_path}")

    # Save training curves
    curves_path = os.path.join(output_dir, "training_curves.png")
    fig = plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    fig.savefig(curves_path, dpi=150)
    print(f"Training curves saved to: {curves_path}")
    plt_close(fig)

    # Also save metrics as numpy
    metrics_path = os.path.join(output_dir, "training_metrics.npz")
    np.savez(
        metrics_path,
        train_losses=train_losses,
        val_losses=val_losses,
        train_accs=train_accs,
        val_accs=val_accs,
    )
    print(f"Metrics saved to: {metrics_path}")


def plt_close(fig):
    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Spectro-Temporal Deepfake Detector")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_ratio=args.val_ratio,
    )


if __name__ == "__main__":
    main()
