"""
Training loop for the PyTorch DistilBERT failure classifier.

Key decisions:
- AdamW + CosineAnnealingLR: standard for transformer fine-tuning.
- Gradient clipping (max_norm=1.0): prevents exploding gradients.
- Early stopping on macro F1: stops before overfitting on small datasets.
- Saves best checkpoint only; final evaluation uses that checkpoint.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from .dataset import FailureDataset, LABEL_NAMES
from .losses import FocalLoss
from .model import FailureClassifier

logger = logging.getLogger(__name__)


def compute_class_weights(records, num_classes: int = 6) -> list:
    from .dataset import LABEL_MAP
    counts = [0] * num_classes
    for r in records:
        counts[LABEL_MAP[r["failure_type"]]] += 1
    total = sum(counts)
    return [total / (num_classes * c) if c > 0 else 1.0 for c in counts]


def train(
    records: list,
    output_dir: str = "checkpoints/failure_classifier",
    num_epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    weight_decay: float = 0.01,
    warmup_epochs: int = 3,
    patience: int = 7,
) -> FailureClassifier:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on: %s", device)

    dataset = FailureDataset(records)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FailureClassifier(num_classes=6).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %d / %d (%.1f%%)", trainable, total, 100 * trainable / total)

    class_weights = compute_class_weights(records)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)

    best_val_f1 = 0.0
    patience_counter = 0
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ragas_scores = batch["ragas_scores"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, ragas_scores)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(logits.argmax(dim=1).cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        if epoch >= warmup_epochs:
            scheduler.step()

        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []

        with torch.no_grad():
            for batch in val_loader:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["ragas_scores"].to(device),
                )
                loss = criterion(logits, batch["label"].to(device))
                val_loss += loss.item()
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_labels.extend(batch["label"].cpu().numpy())

        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f",
            epoch + 1, num_epochs,
            train_loss / len(train_loader), train_f1,
            val_loss / len(val_loader), val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1": val_f1,
                },
                f"{output_dir}/best_model.pt",
            )
            logger.info("  → Saved best model (F1: %.4f)", val_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("  → Early stopping at epoch %d", epoch + 1)
                break

    # ── Final evaluation ───────────────────────────────────────────────────────
    checkpoint = torch.load(f"{output_dir}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["ragas_scores"].to(device),
            )
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, zero_division=0)
    logger.info("\nFinal Classification Report:\n%s", report)

    return model
