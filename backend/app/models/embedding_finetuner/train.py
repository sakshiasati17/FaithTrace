"""
Training loop for domain-adaptive embedding fine-tuning.

Key design choices:
  - AdamW with weight_decay=0.01 (standard for transformer fine-tuning)
  - CosineAnnealingWarmRestarts (T_0=5, T_mult=2) — periodic LR resets
    help escape shallow local minima in contrastive loss landscapes
  - Discriminative LRs: projection head at 5× encoder LR
  - Early stopping on validation loss (patience=5)
  - Alignment & uniformity metrics logged per epoch for embedding quality analysis
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split

from .dataset import TripletDataset
from .losses import CombinedContrastiveLoss
from .model import FineTunableEmbeddingModel

logger = logging.getLogger(__name__)


def _alignment(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean cosine distance between positive pairs (lower = better alignment)."""
    return F.pairwise_distance(
        F.normalize(a, dim=-1),
        F.normalize(b, dim=-1),
        p=2,
    ).mean().item()


def _uniformity(embeddings: torch.Tensor, t: float = 2.0) -> float:
    """
    Uniformity loss: log of mean pairwise Gaussian kernel (Wang & Isola 2020).
    Lower (more negative) = better coverage of the hypersphere.
    """
    e = F.normalize(embeddings, dim=-1)
    sq_dists = torch.pdist(e, p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log().item()


def train(
    triplets_or_records: list[Any],
    output_dir: str = "checkpoints/embedding_finetuner",
    num_epochs: int = 20,
    batch_size: int = 32,
    encoder_lr: float = 2e-5,
    val_split: float = 0.1,
    patience: int = 5,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Fine-tune the embedding model on triplets mined from FaithTrace eval history.

    Args:
        triplets_or_records: list of TrainingTriplet objects (from EvalDataMiner)
        output_dir:          where to save best_model.pt and training_log.json
        num_epochs:          maximum training epochs
        batch_size:          DataLoader batch size (drop_last=True for stable InfoNCE)
        encoder_lr:          base learning rate for encoder parameters;
                             projection head uses 5× this
        val_split:           fraction of triplets held out for validation
        patience:            early stopping — stop after this many epochs
                             without validation loss improvement
        device:              "cuda", "cpu", or None (auto-detect)

    Returns:
        dict with training summary: epochs trained, best_val_loss, alignment,
        uniformity, training_time_s
    """
    from .data_mining import TrainingTriplet

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Accept TrainingTriplet list or raw dict list
    if triplets_or_records and isinstance(triplets_or_records[0], dict):
        triplets = [
            TrainingTriplet(
                anchor=r["anchor"],
                positive=r["positive"],
                negative=r["negative"],
                query_id=r.get("query_id", ""),
            )
            for r in triplets_or_records
        ]
    else:
        triplets = triplets_or_records

    if not triplets:
        return {"error": "No training triplets provided"}

    # Split into train / val
    n_val = max(1, int(len(triplets) * val_split))
    n_train = len(triplets) - n_val
    full_ds = TripletDataset(triplets)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = FineTunableEmbeddingModel().to(device)
    criterion = CombinedContrastiveLoss()
    optimizer = AdamW(model.parameter_groups(encoder_lr), weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, num_epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_steps = 0
        for batch in train_loader:
            anchor = model(
                batch["anchor_input_ids"].to(device),
                batch["anchor_attention_mask"].to(device),
            )
            positive = model(
                batch["pos_input_ids"].to(device),
                batch["pos_attention_mask"].to(device),
            )
            negative = model(
                batch["neg_input_ids"].to(device),
                batch["neg_attention_mask"].to(device),
            )

            loss, breakdown = criterion(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += breakdown["total_loss"]
            train_steps += 1

        scheduler.step(epoch)

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_anchors, all_positives = [], []

        with torch.no_grad():
            for batch in val_loader:
                anchor = model(
                    batch["anchor_input_ids"].to(device),
                    batch["anchor_attention_mask"].to(device),
                )
                positive = model(
                    batch["pos_input_ids"].to(device),
                    batch["pos_attention_mask"].to(device),
                )
                negative = model(
                    batch["neg_input_ids"].to(device),
                    batch["neg_attention_mask"].to(device),
                )

                loss, _ = criterion(anchor, positive, negative)
                val_loss += loss.item()
                val_steps += 1
                all_anchors.append(anchor.cpu())
                all_positives.append(positive.cpu())

        avg_train = train_loss / max(train_steps, 1)
        avg_val = val_loss / max(val_steps, 1)

        # Embedding quality metrics on validation set
        anchors_cat = torch.cat(all_anchors)
        positives_cat = torch.cat(all_positives)
        align = _alignment(anchors_cat, positives_cat)
        uniform = _uniformity(torch.cat([anchors_cat, positives_cat]))

        row = {
            "epoch": epoch,
            "train_loss": round(avg_train, 6),
            "val_loss": round(avg_val, 6),
            "alignment": round(align, 6),
            "uniformity": round(uniform, 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        logger.info(
            "Epoch %d/%d | train=%.4f | val=%.4f | align=%.4f | uniform=%.4f",
            epoch, num_epochs, avg_train, avg_val, align, uniform,
        )

        # Early stopping & checkpointing
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": best_val_loss,
                    "alignment": align,
                    "uniformity": uniform,
                },
                Path(output_dir) / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Save training log
    with open(Path(output_dir) / "training_log.json", "w") as f:
        json.dump(history, f, indent=2)

    elapsed = time.time() - start
    best = min(history, key=lambda r: r["val_loss"])
    return {
        "epochs_trained": len(history),
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_alignment": best["alignment"],
        "best_uniformity": best["uniformity"],
        "training_time_s": round(elapsed, 1),
        "checkpoint": str(Path(output_dir) / "best_model.pt"),
        "samples": len(triplets),
    }
