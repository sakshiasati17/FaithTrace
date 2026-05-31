"""
Contrastive losses for domain-adaptive embedding fine-tuning.

TripletLoss         — cosine triplet with hard negative weighting
InfoNCELoss         — in-batch negatives (NT-Xent / SimCLR style)
CombinedContrastiveLoss — 60 % triplet + 40 % InfoNCE
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Cosine-distance triplet loss with optional hard-negative upweighting.

    Loss per sample = max(0, d(a, p) - d(a, n) + margin)
    where d(x, y) = 1 - cosine_similarity(x, y)

    Hard negatives (retrieved but irrelevant chunks) receive 2× weight
    so the model learns harder from its own retrieval mistakes.
    """

    def __init__(
        self,
        margin: float = 0.3,
        hard_negative_weight: float = 2.0,
    ):
        super().__init__()
        self.margin = margin
        self.hard_negative_weight = hard_negative_weight

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        is_hard_negative: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            anchor:           (B, D) normalised embeddings
            positive:         (B, D) normalised embeddings
            negative:         (B, D) normalised embeddings
            is_hard_negative: (B,) bool tensor; hard negatives get 2× weight
        """
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negative = F.normalize(negative, dim=-1)

        dist_pos = 1.0 - (anchor * positive).sum(dim=-1)  # (B,)
        dist_neg = 1.0 - (anchor * negative).sum(dim=-1)  # (B,)

        losses = F.relu(dist_pos - dist_neg + self.margin)  # (B,)

        if is_hard_negative is not None:
            weights = torch.where(
                is_hard_negative,
                torch.full_like(losses, self.hard_negative_weight),
                torch.ones_like(losses),
            )
            losses = losses * weights

        return losses.mean()


class InfoNCELoss(nn.Module):
    """
    In-batch negative contrastive loss (NT-Xent / SimCLR).

    For a batch of B (query, chunk) positive pairs, treats the other
    B-1 queries' chunks as negatives. The loss is a B-way classification
    problem: which chunk in the batch belongs to this query?

    Temperature τ=0.07 produces sharp distributions that force the model
    to cleanly separate even similar queries.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings_a: (B, D) query embeddings
            embeddings_b: (B, D) chunk embeddings (positive pairs with a)
        """
        a = F.normalize(embeddings_a, dim=-1)
        b = F.normalize(embeddings_b, dim=-1)

        # (B, B) similarity matrix; diagonal = positive pairs
        logits = torch.matmul(a, b.T) / self.temperature  # (B, B)
        labels = torch.arange(len(a), device=a.device)

        # Symmetric: query→chunk loss + chunk→query loss
        loss_ab = F.cross_entropy(logits, labels)
        loss_ba = F.cross_entropy(logits.T, labels)
        return (loss_ab + loss_ba) / 2.0


class CombinedContrastiveLoss(nn.Module):
    """
    60 % TripletLoss + 40 % InfoNCE.

    Triplet loss trains on curated hard negatives; InfoNCE exploits the full
    batch as additional in-batch negatives, regularising the embedding space.
    """

    def __init__(
        self,
        triplet_margin: float = 0.3,
        hard_negative_weight: float = 2.0,
        infonce_temperature: float = 0.07,
        triplet_weight: float = 0.6,
    ):
        super().__init__()
        self.triplet = TripletLoss(
            margin=triplet_margin,
            hard_negative_weight=hard_negative_weight,
        )
        self.infonce = InfoNCELoss(temperature=infonce_temperature)
        self.triplet_weight = triplet_weight
        self.infonce_weight = 1.0 - triplet_weight

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        is_hard_negative: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Returns:
            total_loss: scalar tensor
            breakdown:  dict with individual loss values for logging
        """
        triplet_loss = self.triplet(anchor, positive, negative, is_hard_negative)
        infonce_loss = self.infonce(anchor, positive)

        total = self.triplet_weight * triplet_loss + self.infonce_weight * infonce_loss
        breakdown = {
            "triplet_loss": triplet_loss.item(),
            "infonce_loss": infonce_loss.item(),
            "total_loss": total.item(),
        }
        return total, breakdown
