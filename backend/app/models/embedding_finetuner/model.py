"""
Fine-tunable embedding model wrapping all-MiniLM-L6-v2.

Architecture:
  Encoder (all-MiniLM-L6-v2, 6 transformer layers)
    → Mean pool over non-padding tokens
    → Projection head: Linear(384→256) → GELU → Linear(256→384)
    → L2 normalisation

Discriminative fine-tuning: first 3 encoder layers are frozen;
the projection head trains at 5× the encoder learning rate to adapt
faster without distorting the pretrained representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class FineTunableEmbeddingModel(nn.Module):
    """
    Wraps a sentence-transformers encoder with a trainable projection head.

    The projection head creates a dedicated fine-tuning space so the backbone
    is not pulled too far from its pretrained representations.
    """

    BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    PROJECTION_DIM = 256

    def __init__(
        self,
        model_name: str | None = None,
        freeze_first_n_layers: int = 3,
    ):
        super().__init__()
        name = model_name or self.BASE_MODEL
        self.encoder = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)

        # Projection head: linear → GELU → linear (back to embedding dim)
        self.projection = nn.Sequential(
            nn.Linear(self.EMBEDDING_DIM, self.PROJECTION_DIM),
            nn.GELU(),
            nn.Linear(self.PROJECTION_DIM, self.EMBEDDING_DIM),
        )

        self._freeze_encoder_layers(freeze_first_n_layers)

    def _freeze_encoder_layers(self, n: int) -> None:
        """Freeze the first n transformer layers (discriminative fine-tuning)."""
        if n <= 0:
            return
        layers = self.encoder.encoder.layer  # type: ignore[attr-defined]
        for layer in layers[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def _mean_pool(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Average non-padding token embeddings."""
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_projection: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass returning normalised embeddings.

        Args:
            use_projection: False → return raw mean-pooled embeddings
                            (used for before/after comparison in evaluation)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)

        if use_projection:
            pooled = self.projection(pooled)

        return F.normalize(pooled, dim=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode(input_ids, attention_mask, use_projection=True)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int = 64,
        device: str | None = None,
    ) -> torch.Tensor:
        """Utility: encode a list of raw strings → normalised embedding tensor."""
        if device is None:
            device = next(self.parameters()).device.type

        all_embeddings: list[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch,
                    max_length=256,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)
                emb = self.encode(ids, mask, use_projection=True)
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    def parameter_groups(self, encoder_lr: float) -> list[dict]:
        """
        Discriminative learning rate groups.

        Projection head trains at 5× the encoder LR so it adapts the embedding
        space faster without pulling the pretrained backbone off its optimum.
        """
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        proj_params = list(self.projection.parameters())
        return [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": proj_params,   "lr": encoder_lr * 5},
        ]
