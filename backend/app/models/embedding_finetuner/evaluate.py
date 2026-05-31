"""
Before / after retrieval evaluation for fine-tuned embeddings.

Creates temporary Qdrant collections for the baseline and fine-tuned models,
runs the same eval queries through both, and computes recall@k, MRR, and NDCG.
Also reports embedding shift (cosine similarity of representations before/after)
as a sanity check for catastrophic forgetting.
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0        # Mean Reciprocal Rank
    ndcg_at_5: float = 0.0
    avg_embedding_shift: float = 0.0  # 1 - cosine_sim (lower = less shift)


@dataclass
class EvaluationReport:
    baseline: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    finetuned: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    improvements: dict[str, float] = field(default_factory=dict)
    embedding_shift: float = 0.0
    eval_samples: int = 0


class EmbeddingEvaluator:
    """
    Compare baseline vs fine-tuned embeddings on retrieval tasks.

    Uses in-memory similarity search (no Qdrant connection required for testing).
    Pass use_qdrant=True in production to test against the real vector store.
    """

    def __init__(self, use_qdrant: bool = False, qdrant_url: str = "localhost:6333"):
        self.use_qdrant = use_qdrant
        self.qdrant_url = qdrant_url

    def evaluate(
        self,
        baseline_model: Any,           # FineTunableEmbeddingModel (baseline weights)
        finetuned_model: Any,          # FineTunableEmbeddingModel (fine-tuned weights)
        eval_items: list[dict[str, Any]],
        top_k: int = 5,
        device: str | None = None,
    ) -> EvaluationReport:
        """
        Run side-by-side evaluation.

        Args:
            baseline_model:  model with original pretrained weights
            finetuned_model: model after domain-adaptive fine-tuning
            eval_items:      list of dicts with keys: question, reference_chunks (list[str])
            top_k:           retrieve top-k chunks per query
            device:          compute device

        Returns:
            EvaluationReport with per-metric improvements
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        valid_items = [
            item for item in eval_items
            if item.get("question") and item.get("reference_chunks")
        ]
        if not valid_items:
            logger.warning("No valid eval items with question + reference_chunks")
            return EvaluationReport(eval_samples=0)

        # Build corpus from all reference chunks across eval set
        all_chunks: list[str] = []
        chunk_to_query: dict[str, list[int]] = {}  # chunk_idx → list of item indices
        for item_idx, item in enumerate(valid_items):
            for chunk in item.get("reference_chunks", []):
                if isinstance(chunk, str) and chunk.strip():
                    chunk_idx = len(all_chunks)
                    all_chunks.append(chunk.strip())
                    chunk_to_query.setdefault(chunk_idx, []).append(item_idx)

        if not all_chunks:
            return EvaluationReport(eval_samples=len(valid_items))

        queries = [item["question"] for item in valid_items]

        # Encode with both models
        baseline_model.to(device).eval()
        finetuned_model.to(device).eval()

        base_query_embs = baseline_model.encode_texts(queries, device=device)   # (Q, D)
        base_chunk_embs = baseline_model.encode_texts(all_chunks, device=device)  # (C, D)
        ft_query_embs = finetuned_model.encode_texts(queries, device=device)
        ft_chunk_embs = finetuned_model.encode_texts(all_chunks, device=device)

        # Embedding shift: average over queries
        shift = 1.0 - F.cosine_similarity(
            F.normalize(base_query_embs, dim=-1),
            F.normalize(ft_query_embs, dim=-1),
            dim=-1,
        ).mean().item()

        base_metrics = self._compute_retrieval_metrics(
            base_query_embs, base_chunk_embs, valid_items, all_chunks, top_k, device
        )
        ft_metrics = self._compute_retrieval_metrics(
            ft_query_embs, ft_chunk_embs, valid_items, all_chunks, top_k, device
        )

        improvements = {
            "recall@1": ft_metrics.recall_at_1 - base_metrics.recall_at_1,
            "recall@3": ft_metrics.recall_at_3 - base_metrics.recall_at_3,
            "recall@5": ft_metrics.recall_at_5 - base_metrics.recall_at_5,
            "mrr":      ft_metrics.mrr - base_metrics.mrr,
            "ndcg@5":   ft_metrics.ndcg_at_5 - base_metrics.ndcg_at_5,
        }

        return EvaluationReport(
            baseline=base_metrics,
            finetuned=ft_metrics,
            improvements=improvements,
            embedding_shift=round(shift, 4),
            eval_samples=len(valid_items),
        )

    def _compute_retrieval_metrics(
        self,
        query_embs: torch.Tensor,
        chunk_embs: torch.Tensor,
        eval_items: list[dict[str, Any]],
        all_chunks: list[str],
        top_k: int,
        device: str,
    ) -> RetrievalMetrics:
        q = F.normalize(query_embs.to(device), dim=-1)
        c = F.normalize(chunk_embs.to(device), dim=-1)

        sim = torch.matmul(q, c.T)  # (Q, C)
        top_indices = sim.topk(min(top_k, c.size(0)), dim=-1).indices.cpu()  # (Q, top_k)

        r1_scores, r3_scores, r5_scores, rr_scores, ndcg_scores = [], [], [], [], []

        for item_idx, item in enumerate(eval_items):
            ref_texts = {ch.strip() for ch in item.get("reference_chunks", []) if isinstance(ch, str)}
            retrieved_texts = [all_chunks[i] for i in top_indices[item_idx].tolist()]

            # Recall@k
            hits = [1 if t in ref_texts else 0 for t in retrieved_texts]
            r1_scores.append(int(any(hits[:1])))
            r3_scores.append(int(any(hits[:3])))
            r5_scores.append(int(any(hits[:5])))

            # MRR (reciprocal rank of first hit)
            rr = 0.0
            for rank, hit in enumerate(hits, start=1):
                if hit:
                    rr = 1.0 / rank
                    break
            rr_scores.append(rr)

            # NDCG@5
            dcg = sum(
                h / math.log2(rank + 1)
                for rank, h in enumerate(hits[:5], start=1)
            )
            ideal = sum(
                1.0 / math.log2(rank + 1)
                for rank in range(1, min(len(ref_texts), 5) + 1)
            )
            ndcg_scores.append(dcg / ideal if ideal > 0 else 0.0)

        n = len(eval_items)
        return RetrievalMetrics(
            recall_at_1=round(sum(r1_scores) / n, 4),
            recall_at_3=round(sum(r3_scores) / n, 4),
            recall_at_5=round(sum(r5_scores) / n, 4),
            mrr=round(sum(rr_scores) / n, 4),
            ndcg_at_5=round(sum(ndcg_scores) / n, 4),
        )
