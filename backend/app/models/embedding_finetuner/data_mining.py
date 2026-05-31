"""
Hard negative mining from FaithTrace evaluation history.

Positive pairs: (query, chunk) where chunk was retrieved and the run scored
high context_recall — the model already "got it right."

Hard negatives: (query, chunk) where chunk was retrieved (so the model
thought it was relevant) but context_precision was low — the model's
actual retrieval mistakes, the most informative training signal.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class TrainingTriplet:
    anchor: str       # query text
    positive: str     # relevant chunk text
    negative: str     # hard negative chunk text
    query_id: str = ""
    source_run_id: str = ""


@dataclass
class TrainingPair:
    text_a: str
    text_b: str
    label: float      # 1.0 = similar, 0.0 = dissimilar
    query_id: str = ""


class EvalDataMiner:
    """
    Mines training signal from FaithTrace QueryResult rows.

    Design:
      - Positive: high-recall run → first retrieved chunk is a true positive
      - Hard negative: low-precision run → retrieved but irrelevant chunk
        (semi-hard: not the absolute worst, but wrong enough to be instructive)
      - Cross-query negatives for InfoNCE in-batch negatives (automatic)
    """

    RECALL_THRESHOLD = 0.70     # run must have context_recall >= this for positives
    PRECISION_THRESHOLD = 0.35  # run must have context_precision < this for hard negatives

    def __init__(
        self,
        min_chunk_length: int = 30,
        max_chunk_length: int = 512,
        hard_negative_ratio: float = 0.6,
        random_seed: int = 42,
    ):
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        self.hard_negative_ratio = hard_negative_ratio
        random.seed(random_seed)

    def mine_triplets(
        self,
        query_results: list[dict[str, Any]],
        eval_items: list[dict[str, Any]],
        ragas_scores: list[dict[str, float]],
    ) -> list[TrainingTriplet]:
        """
        Build (anchor, positive, negative) triplets from evaluation history.

        Args:
            query_results: list of QueryResult-like dicts with keys
                           query_id, question, retrieved_chunks, run_id
            eval_items:    ground-truth eval set items (query_id → reference_chunks)
            ragas_scores:  per-query Ragas metric dicts (indexed same as query_results)

        Returns:
            List of TrainingTriplet instances ready for TripletDataset
        """
        eval_by_id = {item.get("query_id", ""): item for item in eval_items}
        triplets: list[TrainingTriplet] = []

        # Separate high-recall and low-precision results
        high_recall, low_precision = [], []
        for qr, scores in zip(query_results, ragas_scores):
            recall = scores.get("context_recall", 0.0)
            precision = scores.get("context_precision", 1.0)
            if recall >= self.RECALL_THRESHOLD:
                high_recall.append((qr, scores))
            if precision < self.PRECISION_THRESHOLD and qr.get("retrieved_chunks"):
                low_precision.append((qr, scores))

        if not high_recall or not low_precision:
            return triplets

        # Index hard negatives by query_id so we avoid same-query negatives
        hard_neg_pool: dict[str, list[str]] = {}
        for qr, _ in low_precision:
            qid = qr.get("query_id", "")
            chunks = self._extract_chunks(qr.get("retrieved_chunks", []))
            if chunks:
                hard_neg_pool.setdefault(qid, []).extend(chunks)

        all_hard_neg_chunks: list[str] = [
            c for chunks in hard_neg_pool.values() for c in chunks
        ]
        if not all_hard_neg_chunks:
            return triplets

        for qr, scores in high_recall:
            question = qr.get("question", "").strip()
            if not question:
                continue

            # Positive: use reference chunk from eval set if available,
            # else fall back to first retrieved chunk (it was high-recall)
            eval_item = eval_by_id.get(qr.get("query_id", ""), {})
            reference_chunks = eval_item.get("reference_chunks", [])
            positive = self._pick_positive(reference_chunks, qr.get("retrieved_chunks", []))
            if not positive:
                continue

            # Negative: prefer from different query to avoid trivial overlap
            qid = qr.get("query_id", "")
            cross_query_negs = [
                c for cid, chunks in hard_neg_pool.items()
                if cid != qid for c in chunks
            ]
            neg_pool = cross_query_negs if cross_query_negs else all_hard_neg_chunks
            negative = random.choice(neg_pool)

            triplets.append(TrainingTriplet(
                anchor=question,
                positive=positive,
                negative=negative,
                query_id=qid,
                source_run_id=qr.get("run_id", ""),
            ))

        random.shuffle(triplets)
        return triplets

    def mine_pairs(
        self,
        query_results: list[dict[str, Any]],
        eval_items: list[dict[str, Any]],
        ragas_scores: list[dict[str, float]],
    ) -> list[TrainingPair]:
        """Build (text_a, text_b, label) pairs for PairDataset / InfoNCE."""
        triplets = self.mine_triplets(query_results, eval_items, ragas_scores)
        pairs: list[TrainingPair] = []
        for t in triplets:
            pairs.append(TrainingPair(
                text_a=t.anchor, text_b=t.positive, label=1.0, query_id=t.query_id
            ))
            pairs.append(TrainingPair(
                text_a=t.anchor, text_b=t.negative, label=0.0, query_id=t.query_id
            ))
        return pairs

    # ── helpers ───────────────────────────────────────────────────────────────

    def _extract_chunks(self, retrieved_chunks: list[Any]) -> list[str]:
        """Normalize retrieved chunk format → list of text strings."""
        texts: list[str] = []
        for chunk in retrieved_chunks:
            if isinstance(chunk, str):
                text = chunk
            elif isinstance(chunk, dict):
                text = chunk.get("text") or chunk.get("content") or chunk.get("page_content", "")
            else:
                text = str(chunk)
            text = text.strip()
            if self.min_chunk_length <= len(text) <= self.max_chunk_length:
                texts.append(text)
        return texts

    def _pick_positive(
        self,
        reference_chunks: list[Any],
        retrieved_chunks: list[Any],
    ) -> str | None:
        """Return the best positive text from reference or retrieved chunks."""
        ref_texts = self._extract_chunks(reference_chunks)
        if ref_texts:
            return ref_texts[0]
        ret_texts = self._extract_chunks(retrieved_chunks)
        return ret_texts[0] if ret_texts else None
