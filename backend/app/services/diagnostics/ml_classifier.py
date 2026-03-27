"""
XGBoost-based failure classifier.

Trains a multi-class XGBoost model to predict RAG failure categories from
per-query retrieval features. Falls back to heuristic classifier when the
model hasn't been trained yet or if xgboost is not installed.

Feature vector (14 features):
    0  faithfulness          (0-1)
    1  context_recall        (0-1)
    2  context_precision     (0-1)
    3  answer_correctness    (0-1)
    4  answer_relevance      (0-1)
    5  latency_ms            (normalized)
    6  cost_usd              (normalized)
    7  num_chunks_retrieved  (int)
    8  has_table_chunk       (0/1)
    9  has_image_chunk       (0/1)
    10 has_vision_chunk      (0/1)
    11 modality_text         (0/1)
    12 modality_table        (0/1)
    13 modality_chart        (0/1)
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from app.services.diagnostics.classifier import FailureCategory

logger = logging.getLogger(__name__)

# Path where the trained model is persisted
_MODEL_PATH = Path(os.getenv("ML_CLASSIFIER_PATH", "/tmp/faithtrace_xgb_classifier.pkl"))

# Ordered list of labels the model is trained on
LABEL_CLASSES: list[str] = [
    FailureCategory.NO_FAILURE,
    FailureCategory.STALE_ANSWER,
    FailureCategory.WRONG_VERSION,
    FailureCategory.TABLE_RETRIEVAL_MISS,
    FailureCategory.CHART_LAYOUT_BLINDNESS,
    FailureCategory.CHUNKING_BOUNDARY_ERROR,
    FailureCategory.LOW_RECALL_RETRIEVAL,
    FailureCategory.IRRELEVANT_CONTEXT_POLLUTION,
    FailureCategory.UNSUPPORTED_SYNTHESIS,
]


# ─── Feature extraction ───────────────────────────────────────────────────────

def extract_features(
    metrics: dict,
    retrieved_chunks: list[dict],
    eval_item: dict,
) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a single query result.

    Args:
        metrics: Per-query metric dict (faithfulness, context_recall, etc.)
        retrieved_chunks: List of chunk dicts from the pipeline result
        eval_item: Ground-truth eval item (provides modality)

    Returns:
        1-D numpy array of shape (14,)
    """
    faithfulness      = float(metrics.get("faithfulness", 0.0) or 0.0)
    context_recall    = float(metrics.get("context_recall", 0.0) or 0.0)
    context_precision = float(metrics.get("context_precision", 0.0) or 0.0)
    answer_correctness = float(metrics.get("answer_correctness", 0.0) or 0.0)
    answer_relevance  = float(metrics.get("answer_relevance", 0.0) or 0.0)

    # Normalize latency to 0-1 (cap at 10 s) and cost (cap at $0.10)
    latency_ms = float(metrics.get("latency_ms", 0.0) or 0.0)
    cost_usd   = float(metrics.get("cost_usd", 0.0) or 0.0)
    latency_norm = min(latency_ms / 10_000.0, 1.0)
    cost_norm    = min(cost_usd / 0.10, 1.0)

    chunk_types = [c.get("chunk_type", "text") for c in retrieved_chunks]
    has_table   = float(any(ct in ("table", "spreadsheet_cell") for ct in chunk_types))
    has_image   = float(any(ct == "image" for ct in chunk_types))
    has_vision  = float(
        any(c.get("metadata", {}).get("source") == "gpt4o_vision" for c in retrieved_chunks)
    )
    num_chunks  = float(len(retrieved_chunks))

    modality = (eval_item.get("modality") or "text").lower()
    mod_text  = float(modality == "text")
    mod_table = float(modality in ("table", "spreadsheet", "mixed"))
    mod_chart = float(modality == "chart")

    return np.array([
        faithfulness, context_recall, context_precision,
        answer_correctness, answer_relevance,
        latency_norm, cost_norm, num_chunks,
        has_table, has_image, has_vision,
        mod_text, mod_table, mod_chart,
    ], dtype=np.float32)


def build_training_data(
    all_metrics: list[dict],
    all_chunks: list[list[dict]],
    all_eval_items: list[dict],
    all_labels: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build X (feature matrix), y (contiguous integer label array), and the
    active label list (only classes present in this dataset).

    Returns:
        X:             shape (n_samples, 14)
        y:             shape (n_samples,) — contiguous 0..n_classes-1
        active_labels: list of class names in y-index order
    """
    # Only include classes that actually appear in the dataset
    seen = sorted(set(all_labels), key=lambda l: LABEL_CLASSES.index(l)
                  if l in LABEL_CLASSES else len(LABEL_CLASSES))
    label_index = {lbl: i for i, lbl in enumerate(seen)}

    X_rows, y_rows = [], []
    for metrics, chunks, eval_item, label in zip(
        all_metrics, all_chunks, all_eval_items, all_labels
    ):
        X_rows.append(extract_features(metrics, chunks, eval_item))
        y_rows.append(label_index.get(label, 0))

    return np.vstack(X_rows), np.array(y_rows, dtype=np.int32), seen


# ─── Training ─────────────────────────────────────────────────────────────────

def train(
    all_metrics: list[dict],
    all_chunks: list[list[dict]],
    all_eval_items: list[dict],
    all_labels: list[str],
) -> dict:
    """
    Train an XGBoost multi-class classifier and persist it to disk.

    Args:
        all_metrics:    List of per-query metric dicts
        all_chunks:     List of per-query retrieved_chunks lists
        all_eval_items: List of per-query eval set items
        all_labels:     List of ground-truth FailureCategory strings

    Returns:
        Training summary dict with accuracy and sample counts.
    """
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
    except ImportError as exc:
        raise RuntimeError(
            "xgboost and scikit-learn are required for ML training. "
            "Run: pip install xgboost scikit-learn"
        ) from exc

    if len(all_labels) < 10:
        raise ValueError(
            f"Need at least 10 labeled samples to train; got {len(all_labels)}. "
            "Run more experiments or add failure_type labels to your eval set."
        )

    X, y, active_labels = build_training_data(all_metrics, all_chunks, all_eval_items, all_labels)

    # Use stratified split only if we have enough samples per class
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    use_stratify = min_count >= 2 and len(X) >= 20

    if use_stratify:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))

    # Feature importance
    importance = dict(zip(
        ["faithfulness", "context_recall", "context_precision",
         "answer_correctness", "answer_relevance",
         "latency_norm", "cost_norm", "num_chunks",
         "has_table", "has_image", "has_vision",
         "mod_text", "mod_table", "mod_chart"],
        model.feature_importances_.tolist(),
    ))

    # Persist — store active_labels so predict() uses the same mapping
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "label_classes": active_labels}, f)

    logger.info("XGBoost classifier trained: acc=%.3f, samples=%d, classes=%d",
                acc, len(X), len(active_labels))

    report = classification_report(
        y_val, y_pred,
        labels=list(range(len(active_labels))),
        target_names=active_labels,
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": acc,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "n_classes": len(unique),
        "feature_importance": importance,
        "per_class_report": report,
        "model_path": str(_MODEL_PATH),
    }


# ─── Inference ────────────────────────────────────────────────────────────────

_cached_model: Optional[dict] = None


def _load_model() -> Optional[dict]:
    """Load the trained model from disk (cached in memory)."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    if not _MODEL_PATH.exists():
        return None
    try:
        with open(_MODEL_PATH, "rb") as f:
            _cached_model = pickle.load(f)
        logger.info("Loaded XGBoost classifier from %s", _MODEL_PATH)
        return _cached_model
    except Exception as exc:
        logger.warning("Failed to load XGBoost classifier: %s", exc)
        return None


def reload_model() -> bool:
    """Force reload of the model from disk (call after training)."""
    global _cached_model
    _cached_model = None
    return _load_model() is not None


def is_trained() -> bool:
    """Return True if a trained model is available."""
    return _MODEL_PATH.exists()


def predict(
    metrics: dict,
    retrieved_chunks: list[dict],
    eval_item: dict,
) -> Optional[tuple[FailureCategory, float]]:
    """
    Predict the failure category for a single query.

    Returns:
        Tuple of (FailureCategory, confidence) if model is available,
        None if no trained model exists (caller should fall back to heuristics).
    """
    bundle = _load_model()
    if bundle is None:
        return None

    try:
        import xgboost  # noqa: F401 — verify still importable
    except ImportError:
        return None

    model = bundle["model"]
    label_classes = bundle.get("label_classes", LABEL_CLASSES)

    features = extract_features(metrics, retrieved_chunks, eval_item).reshape(1, -1)
    pred_idx = int(model.predict(features)[0])
    proba = model.predict_proba(features)[0]
    confidence = float(proba[pred_idx])

    label = label_classes[pred_idx] if pred_idx < len(label_classes) else FailureCategory.NO_FAILURE
    return FailureCategory(label), confidence
