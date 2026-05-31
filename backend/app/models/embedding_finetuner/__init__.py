"""
Extension 4: Domain-Adaptive Embedding Fine-Tuning

Contrastive learning pipeline that adapts all-MiniLM-L6-v2 to FaithTrace's
enterprise RAG domain using hard negatives mined from evaluation history.
"""

from .model import FineTunableEmbeddingModel
from .losses import TripletLoss, InfoNCELoss, CombinedContrastiveLoss
from .data_mining import EvalDataMiner, TrainingTriplet, TrainingPair

__all__ = [
    "FineTunableEmbeddingModel",
    "TripletLoss",
    "InfoNCELoss",
    "CombinedContrastiveLoss",
    "EvalDataMiner",
    "TrainingTriplet",
    "TrainingPair",
]
