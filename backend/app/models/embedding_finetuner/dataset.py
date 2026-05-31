"""
PyTorch Datasets for contrastive embedding fine-tuning.

TripletDataset — for TripletLoss training
PairDataset    — for InfoNCE in-batch negative training
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from .data_mining import TrainingTriplet, TrainingPair


class TripletDataset(Dataset):
    """
    Tokenises (anchor, positive, negative) triplets for contrastive training.

    Each item returns three dicts of input_ids / attention_mask tensors.
    Collate with the default DataLoader collate_fn (stacks tensors).
    """

    def __init__(
        self,
        triplets: list[TrainingTriplet],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 256,
    ):
        self.triplets = triplets
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t = self.triplets[idx]
        anchor = self._tokenize(t.anchor)
        positive = self._tokenize(t.positive)
        negative = self._tokenize(t.negative)
        return {
            "anchor_input_ids":      anchor["input_ids"].squeeze(0),
            "anchor_attention_mask": anchor["attention_mask"].squeeze(0),
            "pos_input_ids":         positive["input_ids"].squeeze(0),
            "pos_attention_mask":    positive["attention_mask"].squeeze(0),
            "neg_input_ids":         negative["input_ids"].squeeze(0),
            "neg_attention_mask":    negative["attention_mask"].squeeze(0),
        }

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )


class PairDataset(Dataset):
    """
    Tokenises (text_a, text_b, label) pairs.

    Used for InfoNCE where in-batch negatives are constructed from
    the batch itself (each item's negative = other items in the same batch).
    Label=1.0 marks positive pairs; used to build the similarity target matrix.
    """

    def __init__(
        self,
        pairs: list[TrainingPair],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 256,
        drop_last: bool = True,
    ):
        # Keep only positive pairs; InfoNCE treats all other batch items as negatives
        self.pairs = [p for p in pairs if p.label == 1.0]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.drop_last = drop_last

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p = self.pairs[idx]
        enc_a = self._tokenize(p.text_a)
        enc_b = self._tokenize(p.text_b)
        return {
            "input_ids_a":      enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b":      enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
        }

    def _tokenize(self, text: str) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
