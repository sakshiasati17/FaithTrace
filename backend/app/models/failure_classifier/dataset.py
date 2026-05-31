import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

LABEL_MAP = {
    "no_failure": 0,
    "retrieval_miss": 1,
    "context_insufficient": 2,
    "hallucination": 3,
    "prompt_weakness": 4,
    "ranking_failure": 5,
}
LABEL_NAMES = list(LABEL_MAP.keys())

# FailureCategory enum (classifier.py) stores UPPERCASE values in the DB.
# Map them to the 6 PyTorch labels above so training data is consistent.
FAILURE_CATEGORY_TO_LABEL = {
    "NO_FAILURE": "no_failure",
    "LOW_RECALL_RETRIEVAL": "retrieval_miss",
    "TABLE_RETRIEVAL_MISS": "retrieval_miss",
    "CHUNKING_BOUNDARY_ERROR": "context_insufficient",
    "IRRELEVANT_CONTEXT_POLLUTION": "context_insufficient",
    "UNSUPPORTED_SYNTHESIS": "hallucination",
    "STALE_ANSWER": "hallucination",
    "WRONG_VERSION": "hallucination",
    "CHART_LAYOUT_BLINDNESS": "ranking_failure",
}


class FailureDataset(Dataset):
    """
    Each sample: query + retrieved context + generated answer + RAGAS scores → failure class.
    Input text is formatted as: query [SEP] context (truncated) [SEP] answer
    """

    def __init__(self, records, tokenizer_name="distilbert-base-uncased", max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.records = records
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        text = (
            f"{record['query']} [SEP] "
            f"{record['retrieved_context'][:300]} [SEP] "
            f"{record['generated_answer']}"
        )

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        ragas_scores = torch.tensor(
            [
                record["faithfulness"],
                record["answer_relevance"],
                record["context_recall"],
                record["context_precision"],
                record["correctness"],
            ],
            dtype=torch.float32,
        )

        raw = record["failure_type"]
        # Normalise UPPERCASE DB values → lowercase PyTorch labels
        normalized = FAILURE_CATEGORY_TO_LABEL.get(raw, raw.lower() if raw else "no_failure")
        label = LABEL_MAP.get(normalized, 0)  # default to no_failure (0) if unknown

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ragas_scores": ragas_scores,
            "label": torch.tensor(label, dtype=torch.long),
        }
