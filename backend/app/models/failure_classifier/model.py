import torch
import torch.nn as nn
from transformers import AutoModel


class FailureClassifier(nn.Module):
    """
    DistilBERT encoder + classification head for RAG failure categorisation.

    Architecture decisions:
    - DistilBERT: 40% smaller than BERT, retains 97% performance, faster iteration.
    - Freeze first 4 of 6 layers: prevents catastrophic forgetting on ~1 000 samples.
    - Concatenate RAGAS scores: gives the model numerical signals XGBoost had
      exclusively, alongside semantic text understanding.
    - GELU + LayerNorm: standard stable combination from transformer literature.
    """

    def __init__(
        self,
        num_classes: int = 6,
        encoder_name: str = "distilbert-base-uncased",
        ragas_feature_dim: int = 5,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_layers: int = 4,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        encoder_dim = self.encoder.config.hidden_size  # 768

        # Freeze early layers to reduce overfitting risk on small datasets
        for i, layer in enumerate(self.encoder.transformer.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # 768 (CLS) + 5 (RAGAS) = 773 → 256 → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(encoder_dim + ragas_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, ragas_scores):
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = encoder_output.last_hidden_state[:, 0, :]   # (B, 768)
        combined = torch.cat([cls_embedding, ragas_scores], dim=1)  # (B, 773)
        return self.classifier(combined)                             # (B, 6)
