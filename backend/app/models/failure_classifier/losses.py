import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss with label smoothing for imbalanced failure-type classification.

    Why Focal Loss over CrossEntropy:
    - Failure classes are highly skewed (hallucination >> prompt_weakness).
    - Standard CE optimises easily on the majority class and ignores rare ones.
    - Focal Loss down-weights easy/frequent examples via (1 - p_t)^gamma,
      forcing the model to focus on hard, under-represented failure modes.

    Label smoothing (0.1) prevents overconfidence and improves generalisation
    on the small training sets FaithTrace typically produces (~500-1 000 rows).
    """

    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
        num_classes: int = 6,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.alpha = torch.ones(num_classes) if alpha is None else torch.tensor(alpha)

    def forward(self, logits, targets):
        confidence = 1.0 - self.label_smoothing
        smooth_value = self.label_smoothing / (self.num_classes - 1)

        one_hot = torch.zeros_like(logits).scatter(1, targets.unsqueeze(1), 1)
        smooth_targets = one_hot * confidence + (1 - one_hot) * smooth_value

        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        focal_weight = (1 - probs) ** self.gamma
        alpha = self.alpha.to(logits.device)
        alpha_weight = alpha[targets].unsqueeze(1)

        loss = -alpha_weight * focal_weight * smooth_targets * log_probs
        return loss.sum(dim=1).mean()
