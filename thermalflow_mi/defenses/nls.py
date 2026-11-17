"""Negative label smoothing (Struppek et al., "Be Careful What You Smooth For", ICLR 2024).

Cross-entropy with smoothed targets q = (1 - a) onehot + a / K and a negative factor a. The model is trained with
a = 0 for the first half of the epochs, then a decreases linearly to `alpha` (-0.05) at the last epoch, so it first
learns the task and the negative smoothing then flattens what its outputs reveal.
"""
import torch
import torch.nn.functional as F


class NLSLoss(torch.nn.Module):
    def __init__(self, alpha=-0.05):
        super().__init__()
        self.final_alpha, self.alpha = float(alpha), 0.0

    def forward(self, logits, labels):
        logp = F.log_softmax(logits.float(), dim=1)
        k = logits.shape[1]
        q = F.one_hot(labels, k).float() * (1 - self.alpha) + self.alpha / k
        return -(q * logp).sum(1).mean()

    def schedule(self, epoch, epochs):
        """Call before every epoch (1-based)."""
        half = epochs // 2
        self.alpha = 0.0 if epoch <= half else self.final_alpha * (epoch - half) / (epochs - half)
