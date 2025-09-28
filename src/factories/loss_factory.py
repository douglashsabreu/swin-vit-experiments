from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.config import FullConfig


class LossFactory:
    """Factory to create loss functions configured via the experiment config.

    Supports cross-entropy with label smoothing and optional class weights.
    """

    def __init__(self, config: FullConfig):
        self._config = config

    def build(self) -> nn.Module:
        loss_cfg = self._config.loss
        if loss_cfg.name.lower() != 'cross_entropy':
            raise ValueError(f"Unsupported loss: {loss_cfg.name}")

        label_smoothing = float(loss_cfg.label_smoothing or 0.0)

        weight = None
        if loss_cfg.class_weights and loss_cfg.class_weights.get('enabled'):
            values = loss_cfg.class_weights.get('values')
            if values:
                weight = torch.tensor(values, dtype=torch.float)

        # Use built-in label_smoothing if available; otherwise implement manually
        try:
            return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        except TypeError:
            # Fallback for older torch versions: wrap logits processing
            return _CrossEntropyWithLabelSmoothing(weight=weight, smoothing=label_smoothing)


class _CrossEntropyWithLabelSmoothing(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, smoothing: float = 0.0):
        super().__init__()
        self.register_buffer('weight', weight) if weight is not None else None
        self.smoothing = float(smoothing)

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
        loss = - (true_dist * log_probs)
        if self.weight is not None:
            loss = loss * self.weight.unsqueeze(0)
        return loss.sum(dim=1).mean()
