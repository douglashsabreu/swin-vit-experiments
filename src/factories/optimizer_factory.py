from __future__ import annotations

from typing import Iterable, Dict

import torch
from torch import nn

from src.config import FullConfig


class OptimizerFactory:
    """Factory to create optimizer instances from configuration.

    The factory supports discriminative learning rates by grouping parameters
    into 'head' and 'backbone' parameter groups. It also exempts parameters
    like biases and LayerNorm/BatchNorm weights from weight decay when
    appropriate.
    """

    def __init__(self, config: FullConfig):
        self._config = config

    def build(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create an optimizer for the provided model.

        Args:
            model: The model whose parameters will be optimized.

        Returns:
            An instance of torch.optim.Optimizer configured per the config.
        """
        opt_cfg = self._config.optimization.optimizer

        head_lr = float(opt_cfg.lr_head)
        backbone_lr = float(opt_cfg.lr_backbone)
        weight_decay = float(opt_cfg.weight_decay)

        param_groups = self._param_groups(model, backbone_lr, head_lr, weight_decay)

        if opt_cfg.name.lower() == 'adamw':
            betas = tuple(opt_cfg.betas)
            return torch.optim.AdamW(param_groups, betas=betas)

        raise ValueError(f"Unsupported optimizer: {opt_cfg.name}")

    def _param_groups(self, model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float):
        """Create parameter groups distinguishing backbone and head.

        The method detects 'head' attribute on the top-level model to separate
        parameters; all remaining parameters are treated as backbone.
        """
        head_params = []
        backbone_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('head'):
                head_params.append((name, param))
            else:
                backbone_params.append((name, param))

        def _exclude_from_weight_decay(n):
            n = n.lower()
            for token in ('bias', 'bn', 'layernorm', 'ln'):
                if token in n:
                    return True
            return False

        def _groupify(pairs, lr):
            wd_params = [p for n, p in pairs if not _exclude_from_weight_decay(n)]
            no_wd_params = [p for n, p in pairs if _exclude_from_weight_decay(n)]
            groups = []
            if wd_params:
                groups.append({"params": wd_params, "lr": lr, "weight_decay": weight_decay})
            if no_wd_params:
                groups.append({"params": no_wd_params, "lr": lr, "weight_decay": 0.0})
            return groups

        groups = _groupify(backbone_params, backbone_lr) + _groupify(head_params, head_lr)
        return groups
