from __future__ import annotations

from typing import Optional

import math
import torch

from src.config import FullConfig


class SchedulerFactory:
    """Factory that constructs LR schedulers from configuration.

    Supports 'cosine_warmup' and 'reduce_on_plateau'. Warmup is implemented
    via a chained scheduler when requested in the config parameters.
    """

    def __init__(self, config: FullConfig):
        self._config = config

    def build(self, optimizer: torch.optim.Optimizer):
        sched_cfg = self._config.optimization.scheduler
        name = sched_cfg.name.lower()
        params = sched_cfg.params

        if name == 'reduce_on_plateau' or name == 'reduceonplateau':
            factor = float(params.factor or 0.1)
            patience = int(params.patience or 10)
            min_lr = float(params.min_lr or 0.0)
            mode = params.mode or 'max'
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)

        if name == 'cosine_warmup' or name == 'cosine':
            warmup_epochs = int(params.warmup_epochs or 0)
            total_epochs = int(self._config.training.epochs_max)
            min_lr = float(params.min_lr or 0.0)

            # cosine annealing scheduler
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=min_lr)

            if warmup_epochs > 0:
                def _lr_lambda(current_epoch):
                    if current_epoch < warmup_epochs:
                        return float(current_epoch) / float(max(1, warmup_epochs))
                    return 1.0

                warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

                # Chain warmup then cosine by stepping both appropriately in training loop.
                # Provide a simple object to hold both schedulers and a mode flag.
                return _ChainedScheduler(warmup_scheduler=warmup, main_scheduler=cosine, warmup_epochs=warmup_epochs)

            return cosine

        raise ValueError(f"Unsupported scheduler: {sched_cfg.name}")


class _ChainedScheduler:
    """Helper that wraps warmup scheduler and main scheduler.

    The training loop should call `step()` each epoch; this wrapper ensures
    warmup is applied for initial epochs then the main scheduler is stepped.
    """

    def __init__(self, warmup_scheduler, main_scheduler, warmup_epochs: int):
        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.warmup_epochs = warmup_epochs
        self._epoch = 0

    def step(self, metric: Optional[float] = None):
        if self._epoch < self.warmup_epochs:
            self.warmup_scheduler.step()
        else:
            # step main scheduler; it expects epochs counting from 0
            self.main_scheduler.step()
        self._epoch += 1

    def state_dict(self):
        return {'warmup': self.warmup_scheduler.state_dict(), 'main': self.main_scheduler.state_dict(), 'epoch': self._epoch}

    def load_state_dict(self, state):
        self.warmup_scheduler.load_state_dict(state['warmup'])
        self.main_scheduler.load_state_dict(state['main'])
        self._epoch = int(state.get('epoch', 0))
