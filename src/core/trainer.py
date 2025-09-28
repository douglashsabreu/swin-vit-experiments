from __future__ import annotations

import math
import time
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

from src.config import FullConfig
from src.factories.model_factory import ModelFactory
from src.factories.optimizer_factory import OptimizerFactory
from src.factories.scheduler_factory import SchedulerFactory
from src.factories.loss_factory import LossFactory
from src.data.datamodule import DataModule
from pathlib import Path


class Trainer:
    """Template Method trainer orchestrating transfer learning lifecycle.

    The trainer runs the configured phases: linear probe (head-only), selective
    fine-tune (unfreeze upper stages), and optional progressive unfreeze. It
    supports mixed precision, gradient accumulation, clipping, early stopping,
    checkpointing, and learning-rate scheduling via injected factories.
    """

    def __init__(self, config: FullConfig):
        self._config = config
        self._device = config.experiment.device
        self._model = None
        self._optimizer = None
        self._scheduler = None
        self._loss_fn = None
        self._scaler = GradScaler() if config.optimization.amp.enabled else None

        self._datamodule = DataModule(config)

        self._best_metric = -math.inf
        self._best_state: Optional[Dict[str, Any]] = None
        self._epochs_no_improve = 0

    def setup(self) -> None:
        self._datamodule.setup()
        factory = ModelFactory(self._config)
        self._model = factory.build()

        opt_factory = OptimizerFactory(self._config)
        self._optimizer = opt_factory.build(self._model)

        sched_factory = SchedulerFactory(self._config)
        self._scheduler = sched_factory.build(self._optimizer)

        loss_factory = LossFactory(self._config)
        self._loss_fn = loss_factory.build()

    def run(self) -> None:
        self.setup()

        train_loader = self._datamodule.train_dataloader()
        val_loader = self._datamodule.val_dataloader()

        epochs_max = int(self._config.training.epochs_max)
        probe_epochs = int(self._config.model.freezing_policy.phase1_linear_probe_epochs)
        grad_accum = int(self._config.training.grad_accum_steps)

        # Phase 1: linear probe (head only)
        self._set_train_mode('linear_probe')
        self._run_epochs(0, probe_epochs, train_loader, val_loader, grad_accum)

        # Phase 2: selective fine-tune
        self._apply_selective_unfreeze()
        self._set_train_mode('selective_finetune')
        remaining = max(0, epochs_max - probe_epochs)
        self._run_epochs(probe_epochs, probe_epochs + remaining, train_loader, val_loader, grad_accum)

    def _run_epochs(self, start_epoch: int, end_epoch: int, train_loader, val_loader, grad_accum: int) -> None:
        for epoch in range(start_epoch, end_epoch):
            t0 = time.time()
            train_metrics = self._train_one_epoch(train_loader, epoch, grad_accum)
            val_metrics = self._validate_one_epoch(val_loader, epoch)

            # Scheduler step
            if hasattr(self._scheduler, 'step') and isinstance(self._scheduler, type(torch.optim.lr_scheduler.ReduceLROnPlateau)):
                # ReduceLROnPlateau expects metric
                self._scheduler.step(val_metrics.get(self._config.checkpointing.monitor))
            elif hasattr(self._scheduler, 'step'):
                # Our chained scheduler provides step() without args
                try:
                    self._scheduler.step()
                except TypeError:
                    self._scheduler.step()

            improved = self._check_improvement(val_metrics)
            if improved:
                self._save_checkpoint(epoch, val_metrics)

            if self._early_stop_check(val_metrics):
                if self._config.training.early_stopping.restore_best:
                    self._restore_best()
                break

            t1 = time.time()
            epoch_time = t1 - t0
            print(f"Epoch {epoch+1}/{end_epoch} - train:{train_metrics} val:{val_metrics} time:{epoch_time:.1f}s")

    def _train_one_epoch(self, loader, epoch: int, grad_accum: int) -> Dict[str, float]:
        self._model.train()
        running_loss = 0.0
        n_samples = 0

        optimizer = self._optimizer
        loss_fn = self._loss_fn

        optimizer.zero_grad()
        for step, (images, targets) in enumerate(loader):
            images = images.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)

            with autocast(enabled=(self._scaler is not None)):
                logits = self._model(images)
                loss = loss_fn(logits, targets)
                loss_value = loss.item() if isinstance(loss, torch.Tensor) else float(loss)

            scaled = loss / float(grad_accum)

            if self._scaler is not None:
                self._scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if (step + 1) % grad_accum == 0:
                if self._config.optimization.gradient_clipping.enabled:
                    max_norm = float(self._config.optimization.gradient_clipping.max_norm)
                    if self._scaler is not None:
                        self._scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)

                if self._scaler is not None:
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss_value * images.size(0)
            n_samples += images.size(0)

        avg_loss = running_loss / max(1, n_samples)
        return {'loss': float(avg_loss)}

    def _validate_one_epoch(self, loader, epoch: int) -> Dict[str, float]:
        self._model.eval()
        running_loss = 0.0
        n_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                with autocast(enabled=(self._scaler is not None)):
                    logits = self._model(images)
                    loss = self._loss_fn(logits, targets)

                running_loss += float(loss.item()) * images.size(0)
                n_samples += images.size(0)

                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())

        avg_loss = running_loss / max(1, n_samples)
        preds_cat = torch.cat(all_preds) if all_preds else torch.tensor([])
        targets_cat = torch.cat(all_targets) if all_targets else torch.tensor([])

        metrics = {'loss': float(avg_loss)}
        # compute macro-f1 and accuracy if possible
        try:
            from sklearn.metrics import f1_score, accuracy_score
            if preds_cat.numel() > 0:
                metrics['val_macro_f1'] = float(f1_score(targets_cat.numpy(), preds_cat.numpy(), average='macro'))
                metrics['val_accuracy'] = float(accuracy_score(targets_cat.numpy(), preds_cat.numpy()))
        except Exception:
            pass

        return metrics

    def _set_train_mode(self, phase: str) -> None:
        # placeholder for mode-specific adjustments
        return

    def _apply_selective_unfreeze(self) -> None:
        # The ModelFactory already applied initial freezing; here we re-enable gradients for configured stages
        policy = self._config.model.freezing_policy
        # Best-effort: attempt to set requires_grad True for unfreeze_stages if not already
        for name, param in self._model.named_parameters():
            for stage in policy.unfreeze_stages:
                token = f'layers.{stage-1}'
                if token in name:
                    param.requires_grad = True

    def _check_improvement(self, metrics: Dict[str, float]) -> bool:
        key = self._config.checkpointing.monitor
        value = metrics.get(key) or metrics.get('val_macro_f1') or -math.inf
        if value is None:
            return False
        if value > self._best_metric:
            self._best_metric = value
            self._epochs_no_improve = 0
            return True
        else:
            self._epochs_no_improve += 1
            return False

    def _early_stop_check(self, metrics: Dict[str, float]) -> bool:
        if not self._config.training.early_stopping.enabled:
            return False
        if self._epochs_no_improve > int(self._config.training.early_stopping.patience):
            return True
        return False

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> None:
        state = {
            'epoch': epoch,
            'model_state': self._model.state_dict(),
            'optimizer_state': self._optimizer.state_dict(),
            'scheduler_state': getattr(self._scheduler, 'state_dict', lambda: None)(),
            'metrics': metrics,
        }
        self._best_state = state
        save_dir = self._config.logging.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / f"best_checkpoint_{epoch}.pt"
        torch.save(state, str(save_path))

    def _restore_best(self) -> None:
        if self._best_state is None:
            return
        self._model.load_state_dict(self._best_state['model_state'])
        self._optimizer.load_state_dict(self._best_state['optimizer_state'])
        if hasattr(self._scheduler, 'load_state_dict') and self._best_state.get('scheduler_state') is not None:
            try:
                self._scheduler.load_state_dict(self._best_state['scheduler_state'])
            except Exception:
                pass
