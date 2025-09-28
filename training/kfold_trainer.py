"""Professional K-Fold Cross-Validation for PhD Thesis with WandB Integration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import wandb

from src.config import FullConfig
from src.factories.model_factory import ModelFactory
from src.factories.optimizer_factory import OptimizerFactory
from src.factories.scheduler_factory import SchedulerFactory
from src.factories.loss_factory import LossFactory
from src.data.datamodule import DataModule


class ProfessionalKFoldTrainer:
    """PhD-grade K-Fold trainer with WandB integration and minimal checkpointing."""
    
    def __init__(self, config: FullConfig, n_folds: int = 5, output_dir: str = "kfold_results"):
        self.config = config
        self.n_folds = n_folds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = config.experiment.device
        self.fold_results = []
        self.wandb_run = None
        
        # Initialize WandB if enabled
        if config.logging.backend == "wandb":
            self._init_wandb_experiment()
    
    def _init_wandb_experiment(self):
        """Initialize WandB experiment for K-fold tracking."""
        self.wandb_run = wandb.init(
            project=self.config.logging.project_name,
            name=f"kfold-{self.n_folds}fold-{self.config.experiment.name}",
            tags=self.config.logging.run_tags + ["kfold_cv", "phd_thesis"],
            config={
                **self.config.dict(),
                "n_folds": self.n_folds,
                "cv_strategy": "stratified_kfold"
            },
            notes=f"PhD Thesis {self.n_folds}-Fold Cross-Validation for Spatial Audio Classification"
        )
    
    def prepare_kfold_splits(self, datamodule: DataModule) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Prepare stratified K-fold splits."""
        print(f"📊 Preparing {self.n_folds}-fold stratified splits...")
        
        # Get all data and labels
        datamodule.setup()
        full_dataset = datamodule._train_dataset
        
        # Extract file paths and labels for tracking
        all_files = []
        all_labels = []
        
        for idx in range(len(full_dataset)):
            file_path, label = full_dataset.dataset.samples[full_dataset.indices[idx]]
            all_files.append(Path(file_path).name)  # Store filename only
            all_labels.append(label)
        
        # Create stratified K-fold splitter
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.config.experiment.seed)
        
        splits = []
        split_info = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):
            splits.append((train_idx, val_idx))
            
            # Track split information
            train_files = [all_files[i] for i in train_idx]
            val_files = [all_files[i] for i in val_idx]
            train_labels = [all_labels[i] for i in train_idx]
            val_labels = [all_labels[i] for i in val_idx]
            
            split_info.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_files': train_files,
                'val_files': val_files,
                'train_labels': train_labels,
                'val_labels': val_labels,
                'train_class_dist': {str(cls): train_labels.count(cls) for cls in set(train_labels)},
                'val_class_dist': {str(cls): val_labels.count(cls) for cls in set(val_labels)}
            })
        
        # Save split information for thesis documentation
        with open(self.output_dir / 'kfold_splits.json', 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"✅ {self.n_folds} splits prepared and documented")
        return splits
    
    def train_single_fold(self, fold_idx: int, train_idx: np.ndarray, val_idx: np.ndarray, 
                         datamodule: DataModule) -> Dict[str, Any]:
        """Train a single fold with minimal checkpointing."""
        print(f"\n🔄 Training Fold {fold_idx + 1}/{self.n_folds}")
        print("-" * 50)
        
        fold_start_time = time.time()
        
        # Create fold-specific datasets
        train_subset = torch.utils.data.Subset(datamodule._train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(datamodule._train_dataset, val_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_subset, 
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory
        )
        
        # Initialize model, optimizer, scheduler
        model_factory = ModelFactory(self.config)
        model = model_factory.build().to(self.device)
        
        optimizer_factory = OptimizerFactory(self.config)
        optimizer = optimizer_factory.build(model)
        
        scheduler_factory = SchedulerFactory(self.config)
        scheduler = scheduler_factory.build(optimizer)
        
        loss_factory = LossFactory(self.config)
        criterion = loss_factory.build()
        
        scaler = GradScaler() if self.config.optimization.amp.enabled else None
        
        # Training loop
        best_val_metric = -float('inf')
        best_epoch = 0
        fold_history = {
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': [],
            'val_f1_scores': []
        }
        
        for epoch in range(self.config.training.epochs_max):
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion, scaler, epoch)
            
            # Validate epoch
            val_metrics = self._validate_epoch(model, val_loader, criterion, scaler, epoch)
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Track metrics
            fold_history['train_losses'].append(train_loss)
            fold_history['val_losses'].append(val_metrics['loss'])
            fold_history['val_accuracies'].append(val_metrics['accuracy'])
            fold_history['val_f1_scores'].append(val_metrics['f1_score'])
            
            # Log to WandB
            if self.wandb_run:
                self.wandb_run.log({
                    f'fold_{fold_idx}/train_loss': train_loss,
                    f'fold_{fold_idx}/val_loss': val_metrics['loss'],
                    f'fold_{fold_idx}/val_accuracy': val_metrics['accuracy'],
                    f'fold_{fold_idx}/val_f1': val_metrics['f1_score'],
                    f'fold_{fold_idx}/epoch': epoch,
                    'global_step': fold_idx * self.config.training.epochs_max + epoch
                })
            
            # Track best model
            current_metric = val_metrics['accuracy']  # or f1_score
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_epoch = epoch
                
                # Save ONLY the best checkpoint for this fold (space-efficient)
                self._save_fold_checkpoint(model, optimizer, fold_idx, epoch, val_metrics)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{self.config.training.epochs_max} - "
                  f"Loss: {train_loss:.4f} - Val Acc: {val_metrics['accuracy']:.4f} - "
                  f"Val F1: {val_metrics['f1_score']:.4f} - Time: {epoch_time:.1f}s")
        
        fold_time = time.time() - fold_start_time
        
        fold_results = {
            'fold_idx': fold_idx,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_metric,
            'best_val_f1': max(fold_history['val_f1_scores']),
            'final_train_loss': fold_history['train_losses'][-1],
            'final_val_loss': fold_history['val_losses'][-1],
            'training_time_minutes': fold_time / 60,
            'history': fold_history
        }
        
        print(f"✅ Fold {fold_idx + 1} completed - Best Val Acc: {best_val_metric:.4f} - Time: {fold_time/60:.1f}min")
        
        return fold_results
    
    def _train_epoch(self, model: nn.Module, loader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module, 
                    scaler: Optional[GradScaler], epoch: int) -> float:
        """Train single epoch."""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def _validate_epoch(self, model: nn.Module, loader: torch.utils.data.DataLoader,
                       criterion: nn.Module, scaler: Optional[GradScaler], epoch: int) -> Dict[str, float]:
        """Validate single epoch."""
        model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if scaler:
                    with autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1_score_macro = f1_score(all_targets, all_preds, average='macro')
        
        return {
            'loss': total_loss / len(loader),
            'accuracy': accuracy,
            'f1_score': f1_score_macro
        }
    
    def _save_fold_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                             fold_idx: int, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint for fold (only best model per fold)."""
        checkpoint = {
            'fold': fold_idx,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.dict()
        }
        
        checkpoint_path = self.output_dir / f'fold_{fold_idx}_best.pt'
        torch.save(checkpoint, checkpoint_path)
    
    def run_kfold_cv(self) -> Dict[str, Any]:
        """Execute complete K-fold cross-validation."""
        print("🎓 Starting Professional K-Fold Cross-Validation")
        print(f"📊 Configuration: {self.n_folds} folds, {self.config.training.epochs_max} epochs per fold")
        print("=" * 70)
        
        cv_start_time = time.time()
        
        # Prepare data
        datamodule = DataModule(self.config)
        splits = self.prepare_kfold_splits(datamodule)
        
        # Train each fold
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            fold_result = self.train_single_fold(fold_idx, train_idx, val_idx, datamodule)
            fold_results.append(fold_result)
        
        cv_time = time.time() - cv_start_time
        
        # Aggregate results
        cv_summary = self._aggregate_results(fold_results, cv_time)
        
        # Save comprehensive report
        self._save_cv_report(cv_summary, fold_results)
        
        # Log summary to WandB
        if self.wandb_run:
            self._log_cv_summary_to_wandb(cv_summary)
        
        print("=" * 70)
        print("✅ K-Fold Cross-Validation Completed!")
        print(f"📊 Mean Validation Accuracy: {cv_summary['mean_val_accuracy']:.4f} ± {cv_summary['std_val_accuracy']:.4f}")
        print(f"📊 Mean Validation F1: {cv_summary['mean_val_f1']:.4f} ± {cv_summary['std_val_f1']:.4f}")
        print(f"⏱️ Total Time: {cv_time/3600:.2f} hours")
        print(f"💾 Results saved to: {self.output_dir}")
        
        return cv_summary
    
    def _aggregate_results(self, fold_results: List[Dict], cv_time: float) -> Dict[str, Any]:
        """Aggregate results across folds."""
        accuracies = [r['best_val_accuracy'] for r in fold_results]
        f1_scores = [r['best_val_f1'] for r in fold_results]
        training_times = [r['training_time_minutes'] for r in fold_results]
        
        return {
            'n_folds': self.n_folds,
            'mean_val_accuracy': float(np.mean(accuracies)),
            'std_val_accuracy': float(np.std(accuracies)),
            'mean_val_f1': float(np.mean(f1_scores)),
            'std_val_f1': float(np.std(f1_scores)),
            'min_val_accuracy': float(np.min(accuracies)),
            'max_val_accuracy': float(np.max(accuracies)),
            'min_val_f1': float(np.min(f1_scores)),
            'max_val_f1': float(np.max(f1_scores)),
            'total_cv_time_hours': cv_time / 3600,
            'mean_fold_time_minutes': float(np.mean(training_times)),
            'individual_fold_results': fold_results
        }
    
    def _save_cv_report(self, cv_summary: Dict, fold_results: List[Dict]):
        """Save comprehensive CV report."""
        with open(self.output_dir / 'kfold_cv_results.json', 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        # Create human-readable report
        report_md = f"""# {self.n_folds}-Fold Cross-Validation Results

## Summary Statistics
- **Mean Validation Accuracy**: {cv_summary['mean_val_accuracy']:.4f} ± {cv_summary['std_val_accuracy']:.4f}
- **Mean Validation F1-Score**: {cv_summary['mean_val_f1']:.4f} ± {cv_summary['std_val_f1']:.4f}
- **Accuracy Range**: [{cv_summary['min_val_accuracy']:.4f}, {cv_summary['max_val_accuracy']:.4f}]
- **F1 Range**: [{cv_summary['min_val_f1']:.4f}, {cv_summary['max_val_f1']:.4f}]

## Training Details
- **Total CV Time**: {cv_summary['total_cv_time_hours']:.2f} hours
- **Mean Fold Time**: {cv_summary['mean_fold_time_minutes']:.1f} minutes
- **Epochs per Fold**: {self.config.training.epochs_max}

## Individual Fold Results
"""
        for i, result in enumerate(fold_results):
            report_md += f"""
### Fold {i+1}
- **Best Validation Accuracy**: {result['best_val_accuracy']:.4f}
- **Best Validation F1**: {result['best_val_f1']:.4f}
- **Best Epoch**: {result['best_epoch']}
- **Training Time**: {result['training_time_minutes']:.1f} minutes
"""
        
        with open(self.output_dir / 'CV_REPORT.md', 'w') as f:
            f.write(report_md)
    
    def _log_cv_summary_to_wandb(self, cv_summary: Dict):
        """Log CV summary to WandB."""
        self.wandb_run.log({
            'cv_summary/mean_val_accuracy': cv_summary['mean_val_accuracy'],
            'cv_summary/std_val_accuracy': cv_summary['std_val_accuracy'],
            'cv_summary/mean_val_f1': cv_summary['mean_val_f1'],
            'cv_summary/std_val_f1': cv_summary['std_val_f1'],
            'cv_summary/total_time_hours': cv_summary['total_cv_time_hours']
        })
        
        # Create summary table
        fold_data = []
        for i, result in enumerate(cv_summary['individual_fold_results']):
            fold_data.append([
                i+1,
                result['best_val_accuracy'],
                result['best_val_f1'],
                result['training_time_minutes']
            ])
        
        table = wandb.Table(
            data=fold_data,
            columns=["Fold", "Best_Val_Accuracy", "Best_Val_F1", "Training_Time_Min"]
        )
        
        self.wandb_run.log({"cv_results_table": table})
    
    def __del__(self):
        """Clean up WandB run."""
        if self.wandb_run:
            wandb.finish()
