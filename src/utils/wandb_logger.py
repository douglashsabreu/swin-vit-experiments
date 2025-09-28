"""WandB Logger for PhD Thesis Experiments.

Comprehensive logging utilities for academic research with WandB integration.
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional, List
from pathlib import Path

import wandb
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class ThesisWandBLogger:
    """Academic-focused WandB logger for PhD thesis experiments."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.run = None
        self._setup_wandb()
        
    def _setup_wandb(self) -> None:
        """Initialize WandB run with thesis-specific configuration."""
        logging_config = self.config.get('logging', {})
        
        # Initialize WandB run
        self.run = wandb.init(
            project=logging_config.get('project_name', 'spatial-audio-classification-phd'),
            name=logging_config.get('run_name', 'swin-transformer-experiment'),
            tags=logging_config.get('run_tags', []),
            config=self.config,
            save_code=True,
            notes="PhD Thesis Experiment: Spatial Audio Classification using Swin Vision Transformer"
        )
        
        # Log system information
        self._log_system_info()
        
    def _log_system_info(self) -> None:
        """Log system and hardware information."""
        system_info = {
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'pytorch_version': torch.__version__,
            'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
            'mixed_precision': self.config.get('experiment', {}).get('mixed_precision', False),
            'batch_size': self.config.get('training', {}).get('batch_size', 0),
            'max_epochs': self.config.get('training', {}).get('epochs_max', 0)
        }
        
        wandb.config.update(system_info)
        
    def log_epoch_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], lr: float = None) -> None:
        """Log training and validation metrics for each epoch."""
        log_dict = {
            'epoch': epoch,
            'train/loss': train_metrics.get('loss', 0),
            'val/loss': val_metrics.get('loss', 0),
            'val/accuracy': val_metrics.get('val_accuracy', 0),
            'val/macro_f1': val_metrics.get('val_macro_f1', 0),
        }
        
        if lr is not None:
            log_dict['learning_rate'] = lr
            
        # Add GPU utilization if available
        if torch.cuda.is_available():
            log_dict['gpu/memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            log_dict['gpu/memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            
        wandb.log(log_dict, step=epoch)
        
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str], epoch: int) -> None:
        """Log confusion matrix as WandB artifact."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        
        # Log to WandB
        wandb.log({
            "confusion_matrix": wandb.Image(plt),
            "confusion_matrix_table": wandb.Table(
                data=cm.tolist(),
                columns=class_names,
                rows=class_names
            )
        }, step=epoch)
        
        plt.close()
        
    def log_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                class_names: List[str], epoch: int) -> None:
        """Log detailed classification report."""
        report = classification_report(y_true, y_pred, target_names=class_names, 
                                     output_dict=True, zero_division=0)
        
        # Log per-class metrics
        for class_name in class_names:
            if class_name in report:
                wandb.log({
                    f"class_{class_name}/precision": report[class_name]['precision'],
                    f"class_{class_name}/recall": report[class_name]['recall'],
                    f"class_{class_name}/f1_score": report[class_name]['f1-score'],
                    f"class_{class_name}/support": report[class_name]['support']
                }, step=epoch)
                
        # Log aggregate metrics
        wandb.log({
            "metrics/macro_precision": report['macro avg']['precision'],
            "metrics/macro_recall": report['macro avg']['recall'],
            "metrics/macro_f1": report['macro avg']['f1-score'],
            "metrics/weighted_f1": report['weighted avg']['f1-score'],
            "metrics/accuracy": report['accuracy']
        }, step=epoch)
        
    def log_model_checkpoint(self, checkpoint_path: str, epoch: int, 
                           metrics: Dict[str, float]) -> None:
        """Log model checkpoint as WandB artifact."""
        artifact = wandb.Artifact(
            name=f"model_checkpoint_epoch_{epoch}",
            type="model",
            description=f"Model checkpoint at epoch {epoch}",
            metadata=metrics
        )
        
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        
    def log_dataset_info(self, train_size: int, val_size: int, test_size: int,
                        class_distribution: Dict[str, int]) -> None:
        """Log dataset information and class distribution."""
        # Log dataset sizes
        wandb.log({
            "dataset/train_size": train_size,
            "dataset/val_size": val_size,
            "dataset/test_size": test_size,
            "dataset/total_size": train_size + val_size + test_size
        })
        
        # Create class distribution plot
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, counts, color='skyblue', alpha=0.7)
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        wandb.log({"dataset/class_distribution": wandb.Image(plt)})
        plt.close()
        
        # Log class distribution table
        wandb.log({
            "dataset/class_distribution_table": wandb.Table(
                data=[[k, v] for k, v in class_distribution.items()],
                columns=["Class", "Count"]
            )
        })
        
    def log_training_summary(self, final_metrics: Dict[str, float], 
                           total_training_time: float, best_epoch: int) -> None:
        """Log final training summary."""
        summary = {
            "summary/final_accuracy": final_metrics.get('accuracy', 0),
            "summary/final_macro_f1": final_metrics.get('macro_f1', 0),
            "summary/final_weighted_f1": final_metrics.get('weighted_f1', 0),
            "summary/best_epoch": best_epoch,
            "summary/total_training_time_hours": total_training_time / 3600,
            "summary/avg_time_per_epoch_minutes": (total_training_time / best_epoch) / 60
        }
        
        wandb.log(summary)
        
        # Create summary table
        summary_table = wandb.Table(
            data=[[k.replace('summary/', ''), v] for k, v in summary.items()],
            columns=["Metric", "Value"]
        )
        wandb.log({"summary/final_results": summary_table})
        
    def log_attention_visualizations(self, attention_maps: np.ndarray, 
                                   sample_images: np.ndarray, epoch: int) -> None:
        """Log attention visualization samples."""
        if len(attention_maps) == 0 or len(sample_images) == 0:
            return
            
        # Create attention visualization
        fig, axes = plt.subplots(2, min(5, len(sample_images)), figsize=(15, 6))
        if len(sample_images) == 1:
            axes = axes.reshape(2, 1)
            
        for i in range(min(5, len(sample_images))):
            # Original image
            axes[0, i].imshow(sample_images[i].transpose(1, 2, 0))
            axes[0, i].set_title(f'Sample {i+1}')
            axes[0, i].axis('off')
            
            # Attention map
            axes[1, i].imshow(attention_maps[i], cmap='hot', alpha=0.7)
            axes[1, i].set_title(f'Attention Map {i+1}')
            axes[1, i].axis('off')
            
        plt.suptitle(f'Attention Visualizations - Epoch {epoch}')
        plt.tight_layout()
        
        wandb.log({"attention_maps": wandb.Image(plt)}, step=epoch)
        plt.close()
        
    def finish(self) -> None:
        """Finish WandB run."""
        if self.run:
            wandb.finish()
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

