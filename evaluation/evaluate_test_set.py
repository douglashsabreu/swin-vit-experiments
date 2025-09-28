"""Evaluate the best trained model on the test set for PhD thesis."""

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)

from src.config import load_config
from src.factories.model_factory import ModelFactory
from src.data.datamodule import DataModule


class TestSetEvaluator:
    """Evaluate trained model on test set for PhD thesis results."""
    
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config = load_config(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = self.config.experiment.device
        
        # Initialize components
        self.datamodule = DataModule(self.config)
        self.model = None
        
    def load_model(self) -> None:
        """Load the best trained model from checkpoint."""
        print(f"📦 Loading model from: {self.checkpoint_path}")
        
        # Build model architecture
        model_factory = ModelFactory(self.config)
        self.model = model_factory.build()
        
        # Load checkpoint
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
            
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Checkpoint metrics: {checkpoint.get('metrics', 'N/A')}")
        
    def evaluate_test_set(self) -> Dict[str, float]:
        """Evaluate model on test set and return comprehensive metrics."""
        print("🔬 Evaluating on test set...")
        
        # Setup data
        self.datamodule.setup()
        test_loader = self.datamodule.test_dataloader()
        
        if test_loader is None:
            raise ValueError("No test set available! Check data configuration.")
            
        print(f"📊 Test set size: {len(test_loader.dataset)} samples")
        
        # Collect predictions
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(test_loader):
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Collect results
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"   Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Concatenate results
        y_true = np.concatenate(all_targets)
        y_pred = np.concatenate(all_preds)
        y_probs = np.concatenate(all_probs)
        
        print(f"✅ Evaluation completed on {len(y_true)} samples")
        
        return self._calculate_metrics(y_true, y_pred, y_probs)
        
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_probs: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive test metrics."""
        class_names = self.config.classes.labels
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None)
        per_class_precision = precision_score(y_true, y_pred, average=None)
        per_class_recall = recall_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Compile results
        metrics = {
            'test_accuracy': float(accuracy),
            'test_macro_f1': float(macro_f1),
            'test_weighted_f1': float(weighted_f1),
            'test_macro_precision': float(macro_precision),
            'test_macro_recall': float(macro_recall),
        }
        
        # Add per-class metrics
        for i, class_name in enumerate(class_names):
            metrics[f'test_f1_class_{class_name}'] = float(per_class_f1[i])
            metrics[f'test_precision_class_{class_name}'] = float(per_class_precision[i])
            metrics[f'test_recall_class_{class_name}'] = float(per_class_recall[i])
        
        # Store additional data for analysis
        self.confusion_matrix = cm
        self.classification_report = class_report
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_probs = y_probs
        
        return metrics
        
    def save_results(self, metrics: Dict[str, float], output_dir: str = "test_results") -> None:
        """Save test results for thesis documentation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics JSON
        with open(output_path / "test_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save confusion matrix
        np.save(output_path / "confusion_matrix.npy", self.confusion_matrix)
        
        # Save classification report
        with open(output_path / "classification_report.json", 'w') as f:
            json.dump(self.classification_report, f, indent=2)
        
        # Save predictions for further analysis
        results_data = {
            'y_true': self.y_true.tolist(),
            'y_pred': self.y_pred.tolist(),
            'y_probs': self.y_probs.tolist(),
            'class_names': self.config.classes.labels
        }
        
        with open(output_path / "predictions.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"💾 Results saved to: {output_path}")
        
    def print_results(self, metrics: Dict[str, float]) -> None:
        """Print formatted test results for thesis."""
        print("\n" + "="*60)
        print("🎓 PhD THESIS - FINAL TEST SET RESULTS")
        print("="*60)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   • Test Accuracy:      {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)")
        print(f"   • Macro F1-Score:     {metrics['test_macro_f1']:.4f} ({metrics['test_macro_f1']*100:.2f}%)")
        print(f"   • Weighted F1-Score:  {metrics['test_weighted_f1']:.4f} ({metrics['test_weighted_f1']*100:.2f}%)")
        print(f"   • Macro Precision:    {metrics['test_macro_precision']:.4f} ({metrics['test_macro_precision']*100:.2f}%)")
        print(f"   • Macro Recall:       {metrics['test_macro_recall']:.4f} ({metrics['test_macro_recall']*100:.2f}%)")
        
        print(f"\n🎯 PER-CLASS PERFORMANCE:")
        class_names = self.config.classes.labels
        for class_name in class_names:
            f1 = metrics[f'test_f1_class_{class_name}']
            precision = metrics[f'test_precision_class_{class_name}']
            recall = metrics[f'test_recall_class_{class_name}']
            print(f"   • Class {class_name}:")
            print(f"     - F1:        {f1:.4f} ({f1*100:.2f}%)")
            print(f"     - Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"     - Recall:    {recall:.4f} ({recall*100:.2f}%)")
        
        print(f"\n📈 CONFUSION MATRIX:")
        print("     Predicted →")
        print("   ", end="")
        for name in class_names:
            print(f"{name:>6}", end="")
        print()
        
        for i, true_name in enumerate(class_names):
            print(f"{true_name:>3}|", end="")
            for j in range(len(class_names)):
                print(f"{self.confusion_matrix[i,j]:>6}", end="")
            print()
        
        print(f"\n🏆 ACADEMIC SIGNIFICANCE:")
        print(f"   • Model: Swin Vision Transformer Base")
        print(f"   • Task: 4-class spatial audio classification")
        print(f"   • Test Set: {len(self.y_true)} samples (10% of dataset)")
        print(f"   • Performance Level: {'Excellent' if metrics['test_accuracy'] > 0.9 else 'Good' if metrics['test_accuracy'] > 0.8 else 'Moderate'}")
        print(f"   • Generalization: {'Strong' if abs(metrics['test_accuracy'] - 0.9446) < 0.05 else 'Moderate'}")
        
        print("\n" + "="*60)


def find_best_checkpoint() -> str:
    """Find the best checkpoint file."""
    logs_dir = Path("logs")
    checkpoints = list(logs_dir.glob("best_checkpoint_*.pt"))
    
    if not checkpoints:
        raise FileNotFoundError("No checkpoint files found in logs/")
    
    # Find the checkpoint with highest epoch number (latest)
    best_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    return str(best_checkpoint)


def main():
    """Main evaluation function."""
    print("🎓 PhD Thesis - Test Set Evaluation")
    print("="*50)
    
    try:
        # Find best checkpoint
        checkpoint_path = find_best_checkpoint()
        print(f"📦 Using checkpoint: {checkpoint_path}")
        
        # Initialize evaluator
        evaluator = TestSetEvaluator(
            config_path="experiments/spatial_experiment.yaml",
            checkpoint_path=checkpoint_path
        )
        
        # Load model
        evaluator.load_model()
        
        # Evaluate on test set
        metrics = evaluator.evaluate_test_set()
        
        # Print and save results
        evaluator.print_results(metrics)
        evaluator.save_results(metrics)
        
        print("\n✅ Test evaluation completed successfully!")
        print("📊 Results ready for PhD thesis documentation")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
