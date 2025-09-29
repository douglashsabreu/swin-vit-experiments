"""Detailed Test Evaluation with Per-File Analysis and Automatic WandB Logging."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import autocast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import wandb

from src.config import FullConfig, load_config
from src.factories.model_factory import ModelFactory
from src.data.datamodule import DataModule


class DetailedTestEvaluator:
    """PhD-grade test evaluator with per-file analysis and automatic WandB logging."""
    
    def __init__(self, config_path: str, checkpoint_path: str, output_dir: str = "detailed_evaluation"):
        self.config = load_config(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Safe config access 
        self.device = self.config.experiment.device if hasattr(self.config, 'experiment') else 'cuda'
        
        # Handle class names - can be dict or object
        try:
            self.class_names = self.config.classes.labels  # If classes is an object
        except AttributeError:
            self.class_names = self.config.classes['labels']  # If classes is a dict
        
        self.model = None
        
        # Results storage
        self.file_results = []
        self.aggregate_results = {}
        
        # Initialize WandB for test evaluation
        if hasattr(self.config, 'logging') and self.config.logging.backend == "wandb":
            self._init_wandb_test_run()
    
    def _init_wandb_test_run(self):
        """Initialize WandB run specifically for test evaluation."""
        # Access config safely
        project_name = getattr(self.config.logging, 'project_name', 'spatial-audio-classification-phd')
        experiment_name = getattr(self.config.experiment, 'name', 'test_evaluation')  
        run_tags = getattr(self.config.logging, 'run_tags', [])
        
        self.wandb_run = wandb.init(
            project=project_name,
            name=f"detailed-test-evaluation-{experiment_name}",
            tags=run_tags + ["detailed_test", "per_file_analysis", "phd_thesis"],
            config={
                **(self.config.dict() if hasattr(self.config, 'dict') else self.config),
                "evaluation_mode": "detailed_per_file",
                "checkpoint": str(self.checkpoint_path)
            },
            notes="Detailed per-file test evaluation with comprehensive analysis for PhD thesis"
        )
    
    def load_model_and_checkpoint(self):
        """Load model and checkpoint with detailed logging."""
        print(f"📦 Loading model from checkpoint: {self.checkpoint_path}")
        
        # Build model
        model_factory = ModelFactory(self.config)
        self.model = model_factory.build().to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_metrics = checkpoint.get('metrics', {})
        else:
            self.model.load_state_dict(checkpoint['model_state'])
            checkpoint_metrics = checkpoint.get('metrics', {})
        
        self.model.eval()
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Checkpoint metrics: {checkpoint_metrics}")
        
        # Log checkpoint info to WandB
        if hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.log({
                "checkpoint/validation_accuracy": checkpoint_metrics.get('val_accuracy', 0),
                "checkpoint/validation_f1": checkpoint_metrics.get('val_macro_f1', 0),
                "checkpoint/validation_loss": checkpoint_metrics.get('loss', 0)
            })
    
    def evaluate_per_file(self) -> List[Dict[str, Any]]:
        """Evaluate model performance on each individual test file."""
        print("🔍 Starting detailed per-file evaluation...")
        
        # Setup test data
        datamodule = DataModule(self.config)
        datamodule.setup()
        test_loader = datamodule.test_dataloader()
        
        if test_loader is None:
            raise ValueError("No test set available!")
        
        print(f"📊 Test set: {len(test_loader.dataset)} files")
        
        file_results = []
        batch_idx = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                # Check if AMP is enabled
                amp_enabled = (hasattr(self.config, 'optimization') and 
                             hasattr(self.config.optimization, 'amp') and 
                             self.config.optimization.amp.enabled)
                
                if amp_enabled:
                    with autocast('cuda'):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)
                
                # Get predictions and confidence scores
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                confidence_scores = torch.max(probabilities, dim=1)[0]
                
                # Process each file in batch
                for i in range(len(images)):
                    # Get file information (this is a simplified version)
                    # In practice, you'd want to track the actual filenames
                    file_idx = batch_idx * test_loader.batch_size + i
                    
                    true_label = targets[i].item()
                    pred_label = predictions[i].item()
                    confidence = confidence_scores[i].item()
                    probs = probabilities[i].cpu().numpy()
                    
                    file_result = {
                        'file_index': file_idx,
                        'file_name': f"test_file_{file_idx:04d}",  # Placeholder - replace with actual filename
                        'true_label': int(true_label),
                        'true_class': self.class_names[true_label],
                        'predicted_label': int(pred_label),
                        'predicted_class': self.class_names[pred_label],
                        'confidence': float(confidence),
                        'is_correct': bool(true_label == pred_label),
                        'class_probabilities': {
                            self.class_names[j]: float(probs[j]) for j in range(len(self.class_names))
                        }
                    }
                    
                    file_results.append(file_result)
                
                batch_idx += 1
                
                if batch_idx % 10 == 0:
                    print(f"   Processed {batch_idx}/{len(test_loader)} batches")
        
        print(f"✅ Per-file evaluation completed: {len(file_results)} files analyzed")
        
        self.file_results = file_results
        return file_results
    
    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive test metrics from per-file results."""
        print("📊 Calculating comprehensive metrics...")
        
        # Extract predictions and ground truth
        y_true = [r['true_label'] for r in self.file_results]
        y_pred = [r['predicted_label'] for r in self.file_results]
        y_probs = np.array([[r['class_probabilities'][cls] for cls in self.class_names] 
                           for r in self.file_results])
        
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
        
        # Confidence analysis
        confidences = [r['confidence'] for r in self.file_results]
        correct_confidences = [r['confidence'] for r in self.file_results if r['is_correct']]
        incorrect_confidences = [r['confidence'] for r in self.file_results if not r['is_correct']]
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Multi-class AUC (if possible)
        try:
            y_true_binarized = label_binarize(y_true, classes=list(range(len(self.class_names))))
            if y_true_binarized.shape[1] > 1:
                auc_macro = roc_auc_score(y_true_binarized, y_probs, average='macro', multi_class='ovr')
                auc_weighted = roc_auc_score(y_true_binarized, y_probs, average='weighted', multi_class='ovr')
            else:
                auc_macro = auc_weighted = 0.0
        except:
            auc_macro = auc_weighted = 0.0
        
        # Compile comprehensive results
        self.aggregate_results = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'macro_f1': float(macro_f1),
                'weighted_f1': float(weighted_f1),
                'macro_precision': float(macro_precision),
                'macro_recall': float(macro_recall),
                'macro_auc': float(auc_macro),
                'weighted_auc': float(auc_weighted)
            },
            'per_class_metrics': {
                self.class_names[i]: {
                    'f1_score': float(per_class_f1[i]),
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i])
                } for i in range(len(self.class_names))
            },
            'confidence_analysis': {
                'mean_confidence': float(np.mean(confidences)),
                'std_confidence': float(np.std(confidences)),
                'mean_correct_confidence': float(np.mean(correct_confidences)) if correct_confidences else 0.0,
                'mean_incorrect_confidence': float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
                'confidence_gap': float(np.mean(correct_confidences) - np.mean(incorrect_confidences)) if correct_confidences and incorrect_confidences else 0.0
            },
            'confusion_matrix': cm.tolist(),
            'error_analysis': self._analyze_errors(),
            'test_set_size': len(self.file_results),
            'correct_predictions': sum(1 for r in self.file_results if r['is_correct']),
            'incorrect_predictions': sum(1 for r in self.file_results if not r['is_correct'])
        }
        
        return self.aggregate_results
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns in detail."""
        errors = [r for r in self.file_results if not r['is_correct']]
        
        if not errors:
            return {'total_errors': 0}
        
        # Error patterns by class
        error_by_true_class = {}
        error_by_pred_class = {}
        confusion_pairs = {}
        
        for error in errors:
            true_cls = error['true_class']
            pred_cls = error['predicted_class']
            
            error_by_true_class[true_cls] = error_by_true_class.get(true_cls, 0) + 1
            error_by_pred_class[pred_cls] = error_by_pred_class.get(pred_cls, 0) + 1
            
            pair = f"{true_cls}→{pred_cls}"
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
        
        # Find most problematic classes
        most_confused_true = max(error_by_true_class.items(), key=lambda x: x[1]) if error_by_true_class else None
        most_confused_pred = max(error_by_pred_class.items(), key=lambda x: x[1]) if error_by_pred_class else None
        most_common_confusion = max(confusion_pairs.items(), key=lambda x: x[1]) if confusion_pairs else None
        
        return {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(self.file_results),
            'errors_by_true_class': error_by_true_class,
            'errors_by_predicted_class': error_by_pred_class,
            'confusion_pairs': confusion_pairs,
            'most_confused_true_class': most_confused_true[0] if most_confused_true else None,
            'most_confused_predicted_class': most_confused_pred[0] if most_confused_pred else None,
            'most_common_confusion': most_common_confusion[0] if most_common_confusion else None,
            'low_confidence_errors': len([e for e in errors if e['confidence'] < 0.5]),
            'high_confidence_errors': len([e for e in errors if e['confidence'] >= 0.8])
        }
    
    def create_visualizations(self):
        """Create comprehensive visualizations for thesis."""
        print("📊 Creating visualizations...")
        
        # Set style for academic plots
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix()
        
        # 2. Per-class performance
        self._plot_per_class_metrics()
        
        # 3. Confidence distribution
        self._plot_confidence_analysis()
        
        # 4. Error analysis
        self._plot_error_analysis()
        
        print("✅ Visualizations created")
    
    def _plot_confusion_matrix(self):
        """Create detailed confusion matrix plots."""
        cm = np.array(self.aggregate_results['confusion_matrix'])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Raw Counts)')
        ax1.set_xlabel('Predicted Class')
        ax1.set_ylabel('True Class')
        
        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)')
        ax2.set_xlabel('Predicted Class')
        ax2.set_ylabel('True Class')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        
        if hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.log({"confusion_matrix": wandb.Image(fig)})
        
        plt.close()
    
    def _plot_per_class_metrics(self):
        """Plot per-class performance metrics."""
        metrics_data = []
        for class_name in self.class_names:
            class_metrics = self.aggregate_results['per_class_metrics'][class_name]
            metrics_data.append([
                class_name,
                class_metrics['precision'],
                class_metrics['recall'],
                class_metrics['f1_score']
            ])
        
        df = pd.DataFrame(metrics_data, columns=['Class', 'Precision', 'Recall', 'F1-Score'])
        
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax.bar(x - width, df['Precision'], width, label='Precision', alpha=0.8)
        ax.bar(x, df['Recall'], width, label='Recall', alpha=0.8)
        ax.bar(x + width, df['F1-Score'], width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for i, (p, r, f1) in enumerate(zip(df['Precision'], df['Recall'], df['F1-Score'])):
            ax.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
        
        if hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.log({"per_class_metrics": wandb.Image(fig)})
        
        plt.close()
    
    def _plot_confidence_analysis(self):
        """Plot confidence score analysis."""
        confidences = [r['confidence'] for r in self.file_results]
        correct_confidences = [r['confidence'] for r in self.file_results if r['is_correct']]
        incorrect_confidences = [r['confidence'] for r in self.file_results if not r['is_correct']]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Confidence distribution
        ax1.hist(correct_confidences, bins=20, alpha=0.7, label='Correct', color='green')
        ax1.hist(incorrect_confidences, bins=20, alpha=0.7, label='Incorrect', color='red')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Score Distribution')
        ax1.legend()
        
        # Box plot
        ax2.boxplot([correct_confidences, incorrect_confidences], 
                   labels=['Correct', 'Incorrect'])
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence Score Comparison')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        
        if hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.log({"confidence_analysis": wandb.Image(fig)})
        
        plt.close()
    
    def _plot_error_analysis(self):
        """Plot error analysis."""
        error_data = self.aggregate_results['error_analysis']
        
        if error_data['total_errors'] == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Errors by true class
        if error_data['errors_by_true_class']:
            classes = list(error_data['errors_by_true_class'].keys())
            counts = list(error_data['errors_by_true_class'].values())
            ax1.bar(classes, counts, alpha=0.7, color='red')
            ax1.set_title('Errors by True Class')
            ax1.set_xlabel('True Class')
            ax1.set_ylabel('Number of Errors')
        
        # Most common confusions
        if error_data['confusion_pairs']:
            pairs = list(error_data['confusion_pairs'].items())
            pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]  # Top 10
            pair_names = [p[0] for p in pairs]
            pair_counts = [p[1] for p in pairs]
            
            ax2.barh(pair_names, pair_counts, alpha=0.7, color='orange')
            ax2.set_title('Top Confusion Pairs')
            ax2.set_xlabel('Number of Confusions')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        
        if hasattr(self, 'wandb_run') and self.wandb_run:
            self.wandb_run.log({"error_analysis": wandb.Image(fig)})
        
        plt.close()
    
    def log_to_wandb(self):
        """Log all results to WandB automatically."""
        if not hasattr(self, 'wandb_run') or not self.wandb_run:
            return
        
        print("📤 Logging results to WandB...")
        
        # Log overall metrics
        overall = self.aggregate_results['overall_metrics']
        self.wandb_run.log({
            'test/accuracy': overall['accuracy'],
            'test/macro_f1': overall['macro_f1'],
            'test/weighted_f1': overall['weighted_f1'],
            'test/macro_precision': overall['macro_precision'],
            'test/macro_recall': overall['macro_recall'],
            'test/macro_auc': overall['macro_auc'],
            'test/weighted_auc': overall['weighted_auc']
        })
        
        # Log per-class metrics
        for class_name, metrics in self.aggregate_results['per_class_metrics'].items():
            self.wandb_run.log({
                f'test_per_class/{class_name}_f1': metrics['f1_score'],
                f'test_per_class/{class_name}_precision': metrics['precision'],
                f'test_per_class/{class_name}_recall': metrics['recall']
            })
        
        # Log confidence analysis
        conf = self.aggregate_results['confidence_analysis']
        self.wandb_run.log({
            'confidence/mean_confidence': conf['mean_confidence'],
            'confidence/confidence_gap': conf['confidence_gap'],
            'confidence/mean_correct': conf['mean_correct_confidence'],
            'confidence/mean_incorrect': conf['mean_incorrect_confidence']
        })
        
        # Log error analysis
        error = self.aggregate_results['error_analysis']
        self.wandb_run.log({
            'errors/total_errors': error['total_errors'],
            'errors/error_rate': error['error_rate'],
            'errors/low_confidence_errors': error['low_confidence_errors'],
            'errors/high_confidence_errors': error['high_confidence_errors']
        })
        
        # Create and log detailed results table
        file_data = []
        for result in self.file_results[:100]:  # Limit to first 100 for WandB table
            file_data.append([
                result['file_name'],
                result['true_class'],
                result['predicted_class'],
                result['confidence'],
                result['is_correct']
            ])
        
        table = wandb.Table(
            data=file_data,
            columns=["File", "True_Class", "Predicted_Class", "Confidence", "Correct"]
        )
        self.wandb_run.log({"detailed_results_sample": table})
        
        print("✅ Results logged to WandB")
    
    def save_results(self):
        """Save all results to files."""
        print("💾 Saving results...")
        
        # Save per-file results
        with open(self.output_dir / 'per_file_results.json', 'w') as f:
            json.dump(self.file_results, f, indent=2)
        
        # Save aggregate results
        with open(self.output_dir / 'aggregate_results.json', 'w') as f:
            json.dump(self.aggregate_results, f, indent=2)
        
        # Save as CSV for easy analysis
        file_df = pd.DataFrame(self.file_results)
        file_df.to_csv(self.output_dir / 'per_file_results.csv', index=False)
        
        # Create comprehensive report
        self._create_comprehensive_report()
        
        print(f"✅ Results saved to: {self.output_dir}")
    
    def _create_comprehensive_report(self):
        """Create comprehensive markdown report."""
        overall = self.aggregate_results['overall_metrics']
        
        report_md = f"""# Detailed Test Evaluation Report

## Overall Performance
- **Test Accuracy**: {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)
- **Macro F1-Score**: {overall['macro_f1']:.4f} ({overall['macro_f1']*100:.2f}%)
- **Weighted F1-Score**: {overall['weighted_f1']:.4f} ({overall['weighted_f1']*100:.2f}%)
- **Macro Precision**: {overall['macro_precision']:.4f} ({overall['macro_precision']*100:.2f}%)
- **Macro Recall**: {overall['macro_recall']:.4f} ({overall['macro_recall']*100:.2f}%)

## Test Set Details
- **Total Files**: {self.aggregate_results['test_set_size']}
- **Correct Predictions**: {self.aggregate_results['correct_predictions']}
- **Incorrect Predictions**: {self.aggregate_results['incorrect_predictions']}

## Per-Class Performance
"""
        
        for class_name, metrics in self.aggregate_results['per_class_metrics'].items():
            report_md += f"""
### Class {class_name}
- **F1-Score**: {metrics['f1_score']:.4f}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
"""
        
        # Add confidence analysis
        conf = self.aggregate_results['confidence_analysis']
        report_md += f"""
## Confidence Analysis
- **Mean Confidence**: {conf['mean_confidence']:.4f}
- **Confidence Gap**: {conf['confidence_gap']:.4f}
- **Mean Correct Confidence**: {conf['mean_correct_confidence']:.4f}
- **Mean Incorrect Confidence**: {conf['mean_incorrect_confidence']:.4f}
"""
        
        # Add error analysis
        error = self.aggregate_results['error_analysis']
        report_md += f"""
## Error Analysis
- **Total Errors**: {error['total_errors']}
- **Error Rate**: {error['error_rate']:.4f} ({error['error_rate']*100:.2f}%)
- **Low Confidence Errors**: {error['low_confidence_errors']}
- **High Confidence Errors**: {error['high_confidence_errors']}
"""
        
        with open(self.output_dir / 'DETAILED_EVALUATION_REPORT.md', 'w') as f:
            f.write(report_md)
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run complete detailed evaluation pipeline."""
        print("🎓 Starting Detailed Test Evaluation for PhD Thesis")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load model
        self.load_model_and_checkpoint()
        
        # Evaluate per file
        self.evaluate_per_file()
        
        # Calculate comprehensive metrics
        self.calculate_comprehensive_metrics()
        
        # Create visualizations
        self.create_visualizations()
        
        # Log to WandB
        self.log_to_wandb()
        
        # Save results
        self.save_results()
        
        eval_time = time.time() - start_time
        
        print("=" * 60)
        print("✅ Detailed evaluation completed!")
        print(f"📊 Test Accuracy: {self.aggregate_results['overall_metrics']['accuracy']:.4f}")
        print(f"📊 Test F1-Score: {self.aggregate_results['overall_metrics']['macro_f1']:.4f}")
        print(f"⏱️ Evaluation time: {eval_time:.1f} seconds")
        print(f"💾 Results saved to: {self.output_dir}")
        
        return self.aggregate_results
    
    def __del__(self):
        """Clean up WandB run."""
        if hasattr(self, 'wandb_run') and self.wandb_run:
            wandb.finish()
