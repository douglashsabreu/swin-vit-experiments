"""Upload test results to WandB for PhD thesis documentation."""

import json
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def upload_test_results():
    """Upload test results to WandB."""
    print("🎓 Uploading Test Results to WandB...")
    
    # Load test results
    if not Path("test_results_final.json").exists():
        print("❌ test_results_final.json not found. Run test evaluation first.")
        return
    
    with open("test_results_final.json", 'r') as f:
        test_results = json.load(f)
    
    # Initialize WandB run
    run = wandb.init(
        project="spatial-audio-classification-phd",
        name="test-set-evaluation-final",
        tags=["test_evaluation", "final_results", "phd_thesis"],
        notes="Final test set evaluation results for PhD thesis"
    )
    
    # Log test metrics
    test_metrics = {
        "test/accuracy": test_results["test_accuracy"],
        "test/macro_f1": test_results["test_macro_f1"],
        "test/weighted_f1": test_results["test_weighted_f1"],
        "test/macro_precision": test_results["test_macro_precision"],
        "test/macro_recall": test_results["test_macro_recall"],
        "generalization/accuracy_drop": test_results["accuracy_drop"],
        "generalization/f1_drop": test_results["f1_drop"],
        "dataset/test_samples": test_results["test_samples"]
    }
    
    wandb.log(test_metrics)
    
    # Log per-class metrics
    class_names = test_results["class_names"]
    for i, class_name in enumerate(class_names):
        wandb.log({
            f"test_per_class/f1_class_{class_name}": test_results["per_class_f1"][i],
            f"test_per_class/precision_class_{class_name}": test_results["per_class_precision"][i],
            f"test_per_class/recall_class_{class_name}": test_results["per_class_recall"][i]
        })
    
    # Create and log confusion matrix visualization
    cm = np.array(test_results["confusion_matrix"])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Set Confusion Matrix\nSwin Vision Transformer - Spatial Audio Classification')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    
    wandb.log({"test_confusion_matrix": wandb.Image(plt)})
    plt.close()
    
    # Create normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Set Confusion Matrix (Normalized)\nSwin Vision Transformer - Spatial Audio Classification')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.tight_layout()
    
    wandb.log({"test_confusion_matrix_normalized": wandb.Image(plt)})
    plt.close()
    
    # Create per-class performance bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    ax.bar(x - width, test_results["per_class_precision"], width, label='Precision', alpha=0.8)
    ax.bar(x, test_results["per_class_recall"], width, label='Recall', alpha=0.8)
    ax.bar(x + width, test_results["per_class_f1"], width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Test Set Per-Class Performance\nSwin Vision Transformer - Spatial Audio Classification')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for i, (p, r, f1) in enumerate(zip(test_results["per_class_precision"], 
                                      test_results["per_class_recall"], 
                                      test_results["per_class_f1"])):
        ax.text(i - width, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width, f1 + 0.01, f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    wandb.log({"test_per_class_metrics": wandb.Image(plt)})
    plt.close()
    
    # Create generalization comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    categories = ['Validation', 'Test']
    accuracy_values = [test_results["validation_accuracy"], test_results["test_accuracy"]]
    f1_values = [test_results["validation_f1"], test_results["test_macro_f1"]]
    
    bars1 = ax1.bar(categories, accuracy_values, color=['skyblue', 'lightcoral'], alpha=0.7)
    ax1.set_title('Validation vs Test Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.9, 1.0)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, accuracy_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # F1 comparison
    bars2 = ax2.bar(categories, f1_values, color=['skyblue', 'lightcoral'], alpha=0.7)
    ax2.set_title('Validation vs Test F1-Score')
    ax2.set_ylabel('F1-Score')
    ax2.set_ylim(0.9, 1.0)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, f1_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Generalization Analysis: Validation vs Test Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    wandb.log({"generalization_comparison": wandb.Image(plt)})
    plt.close()
    
    # Create summary table
    summary_data = [
        ["Test Accuracy", f"{test_results['test_accuracy']:.4f} ({test_results['test_accuracy']*100:.2f}%)"],
        ["Test Macro F1", f"{test_results['test_macro_f1']:.4f} ({test_results['test_macro_f1']*100:.2f}%)"],
        ["Test Weighted F1", f"{test_results['test_weighted_f1']:.4f} ({test_results['test_weighted_f1']*100:.2f}%)"],
        ["Test Precision", f"{test_results['test_macro_precision']:.4f} ({test_results['test_macro_precision']*100:.2f}%)"],
        ["Test Recall", f"{test_results['test_macro_recall']:.4f} ({test_results['test_macro_recall']*100:.2f}%)"],
        ["Accuracy Drop", f"{test_results['accuracy_drop']:.4f} ({test_results['accuracy_drop']*100:.2f} pp)"],
        ["F1 Drop", f"{test_results['f1_drop']:.4f} ({test_results['f1_drop']*100:.2f} pp)"],
        ["Test Samples", str(test_results['test_samples'])],
        ["Generalization", "Excellent" if abs(test_results['accuracy_drop']) < 0.02 else "Good"]
    ]
    
    table = wandb.Table(data=summary_data, columns=["Metric", "Value"])
    wandb.log({"test_results_summary": table})
    
    # Log final summary to wandb.summary for easy access
    wandb.summary.update({
        "final_test_accuracy": test_results["test_accuracy"],
        "final_test_f1": test_results["test_macro_f1"],
        "generalization_quality": "Excellent" if abs(test_results['accuracy_drop']) < 0.02 else "Good",
        "best_performing_class": class_names[np.argmax(test_results["per_class_f1"])],
        "worst_performing_class": class_names[np.argmin(test_results["per_class_f1"])],
        "test_samples": test_results["test_samples"]
    })
    
    print("✅ Test results uploaded successfully!")
    print(f"🔗 View results at: {run.url}")
    
    wandb.finish()

if __name__ == "__main__":
    upload_test_results()
