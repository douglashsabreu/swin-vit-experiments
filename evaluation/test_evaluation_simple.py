"""Simple test set evaluation for PhD thesis."""

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import json

from src.config import load_config
from src.factories.model_factory import ModelFactory
from src.data.datamodule import DataModule

def main():
    print("🎓 PhD Thesis - Test Set Evaluation")
    print("="*50)
    
    # Load config and find best checkpoint
    config = load_config("experiments/spatial_experiment.yaml")
    
    logs_dir = Path("logs")
    checkpoints = list(logs_dir.glob("best_checkpoint_*.pt"))
    best_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
    
    print(f"📦 Using checkpoint: {best_checkpoint}")
    
    # Load model
    model_factory = ModelFactory(config)
    model = model_factory.build()
    
    checkpoint = torch.load(best_checkpoint, map_location=config.experiment.device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(config.experiment.device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Validation metrics: Acc={checkpoint['metrics']['val_accuracy']:.4f}, F1={checkpoint['metrics']['val_macro_f1']:.4f}")
    
    # Setup data
    datamodule = DataModule(config)
    datamodule.setup()
    test_loader = datamodule.test_dataloader()
    
    print(f"📊 Test set size: {len(test_loader.dataset)} samples")
    
    # Evaluate
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            images = images.to(config.experiment.device, non_blocking=True)
            targets = targets.to(config.experiment.device, non_blocking=True)
            
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"   Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Calculate metrics
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    
    cm = confusion_matrix(y_true, y_pred)
    class_names = ["100", "200", "510", "514"]  # From config
    
    # Print results
    print("\n" + "="*60)
    print("🎓 FINAL TEST SET RESULTS")
    print("="*60)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   • Test Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Macro F1-Score:     {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"   • Weighted F1-Score:  {weighted_f1:.4f} ({weighted_f1*100:.2f}%)")
    print(f"   • Macro Precision:    {macro_precision:.4f} ({macro_precision*100:.2f}%)")
    print(f"   • Macro Recall:       {macro_recall:.4f} ({macro_recall*100:.2f}%)")
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    per_class_precision = precision_score(y_true, y_pred, average=None)
    per_class_recall = recall_score(y_true, y_pred, average=None)
    
    print(f"\n🎯 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        print(f"   • Class {class_name}:")
        print(f"     - F1:        {per_class_f1[i]:.4f} ({per_class_f1[i]*100:.2f}%)")
        print(f"     - Precision: {per_class_precision[i]:.4f} ({per_class_precision[i]*100:.2f}%)")
        print(f"     - Recall:    {per_class_recall[i]:.4f} ({per_class_recall[i]*100:.2f}%)")
    
    print(f"\n📈 CONFUSION MATRIX:")
    print("     Predicted →")
    print("   ", end="")
    for name in class_names:
        print(f"{name:>6}", end="")
    print()
    
    for i, true_name in enumerate(class_names):
        print(f"{true_name:>3}|", end="")
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>6}", end="")
        print()
    
    # Validation vs Test comparison
    val_acc = checkpoint['metrics']['val_accuracy']
    val_f1 = checkpoint['metrics']['val_macro_f1']
    
    print(f"\n🔍 GENERALIZATION ANALYSIS:")
    print(f"   • Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"   • Test Accuracy:       {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   • Accuracy Drop:       {(val_acc-accuracy)*100:.2f} percentage points")
    print(f"   • Validation F1:       {val_f1:.4f} ({val_f1*100:.2f}%)")
    print(f"   • Test F1:             {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"   • F1 Drop:             {(val_f1-macro_f1)*100:.2f} percentage points")
    
    generalization = "Excellent" if abs(val_acc - accuracy) < 0.02 else "Good" if abs(val_acc - accuracy) < 0.05 else "Moderate"
    print(f"   • Generalization:      {generalization}")
    
    print(f"\n🏆 ACADEMIC SIGNIFICANCE:")
    print(f"   • Model: Swin Vision Transformer Base")
    print(f"   • Task: 4-class spatial audio classification")
    print(f"   • Test Set: {len(y_true)} samples (10% of dataset)")
    performance_level = "Excellent" if accuracy > 0.9 else "Good" if accuracy > 0.8 else "Moderate"
    print(f"   • Performance Level: {performance_level}")
    print(f"   • Cross-Modal Learning: Vision Transformer → Audio Domain")
    
    # Save results
    results = {
        'test_accuracy': float(accuracy),
        'test_macro_f1': float(macro_f1),
        'test_weighted_f1': float(weighted_f1),
        'test_macro_precision': float(macro_precision),
        'test_macro_recall': float(macro_recall),
        'validation_accuracy': float(val_acc),
        'validation_f1': float(val_f1),
        'accuracy_drop': float(val_acc - accuracy),
        'f1_drop': float(val_f1 - macro_f1),
        'test_samples': int(len(y_true)),
        'confusion_matrix': cm.tolist(),
        'per_class_f1': per_class_f1.tolist(),
        'per_class_precision': per_class_precision.tolist(),
        'per_class_recall': per_class_recall.tolist(),
        'class_names': class_names
    }
    
    with open('test_results_final.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: test_results_final.json")
    print("="*60)
    print("✅ Test evaluation completed successfully!")
    print("🎓 Results ready for PhD thesis documentation")

if __name__ == "__main__":
    main()
