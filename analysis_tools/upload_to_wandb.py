"""Upload historical training results to WandB."""

import re
import json
from pathlib import Path
import wandb
from datetime import datetime

def parse_training_log():
    """Parse training log to extract all metrics."""
    log_file = Path("training.log")
    if not log_file.exists():
        print("❌ training.log not found")
        return None
        
    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    epoch_times = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Epoch" in line and "train:" in line and "val:" in line:
                # Extract epoch number
                epoch_match = re.search(r'Epoch (\d+)/\d+', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    epochs.append(epoch)
                
                # Extract train loss
                train_loss_match = re.search(r"train:\{'loss': ([\d.]+)", line)
                if train_loss_match:
                    train_losses.append(float(train_loss_match.group(1)))
                
                # Extract val metrics
                val_loss_match = re.search(r"'loss': ([\d.]+)", line.split("val:")[1])
                val_acc_match = re.search(r"'val_accuracy': ([\d.]+)", line)
                val_f1_match = re.search(r"'val_macro_f1': ([\d.]+)", line)
                time_match = re.search(r"time:([\d.]+)s", line)
                
                if val_loss_match:
                    val_losses.append(float(val_loss_match.group(1)))
                if val_acc_match:
                    val_accuracies.append(float(val_acc_match.group(1)))
                if val_f1_match:
                    val_f1_scores.append(float(val_f1_match.group(1)))
                if time_match:
                    epoch_times.append(float(time_match.group(1)))
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'epoch_times': epoch_times
    }

def upload_to_wandb():
    """Upload historical data to WandB."""
    print("🚀 Uploading PhD Thesis Results to WandB...")
    
    # Parse training data
    data = parse_training_log()
    if not data:
        return
    
    # Initialize WandB
    run = wandb.init(
        project="spatial-audio-classification-phd",
        name="swin-base-80epochs-rtx4070ti-historical",
        tags=["spatial_audio", "swin_transformer", "phd_thesis", "rtx4070ti", "historical_upload"],
        notes="Historical upload of completed PhD thesis experiment results"
    )
    
    # Log system configuration
    config = {
        "model": "swin_base_patch4_window12_384",
        "input_resolution": "384x384",
        "batch_size": 16,
        "epochs_completed": len(data['epochs']),
        "gpu": "RTX_4070_Ti",
        "mixed_precision": True,
        "optimizer": "AdamW",
        "scheduler": "cosine_warmup",
        "dataset_split": "70/20/10",
        "classes": 4,
        "early_stopping": True,
        "patience": 15
    }
    
    wandb.config.update(config)
    
    # Upload metrics for each epoch
    print(f"📊 Uploading {len(data['epochs'])} epochs of data...")
    
    for i, epoch in enumerate(data['epochs']):
        metrics = {
            'epoch': epoch,
            'train/loss': data['train_losses'][i] if i < len(data['train_losses']) else None,
            'val/loss': data['val_losses'][i] if i < len(data['val_losses']) else None,
            'val/accuracy': data['val_accuracies'][i] if i < len(data['val_accuracies']) else None,
            'val/macro_f1': data['val_f1_scores'][i] if i < len(data['val_f1_scores']) else None,
            'epoch_time_seconds': data['epoch_times'][i] if i < len(data['epoch_times']) else None,
        }
        
        # Remove None values
        metrics = {k: v for k, v in metrics.items() if v is not None}
        
        wandb.log(metrics, step=epoch)
    
    # Log final summary
    if data['val_accuracies']:
        best_accuracy = max(data['val_accuracies'])
        best_f1 = max(data['val_f1_scores']) if data['val_f1_scores'] else 0
        final_accuracy = data['val_accuracies'][-1]
        final_f1 = data['val_f1_scores'][-1] if data['val_f1_scores'] else 0
        
        summary = {
            "best_accuracy": best_accuracy,
            "best_f1_score": best_f1,
            "final_accuracy": final_accuracy,
            "final_f1_score": final_f1,
            "total_epochs": len(data['epochs']),
            "avg_epoch_time": sum(data['epoch_times']) / len(data['epoch_times']) if data['epoch_times'] else 0,
            "total_training_time_hours": sum(data['epoch_times']) / 3600 if data['epoch_times'] else 0
        }
        
        wandb.summary.update(summary)
    
    # Create and log training curves table
    table_data = []
    for i, epoch in enumerate(data['epochs']):
        row = [
            epoch,
            data['train_losses'][i] if i < len(data['train_losses']) else None,
            data['val_losses'][i] if i < len(data['val_losses']) else None,
            data['val_accuracies'][i] if i < len(data['val_accuracies']) else None,
            data['val_f1_scores'][i] if i < len(data['val_f1_scores']) else None,
            data['epoch_times'][i] if i < len(data['epoch_times']) else None
        ]
        table_data.append(row)
    
    table = wandb.Table(
        data=table_data,
        columns=["Epoch", "Train_Loss", "Val_Loss", "Val_Accuracy", "Val_F1", "Epoch_Time"]
    )
    
    wandb.log({"training_results_table": table})
    
    print("✅ Upload completed successfully!")
    print(f"🔗 View results at: {run.url}")
    
    wandb.finish()

if __name__ == "__main__":
    upload_to_wandb()
