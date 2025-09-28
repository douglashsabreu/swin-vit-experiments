#!/bin/bash

# PhD Thesis Training Monitor
# Use this script to check training progress remotely

cd /home/home/Documents/PhD_sept_2025/swin-vit-experiments

echo "🎓 PhD Thesis Training Monitor"
echo "=============================="
echo "📅 Current time: $(date)"
echo ""

# Check if training is running
TRAIN_PID=$(ps aux | grep "python main.py" | grep -v grep | awk '{print $2}' | head -1)

if [ -z "$TRAIN_PID" ]; then
    echo "❌ Training process not found!"
    echo "💡 Start training with: bash scripts/start_training.sh"
    exit 1
else
    echo "✅ Training is running (PID: $TRAIN_PID)"
    
    # Show process info
    echo ""
    echo "📊 Process Information:"
    ps aux | grep $TRAIN_PID | grep -v grep
    
    # Show GPU usage
    echo ""
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits
    
    # Show latest logs (last 10 lines)
    echo ""
    echo "📝 Latest Training Logs:"
    echo "------------------------"
    if [ -f "training.log" ]; then
        tail -10 training.log
    else
        echo "No training.log found"
    fi
    
    # Show checkpoints
    echo ""
    echo "💾 Saved Checkpoints:"
    ls -la logs/best_checkpoint_*.pt 2>/dev/null | tail -5 || echo "No checkpoints found yet"
    
    echo ""
    echo "🔗 Monitoring Links:"
    echo "• WandB: https://wandb.ai/your-username/spatial-audio-classification-phd"
    echo "• Live logs: tail -f training.log"
    echo "• GPU monitor: watch -n 2 nvidia-smi"
fi

