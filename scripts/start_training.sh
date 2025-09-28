#!/bin/bash

# PhD Thesis Training Script - Remote Monitoring
# Starts training with comprehensive logging and monitoring

cd /home/home/Documents/PhD_sept_2025/swin-vit-experiments

echo "🎓 Starting PhD Thesis Training - Spatial Audio Classification"
echo "📅 Started at: $(date)"
echo "🖥️  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "💾 VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
echo ""

# Activate virtual environment
source .venv/bin/activate

# Start training with nohup
nohup python main.py --config experiments/spatial_experiment.yaml > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Get process ID
TRAIN_PID=$!
echo "🚀 Training started with PID: $TRAIN_PID"
echo $TRAIN_PID > logs/training.pid

echo ""
echo "📊 Monitoring options:"
echo "1. WandB Dashboard: https://wandb.ai/your-username/spatial-audio-classification-phd"
echo "2. Local logs: tail -f logs/training_*.log"
echo "3. GPU usage: watch -n 2 nvidia-smi"
echo "4. Process status: ps aux | grep $TRAIN_PID"
echo ""
echo "✅ Training is now running in background!"
echo "💡 You can safely disconnect SSH - training will continue"

