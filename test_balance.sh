#!/bin/bash
# Quick test script for balance policy

echo "=========================================="
echo "  Balance Policy Test"
echo "=========================================="
echo ""

# 检查是否有训练好的模型
if [ ! -d "logs/wheel_legged_vmc_balance" ]; then
    echo "❌ No training logs found!"
    echo "Please train the model first:"
    echo "  python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096"
    exit 1
fi

# 查找最新的模型
LATEST_LOG=$(ls -td logs/wheel_legged_vmc_balance/*/ | head -1)
if [ -z "$LATEST_LOG" ]; then
    echo "❌ No model checkpoint found!"
    exit 1
fi

echo "✅ Found model: $LATEST_LOG"
echo ""

# 激活环境并运行
echo "🚀 Starting balance test..."
echo ""
echo "Controls:"
echo "  [C] - Start control (enable policy)"
echo "  [R] - Reset environment"
echo "  [S] - Toggle statistics display"
echo "  [P] - Pause/Resume"
echo "  [ESC] - Exit"
echo ""
echo "=========================================="
echo ""

conda run -n gym_env python wheel_legged_gym/scripts/play_balance.py --task=wheel_legged_vmc_balance
