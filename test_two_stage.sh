#!/bin/bash
# 两阶段控制训练和测试脚本

echo "=========================================="
echo "  Two-Stage Control: Balance → Flat"
echo "=========================================="
echo ""

# 检查 Flat 模型
if [ ! -d "logs/wheel_legged_vmc_flat" ]; then
    echo "❌ Flat policy not found!"
    echo "Please train Flat policy first:"
    echo "  python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat --num_envs=4096"
    exit 1
fi

echo "✅ Flat policy found"
echo ""

# 检查 Balance 模型
if [ ! -d "logs/wheel_legged_vmc_balance" ]; then
    echo "⚠️  Balance policy not found. Training now..."
    echo ""
    python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
else
    echo "✅ Balance policy found"
fi

echo ""
echo "=========================================="
echo "  Starting Two-Stage Control Test"
echo "=========================================="
echo ""
echo "Controls:"
echo "  [C] - Start control (Balance stage)"
echo "  [M] - Manual stage switch"
echo "  [R] - Reset"
echo "  [S] - Toggle statistics"
echo "  [ESC] - Exit"
echo ""
echo "Automatic switching:"
echo "  Balance → Flat when conditions met"
echo "=========================================="
echo ""

conda run -n gym_env python wheel_legged_gym/scripts/play_two_stage.py --task=wheel_legged_vmc_balance
