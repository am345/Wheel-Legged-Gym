#!/bin/bash
# 重新训练 Balance 任务 - 应用奖励修复后

set -e

echo "=========================================="
echo "Balance 任务重新训练脚本"
echo "=========================================="
echo ""

# 检查 conda 环境
if ! conda info --envs | grep -q "gym_env"; then
    echo "❌ 错误: conda 环境 'gym_env' 不存在"
    echo "请先创建环境: conda create -n gym_env python=3.8"
    exit 1
fi

echo "✅ 找到 conda 环境: gym_env"
echo ""

# 询问是否清理旧日志
read -p "是否清理旧的训练日志? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  清理旧日志..."
    rm -rf logs/wheel_legged_vmc_balance/Feb23_*
    echo "✅ 清理完成"
else
    echo "⏭️  保留旧日志"
fi
echo ""

# 显示修改摘要
echo "=========================================="
echo "已应用的修改:"
echo "=========================================="
echo "1. ✅ orientation 惩罚: -40.0 → -80.0"
echo "2. ✅ lin_vel_z 惩罚: -2.0 → -4.0"
echo "3. ✅ ang_vel_xy 惩罚: -1.0 → -2.0"
echo "4. ✅ stand_still 奖励: 3.0 → 5.0"
echo "5. ✅ base_height 衰减: 0.01 → 0.05"
echo "6. ✅ 终止条件: -0.1 → -0.5 (60度)"
echo "7. ✅ upright_bonus 条件放宽"
echo ""

# 显示训练参数
echo "=========================================="
echo "训练参数:"
echo "=========================================="
echo "任务: wheel_legged_vmc_balance"
echo "环境数: 4096"
echo "最大迭代: 1000"
echo "学习率: 1e-4"
echo ""

# 询问是否开始训练
read -p "开始训练? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 取消训练"
    exit 0
fi

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""
echo "💡 提示:"
echo "  - 按 'v' 关闭渲染以提高性能"
echo "  - 使用 TensorBoard 监控: tensorboard --logdir=logs/wheel_legged_vmc_balance --port=6006"
echo "  - 关键指标: mean_episode_length (期望 > 3500)"
echo ""
echo "按 Ctrl+C 停止训练"
echo ""

# 激活环境并开始训练
conda run -n gym_env python wheel_legged_gym/scripts/train.py \
    --task=wheel_legged_vmc_balance \
    --num_envs=4096 \
    --headless

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 查看 TensorBoard: tensorboard --logdir=logs/wheel_legged_vmc_balance --port=6006"
echo "2. 测试策略: python wheel_legged_gym/scripts/play_balance.py"
echo ""
