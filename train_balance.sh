#!/bin/bash
# 训练 Balance 网络 - 从任意姿态恢复到 Flat 初始条件

echo "=========================================="
echo "  Balance Network Training"
echo "=========================================="
echo ""
echo "目标: 训练机器人从任意姿态恢复到 Flat 初始条件"
echo "  - 初始化范围: ±45.8° 姿态, ±0.5 m/s 速度"
echo "  - 目标状态: 高度 0.20m, 姿态 ±5.7°, 速度 < 0.2"
echo ""
echo "关键指标 (TensorBoard):"
echo "  - Train/rew_reach_flat_target > 80"
echo "  - Train/rew_pitch_angle → 0"
echo "  - Train/rew_roll_angle → 0"
echo "  - Train/mean_episode_length → max"
echo ""
echo "训练配置:"
echo "  - Environments: 4096"
echo "  - Max iterations: 2000"
echo "  - action_scale_theta: 1.0 (允许大幅度恢复)"
echo ""
echo "开始训练..."
echo ""

python wheel_legged_gym/scripts/train.py \
    --task=wheel_legged_vmc_balance \
    --num_envs=4096

echo ""
echo "=========================================="
echo "  训练完成!"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 查看训练曲线: tensorboard --logdir=logs/wheel_legged_vmc_balance"
echo "  2. 测试两阶段系统: ./test_two_stage.sh"
echo ""
