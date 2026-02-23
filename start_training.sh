#!/bin/bash

# Balance 任务训练启动脚本
# 使用方法: bash start_training.sh

echo "=========================================="
echo "  Wheel-Legged Balance Training Launcher"
echo "=========================================="
echo ""

# 检查 conda 环境
if [[ "$CONDA_DEFAULT_ENV" != "gym_env" ]]; then
    echo "⚠️  警告: 当前不在 gym_env 环境中"
    echo "正在激活 gym_env..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate gym_env
fi

echo "✓ Conda 环境: $CONDA_DEFAULT_ENV"
echo ""

# 配置参数
TASK="wheel_legged_vmc_balance"
NUM_ENVS=4096
HEADLESS=""  # 留空表示显示渲染，设置为 "--headless" 表示无头模式

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "训练配置:"
echo "  - 任务: $TASK"
echo "  - 环境数量: $NUM_ENVS"
echo "  - 无头模式: ${HEADLESS:-否}"
echo "  - 恢复训练: ${RESUME:-否}"
echo ""

# 检查是否已有 TensorBoard 在运行
TB_PID=$(pgrep -f "tensorboard.*--logdir=./logs")
if [ -n "$TB_PID" ]; then
    echo "✓ TensorBoard 已在运行 (PID: $TB_PID)"
    echo "  访问地址: http://localhost:6006"
else
    echo "启动 TensorBoard..."
    nohup tensorboard --logdir=./logs --port=6006 > tensorboard.log 2>&1 &
    TB_PID=$!
    echo "✓ TensorBoard 已启动 (PID: $TB_PID)"
    echo "  访问地址: http://localhost:6006"
    echo "  日志文件: tensorboard.log"
fi

echo ""
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""
echo "提示:"
echo "  - 按 'V' 键关闭渲染以提升性能"
echo "  - 按 Ctrl+C 停止训练"
echo "  - 在浏览器打开 http://localhost:6006 查看 TensorBoard"
echo ""

# 启动训练
python wheel_legged_gym/scripts/train.py \
    --task=$TASK \
    --num_envs=$NUM_ENVS \
    $HEADLESS \
    $RESUME

echo ""
echo "训练结束"
