# TensorBoard 监控指南

## 快速启动

### 方法 1: 使用启动脚本（推荐）
```bash
conda activate gym_env
bash start_training.sh
```

这个脚本会自动：
- 激活 conda 环境
- 启动 TensorBoard
- 开始训练

### 方法 2: 手动启动

**终端 1 - 启动 TensorBoard**:
```bash
conda activate gym_env
tensorboard --logdir=./logs --port=6006
```

**终端 2 - 启动训练**:
```bash
conda activate gym_env
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 方法 3: 后台运行 TensorBoard
```bash
conda activate gym_env
nohup tensorboard --logdir=./logs --port=6006 > tensorboard.log 2>&1 &
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance
```

## 访问 TensorBoard

### 本地访问
在浏览器打开: `http://localhost:6006`

### 远程服务器访问
在本地电脑执行端口转发：
```bash
ssh -L 6006:localhost:6006 username@server_ip
```
然后在本地浏览器访问: `http://localhost:6006`

## 关键监控指标

### 1. 总体性能指标

| 指标 | 说明 | 期望趋势 |
|------|------|----------|
| `Train/mean_reward` | 平均总奖励 | 📈 持续上升 |
| `Train/mean_episode_length` | 平均 episode 长度 | 📈 逐渐增加 |

### 2. Balance 任务专属指标

#### 核心奖励（应该增加）
- ✅ `Train/rew_base_height`: 高度跟踪（目标: > 10）
- ✅ `Train/rew_upright_bonus`: 直立奖励（目标: > 3）
- ✅ `Train/rew_recovery_speed`: 恢复速度 ⭐（目标: > 1.5）
- ✅ `Train/rew_energy_efficiency`: 能量效率 ⭐（目标: > 0.5）

#### 惩罚项（应该减少/接近 0）
- ❌ `Train/rew_orientation`: 姿态惩罚（目标: > -5）
- ❌ `Train/rew_torques`: 力矩惩罚（目标: > -0.5）
- ❌ `Train/rew_torque_over_limit`: 超限惩罚（目标: ≈ 0）
- ❌ `Train/rew_termination`: 终止惩罚（目标: ≈ 0）

### 3. 训练健康度指标

| 指标 | 健康范围 | 说明 |
|------|----------|------|
| `Loss/value_function` | 0.1 - 10 | 价值函数损失 |
| `Loss/surrogate` | -0.1 - 0.1 | 策略损失 |
| `Policy/mean_noise_std` | 0.5 - 1.5 | 探索噪声 |
| `Policy/learning_rate` | 1e-5 - 1e-3 | 学习率 |

## 训练阶段判断

### 阶段 1: 初期探索（0-500 iterations）
- `mean_reward`: -50 到 0
- 机器人在学习基本的平衡控制
- 可能频繁倒地

**正常现象**:
- 奖励波动大
- episode 长度短
- 各项惩罚都很高

### 阶段 2: 学习平衡（500-1500 iterations）
- `mean_reward`: 0 到 50
- 机器人开始能从小角度恢复
- `rew_upright_bonus` 开始增加

**正常现象**:
- 奖励开始稳定上升
- `rew_orientation` 惩罚减少
- `rew_torque_over_limit` 减少

### 阶段 3: 优化策略（1500-3000 iterations）
- `mean_reward`: 50 到 100+
- 机器人能从大角度恢复
- 开始优化力矩使用

**期望现象**:
- `rew_recovery_speed` 增加
- `rew_energy_efficiency` 增加
- `rew_torques` 惩罚减少

## 常见问题诊断

### 问题 1: 奖励不增长
**症状**: `mean_reward` 在 -50 附近徘徊

**可能原因**:
- 初始化难度太大
- 学习率过高/过低
- 奖励权重不合理

**解决方案**:
```python
# 降低初始化难度
roll = [-1.57, 1.57]  # 改为 ±90°
pitch = [-1.57, 1.57]

# 调整学习率
learning_rate = 1e-4  # 降低学习率
```

### 问题 2: 训练不稳定
**症状**: `mean_reward` 剧烈波动

**可能原因**:
- batch size 太小
- 学习率太高
- 奖励 clip 范围不合理

**解决方案**:
```python
# 增加 batch size
num_mini_batches = 8  # 增加到 8

# 降低学习率
learning_rate = 5e-5

# 调整奖励 clip
clip_single_reward = 5.0  # 增加到 5.0
```

### 问题 3: 力矩过大
**症状**: `rew_torque_over_limit` 持续为负

**可能原因**:
- PD 增益过大
- 力矩惩罚权重太小

**解决方案**:
```python
# 降低 PD 增益
kp_theta = 8.0  # 从 10.0 降低
kp_l0 = 600.0   # 从 800.0 降低

# 增加力矩惩罚
torques = -5e-4  # 从 -2e-4 增加
torque_over_limit = -100.0  # 从 -50.0 增加
```

### 问题 4: 恢复速度慢
**症状**: `rew_recovery_speed` 很低

**可能原因**:
- 能量效率权重太高
- 恢复速度权重太低

**解决方案**:
```python
# 调整权重平衡
recovery_speed = 3.0  # 从 2.0 增加
energy_efficiency = 0.5  # 从 1.0 降低
```

## TensorBoard 高级功能

### 1. 比较多次训练
```bash
# 同时监控多个实验
tensorboard --logdir_spec=exp1:./logs/run1,exp2:./logs/run2 --port=6006
```

### 2. 平滑曲线
在 TensorBoard 界面左侧调整 "Smoothing" 滑块（推荐 0.6-0.8）

### 3. 导出数据
点击右上角的下载按钮，可以导出 CSV 格式的数据

### 4. 查看分布
切换到 "Distributions" 标签查看：
- 动作分布
- 观测分布
- 奖励分布

## 停止 TensorBoard

### 查找进程
```bash
ps aux | grep tensorboard
```

### 停止进程
```bash
# 方法 1: 使用 pkill
pkill -f tensorboard

# 方法 2: 使用 kill
kill <PID>
```

## 训练完成后的分析

### 1. 查看最终性能
在 TensorBoard 中查看最后 100 个 iterations 的平均值

### 2. 导出关键指标
```python
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

ea = event_accumulator.EventAccumulator('logs/wheel_legged_vmc_balance/latest')
ea.Reload()

# 导出奖励数据
rewards = ea.Scalars('Train/mean_reward')
df = pd.DataFrame(rewards)
df.to_csv('training_rewards.csv')
```

### 3. 生成训练报告
记录以下关键信息：
- 最终平均奖励
- 收敛所需迭代次数
- 各项奖励的最终值
- 训练总时长

## 实时监控脚本

创建一个简单的监控脚本：
```bash
#!/bin/bash
# monitor_training.sh

while true; do
    clear
    echo "=== Training Monitor ==="
    echo "Time: $(date)"
    echo ""

    # 显示最新的日志
    tail -n 20 logs/wheel_legged_vmc_balance/*/train.log | grep "mean_reward"

    sleep 10
done
```

使用方法：
```bash
bash monitor_training.sh
```
