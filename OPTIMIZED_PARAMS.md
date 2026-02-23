# 优化的 Balance 参数配置

## 参数调整理由

基于双轮足机器人平衡控制的物理原理和强化学习经验，我对参数进行了以下优化：

## 1. PD 控制参数

### 降低 PD 增益（提高稳定性）

```python
# 原始参数（过于激进）
kp_theta = 10.0
kd_theta = 5.0
kp_l0 = 800.0
kd_l0 = 7.0

# 优化参数（更稳定）
kp_theta = 8.0   # ↓ 20% - 减少角度控制的过冲
kd_theta = 4.0   # ↓ 20% - 减少阻尼振荡
kp_l0 = 600.0    # ↓ 25% - 减少腿长控制的过冲
kd_l0 = 6.0      # ↓ 14% - 减少阻尼振荡
```

**原理**:
- 过高的 P 增益会导致过冲和振荡
- 过高的 D 增益会放大噪声
- 对于平衡任务，稳定性比响应速度更重要

### 降低动作幅度（更平滑的控制）

```python
# 原始参数
action_scale_theta = 0.5
action_scale_l0 = 0.07
action_scale_vel = 10.0

# 优化参数
action_scale_theta = 0.3  # ↓ 40% - 更小的角度变化
action_scale_l0 = 0.05    # ↓ 29% - 更小的腿长变化
action_scale_vel = 8.0    # ↓ 20% - 更小的轮速变化
```

**原理**:
- 从直立姿态开始，不需要大幅度动作
- 小动作更容易学习和控制
- 减少动作空间，加快收敛

### 增加轮子阻尼

```python
damping = {"wheel": 0.15}  # 从 0.1 增加到 0.15
```

**原理**:
- 增加阻尼可以减少轮子的自由滚动
- 提高系统稳定性

## 2. 奖励函数权重

### 增加核心奖励权重

```python
# 原始权重
base_height = 30.0
orientation = -30.0
upright_bonus = 15.0

# 优化权重
base_height = 40.0    # ↑ 33% - 高度是最重要的
orientation = -40.0   # ↑ 33% - 姿态是最重要的
upright_bonus = 20.0  # ↑ 33% - 增加直立激励
```

**原理**:
- 高度和姿态是平衡的核心
- 更高的权重可以加快学习这些关键行为
- 直立奖励提供明确的目标信号

### 增加静止约束

```python
# 原始权重
lin_vel_z = -1.0
ang_vel_xy = -0.5
stand_still = 2.0

# 优化权重
lin_vel_z = -2.0      # ↑ 100% - 严格限制 z 方向运动
ang_vel_xy = -1.0     # ↑ 100% - 严格限制 roll/pitch 晃动
stand_still = 3.0     # ↑ 50% - 增加静止奖励
```

**原理**:
- 平衡任务的目标是静止，不是运动
- 更强的静止约束可以防止机器人乱动
- 减少不必要的探索

### 增加力矩惩罚

```python
torques = -5e-5  # 从 -1e-5 增加 5 倍
```

**原理**:
- 鼓励使用更小的力矩
- 提高能量效率
- 减少机械磨损

## 3. 训练参数

### 提高学习率

```python
learning_rate = 1e-4  # 从 5e-5 提高到 1e-4
```

**原理**:
- 任务简单（从直立开始），可以用更高的学习率
- 加快收敛速度
- 5e-5 太保守，会导致训练过慢

### 优化 PPO 参数

```python
num_learning_epochs = 5   # 增加学习轮数
num_mini_batches = 8      # 增加 mini batch
entropy_coef = 0.01       # 增加探索
```

**原理**:
- 更多的学习轮数可以更充分地利用数据
- 更多的 mini batch 可以提高训练稳定性
- 适当的熵系数鼓励探索，避免过早收敛到局部最优

### 缩短 Episode 长度

```python
episode_length_s = 20      # 从 30 降低到 20
fail_to_terminal_time_s = 5.0  # 从 10 降低到 5
```

**原理**:
- 平衡任务不需要长时间 episode
- 更短的 episode 可以加快训练速度
- 更快终止失败的尝试，提高数据效率

## 4. 预期训练曲线

### 阶段 1: 初期探索（0-100 iterations）
- `mean_reward`: -10 → 10
- 机器人学习基本的高度和姿态控制
- 可能会频繁倒地

### 阶段 2: 快速学习（100-500 iterations）
- `mean_reward`: 10 → 40
- 机器人学会保持直立
- `rew_upright_bonus` 开始增加

### 阶段 3: 精细调优（500-1000 iterations）
- `mean_reward`: 40 → 60+
- 机器人学会稳定平衡
- 所有奖励指标趋于稳定

### 阶段 4: 收敛（1000+ iterations）
- `mean_reward`: 稳定在 60-80
- 机器人可以长时间保持平衡
- 训练完成

## 5. 监控关键指标

### 必须监控
1. `Train/mean_reward`: 主要指标，应该持续上升
2. `Train/rew_base_height`: 应该 > 30
3. `Train/rew_orientation`: 应该 > -10
4. `Train/rew_upright_bonus`: 应该 > 15

### 次要监控
5. `Train/mean_episode_length`: 应该接近最大长度（20s）
6. `Loss/value_function`: 应该逐渐降低
7. `Policy/learning_rate`: 保持在 1e-4

## 6. 如果仍然失败

### 方案 A: 进一步降低难度
```python
# 增加奖励权重
base_height = 60.0
orientation = -60.0
upright_bonus = 30.0

# 降低 PD 增益
kp_theta = 6.0
kp_l0 = 400.0
```

### 方案 B: 调整学习率
```python
# 如果训练不稳定
learning_rate = 5e-5  # 降低

# 如果训练太慢
learning_rate = 2e-4  # 提高
```

### 方案 C: 检查物理参数
```bash
# 查看机器人质量和惯量
cat resources/robots/serialleg/urdf/serialleg.urdf | grep -A 5 "inertial"
```

## 7. 参数调整策略

### 如果机器人倒地太快
1. 降低 PD 增益（kp_theta, kp_l0）
2. 降低动作幅度（action_scale_*）
3. 增加高度奖励（base_height）

### 如果机器人晃动不停
1. 增加阻尼（kd_theta, kd_l0）
2. 增加角速度惩罚（ang_vel_xy）
3. 增加静止奖励（stand_still）

### 如果训练不收敛
1. 降低学习率（learning_rate）
2. 增加 mini batch（num_mini_batches）
3. 检查奖励权重是否合理

### 如果奖励振荡
1. 降低学习率
2. 增加 clip 范围（clip_single_reward）
3. 减少探索（entropy_coef）

## 8. 完整配置总结

```python
# PD 控制
kp_theta = 8.0, kd_theta = 4.0
kp_l0 = 600.0, kd_l0 = 6.0

# 动作幅度
action_scale_theta = 0.3
action_scale_l0 = 0.05
action_scale_vel = 8.0

# 核心奖励
base_height = 40.0
orientation = -40.0
upright_bonus = 20.0

# 静止约束
lin_vel_z = -2.0
ang_vel_xy = -1.0
stand_still = 3.0

# 训练参数
learning_rate = 1e-4
num_learning_epochs = 5
num_mini_batches = 8
```

## 9. 开始训练

```bash
conda activate gym_env
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

预计训练时间：
- GPU: RTX 3090 → 约 30-60 分钟
- GPU: RTX 4090 → 约 20-40 分钟

## 10. 验证成功

训练成功的标志：
- ✅ `mean_reward` > 50
- ✅ `rew_base_height` > 30
- ✅ `rew_upright_bonus` > 15
- ✅ 机器人可以稳定站立 20 秒

测试命令：
```bash
python wheel_legged_gym/scripts/play_balance.py --task=wheel_legged_vmc_balance
```
