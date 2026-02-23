# 两阶段控制策略：Balance → Flat

## 🎯 目标

创建一个两阶段控制系统：

1. **Balance 网络**: 从任意初始姿态 → 恢复到 Flat 任务的初始条件
2. **Flat 网络**: 从 Flat 初始条件 → 保持平衡

## 📊 Flat 任务分析

### Flat 任务的初始条件

根据代码分析，Flat 任务使用默认初始化：

```python
# 基类默认初始化（legged_robot_config.py）
class init_state:
    pos = [0.0, 0.0, 1.0]          # 高度 1.0m
    rot = [0.0, 0.0, 0.0, 1.0]     # 完全直立（四元数）
    lin_vel = [0.0, 0.0, 0.0]      # 静止
    ang_vel = [0.0, 0.0, 0.0]      # 无旋转
```

**转换为欧拉角**:
- Roll = 0°
- Pitch = 0°
- Yaw = 0°
- 高度 = 1.0m（或根据机器人调整）
- 所有速度 = 0

### Flat 任务的能力

Flat 网络已经训练好，能够：
- ✅ 在完全直立状态下保持平衡
- ✅ 处理小幅度扰动（±5°）
- ❌ 无法从大角度倾斜恢复

## 🎯 Balance 任务设计

### 目标

Balance 网络的任务是：
- **输入**: 任意初始姿态（大角度倾斜）
- **输出**: 恢复到 Flat 初始条件
- **切换条件**: 达到 Flat 初始条件后，切换到 Flat 网络

### 目标状态（Flat 初始条件）

```python
# Balance 网络的目标
target_roll = 0.0 rad
target_pitch = 0.0 rad
target_height = 0.25 m  # 根据实际机器人调整
target_lin_vel = 0.0 m/s
target_ang_vel = 0.0 rad/s
```

### 奖励函数设计

Balance 网络需要奖励"接近 Flat 初始条件"：

```python
# 1. 姿态接近目标
reward_pitch_to_zero = -100.0 * pitch_angle²
reward_roll_to_zero = -100.0 * roll_angle²

# 2. 高度接近目标
reward_height_to_target = -50.0 * (height - 0.25)²

# 3. 速度接近零
reward_lin_vel_to_zero = -10.0 * ||lin_vel||²
reward_ang_vel_to_zero = -10.0 * ||ang_vel||²

# 4. 达到目标的奖励
reward_reach_target = 100.0  # 当所有条件满足时
```

### 初始化范围

Balance 网络需要从各种姿态开始训练：

```python
# 大范围初始化
roll = [-0.8, 0.8]      # ±45.8°
pitch = [-0.8, 0.8]     # ±45.8°
height = [0.15, 0.35]   # ±0.1m
lin_vel = [-0.5, 0.5]   # ±0.5 m/s
ang_vel = [-1.0, 1.0]   # ±1.0 rad/s
```

## 📝 实施方案

### 方案 1: 修改 Balance 配置（推荐）

修改 `wheel_legged_vmc_balance_config.py`：

```python
class WheelLeggedVMCBalanceCfg(WheelLeggedVMCCfg):

    class control:
        # 允许大幅度恢复
        action_scale_theta = 1.0   # 57.3°
        action_scale_l0 = 0.1
        action_scale_vel = 10.0

        # PD 增益
        kp_theta = 8.0
        kd_theta = 4.0
        kp_l0 = 600.0
        kd_l0 = 6.0

    class rewards:
        class scales:
            # 目标：达到 Flat 初始条件
            pitch_angle = -100.0      # 强烈惩罚偏离 0°
            roll_angle = -100.0
            pitch_vel = -10.0         # 惩罚角速度
            roll_vel = -10.0

            # 高度控制到目标
            base_height = 50.0        # 奖励接近目标高度

            # 速度控制到零
            lin_vel_z = -10.0
            base_lin_vel_xy = -10.0   # 惩罚 xy 方向速度
            ang_vel_yaw = -5.0        # 惩罚 yaw 角速度

            # 腿部控制
            leg_angle_zero = -5.0     # 鼓励腿垂直

            # 达到目标奖励
            reach_flat_target = 100.0  # 新增

            # 其他
            stand_still = 10.0
            torques = -1e-4
            termination = -10.0

    class init_state:
        # 大范围初始化
        pos = [0.0, 0.0, 0.25]  # 目标高度
        default_joint_angles = {...}  # 根据机器人设置

    class balance_reset:
        # 大范围扰动
        x_pos_offset = [0.0, 0.0]
        y_pos_offset = [0.0, 0.0]
        z_pos_offset = [-0.1, 0.1]    # ±10cm
        roll = [-0.8, 0.8]            # ±45.8°
        pitch = [-0.8, 0.8]           # ±45.8°
        yaw = [-0.5, 0.5]             # ±28.6°
        lin_vel_x = [-0.5, 0.5]       # ±0.5 m/s
        lin_vel_y = [-0.5, 0.5]
        lin_vel_z = [-0.3, 0.3]
        ang_vel_roll = [-1.0, 1.0]    # ±1.0 rad/s
        ang_vel_pitch = [-1.0, 1.0]
        ang_vel_yaw = [-0.5, 0.5]
```

### 方案 2: 添加新的奖励函数

在 `wheel_legged_vmc_balance.py` 中添加：

```python
def _reward_reach_flat_target(self):
    """奖励达到 Flat 初始条件"""
    # 定义 Flat 目标
    target_height = 0.25  # m
    angle_threshold = 0.1  # 约 5.7°
    vel_threshold = 0.2    # m/s 或 rad/s

    # 检查是否达到目标
    height_ok = torch.abs(self.base_height - target_height) < 0.05
    pitch_ok = torch.abs(self.pitch_angle) < angle_threshold
    roll_ok = torch.abs(self.roll_angle) < angle_threshold
    lin_vel_ok = torch.norm(self.base_lin_vel, dim=1) < vel_threshold
    ang_vel_ok = torch.norm(self.base_ang_vel, dim=1) < vel_threshold

    # 所有条件都满足时给予大奖励
    all_ok = height_ok & pitch_ok & roll_ok & lin_vel_ok & ang_vel_ok
    return all_ok.float()
```

## 🔄 两阶段切换逻辑

### 切换条件

```python
def should_switch_to_flat(self):
    """判断是否应该切换到 Flat 网络"""
    # Flat 初始条件
    target_height = 0.25
    angle_threshold = 0.1  # 5.7°
    vel_threshold = 0.2

    # 检查条件
    height_ok = abs(self.base_height - target_height) < 0.05
    pitch_ok = abs(self.pitch_angle) < angle_threshold
    roll_ok = abs(self.roll_angle) < angle_threshold
    lin_vel_ok = torch.norm(self.base_lin_vel) < vel_threshold
    ang_vel_ok = torch.norm(self.base_ang_vel) < vel_threshold

    return height_ok and pitch_ok and roll_ok and lin_vel_ok and ang_vel_ok
```

### 实现方式

#### 选项 A: 在 play 脚本中切换

```python
# play_two_stage.py
def play_two_stage(args):
    # 加载两个网络
    balance_policy = load_policy("balance")
    flat_policy = load_policy("flat")

    current_policy = balance_policy
    stage = "balance"

    while True:
        # 选择策略
        if stage == "balance":
            actions = balance_policy(obs)

            # 检查是否达到 Flat 条件
            if should_switch_to_flat(env):
                print("Switching to Flat policy!")
                current_policy = flat_policy
                stage = "flat"

        else:  # stage == "flat"
            actions = flat_policy(obs)

            # 如果偏离太多，切回 Balance
            if should_switch_to_balance(env):
                print("Switching back to Balance policy!")
                current_policy = balance_policy
                stage = "balance"

        env.step(actions)
```

#### 选项 B: 训练一个切换策略

训练一个高层策略，决定何时切换：

```python
# 高层策略输出
switch_signal = high_level_policy(obs)

if switch_signal > 0.5:
    actions = flat_policy(obs)
else:
    actions = balance_policy(obs)
```

## 📈 训练流程

### 阶段 1: 训练 Balance 网络

```bash
# 目标：从任意姿态恢复到 Flat 初始条件
python train.py --task=wheel_legged_vmc_balance --num_envs=4096

# 训练 1000-2000 iterations
# 监控 rew_reach_flat_target 是否增加
```

### 阶段 2: 验证 Flat 网络

```bash
# 确认 Flat 网络能在初始条件下保持平衡
python play.py --task=wheel_legged_vmc_flat
```

### 阶段 3: 集成测试

```bash
# 测试两阶段切换
python play_two_stage.py
```

## 🎯 成功标准

### Balance 网络

- ✅ 能从 ±45° 倾斜恢复到 ±5°
- ✅ 能将高度调整到目标 ±5cm
- ✅ 能将速度降到接近 0
- ✅ `rew_reach_flat_target` > 80（80% 的 episode 达到目标）

### 两阶段系统

- ✅ Balance 阶段能在 5-10 秒内达到 Flat 条件
- ✅ 切换到 Flat 后能保持平衡 > 20 秒
- ✅ 整体成功率 > 80%

## 📊 监控指标

### Balance 训练

```
Train/rew_pitch_angle      # 应该接近 0
Train/rew_roll_angle       # 应该接近 0
Train/rew_base_height      # 应该 > 40
Train/rew_reach_flat_target  # 应该 > 80
Train/mean_episode_length  # 应该接近最大值
```

### 两阶段测试

```
Balance 阶段时长: 平均 5-10 秒
Flat 阶段时长: 平均 > 20 秒
切换成功率: > 80%
整体平衡时长: > 25 秒
```

## 🔧 调试技巧

### 如果 Balance 网络学不会恢复

1. 降低初始化难度：
   ```python
   roll = [-0.5, 0.5]  # 从 ±45° 降到 ±28°
   pitch = [-0.5, 0.5]
   ```

2. 增加恢复奖励：
   ```python
   pitch_angle = -150.0  # 从 -100.0 增加
   roll_angle = -150.0
   ```

3. 使用渐进式训练（见 PROGRESSIVE_TRAINING.md）

### 如果切换不平滑

1. 放宽切换条件：
   ```python
   angle_threshold = 0.15  # 从 0.1 增加到 0.15
   ```

2. 添加切换缓冲区（hysteresis）：
   ```python
   # 切换到 Flat: 条件严格
   # 切回 Balance: 条件宽松
   ```

## 💡 关键洞察

### 为什么需要两阶段？

1. **任务分解**:
   - Balance: 专注于大角度恢复
   - Flat: 专注于小角度平衡
   - 各自优化，性能更好

2. **训练效率**:
   - Flat 已经训练好，不需要重新训练
   - Balance 只需要学习恢复，不需要学习平衡

3. **鲁棒性**:
   - 两个专家网络比一个通用网络更鲁棒
   - 可以独立调试和优化

### Flat 初始条件的重要性

Flat 初始条件是两个网络的"交接点"：
- Balance 的目标
- Flat 的起点

必须精确定义，确保：
- Balance 能稳定达到
- Flat 能稳定接管

## 📚 相关文档

- [PROGRESSIVE_TRAINING.md](PROGRESSIVE_TRAINING.md) - 渐进式训练
- [BALANCE_RECOVERY_TRADEOFF.md](BALANCE_RECOVERY_TRADEOFF.md) - 平衡与恢复
- [ANGLE_BASED_REWARDS.md](ANGLE_BASED_REWARDS.md) - 奖励函数设计

## 🚀 快速开始

### 1. 修改 Balance 配置

```bash
# 编辑 wheel_legged_vmc_balance_config.py
# 按照上面的方案修改
```

### 2. 训练 Balance 网络

```bash
python train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 3. 创建两阶段测试脚本

```bash
# 创建 play_two_stage.py
# 实现切换逻辑
```

### 4. 测试

```bash
python play_two_stage.py
```

---

**目标**: Balance 网络从任意姿态恢复到 Flat 初始条件，然后切换到 Flat 网络保持平衡。

**关键**: 精确定义 Flat 初始条件，作为两个网络的交接点。
