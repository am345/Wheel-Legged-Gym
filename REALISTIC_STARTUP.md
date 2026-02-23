# 真实启动流程 - Balance 训练

## 🎯 设计理念

模拟真实机器人的启动过程，而不是从理想状态开始训练。

## 📋 启动流程

### 1. 断电状态（初始化）

**物理状态**:
- 腿处于最长位置（气弹簧撑开）
- lf1_Joint 和 rf1_Joint 在 lower_limit
- 机器人以随机姿态悬空

**代码实现**:
```python
# reset_idx() 中设置
self.dof_pos[env_ids, lf1_idx] = self.dof_pos_limits[lf1_idx, 0]  # lower_limit
self.dof_pos[env_ids, rf1_idx] = self.dof_pos_limits[rf1_idx, 0]  # lower_limit
```

### 2. 自由落体

**物理过程**:
- 机器人从随机姿态落下
- 腿保持最长（断电状态，零动作）
- 受重力和碰撞影响

**控制状态**:
```python
self.ready_for_control = False  # 还未允许控制
actions = torch.zeros_like(actions)  # 零动作
```

### 3. 等待稳定

**稳定判断**:
- 线速度 < 0.1 m/s
- 角速度 < 0.1 rad/s
- 持续时间 > 0.5 秒

**代码实现**:
```python
# post_physics_step() 中检测
lin_vel_norm = torch.norm(self.base_lin_vel, dim=1)
ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)
is_stable = (lin_vel_norm < 0.1) & (ang_vel_norm < 0.1)

# 累积稳定时间
self.settling_time = torch.where(
    is_stable,
    self.settling_time + self.dt,
    torch.zeros_like(self.settling_time)
)

# 稳定后允许控制
self.ready_for_control = self.settling_time >= 0.5
```

### 4. 开始控制

**学习目标**:
- 从稳定后的姿态恢复到 Flat 初始条件
- 高度: 0.20m ± 5cm
- 姿态: ±5.7°
- 速度: < 0.2 m/s

**控制逻辑**:
```python
# step() 中应用动作
actions = torch.where(
    self.ready_for_control.unsqueeze(1),
    original_actions,  # 稳定后使用策略动作
    torch.zeros_like(original_actions)  # 稳定前零动作
)
```

## 🔄 完整时间线

```
t=0.0s: Reset
  ├─ 设置腿最长（lf1/rf1 = lower_limit）
  ├─ 随机姿态（±45.8°）
  └─ ready_for_control = False

t=0.0-1.0s: 自由落体
  ├─ 零动作（断电状态）
  ├─ 机器人落下并碰撞地面
  └─ 检测稳定性

t=1.0s: 达到稳定
  ├─ 速度 < 0.1 m/s
  ├─ 持续 0.5 秒
  └─ ready_for_control = True

t=1.0-20.0s: 控制阶段
  ├─ 应用策略动作
  ├─ 学习恢复到 Flat 条件
  └─ 计算奖励
```

## 📊 训练数据

### 只有控制阶段的数据用于学习

**不学习的阶段**:
- ❌ 自由落体阶段（零动作）
- ❌ 等待稳定阶段（零动作）

**学习的阶段**:
- ✅ 稳定后的控制阶段
- ✅ 从稳定姿态恢复到 Flat

### 为什么这样设计？

1. **符合实际**:
   - 真实机器人断电时腿最长
   - 启动前需要等待稳定
   - 不会从空中开始控制

2. **训练效率**:
   - 只学习有意义的控制策略
   - 不浪费时间学习"等待"
   - 初始状态更真实

3. **鲁棒性**:
   - 从各种稳定姿态开始
   - 覆盖真实启动场景
   - 不依赖理想初始条件

## 🎮 初始化范围

### 姿态扰动（balance_reset）

```python
roll = [-0.8, 0.8]      # ±45.8°
pitch = [-0.8, 0.8]     # ±45.8°
yaw = [-0.5, 0.5]       # ±28.6°
z_pos_offset = [-0.1, 0.1]  # ±10cm
```

### 稳定后的实际姿态

落地后的姿态取决于：
- 初始姿态
- 腿长（最长）
- 地面碰撞
- 重力方向

**预期范围**:
- 大部分情况: ±30° 左右
- 极端情况: 可能倒地（触发 termination）

## 🔧 关键参数

### 稳定性检测

```python
vel_stable_threshold = 0.1  # m/s 或 rad/s
settling_threshold = 0.5    # 秒
```

**调整建议**:
- 如果稳定判断太严格 → 增加 `vel_stable_threshold`
- 如果等待时间太长 → 减少 `settling_threshold`
- 如果频繁误判 → 增加 `settling_threshold`

### 腿长设置

```python
# 腿最长 = lf1/rf1 在 lower_limit
self.dof_pos[env_ids, lf1_idx] = self.dof_pos_limits[lf1_idx, 0]
self.dof_pos[env_ids, rf1_idx] = self.dof_pos_limits[rf1_idx, 0]
```

**验证方法**:
```bash
# 运行仿真，观察初始腿长
python wheel_legged_gym/scripts/play_balance.py

# 检查 lf1/rf1 是否在 lower_limit
# 腿应该完全伸展
```

## 📈 预期训练效果

### Episode 结构

```
Episode 长度: 20 秒 (4000 steps @ 200Hz)

├─ 0.0-1.0s: 等待稳定 (不学习)
│   └─ 零动作，自由落体
│
└─ 1.0-20.0s: 控制学习 (学习)
    ├─ 应用策略动作
    ├─ 恢复到 Flat 条件
    └─ 计算奖励
```

### 成功标准

**稳定阶段**:
- ✅ 1 秒内达到稳定
- ✅ 不立即倒地

**控制阶段**:
- ✅ 5-10 秒达到 Flat 条件
- ✅ `rew_reach_flat_target` > 80
- ✅ 能从各种稳定姿态恢复

## 🔍 调试技巧

### 问题 1: 一直不稳定

**症状**: `ready_for_control` 始终为 False

**检查**:
```python
# 在 post_physics_step() 中打印
print(f"Lin vel: {lin_vel_norm.mean():.3f}")
print(f"Ang vel: {ang_vel_norm.mean():.3f}")
print(f"Settling time: {self.settling_time.mean():.3f}")
```

**解决**:
- 放宽稳定阈值: `vel_stable_threshold = 0.2`
- 减少等待时间: `settling_threshold = 0.3`

### 问题 2: 腿长设置不正确

**症状**: 初始腿长不是最长

**检查**:
```python
# 在 reset_idx() 中打印
print(f"lf1 limit: {self.dof_pos_limits[lf1_idx, 0]:.3f}")
print(f"lf1 pos: {self.dof_pos[env_ids[0], lf1_idx]:.3f}")
```

**解决**:
- 检查 URDF 中的 joint limits
- 确认 lower_limit 对应腿最长

### 问题 3: 稳定后立即倒地

**症状**: 稳定后马上 termination

**原因**: 稳定姿态太极端（接近倒地）

**解决**:
- 减小初始化范围: `roll/pitch = [-0.5, 0.5]`
- 放宽 termination 条件
- 增加恢复能力训练

## 💡 与标准训练的对比

### 标准训练（之前）

```python
# 直接从随机姿态开始控制
reset() → 立即应用动作 → 学习
```

**问题**:
- 不符合实际启动流程
- 可能从空中开始控制
- 初始状态不真实

### 真实启动训练（现在）

```python
# 模拟真实启动流程
reset() → 断电落下 → 等待稳定 → 开始控制 → 学习
```

**优势**:
- ✅ 符合真实机器人行为
- ✅ 从稳定状态开始学习
- ✅ 更鲁棒的策略

## 🚀 使用方法

### 训练

```bash
# 使用新的真实启动流程训练
./train_balance.sh
```

### 监控

```bash
tensorboard --logdir=logs/wheel_legged_vmc_balance
```

**关注指标**:
- `Train/ready_for_control_ratio` - 稳定成功率
- `Train/settling_time_mean` - 平均稳定时间
- `Train/rew_reach_flat_target` - 达到 Flat 条件

### 测试

```bash
# 观察启动流程
python wheel_legged_gym/scripts/play_balance.py

# 观察:
# 1. 初始腿是否最长
# 2. 是否自由落体
# 3. 是否等待稳定
# 4. 稳定后是否开始控制
```

## 📚 相关文档

- [TWO_STAGE_IMPLEMENTATION.md](TWO_STAGE_IMPLEMENTATION.md) - 两阶段控制
- [QUICK_START.md](QUICK_START.md) - 快速开始
- [BALANCE_RECOVERY_TRADEOFF.md](BALANCE_RECOVERY_TRADEOFF.md) - 平衡与恢复

---

**实施日期**: 2026-02-23
**状态**: ✅ 已实现
**关键**: 模拟真实启动流程，只学习稳定后的控制阶段
