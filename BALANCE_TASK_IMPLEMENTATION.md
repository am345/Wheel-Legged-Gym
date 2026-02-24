# Balance 任务实现文档

## 📋 任务目标

训练一个 Balance 网络，使轮腿机器人能够从**任意初始姿态**恢复到 **Flat 任务的初始条件**，然后切换到 Flat 网络保持平衡。

## 🎯 设计理念

模拟真实机器人的启动过程：
1. **断电状态**：机器人断电时，气弹簧撑开，腿处于最长位置
2. **自由落体**：以不同随机姿态落下
3. **等待稳定**：落地后等待机器人静止（速度接近 0）
4. **开始控制**：稳定后才启动控制，学习从稳定姿态恢复到 Flat 条件

**关键点**：只有稳定后的控制阶段数据用于训练，不学习自由落体过程。

## 🔄 完整启动流程

```
t=0.0s: Reset
  ├─ 随机姿态：roll/pitch ±45.8°, yaw ±28.6°
  ├─ 随机位置：z ±10cm
  ├─ 腿最长：lf1/rf1 = lower_limit
  └─ ready_for_control = False

t=0.0-2.0s: 自由落体阶段
  ├─ 小腿扭矩：-50 N·m（负扭矩让腿伸长）
  ├─ 其他关节：0 N·m（不输出）
  ├─ 机器人以不同姿态落下
  └─ 检测稳定性

t=2.0s: 达到稳定
  ├─ 线速度 < 0.1 m/s
  ├─ 角速度 < 0.1 rad/s
  ├─ 持续 1 秒
  └─ ready_for_control = True

t=2.0-60.0s: 控制学习阶段
  ├─ 使用策略计算的正常扭矩
  ├─ 学习从稳定姿态恢复到 Flat 条件
  ├─ 目标：高度 0.20m, 姿态 ±5.7°, 速度 < 0.2
  └─ 计算奖励

t=60.0s: 超时重启
  └─ 唯一的 termination 条件
```

## 📁 核心文件

### 1. wheel_legged_vmc_balance.py

**关键实现**：

#### `__init__()`
```python
# 稳定性跟踪
self.ready_for_control = torch.zeros(self.num_envs, dtype=torch.bool)
self.settling_time = torch.zeros(self.num_envs)
self.settling_threshold = 1.0  # 秒
self.vel_stable_threshold = 0.1  # m/s 或 rad/s
```

#### `post_physics_step()`
```python
# 计算 pitch/roll 角度
self.pitch_angle = torch.atan2(self.projected_gravity[:, 1], -self.projected_gravity[:, 2])
self.roll_angle = torch.atan2(self.projected_gravity[:, 0], -self.projected_gravity[:, 2])

# 检测稳定性
lin_vel_norm = torch.norm(self.base_lin_vel, dim=1)
ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)
is_stable = (lin_vel_norm < 0.1) & (ang_vel_norm < 0.1)

# 累积稳定时间
self.settling_time = torch.where(is_stable, self.settling_time + self.dt, 0.0)

# 稳定后允许控制
self.ready_for_control = self.settling_time >= 1.0
```

#### `_compute_torques()`
```python
# 调用父类计算正常力矩
torques = super()._compute_torques(actions)

# 稳定前：强制腿伸长
if not self.ready_for_control.all():
    not_ready = ~self.ready_for_control

    # 所有关节零扭矩
    torques[not_ready, :] = 0.0

    # 小腿负扭矩（模拟气弹簧）
    torques[not_ready, lf1_idx] = -50.0  # N·m
    torques[not_ready, rf1_idx] = -50.0
```

#### `check_termination()`
```python
# 禁用所有失败条件
self.fail_buf[:] = 0

# 只有超时才重启（60 秒）
self.time_out_buf = self.episode_length_buf > self.max_episode_length
self.reset_buf = self.time_out_buf.clone()
```

#### `_reset_dofs()`
```python
# 调用父类设置基本状态
super()._reset_dofs(env_ids)

# 设置腿最长（lf1/rf1 = lower_limit）
lf1_idx = self.dof_names.index("lf1_Joint")
rf1_idx = self.dof_names.index("rf1_Joint")
self.dof_pos[env_ids, lf1_idx] = self.dof_pos_limits[lf1_idx, 0]
self.dof_pos[env_ids, rf1_idx] = self.dof_pos_limits[rf1_idx, 0]
```

#### `_reset_root_states()`
```python
# 应用 balance_reset 配置的随机姿态
cfg = self.cfg.balance_reset

# 随机姿态（欧拉角 -> 四元数）
roll = torch_rand_float(cfg.roll[0], cfg.roll[1], (len(env_ids), 1), device=self.device)
pitch = torch_rand_float(cfg.pitch[0], cfg.pitch[1], (len(env_ids), 1), device=self.device)
yaw = torch_rand_float(cfg.yaw[0], cfg.yaw[1], (len(env_ids), 1), device=self.device)
quat = quat_from_euler_xyz(roll, pitch, yaw).squeeze(1)
self.root_states[env_ids, 3:7] = quat

# 随机位置和速度...
```

#### 奖励函数
```python
def _reward_pitch_angle(self):
    """惩罚 pitch 角度偏差"""
    return torch.square(self.pitch_angle)

def _reward_roll_angle(self):
    """惩罚 roll 角度偏差"""
    return torch.square(self.roll_angle)

def _reward_reach_flat_target(self):
    """奖励达到 Flat 初始条件"""
    height_ok = torch.abs(self.base_height - 0.20) < 0.05
    pitch_ok = torch.abs(self.pitch_angle) < 0.1
    roll_ok = torch.abs(self.roll_angle) < 0.1
    lin_vel_ok = torch.norm(self.base_lin_vel, dim=1) < 0.2
    ang_vel_ok = torch.norm(self.base_ang_vel, dim=1) < 0.2
    all_ok = height_ok & pitch_ok & roll_ok & lin_vel_ok & ang_vel_ok
    return all_ok.float()
```

### 2. wheel_legged_vmc_balance_config.py

**关键配置**：

#### Episode 设置
```python
class env(WheelLeggedVMCCfg.env):
    num_envs = 4096
    episode_length_s = 60  # 1 分钟超时重启
    fail_to_terminal_time_s = 60.0
```

#### 控制参数
```python
class control(WheelLeggedVMCCfg.control):
    action_scale_theta = 1.0   # 57.3° - 允许大幅度恢复
    action_scale_l0 = 0.1      # 10cm
    action_scale_vel = 10.0    # 轮速

    l0_offset = 0.23
    feedforward_force = 40.0

    kp_theta = 8.0
    kd_theta = 4.0
    kp_l0 = 600.0
    kd_l0 = 6.0
```

#### 奖励权重
```python
class scales(WheelLeggedVMCCfg.rewards.scales):
    # 核心目标
    base_height = 50.0
    pitch_angle = -100.0      # 强烈惩罚偏离 0°
    roll_angle = -100.0
    pitch_vel = -10.0
    roll_vel = -10.0
    leg_angle_zero = -5.0
    reach_flat_target = 100.0  # 达到 Flat 条件的大奖励

    # 辅助目标
    upright_bonus = 20.0
    stand_still = 10.0
    lin_vel_z = -10.0
    ang_vel_yaw = -5.0
    base_lin_vel_xy = -10.0

    # 禁用的奖励
    tracking_lin_vel = 0.0
    tracking_ang_vel = 0.0
    orientation = 0.0
    ang_vel_xy = 0.0
```

#### 初始化范围
```python
class balance_reset:
    """大范围随机初始化"""
    # 位置扰动
    x_pos_offset = [0.0, 0.0]
    y_pos_offset = [0.0, 0.0]
    z_pos_offset = [-0.1, 0.1]  # ±10cm

    # 姿态扰动（大范围！）
    roll = [-0.8, 0.8]          # ±45.8°
    pitch = [-0.8, 0.8]         # ±45.8°
    yaw = [-0.5, 0.5]           # ±28.6°

    # 速度扰动（当前设为 0，可根据需要调整）
    lin_vel_x = [0, 0]
    lin_vel_y = [0, 0]
    lin_vel_z = [0, 0]
    ang_vel_roll = [0, 0]
    ang_vel_pitch = [0, 0]
    ang_vel_yaw = [0, 0]
```

#### Termination 设置
```python
class asset(WheelLeggedVMCCfg.asset):
    terminate_after_contacts_on = []  # 不因接触而终止
```

## 🎯 Flat 初始条件（目标状态）

Balance 网络的目标 = Flat 网络的起点：

| 参数 | 目标值 | 容差 |
|------|--------|------|
| 高度 | 0.20 m | ±5 cm |
| Pitch | 0° | ±5.7° (0.1 rad) |
| Roll | 0° | ±5.7° (0.1 rad) |
| 线速度 | 0 m/s | < 0.2 m/s |
| 角速度 | 0 rad/s | < 0.2 rad/s |

## 🔧 关键参数调整指南

### 稳定性检测

**如果机器人一直不稳定**（`ready_for_control` 始终为 False）：
```python
# 在 __init__() 中调整
self.vel_stable_threshold = 0.2  # 从 0.1 增加到 0.2
self.settling_threshold = 0.5    # 从 1.0 减少到 0.5
```

**如果稳定判断太早**（还在晃动就开始控制）：
```python
self.vel_stable_threshold = 0.05  # 从 0.1 减少到 0.05
self.settling_threshold = 1.5     # 从 1.0 增加到 1.5
```

### 腿伸长扭矩

**如果腿伸长不够**（落地时腿不是最长）：
```python
# 在 _compute_torques() 中调整
torques[not_ready, lf1_idx] = -80.0  # 从 -50.0 增加到 -80.0
torques[not_ready, rf1_idx] = -80.0
```

**如果腿伸长过度**（关节到达极限并震荡）：
```python
torques[not_ready, lf1_idx] = -30.0  # 从 -50.0 减少到 -30.0
torques[not_ready, rf1_idx] = -30.0
```

### 初始化范围

**如果训练太难**（机器人总是倒地）：
```python
# 在 balance_reset 中减小范围
roll = [-0.5, 0.5]   # 从 ±0.8 减少到 ±0.5 (±28.6°)
pitch = [-0.5, 0.5]
```

**如果想增加难度**（训练更鲁棒的策略）：
```python
roll = [-1.0, 1.0]   # 从 ±0.8 增加到 ±1.0 (±57.3°)
pitch = [-1.0, 1.0]

# 或添加初始速度
lin_vel_x = [-0.5, 0.5]
lin_vel_y = [-0.5, 0.5]
ang_vel_roll = [-1.0, 1.0]
ang_vel_pitch = [-1.0, 1.0]
```

### 奖励权重

**如果机器人不学习恢复姿态**：
```python
# 增加姿态惩罚
pitch_angle = -150.0  # 从 -100.0 增加
roll_angle = -150.0
```

**如果机器人恢复太慢**：
```python
# 增加目标奖励
reach_flat_target = 150.0  # 从 100.0 增加
```

**如果机器人晃动太多**：
```python
# 增加速度惩罚
pitch_vel = -20.0  # 从 -10.0 增加
roll_vel = -20.0
```

## 📊 训练监控

### TensorBoard 关键指标

启动 TensorBoard：
```bash
tensorboard --logdir=logs/wheel_legged_vmc_balance
```

**必看指标**：

| 指标 | 目标值 | 含义 |
|------|--------|------|
| `Train/rew_reach_flat_target` | > 80 | 达到 Flat 条件的比例（%） |
| `Train/rew_pitch_angle` | → 0 | Pitch 角度接近 0° |
| `Train/rew_roll_angle` | → 0 | Roll 角度接近 0° |
| `Train/rew_base_height` | > 40 | 高度接近 0.20m |
| `Train/mean_episode_length` | → max | Episode 长度接近 60 秒 |

**调试指标**（需要添加到代码中）：

```python
# 在 post_physics_step() 中添加
self.extras["ready_ratio"] = self.ready_for_control.float().mean()
self.extras["settling_time_mean"] = self.settling_time.mean()
```

### 预期训练曲线

```
Iterations 0-500:
  rew_reach_flat_target: 0 → 30
  学习基本姿态控制

Iterations 500-1000:
  rew_reach_flat_target: 30 → 60
  提高恢复成功率

Iterations 1000-2000:
  rew_reach_flat_target: 60 → 85+
  稳定的恢复能力
```

## 🚀 训练流程

### 1. 训练 Balance 网络

```bash
# 使用快速脚本
./train_balance.sh

# 或手动运行
python wheel_legged_gym/scripts/train.py \
    --task=wheel_legged_vmc_balance \
    --num_envs=4096
```

**训练参数**：
- Environments: 4096
- Max iterations: 2000
- Learning rate: 1e-4
- Episode length: 60 秒

### 2. 监控训练

```bash
tensorboard --logdir=logs/wheel_legged_vmc_balance
```

关注 `rew_reach_flat_target` 是否增加。

### 3. 测试 Balance 网络

```bash
python wheel_legged_gym/scripts/play_balance.py
```

观察：
- 初始腿是否最长
- 是否自由落体
- 是否等待稳定（约 1-2 秒）
- 稳定后是否开始控制
- 能否恢复到直立姿态

### 4. 测试两阶段系统

```bash
./test_two_stage.sh
```

观察：
- Balance 阶段：能否从大角度恢复
- 切换时刻：是否满足 Flat 条件
- Flat 阶段：能否保持平衡 > 20 秒

## 🐛 常见问题

### 问题 1: 机器人一直不稳定

**症状**：`ready_for_control` 始终为 False，机器人一直在晃动

**原因**：稳定阈值太严格

**解决**：
```python
self.vel_stable_threshold = 0.2  # 放宽速度阈值
self.settling_threshold = 0.5    # 减少等待时间
```

### 问题 2: 腿没有伸长

**症状**：落地时腿不是最长状态

**原因**：负扭矩不够大

**解决**：
```python
torques[not_ready, lf1_idx] = -80.0  # 增加负扭矩
torques[not_ready, rf1_idx] = -80.0
```

### 问题 3: 训练不收敛

**症状**：`rew_reach_flat_target` 一直很低

**原因**：初始化范围太大或奖励设计不合理

**解决**：
1. 减小初始化范围：
```python
roll = [-0.5, 0.5]  # 从 ±45.8° 减到 ±28.6°
pitch = [-0.5, 0.5]
```

2. 增加目标奖励：
```python
reach_flat_target = 150.0  # 从 100.0 增加
```

3. 使用渐进式训练（见 PROGRESSIVE_TRAINING.md）

### 问题 4: 机器人立即倒地

**症状**：稳定后立即 termination

**原因**：稳定姿态太极端

**解决**：
1. 检查是否真的禁用了接触终止：
```python
terminate_after_contacts_on = []  # 必须为空
```

2. 检查 `check_termination()` 是否正确：
```python
self.fail_buf[:] = 0  # 必须禁用失败条件
```

### 问题 5: 每次姿态相同

**症状**：机器人总是以相同姿态落下

**原因**：`_reset_root_states()` 没有被调用或随机范围为 0

**解决**：
1. 检查 `balance_reset` 配置：
```python
roll = [-0.8, 0.8]  # 不能是 [0, 0]
pitch = [-0.8, 0.8]
```

2. 确认 `_reset_root_states()` 被调用

## 📚 相关文档

- [REALISTIC_STARTUP.md](REALISTIC_STARTUP.md) - 真实启动流程设计
- [TWO_STAGE_IMPLEMENTATION.md](TWO_STAGE_IMPLEMENTATION.md) - 两阶段控制实施
- [QUICK_START.md](QUICK_START.md) - 快速开始指南
- [PROGRESSIVE_TRAINING.md](PROGRESSIVE_TRAINING.md) - 渐进式训练方案

## ✅ 实施检查清单

在开始训练前，确认以下内容：

- [ ] `episode_length_s = 60`（1 分钟）
- [ ] `terminate_after_contacts_on = []`（不因接触终止）
- [ ] `check_termination()` 只检查超时
- [ ] `balance_reset` 配置了大范围随机姿态
- [ ] `_reset_dofs()` 设置腿最长
- [ ] `_reset_root_states()` 应用随机姿态
- [ ] `_compute_torques()` 稳定前强制腿伸长
- [ ] `post_physics_step()` 检测稳定性
- [ ] 所有奖励函数已实现

## 🎯 成功标准

### Balance 网络训练成功

- ✅ `rew_reach_flat_target` > 80
- ✅ 能从 ±45° 恢复到 ±5°
- ✅ 平均 5-10 秒达到 Flat 条件
- ✅ Episode 长度接近 60 秒

### 两阶段系统成功

- ✅ Balance 阶段: 5-10 秒达到 Flat 条件
- ✅ 切换成功率 > 80%
- ✅ Flat 阶段: 保持平衡 > 20 秒
- ✅ 整体平衡时长 > 25 秒

## 🔄 下一步工作

### 短期优化

1. **调整稳定性检测**：
   - 根据实际测试调整 `vel_stable_threshold` 和 `settling_threshold`
   - 可能需要添加高度检测（确保落地）

2. **优化腿伸长扭矩**：
   - 根据实际机器人调整 `-50.0` N·m
   - 可能需要根据当前腿长动态调整

3. **奖励函数微调**：
   - 根据训练曲线调整权重
   - 可能需要添加恢复速度奖励

### 中期改进

1. **渐进式训练**：
   - 从小范围初始化开始（±20°）
   - 逐步增加到 ±45°
   - 见 PROGRESSIVE_TRAINING.md

2. **Curriculum Learning**：
   - 根据成功率动态调整初始化范围
   - 成功率高 → 增加难度
   - 成功率低 → 降低难度

3. **两阶段切换优化**：
   - 添加切换缓冲区（hysteresis）
   - 平滑切换，避免抖动

### 长期扩展

1. **多种地形**：
   - 在不同地形上训练（斜坡、台阶）
   - 增强鲁棒性

2. **外部扰动**：
   - 添加推力扰动
   - 训练抗干扰能力

3. **实机部署**：
   - Sim-to-Real 迁移
   - 域随机化
   - 系统辨识

## 📝 代码修改记录

### 2026-02-23

**实现真实启动流程**：

1. **wheel_legged_vmc_balance.py**：
   - 添加稳定性检测（`ready_for_control`, `settling_time`）
   - 重写 `_compute_torques()`：稳定前强制腿伸长
   - 重写 `check_termination()`：只有超时才重启
   - 重写 `_reset_dofs()`：设置腿最长初始位置
   - 重写 `_reset_root_states()`：应用大范围随机姿态
   - 添加所有奖励函数实现

2. **wheel_legged_vmc_balance_config.py**：
   - 设置 `episode_length_s = 60`
   - 设置 `terminate_after_contacts_on = []`
   - 配置 `balance_reset` 大范围初始化
   - 配置奖励权重
   - 添加 `height` 到 `commands.ranges`

3. **关键修复**：
   - 修复四元数形状不匹配（`.squeeze(1)`）
   - 修复 PhysX 错误（使用 `_reset_dofs()` 而非手动调用 `set_dof_state_tensor_indexed`）
   - 修复稳定前腿伸长逻辑（在 `_compute_torques()` 中强制扭矩）

---

**实施日期**: 2026-02-23
**状态**: ✅ 完成，准备训练
**下一步**: 运行 `./train_balance.sh` 开始训练
