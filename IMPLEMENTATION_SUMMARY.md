# 两阶段控制系统 - 实施完成总结

## ✅ 已完成的工作

### 1. 核心实现

#### Balance 环境 ([wheel_legged_vmc_balance.py](wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance.py))
- ✅ 添加 Flat 目标状态定义（高度 0.20m，角度 ±5.7°，速度 < 0.2）
- ✅ 实现 pitch/roll 角度和角速度计算
- ✅ 添加 `_reward_reach_flat_target()` 奖励函数
- ✅ 所有奖励函数基于物理量（角度、角速度）设计

#### Balance 配置 ([wheel_legged_vmc_balance_config.py](wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance_config.py))
- ✅ 大范围初始化（±45.8° 姿态，±0.5 m/s 速度）
- ✅ 奖励函数优化（强烈惩罚偏离目标，大奖励达到目标）
- ✅ 允许大幅度恢复动作（action_scale_theta = 1.0）
- ✅ 配置 PD 增益和控制参数

#### 两阶段切换脚本 ([play_two_stage.py](wheel_legged_gym/scripts/play_two_stage.py))
- ✅ 加载 Balance 和 Flat 两个策略
- ✅ 实时监控 Flat 条件
- ✅ 自动切换逻辑
- ✅ 手动切换功能（M 键）
- ✅ 统计信息显示
- ✅ 键盘控制（C/R/M/S/ESC）

### 2. 快速启动脚本

- ✅ [train_balance.sh](train_balance.sh) - 一键训练 Balance 网络
- ✅ [test_two_stage.sh](test_two_stage.sh) - 一键测试两阶段系统

### 3. 文档

- ✅ [QUICK_START.md](QUICK_START.md) - 快速开始指南
- ✅ [TWO_STAGE_IMPLEMENTATION.md](TWO_STAGE_IMPLEMENTATION.md) - 完整实施指南
- ✅ [TWO_STAGE_CONTROL.md](TWO_STAGE_CONTROL.md) - 设计文档
- ✅ [PROGRESSIVE_TRAINING.md](PROGRESSIVE_TRAINING.md) - 渐进式训练方案
- ✅ [BALANCE_RECOVERY_TRADEOFF.md](BALANCE_RECOVERY_TRADEOFF.md) - 平衡与恢复权衡
- ✅ [ACTION_SCALE_FIX.md](ACTION_SCALE_FIX.md) - Action scale 说明

### 4. 关键修正

- ✅ **Flat 目标高度**: 从 0.25m 更正为 0.20m（匹配实际机器人）
- ✅ **语法验证**: 所有 Python 文件编译通过
- ✅ **一致性检查**: Balance 环境和 play 脚本使用相同的目标参数

## 🎯 系统架构

```
任意初始姿态（±45.8°）
    ↓
[Balance 网络]
    ↓ (5-10 秒)
    ↓ 达到 Flat 条件:
    ↓ - 高度: 0.20m ± 5cm
    ↓ - 姿态: ±5.7°
    ↓ - 速度: < 0.2 m/s
    ↓
[自动切换]
    ↓
[Flat 网络]
    ↓ (> 20 秒)
保持平衡
```

## 📋 Flat 初始条件（交接点）

Balance 网络的目标 = Flat 网络的起点：

| 参数 | 目标值 | 容差 |
|------|--------|------|
| 高度 | 0.20 m | ±5 cm |
| Pitch | 0° | ±5.7° |
| Roll | 0° | ±5.7° |
| 线速度 | 0 m/s | < 0.2 m/s |
| 角速度 | 0 rad/s | < 0.2 rad/s |

## 🚀 使用流程

### 步骤 1: 确认 Flat 网络已训练

```bash
# 检查 Flat 模型
ls logs/wheel_legged_vmc_flat/

# 如果没有，先训练 Flat
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat --num_envs=4096
```

### 步骤 2: 训练 Balance 网络

```bash
# 使用快速脚本
./train_balance.sh

# 或手动运行
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

**训练参数**:
- Environments: 4096
- Max iterations: 2000
- 初始化: ±45.8° 姿态，±0.5 m/s 速度
- 目标: 达到 Flat 初始条件

### 步骤 3: 监控训练

```bash
tensorboard --logdir=logs/wheel_legged_vmc_balance
```

**关键指标**:
- `Train/rew_reach_flat_target` - 应该 > 80（80% 达到目标）
- `Train/rew_pitch_angle` - 应该接近 0
- `Train/rew_roll_angle` - 应该接近 0
- `Train/mean_episode_length` - 应该接近最大值

### 步骤 4: 测试两阶段系统

```bash
# 使用快速脚本
./test_two_stage.sh

# 或手动运行
python wheel_legged_gym/scripts/play_two_stage.py
```

**观察**:
- Balance 阶段能否从大角度恢复
- 切换是否平滑
- Flat 阶段能否保持平衡

## 📊 成功标准

### Balance 网络训练

训练成功的标志：
- ✅ `rew_reach_flat_target` > 80
- ✅ 能从 ±45° 恢复到 ±5°
- ✅ 平均 5-10 秒达到 Flat 条件

### 两阶段系统

系统成功的标志：
- ✅ Balance 阶段: 5-10 秒达到 Flat 条件
- ✅ 切换成功率 > 80%
- ✅ Flat 阶段: 保持平衡 > 20 秒
- ✅ 整体平衡时长 > 25 秒

## 🔧 配置参数

### Balance 网络

```python
# 控制参数
action_scale_theta = 1.0   # 57.3° - 允许大幅度恢复
action_scale_l0 = 0.1      # 10cm
action_scale_vel = 10.0    # 轮速

# PD 增益
kp_theta = 8.0
kd_theta = 4.0
kp_l0 = 600.0
kd_l0 = 6.0

# 奖励权重
base_height = 50.0
pitch_angle = -100.0       # 强烈惩罚偏离 0°
roll_angle = -100.0
pitch_vel = -10.0
roll_vel = -10.0
leg_angle_zero = -5.0
reach_flat_target = 100.0  # 大奖励达到目标
```

### 初始化范围

```python
# 大范围初始化
roll = [-0.8, 0.8]         # ±45.8°
pitch = [-0.8, 0.8]        # ±45.8°
yaw = [-0.5, 0.5]          # ±28.6°
z_pos_offset = [-0.1, 0.1] # ±10cm
lin_vel_x = [-0.5, 0.5]    # ±0.5 m/s
lin_vel_y = [-0.5, 0.5]
lin_vel_z = [-0.3, 0.3]
ang_vel_roll = [-1.0, 1.0] # ±1.0 rad/s
ang_vel_pitch = [-1.0, 1.0]
ang_vel_yaw = [-0.5, 0.5]
```

## 📁 文件清单

### 核心文件
- `wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance.py` - Balance 环境
- `wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance_config.py` - Balance 配置
- `wheel_legged_gym/scripts/play_two_stage.py` - 两阶段测试脚本

### 快速脚本
- `train_balance.sh` - 训练脚本
- `test_two_stage.sh` - 测试脚本

### 文档
- `QUICK_START.md` - 快速开始指南（推荐首先阅读）
- `TWO_STAGE_IMPLEMENTATION.md` - 完整实施指南
- `TWO_STAGE_CONTROL.md` - 设计文档
- `PROGRESSIVE_TRAINING.md` - 渐进式训练
- `BALANCE_RECOVERY_TRADEOFF.md` - 平衡与恢复权衡
- `ACTION_SCALE_FIX.md` - Action scale 说明
- `IMPLEMENTATION_SUMMARY.md` - 本文件

## 💡 关键设计决策

### 1. Flat 目标高度: 0.20m

根据 `wheel_legged_config.py` 中的实际机器人初始高度确定：
```python
class init_state(LeggedRobotCfg.init_state):
    pos = [0.0, 0.0, 0.20]  # 实际高度
```

### 2. action_scale_theta: 1.0

允许 57.3° 的腿部摆动，足够从中等到大角度倾斜恢复，同时保持可控性。

### 3. 大范围初始化

±45.8° 姿态初始化确保网络学习真正的恢复能力，而不仅仅是局部平衡。

### 4. 奖励函数设计

基于物理量（角度、角速度）而非重力分量，更直观易调。

### 5. 两阶段切换

自动切换 + 手动切换，既保证系统自主性，又允许调试和干预。

## 🎯 下一步行动

### 立即执行

```bash
# 1. 训练 Balance 网络
./train_balance.sh

# 2. 在另一个终端监控训练
tensorboard --logdir=logs/wheel_legged_vmc_balance

# 3. 训练完成后测试
./test_two_stage.sh
```

### 预期时间线

- **训练**: 2000 iterations（约 2-4 小时，取决于硬件）
- **监控**: 实时查看 TensorBoard
- **测试**: 训练完成后立即测试

### 成功指标

训练过程中关注：
- Iteration 500: `rew_reach_flat_target` 应该 > 40
- Iteration 1000: `rew_reach_flat_target` 应该 > 70
- Iteration 2000: `rew_reach_flat_target` 应该 > 85

## 🔍 故障排除

如果遇到问题，参考：
- [QUICK_START.md](QUICK_START.md) - 故障排除章节
- [TWO_STAGE_IMPLEMENTATION.md](TWO_STAGE_IMPLEMENTATION.md) - 调试指南
- [PROGRESSIVE_TRAINING.md](PROGRESSIVE_TRAINING.md) - 渐进式训练方案

## 📚 技术细节

### 奖励函数

```python
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

### 切换逻辑

```python
def check_flat_condition(self, env):
    """检查是否达到 Flat 条件"""
    # 计算 pitch/roll 角度
    pitch = np.arctan2(gravity[1], -gravity[2])
    roll = np.arctan2(gravity[0], -gravity[2])

    # 检查所有条件
    height_ok = abs(height - 0.20) < 0.05
    pitch_ok = abs(pitch) < 0.1
    roll_ok = abs(roll) < 0.1
    lin_vel_ok = np.linalg.norm(lin_vel) < 0.2
    ang_vel_ok = np.linalg.norm(ang_vel) < 0.2

    return height_ok and pitch_ok and roll_ok and lin_vel_ok and ang_vel_ok
```

---

**实施日期**: 2026-02-23
**状态**: ✅ 完成，准备训练
**验证**: ✅ 所有文件语法正确
**下一步**: 运行 `./train_balance.sh` 开始训练
