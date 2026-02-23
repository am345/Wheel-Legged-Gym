# 两阶段控制系统 - 完整实施指南

## ✅ 已完成的工作

### 1. Balance 环境修改
- ✅ 添加 `_reward_reach_flat_target()` - 奖励达到 Flat 初始条件
- ✅ 计算 pitch/roll 角度和角速度
- ✅ 定义 Flat 目标状态（高度、姿态、速度）

### 2. Balance 配置修改
- ✅ 大范围初始化（±45.8° 姿态，±0.5 m/s 速度）
- ✅ 奖励函数优化（强烈惩罚偏离目标）
- ✅ 允许大幅度恢复动作（action_scale_theta = 1.0）

### 3. 两阶段切换脚本
- ✅ `play_two_stage.py` - 自动/手动切换
- ✅ 实时监控 Flat 条件
- ✅ 统计信息显示

### 4. 快速启动脚本
- ✅ `test_two_stage.sh` - 一键测试

## 🎯 系统架构

```
任意初始姿态
    ↓
[Balance 网络]
    ↓ (达到 Flat 条件)
    ↓ - 高度: 0.25m ± 5cm
    ↓ - 姿态: ±5.7°
    ↓ - 速度: < 0.2 m/s
    ↓
[Flat 网络]
    ↓
保持平衡
```

## 📋 Flat 初始条件（交接点）

Balance 网络的目标 = Flat 网络的起点：

| 参数 | 目标值 | 容差 |
|------|--------|------|
| 高度 | 0.25 m | ±5 cm |
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
python train.py --task=wheel_legged_vmc_flat --num_envs=4096
```

### 步骤 2: 训练 Balance 网络

```bash
# 训练 Balance 网络（从任意姿态恢复到 Flat 条件）
python train.py --task=wheel_legged_vmc_balance --num_envs=4096

# 监控关键指标
tensorboard --logdir=logs/wheel_legged_vmc_balance
```

**关键指标**:
- `Train/rew_reach_flat_target` - 应该 > 80（80% 达到目标）
- `Train/rew_pitch_angle` - 应该接近 0
- `Train/rew_roll_angle` - 应该接近 0
- `Train/mean_episode_length` - 应该接近最大值

### 步骤 3: 测试两阶段系统

```bash
# 方式 1: 使用快速脚本
./test_two_stage.sh

# 方式 2: 手动运行
python wheel_legged_gym/scripts/play_two_stage.py
```

## 🎮 操作说明

### 键盘控制

| 按键 | 功能 |
|------|------|
| C | 启动控制（Balance 阶段） |
| M | 手动切换阶段 |
| R | 重置环境 |
| S | 切换统计显示 |
| ESC | 退出 |

### 自动切换

当满足以下所有条件时，自动从 Balance 切换到 Flat：
- ✅ 高度误差 < 5cm
- ✅ Pitch 角度 < 5.7°
- ✅ Roll 角度 < 5.7°
- ✅ 线速度 < 0.2 m/s
- ✅ 角速度 < 0.2 rad/s

### 显示信息

```
============================================================
  Two-Stage Control Monitor - BALANCE - RECOVERING
============================================================

Current Stage: BALANCE
  Balance steps: 245
  Flat steps:    0
  Total steps:   245

Flat Condition Check:
  Height:    0.268 m  (target: 0.250 m)  ✗
  Pitch:     +12.34°  ✗
  Roll:      -8.76°   ✗
  Lin Vel:   0.156 m/s  ✓
  Ang Vel:   0.234 rad/s  ✗

  Ready for Flat: NO

Overall Stats:
  Episodes:       3
  Success:        2 (66.7%)
  Stage switches: 2
  Avg switch time: 8.5 s

Controls:
  [C] Start  [R] Reset  [M] Manual Switch  [S] Toggle Stats
============================================================
```

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

## 🔧 调试指南

### 问题 1: Balance 网络学不会恢复

**症状**: `rew_reach_flat_target` 一直很低

**解决方案**:
1. 降低初始化难度：
   ```python
   # balance_config.py
   roll = [-0.5, 0.5]  # 从 ±45° 降到 ±28°
   pitch = [-0.5, 0.5]
   ```

2. 增加目标奖励：
   ```python
   reach_flat_target = 150.0  # 从 100.0 增加
   ```

3. 使用渐进式训练（见 PROGRESSIVE_TRAINING.md）

### 问题 2: 切换后 Flat 网络失控

**症状**: 切换到 Flat 后立即倒地

**原因**: Flat 条件定义不准确

**解决方案**:
1. 检查 Flat 网络的实际初始条件：
   ```bash
   python play.py --task=wheel_legged_vmc_flat
   # 观察初始高度、姿态
   ```

2. 调整 Balance 的目标：
   ```python
   # balance.py
   self.flat_target_height = 0.XX  # 根据实际调整
   ```

3. 放宽切换条件：
   ```python
   self.flat_angle_threshold = 0.15  # 从 0.1 增加
   ```

### 问题 3: 切换不平滑

**症状**: 切换时有明显抖动

**解决方案**:
1. 添加切换缓冲区：
   ```python
   # 切换到 Flat: 严格条件
   # 切回 Balance: 宽松条件（hysteresis）
   ```

2. 使用软切换（混合策略）：
   ```python
   alpha = smooth_transition_weight()
   actions = alpha * flat_actions + (1-alpha) * balance_actions
   ```

### 问题 4: Balance 阶段太慢

**症状**: 需要 > 15 秒才能达到 Flat 条件

**解决方案**:
1. 增加 action_scale：
   ```python
   action_scale_theta = 1.5  # 从 1.0 增加
   ```

2. 增加姿态惩罚：
   ```python
   pitch_angle = -150.0  # 从 -100.0 增加
   roll_angle = -150.0
   ```

## 💡 优化建议

### 1. 渐进式训练

如果直接训练困难，使用 3 阶段：

```bash
# 阶段 1: 小角度恢复（500 iter）
roll = [-0.3, 0.3]  # ±17°
pitch = [-0.3, 0.3]

# 阶段 2: 中等角度恢复（500 iter，从阶段 1 继续）
roll = [-0.6, 0.6]  # ±34°
pitch = [-0.6, 0.6]

# 阶段 3: 大角度恢复（1000 iter，从阶段 2 继续）
roll = [-0.8, 0.8]  # ±45°
pitch = [-0.8, 0.8]
```

### 2. 课程学习

动态调整初始化难度：

```python
# 根据成功率调整
if success_rate > 0.8:
    increase_difficulty()
elif success_rate < 0.5:
    decrease_difficulty()
```

### 3. 奖励塑形

添加中间奖励：

```python
# 奖励接近目标
distance_to_target = compute_distance()
reward_approaching = -10.0 * distance_to_target
```

## 📈 预期训练曲线

### Balance 网络

```
Iterations 0-500:
- rew_reach_flat_target: 0 → 40
- 学习基本恢复能力

Iterations 500-1000:
- rew_reach_flat_target: 40 → 70
- 提高恢复成功率

Iterations 1000-2000:
- rew_reach_flat_target: 70 → 85+
- 稳定的恢复能力
```

### 两阶段系统

```
测试 10 次:
- 切换成功: 8-9 次
- 平均切换时间: 6-10 秒
- Flat 阶段时长: 20-30 秒
- 整体成功率: 80-90%
```

## 🎯 下一步

### 1. 训练 Balance 网络

```bash
python train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 2. 监控训练

```bash
tensorboard --logdir=logs/wheel_legged_vmc_balance
```

关注：
- `rew_reach_flat_target` 是否增加
- `rew_pitch_angle` / `rew_roll_angle` 是否接近 0

### 3. 测试系统

```bash
./test_two_stage.sh
```

观察：
- 能否从大角度恢复
- 切换是否平滑
- Flat 阶段是否稳定

### 4. 优化调整

根据测试结果调整：
- Flat 目标条件
- Balance 奖励权重
- 初始化范围

## 📚 相关文档

- [TWO_STAGE_CONTROL.md](TWO_STAGE_CONTROL.md) - 详细设计文档
- [PROGRESSIVE_TRAINING.md](PROGRESSIVE_TRAINING.md) - 渐进式训练
- [BALANCE_RECOVERY_TRADEOFF.md](BALANCE_RECOVERY_TRADEOFF.md) - 平衡与恢复

## 🎉 总结

你现在有了一个完整的两阶段控制系统：

1. **Balance 网络**: 从任意姿态恢复到 Flat 初始条件
2. **Flat 网络**: 从 Flat 初始条件保持平衡
3. **自动切换**: 达到条件时自动切换
4. **监控系统**: 实时显示状态和统计

**关键**: Flat 初始条件是两个网络的"交接点"，必须精确定义和训练。

---

**状态**: ✅ 系统已实现，准备训练和测试
**下一步**: 训练 Balance 网络，然后测试两阶段切换
