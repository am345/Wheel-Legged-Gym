# 快速开始指南 - 两阶段控制系统

## 🎯 系统概述

两阶段控制系统：
1. **Balance 网络**: 从任意姿态 → 恢复到 Flat 初始条件
2. **Flat 网络**: 从 Flat 初始条件 → 保持平衡

**关键**: Flat 初始条件是两个网络的"交接点"
- 高度: 0.20m ± 5cm
- 姿态: ±5.7° (pitch/roll)
- 速度: < 0.2 m/s (线速度/角速度)

## 📋 前置条件

### 1. 确认 Flat 网络已训练

```bash
# 检查 Flat 模型
ls logs/wheel_legged_vmc_flat/

# 如果没有，先训练 Flat
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat --num_envs=4096
```

### 2. 验证 Flat 网络性能

```bash
# 测试 Flat 网络
python wheel_legged_gym/scripts/play.py --task=wheel_legged_vmc_flat

# 观察:
# - 能否在初始条件下保持平衡 > 20 秒
# - 初始高度是否约 0.20m
# - 初始姿态是否接近直立
```

## 🚀 训练 Balance 网络

### 方式 1: 使用快速脚本（推荐）

```bash
./train_balance.sh
```

### 方式 2: 手动训练

```bash
python wheel_legged_gym/scripts/train.py \
    --task=wheel_legged_vmc_balance \
    --num_envs=4096
```

### 训练参数

- **Environments**: 4096
- **Max iterations**: 2000
- **初始化范围**:
  - Roll/Pitch: ±45.8°
  - 速度: ±0.5 m/s
- **目标状态**: Flat 初始条件

## 📊 监控训练

### 启动 TensorBoard

```bash
tensorboard --logdir=logs/wheel_legged_vmc_balance
```

### 关键指标

| 指标 | 目标值 | 含义 |
|------|--------|------|
| `Train/rew_reach_flat_target` | > 80 | 达到 Flat 条件的比例 |
| `Train/rew_pitch_angle` | → 0 | Pitch 角度接近 0° |
| `Train/rew_roll_angle` | → 0 | Roll 角度接近 0° |
| `Train/rew_base_height` | > 40 | 高度接近目标 |
| `Train/mean_episode_length` | → max | Episode 长度接近最大值 |

### 预期训练曲线

```
Iterations 0-500:
  rew_reach_flat_target: 0 → 40
  学习基本恢复能力

Iterations 500-1000:
  rew_reach_flat_target: 40 → 70
  提高恢复成功率

Iterations 1000-2000:
  rew_reach_flat_target: 70 → 85+
  稳定的恢复能力
```

## 🧪 测试两阶段系统

### 方式 1: 使用快速脚本（推荐）

```bash
./test_two_stage.sh
```

### 方式 2: 手动运行

```bash
python wheel_legged_gym/scripts/play_two_stage.py
```

### 键盘控制

| 按键 | 功能 |
|------|------|
| C | 启动控制（Balance 阶段） |
| M | 手动切换阶段 |
| R | 重置环境 |
| S | 切换统计显示 |
| ESC | 退出 |

### 观察指标

**Balance 阶段**:
- 能否从大角度倾斜恢复
- 恢复时间（目标: 5-10 秒）
- 腿部摆动是否受控

**切换时刻**:
- 切换是否平滑
- 是否满足所有 Flat 条件

**Flat 阶段**:
- 能否保持平衡 > 20 秒
- 姿态是否稳定

## ✅ 成功标准

### Balance 网络训练成功

- ✅ `rew_reach_flat_target` > 80
- ✅ 能从 ±45° 恢复到 ±5°
- ✅ 平均 5-10 秒达到 Flat 条件

### 两阶段系统成功

- ✅ Balance 阶段: 5-10 秒达到 Flat 条件
- ✅ 切换成功率 > 80%
- ✅ Flat 阶段: 保持平衡 > 20 秒
- ✅ 整体平衡时长 > 25 秒

## 🔧 故障排除

### 问题 1: Balance 网络学不会恢复

**症状**: `rew_reach_flat_target` 一直很低

**解决方案**:
1. 降低初始化难度:
   ```python
   # balance_config.py
   roll = [-0.5, 0.5]  # 从 ±45° 降到 ±28°
   pitch = [-0.5, 0.5]
   ```

2. 增加目标奖励:
   ```python
   reach_flat_target = 150.0  # 从 100.0 增加
   ```

3. 使用渐进式训练（见 [PROGRESSIVE_TRAINING.md](PROGRESSIVE_TRAINING.md)）

### 问题 2: 切换后 Flat 网络失控

**症状**: 切换到 Flat 后立即倒地

**原因**: Flat 条件定义不准确

**解决方案**:
1. 检查 Flat 网络的实际初始条件:
   ```bash
   python wheel_legged_gym/scripts/play.py --task=wheel_legged_vmc_flat
   # 观察初始高度、姿态
   ```

2. 调整 Balance 的目标:
   ```python
   # balance.py
   self.flat_target_height = 0.XX  # 根据实际调整
   ```

3. 放宽切换条件:
   ```python
   self.flat_angle_threshold = 0.15  # 从 0.1 增加
   ```

### 问题 3: 切换不平滑

**症状**: 切换时有明显抖动

**解决方案**:
1. 添加切换缓冲区（hysteresis）
2. 增加切换条件的持续时间要求

### 问题 4: Balance 阶段太慢

**症状**: 需要 > 15 秒才能达到 Flat 条件

**解决方案**:
1. 增加 action_scale:
   ```python
   action_scale_theta = 1.5  # 从 1.0 增加
   ```

2. 增加姿态惩罚:
   ```python
   pitch_angle = -150.0  # 从 -100.0 增加
   roll_angle = -150.0
   ```

## 📁 文件结构

```
Wheel-Legged-Gym/
├── wheel_legged_gym/
│   ├── envs/
│   │   ├── wheel_legged_vmc_balance/
│   │   │   ├── wheel_legged_vmc_balance.py          # Balance 环境
│   │   │   └── wheel_legged_vmc_balance_config.py   # Balance 配置
│   │   └── wheel_legged_vmc_flat/
│   │       └── wheel_legged_vmc_flat_config.py      # Flat 配置
│   └── scripts/
│       ├── train.py                                  # 训练脚本
│       ├── play_two_stage.py                         # 两阶段测试
│       └── play.py                                   # 单阶段测试
├── logs/
│   ├── wheel_legged_vmc_balance/                     # Balance 训练日志
│   └── wheel_legged_vmc_flat/                        # Flat 训练日志
├── train_balance.sh                                  # Balance 训练快速脚本
├── test_two_stage.sh                                 # 两阶段测试快速脚本
└── 文档/
    ├── QUICK_START.md                                # 本文件
    ├── TWO_STAGE_IMPLEMENTATION.md                   # 完整实施指南
    ├── TWO_STAGE_CONTROL.md                          # 设计文档
    ├── PROGRESSIVE_TRAINING.md                       # 渐进式训练
    └── BALANCE_RECOVERY_TRADEOFF.md                  # 平衡与恢复权衡
```

## 📚 相关文档

- [TWO_STAGE_IMPLEMENTATION.md](TWO_STAGE_IMPLEMENTATION.md) - 完整实施指南
- [TWO_STAGE_CONTROL.md](TWO_STAGE_CONTROL.md) - 详细设计文档
- [PROGRESSIVE_TRAINING.md](PROGRESSIVE_TRAINING.md) - 渐进式训练方案
- [BALANCE_RECOVERY_TRADEOFF.md](BALANCE_RECOVERY_TRADEOFF.md) - 平衡与恢复权衡

## 🎯 下一步

### 1. 训练 Balance 网络

```bash
./train_balance.sh
```

### 2. 监控训练

```bash
tensorboard --logdir=logs/wheel_legged_vmc_balance
```

关注 `rew_reach_flat_target` 是否增加

### 3. 测试系统

```bash
./test_two_stage.sh
```

观察:
- 能否从大角度恢复
- 切换是否平滑
- Flat 阶段是否稳定

### 4. 优化调整

根据测试结果调整:
- Flat 目标条件
- Balance 奖励权重
- 初始化范围

---

**状态**: ✅ 系统已实现，准备训练和测试
**关键修正**: Flat 目标高度已更新为 0.20m（匹配实际机器人）
**下一步**: 运行 `./train_balance.sh` 开始训练
