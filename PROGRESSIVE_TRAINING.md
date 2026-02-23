# 渐进式平衡训练方案

## 问题分析

你的观察非常准确：
- **当前**: 只能在平衡点附近保持平衡（小角度恢复）
- **问题**: 无法从大角度倾斜恢复（需要正周转）
- **原因**: `action_scale_theta = 0.2` 太小 + `leg_angle_zero` 惩罚限制了腿部摆动

## 核心矛盾

```
小 action_scale → 稳定但恢复能力弱
大 action_scale → 恢复能力强但容易失控
```

## 解决方案：渐进式训练

### 阶段 1: 小角度平衡（当前）

**目标**: 学会在平衡点附近保持稳定

```python
# 配置
action_scale_theta = 0.2   # 小幅度
leg_angle_zero = -10.0     # 惩罚腿摆动

# 初始化
roll = [0.0, 0.0]          # 完全直立
pitch = [0.0, 0.0]
```

**训练**: 500-1000 iterations

**效果**: 机器人学会小角度平衡

---

### 阶段 2: 中等角度恢复

**目标**: 学会从中等倾斜恢复

```python
# 配置
action_scale_theta = 0.5   # 增加到 28.6°
leg_angle_zero = -5.0      # 减少惩罚

# 初始化（增加扰动）
roll = [-0.2, 0.2]         # ±11.5°
pitch = [-0.2, 0.2]
```

**训练**: 从阶段1的模型继续训练 500 iterations

**效果**: 机器人学会中等角度恢复

---

### 阶段 3: 大角度恢复（正周转）

**目标**: 学会从大角度倾斜恢复

```python
# 配置
action_scale_theta = 1.0   # 增加到 57.3°
leg_angle_zero = 0.0       # 移除惩罚！

# 初始化（大扰动）
roll = [-0.5, 0.5]         # ±28.6°
pitch = [-0.5, 0.5]
```

**训练**: 从阶段2的模型继续训练 500 iterations

**效果**: 机器人学会大角度恢复（正周转）

---

### 阶段 4: 极限恢复（可选）

**目标**: 学会从极限角度恢复

```python
# 配置
action_scale_theta = 2.0   # 114.6°
leg_angle_zero = 0.0

# 初始化（极限扰动）
roll = [-0.8, 0.8]         # ±45.8°
pitch = [-0.8, 0.8]
```

**训练**: 从阶段3的模型继续训练 500 iterations

**效果**: 机器人可以从接近倒地的状态恢复

## 实施步骤

### 1. 创建阶段配置文件

创建 4 个配置文件：
- `wheel_legged_vmc_balance_stage1.py` - 小角度
- `wheel_legged_vmc_balance_stage2.py` - 中等角度
- `wheel_legged_vmc_balance_stage3.py` - 大角度
- `wheel_legged_vmc_balance_stage4.py` - 极限角度

### 2. 训练脚本

```bash
# 阶段 1: 小角度平衡
python train.py --task=wheel_legged_vmc_balance_stage1 --num_envs=4096

# 阶段 2: 中等角度恢复（加载阶段1模型）
python train.py --task=wheel_legged_vmc_balance_stage2 --num_envs=4096 \
    --load_run=logs/wheel_legged_vmc_balance_stage1/最新日志

# 阶段 3: 大角度恢复（加载阶段2模型）
python train.py --task=wheel_legged_vmc_balance_stage3 --num_envs=4096 \
    --load_run=logs/wheel_legged_vmc_balance_stage2/最新日志

# 阶段 4: 极限恢复（加载阶段3模型）
python train.py --task=wheel_legged_vmc_balance_stage4 --num_envs=4096 \
    --load_run=logs/wheel_legged_vmc_balance_stage3/最新日志
```

### 3. 监控指标

每个阶段关注：
- `mean_episode_length` - 应该接近最大值
- `rew_pitch_angle` / `rew_roll_angle` - 应该接近 0
- `rew_leg_angle_zero` - 阶段3/4可以忽略

## 参数对比表

| 阶段 | action_scale_theta | leg_angle_zero | 初始角度 | 恢复能力 |
|------|-------------------|----------------|----------|----------|
| 1 | 0.2 (11.5°) | -10.0 | 0° | 小角度 |
| 2 | 0.5 (28.6°) | -5.0 | ±11.5° | 中等角度 |
| 3 | 1.0 (57.3°) | 0.0 | ±28.6° | 大角度 |
| 4 | 2.0 (114.6°) | 0.0 | ±45.8° | 极限角度 |

## 关键洞察

### 为什么需要渐进式训练？

1. **探索-利用权衡**:
   - 小 scale: 容易学习稳定策略，但探索空间小
   - 大 scale: 探索空间大，但难以学习稳定策略

2. **课程学习**:
   - 先学简单的（小角度平衡）
   - 再学复杂的（大角度恢复）
   - 避免一开始就面对困难任务

3. **策略迁移**:
   - 阶段1学到的"保持平衡"策略
   - 可以迁移到阶段2/3/4
   - 只需要学习"如何恢复"

### leg_angle_zero 的作用变化

```python
# 阶段 1-2: 惩罚腿摆动
leg_angle_zero = -10.0  # 鼓励腿保持垂直

# 阶段 3-4: 允许腿摆动
leg_angle_zero = 0.0    # 不限制腿部运动
```

**原因**: 大角度恢复时，腿必须大幅度摆动（正周转），不能惩罚！

## 替代方案

### 方案 2: 自适应 action_scale

动态调整 action_scale，根据倾斜角度：

```python
def compute_action_scale(self):
    # 根据倾斜角度动态调整
    tilt = torch.sqrt(self.pitch_angle**2 + self.roll_angle**2)

    # 小角度 → 小 scale
    # 大角度 → 大 scale
    scale = 0.2 + 2.0 * torch.tanh(tilt / 0.5)
    return scale
```

**优点**: 自动适应
**缺点**: 实现复杂，可能不稳定

### 方案 3: 混合奖励

同时鼓励平衡和恢复能力：

```python
# 奖励函数
leg_angle_zero = -5.0      # 轻微惩罚
recovery_bonus = 10.0      # 奖励从大角度恢复

# 配置
action_scale_theta = 0.8   # 中等值
```

**优点**: 一次训练完成
**缺点**: 可能两个目标都学不好

## 推荐实施

### 快速验证（2阶段）

```python
# 阶段 1: 基础平衡
action_scale_theta = 0.2
leg_angle_zero = -10.0
初始角度 = 0°
训练 1000 iterations

# 阶段 2: 恢复能力
action_scale_theta = 1.0
leg_angle_zero = 0.0
初始角度 = ±30°
从阶段1继续训练 1000 iterations
```

### 完整训练（4阶段）

按照上面的4阶段方案，每阶段 500-1000 iterations。

## 预期效果

### 阶段 1 后
- ✅ 能在平衡点附近保持稳定
- ❌ 无法从大角度恢复

### 阶段 2 后
- ✅ 能从 ±20° 恢复
- ❌ 无法从 ±40° 恢复

### 阶段 3 后
- ✅ 能从 ±40° 恢复（正周转）
- ✅ 鲁棒性大幅提升

### 阶段 4 后
- ✅ 能从接近倒地状态恢复
- ✅ 极强的鲁棒性

## 总结

你的观察揭示了一个重要问题：
- **当前训练**: 只是"局部平衡"，不是"全局平衡"
- **解决方案**: 渐进式训练，逐步增加难度和 action_scale
- **关键**: 阶段3/4 必须移除 `leg_angle_zero` 惩罚，允许正周转

这就是为什么很多机器人学习论文都使用课程学习（Curriculum Learning）的原因！

---

**建议**: 先完成阶段1训练，验证基础平衡能力，然后再进行阶段2/3的恢复能力训练。
