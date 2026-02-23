# Pitch 和 Roll 分开惩罚

## 修改内容

将原来的 `orientation` 奖励（同时惩罚 pitch 和 roll）拆分为两个独立的奖励函数，可以分别调整前后倾斜和左右倾斜的惩罚力度。

## 代码修改

### 1. 新增奖励函数 (wheel_legged_vmc_balance.py)

```python
def _reward_orientation_pitch(self):
    """Penalize pitch (forward/backward tilt) - gravity_y component"""
    return torch.square(self.projected_gravity[:, 1])

def _reward_orientation_roll(self):
    """Penalize roll (left/right tilt) - gravity_x component"""
    return torch.square(self.projected_gravity[:, 0])
```

### 2. 配置文件修改 (wheel_legged_vmc_balance_config.py)

```python
# 禁用原始的 orientation
orientation = 0.0

# 新增分开的惩罚
orientation_pitch = -80.0  # Pitch 惩罚（前后倾斜）
orientation_roll = -80.0   # Roll 惩罚（左右倾斜）
```

## 工作原理

### Projected Gravity 向量

机器人的 `projected_gravity` 是重力在机器人坐标系中的投影：
- `gravity[0]` (x): Roll 方向（左右倾斜）
- `gravity[1]` (y): Pitch 方向（前后倾斜）
- `gravity[2]` (z): 垂直方向（应该接近 -1）

### 倾斜角度计算

```python
# 完全直立时
gravity = [0, 0, -1]  # 重力完全向下

# Roll 倾斜 θ 度
gravity_x = sin(θ)
roll_angle = arctan2(gravity_x, -gravity_z) * 180/π

# Pitch 倾斜 φ 度
gravity_y = sin(φ)
pitch_angle = arctan2(gravity_y, -gravity_z) * 180/π
```

### 奖励计算

```python
# Pitch 惩罚
reward_pitch = -80.0 * gravity_y²

# Roll 惩罚
reward_roll = -80.0 * gravity_x²

# 总姿态惩罚
total_orientation_penalty = reward_pitch + reward_roll
```

## 优势

### 1. 独立调整
可以根据机器人特性分别调整：
```python
# 如果机器人前后倾斜更危险
orientation_pitch = -100.0
orientation_roll = -60.0

# 如果机器人左右倾斜更危险
orientation_pitch = -60.0
orientation_roll = -100.0
```

### 2. 更精细的控制
- **Pitch**: 控制前后平衡（通常由轮子速度控制）
- **Roll**: 控制左右平衡（通常由腿部角度控制）

### 3. 调试更容易
可以在 TensorBoard 中分别查看：
- `Train/rew_orientation_pitch`
- `Train/rew_orientation_roll`

了解哪个方向的平衡更困难。

## 对比

### 原始方法
```python
# 同时惩罚 pitch 和 roll
orientation = -80.0
reward = -80.0 * (gravity_x² + gravity_y²)
```

**问题**: 无法区分是前后倾斜还是左右倾斜导致的惩罚。

### 新方法
```python
# 分开惩罚
orientation_pitch = -80.0
orientation_roll = -80.0
reward_pitch = -80.0 * gravity_y²
reward_roll = -80.0 * gravity_x²
total = reward_pitch + reward_roll
```

**优势**:
- 可以独立调整权重
- 可以分别监控
- 更容易诊断问题

## 使用场景

### 场景 1: 标准平衡（默认）
```python
orientation_pitch = -80.0
orientation_roll = -80.0
```
两个方向同等重要。

### 场景 2: 强调前后平衡
```python
orientation_pitch = -120.0  # 更严格
orientation_roll = -60.0    # 较宽松
```
适合轮足机器人，前后倾斜更危险。

### 场景 3: 强调左右平衡
```python
orientation_pitch = -60.0   # 较宽松
orientation_roll = -120.0   # 更严格
```
适合窄底盘机器人，左右倾斜更危险。

### 场景 4: 渐进式训练
```python
# 阶段 1: 先学会左右平衡
orientation_pitch = -40.0
orientation_roll = -120.0

# 阶段 2: 再学会前后平衡
orientation_pitch = -120.0
orientation_roll = -120.0
```

## 监控指标

### TensorBoard 中查看

训练时关注这些指标：
```
Train/rew_orientation_pitch  # 应该接近 0（不是 -40）
Train/rew_orientation_roll   # 应该接近 0（不是 -40）
```

### 理想值

| 倾斜角度 | Pitch 惩罚 | Roll 惩罚 | 说明 |
|----------|------------|-----------|------|
| 0° | 0.0 | 0.0 | 完美直立 |
| 5° | -0.6 | -0.6 | 优秀 |
| 10° | -2.4 | -2.4 | 良好 |
| 20° | -9.5 | -9.5 | 不稳定 |
| 30° | -20.0 | -20.0 | 危险 |

### 诊断

**如果 `rew_orientation_pitch` 很负**:
- 机器人前后倾斜严重
- 可能需要调整轮速控制
- 考虑增加 `orientation_pitch` 权重

**如果 `rew_orientation_roll` 很负**:
- 机器人左右倾斜严重
- 可能需要调整腿部角度控制
- 考虑增加 `orientation_roll` 权重

## 调试技巧

### 1. 可视化倾斜方向

在 `play_balance.py` 中已经显示：
```
Roll:   +2.34 deg  # 正值 = 向右倾斜
Pitch:  -1.87 deg  # 负值 = 向前倾斜
```

### 2. 记录历史数据

```python
# 在训练中添加
if self.common_step_counter % 100 == 0:
    pitch = torch.arctan2(self.projected_gravity[:, 1], -self.projected_gravity[:, 2])
    roll = torch.arctan2(self.projected_gravity[:, 0], -self.projected_gravity[:, 2])
    print(f"Avg pitch: {pitch.mean()*180/3.14:.2f}°, Avg roll: {roll.mean()*180/3.14:.2f}°")
```

### 3. 单独测试

```python
# 只惩罚 pitch，看机器人行为
orientation_pitch = -80.0
orientation_roll = 0.0

# 只惩罚 roll，看机器人行为
orientation_pitch = 0.0
orientation_roll = -80.0
```

## 预期效果

### 修改前
- 无法区分 pitch 和 roll 的问题
- 调试困难
- 权重调整不灵活

### 修改后
- ✅ 可以分别监控 pitch 和 roll
- ✅ 可以独立调整权重
- ✅ 更容易诊断平衡问题
- ✅ 支持渐进式训练策略

## 下一步

### 1. 重新训练
```bash
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 2. 监控新指标
在 TensorBoard 中查看：
- `Train/rew_orientation_pitch`
- `Train/rew_orientation_roll`

### 3. 根据结果调整
如果发现某个方向的倾斜更严重，增加对应的权重。

## 相关文档

- [READY_TO_TRAIN.md](READY_TO_TRAIN.md) - 训练指南
- [REWARD_FIX_PLAN.md](REWARD_FIX_PLAN.md) - 奖励修复方案
- [PLAY_BALANCE_GUIDE.md](PLAY_BALANCE_GUIDE.md) - 测试指南

---

**修改时间**: 2026-02-23
**状态**: ✅ 已完成
**优势**: 更精细的姿态控制
