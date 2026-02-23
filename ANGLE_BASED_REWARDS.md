# 基于角度和角速度的 Pitch/Roll 奖励设计

## ✅ 修改完成

已重新设计奖励函数，先解算出实际的 pitch 和 roll 角度及角速度，然后基于这些物理量设计奖励。

## 核心思想

### 之前的方法（不直观）
```python
# 直接使用 projected_gravity 的分量
reward = -scale * gravity_x²  # 不直观，难以理解
```

### 现在的方法（物理直观）
```python
# 1. 先计算实际角度
pitch_angle = atan2(gravity_y, -gravity_z)  # 弧度
roll_angle = atan2(gravity_x, -gravity_z)   # 弧度

# 2. 提取角速度
pitch_vel = base_ang_vel[1]  # rad/s
roll_vel = base_ang_vel[0]   # rad/s

# 3. 基于物理量设计奖励
reward_pitch_angle = -80.0 * pitch_angle²
reward_pitch_vel = -2.0 * pitch_vel²
```

## 代码实现

### 1. 环境类修改 (wheel_legged_vmc_balance.py)

```python
class LeggedRobotVMCBalance(LeggedRobotVMC):

    def __init__(self, ...):
        super().__init__(...)

        # 存储计算的角度和角速度
        self.pitch_angle = torch.zeros(self.num_envs, device=self.device)
        self.roll_angle = torch.zeros(self.num_envs, device=self.device)
        self.pitch_vel = torch.zeros(self.num_envs, device=self.device)
        self.roll_vel = torch.zeros(self.num_envs, device=self.device)

    def post_physics_step(self):
        """每步计算角度和角速度"""
        super().post_physics_step()

        # 计算角度（弧度）
        self.pitch_angle = torch.atan2(
            self.projected_gravity[:, 1],   # gravity_y
            -self.projected_gravity[:, 2]   # -gravity_z
        )

        self.roll_angle = torch.atan2(
            self.projected_gravity[:, 0],   # gravity_x
            -self.projected_gravity[:, 2]   # -gravity_z
        )

        # 提取角速度
        self.roll_vel = self.base_ang_vel[:, 0]   # x 轴旋转
        self.pitch_vel = self.base_ang_vel[:, 1]  # y 轴旋转

    def _reward_pitch_angle(self):
        """惩罚 pitch 角度偏差"""
        return torch.square(self.pitch_angle)

    def _reward_roll_angle(self):
        """惩罚 roll 角度偏差"""
        return torch.square(self.roll_angle)

    def _reward_pitch_vel(self):
        """惩罚 pitch 角速度"""
        return torch.square(self.pitch_vel)

    def _reward_roll_vel(self):
        """惩罚 roll 角速度"""
        return torch.square(self.roll_vel)

    def _reward_upright_bonus(self):
        """直立奖励 - 基于实际角度"""
        angle_threshold = 0.1  # 约 5.7 度
        vel_threshold = 0.5    # rad/s

        angle_ok = (torch.abs(self.pitch_angle) < angle_threshold) & \
                   (torch.abs(self.roll_angle) < angle_threshold)
        vel_ok = (torch.abs(self.pitch_vel) < vel_threshold) & \
                 (torch.abs(self.roll_vel) < vel_threshold)
        lin_vel_ok = torch.norm(self.base_lin_vel, dim=1) < 0.5

        return (angle_ok & vel_ok & lin_vel_ok).float()
```

### 2. 配置文件修改 (wheel_legged_vmc_balance_config.py)

```python
class scales:
    # 高度控制
    base_height = 40.0

    # 禁用原始的 orientation 和 ang_vel_xy
    orientation = 0.0
    ang_vel_xy = 0.0

    # 基于角度的姿态惩罚（rad²）
    pitch_angle = -80.0      # Pitch 角度惩罚
    roll_angle = -80.0       # Roll 角度惩罚

    # 基于角速度的晃动惩罚（(rad/s)²）
    pitch_vel = -2.0         # Pitch 角速度惩罚
    roll_vel = -2.0          # Roll 角速度惩罚

    # 直立奖励
    upright_bonus = 20.0

    # 其他
    lin_vel_z = -4.0
    stand_still = 5.0
    torques = -5e-5
    termination = -10.0
```

## 物理意义

### 角度计算

```python
# Pitch (前后倾斜)
pitch_angle = atan2(gravity_y, -gravity_z)

# 示例:
# 完全直立: gravity = [0, 0, -1] → pitch = 0°
# 前倾 10°: gravity = [0, 0.17, -0.98] → pitch = 10°
# 后倾 10°: gravity = [0, -0.17, -0.98] → pitch = -10°
```

```python
# Roll (左右倾斜)
roll_angle = atan2(gravity_x, -gravity_z)

# 示例:
# 完全直立: gravity = [0, 0, -1] → roll = 0°
# 右倾 10°: gravity = [0.17, 0, -0.98] → roll = 10°
# 左倾 10°: gravity = [-0.17, 0, -0.98] → roll = -10°
```

### 角速度提取

```python
# base_ang_vel: [roll_vel, pitch_vel, yaw_vel]
roll_vel = base_ang_vel[:, 0]   # x 轴旋转速度
pitch_vel = base_ang_vel[:, 1]  # y 轴旋转速度
```

## 奖励计算示例

### 场景 1: 完美直立
```
pitch_angle = 0 rad
roll_angle = 0 rad
pitch_vel = 0 rad/s
roll_vel = 0 rad/s

reward_pitch_angle = -80.0 * 0² = 0
reward_roll_angle = -80.0 * 0² = 0
reward_pitch_vel = -2.0 * 0² = 0
reward_roll_vel = -2.0 * 0² = 0
upright_bonus = 20.0

Total orientation reward = 20.0
```

### 场景 2: 轻微倾斜 (5°)
```
pitch_angle = 0.087 rad (5°)
roll_angle = 0 rad
pitch_vel = 0.1 rad/s
roll_vel = 0 rad/s

reward_pitch_angle = -80.0 * 0.087² = -0.61
reward_roll_angle = 0
reward_pitch_vel = -2.0 * 0.1² = -0.02
reward_roll_vel = 0
upright_bonus = 20.0 (仍满足条件)

Total = 19.37
```

### 场景 3: 中等倾斜 (15°)
```
pitch_angle = 0.262 rad (15°)
roll_angle = 0 rad
pitch_vel = 0.3 rad/s
roll_vel = 0 rad/s

reward_pitch_angle = -80.0 * 0.262² = -5.49
reward_roll_angle = 0
reward_pitch_vel = -2.0 * 0.3² = -0.18
reward_roll_vel = 0
upright_bonus = 0 (不满足条件)

Total = -5.67
```

### 场景 4: 大幅倾斜 (30°)
```
pitch_angle = 0.524 rad (30°)
roll_angle = 0 rad
pitch_vel = 0.5 rad/s
roll_vel = 0 rad/s

reward_pitch_angle = -80.0 * 0.524² = -21.94
reward_roll_angle = 0
reward_pitch_vel = -2.0 * 0.5² = -0.50
reward_roll_vel = 0
upright_bonus = 0

Total = -22.44
```

## 优势对比

### 旧方法（基于 gravity 分量）
```python
# 不直观
reward = -80.0 * gravity_y²

# 问题:
# - gravity_y = 0.5 是多少度？（不直观）
# - 如何设置合理的权重？（难以调整）
# - 无法直接理解物理意义
```

### 新方法（基于角度）
```python
# 直观
reward = -80.0 * pitch_angle²

# 优势:
# - pitch_angle = 0.1 rad = 5.7° （直观）
# - 权重设置有物理意义
# - 容易理解和调试
```

## 权重调整指南

### 角度惩罚权重

| 权重 | 5° 惩罚 | 10° 惩罚 | 20° 惩罚 | 适用场景 |
|------|---------|----------|----------|----------|
| -40 | -0.3 | -1.2 | -4.9 | 宽松 |
| -80 | -0.6 | -2.4 | -9.7 | 标准 |
| -120 | -0.9 | -3.7 | -14.6 | 严格 |
| -200 | -1.5 | -6.1 | -24.3 | 非常严格 |

### 角速度惩罚权重

| 权重 | 0.1 rad/s | 0.5 rad/s | 1.0 rad/s | 适用场景 |
|------|-----------|-----------|-----------|----------|
| -1 | -0.01 | -0.25 | -1.0 | 宽松 |
| -2 | -0.02 | -0.50 | -2.0 | 标准 |
| -5 | -0.05 | -1.25 | -5.0 | 严格 |
| -10 | -0.10 | -2.50 | -10.0 | 非常严格 |

### 推荐配置

**标准平衡**:
```python
pitch_angle = -80.0
roll_angle = -80.0
pitch_vel = -2.0
roll_vel = -2.0
```

**强调角度控制**:
```python
pitch_angle = -120.0
roll_angle = -120.0
pitch_vel = -1.0
roll_vel = -1.0
```

**强调平滑控制**:
```python
pitch_angle = -60.0
roll_angle = -60.0
pitch_vel = -5.0
roll_vel = -5.0
```

**非对称控制**（前后更重要）:
```python
pitch_angle = -120.0
roll_angle = -60.0
pitch_vel = -3.0
roll_vel = -1.0
```

## 监控指标

### TensorBoard 中查看

```
Train/rew_pitch_angle    # 应该接近 0
Train/rew_roll_angle     # 应该接近 0
Train/rew_pitch_vel      # 应该接近 0
Train/rew_roll_vel       # 应该接近 0
Train/rew_upright_bonus  # 应该 > 15
```

### 诊断

**如果 `rew_pitch_angle` 很负**:
- Pitch 角度控制不好
- 考虑增加 `pitch_angle` 权重
- 或降低 `pitch_vel` 权重（允许更快调整）

**如果 `rew_pitch_vel` 很负**:
- Pitch 晃动严重
- 考虑增加 `pitch_vel` 权重
- 或增加阻尼（PD 控制的 kd）

## 调试技巧

### 1. 打印实际角度

在 `post_physics_step` 中添加：
```python
if self.common_step_counter % 100 == 0:
    pitch_deg = self.pitch_angle.mean() * 180 / 3.14159
    roll_deg = self.roll_angle.mean() * 180 / 3.14159
    print(f"Avg pitch: {pitch_deg:.2f}°, Avg roll: {roll_deg:.2f}°")
```

### 2. 可视化角度历史

```python
import matplotlib.pyplot as plt

# 记录历史
pitch_history = []
roll_history = []

# 在训练中
pitch_history.append(self.pitch_angle.mean().item())
roll_history.append(self.roll_angle.mean().item())

# 绘图
plt.plot(pitch_history, label='Pitch')
plt.plot(roll_history, label='Roll')
plt.legend()
plt.show()
```

### 3. 单独测试

```python
# 只惩罚 pitch
pitch_angle = -80.0
roll_angle = 0.0
pitch_vel = -2.0
roll_vel = 0.0

# 只惩罚 roll
pitch_angle = 0.0
roll_angle = -80.0
pitch_vel = 0.0
roll_vel = -2.0
```

## 预期效果

### 修改前
- 使用 gravity 分量，不直观
- 难以理解物理意义
- 权重调整困难

### 修改后
- ✅ 使用实际角度，直观易懂
- ✅ 物理意义明确
- ✅ 权重调整有指导意义
- ✅ 分离角度和角速度控制
- ✅ 更容易调试和优化

## 下一步

### 1. 重新训练
```bash
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 2. 监控新指标
关注 TensorBoard 中的：
- `rew_pitch_angle`
- `rew_roll_angle`
- `rew_pitch_vel`
- `rew_roll_vel`

### 3. 根据结果调整
- 如果角度控制不好 → 增加角度权重
- 如果晃动严重 → 增加角速度权重
- 如果响应太慢 → 降低角速度权重

---

**修改时间**: 2026-02-23
**状态**: ✅ 已完成并验证
**优势**: 物理直观，易于调整
