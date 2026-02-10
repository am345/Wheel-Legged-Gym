# 机器人学不会站高问题诊断报告

## 问题概述
尽管在代码中添加了随机高度指令（[`commands.ranges.height`](wheel_legged_gym/envs/base/legged_robot_config.py:101)），机器人仍然无法学会根据指令调整站立高度。

## 根本原因分析

### 🔴 问题1：高度指令的观测缩放不合理
**位置**: [`legged_robot.py:1000-1008`](wheel_legged_gym/envs/base/legged_robot.py:1000)

```python
self.commands_scale = torch.tensor(
    [
        self.obs_scales.lin_vel,      # 2.0
        self.obs_scales.ang_vel,      # 0.25
        self.obs_scales.height_measurements,  # 5.0 ❌ 问题所在
    ],
    device=self.device,
    requires_grad=False,
)
```

**问题**: 高度指令使用了 `height_measurements` (5.0) 作为缩放因子，但这个值是用于地形高度测量的，不适合高度指令。

**影响**: 
- 高度指令范围 [0.14, 0.31] 米
- 缩放后变成 [0.7, 1.55]
- 这个缩放值过大，导致神经网络难以学习高度指令的细微差异

### 🔴 问题2：高度奖励权重可能不足
**位置**: [`legged_robot_config.py:182`](wheel_legged_gym/envs/base/legged_robot_config.py:182)

```python
class scales:
    tracking_lin_vel = 1.0
    tracking_ang_vel = 1.0
    base_height = 3.0  # ⚠️ 可能需要调整
```

**问题**: 
- 线速度和角速度跟踪奖励权重都是 1.0
- 高度奖励权重是 3.0，看似较高
- 但由于高度变化范围小（0.14-0.31米），实际奖励信号可能被速度跟踪奖励淹没

### 🔴 问题3：观测空间中高度指令位置
**位置**: [`legged_robot.py:345`](wheel_legged_gym/envs/base/legged_robot.py:345)

```python
self.commands[:, :3] * self.commands_scale,
# commands[:, 0] = lin_vel_x
# commands[:, 1] = ang_vel_yaw  
# commands[:, 2] = height ✓ 已包含
```

**状态**: ✅ 高度指令已正确包含在观测中

### 🟡 问题4：高度指令范围可能过窄
**位置**: [`legged_robot_config.py:101`](wheel_legged_gym/envs/base/legged_robot_config.py:101)

```python
height = [0.14, 0.31]  # 仅 0.17 米的变化范围
```

**问题**: 
- 变化范围只有 17 厘米
- 对于轮腿机器人，这个范围可能不够明显
- 机器人可能更倾向于保持一个中间高度来平衡其他任务

### 🟡 问题5：base_height 计算方式
**位置**: [`legged_robot.py:616-618`](wheel_legged_gym/envs/base/legged_robot.py:616)

```python
self.base_height = torch.mean(
    self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1
)
```

**问题**: 
- 使用地形高度测量的平均值
- 在平地上 `measured_heights` 为 0，所以 `base_height` 就是 `root_states[:, 2]`（机器人基座的 z 坐标）
- 这个计算方式是正确的，但需要确认是否与期望的高度定义一致

### 🔴 问题6：奖励函数的敏感度
**位置**: [`legged_robot.py:1642-1649`](wheel_legged_gym/envs/base/legged_robot.py:1642)

```python
def _reward_base_height(self):
    if self.reward_scales["base_height"] < 0:
        return torch.abs(self.base_height - self.commands[:, 2])
    else:
        base_height_error = torch.square(self.base_height - self.commands[:, 2])
        return torch.exp(-base_height_error / 0.001)  # ⚠️ 敏感度极高
```

**问题**: 
- 当 `base_height_error = 0.001` 时，奖励降到 `exp(-1) ≈ 0.37`
- 这意味着 3.16 厘米的误差就会让奖励减半
- 对于动态运动的机器人来说，这个容差太小了

## 解决方案

### 方案1：修正观测缩放（推荐）⭐

在 [`legged_robot_config.py`](wheel_legged_gym/envs/base/legged_robot_config.py:209) 中添加高度指令的专用缩放：

```python
class obs_scales:
    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05
    dof_acc = 0.0025
    height_measurements = 5.0
    height_command = 5.0  # 新增：高度指令缩放
    torque = 0.05
```

然后在 [`legged_robot.py:1000`](wheel_legged_gym/envs/base/legged_robot.py:1000) 修改：

```python
self.commands_scale = torch.tensor(
    [
        self.obs_scales.lin_vel,
        self.obs_scales.ang_vel,
        self.obs_scales.height_command,  # 使用专用缩放
    ],
    device=self.device,
    requires_grad=False,
)
```

### 方案2：调整奖励函数敏感度（推荐）⭐

修改 [`_reward_base_height()`](wheel_legged_gym/envs/base/legged_robot.py:1642)：

```python
def _reward_base_height(self):
    if self.reward_scales["base_height"] < 0:
        return torch.abs(self.base_height - self.commands[:, 2])
    else:
        base_height_error = torch.square(self.base_height - self.commands[:, 2])
        return torch.exp(-base_height_error / 0.01)  # 从 0.001 改为 0.01
```

这样 10 厘米误差时奖励约为 0.37，更合理。

### 方案3：增加高度奖励权重

在配置文件中调整：

```python
class scales:
    tracking_lin_vel = 1.0
    tracking_ang_vel = 1.0
    base_height = 5.0  # 从 3.0 增加到 5.0
```

### 方案4：扩大高度指令范围（可选）

```python
height = [0.10, 0.35]  # 从 [0.14, 0.31] 扩大到 25 厘米范围
```

## 推荐实施顺序

1. **首先**：修正观测缩放（方案1）- 这是最关键的问题
2. **其次**：调整奖励函数敏感度（方案2）- 让学习更稳定
3. **然后**：增加高度奖励权重（方案3）- 如果前两步效果不明显
4. **最后**：扩大高度范围（方案4）- 如果需要更明显的高度变化

## 验证方法

训练后检查以下指标：
- `rew_base_height`: 应该逐渐增加
- 观察机器人是否能在不同高度指令下调整站立高度
- 检查 tensorboard 中高度跟踪误差是否下降

## 额外建议

考虑添加高度跟踪的增强奖励（类似于速度跟踪）：

```python
def _reward_base_height_enhance(self):
    base_height_error = torch.square(self.base_height - self.commands[:, 2])
    return torch.exp(-base_height_error / 0.1) - 1
```

并在配置中添加：

```python
base_height_enhance = 0.5
```
