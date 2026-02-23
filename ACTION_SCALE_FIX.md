# Action Scale 调整说明

## 问题诊断

### 现象
仿真中腿一直在乱摆，无法稳定平衡。

### 原因
```python
action_scale_theta = 3.14  # 180度 - 太大了！
```

这导致：
- 策略输出 `action = 1.0` → `theta0_ref` 变化 **180°**
- 策略输出 `action = -1.0` → `theta0_ref` 变化 **-180°**
- 腿部剧烈摆动，完全无法控制

## 核心矛盾

### 平衡任务 vs 正周转

你提到希望腿可以"正周转"（转一圈），但这与平衡任务冲突：

| 任务类型 | theta0 范围 | action_scale_theta | 行为 |
|----------|-------------|-------------------|------|
| **平衡** | ±10° | 0.1 - 0.3 | 腿保持垂直，小幅调整 |
| **行走** | ±30° | 0.5 - 1.0 | 腿前后摆动 |
| **正周转** | ±180° | 3.14 | 腿可以转一圈 |

**关键**: 平衡任务中，腿应该保持接近垂直（theta0 ≈ 0），不需要大幅度摆动！

## 修复方案

### 方案 1: 平衡任务（推荐）

```python
action_scale_theta = 0.2   # 约 11.5 度
action_scale_l0 = 0.05     # 5cm
action_scale_vel = 8.0     # 轮速
```

**效果**:
- 策略输出 `action = 1.0` → theta0 变化 **11.5°**
- 策略输出 `action = 0.5` → theta0 变化 **5.7°**
- 腿部小幅度调整，足够平衡控制

**适用场景**:
- ✅ 静态平衡
- ✅ 原地保持
- ✅ 小幅度扰动恢复

### 方案 2: 保守平衡

如果机器人还是不稳定，进一步降低：

```python
action_scale_theta = 0.1   # 约 5.7 度
action_scale_l0 = 0.03     # 3cm
action_scale_vel = 5.0     # 更慢的轮速
```

**效果**: 更平滑，但响应更慢

### 方案 3: 激进平衡

如果需要更快的响应：

```python
action_scale_theta = 0.3   # 约 17 度
action_scale_l0 = 0.07     # 7cm
action_scale_vel = 10.0    # 更快的轮速
```

**效果**: 响应更快，但可能不稳定

### 方案 4: 行走任务（不是平衡）

如果你真的需要正周转（行走/跑步）：

```python
action_scale_theta = 1.0   # 约 57 度
action_scale_l0 = 0.1      # 10cm
action_scale_vel = 15.0    # 快速轮速
```

**注意**: 这需要完全不同的奖励函数和训练策略！

## 物理意义

### action_scale_theta 的含义

```python
# 在 VMC 控制中
theta0_ref = action * action_scale_theta

# 示例
action_scale_theta = 0.2
action = 1.0
theta0_ref = 1.0 * 0.2 = 0.2 rad = 11.5°
```

### 不同 scale 的效果

| action_scale_theta | action=1.0 时的角度 | 适用任务 |
|-------------------|-------------------|----------|
| 0.05 | 2.9° | 极保守平衡 |
| 0.1 | 5.7° | 保守平衡 |
| 0.2 | 11.5° | **标准平衡** ✅ |
| 0.3 | 17.2° | 激进平衡 |
| 0.5 | 28.6° | 慢速行走 |
| 1.0 | 57.3° | 快速行走 |
| 3.14 | 180° | ❌ 太大，无法控制 |

## 推荐配置

### 当前修复（已应用）

```python
class control:
    # 平衡任务 - 小幅度调整
    action_scale_theta = 0.2   # 11.5 度
    action_scale_l0 = 0.05     # 5cm
    action_scale_vel = 8.0     # 轮速

    # PD 增益
    kp_theta = 8.0
    kd_theta = 4.0
    kp_l0 = 600.0
    kd_l0 = 6.0
```

### 配合的奖励函数

```python
class scales:
    # 惩罚腿部摆角偏离垂直
    leg_angle_zero = -10.0   # 鼓励 theta0 = 0

    # 姿态控制
    pitch_angle = -80.0
    roll_angle = -80.0
    pitch_vel = -2.0
    roll_vel = -2.0

    # 高度控制
    base_height = 40.0
```

**关键**: `leg_angle_zero` 奖励会鼓励腿保持垂直，配合小的 `action_scale_theta`，腿不会乱摆。

## 调试步骤

### 1. 验证修改

```bash
python3 -m py_compile wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance_config.py
```

### 2. 重新训练

```bash
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 3. 监控指标

在 TensorBoard 中查看：
- `Train/rew_leg_angle_zero` - 应该接近 0（腿保持垂直）
- `Train/rew_pitch_angle` - 应该接近 0
- `Train/rew_roll_angle` - 应该接近 0

### 4. 可视化测试

```bash
python wheel_legged_gym/scripts/play_balance.py
```

观察：
- ✅ 腿部应该保持接近垂直
- ✅ 只有小幅度摆动
- ✅ 机器人能稳定平衡

## 如果还是不稳定

### 诊断清单

1. **腿还在乱摆**:
   - 降低 `action_scale_theta` 到 0.1
   - 增加 `leg_angle_zero` 惩罚到 -20.0

2. **响应太慢**:
   - 增加 `kp_theta` 到 10.0
   - 增加 `action_scale_theta` 到 0.3

3. **振荡不停**:
   - 增加 `kd_theta` 到 5.0
   - 增加 `pitch_vel` 和 `roll_vel` 惩罚

4. **倒地太快**:
   - 降低 `action_scale_theta` 到 0.1
   - 增加 `pitch_angle` 和 `roll_angle` 惩罚

## 理解 VMC 控制

### 控制流程

```python
# 1. 策略输出动作
action = policy(obs)  # [-1, 1]

# 2. 缩放到参考值
theta0_ref = action * action_scale_theta  # 弧度
l0_ref = action * action_scale_l0 + l0_offset  # 米

# 3. PD 控制
torque = kp_theta * (theta0_ref - theta0) - kd_theta * theta0_dot
force = kp_l0 * (l0_ref - L0) - kd_l0 * L0_dot

# 4. VMC 转换到关节力矩
joint_torques = VMC(torque, force)
```

### 关键参数关系

```
action_scale_theta ↑ → 腿摆动幅度 ↑ → 不稳定 ↑
kp_theta ↑ → 响应速度 ↑ → 可能振荡 ↑
kd_theta ↑ → 阻尼 ↑ → 平滑 ↑
leg_angle_zero ↓ (更负) → 腿更垂直 → 更稳定
```

## 总结

### 核心问题
`action_scale_theta = 3.14` 太大，导致腿部剧烈摆动。

### 解决方案
降低到 `action_scale_theta = 0.2`，配合 `leg_angle_zero = -10.0` 奖励。

### 预期效果
- ✅ 腿部保持接近垂直
- ✅ 只有小幅度调整
- ✅ 机器人能稳定平衡
- ✅ 不会乱摆

### 如果需要正周转
那不是平衡任务，需要：
1. 创建新的行走任务
2. 使用不同的奖励函数
3. 允许更大的 `action_scale_theta`
4. 不惩罚 `leg_angle_zero`

---

**修改时间**: 2026-02-23
**状态**: ✅ 已修复
**关键**: action_scale_theta 从 3.14 降低到 0.2
