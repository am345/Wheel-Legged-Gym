# Balance 任务问题修复说明

## 问题描述

### 问题 1: 机器人自转且 pitch 不平
- 机器人在恢复平衡时会绕 yaw 轴自转
- pitch 轴（前后倾斜）不能保持水平
- 无法保持原地静止

### 问题 2: 腿部收腿策略不合理
- 当腿在前方时，机器人直接收腿
- 正确做法应该是：先将腿摆到后方，再收腿保持平衡
- 这样更符合物理直觉，也更稳定

## 解决方案

### 1. 增强静止约束

#### 新增奖励函数

**`_reward_ang_vel_yaw()`** - 惩罚 yaw 自转
```python
# 严格惩罚 yaw 角速度
return torch.square(self.base_ang_vel[:, 2])
```
- 权重: `-1.0`
- 作用: 防止机器人绕竖直轴旋转

**`_reward_base_lin_vel_xy()`** - 惩罚 xy 平面移动
```python
# 惩罚 x 和 y 方向的线速度
return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
```
- 权重: `-0.5`
- 作用: 保持机器人原地不动

#### 调整现有奖励

- `ang_vel_xy`: `-0.2` → `-0.5` (增加 roll/pitch 角速度惩罚)
- `stand_still`: `1.0` → `2.0` (增加静止奖励)

### 2. 约束髋关节位置

#### 新增奖励函数

**`_reward_hip_pos_constraint()`** - 引导腿部摆动策略
```python
# 目标：髋关节保持在 -0.1 rad (略微向后)
target_hip_pos = -0.1

# 如果腿在前方（hip_pos > 0），给予双倍惩罚
penalty = torch.where(
    hip_pos > 0.0,
    forward_penalty * 2.0 + deviation,  # 腿在前方
    deviation * 0.5  # 腿在后方
)
```
- 权重: `-8.0`
- 作用:
  - 强烈惩罚腿在前方的姿态
  - 鼓励髋关节保持在 -0.2 到 0.1 rad 范围
  - 引导机器人先摆腿再收腿

## 工作原理

### 场景 1: 腿在前方（图二）

1. `hip_pos > 0` → 触发双倍惩罚
2. 机器人学会先将髋关节向后摆动
3. 当 `hip_pos < 0` 时，惩罚减小
4. 然后再收腿（调整小腿关节）

### 场景 2: 腿在后方

1. `hip_pos < 0` → 较小惩罚
2. 机器人可以直接收腿
3. 保持髋关节在 -0.1 rad 附近

### 场景 3: 保持平衡

1. `ang_vel_yaw` 惩罚 → 减少自转
2. `base_lin_vel_xy` 惩罚 → 保持原地
3. `ang_vel_xy` 惩罚 → 保持 pitch/roll 稳定

## 预期效果

### 修复前
- ❌ 机器人会自转
- ❌ pitch 轴不平
- ❌ 腿在前方时直接收腿（不稳定）

### 修复后
- ✅ 机器人保持原地不动
- ✅ pitch 和 roll 都保持水平
- ✅ 腿在前方时先摆到后方再收腿
- ✅ 更稳定的平衡策略

## 训练建议

### 监控指标

在 TensorBoard 中重点关注：

1. **静止性能**
   - `Train/rew_ang_vel_yaw`: 应该接近 0（无自转）
   - `Train/rew_base_lin_vel_xy`: 应该接近 0（无移动）
   - `Train/rew_ang_vel_xy`: 应该接近 0（pitch/roll 稳定）

2. **腿部策略**
   - `Train/rew_hip_pos_constraint`: 应该逐渐减小
   - 观察机器人是否先摆腿再收腿

3. **整体性能**
   - `Train/mean_reward`: 应该持续上升
   - `Train/rew_upright_bonus`: 应该增加

### 调试技巧

如果机器人仍然自转：
```python
# 增加 yaw 惩罚权重
ang_vel_yaw = -2.0  # 从 -1.0 增加
```

如果腿部策略仍不理想：
```python
# 增加髋关节约束权重
hip_pos_constraint = -12.0  # 从 -8.0 增加

# 或调整目标位置
target_hip_pos = -0.15  # 更靠后
```

如果 pitch 仍不平：
```python
# 增加 pitch 角速度惩罚
ang_vel_xy = -1.0  # 从 -0.5 增加
```

## 代码位置

### 配置文件
[wheel_legged_vmc_balance_config.py:69-106](wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance_config.py#L69-L106)

新增奖励权重：
- `ang_vel_yaw = -1.0`
- `base_lin_vel_xy = -0.5`
- `hip_pos_constraint = -8.0`

### 环境实现
[wheel_legged_vmc_balance.py:311-347](wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance.py#L311-L347)

新增奖励函数：
- `_reward_ang_vel_yaw()`
- `_reward_base_lin_vel_xy()`
- `_reward_hip_pos_constraint()`

## 重新训练

修改后需要重新开始训练：

```bash
conda activate gym_env

# 删除旧的训练日志（可选）
rm -rf logs/wheel_legged_vmc_balance/latest

# 开始新的训练
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

或使用启动脚本：
```bash
bash start_training.sh
```

## 验证方法

训练 500-1000 iterations 后，使用 play 脚本测试：

```bash
python wheel_legged_gym/scripts/play_balance.py --task=wheel_legged_vmc_balance
```

观察：
1. 机器人是否保持原地不动（无自转、无移动）
2. pitch 和 roll 是否保持水平
3. 当腿在前方时，是否先摆到后方再收腿

## 进一步优化

如果效果仍不理想，可以考虑：

1. **增加观测信息**
   - 添加髋关节位置到观测空间
   - 让网络明确知道腿的位置

2. **调整 PD 增益**
   - 降低髋关节的 kp_theta
   - 提高响应速度

3. **使用课程学习**
   - 先训练静止平衡
   - 再训练动态恢复

4. **添加示教数据**
   - 录制正确的恢复动作
   - 使用模仿学习辅助训练
