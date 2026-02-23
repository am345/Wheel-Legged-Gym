# Balance 任务课程学习指南

## 训练策略：从简单到难

由于直接从 0-360° 训练太难，我们采用**分阶段课程学习**策略。

## 阶段 1: 基础平衡（当前配置）⭐

### 目标
让机器人学会从小角度倾斜恢复到直立平衡。

### 配置参数
```python
# balance_reset
roll = [-0.5, 0.5]      # ±30° (0.5 rad)
pitch = [-0.5, 0.5]     # ±30°
lin_vel_x = [-0.5, 0.5]
ang_vel_roll = [-1.5, 1.5]

# rewards.scales
base_height = 20.0
orientation = -20.0
upright_bonus = 10.0
recovery_speed = 0.0    # 禁用
energy_efficiency = 0.0 # 禁用
```

### 训练命令
```bash
conda activate gym_env
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 成功标准
- `mean_reward` > 30
- `rew_upright_bonus` > 5
- `rew_orientation` > -5
- 机器人能稳定站立 10 秒以上

### 预计训练时间
1000-2000 iterations

---

## 阶段 2: 中等角度恢复

### 目标
从 ±60° 角度恢复平衡。

### 配置修改
```python
# 在 wheel_legged_vmc_balance_config.py 中修改
roll = [-1.0, 1.0]      # ±60° (1.0 rad)
pitch = [-1.0, 1.0]     # ±60°
lin_vel_x = [-0.8, 0.8]
ang_vel_roll = [-2.0, 2.0]
```

### 训练方式
```bash
# 从阶段1的模型继续训练
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --resume
```

### 成功标准
- `mean_reward` > 40
- 能从 60° 倾斜恢复

### 预计训练时间
1000-1500 iterations

---

## 阶段 3: 大角度恢复

### 目标
从 ±90° 角度恢复平衡。

### 配置修改
```python
roll = [-1.57, 1.57]    # ±90° (π/2 rad)
pitch = [-1.57, 1.57]   # ±90°
lin_vel_x = [-1.0, 1.0]
ang_vel_roll = [-2.5, 2.5]

# 启用恢复速度奖励
recovery_speed = 1.0
```

### 成功标准
- `mean_reward` > 50
- 能从 90° 倾斜恢复

### 预计训练时间
1500-2000 iterations

---

## 阶段 4: 倒立恢复（高级）

### 目标
从 ±180° (倒立) 恢复平衡。

### 配置修改
```python
roll = [-3.14, 3.14]    # ±180° (π rad)
pitch = [-3.14, 3.14]   # ±180°

# 修改终止条件，允许倒立
# 在 wheel_legged_vmc_balance.py 的 check_termination() 中
# 注释掉: fail_buf |= self.projected_gravity[:, 2] > -0.1

# 启用能量效率奖励
energy_efficiency = 0.5
torques = -2e-4  # 增加力矩惩罚
```

### 成功标准
- `mean_reward` > 60
- 能从倒立恢复

### 预计训练时间
2000-3000 iterations

---

## 快速配置切换

### 方法 1: 手动修改配置文件
编辑 `wheel_legged_vmc_balance_config.py` 中的 `balance_reset` 类。

### 方法 2: 创建多个配置类
```python
# 在 wheel_legged_vmc_balance_config.py 中添加

class WheelLeggedVMCBalanceStage2Cfg(WheelLeggedVMCBalanceCfg):
    class balance_reset(WheelLeggedVMCBalanceCfg.balance_reset):
        roll = [-1.0, 1.0]
        pitch = [-1.0, 1.0]

class WheelLeggedVMCBalanceStage3Cfg(WheelLeggedVMCBalanceCfg):
    class balance_reset(WheelLeggedVMCBalanceCfg.balance_reset):
        roll = [-1.57, 1.57]
        pitch = [-1.57, 1.57]
```

然后在 `envs/__init__.py` 中注册新任务。

---

## 监控关键指标

### 阶段 1 重点监控
- ✅ `Train/mean_reward`: 应该从 -20 上升到 30+
- ✅ `Train/rew_upright_bonus`: 应该逐渐增加到 5+
- ✅ `Train/rew_base_height`: 应该 > 10
- ❌ `Train/rew_orientation`: 应该 > -5

### 阶段 2-4 重点监控
- ✅ `Train/rew_recovery_speed`: 应该逐渐增加
- ✅ `Train/mean_episode_length`: 应该接近最大长度
- ❌ `Train/rew_torque_over_limit`: 应该接近 0

---

## 常见问题

### Q1: 阶段1训练不收敛怎么办？
**A**: 进一步降低难度
```python
roll = [-0.3, 0.3]  # 降低到 ±17°
pitch = [-0.3, 0.3]
lin_vel_x = [-0.3, 0.3]
ang_vel_roll = [-1.0, 1.0]
```

### Q2: 什么时候进入下一阶段？
**A**: 满足以下条件：
1. 当前阶段的成功标准达到
2. 奖励曲线稳定（不再上升）
3. 训练至少 1000 iterations

### Q3: 能跳过某个阶段吗？
**A**: 不建议。每个阶段都是下一阶段的基础。跳过会导致训练不稳定。

### Q4: 如何从上一阶段的模型继续训练？
**A**: 使用 `--resume` 参数：
```bash
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --resume
```

---

## 当前状态

**你现在处于: 阶段 1 - 基础平衡**

配置已调整为：
- 初始角度: ±30°
- 初始速度: 降低
- 奖励函数: 简化，聚焦核心目标
- 学习率: 降低到 5e-5

**下一步**: 开始训练，监控 TensorBoard，等待 `mean_reward` 达到 30+
