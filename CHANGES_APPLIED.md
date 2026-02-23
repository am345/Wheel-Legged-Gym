# Balance 任务修复 - 已应用的修改

## 修改时间
2026-02-23

## 问题描述
奖励达到 300+ 但机器人仍不能平衡，诊断发现是**奖励欺骗 (Reward Hacking)** 问题。

## 已应用的修改

### 1. 增强 orientation 惩罚 ✅
**文件**: `wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance_config.py:103`

```python
# 修改前
orientation = -40.0

# 修改后
orientation = -80.0  # 翻倍，增强对倾斜的惩罚
```

**效果**:
- 10度倾斜: 惩罚从 -1.2 → -2.4
- 30度倾斜: 惩罚从 -10.0 → -20.0
- 机器人将更难通过「倾斜但不倒」获得高奖励

### 2. 增强静止约束 ✅
**文件**: `wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance_config.py:107-109`

```python
# 修改前
lin_vel_z = -2.0
ang_vel_xy = -1.0
stand_still = 3.0

# 修改后
lin_vel_z = -4.0      # 翻倍
ang_vel_xy = -2.0     # 翻倍
stand_still = 5.0     # 增加 67%
```

**效果**: 更强烈地鼓励机器人保持静止

### 3. 减缓 base_height 奖励衰减 ✅
**文件**: `wheel_legged_gym/envs/base/legged_robot.py:1649`

```python
# 修改前
return torch.exp(-base_height_error / 0.01)

# 修改后
return torch.exp(-base_height_error / 0.05)  # 分母增大 5 倍
```

**效果**:
- 高度差 0.1m: 奖励从 0.6 → 13.5 (保留更多信号)
- 高度差 0.2m: 奖励从 0.0 → 1.8 (仍有梯度)
- 机器人将更容易学习正确的高度控制

### 4. 收紧终止条件 ✅
**文件**: `wheel_legged_gym/envs/base/legged_robot.py:218`

```python
# 修改前
fail_buf |= self.projected_gravity[:, 2] > -0.1  # 约 84 度

# 修改后
fail_buf |= self.projected_gravity[:, 2] > -0.5  # 约 60 度
```

**效果**: 机器人倾斜超过 60 度就会终止，无法在倒地前积累高奖励

### 5. 放宽 upright_bonus 条件 ✅
**文件**: `wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance.py:29-32`

```python
# 修改前
grav_ok = (torch.abs(self.projected_gravity[:, 0]) < 0.05) & (
    torch.abs(self.projected_gravity[:, 1]) < 0.05
)  # 约 3 度
lin_ok = torch.norm(self.base_lin_vel, dim=1) < 0.3
ang_ok = torch.norm(self.base_ang_vel, dim=1) < 0.3

# 修改后
grav_ok = (torch.abs(self.projected_gravity[:, 0]) < 0.1) & (
    torch.abs(self.projected_gravity[:, 1]) < 0.1
)  # 约 6 度
lin_ok = torch.norm(self.base_lin_vel, dim=1) < 0.5
ang_ok = torch.norm(self.base_ang_vel, dim=1) < 0.5
```

**效果**: upright_bonus (20 分) 更容易获得，提供更强的正向激励

## 预期效果

### 修改前
- ❌ 奖励: 300+
- ❌ Episode 长度: 短 (几秒)
- ❌ 行为: 倾斜、蹲下、快速倒地
- ❌ 奖励来源: 倒地前的短暂时刻

### 修改后
- ✅ 奖励: 50-100 (更真实)
- ✅ Episode 长度: 接近 20s (4000 steps)
- ✅ 行为: 真正的直立平衡
- ✅ 奖励来源: 持续保持平衡

## 重新训练步骤

### 1. 清理旧的训练结果
```bash
cd /home/am345/Wheel-Legged-Gym
rm -rf logs/wheel_legged_vmc_balance/Feb23_*
```

### 2. 开始新的训练
```bash
conda activate gym_env
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 3. 监控 TensorBoard
```bash
tensorboard --logdir=logs/wheel_legged_vmc_balance --port=6006
```

## 新的监控指标

在 TensorBoard 中重点关注：

### 主要指标
1. **Train/mean_episode_length**
   - 期望: 接近 4000 steps (20s)
   - 如果 < 1000: 机器人仍然倒得太快

2. **Train/rew_orientation**
   - 期望: > -5
   - 如果 < -20: 机器人仍在大幅倾斜

3. **Train/rew_upright_bonus**
   - 期望: > 10
   - 如果 = 0: 条件仍然太严格

4. **Train/mean_reward**
   - 期望: 50-100
   - 注意: 奖励降低是正常的，关键看行为

### 次要指标
5. **Train/rew_base_height** - 应该 > 30
6. **Loss/value_function** - 应该逐渐降低
7. **Policy/learning_rate** - 保持在 1e-4

## 验证方法

### 训练完成后测试
```bash
python wheel_legged_gym/scripts/play_balance.py --task=wheel_legged_vmc_balance
```

按 'C' 启动控制，观察机器人是否能：
- ✅ 稳定站立 20 秒
- ✅ 保持直立 (倾斜 < 10 度)
- ✅ 高度稳定在 0.25m 附近
- ✅ 没有明显晃动

## 如果仍然失败

### 诊断步骤
1. 检查 episode 长度是否增加
2. 检查 orientation 惩罚是否生效
3. 查看机器人在仿真中的实际行为
4. 检查 upright_bonus 是否被触发

### 进一步调整
如果机器人仍然倒地太快，可以：
1. 进一步增加 orientation 惩罚到 -120.0
2. 降低 PD 增益 (kp_theta, kp_l0)
3. 减小动作幅度 (action_scale_*)
4. 收紧终止条件到 -0.7 (约 45 度)

## 修改的文件列表

1. ✅ `wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance_config.py`
2. ✅ `wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance.py`
3. ✅ `wheel_legged_gym/envs/base/legged_robot.py`

## 备份建议

如果需要回滚，可以使用 git:
```bash
git diff  # 查看修改
git checkout -- <file>  # 恢复单个文件
git stash  # 暂存所有修改
```

## 理论依据

这些修改基于以下原理：
1. **奖励塑形**: 增强关键行为的奖励/惩罚权重
2. **梯度保持**: 减缓奖励衰减以保持学习信号
3. **终止条件**: 防止在失败状态积累奖励
4. **可达性**: 放宽条件使奖励更容易获得

## 预计训练时间

- GPU: RTX 3090 → 约 30-60 分钟
- GPU: RTX 4090 → 约 20-40 分钟
- 1000 iterations @ 4096 envs

## 成功标准

训练成功的标志：
- ✅ mean_episode_length > 3500 steps
- ✅ mean_reward > 50
- ✅ rew_orientation > -5
- ✅ rew_upright_bonus > 10
- ✅ 机器人在 play 模式下能稳定站立 20 秒

---

**修改完成时间**: 2026-02-23
**下一步**: 清理旧日志并重新训练
