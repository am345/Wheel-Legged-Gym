# Balance 任务奖励函数修复方案

## 问题诊断

### 当前问题
奖励达到 300+ 但机器人仍不能平衡，这是典型的**奖励欺骗 (Reward Hacking)** 问题。

### 根本原因

通过诊断脚本发现了 4 个关键问题：

#### 1. **base_height 奖励衰减过快**
```python
# 当前实现
reward = 40.0 * exp(-error^2 / 0.01)

# 问题:
- 高度差 0.1m: 奖励从 40 → 0.6 (衰减 98.5%)
- 高度差 0.2m: 奖励 ≈ 0
```
**后果**: 机器人可能学会「蹲下」到某个高度，然后就不管了。

#### 2. **orientation 惩罚太弱**
```python
# 当前实现
penalty = -40.0 * (gravity_x^2 + gravity_y^2)

# 问题:
- 10度倾斜: -1.2 (很小)
- 30度倾斜: -10.0 (中等)
- 60度倾斜: -30.0 (才接近 base_height 奖励)
```
**后果**: 机器人可以倾斜 10-20 度而不受太大惩罚，只要保持高度就能获得高奖励。

#### 3. **upright_bonus 条件过于严格**
```python
# 当前条件
grav_ok = (abs(gravity_x) < 0.05) & (abs(gravity_y) < 0.05)  # 约 3 度
lin_ok = norm(lin_vel) < 0.3
ang_ok = norm(ang_vel) < 0.3
```
**后果**: 机器人很难同时满足所有条件，这个 20 分的奖励几乎拿不到。

#### 4. **终止条件宽松**
```python
# 终止条件
fail = projected_gravity.z > -0.1  # 约 84 度才终止
```
**后果**: 机器人可以倾斜很大角度而不终止，在倒地前的短暂时刻积累高奖励。

### 奖励欺骗场景

机器人可能学会了这样的策略：
1. 保持高度在 0.25m 附近 → 获得 40 分
2. 轻微倾斜 10-20 度 → 只损失 1-5 分
3. 在倒地前的短暂时刻 → 总奖励 35-39 分
4. Episode 很快终止 → 平均奖励看起来很高

**关键**: 奖励高不代表平衡好，可能只是「倒得慢」。

## 修复方案

### 方案 A: 修改奖励函数 (推荐)

#### 1. 修改 base_height 奖励函数
```python
# 在 legged_robot.py 中修改
def _reward_base_height(self):
    if self.reward_scales["base_height"] < 0:
        return torch.abs(self.base_height - self.commands[:, 2])
    else:
        base_height_error = torch.square(self.base_height - self.commands[:, 2])
        # 修改: 增大分母，减缓衰减
        return torch.exp(-base_height_error / 0.05)  # 从 0.01 改为 0.05
```

**效果**:
- 高度差 0.1m: 奖励从 0.6 → 13.5 (衰减 66%)
- 高度差 0.2m: 奖励从 0.0 → 1.8 (仍有信号)

#### 2. 增强 orientation 惩罚
```python
# 在 balance_config.py 中修改
orientation = -80.0  # 从 -40.0 增加到 -80.0
```

**效果**:
- 10度倾斜: -2.4 (从 -1.2 翻倍)
- 30度倾斜: -20.0 (从 -10.0 翻倍)
- 60度倾斜: -60.0 (从 -30.0 翻倍)

#### 3. 放宽 upright_bonus 条件
```python
# 在 wheel_legged_vmc_balance.py 中修改
def _reward_upright_bonus(self):
    # 放宽条件
    grav_ok = (torch.abs(self.projected_gravity[:, 0]) < 0.1) & (
        torch.abs(self.projected_gravity[:, 1]) < 0.1
    )  # 从 0.05 改为 0.1 (约 6 度)
    lin_ok = torch.norm(self.base_lin_vel, dim=1) < 0.5  # 从 0.3 改为 0.5
    ang_ok = torch.norm(self.base_ang_vel, dim=1) < 0.5  # 从 0.3 改为 0.5
    return (grav_ok & lin_ok & ang_ok).float()
```

#### 4. 收紧终止条件
```python
# 在 legged_robot.py 中修改 check_termination
fail_buf |= self.projected_gravity[:, 2] > -0.5  # 从 -0.1 改为 -0.5 (约 60 度)
```

**效果**: 机器人倾斜超过 60 度就会终止，无法在倒地前积累高奖励。

### 方案 B: 添加新的奖励项 (补充)

在 `wheel_legged_vmc_balance.py` 中添加：

```python
def _reward_stay_upright(self):
    """持续直立奖励 - 每个时间步都给"""
    # 只要 gravity.z < -0.8 (约 36 度内) 就给奖励
    return (self.projected_gravity[:, 2] < -0.8).float()
```

在 `balance_config.py` 中添加：
```python
stay_upright = 5.0  # 每步 5 分，20 秒 = 2000 步 = 10000 分
```

**效果**: 鼓励机器人尽可能长时间保持直立。

### 方案 C: 使用负奖励基线 (激进)

```python
# 在 balance_config.py 中
class scales:
    # 所有奖励改为负惩罚
    base_height = -40.0      # 惩罚高度偏差
    orientation = -80.0      # 惩罚倾斜
    upright_bonus = 0.0      # 移除

    # 添加持续奖励
    stay_upright = 10.0      # 唯一的正奖励
```

**效果**: 只有真正保持直立才能获得正奖励。

## 推荐实施步骤

### 第 1 步: 快速修复 (5 分钟)

只修改配置文件，不改代码：

```python
# wheel_legged_vmc_balance_config.py
class scales:
    base_height = 40.0
    orientation = -80.0      # ← 从 -40.0 改为 -80.0
    upright_bonus = 20.0

    # 添加新奖励
    stand_still = 5.0        # ← 从 3.0 改为 5.0
    lin_vel_z = -4.0         # ← 从 -2.0 改为 -4.0
    ang_vel_xy = -2.0        # ← 从 -1.0 改为 -2.0
```

### 第 2 步: 修改奖励函数 (10 分钟)

修改 `legged_robot.py`:

```python
def _reward_base_height(self):
    if self.reward_scales["base_height"] < 0:
        return torch.abs(self.base_height - self.commands[:, 2])
    else:
        base_height_error = torch.square(self.base_height - self.commands[:, 2])
        return torch.exp(-base_height_error / 0.05)  # ← 从 0.01 改为 0.05
```

### 第 3 步: 收紧终止条件 (5 分钟)

修改 `legged_robot.py`:

```python
def check_termination(self):
    # ...
    fail_buf |= self.projected_gravity[:, 2] > -0.5  # ← 从 -0.1 改为 -0.5
```

### 第 4 步: 重新训练

```bash
# 删除旧的训练结果
rm -rf logs/wheel_legged_vmc_balance/Feb23_*

# 重新训练
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 第 5 步: 监控新指标

在 TensorBoard 中关注：
1. `Train/mean_episode_length` - 应该接近最大长度 (20s = 4000 steps)
2. `Train/rew_orientation` - 应该接近 0 (不是 -10 或更负)
3. `Train/rew_upright_bonus` - 应该 > 10 (不是 0)

## 预期效果

### 修复前
- 奖励: 300+
- Episode 长度: 短 (可能只有几秒)
- 行为: 倾斜、蹲下、快速倒地

### 修复后
- 奖励: 可能降到 50-100 (但更真实)
- Episode 长度: 接近最大长度 (20s)
- 行为: 真正的直立平衡

## 验证方法

### 1. 可视化测试
```bash
python wheel_legged_gym/scripts/play_balance.py --task=wheel_legged_vmc_balance
```
按 'C' 启动，观察机器人是否能稳定站立 20 秒。

### 2. 检查奖励分解
在训练日志中查看：
- `rew_base_height` 应该 > 30
- `rew_orientation` 应该 > -5
- `rew_upright_bonus` 应该 > 10

### 3. 检查终止原因
添加调试代码：
```python
# 在 check_termination 中
if self.common_step_counter % 1000 == 0:
    print(f"Fail rate: {fail_buf.float().mean():.2%}")
    print(f"Timeout rate: {self.time_out_buf.float().mean():.2%}")
```

**期望**: Timeout rate > 80% (大部分 episode 是超时而非倒地)

## 如果仍然失败

### 诊断步骤
1. 检查机器人 URDF 质量分布
2. 检查 PD 增益是否合理
3. 检查初始姿态是否真的直立
4. 添加更多调试输出

### 最后手段
使用 Imitation Learning:
1. 手动控制机器人平衡
2. 记录轨迹
3. 用行为克隆预训练
4. 再用 RL 微调

## 总结

**核心问题**: 奖励函数设计不当，导致机器人学会「看起来奖励高但实际不平衡」的策略。

**解决方案**:
1. 增强 orientation 惩罚 (最重要)
2. 减缓 base_height 衰减
3. 收紧终止条件
4. 放宽 upright_bonus 条件

**预期**: 修复后奖励可能降低，但机器人行为会更好。
