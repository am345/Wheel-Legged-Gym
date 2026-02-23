# 最小化 Balance 任务配置

## 简化目标

**唯一目标**: 让机器人从直立姿态保持平衡

## 移除的内容

### ❌ 移除的功能
1. **气弹簧模型** - 不再模拟气弹簧力
2. **复杂初始化** - 不从倾斜/倒立姿态开始
3. **渐进式控制** - 不需要 ready_for_control 逻辑
4. **推力扰动** - 禁用 push_robots
5. **所有非核心奖励** - 只保留 3 个核心奖励

### ✅ 保留的内容
1. **VMC 控制** - 极坐标空间控制（父类提供）
2. **基础 PD 控制** - 标准的力矩计算（父类提供）
3. **核心奖励** - 只有 3 个：
   - `base_height`: 保持高度
   - `orientation`: 保持直立
   - `upright_bonus`: 直立奖励

## 当前配置

### 初始化
```python
# 完全直立，无任何扰动
roll = [0.0, 0.0]
pitch = [0.0, 0.0]
yaw = [0.0, 0.0]
所有速度 = [0.0, 0.0]
```

### 奖励函数（只有 3 个）
```python
base_height = 30.0      # 保持高度 0.18m
orientation = -30.0     # 保持直立（惩罚倾斜）
upright_bonus = 15.0    # 直立且静止时的奖励
```

### 训练参数
```python
learning_rate = 5e-5    # 较低的学习率
max_iterations = 5000
episode_length_s = 30
```

## 代码结构

### wheel_legged_vmc_balance.py
```python
class LeggedRobotVMCBalance(LeggedRobotVMC):
    # 只有 35 行代码
    # 直接继承父类的所有功能
    # 只添加一个奖励函数: _reward_upright_bonus()
```

### wheel_legged_vmc_balance_config.py
```python
class WheelLeggedVMCBalanceCfg(WheelLeggedVMCCfg):
    # 只配置必要的参数
    # 禁用所有非核心奖励
    # 初始化全部设为 0（直立）
```

## 训练方法

### 开始训练
```bash
conda activate gym_env
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096
```

### 预期结果

由于任务极其简单（从直立开始保持平衡），训练应该很快收敛：

- **0-100 iterations**: 学习基本控制
- **100-500 iterations**: 奖励稳定上升
- **500+ iterations**: 达到稳定平衡

### 监控指标

在 TensorBoard 中只需关注：
1. `Train/mean_reward`: 应该快速上升到 40+
2. `Train/rew_base_height`: 应该 > 20
3. `Train/rew_orientation`: 应该 > -5
4. `Train/rew_upright_bonus`: 应该 > 10

## 如果仍然失败

### 进一步简化

如果机器人仍然无法平衡，可以：

1. **增加高度奖励权重**
```python
base_height = 50.0  # 从 30.0 增加
```

2. **降低学习率**
```python
learning_rate = 1e-5  # 从 5e-5 降低
```

3. **减少环境数量**（更稳定但更慢）
```bash
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=2048
```

4. **检查父类控制参数**
查看 `wheel_legged_vmc_config.py` 中的 PD 增益：
```python
kp_theta = 10.0
kd_theta = 5.0
kp_l0 = 800.0
kd_l0 = 7.0
```

## 调试技巧

### 1. 可视化观察
```bash
# 不使用 headless 模式，观察机器人行为
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=16
```

### 2. 打印调试信息
在 `wheel_legged_vmc_balance.py` 中添加：
```python
def _reward_upright_bonus(self):
    grav_ok = (torch.abs(self.projected_gravity[:, 0]) < 0.05) & (
        torch.abs(self.projected_gravity[:, 1]) < 0.05
    )
    if self.common_step_counter % 100 == 0:
        print(f"Gravity: {self.projected_gravity[0]}")
        print(f"Height: {self.root_states[0, 2]}")
    return (grav_ok & lin_ok & ang_ok).float()
```

### 3. 检查物理参数
确保机器人 URDF 中的质量、惯量等参数合理。

## 下一步

如果这个最小化版本能成功训练，再逐步添加功能：

1. ✅ **阶段 0**: 从直立保持平衡（当前）
2. **阶段 1**: 添加小角度初始化（±10°）
3. **阶段 2**: 添加推力扰动
4. **阶段 3**: 增加初始角度（±30°）
5. **阶段 4**: 添加气弹簧模型
6. **阶段 5**: 大角度恢复（±90°）
7. **阶段 6**: 倒立恢复（±180°）

## 文件位置

- 配置: [wheel_legged_vmc_balance_config.py](wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance_config.py)
- 环境: [wheel_legged_vmc_balance.py](wheel_legged_gym/envs/wheel_legged_vmc_balance/wheel_legged_vmc_balance.py)

## 总结

现在的配置是**最简单的平衡任务**：
- 从完全直立开始
- 只需要保持不倒
- 没有任何扰动
- 只有 3 个奖励函数
- 代码只有 35 行

如果这个都无法训练成功，那么问题可能在于：
1. 父类 VMC 控制有问题
2. 机器人 URDF 模型有问题
3. PD 增益参数不合理
4. 物理仿真参数有问题
