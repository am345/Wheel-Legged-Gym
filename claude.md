# Wheel-Legged-Gym 项目概述

这是一个基于 Isaac Gym 的双轮足机器人强化学习训练框架，使用 PPO 算法训练机器人在复杂地形中的运动控制策略。

## 技术栈
- **仿真引擎**: NVIDIA Isaac Gym (GPU 加速物理仿真)
- **深度学习**: PyTorch + RSL_RL (PPO 算法)
- **机器人控制**: VMC (Virtual Model Control) - 极坐标空间控制
- **Python**: 3.8 推荐
- **Conda 环境**: gym_env

## 环境准备

**重要**: 运行任何训练或测试命令前，必须先激活 conda 环境：
```bash
conda activate gym_env
```

## 项目结构
```
wheel_legged_gym/
├── envs/                          # 环境定义
│   ├── base/                      # 基础环境类
│   │   ├── legged_robot.py        # 基础机器人环境
│   │   └── legged_robot_config.py # 基础配置
│   ├── wheel_legged/              # 端到端训练 (关节空间)
│   ├── wheel_legged_vmc/          # VMC 控制 (极坐标空间)
│   ├── wheel_legged_vmc_flat/     # VMC 平地训练 (低显存)
│   └── wheel_legged_vmc_balance/  # 平衡恢复任务 ⭐
├── rsl_rl/                        # PPO 训练算法
├── scripts/                       # 训练和测试脚本
│   ├── train.py                   # 训练入口
│   └── play.py                    # 策略测试
└── utils/                         # 工具函数
    ├── terrain.py                 # 地形生成
    └── math.py                    # 数学工具

resources/robots/                  # 机器人 URDF 模型
logs/                              # 训练日志和模型检查点
```

## 核心概念

### 1. VMC (Virtual Model Control)
将关节空间 (theta1, theta2) 转换为极坐标空间 (L0, theta0):
- **L0**: 腿长 (径向距离)
- **theta0**: 腿的角度 (极角)
- **优势**: 控制空间更直观，便于部署到闭链机构

实现位置: `wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py:191-201`

### 2. 平衡恢复任务 (wheel_legged_vmc_balance)
训练机器人从任意姿态恢复到直立平衡状态。

**特点**:
- 随机初始化姿态 (roll/pitch ±0.8 rad, 全方向 yaw)
- 随机初始速度 (线速度 ±1.0 m/s, 角速度 ±3.0 rad/s)
- 渐进式控制启用 (稳定后才启用完整控制)
- 气弹簧模型 (被动弹性支撑)

实现位置: `wheel_legged_gym/envs/wheel_legged_vmc_balance/`

### 3. 观测空间 (26 维)
```python
[
    base_ang_vel (3),           # 基座角速度
    projected_gravity (3),      # 重力投影
    commands (3),               # 速度/高度命令
    theta0 (2),                 # 腿部极角
    theta0_dot (2),             # 腿部极角速度
    L0 (2),                     # 腿长
    L0_dot (2),                 # 腿长变化率
    wheel_pos (2),              # 轮子位置
    wheel_vel (2),              # 轮子速度
    last_actions (6)            # 上一步动作
]
```

### 4. 动作空间 (6 维)
```python
每条腿 3 个动作:
[
    theta0_ref,    # 目标极角 (±0.5 rad)
    l0_ref,        # 目标腿长 (±0.07 m + 0.22 m offset)
    wheel_vel_ref  # 目标轮速 (±10 rad/s)
]
```

## 编码规范

### 环境继承结构
```
LeggedRobot (base)
    └── LeggedRobotVMC (wheel_legged_vmc)
            └── LeggedRobotVMCBalance (wheel_legged_vmc_balance)
```

### 配置文件规范
每个环境包含两个配置类:
1. **EnvCfg**: 环境参数 (奖励权重、地形、初始化等)
2. **EnvCfgPPO**: 训练参数 (学习率、batch size、迭代次数等)

### 奖励函数命名
- 奖励函数名必须与 `cfg.rewards.scales` 中的键对应
- 函数命名格式: `_reward_<name>(self) -> Tensor`
- 返回值: `(num_envs,)` 形状的张量

示例:
```python
class rewards:
    class scales:
        base_height = 15.0
        orientation = -15.0

# 对应的奖励函数
def _reward_base_height(self):
    return torch.square(self.root_states[:, 2] - self.commands[:, 2])

def _reward_orientation(self):
    return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
```

### 重要约定
- 所有物理量使用 SI 单位 (m, rad, N, Nm)
- 张量操作优先使用 PyTorch (避免 numpy 转换)
- 环境重置时必须调用 `super().reset_idx(env_ids)`
- 修改观测空间后必须同步更新 `_get_noise_scale_vec()`

## 常用命令

**注意**: 所有命令执行前必须先激活环境：`conda activate gym_env`

### 训练
```bash
# 平衡恢复任务训练
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance

# 平地 VMC 训练 (低显存)
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_flat

# 复杂地形训练
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc

# 无头模式 (不渲染，更快)
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --headless

# 指定环境数量
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --num_envs=4096

# 恢复训练
python wheel_legged_gym/scripts/train.py --task=wheel_legged_vmc_balance --resume
```

### 测试策略
```bash
# 测试最新模型
python wheel_legged_gym/scripts/play.py --task=wheel_legged_vmc_balance

# 测试特定检查点
python wheel_legged_gym/scripts/play.py --task=wheel_legged_vmc_balance --checkpoint=1000
```

### 监控训练
```bash
# 启动 TensorBoard
tensorboard --logdir=./logs --port=8080

# 查看训练日志
tail -f logs/wheel_legged_vmc_balance/latest/train.log
```

## 关键参数调优指南

### 平衡恢复任务 (wheel_legged_vmc_balance)

**奖励权重** (`cfg.rewards.scales`):
```python
base_height = 15.0          # 高度跟踪 (主要目标)
orientation = -15.0         # 姿态惩罚 (保持直立)
orientation_flip = -20.0    # 翻转惩罚 (严重)
hip_upright = -10.0         # 髋关节归零
upright_bonus = 5.0         # 稳定奖励
torque_over_limit = -50.0   # 力矩限制 (>30Nm)
```

**控制参数** (`cfg.control`):
```python
kp_theta = 10.0             # 极角 P 增益
kd_theta = 5.0              # 极角 D 增益
kp_l0 = 800.0               # 腿长 P 增益
kd_l0 = 7.0                 # 腿长 D 增益
feedforward_force = 60.0    # 前馈力 (抵消重力)
```

**初始化范围** (`cfg.balance_reset`):
```python
roll = [-0.8, 0.8]          # 初始横滚角
pitch = [-0.8, 0.8]         # 初始俯仰角
yaw = [-π, π]               # 初始偏航角 (全范围)
lin_vel_x = [-1.0, 1.0]     # 初始线速度
ang_vel_roll = [-3.0, 3.0]  # 初始角速度
```

## 调试技巧

### 1. 可视化调试
训练时按键盘快捷键:
- `v`: 切换渲染 (提升性能)
- `c`: 切换相机跟随
- `r`: 重置所有环境

### 2. 打印调试信息
在环境类中添加:
```python
def _post_physics_step_callback(self):
    super()._post_physics_step_callback()
    if self.common_step_counter % 100 == 0:
        print(f"L0: {self.L0[0]}, theta0: {self.theta0[0]}")
```

### 3. 检查奖励分布
```python
# 在 compute_reward() 后添加
if self.common_step_counter % 1000 == 0:
    for key, value in self.episode_sums.items():
        print(f"{key}: {value.mean():.3f}")
```

### 4. 验证 VMC 雅可比
```python
# 测试正逆运动学一致性
theta1_test = torch.tensor([[0.5, 0.5]])
theta2_test = torch.tensor([[-0.3, -0.3]])
L0, theta0 = self.forward_kinematics(theta1_test, theta2_test)
print(f"L0: {L0}, theta0: {theta0}")
```

## 常见问题

### 1. 训练不稳定
- 降低学习率 (`cfg_train.algorithm.learning_rate`)
- 增加 batch size (`cfg_train.algorithm.num_mini_batches`)
- 检查奖励权重是否过大

### 2. 机器人倒地
- 增加 `orientation` 惩罚权重
- 调整初始化范围 (降低难度)
- 检查力矩限制是否合理

### 3. 显存不足
- 减少环境数量 (`--num_envs=2048`)
- 使用 `wheel_legged_vmc_flat` (平地地形)
- 关闭高度测量 (`cfg.terrain.measure_heights = False`)

### 4. 训练速度慢
- 使用无头模式 (`--headless`)
- 训练时按 `v` 关闭渲染
- 增加 decimation (`cfg.control.decimation`)

## 模型部署

训练完成后，模型保存在:
```
logs/<task_name>/<timestamp>/model_<iteration>.pt
```

加载模型:
```python
from wheel_legged_gym.utils import get_args, task_registry

args = get_args()
env, env_cfg = task_registry.make_env(name=args.task, args=args)
policy = torch.jit.load('path/to/model.pt')

obs = env.get_observations()
actions = policy(obs.detach())
```