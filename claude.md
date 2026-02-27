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
**重要** 每次更改完代码后先自己运行一遍任务看有没有语法错误 
若输出以下类似调试信息说明代码成功跑起来了

################################################################################
                      Learning iteration 305/2000                       

                       Computation: 7798 steps/s (collection: 0.644s, learning 0.144s)
               Value function loss: 42820206.6750
                    Surrogate loss: 0.0003
             Mean action noise std: 0.66
                       Mean reward: 2565.63
                       Mean length: 6002.00
              Mean rew_ang_vel_yaw: -0.0020
          Mean rew_base_lin_vel_xy: -0.0114
           Mean rew_leg_angle_zero: 5.6504
                Mean rew_lin_vel_z: -0.0012
              Mean rew_pitch_angle: 10.0033
                Mean rew_pitch_vel: -0.1078
        Mean rew_reach_flat_target: 0.0033
               Mean rew_roll_angle: 10.0033
                 Mean rew_roll_vel: -0.0233
              Mean rew_stand_still: 8.5224
              Mean rew_termination: 0.0000
                  Mean rew_torques: -0.0016
            Mean rew_upright_bonus: 9.0815
--------------------------------------------------------------------------------
                   Total timesteps: 1880064
                    Iteration time: 0.79s
                        Total time: 272.65s
                               ETA: 1510.3s

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