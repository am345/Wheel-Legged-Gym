#!/usr/bin/env python3
"""
对比IsaacGym和MuJoCo的VMC运动学计算
"""

import torch
import numpy as np
import sys
sys.path.append('.')

from wheel_legged_gym.utils import get_args, task_registry

# 加载IsaacGym环境
print("加载IsaacGym环境...")
args = get_args()
args.task = "wheel_legged_vmc_balance"
args.headless = True

env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
env_cfg.env.num_envs = 1
env_cfg.terrain.num_rows = 1
env_cfg.terrain.num_cols = 1

env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

# 设置特定的关节位置
print("\n测试不同的关节位置...")
test_cases = [
    ([0.0, 0.0], "零位"),
    ([0.4, 0.25], "默认位置"),
    ([0.5, 0.0], "theta1=0.5, theta2=0"),
    ([0.0, 0.5], "theta1=0, theta2=0.5"),
]

for joint_pos, desc in test_cases:
    # 设置关节位置
    env.dof_pos[0, 0] = joint_pos[0]  # lf0
    env.dof_pos[0, 1] = joint_pos[1]  # lf1
    env.dof_pos[0, 3] = joint_pos[0]  # rf0
    env.dof_pos[0, 4] = joint_pos[1]  # rf1

    # 计算VMC
    env.leg_post_physics_step()

    print(f"\n{desc}: theta1={joint_pos[0]:.3f}, theta2={joint_pos[1]:.3f}")
    print(f"  IsaacGym L0: {env.L0[0, 0].item():.4f}")
    print(f"  IsaacGym theta0: {env.theta0[0, 0].item():.4f} rad = {np.degrees(env.theta0[0, 0].item()):.1f} deg")

    # 我们的计算
    l1 = 0.167
    l2 = 0.200
    theta1 = joint_pos[0]
    theta2 = joint_pos[1]
    end_x = l1 * np.cos(theta1) - l2 * np.sin(theta1 + theta2)
    end_y = l1 * np.sin(theta1) + l2 * np.cos(theta1 + theta2)
    L0_ours = np.sqrt(end_x**2 + end_y**2)
    theta0_ours = np.arctan2(end_x, end_y)

    print(f"  我们的 L0: {L0_ours:.4f}")
    print(f"  我们的 theta0: {theta0_ours:.4f} rad = {np.degrees(theta0_ours):.1f} deg")
    print(f"  差异: ΔL0={abs(env.L0[0,0].item()-L0_ours):.6f}, Δtheta0={abs(env.theta0[0,0].item()-theta0_ours):.6f}")

print("\n完成！")
