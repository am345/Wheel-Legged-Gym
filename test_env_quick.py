#!/usr/bin/env python3
"""
测试修改后的MuJoCo环境
"""

import numpy as np
import sys
sys.path.append('.')

from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv

print("加载环境...")
env = MuJoCoBalanceEnv('resources/robots/serialleg/mjcf/serialleg_simple.xml', render=False)

print("\n测试1: 零位重置")
obs = env.reset(randomize=False)
print(f"  观测形状: {obs.shape}")
print(f"  基座位置: {env.data.qpos[0:3]}")
print(f"  关节位置: {env.data.qpos[7:13]}")
print(f"  theta0 (观测[9:11]): {obs[9:11]}")
print(f"  L0 (观测[13:15]): {obs[13:15]}")

# 手动计算VMC
from mujoco_sim.vmc_kinematics import VMCKinematics
vmc = VMCKinematics()
L0_l, theta0_l = vmc.forward_kinematics(0.0, 0.0)
print(f"  VMC计算: L0={L0_l:.4f}, theta0={theta0_l:.4f}")
print(f"  观测中的L0 (缩放后): {L0_l * 5.0:.4f}")
print(f"  观测中的theta0 (缩放后): {theta0_l * 1.0:.4f}")

print("\n测试2: 默认位置")
env.data.qpos[7:13] = [0.4, 0.25, 0.0, 0.4, 0.25, 0.0]
obs = env.obs_computer.compute(env.data, np.zeros(6))
print(f"  关节位置: {env.data.qpos[7:13]}")
print(f"  theta0 (观测[9:11]): {obs[9:11]}")
print(f"  L0 (观测[13:15]): {obs[13:15]}")

L0_l, theta0_l = vmc.forward_kinematics(0.4, 0.25)
print(f"  VMC计算: L0={L0_l:.4f}, theta0={theta0_l:.4f}")
print(f"  观测中的L0 (缩放后): {L0_l * 5.0:.4f}")
print(f"  观测中的theta0 (缩放后): {theta0_l * 1.0:.4f}")

print("\n测试3: 执行零动作")
obs = env.reset(randomize=False)
action = np.zeros(6)
obs, reward, done, info = env.step(action)
print(f"  奖励: {reward:.4f}")
print(f"  高度: {info['base_height']:.4f}")
print(f"  投影重力 (观测[3:6]): {obs[3:6]}")

print("\n✓ 环境测试完成")
