#!/usr/bin/env python3
"""
调试脚本：检查MuJoCo环境的观测和控制
"""

import numpy as np
import sys
sys.path.append('.')

from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv
from mujoco_sim.policy_loader import PolicyLoader

# 加载环境和策略
print("加载环境...")
env = MuJoCoBalanceEnv('resources/robots/serialleg/mjcf/serialleg_simple.xml', render=False)
policy = PolicyLoader('logs/wheel_legged_vmc_balance/Feb25_16-51-57_/model_950.pt')

# 重置（不随机化，方便调试）
print("\n重置环境（无随机化）...")
obs = env.reset(randomize=False)

print(f"\n初始状态:")
print(f"  基座位置: {env.data.qpos[0:3]}")
print(f"  基座四元数: {env.data.qpos[3:7]}")
print(f"  关节位置: {env.data.qpos[7:13]}")
print(f"  关节速度: {env.data.qvel[6:12]}")

print(f"\n初始观测 (27维):")
print(f"  基座角速度 (3): {obs[0:3]}")
print(f"  投影重力 (3): {obs[3:6]}")
print(f"  指令 (3): {obs[6:9]}")
print(f"  theta0 (2): {obs[9:11]}")
print(f"  theta0_dot (2): {obs[11:13]}")
print(f"  L0 (2): {obs[13:15]}")
print(f"  L0_dot (2): {obs[15:17]}")
print(f"  轮子位置 (2): {obs[17:19]}")
print(f"  轮子速度 (2): {obs[19:21]}")
print(f"  上一步动作 (6): {obs[21:27]}")

# 获取动作
print(f"\n获取策略动作...")
action = policy.get_action(obs)
print(f"  动作: {action}")
print(f"  动作范围: [{action.min():.3f}, {action.max():.3f}]")

# 计算力矩
print(f"\n计算PD控制力矩...")
dof_pos = env.data.qpos[7:13]
dof_vel = env.data.qvel[6:12]
pos_ref = action * env.pos_action_scale + env.default_dof_pos
vel_ref = action * env.vel_action_scale
p_gains = np.array([40.0, 40.0, 0.0, 40.0, 40.0, 0.0])
d_gains = np.array([1.0, 1.0, 0.5, 1.0, 1.0, 0.5])
torques = p_gains * (pos_ref - dof_pos) + d_gains * (vel_ref - dof_vel)
torques = np.clip(torques, -env.torque_limits, env.torque_limits)

print(f"  位置参考: {pos_ref}")
print(f"  当前位置: {dof_pos}")
print(f"  位置误差: {pos_ref - dof_pos}")
print(f"  速度参考: {vel_ref}")
print(f"  当前速度: {dof_vel}")
print(f"  计算力矩: {torques}")

# 执行几步
print(f"\n执行10步...")
for i in range(10):
    obs, reward, done, info = env.step(action)
    if i == 0 or i == 9:
        print(f"  步骤 {i}: 高度={info['base_height']:.3f}, 奖励={reward:.3f}")
        print(f"    关节位置: {env.data.qpos[7:13]}")
        print(f"    关节速度: {env.data.qvel[6:12]}")

print("\n完成！")
