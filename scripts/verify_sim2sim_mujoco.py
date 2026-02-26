#!/usr/bin/env python3
"""
Sim2Sim验证脚本：在MuJoCo中运行训练好的IsaacGym策略
"""

import argparse
import numpy as np
import time
import json
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv
from mujoco_sim.policy_loader import PolicyLoader


class BalanceMetrics:
    """跟踪平衡性能指标"""

    def __init__(self):
        self.reset()

    def reset(self):
        """重置统计"""
        self.episode_data = []
        self.current_episode = {
            'heights': [],
            'tilts': [],
            'angular_velocities': [],
            'upright_time': 0,
            'max_tilt': 0,
        }

    def update(self, data, step):
        """更新当前时间步的指标"""
        # 提取状态
        base_pos = data.qpos[0:3]
        base_quat = data.qpos[3:7]
        base_angvel = data.qvel[3:6]

        # 计算指标
        height = base_pos[2]

        # 投影重力（直立度）
        from scipy.spatial.transform import Rotation
        quat_scipy = np.array([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])
        rot = Rotation.from_quat(quat_scipy)
        projected_gravity = rot.inv().apply(np.array([0, 0, -1]))

        # 倾角（度）
        tilt_angle = np.arccos(np.clip(-projected_gravity[2], -1, 1)) * 180 / np.pi

        # 跟踪数据
        self.current_episode['heights'].append(height)
        self.current_episode['tilts'].append(tilt_angle)
        self.current_episode['angular_velocities'].append(np.linalg.norm(base_angvel))

        # 直立时间（倾角<10度）
        if tilt_angle < 10:
            self.current_episode['upright_time'] += 1

        self.current_episode['max_tilt'] = max(self.current_episode['max_tilt'], tilt_angle)

    def log_episode(self, episode_num, steps, reward):
        """记录完成的episode"""
        episode_summary = {
            'episode': episode_num,
            'steps': steps,
            'reward': reward,
            'mean_height': np.mean(self.current_episode['heights']),
            'std_height': np.std(self.current_episode['heights']),
            'upright_ratio': self.current_episode['upright_time'] / steps if steps > 0 else 0,
            'max_tilt': self.current_episode['max_tilt'],
            'mean_tilt': np.mean(self.current_episode['tilts']),
            'mean_angvel': np.mean(self.current_episode['angular_velocities']),
        }

        self.episode_data.append(episode_summary)

        # 重置当前episode
        self.current_episode = {
            'heights': [],
            'tilts': [],
            'angular_velocities': [],
            'upright_time': 0,
            'max_tilt': 0,
        }

    def print_summary(self):
        """打印汇总统计"""
        if not self.episode_data:
            return

        steps = [e['steps'] for e in self.episode_data]
        rewards = [e['reward'] for e in self.episode_data]
        upright_ratios = [e['upright_ratio'] for e in self.episode_data]
        max_tilts = [e['max_tilt'] for e in self.episode_data]
        mean_tilts = [e['mean_tilt'] for e in self.episode_data]

        print("\n" + "="*60)
        print("SIM2SIM验证结果")
        print("="*60)
        print(f"Episodes: {len(self.episode_data)}")
        print(f"\nEpisode长度:")
        print(f"  平均: {np.mean(steps):.1f} 步 ({np.mean(steps)*0.01:.1f}秒)")
        print(f"  标准差: {np.std(steps):.1f} 步")
        print(f"  最大: {np.max(steps)} 步 ({np.max(steps)*0.01:.1f}秒)")
        print(f"\n平衡性能:")
        print(f"  平均直立率: {np.mean(upright_ratios)*100:.1f}%")
        print(f"  平均倾角: {np.mean(mean_tilts):.1f}度")
        print(f"  平均最大倾角: {np.mean(max_tilts):.1f}度")
        print(f"\n奖励:")
        print(f"  平均: {np.mean(rewards):.2f}")
        print(f"  标准差: {np.std(rewards):.2f}")
        print("="*60)

    def save_results(self, filepath):
        """保存结果到JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.episode_data, f, indent=2)
        print(f"\n结果已保存到: {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Sim2Sim验证：在MuJoCo中运行IsaacGym策略')
    parser.add_argument('--checkpoint', type=str,
                       default='logs/wheel_legged_vmc_balance/Feb25_16-51-57_/model_950.pt',
                       help='训练好的策略checkpoint路径')
    parser.add_argument('--model', type=str,
                       default='resources/robots/serialleg/mjcf/serialleg_simple.xml',
                       help='MuJoCo MJCF模型路径')
    parser.add_argument('--episodes', type=int, default=10,
                       help='运行的episode数量')
    parser.add_argument('--max_steps', type=int, default=3000,
                       help='每个episode的最大步数 (30秒 @ 100Hz)')
    parser.add_argument('--render', action='store_true',
                       help='启用可视化')
    parser.add_argument('--device', type=str, default='cpu',
                       help='策略推理设备')
    args = parser.parse_args()

    # 初始化环境
    print(f"\n加载MuJoCo模型: {args.model}")
    env = MuJoCoBalanceEnv(args.model, render=args.render)

    # 加载策略
    print(f"加载策略checkpoint: {args.checkpoint}")
    policy = PolicyLoader(args.checkpoint, device=args.device)

    # 初始化指标跟踪
    metrics = BalanceMetrics()

    # 启动渲染
    if args.render:
        env.render()

    # 运行episodes
    print(f"\n开始运行 {args.episodes} 个episodes...")
    print("-" * 60)

    for episode in range(args.episodes):
        obs = env.reset(randomize=True)
        episode_reward = 0
        episode_steps = 0

        for step in range(args.max_steps):
            # 获取动作
            action = policy.get_action(obs)

            # 执行步骤
            obs, reward, done, info = env.step(action)

            # 跟踪指标
            metrics.update(env.data, step)
            episode_reward += reward
            episode_steps += 1

            if done:
                break

        # 记录episode结果
        metrics.log_episode(episode, episode_steps, episode_reward)

        print(f"Episode {episode+1}/{args.episodes} - "
              f"步数: {episode_steps}, "
              f"奖励: {episode_reward:.2f}, "
              f"直立率: {metrics.episode_data[-1]['upright_ratio']*100:.1f}%")

    # 打印最终统计
    metrics.print_summary()

    # 保存结果
    output_file = 'mujoco_sim2sim_results.json'
    metrics.save_results(output_file)

    # 关闭环境
    env.close()

    print("\n验证完成！")


if __name__ == '__main__':
    main()
