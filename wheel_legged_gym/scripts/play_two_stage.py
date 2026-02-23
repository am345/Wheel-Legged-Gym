#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
#
# Two-stage control: Balance → Flat
# - Stage 1 (Balance): Recover from any pose to Flat initial condition
# - Stage 2 (Flat): Maintain balance from Flat initial condition
#
# Keyboard Controls:
# - 'C': Enable control
# - 'R': Reset
# - 'S': Toggle stats
# - 'M': Manual stage switch
# - 'ESC': Exit

import os
import time
from isaacgym import gymapi
import torch
import numpy as np

import wheel_legged_gym.envs
from wheel_legged_gym.utils import get_args, task_registry
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR


class TwoStageMonitor:
    """监控两阶段控制"""

    def __init__(self, flat_target_height=0.20):
        self.flat_target_height = flat_target_height
        self.flat_angle_threshold = 0.1  # 5.7°
        self.flat_vel_threshold = 0.2

        self.reset_stats()
        self.show_stats = True

    def reset_stats(self):
        self.episode_count = 0
        self.success_count = 0
        self.total_steps = 0
        self.episode_steps = 0
        self.balance_steps = 0
        self.flat_steps = 0

        self.switch_count = 0
        self.avg_switch_time = 0

    def update(self, env, stage):
        self.episode_steps += 1
        self.total_steps += 1

        if stage == "balance":
            self.balance_steps += 1
        else:
            self.flat_steps += 1

    def on_switch(self):
        """切换到 Flat 阶段"""
        self.switch_count += 1
        switch_time = self.balance_steps * 0.02  # dt = 0.02s
        self.avg_switch_time = (self.avg_switch_time * (self.switch_count - 1) + switch_time) / self.switch_count

    def on_reset(self, success=False):
        if success:
            self.success_count += 1
        self.episode_count += 1
        self.episode_steps = 0
        self.balance_steps = 0
        self.flat_steps = 0

    def check_flat_condition(self, env):
        """检查是否达到 Flat 初始条件"""
        # 计算角度
        gravity = env.projected_gravity[0].cpu().numpy()
        pitch = np.arctan2(gravity[1], -gravity[2])
        roll = np.arctan2(gravity[0], -gravity[2])

        # 检查条件
        height = env.base_height[0].item()
        height_ok = abs(height - self.flat_target_height) < 0.05

        pitch_ok = abs(pitch) < self.flat_angle_threshold
        roll_ok = abs(roll) < self.flat_angle_threshold

        lin_vel = env.base_lin_vel[0].cpu().numpy()
        ang_vel = env.base_ang_vel[0].cpu().numpy()
        lin_vel_ok = np.linalg.norm(lin_vel) < self.flat_vel_threshold
        ang_vel_ok = np.linalg.norm(ang_vel) < self.flat_vel_threshold

        all_ok = height_ok and pitch_ok and roll_ok and lin_vel_ok and ang_vel_ok

        return all_ok, {
            'height': height,
            'height_ok': height_ok,
            'pitch': pitch * 180 / np.pi,
            'roll': roll * 180 / np.pi,
            'pitch_ok': pitch_ok,
            'roll_ok': roll_ok,
            'lin_vel_norm': np.linalg.norm(lin_vel),
            'ang_vel_norm': np.linalg.norm(ang_vel),
            'lin_vel_ok': lin_vel_ok,
            'ang_vel_ok': ang_vel_ok,
        }

    def print_stats(self, env, stage, started):
        if not self.show_stats or self.total_steps % 50 != 0:
            return

        os.system('clear' if os.name == 'posix' else 'cls')

        # 检查 Flat 条件
        flat_ok, conditions = self.check_flat_condition(env)

        # 状态评估
        if not started:
            status = "STOPPED"
        elif stage == "balance":
            if flat_ok:
                status = "BALANCE - READY TO SWITCH"
            else:
                status = "BALANCE - RECOVERING"
        else:
            status = "FLAT - MAINTAINING"

        print("=" * 70)
        print(f"  Two-Stage Control Monitor - {status}")
        print("=" * 70)
        print(f"\nCurrent Stage: {stage.upper()}")
        print(f"  Balance steps: {self.balance_steps}")
        print(f"  Flat steps:    {self.flat_steps}")
        print(f"  Total steps:   {self.episode_steps}")
        print("\nFlat Condition Check:")
        print(f"  Height:    {conditions['height']:.3f} m  (target: {self.flat_target_height:.3f} m)  {'✓' if conditions['height_ok'] else '✗'}")
        print(f"  Pitch:     {conditions['pitch']:+6.2f}°  {'✓' if conditions['pitch_ok'] else '✗'}")
        print(f"  Roll:      {conditions['roll']:+6.2f}°  {'✓' if conditions['roll_ok'] else '✗'}")
        print(f"  Lin Vel:   {conditions['lin_vel_norm']:.3f} m/s  {'✓' if conditions['lin_vel_ok'] else '✗'}")
        print(f"  Ang Vel:   {conditions['ang_vel_norm']:.3f} rad/s  {'✓' if conditions['ang_vel_ok'] else '✗'}")
        print(f"\n  Ready for Flat: {'YES' if flat_ok else 'NO'}")
        print("\nOverall Stats:")
        print(f"  Episodes:       {self.episode_count}")
        print(f"  Success:        {self.success_count} ({self.success_count/self.episode_count*100 if self.episode_count > 0 else 0:.1f}%)")
        print(f"  Stage switches: {self.switch_count}")
        if self.switch_count > 0:
            print(f"  Avg switch time: {self.avg_switch_time:.1f} s")
        print("\nControls:")
        print("  [C] Start  [R] Reset  [M] Manual Switch  [S] Toggle Stats  [ESC] Exit")
        print("=" * 70)


def play_two_stage(args):
    # 加载 Balance 环境和策略
    print("\n" + "=" * 70)
    print("  Loading Balance Policy...")
    print("=" * 70)

    balance_env_cfg, balance_train_cfg = task_registry.get_cfgs(name="wheel_legged_vmc_balance")
    balance_env_cfg.env.num_envs = 1
    balance_env_cfg.env.episode_length_s = 30
    balance_env_cfg.noise.add_noise = False
    balance_env_cfg.domain_rand.push_robots = False

    balance_env, _ = task_registry.make_env(name="wheel_legged_vmc_balance", args=args, env_cfg=balance_env_cfg)

    balance_train_cfg.runner.resume = True
    try:
        balance_runner, _ = task_registry.make_alg_runner(
            env=balance_env, name="wheel_legged_vmc_balance", args=args, train_cfg=balance_train_cfg
        )
        balance_policy = balance_runner.get_inference_policy(device=balance_env.device)
        print("✅ Balance policy loaded!")
    except Exception as e:
        print(f"❌ Failed to load Balance policy: {e}")
        print("💡 Train Balance policy first:")
        print("   python train.py --task=wheel_legged_vmc_balance")
        return

    # 加载 Flat 策略
    print("\n" + "=" * 70)
    print("  Loading Flat Policy...")
    print("=" * 70)

    flat_train_cfg = task_registry.get_train_cfg(name="wheel_legged_vmc_flat")
    flat_train_cfg.runner.resume = True
    try:
        flat_runner, _ = task_registry.make_alg_runner(
            env=balance_env, name="wheel_legged_vmc_flat", args=args, train_cfg=flat_train_cfg
        )
        flat_policy = flat_runner.get_inference_policy(device=balance_env.device)
        print("✅ Flat policy loaded!")
    except Exception as e:
        print(f"❌ Failed to load Flat policy: {e}")
        print("💡 Train Flat policy first:")
        print("   python train.py --task=wheel_legged_vmc_flat")
        return

    # 使用 Balance 环境（两个策略共享同一个环境）
    env = balance_env
    obs, obs_history = env.get_observations()

    # 初始化监控器
    monitor = TwoStageMonitor()

    # 键盘订阅
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_C, "START")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "RESET")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "TOGGLE_STATS")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_M, "MANUAL_SWITCH")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_ESCAPE, "EXIT")

    # 状态变量
    started = False
    stage = "balance"  # 初始阶段
    current_policy = balance_policy
    current_runner = balance_runner

    print("\n" + "=" * 70)
    print("  Two-Stage Control Started!")
    print("=" * 70)
    print("\n⌨️  Press 'C' to start control")
    print("⌨️  Press 'M' to manually switch stage")
    print("⌨️  Press 'R' to reset")
    print("⌨️  Press 'ESC' to exit\n")

    try:
        while True:
            # 处理键盘事件
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "START" and evt.value > 0:
                    started = True
                    stage = "balance"
                    current_policy = balance_policy
                    current_runner = balance_runner
                    print("\n🟢 Control ENABLED - Starting with Balance policy")

                elif evt.action == "RESET" and evt.value > 0:
                    success = monitor.episode_steps > env.max_episode_length * 0.8
                    monitor.on_reset(success=success)
                    env.reset_idx(torch.arange(env.num_envs, device=env.device))
                    started = False
                    stage = "balance"
                    current_policy = balance_policy
                    current_runner = balance_runner
                    print("\n🔴 RESET - Back to Balance stage")

                elif evt.action == "MANUAL_SWITCH" and evt.value > 0 and started:
                    if stage == "balance":
                        stage = "flat"
                        current_policy = flat_policy
                        current_runner = flat_runner
                        monitor.on_switch()
                        print("\n🔄 Manual switch to FLAT policy")
                    else:
                        stage = "balance"
                        current_policy = balance_policy
                        current_runner = balance_runner
                        print("\n🔄 Manual switch to BALANCE policy")

                elif evt.action == "TOGGLE_STATS" and evt.value > 0:
                    monitor.show_stats = not monitor.show_stats

                elif evt.action == "EXIT" and evt.value > 0:
                    print("\n👋 Exiting...")
                    return

            # 选择动作
            if started:
                if current_runner.alg.actor_critic.is_sequence:
                    actions, _ = current_policy(obs, obs_history)
                else:
                    actions = current_policy(obs.detach())

                # 自动切换逻辑
                if stage == "balance":
                    flat_ok, _ = monitor.check_flat_condition(env)
                    if flat_ok:
                        print("\n🎯 Flat condition reached! Switching to Flat policy...")
                        stage = "flat"
                        current_policy = flat_policy
                        current_runner = flat_runner
                        monitor.on_switch()
            else:
                actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

            # 执行步骤
            step_out = env.step(actions)
            obs = step_out[0]
            if len(step_out) > 5:
                obs_history = step_out[5]

            # 更新监控
            if started:
                monitor.update(env, stage)
                monitor.print_stats(env, stage, started)

            # 检查重置
            if env.reset_buf[0]:
                success = monitor.episode_steps > env.max_episode_length * 0.8
                monitor.on_reset(success=success)
                if success:
                    print("\n🎉 SUCCESS! Episode completed!")

    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Exiting...")
    finally:
        print("\n" + "=" * 70)
        print("  Final Statistics")
        print("=" * 70)
        print(f"  Total Episodes:  {monitor.episode_count}")
        print(f"  Success Rate:    {monitor.success_count}/{monitor.episode_count}")
        print(f"  Stage Switches:  {monitor.switch_count}")
        if monitor.switch_count > 0:
            print(f"  Avg Switch Time: {monitor.avg_switch_time:.1f} s")
        print("=" * 70)
        print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    play_two_stage(get_args())
