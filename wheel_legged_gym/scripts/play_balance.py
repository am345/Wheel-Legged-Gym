# SPDX-License-Identifier: BSD-3-Clause
#
# Quick play script for wheel_legged_vmc_balance:
# - Starts in “power off” (ready_for_control=False, zero actions).
# - Press 'C' to enable control (ready_for_control=True) and start policy.
# - Press 'R' to reset (back to power-off).

import os
from isaacgym import gymapi
import torch

import wheel_legged_gym.envs  # ensure env package initializes before task_registry to avoid circular import
from wheel_legged_gym.utils import get_args, task_registry
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR


def play(args):
    args.task = args.task or "wheel_legged_vmc_balance"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # play defaults
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 30
    env_cfg.env.fail_to_terminal_time_s = 10.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs, obs_history = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    # keyboard subscriptions
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_C, "START")
    env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "RESET")

    started = False
    step_count = 0
    while True:
        # handle keyboard
        for evt in env.gym.query_viewer_action_events(env.viewer):
            if evt.action == "START" and evt.value > 0:
                started = True
                env.ready_for_control[:] = True
                print("[PLAY] Start command received -> control enabled")
            if evt.action == "RESET" and evt.value > 0:
                env.reset_idx(torch.arange(env.num_envs, device=env.device))
                started = False
                env.ready_for_control[:] = False
                print("[PLAY] Reset -> control disabled")

        # choose actions
        if started:
            if ppo_runner.alg.actor_critic.is_sequence:
                actions, _ = policy(obs, obs_history)
            else:
                actions = policy(obs.detach())
        else:
            actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)
            env.ready_for_control[:] = False

        step_out = env.step(actions)
        obs = step_out[0]
        if len(step_out) > 5:
            obs_history = step_out[5]
        step_count += 1

        if env.viewer is None:  # headless safety
            if step_count % 100 == 0:
                print(f"[PLAY] step {step_count}, started={started}")


if __name__ == "__main__":
    play(get_args())
