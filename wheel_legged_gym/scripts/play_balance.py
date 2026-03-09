# SPDX-License-Identifier: BSD-3-Clause
#
# Enhanced play script for wheel_legged_vmc_balance with real-time monitoring
#
# Keyboard Controls:
# - 'C': Enable control (start policy)
# - 'R': Reset environment
# - 'S': Toggle statistics display
# - 'P': Pause/Resume
# - 'ESC': Exit

import os
import time
from isaacgym import gymapi, gymtorch
import torch
import numpy as np

import wheel_legged_gym.envs
from wheel_legged_gym.utils import get_args, task_registry
from wheel_legged_gym import WHEEL_LEGGED_GYM_ROOT_DIR

SUPPORTED_TASKS = (
    "wheel_legged_vmc_balance",
    "wheel_legged_fzqver",
)
DEFAULT_TASK = "wheel_legged_vmc_balance"


def custom_reset_dofs(env, env_ids):
    """Custom DOF reset with fixed shin joints and randomized thigh joints

    Joint indices:
    0: lf0_Joint (left thigh)
    1: lf1_Joint (left shin)
    2: l_wheel_Joint (left wheel)
    3: rf0_Joint (right thigh)
    4: rf1_Joint (right shin)
    5: r_wheel_Joint (right wheel)
    """
    if env_ids is None:
        return
    if isinstance(env_ids, torch.Tensor):
        if env_ids.numel() == 0:
            return
    elif len(env_ids) == 0:
        return

    # Randomize thigh joints (f0) in range [-3.14, 3.14]
    env.dof_pos[env_ids, 0] = torch.rand(len(env_ids), device=env.device) * 6.28 - 3.14  # lf0
    env.dof_pos[env_ids, 3] = torch.rand(len(env_ids), device=env.device) * 6.28 - 3.14  # rf0

    # Fix shin joints (f1) to -0.6
    env.dof_pos[env_ids, 1] = -0.6  # lf1
    env.dof_pos[env_ids, 4] = -0.6  # rf1

    # Keep wheel joints at 0
    env.dof_pos[env_ids, 2] = 0.0  # l_wheel
    env.dof_pos[env_ids, 5] = 0.0  # r_wheel

    # Set all velocities to zero
    env.dof_vel[env_ids] = 0.0

    # Apply to simulation
    env_ids_int32 = env_ids.to(dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(env_ids_int32),
        len(env_ids_int32),
    )


class BalanceMonitor:
    """Real-time monitoring for robot balance"""

    def __init__(self):
        self.reset_stats()
        self.show_stats = True

    def reset_stats(self):
        self.episode_count = 0
        self.success_count = 0
        self.total_steps = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.max_episode_length = 0
        self.height_history = []
        self.orientation_history = []
        self.episode_lengths = []

    def update(self, env):
        """Update statistics"""
        self.episode_steps += 1
        self.total_steps += 1
        height = env.base_height[0].item()
        gravity = env.projected_gravity[0].cpu().numpy()
        self.height_history.append(height)
        self.orientation_history.append(gravity)
        if len(self.height_history) > 1000:
            self.height_history.pop(0)
            self.orientation_history.pop(0)

    def on_reset(self, success=False):
        """Called when episode ends"""
        self.episode_lengths.append(self.episode_steps)
        self.max_episode_length = max(self.max_episode_length, self.episode_steps)
        if success:
            self.success_count += 1
        self.episode_count += 1
        self.episode_steps = 0
        self.episode_start_time = time.time()
        if len(self.episode_lengths) > 100:
            self.episode_lengths.pop(0)

    def print_stats(self, env, started):
        """Print statistics to console"""
        if not self.show_stats or self.total_steps % 50 != 0:
            return

        os.system('clear' if os.name == 'posix' else 'cls')

        height = env.base_height[0].item()
        target_height = env.commands[0, 2].item()
        height_error = abs(height - target_height)
        gravity = env.projected_gravity[0].cpu().numpy()
        roll = np.arctan2(gravity[0], -gravity[2]) * 180 / np.pi
        pitch = np.arctan2(gravity[1], -gravity[2]) * 180 / np.pi
        lin_vel = env.base_lin_vel[0].cpu().numpy()
        ang_vel = env.base_ang_vel[0].cpu().numpy()
        tilt_angle = np.sqrt(roll**2 + pitch**2)
        episode_time = self.episode_steps * env.dt
        success_rate = (self.success_count / self.episode_count * 100) if self.episode_count > 0 else 0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0

        status = "STOPPED"
        if started:
            if tilt_angle < 5 and height_error < 0.05:
                status = "EXCELLENT"
            elif tilt_angle < 10 and height_error < 0.1:
                status = "GOOD"
            elif tilt_angle < 20:
                status = "UNSTABLE"
            else:
                status = "FALLING"

        print("=" * 60)
        print(f"  Balance Monitor - {status}")
        print("=" * 60)
        print("\nCurrent State:")
        print(f"  Height:      {height:.3f} m  (target: {target_height:.3f} m, error: {height_error:.3f} m)")
        print(f"  Roll:        {roll:+6.2f} deg")
        print(f"  Pitch:       {pitch:+6.2f} deg")
        print(f"  Tilt Angle:  {tilt_angle:6.2f} deg  {'OK' if tilt_angle < 10 else 'BAD'}")
        print("\nVelocities:")
        print(f"  Linear:  [{lin_vel[0]:+6.3f}, {lin_vel[1]:+6.3f}, {lin_vel[2]:+6.3f}] m/s")
        print(f"  Angular: [{ang_vel[0]:+6.3f}, {ang_vel[1]:+6.3f}, {ang_vel[2]:+6.3f}] rad/s")
        print("\nEpisode Stats:")
        print(f"  Time:        {episode_time:.1f} s  (steps: {self.episode_steps})")
        print(f"  Max Length:  {self.max_episode_length * env.dt:.1f} s  ({self.max_episode_length} steps)")
        print("\nOverall Stats:")
        print(f"  Episodes:    {self.episode_count}")
        print(f"  Success:     {self.success_count} ({success_rate:.1f}%)")
        print(f"  Avg Length:  {avg_length * env.dt:.1f} s  ({avg_length:.0f} steps)")
        print(f"  Total Steps: {self.total_steps}")
        print("\nControls:")
        print("  [C] Start  [R] Reset  [S] Toggle Stats  [P] Pause  [ESC] Exit")
        print("=" * 60)


def play(args):
    if not args.task or args.task == "anymal_c_flat":
        args.task = DEFAULT_TASK
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task '{args.task}'. Expected one of {SUPPORTED_TASKS}.")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    use_custom_reset = args.task == "wheel_legged_vmc_balance"

    # Play configuration
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 30
    env_cfg.env.fail_to_terminal_time_s = 10.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False

    # Disable default joint randomization only when using custom reset.
    if use_custom_reset:
        env_cfg.domain_rand.randomize_default_dof_pos = False

    # Create environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    if use_custom_reset:
        # Wrap reset_idx to apply custom joint initialization for balance-debug task.
        original_reset_idx = env.reset_idx

        def wrapped_reset_idx(env_ids):
            original_reset_idx(env_ids)
            if isinstance(env_ids, torch.Tensor):
                if env_ids.numel() == 0:
                    return
            elif len(env_ids) == 0:
                return
            custom_reset_dofs(env, env_ids)

        env.reset_idx = wrapped_reset_idx
        # Apply initial custom reset
        custom_reset_dofs(env, torch.arange(env.num_envs, device=env.device))

    # Torque gate: force zero torques until control is explicitly enabled.
    control_enabled = {"value": env.viewer is None}
    zero_torques = torch.zeros_like(env.torques)
    original_compute_torques = env._compute_torques

    def wrapped_compute_torques(actions):
        if not control_enabled["value"]:
            return zero_torques
        return original_compute_torques(actions)

    env._compute_torques = wrapped_compute_torques

    obs, obs_history = env.get_observations()

    # Load policy
    print("\nLoading trained policy...")
    train_cfg.runner.resume = True
    try:
        ppo_runner, train_cfg = task_registry.make_alg_runner(
            env=env, name=args.task, args=args, train_cfg=train_cfg
        )
        policy = ppo_runner.get_inference_policy(device=env.device)
        print("Policy loaded successfully!")
    except Exception as e:
        print(f"Failed to load policy: {e}")
        print("Make sure you have trained the model first:")
        print(f"  python wheel_legged_gym/scripts/train.py --task={args.task}")
        return

    # Initialize monitor
    monitor = BalanceMonitor()

    # Keyboard subscriptions
    if env.viewer is not None:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_C, "START")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_R, "RESET")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "TOGGLE_STATS")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_P, "PAUSE")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_ESCAPE, "EXIT")

    # State variables
    started = control_enabled["value"]
    paused = False

    print("\n" + "=" * 60)
    print("  Balance Test Started!")
    print("=" * 60)
    print(f"\nTask: {args.task}  |  obs_dim={env.num_obs}  action_dim={env.num_actions}")
    print(f"Reset mode: {'custom_dof_debug_reset' if use_custom_reset else 'task_default_reset'}")
    print(f"Torque gate: {'enabled' if control_enabled['value'] else 'disabled'}")
    if env.viewer is not None:
        print("\nPress 'C' to start control")
        print("Press 'R' to reset")
        print("Press 'S' to toggle statistics")
        print("Press 'P' to pause/resume")
        print("Press 'ESC' to exit\n")
    else:
        print("\nHeadless mode: control auto-started.\n")

    try:
        while True:
            # Handle keyboard events
            if env.viewer is not None:
                for evt in env.gym.query_viewer_action_events(env.viewer):
                    if evt.action == "START" and evt.value > 0:
                        started = True
                        control_enabled["value"] = True
                        print("\nControl ENABLED - Policy and torque output are now active!")

                    elif evt.action == "RESET" and evt.value > 0:
                        success = monitor.episode_steps > env.max_episode_length * 0.8
                        monitor.on_reset(success=success)
                        env.reset_idx(torch.arange(env.num_envs, device=env.device))
                        started = False
                        control_enabled["value"] = False
                        if use_custom_reset:
                            print(
                                "\nRESET - Control disabled "
                                "(custom joint init: thigh random, shin=-0.6, torque gate closed)"
                            )
                        else:
                            print("\nRESET - Control disabled (task default reset, torque gate closed)")

                    elif evt.action == "TOGGLE_STATS" and evt.value > 0:
                        monitor.show_stats = not monitor.show_stats
                        print(f"\nStatistics display: {'ON' if monitor.show_stats else 'OFF'}")

                    elif evt.action == "PAUSE" and evt.value > 0:
                        paused = not paused
                        print(f"\n{'PAUSED' if paused else 'RESUMED'}")

                    elif evt.action == "EXIT" and evt.value > 0:
                        print("\nExiting...")
                        return

            if paused:
                time.sleep(0.1)
                continue

            # Choose actions
            if started:
                if ppo_runner.alg.actor_critic.is_sequence:
                    actions, _ = policy(obs, obs_history)
                else:
                    actions = policy(obs.detach())
            else:
                actions = torch.zeros((env.num_envs, env.num_actions), device=env.device)

            # Execute step
            step_out = env.step(actions)
            obs = step_out[0]
            if len(step_out) > 5:
                obs_history = step_out[5]

            # Update monitor
            if started:
                monitor.update(env)
                monitor.print_stats(env, started)

            # Check for auto-reset
            if env.reset_buf[0]:
                success = monitor.episode_steps > env.max_episode_length * 0.8
                monitor.on_reset(success=success)
                if success:
                    print("\nSUCCESS! Episode completed!")
                else:
                    print("\nEpisode terminated (robot fell or timeout)")

            # Headless safety
            if env.viewer is None:
                if monitor.total_steps % 100 == 0:
                    print(f"[HEADLESS] step {monitor.total_steps}, started={started}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Print final statistics
        print("\n" + "=" * 60)
        print("  Final Statistics")
        print("=" * 60)
        print(f"  Total Episodes:  {monitor.episode_count}")
        print(f"  Success Rate:    {monitor.success_count}/{monitor.episode_count} ({monitor.success_count/monitor.episode_count*100 if monitor.episode_count > 0 else 0:.1f}%)")
        print(f"  Max Length:      {monitor.max_episode_length * env.dt:.1f} s")
        if monitor.episode_lengths:
            print(f"  Avg Length:      {np.mean(monitor.episode_lengths) * env.dt:.1f} s")
        print(f"  Total Steps:     {monitor.total_steps}")
        print("=" * 60)
        print("\nGoodbye!\n")


if __name__ == "__main__":
    play(get_args())
