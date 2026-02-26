# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)


class WheelLeggedVMCBalanceCfg(WheelLeggedVMCCfg):
    """
    Balance task: same as Flat but with random initialization and upward reward
    """

    class terrain(WheelLeggedVMCCfg.terrain):
        mesh_type = "plane"

    class env(WheelLeggedVMCCfg.env):
        # Avoid timeout bootstrap feedback loop while debugging critic stability.
        send_timeouts = False
        # Shorter episodes reduce return magnitude/variance during early stabilization.
        episode_length_s = 10

    class commands(WheelLeggedVMCCfg.commands):
        # 关闭curriculum，因为不需要速度跟踪
        curriculum = False

        class ranges(WheelLeggedVMCCfg.commands.ranges):
            # Fix balance height command to a single target height.
            height = [0.25, 0.25]

    class control(WheelLeggedVMCCfg.control):
        # Replace constant leg feedforward with measured linear gas spring in balance task.
        feedforward_force = 0.0

        # Gas spring model: F[N] = gain * (k * l[m] + b), where l is current virtual leg length L0.
        gas_spring_enable = True
        gas_spring_gain = 1.5  # dimensionless scale for easy tuning
        gas_spring_k = 188.3447  # [N/m]
        gas_spring_b = 1.2055  # [N]

        # Annealed stand-up hint: if |pitch| exceeds threshold, blend l0_ref toward max leg length target.
        pitch_l0_hint_enable = True
        pitch_l0_hint_threshold_rad = 0.349  # 20 deg
        pitch_l0_hint_anneal_steps = 100000  # common env steps (post_physics_step counter)
        pitch_l0_hint_target_l0 = 0.32  # [m], close to action-space max l0_offset + action_scale_l0

        # Terminal torque debug print (env-level cadence, prints env0 by default).
        debug_print_torque_breakdown = True
        debug_print_torque_interval = 100
        debug_print_torque_env_id = 0

    class rewards(WheelLeggedVMCCfg.rewards):
        # Keep config-level target consistent with the fixed height command.
        base_height_target = 0.25

        class scales(WheelLeggedVMCCfg.rewards.scales):
            # 关闭速度跟踪奖励，专注于站起来
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

            # Slightly strengthen height-seeking behavior for stand-up / leg retraction.
            base_height = -8.0
            base_height_enhance = 1.0

            # 添加 upward 奖励（倒地自救核心）
            upward = -1.5

    class balance_reset:
        """Random initialization for fall recovery training"""
        # 位置扰动
        x_pos_offset = [0.0, 0.0]
        y_pos_offset = [0.0, 0.0]
        z_pos_offset = [0.0, 0.0]

        # 姿态扰动（全范围随机，参考 robot_lab）
        roll = [-3.14, 3.14]
        pitch = [-3.14, 3.14]
        yaw = [-3.14, 3.14]

        # 速度扰动
        lin_vel_x = [-0.5, 0.5]
        lin_vel_y = [-0.5, 0.5]
        lin_vel_z = [-0.5, 0.5]

        # 角速度扰动
        ang_vel_roll = [-0.5, 0.5]
        ang_vel_pitch = [-0.5, 0.5]
        ang_vel_yaw = [-0.5, 0.5]


class WheelLeggedVMCBalanceCfgPPO(WheelLeggedVMCCfgPPO):
    class policy(WheelLeggedVMCCfgPPO.policy):
        # Lower exploration while the critic is unstable under strong passive dynamics.
        init_noise_std = 0.3

    class algorithm(WheelLeggedVMCCfgPPO.algorithm):
        # Conservative critic optimization to prevent timeout-bootstrapped value blow-ups.
        learning_rate = 1.0e-3
        value_loss_coef = 0.5
        max_grad_norm = 0.5

    class runner(WheelLeggedVMCCfgPPO.runner):
        experiment_name = "wheel_legged_vmc_balance"
        max_iterations = 4000
        save_interval = 50
