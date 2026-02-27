# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)


class WheelLeggedFzqverCfg(WheelLeggedVMCCfg):
    class terrain(WheelLeggedVMCCfg.terrain):
        mesh_type = "plane"

    class env(WheelLeggedVMCCfg.env):
        episode_length_s = 20

    class commands(WheelLeggedVMCCfg.commands):
        curriculum = False
        heading_command = False

        class ranges(WheelLeggedVMCCfg.commands.ranges):
            lin_vel_x = [-0.6, 0.6]
            ang_vel_yaw = [-1.2, 1.2]
            height = [0.20, 0.24]

    class domain_rand(WheelLeggedVMCCfg.domain_rand):
        push_robots = False

    class rewards(WheelLeggedVMCCfg.rewards):
        class scales(WheelLeggedVMCCfg.rewards.scales):
            tracking_lin_vel = 2.5
            tracking_ang_vel = 1.0
            upward = 1.0

            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            base_height = 2.0

            dof_vel = -2e-5
            dof_acc = -2.5e-7
            torques = -2.5e-5
            action_rate = -0.01
            action_smooth = -0.005
            collision = -1.0
            dof_pos_limits = -5.0

            orientation = 0.0
            tracking_lin_vel_enhance = 0.0
            tracking_ang_vel_enhance = 0.0
            base_height_enhance = 0.0

    class fzqver_reset:
        fallen_ratio = 0.7

        fallen_roll_pitch_yaw = [-3.14, 3.14]
        upright_roll_pitch = [-0.25, 0.25]
        upright_yaw = [-3.14, 3.14]

        fallen_z_offset = [0.0, 0.08]
        upright_z_offset = [0.0, 0.03]

        lin_vel = [-0.5, 0.5]
        ang_vel = [-0.5, 0.5]

    class fzqver_command:
        stand_env_ratio = 0.35
        stand_height = 0.22


class WheelLeggedFzqverCfgPPO(WheelLeggedVMCCfgPPO):
    class policy(WheelLeggedVMCCfgPPO.policy):
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]

    class algorithm(WheelLeggedVMCCfgPPO.algorithm):
        desired_kl = 0.01
        entropy_coef = 0.01

    class runner(WheelLeggedVMCCfgPPO.runner):
        experiment_name = "wheel_legged_fzqver"
        max_iterations = 5000
        save_interval = 50
