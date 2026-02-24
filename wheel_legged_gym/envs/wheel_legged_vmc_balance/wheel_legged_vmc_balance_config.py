# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)


class WheelLeggedVMCBalanceCfg(WheelLeggedVMCCfg):
    """
    Balance task configuration: recover from any pose to Flat initial condition
    Two-stage control: Balance → Flat
    """

    class env(WheelLeggedVMCCfg.env):
        num_envs = 4096
        episode_length_s = 60  # 1 分钟超时重启
        fail_to_terminal_time_s = 60.0  # 与 episode_length_s 一致
        # 观测: 基类27项 + base_lin_vel(3) + base_height(1)
        num_observations = 31
        # 31 + measured_heights(77) + last_actions(12) + dof_acc(6) + dof_pos(6) + dof_vel(6) + torques(6) + base_mass/com(4) + default_dof_pos diff(6)
        num_privileged_obs = 157

    class terrain(WheelLeggedVMCCfg.terrain):
        mesh_type = "plane"

    class commands(WheelLeggedVMCCfg.commands):
        # Balance 任务不需要速度命令
        curriculum = False
        max_curriculum = 0.0
        num_commands = 3
        resampling_time = 10.0

        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]
            height = [0.20, 0.20]  # 目标高度固定为 0.20m

    class control(WheelLeggedVMCCfg.control):
        # 允许大幅度恢复动作
        action_scale_theta = 1.5   # 收窄以减少早期饱和但仍保留恢复空间
        action_scale_l0 = 0.1      # 10cm
        action_scale_vel = 10.0    # 轮速

        l0_offset = 0.23
        feedforward_force = 40.0

        # PD 增益
        kp_theta = 12.0
        kd_theta = 4.0
        kp_l0 = 600.0
        kd_l0 = 6.0

        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.15}

    class rewards(WheelLeggedVMCCfg.rewards):
        clip_single_reward = 10.0

        class scales(WheelLeggedVMCCfg.rewards.scales):
            # 关闭所有基类奖励
            tracking_lin_vel = 0.0
            tracking_lin_vel_enhance = 0.0
            tracking_ang_vel = 0.0
            base_height_enhance = 0.0
            nominal_state = 0.0
            collision = 0.0
            dof_pos_limits = 0.0
            action_smooth = 0.0
            dof_vel = 0.0
            dof_acc = 0.0
            orientation = 0.0
            base_height = 0.0
            hip_pos_constraint = 0.0
            orientation_flip = 0.0
            hip_upright = 0.0
            torque_over_limit = 0.0
            recovery_speed = 0.0
            energy_efficiency = 0.0
            ang_vel_yaw = 0.0
            base_lin_vel_xy = 0.0
            stand_still = 0.0
            upright_bonus = 0.0
            leg_angle_zero = 0.0
            pitch_angle = 0.0
            roll_angle = 0.0
            pitch_vel = 0.0
            roll_vel = 0.0
            reach_flat_target = 0.0

            # ========== robot_lab 标准奖励 ==========
            # General
            termination = -10.0

            # Root penalties (参考 robot_lab/velocity_env_cfg.py)
            lin_vel_z = -2.0           # lin_vel_z_l2
            ang_vel_xy = -0.5          # ang_vel_xy_l2
            # flat_orientation_l2 = 0.0  # 由 upward 替代

            # Joint penalties
            torques = -1e-3            # joint_torques_l2 (robot_lab 标准值)
            # joint_vel_l2 = 0.0
            # joint_acc_l2 = -2.5e-6
            # joint_pos_limits = 0.0
            # joint_power = 0.0

            # Action penalties
            action_rate = -0.05        # action_rate_l2 (robot_lab 标准值)

            # Contact sensor
            # undesired_contacts = 0.0

            # 倒地自救核心奖励
            upward = 1.0               # 鼓励直立姿态

    class domain_rand(WheelLeggedVMCCfg.domain_rand):
        push_robots = False
        randomize_default_dof_pos = False

    class asset(WheelLeggedVMCCfg.asset):
        terminate_after_contacts_on = []  # 不因接触而终止

    class init_state(WheelLeggedVMCCfg.init_state):
        # 目标状态（Flat 初始条件）
        pos = [0.0, 0.0, 0.20]  # 目标高度 0.20m（匹配实际机器人）
        rot = [0.0, 0.0, 0.0, 1.0]  # 完全直立
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        # 大范围随机初始化（从各种姿态开始）
        pos_offset = [0.0, 0.0, 0.0]
        rot_offset = [0.0, 0.0, 0.0]  # 将通过 balance_reset 设置

    class balance_reset:
        """渐进式初始化：从小角度开始训练"""
        # 位置扰动
        x_pos_offset = [0.0, 0.0]
        y_pos_offset = [0.0, 0.0]
        z_pos_offset = [0, 0]  # ±5cm（减小范围）

        # 姿态扰动（从小角度开始！）
        roll = [0, 0]            # ±17.2°（从 ±45.8° 减小）
        pitch = [0, 0]           # ±17.2°（从 ±45.8° 减小）
        yaw = [0, 0]             # ±11.5°（从 ±28.6° 减小）

        # 速度扰动
        lin_vel_x = [0, 0]       # ±0.5 m/s
        lin_vel_y = [0, 0]
        lin_vel_z = [0, 0]

        # 角速度扰动
        ang_vel_roll = [0, 0]    # ±1.0 rad/s
        ang_vel_pitch = [0, 0]
        ang_vel_yaw = [0, 0]


class WheelLeggedVMCBalanceCfgPPO(WheelLeggedVMCCfgPPO):
    class policy(WheelLeggedVMCCfgPPO.policy):
        # 确保编码器输入维度与更新后的观测长度一致
        num_encoder_obs = (
            WheelLeggedVMCBalanceCfg.env.num_observations
            * WheelLeggedVMCBalanceCfg.env.obs_history_length
        )
    class algorithm(WheelLeggedVMCCfgPPO.algorithm):
        learning_rate = 1e-4
        num_learning_epochs = 5
        num_mini_batches = 8
        entropy_coef = 0.01

    class runner(WheelLeggedVMCCfgPPO.runner):
        experiment_name = "wheel_legged_vmc_balance"
        max_iterations = 2000     # 需要更多 iterations 学习恢复
        save_interval = 50
