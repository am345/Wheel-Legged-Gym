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
        action_scale_theta = 1.0   # 57.3° - 足够恢复大角度
        action_scale_l0 = 0.1      # 10cm
        action_scale_vel = 10.0    # 轮速

        l0_offset = 0.23
        feedforward_force = 40.0

        # PD 增益
        kp_theta = 8.0
        kd_theta = 4.0
        kp_l0 = 600.0
        kd_l0 = 6.0

        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.15}

    class rewards(WheelLeggedVMCCfg.rewards):
        clip_single_reward = 10.0

        class scales(WheelLeggedVMCCfg.rewards.scales):
            # 禁用所有非核心奖励
            tracking_lin_vel = 0.0
            tracking_lin_vel_enhance = 0.0
            tracking_ang_vel = 0.0
            base_height_enhance = 0.0
            nominal_state = 0.0
            collision = 0.0
            dof_pos_limits = 0.0
            action_rate = 0.0
            action_smooth = 0.0
            dof_vel = 0.0
            dof_acc = 0.0

            # 核心目标：达到 Flat 初始条件
            base_height = 50.0       # 高度控制到目标

            # 禁用原始的 orientation 和 ang_vel_xy
            orientation = 0.0
            ang_vel_xy = 0.0

            # 基于角度的姿态惩罚（强烈惩罚偏离 0°）
            pitch_angle = -100.0     # 目标：0°
            roll_angle = -100.0      # 目标：0°

            # 基于角速度的晃动惩罚
            pitch_vel = -10.0        # 目标：0 rad/s
            roll_vel = -10.0         # 目标：0 rad/s

            # 腿部控制
            leg_angle_zero = -5.0    # 鼓励腿垂直

            # 达到 Flat 目标的大奖励（关键！）
            reach_flat_target = 100.0  # 所有条件满足时

            # 直立奖励
            upright_bonus = 20.0

            # 保持静止
            lin_vel_z = -10.0        # 惩罚 z 方向速度
            stand_still = 10.0       # 奖励静止

            # 基本约束
            torques = -1e-4
            termination = -10.0

            # 禁用所有其他奖励
            orientation_flip = 0.0
            hip_upright = 0.0
            torque_over_limit = 0.0
            recovery_speed = 0.0
            energy_efficiency = 0.0
            ang_vel_yaw = -5.0       # 惩罚 yaw 旋转
            base_lin_vel_xy = -10.0  # 惩罚 xy 方向速度
            hip_pos_constraint = 0.0

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
        """大范围初始化：从任意姿态开始训练"""
        # 位置扰动
        x_pos_offset = [0.0, 0.0]
        y_pos_offset = [0.0, 0.0]
        z_pos_offset = [-0.1, 0.1]    # ±10cm

        # 姿态扰动（大范围！）
        roll = [-0.8, 0.8]            # ±45.8°
        pitch = [-0.8, 0.8]           # ±45.8°
        yaw = [-0.5, 0.5]             # ±28.6°

        # 速度扰动
        lin_vel_x = [0, 0]       # ±0.5 m/s
        lin_vel_y = [0, 0]
        lin_vel_z = [0, 0]

        # 角速度扰动
        ang_vel_roll = [0, 0]    # ±1.0 rad/s
        ang_vel_pitch = [0, 0]
        ang_vel_yaw = [0, 0]


class WheelLeggedVMCBalanceCfgPPO(WheelLeggedVMCCfgPPO):
    class algorithm(WheelLeggedVMCCfgPPO.algorithm):
        learning_rate = 1e-4
        num_learning_epochs = 5
        num_mini_batches = 8
        entropy_coef = 0.01

    class runner(WheelLeggedVMCCfgPPO.runner):
        experiment_name = "wheel_legged_vmc_balance"
        max_iterations = 2000     # 需要更多 iterations 学习恢复
        save_interval = 50
