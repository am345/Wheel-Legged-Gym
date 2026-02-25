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

    class commands(WheelLeggedVMCCfg.commands):
        # 关闭curriculum，因为不需要速度跟踪
        curriculum = False

    class rewards(WheelLeggedVMCCfg.rewards):
        class scales(WheelLeggedVMCCfg.rewards.scales):
            # 关闭速度跟踪奖励，专注于站起来
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0

            # 添加 upward 奖励（倒地自救核心）
            upward = -1.0

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
    class runner(WheelLeggedVMCCfgPPO.runner):
        experiment_name = "wheel_legged_vmc_balance"
        max_iterations = 2000
        save_interval = 50
