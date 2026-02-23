# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)


class WheelLeggedVMCBalanceCfg(WheelLeggedVMCCfg):
    class init_state(WheelLeggedVMCCfg.init_state):
        # keep height, roll/pitch/yaw from parent; set calf joints to fully extended (lower limit = -0.6 rad)
        default_joint_angles = WheelLeggedVMCCfg.init_state.default_joint_angles.copy()
        default_joint_angles.update(
            {
                "lf1_Joint": -0.6,
                "rf1_Joint": -0.6,
            }
        )

    class env(WheelLeggedVMCCfg.env):
        episode_length_s = 30
        fail_to_terminal_time_s = 10.0

    class terrain(WheelLeggedVMCCfg.terrain):
        mesh_type = "plane"

    class commands(WheelLeggedVMCCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 1000.0

        class ranges(WheelLeggedVMCCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            height = [0.18, 0.18]
            heading = [0.0, 0.0]

    class rewards(WheelLeggedVMCCfg.rewards):
        clip_single_reward = 5.0  # 增加 clip 范围

        class scales(WheelLeggedVMCCfg.rewards.scales):
            # 禁用速度跟踪
            tracking_lin_vel = 0.0
            tracking_lin_vel_enhance = 0.0
            tracking_ang_vel = 0.0

            # 核心奖励：保持直立和高度
            base_height = 20.0  # 增加高度奖励权重
            base_height_enhance = 5.0
            orientation = -20.0  # 增加姿态惩罚
            orientation_flip = 0.0
            upright_bonus = 10.0  # 大幅增加直立奖励

            # 保持静止 - 增强
            lin_vel_z = -2.0
            ang_vel_xy = -0.5  # 增加惩罚，减少 roll/pitch 角速度
            ang_vel_yaw = -1.0  # 新增：严格惩罚 yaw 自转
            base_lin_vel_xy = -0.5  # 新增：惩罚 xy 平面移动
            stand_still = 2.0  # 增加静止奖励

            # 关节控制
            hip_upright = -5.0
            nominal_state = 0.3
            hip_pos_constraint = -8.0  # 新增：约束髋关节位置

            # 平滑控制
            dof_vel = -1e-4
            dof_acc = -2e-7
            torques = -1e-4
            action_rate = -0.01
            action_smooth = -0.01

            # 安全约束
            collision = -1.0
            dof_pos_limits = -1.0
            termination = -10.0
            torque_over_limit = -30.0

            # 暂时禁用高级奖励
            recovery_speed = 0.0
            energy_efficiency = 0.0

    class domain_rand(WheelLeggedVMCCfg.domain_rand):
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.0
        randomize_default_dof_pos = False

    class asset(WheelLeggedVMCCfg.asset):
        # terminate episode if thighs/calf links contact the ground (simulate forbidden knee-ground contact)
        terminate_after_contacts_on = ["base", "lf0", "lf1", "rf0", "rf1"]

    class balance_reset:
        x_pos_offset = [-0.1, 0.1]
        y_pos_offset = [-0.1, 0.1]
        z_pos_offset = [-0.02, 0.04]
        # 阶段1: 先学会小角度平衡 (±30°)
        roll = [-0.5, 0.5]   # ±30° 开始训练
        pitch = [-0.5, 0.5]  # ±30° 开始训练
        yaw = [-3.1415926, 3.1415926]
        lin_vel_x = [-0.5, 0.5]   # 降低初始速度
        lin_vel_y = [-0.5, 0.5]
        lin_vel_z = [-0.3, 0.3]
        ang_vel_roll = [-1.5, 1.5]   # 降低初始角速度
        ang_vel_pitch = [-1.5, 1.5]
        ang_vel_yaw = [-1.5, 1.5]

    class balance_control:
        stable_steps = 5          # consecutive steps satisfying thresholds before enabling all motors
        max_wait_steps = 600      # safety cap; fallback enable even if not perfectly stable
        lin_vel_thresh = 0.08
        ang_vel_thresh = 0.15
        grav_xy_thresh = 0.06


class WheelLeggedVMCBalanceCfgPPO(WheelLeggedVMCCfgPPO):
    class algorithm(WheelLeggedVMCCfgPPO.algorithm):
        # 降低学习率，提高稳定性
        learning_rate = 5e-5  # 从默认的 1e-3 降低

    class runner(WheelLeggedVMCCfgPPO.runner):
        # logging
        experiment_name = "wheel_legged_vmc_balance"
        max_iterations = 5000  # 增加训练迭代次数
        save_interval = 100  # 更频繁地保存模型
