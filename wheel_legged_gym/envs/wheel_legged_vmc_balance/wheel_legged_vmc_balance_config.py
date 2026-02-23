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
        # 小腿完全伸展，提供更大的支撑面
        default_joint_angles = WheelLeggedVMCCfg.init_state.default_joint_angles.copy()
        default_joint_angles.update(
            {
                "lf1_Joint": -0.6,
                "rf1_Joint": -0.6,
            }
        )

    class env(WheelLeggedVMCCfg.env):
        episode_length_s = 20  # 缩短 episode，加快训练
        fail_to_terminal_time_s = 5.0  # 更快终止失败的 episode

    class terrain(WheelLeggedVMCCfg.terrain):
        mesh_type = "plane"

    class commands(WheelLeggedVMCCfg.commands):
        curriculum = False
        heading_command = False
        resampling_time = 1000.0

        class ranges(WheelLeggedVMCCfg.commands.ranges):
            lin_vel_x = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            height = [0.25, 0.25]
            heading = [0.0, 0.0]

    class control(WheelLeggedVMCCfg.control):
        # 优化的 PD 参数 - 针对平衡任务
        action_scale_theta = 0.3  # 降低角度动作幅度，更平滑
        action_scale_l0 = 0.05    # 降低腿长动作幅度
        action_scale_vel = 8.0    # 降低轮速动作幅度

        l0_offset = 0.22
        feedforward_force = 40.0  # 降低前馈力，减少过度补偿

        # 降低 PD 增益，提高稳定性
        kp_theta = 8.0   # 从 10.0 降低
        kd_theta = 4.0   # 从 5.0 降低
        kp_l0 = 600.0    # 从 800.0 降低
        kd_l0 = 6.0      # 从 7.0 降低

        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0}
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.15}  # 增加轮子阻尼

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

            # 核心奖励 - 优化权重
            base_height = 40.0       # 增加高度奖励（最重要）
            orientation = -40.0      # 增加姿态惩罚（最重要）
            upright_bonus = 20.0     # 增加直立奖励

            # 保持静止 - 增强
            lin_vel_z = -2.0         # 增加 z 方向速度惩罚
            ang_vel_xy = -1.0        # 增加 roll/pitch 角速度惩罚
            stand_still = 3.0        # 增加静止奖励

            # 基本约束
            torques = -5e-5          # 增加力矩惩罚
            termination = -10.0      # 增加终止惩罚

            # 禁用所有其他奖励
            orientation_flip = 0.0
            hip_upright = 0.0
            torque_over_limit = 0.0
            recovery_speed = 0.0
            energy_efficiency = 0.0
            ang_vel_yaw = 0.0
            base_lin_vel_xy = 0.0
            hip_pos_constraint = 0.0

    class domain_rand(WheelLeggedVMCCfg.domain_rand):
        push_robots = False  # 禁用推力扰动
        randomize_default_dof_pos = False

    class asset(WheelLeggedVMCCfg.asset):
        # terminate episode if thighs/calf links contact the ground (simulate forbidden knee-ground contact)
        terminate_after_contacts_on = ["base", "lf0", "lf1", "rf0", "rf1"]

    class balance_reset:
        # 最小化初始化：只从直立姿态开始，添加微小扰动
        x_pos_offset = [0.0, 0.0]
        y_pos_offset = [0.0, 0.0]
        z_pos_offset = [0.0, 0.0]
        roll = [0.0, 0.0]   # 完全直立
        pitch = [0.0, 0.0]  # 完全直立
        yaw = [0.0, 0.0]    # 固定朝向
        lin_vel_x = [0.0, 0.0]
        lin_vel_y = [0.0, 0.0]
        lin_vel_z = [0.0, 0.0]
        ang_vel_roll = [0.0, 0.0]
        ang_vel_pitch = [0.0, 0.0]
        ang_vel_yaw = [0.0, 0.0]

    class balance_control:
        stable_steps = 5          # consecutive steps satisfying thresholds before enabling all motors
        max_wait_steps = 600      # safety cap; fallback enable even if not perfectly stable
        lin_vel_thresh = 0.08
        ang_vel_thresh = 0.15
        grav_xy_thresh = 0.06


class WheelLeggedVMCBalanceCfgPPO(WheelLeggedVMCCfgPPO):
    class algorithm(WheelLeggedVMCCfgPPO.algorithm):
        # 优化的训练参数
        learning_rate = 1e-4      # 提高学习率，加快收敛
        num_learning_epochs = 5   # 增加学习轮数
        num_mini_batches = 8      # 增加 mini batch 数量
        entropy_coef = 0.01       # 增加探索

    class runner(WheelLeggedVMCCfgPPO.runner):
        experiment_name = "wheel_legged_vmc_balance"
        max_iterations = 2000     # 足够的迭代次数
        save_interval = 50        # 更频繁保存
