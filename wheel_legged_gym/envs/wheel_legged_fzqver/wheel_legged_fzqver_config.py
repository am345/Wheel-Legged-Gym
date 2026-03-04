# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc_config import (
    WheelLeggedVMCCfg,
    WheelLeggedVMCCfgPPO,
)


class WheelLeggedFzqverCfg(WheelLeggedVMCCfg):
    class terrain(WheelLeggedVMCCfg.terrain):
        mesh_type = "plane"
        curriculum = False

    class env(WheelLeggedVMCCfg.env):
        episode_length_s = 20

    class commands(WheelLeggedVMCCfg.commands):
        curriculum = False
        heading_command = False

        class ranges(WheelLeggedVMCCfg.commands.ranges):
            lin_vel_x = [-0.8, 0.8]
            ang_vel_yaw = [-1.5, 1.5]
            height = [0.20, 0.24]

    class domain_rand(WheelLeggedVMCCfg.domain_rand):
        push_robots = False

    class asset(WheelLeggedVMCCfg.asset):
        # Collision penalty only tracks base-link contacts.
        penalize_contacts_on = [ "lf1", "rf1", "base"]

    class control(WheelLeggedVMCCfg.control):
        enable_gas_spring = True
        gas_spring_k = 188.3447 * 1.5
        gas_spring_b = 1.2055 *1.5

    class rewards(WheelLeggedVMCCfg.rewards):
        class scales(WheelLeggedVMCCfg.rewards.scales):
            # Velocity tracking rewards (Go2W)
            tracking_lin_vel = 3.0
            tracking_ang_vel = 1.5
            upward = 1.0

            # Root penalties (Go2W)
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            base_height = 2.0

            # Joint penalties (Go2W)
            torques = -2.5e-5
            torques_wheel = 0.0
            dof_vel = -2.5e-5
            dof_vel_wheel = 0.0
            dof_acc = -2.5e-7
            dof_acc_wheel = -2.5e-9
            power = -2.0e-5
            action_rate = -0.01
            stand_still = -2.0
            joint_pos_penalty = -1.0
            joint_mirror = -0.05
            dof_pos_limits = -0.0
            collision = -1.0
            contact_forces = 0.0
            feet_contact_without_cmd = 0.0

            orientation = 0.0
            nominal_state = 0.0
            tracking_lin_vel_enhance = 0.0
            tracking_ang_vel_enhance = 0.0
            base_height_enhance = 0.0
            action_smooth = 0.0

    class fzqver_rewards:
        upright_gating_max = 0.7
        gate_ang_vel_xy_by_upright = True
        gate_joint_mirror_by_upright = True
        joint_pos_penalty_stand_still_scale = 5.0
        joint_pos_penalty_velocity_threshold = 0.5
        joint_pos_penalty_command_threshold = 0.1
        contact_force_threshold = 100.0
        feet_contact_threshold = 1.0
        feet_contact_cmd_threshold = 0.1

    class fzqver_reset:
        upright_ratio = 0.2
        full_pose = [-3.14, 3.14]
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
        num_steps_per_env = 24
        max_iterations = 20000
        save_interval = 100


class WheelLeggedFzqverComp8Cfg(WheelLeggedFzqverCfg):
    class env(WheelLeggedFzqverCfg.env):
        num_actions = 8
        num_observations = 29
        # 3(base_lin_vel) + obs(29) + last_actions(8*2) + dof_acc(6)
        # + dof_pos(6) + dof_vel(6) + heights(77) + torques(6)
        # + base_mass_delta(1) + base_com(3) + default_pos_delta(6)
        # + friction(1) + restitution(1) = 161
        num_privileged_obs = 161

    class control(WheelLeggedFzqverCfg.control):
        enable_gas_spring = True
        enable_policy_gas_compensation = True
        policy_gas_comp_sigmoid_scale = 1.0

    class rewards(WheelLeggedFzqverCfg.rewards):
        class scales(WheelLeggedFzqverCfg.rewards.scales):
            gas_comp_torque = -2e-6

    class fzqver_rewards(WheelLeggedFzqverCfg.fzqver_rewards):
        gate_ang_vel_xy_by_upright = False
        gate_joint_mirror_by_upright = False


class WheelLeggedFzqverComp8CfgPPO(WheelLeggedFzqverCfgPPO):
    class policy(WheelLeggedFzqverCfgPPO.policy):
        num_encoder_obs = (
            WheelLeggedFzqverComp8Cfg.env.obs_history_length
            * WheelLeggedFzqverComp8Cfg.env.num_observations
        )

    class runner(WheelLeggedFzqverCfgPPO.runner):
        experiment_name = "wheel_legged_fzqver_comp8"
