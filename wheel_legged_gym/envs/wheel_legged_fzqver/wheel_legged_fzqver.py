# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz, torch_rand_float

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_fzqver_config import WheelLeggedFzqverCfg


class LeggedRobotVMCFzqver(LeggedRobotVMC):
    def __init__(
        self, cfg: WheelLeggedFzqverCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def check_termination(self):
        """Go2W-style termination: timeout-only."""
        self.fail_buf[:] = 0
        self.edge_reset_buf[:] = 0
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.time_out_buf.clone()

    def _resample_commands(self, env_ids):
        """Mix locomotion and standing commands in one policy."""
        super()._resample_commands(env_ids)
        if len(env_ids) == 0:
            return

        stand_mask = (
            torch.rand(len(env_ids), device=self.device)
            < self.cfg.fzqver_command.stand_env_ratio
        )
        stand_env_ids = env_ids[stand_mask]
        if len(stand_env_ids) == 0:
            return

        self.commands[stand_env_ids, 0] = 0.0
        self.commands[stand_env_ids, 1] = 0.0
        self.commands[stand_env_ids, 2] = self.cfg.fzqver_command.stand_height
        if self.commands.shape[1] > 3:
            self.commands[stand_env_ids, 3] = 0.0

    def _reset_root_states(self, env_ids):
        """Go2W-style full random pose reset with a small upright subset."""
        if len(env_ids) == 0:
            return

        cfg = self.cfg.fzqver_reset

        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(
                -1.0, 1.0, (len(env_ids), 2), device=self.device
            )
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        upright_mask = (
            torch.rand(len(env_ids), device=self.device) < cfg.upright_ratio
        )
        fallen_mask = ~upright_mask

        roll = torch.zeros(len(env_ids), device=self.device)
        pitch = torch.zeros(len(env_ids), device=self.device)
        yaw = torch.zeros(len(env_ids), device=self.device)

        if fallen_mask.any():
            fallen_count = int(fallen_mask.sum().item())
            roll[fallen_mask] = torch_rand_float(
                cfg.full_pose[0], cfg.full_pose[1], (fallen_count, 1), device=self.device
            ).squeeze(1)
            pitch[fallen_mask] = torch_rand_float(
                cfg.full_pose[0], cfg.full_pose[1], (fallen_count, 1), device=self.device
            ).squeeze(1)
            yaw[fallen_mask] = torch_rand_float(
                cfg.full_pose[0], cfg.full_pose[1], (fallen_count, 1), device=self.device
            ).squeeze(1)
            self.root_states[env_ids[fallen_mask], 2] += torch_rand_float(
                cfg.fallen_z_offset[0],
                cfg.fallen_z_offset[1],
                (fallen_count, 1),
                device=self.device,
            ).squeeze(1)

        if upright_mask.any():
            upright_count = int(upright_mask.sum().item())
            roll[upright_mask] = torch_rand_float(
                cfg.upright_roll_pitch[0],
                cfg.upright_roll_pitch[1],
                (upright_count, 1),
                device=self.device,
            ).squeeze(1)
            pitch[upright_mask] = torch_rand_float(
                cfg.upright_roll_pitch[0],
                cfg.upright_roll_pitch[1],
                (upright_count, 1),
                device=self.device,
            ).squeeze(1)
            yaw[upright_mask] = torch_rand_float(
                cfg.upright_yaw[0], cfg.upright_yaw[1], (upright_count, 1), device=self.device
            ).squeeze(1)
            self.root_states[env_ids[upright_mask], 2] += torch_rand_float(
                cfg.upright_z_offset[0],
                cfg.upright_z_offset[1],
                (upright_count, 1),
                device=self.device,
            ).squeeze(1)

        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        self.root_states[env_ids, 7:10] = torch_rand_float(
            cfg.lin_vel[0], cfg.lin_vel[1], (len(env_ids), 3), device=self.device
        )
        self.root_states[env_ids, 10:13] = torch_rand_float(
            cfg.ang_vel[0], cfg.ang_vel[1], (len(env_ids), 3), device=self.device
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _upright_factor(self):
        gate_max = max(float(self.cfg.fzqver_rewards.upright_gating_max), 1e-6)
        return torch.clamp(-self.projected_gravity[:, 2], 0.0, gate_max) / gate_max

    def _reward_upward(self):
        return torch.square(1.0 - self.projected_gravity[:, 2])

    def _reward_tracking_lin_vel(self):
        return super()._reward_tracking_lin_vel() * self._upright_factor()

    def _reward_tracking_ang_vel(self):
        return super()._reward_tracking_ang_vel() * self._upright_factor()

    def _reward_lin_vel_z(self):
        return super()._reward_lin_vel_z() * self._upright_factor()

    def _reward_ang_vel_xy(self):
        return super()._reward_ang_vel_xy() * self._upright_factor()

    def _reward_base_height(self):
        return super()._reward_base_height() * self._upright_factor()

    def _reward_torques(self):
        return super()._reward_torques() * self._upright_factor()

    def _reward_dof_vel(self):
        return super()._reward_dof_vel() * self._upright_factor()

    def _reward_dof_acc(self):
        return super()._reward_dof_acc() * self._upright_factor()

    def _reward_power(self):
        return super()._reward_power() * self._upright_factor()

    def _reward_action_rate(self):
        return super()._reward_action_rate() * self._upright_factor()

    def _reward_stand_still(self):
        return super()._reward_stand_still() * self._upright_factor()

    def _reward_dof_pos_limits(self):
        return super()._reward_dof_pos_limits() * self._upright_factor()

    def _reward_collision(self):
        return super()._reward_collision() * self._upright_factor()

