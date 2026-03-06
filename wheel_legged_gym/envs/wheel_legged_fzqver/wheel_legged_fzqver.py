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
        self.reset_curriculum_progress = 1.0
        self.current_learning_iteration = 0
        self.max_learning_iterations = 1

    def set_training_iteration(self, current_iter: int, max_iters: int):
        """Update the reset curriculum progress from the PPO iteration."""
        self.current_learning_iteration = max(int(current_iter), 0)
        self.max_learning_iterations = max(int(max_iters), 1)

        cfg = self.cfg.fzqver_reset_curriculum
        if not cfg.enabled:
            self.reset_curriculum_progress = 1.0
            return

        ramp_start_iter = int(cfg.ramp_start_iter)
        ramp_end_iter = max(
            int(float(cfg.ramp_end_frac) * self.max_learning_iterations),
            ramp_start_iter + 1,
        )
        progress = (self.current_learning_iteration - ramp_start_iter) / (
            ramp_end_iter - ramp_start_iter
        )
        self.reset_curriculum_progress = float(min(max(progress, 0.0), 1.0))

    def _get_thigh_reset_ranges(self):
        """Return the current reset sampling range for left and right thighs."""
        progress = float(self.reset_curriculum_progress)
        default_angles = self.cfg.init_state.default_joint_angles
        lf0_default = float(default_angles.get("lf0_Joint", 0.4))
        rf0_default = float(default_angles.get("rf0_Joint", 0.4))
        final_min = float(self.cfg.fzqver_reset_curriculum.thigh_final_range[0])
        final_max = float(self.cfg.fzqver_reset_curriculum.thigh_final_range[1])

        curr_min_l = lf0_default + (final_min - lf0_default) * progress
        curr_max_l = lf0_default + (final_max - lf0_default) * progress
        curr_min_r = rf0_default + (final_min - rf0_default) * progress
        curr_max_r = rf0_default + (final_max - rf0_default) * progress
        return curr_min_l, curr_max_l, curr_min_r, curr_max_r

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

    def _reset_dofs(self, env_ids):
        """Reset DOFs with a curriculum on thigh randomization difficulty."""
        if len(env_ids) == 0:
            return

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids, :]

        lf0_idx = self.dof_names.index("lf0_Joint") if "lf0_Joint" in self.dof_names else 0
        lf1_idx = self.dof_names.index("lf1_Joint") if "lf1_Joint" in self.dof_names else 1
        lw_idx = (
            self.dof_names.index("l_wheel_Joint")
            if "l_wheel_Joint" in self.dof_names
            else 2
        )
        rf0_idx = self.dof_names.index("rf0_Joint") if "rf0_Joint" in self.dof_names else 3
        rf1_idx = self.dof_names.index("rf1_Joint") if "rf1_Joint" in self.dof_names else 4
        rw_idx = (
            self.dof_names.index("r_wheel_Joint")
            if "r_wheel_Joint" in self.dof_names
            else 5
        )
        curr_min_l, curr_max_l, curr_min_r, curr_max_r = self._get_thigh_reset_ranges()

        self.dof_pos[env_ids, lf0_idx] = (
            torch.rand(len(env_ids), device=self.device) * (curr_max_l - curr_min_l)
            + curr_min_l
        )
        self.dof_pos[env_ids, rf0_idx] = (
            torch.rand(len(env_ids), device=self.device) * (curr_max_r - curr_min_r)
            + curr_min_r
        )
        self.dof_pos[env_ids, lf1_idx] = -0.6
        self.dof_pos[env_ids, rf1_idx] = -0.6
        self.dof_pos[env_ids, lw_idx] = 0.0
        self.dof_pos[env_ids, rw_idx] = 0.0
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        super().reset_idx(env_ids)

        if not self.cfg.fzqver_reset_curriculum.log_curriculum_stats:
            return

        curr_min_l, curr_max_l, curr_min_r, curr_max_r = self._get_thigh_reset_ranges()
        self.extras["episode"]["thigh_reset_curriculum_progress"] = (
            self.reset_curriculum_progress
        )
        self.extras["episode"]["thigh_reset_min"] = min(curr_min_l, curr_min_r)
        self.extras["episode"]["thigh_reset_max"] = max(curr_max_l, curr_max_r)

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

    def _get_wheel_dof_indices(self):
        if not hasattr(self, "_wheel_dof_idx_cache"):
            wheel_idx = [i for i, n in enumerate(self.dof_names) if "wheel" in n.lower()]
            if len(wheel_idx) == 0 and self.num_dof >= 6:
                wheel_idx = [2, 5]
            self._wheel_dof_idx_cache = torch.tensor(
                wheel_idx, dtype=torch.long, device=self.device
            )
        return self._wheel_dof_idx_cache

    def _get_leg_dof_indices(self):
        if not hasattr(self, "_leg_dof_idx_cache"):
            wheel_idx = set(self._get_wheel_dof_indices().tolist())
            leg_idx = [i for i in range(self.num_dof) if i not in wheel_idx]
            self._leg_dof_idx_cache = torch.tensor(
                leg_idx, dtype=torch.long, device=self.device
            )
        return self._leg_dof_idx_cache

    def _get_wheel_body_indices(self):
        if not hasattr(self, "_wheel_body_idx_cache"):
            wheel_body_idx = []
            try:
                body_names = self.gym.get_actor_rigid_body_names(
                    self.envs[0], self.actor_handles[0]
                )
                wheel_body_idx = [i for i, n in enumerate(body_names) if "wheel" in n.lower()]
            except Exception:
                wheel_body_idx = []
            if len(wheel_body_idx) == 0 and hasattr(self, "feet_indices"):
                wheel_body_idx = self.feet_indices.tolist()
            self._wheel_body_idx_cache = torch.tensor(
                wheel_body_idx, dtype=torch.long, device=self.device
            )
        return self._wheel_body_idx_cache

    def _get_mirror_dof_pairs(self):
        if not hasattr(self, "_mirror_dof_pairs_cache"):
            pairs = []
            name_to_idx = {n: i for i, n in enumerate(self.dof_names)}
            for name in self.dof_names:
                lname = name.lower()
                if "wheel" in lname:
                    continue
                if name.startswith("lf"):
                    mirror_name = "rf" + name[2:]
                    if mirror_name in name_to_idx:
                        pairs.append((name_to_idx[name], name_to_idx[mirror_name]))
            if len(pairs) == 0 and self.num_dof >= 5:
                pairs = [(0, 3), (1, 4)]
            self._mirror_dof_pairs_cache = pairs
        return self._mirror_dof_pairs_cache

    def _reward_upward(self):
        return torch.square(1.0 - self.projected_gravity[:, 2])

    def _reward_tracking_lin_vel(self):
        return super()._reward_tracking_lin_vel() * self._upright_factor()

    def _reward_tracking_ang_vel(self):
        return super()._reward_tracking_ang_vel() * self._upright_factor()

    def _reward_lin_vel_z(self):
        return super()._reward_lin_vel_z() * self._upright_factor()

    def _reward_ang_vel_xy(self):
        rew = super()._reward_ang_vel_xy()
        if self.cfg.fzqver_rewards.gate_ang_vel_xy_by_upright:
            return rew * self._upright_factor()
        return rew

    def _reward_base_height(self):
        return super()._reward_base_height() * self._upright_factor()

    def _reward_torques(self):
        leg_idx = self._get_leg_dof_indices()
        if leg_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.square(self.torques[:, leg_idx]), dim=1) * self._upright_factor()

    def _reward_torques_wheel(self):
        wheel_idx = self._get_wheel_dof_indices()
        if wheel_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.square(self.torques[:, wheel_idx]), dim=1) * self._upright_factor()

    def _reward_dof_vel(self):
        leg_idx = self._get_leg_dof_indices()
        if leg_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.square(self.dof_vel[:, leg_idx]), dim=1) * self._upright_factor()

    def _reward_dof_vel_wheel(self):
        wheel_idx = self._get_wheel_dof_indices()
        if wheel_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.square(self.dof_vel[:, wheel_idx]), dim=1) * self._upright_factor()

    def _reward_dof_acc(self):
        leg_idx = self._get_leg_dof_indices()
        if leg_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.square(self.dof_acc[:, leg_idx]), dim=1) * self._upright_factor()

    def _reward_dof_acc_wheel(self):
        wheel_idx = self._get_wheel_dof_indices()
        if wheel_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.square(self.dof_acc[:, wheel_idx]), dim=1) * self._upright_factor()

    def _reward_power(self):
        leg_idx = self._get_leg_dof_indices()
        if leg_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return (
            torch.sum(torch.abs(self.torques[:, leg_idx] * self.dof_vel[:, leg_idx]), dim=1)
            * self._upright_factor()
        )

    def _reward_action_rate(self):
        return super()._reward_action_rate() * self._upright_factor()

    def _reward_stand_still(self):
        leg_idx = self._get_leg_dof_indices()
        if leg_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        rew = torch.sum(
            torch.abs(self.dof_pos[:, leg_idx] - self.default_dof_pos[:, leg_idx]), dim=1
        ) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        return rew * self._upright_factor()

    def _reward_joint_pos_penalty(self):
        leg_idx = self._get_leg_dof_indices()
        if leg_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)

        cfg = self.cfg.fzqver_rewards
        cmd = torch.linalg.norm(self.commands[:, :2], dim=1)
        body_vel = torch.linalg.norm(self.base_lin_vel[:, :2], dim=1)
        running = torch.linalg.norm(
            self.dof_pos[:, leg_idx] - self.default_dof_pos[:, leg_idx], dim=1
        )
        rew = torch.where(
            torch.logical_or(
                cmd > cfg.joint_pos_penalty_command_threshold,
                body_vel > cfg.joint_pos_penalty_velocity_threshold,
            ),
            running,
            cfg.joint_pos_penalty_stand_still_scale * running,
        )
        return rew * self._upright_factor()

    def _reward_joint_mirror(self):
        pairs = self._get_mirror_dof_pairs()
        if len(pairs) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        rew = torch.zeros(self.num_envs, device=self.device)
        for li, ri in pairs:
            rew += torch.square(self.dof_pos[:, li] - self.dof_pos[:, ri])
        rew = rew / len(pairs)
        if self.cfg.fzqver_rewards.gate_joint_mirror_by_upright:
            return rew * self._upright_factor()
        return rew

    def _reward_dof_pos_limits(self):
        return super()._reward_dof_pos_limits() * self._upright_factor()

    def _reward_collision(self):
        return super()._reward_collision() * self._upright_factor()

    def _reward_contact_forces(self):
        wheel_body_idx = self._get_wheel_body_indices()
        if wheel_body_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        forces = torch.norm(self.contact_forces[:, wheel_body_idx, :], dim=-1)
        rew = torch.sum(
            (forces - self.cfg.fzqver_rewards.contact_force_threshold).clip(min=0.0), dim=1
        )
        return rew * self._upright_factor()

    def _reward_feet_contact_without_cmd(self):
        wheel_body_idx = self._get_wheel_body_indices()
        if wheel_body_idx.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        contact = (
            torch.norm(self.contact_forces[:, wheel_body_idx, :], dim=-1)
            > self.cfg.fzqver_rewards.feet_contact_threshold
        )
        rew = torch.sum(contact.float(), dim=-1)
        rew *= (
            torch.linalg.norm(self.commands[:, :2], dim=1)
            < self.cfg.fzqver_rewards.feet_contact_cmd_threshold
        )
        return rew * self._upright_factor()

    def _reward_gas_comp_torque(self):
        if not hasattr(self, "comp_torque_leg"):
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(torch.square(self.comp_torque_leg), dim=1) * self._upright_factor()
