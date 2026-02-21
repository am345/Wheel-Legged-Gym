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

import torch

from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_vmc_balance_config import WheelLeggedVMCBalanceCfg


class LeggedRobotVMCBalance(LeggedRobotVMC):
    def __init__(
        self, cfg: WheelLeggedVMCBalanceCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.ready_for_control = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False
        )
        self.stable_steps = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device, requires_grad=False
        )

    def _reset_root_states(self, env_ids):
        """Reset root states with randomized recovery postures."""
        if len(env_ids) == 0:
            return

        reset_cfg = self.cfg.balance_reset
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 0] += torch_rand_float(
            reset_cfg.x_pos_offset[0],
            reset_cfg.x_pos_offset[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 1] += torch_rand_float(
            reset_cfg.y_pos_offset[0],
            reset_cfg.y_pos_offset[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 2] += torch_rand_float(
            reset_cfg.z_pos_offset[0],
            reset_cfg.z_pos_offset[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        roll = torch_rand_float(
            reset_cfg.roll[0],
            reset_cfg.roll[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        pitch = torch_rand_float(
            reset_cfg.pitch[0],
            reset_cfg.pitch[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        yaw = torch_rand_float(
            reset_cfg.yaw[0],
            reset_cfg.yaw[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 3:7] = self._quat_from_euler_xyz(roll, pitch, yaw)

        self.root_states[env_ids, 7] = torch_rand_float(
            reset_cfg.lin_vel_x[0],
            reset_cfg.lin_vel_x[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 8] = torch_rand_float(
            reset_cfg.lin_vel_y[0],
            reset_cfg.lin_vel_y[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 9] = torch_rand_float(
            reset_cfg.lin_vel_z[0],
            reset_cfg.lin_vel_z[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 10] = torch_rand_float(
            reset_cfg.ang_vel_roll[0],
            reset_cfg.ang_vel_roll[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 11] = torch_rand_float(
            reset_cfg.ang_vel_pitch[0],
            reset_cfg.ang_vel_pitch[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)
        self.root_states[env_ids, 12] = torch_rand_float(
            reset_cfg.ang_vel_yaw[0],
            reset_cfg.ang_vel_yaw[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.ready_for_control[env_ids] = False
        self.stable_steps[env_ids] = 0

    def _reset_dofs(self, env_ids):
        """Ensure hips/calf start at fully-extended posture."""
        super()._reset_dofs(env_ids)
        calf_idx = torch.tensor([1, 4], device=self.device)
        # keep calves at lower limit; let hips random (from parent reset) stay
        self.dof_pos[env_ids][:, calf_idx] = self.default_dof_pos[env_ids][:, calf_idx]
        self.dof_vel[env_ids][:, calf_idx] = 0.0

        # add explicit small randomness to hips after parent reset
        hip_idx = torch.tensor([0, 3], device=self.device)
        hip_noise = torch_rand_float(-0.3, 0.3, (len(env_ids), len(hip_idx)), device=self.device)
        self.dof_pos[env_ids][:, hip_idx] = torch.clamp(
            self.dof_pos[env_ids][:, hip_idx] + hip_noise,
            min=self.dof_pos_limits[hip_idx, 0],
            max=self.dof_pos_limits[hip_idx, 1],
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        lin_vel_ok = torch.norm(self.base_lin_vel, dim=1) < self.cfg.balance_control.lin_vel_thresh
        ang_vel_ok = torch.norm(self.base_ang_vel, dim=1) < self.cfg.balance_control.ang_vel_thresh
        stable = lin_vel_ok & ang_vel_ok
        self.stable_steps = torch.where(stable, self.stable_steps + 1, torch.zeros_like(self.stable_steps))
        ready_now = (self.stable_steps >= self.cfg.balance_control.stable_steps) | (
            self.envs_steps_buf >= self.cfg.balance_control.max_wait_steps
        )
        self.ready_for_control |= ready_now

    def step(self, actions):
        actions = actions.clone()
        actions[~self.ready_for_control] = 0.0
        return super().step(actions)

    def _compute_torques(self, actions):
        torques = super()._compute_torques(actions)
        if self.ready_for_control.all():
            return torques
        # Motors off for hips和轮子，给小腿恒定下压扭矩，保持在下限
        hold_torques = torch.zeros_like(torques)
        calf_indices = torch.tensor([1, 4], device=self.device)
        hold_torques[:, calf_indices] = -50.0  # constant negative torque
        hold_torques = torch.clip(hold_torques, -self.torque_limits, self.torque_limits)

        mask = self.ready_for_control.unsqueeze(1)
        return torch.where(mask, torques, hold_torques)

    @staticmethod
    def _quat_from_euler_xyz(roll, pitch, yaw):
        half_roll = roll * 0.5
        half_pitch = pitch * 0.5
        half_yaw = yaw * 0.5

        cr = torch.cos(half_roll)
        sr = torch.sin(half_roll)
        cp = torch.cos(half_pitch)
        sp = torch.sin(half_pitch)
        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return torch.stack((qx, qy, qz, qw), dim=-1)
