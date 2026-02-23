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
        # ----- base torque computation with gas-spring term -----
        if not hasattr(self, "_debug_limits_printed"):
            print("[DEBUG] torque_limits:", self.torque_limits.tolist())
            self._debug_limits_printed = True
        theta0_ref = (
            torch.cat(
                (
                    (actions[:, 0]).unsqueeze(1),
                    (actions[:, 3]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_theta
        )
        l0_ref = (
            torch.cat(
                (
                    (actions[:, 1]).unsqueeze(1),
                    (actions[:, 4]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_l0
        ) + self.cfg.control.l0_offset
        wheel_vel_ref = (
            torch.cat(
                (
                    (actions[:, 2]).unsqueeze(1),
                    (actions[:, 5]).unsqueeze(1),
                ),
                axis=1,
            )
            * self.cfg.control.action_scale_vel
        )

        # PD + gas spring force
        spring_force = 188.3447 * self.L0 + 1.2055  # gas spring reaction along leg
        self.torque_leg = (
            self.theta_kp * (theta0_ref - self.theta0) - self.theta_kd * self.theta0_dot
        )
        self.force_leg = (
            self.l0_kp * (l0_ref - self.L0) - self.l0_kd * self.L0_dot  + self.cfg.control.feedforward_force
        )
        self.torque_wheel = self.d_gains[:, [2, 5]] * (wheel_vel_ref - self.dof_vel[:, [2, 5]])
        T1, T2 = self.VMC(self.force_leg, self.torque_leg)
        torques = torch.cat(
            (
                T1[:, 0].unsqueeze(1),
                T2[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                T1[:, 1].unsqueeze(1),
                T2[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )
        torques = torch.clip(torques * self.torques_scale, -self.torque_limits, self.torque_limits)

        if self.ready_for_control.all():
            return torques

        # Not ready: hips & wheels off; leg force only gas spring (no PD), downward bias
        hold_torques = torch.zeros_like(torques)
        spring_force = 188.3447 * self.L0 + 1.2055
        T1_hold, T2_hold = self.VMC(spring_force, torch.zeros_like(self.torque_leg))
        hold_torques[:, 0] = 0.0  
        hold_torques[:, 3] = 0.0  # rf0 off
        hold_torques[:, 1] = -20
        hold_torques[:, 4] = -20
        hold_torques[:, 2] = 0.0  # wheels off
        hold_torques[:, 5] = 0.0
        hold_torques = torch.clip(hold_torques, -self.torque_limits, self.torque_limits)

        mask = self.ready_for_control.unsqueeze(1)
        return torch.where(mask, torques, hold_torques)

    # Reward: penalize joint torques above 30 Nm on hips/calves
    def _reward_torque_over_limit(self):
        joint_idx = torch.tensor([0, 1, 3, 4], device=self.device)
        over = (torch.abs(self.torques[:, joint_idx]) - 30.0).clip(min=0.0)
        return torch.sum(over, dim=1)

    def _reward_orientation_flip(self):
        # 移除翻转惩罚，允许机器人从倒立状态恢复
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_hip_upright(self):
        hip_idx = torch.tensor([0, 3], device=self.device)
        return torch.sum(torch.abs(self.dof_pos[:, hip_idx]), dim=1)

    def _reward_upright_bonus(self):
        grav_ok = (torch.abs(self.projected_gravity[:, 0]) < 0.05) & (
            torch.abs(self.projected_gravity[:, 1]) < 0.05
        )
        lin_ok = torch.norm(self.base_lin_vel, dim=1) < 0.3
        ang_ok = torch.norm(self.base_ang_vel, dim=1) < 0.3
        return (grav_ok & lin_ok & ang_ok).float()

    def _reward_recovery_speed(self):
        # 奖励快速恢复到直立状态
        # 计算当前姿态与直立姿态的距离
        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        height_error = torch.square(self.root_states[:, 2] - self.commands[:, 2])

        # 如果已经接近直立，给予高奖励
        is_recovering = orientation_error < 0.1
        return is_recovering.float() * (1.0 - orientation_error - 0.1 * height_error)

    def _reward_energy_efficiency(self):
        # 奖励能量效率：在恢复过程中使用更少的力矩
        # 计算归一化的力矩使用量
        joint_idx = torch.tensor([0, 1, 3, 4], device=self.device)
        torque_usage = torch.sum(torch.abs(self.torques[:, joint_idx]), dim=1) / 120.0  # 归一化 (4 joints * 30 Nm)

        # 只在恢复过程中奖励低力矩
        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        is_recovering = orientation_error > 0.05

        return is_recovering.float() * (1.0 - torque_usage)

    def _reward_ang_vel_yaw(self):
        """惩罚 yaw 轴自转，保持原地不动"""
        # 严格惩罚 yaw 角速度，避免自转
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_base_lin_vel_xy(self):
        """惩罚 xy 平面移动，保持原地不动"""
        # 惩罚 x 和 y 方向的线速度
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)

    def _reward_hip_pos_constraint(self):
        """约束髋关节位置，引导腿部摆动到后方"""
        hip_idx = torch.tensor([0, 3], device=self.device)
        hip_pos = self.dof_pos[:, hip_idx]

        # 计算腿部在前方的程度（正值表示在前方）
        # 当髋关节角度为正时，腿在前方
        forward_penalty = torch.relu(hip_pos)  # 只惩罚正值（腿在前方）

        # 鼓励髋关节保持在 -0.2 到 0.1 的范围内（略微向后）
        # 这样可以引导机器人先将腿摆到后方再收腿
        target_hip_pos = -0.1  # 目标位置：略微向后
        deviation = torch.abs(hip_pos - target_hip_pos)

        # 如果腿在前方，给予更大的惩罚
        penalty = torch.where(
            hip_pos > 0.0,
            forward_penalty * 2.0 + deviation,  # 腿在前方：双倍惩罚
            deviation * 0.5  # 腿在后方：较小惩罚
        )

        return torch.sum(penalty, dim=1)

    def check_termination(self):
        """阶段1: 恢复基础终止条件，先学会正立平衡"""
        # 检查身体部件接触地面
        fail_buf = torch.any(
            torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            > 10.0,
            dim=1,
        )
        # 阶段1: 保留重力方向检查，倒地时终止（先学会正立）
        # 当 projected_gravity z > -0.1 时，说明机器人严重倾斜或倒立
        fail_buf |= self.projected_gravity[:, 2] > -0.1

        self.fail_buf *= fail_buf
        self.fail_buf += fail_buf
        self.time_out_buf = (
            self.episode_length_buf > self.max_episode_length
        )

        self.reset_buf = (
            (self.fail_buf > self.cfg.env.fail_to_terminal_time_s / self.dt)
            | self.time_out_buf
        )

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
