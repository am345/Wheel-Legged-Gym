# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float, quat_from_euler_xyz


from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_vmc_balance_config import WheelLeggedVMCBalanceCfg


class LeggedRobotVMCBalance(LeggedRobotVMC):
    """
    Balance task: same as Flat but with random initialization for fall recovery training
    Following robot_lab design
    """

    def __init__(
        self, cfg: WheelLeggedVMCBalanceCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # 计算pitch角度用于提示
        self.pitch_angle = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # Training-visible torques exclude passive gas-spring contribution.
        self.active_motor_torques = self.torques.clone()
        self._last_torque_debug_print_common_step = -1

    def post_physics_step(self):
        """计算pitch角度"""
        super().post_physics_step()

        # 计算pitch角度（从projected_gravity）
        self.pitch_angle = torch.atan2(
            self.projected_gravity[:, 1],
            -self.projected_gravity[:, 2]
        )

    def _get_training_visible_torques(self):
        return getattr(self, "active_motor_torques", self.torques)

    def compute_observations(self):
        """Use active motor torques (excluding gas spring) in privileged torque observations."""
        torques_total = self.torques
        self.torques = self._get_training_visible_torques()
        try:
            super().compute_observations()
        finally:
            self.torques = torques_total

    def _compute_torques(self, actions):
        """Balance task VMC torque with optional axial gas-spring force (no pitch prompt torque)."""
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

        hint_alpha = 0.0
        hint_trigger_mask = None
        if getattr(self.cfg.control, "pitch_l0_hint_enable", False):
            threshold = float(getattr(self.cfg.control, "pitch_l0_hint_threshold_rad", 0.349))
            hint_trigger_mask = (torch.abs(self.pitch_angle) > threshold).unsqueeze(1)
            if bool(hint_trigger_mask.any().item()):
                anneal_steps = int(getattr(self.cfg.control, "pitch_l0_hint_anneal_steps", 0))
                if anneal_steps > 0:
                    step_id = float(getattr(self, "common_step_counter", 0))
                    hint_alpha = max(0.0, 1.0 - step_id / float(anneal_steps))
                else:
                    hint_alpha = 1.0
                if hint_alpha > 0.0:
                    l0_ref_max = torch.full_like(
                        l0_ref,
                        float(
                            getattr(
                                self.cfg.control,
                                "pitch_l0_hint_target_l0",
                                self.cfg.control.l0_offset + self.cfg.control.action_scale_l0,
                            )
                        ),
                    )
                    hinted_l0_ref = torch.where(hint_trigger_mask, l0_ref_max, l0_ref)
                    # Annealed blend: early training uses strong hint, later fades out to policy target.
                    l0_ref = (1.0 - hint_alpha) * l0_ref + hint_alpha * hinted_l0_ref

        self.torque_leg = (
            self.theta_kp * (theta0_ref - self.theta0) - self.theta_kd * self.theta0_dot
        )
        self.force_leg = self.l0_kp * (l0_ref - self.L0) - self.l0_kd * self.L0_dot
        self.torque_wheel = self.d_gains[:, [2, 5]] * (
            wheel_vel_ref - self.dof_vel[:, [2, 5]]
        )

        active_axial_force = self.force_leg + self.cfg.control.feedforward_force
        gas_force = torch.zeros_like(self.force_leg)
        if getattr(self.cfg.control, "gas_spring_enable", False):
            # Linear gas spring: F[N] = gain * (k * l[m] + b), where l is current virtual leg length L0.
            gas_force = (
                getattr(self.cfg.control, "gas_spring_gain", 1.0)
                * (self.cfg.control.gas_spring_k * self.L0 + self.cfg.control.gas_spring_b)
            )

        T1_active, T2_active = self.VMC(active_axial_force, self.torque_leg)
        T1_gas, T2_gas = self.VMC(gas_force, torch.zeros_like(self.torque_leg))

        active_torques = torch.cat(
            (
                T1_active[:, 0].unsqueeze(1),
                T2_active[:, 0].unsqueeze(1),
                self.torque_wheel[:, 0].unsqueeze(1),
                T1_active[:, 1].unsqueeze(1),
                T2_active[:, 1].unsqueeze(1),
                self.torque_wheel[:, 1].unsqueeze(1),
            ),
            axis=1,
        )
        gas_torques = torch.cat(
            (
                T1_gas[:, 0].unsqueeze(1),
                T2_gas[:, 0].unsqueeze(1),
                torch.zeros_like(self.torque_wheel[:, 0]).unsqueeze(1),
                T1_gas[:, 1].unsqueeze(1),
                T2_gas[:, 1].unsqueeze(1),
                torch.zeros_like(self.torque_wheel[:, 1]).unsqueeze(1),
            ),
            axis=1,
        )

        # Motor torque randomization / clipping applies only to active motor torques.
        active_torques_scaled = active_torques * self.torques_scale
        self.active_motor_torques = torch.clip(
            active_torques_scaled, -self.torque_limits, self.torque_limits
        )
        # Passive gas-spring torques are added directly and do not consume motor torque budget.
        total_torques = self.active_motor_torques + gas_torques

        if getattr(self.cfg.control, "debug_print_torque_breakdown", False):
            step_id = int(getattr(self, "common_step_counter", 0))
            interval = max(1, int(getattr(self.cfg.control, "debug_print_torque_interval", 100)))
            if (
                step_id % interval == 0
                and self._last_torque_debug_print_common_step != step_id
            ):
                self._last_torque_debug_print_common_step = step_id
                env_id = int(getattr(self.cfg.control, "debug_print_torque_env_id", 0))
                env_id = max(0, min(env_id, self.num_envs - 1))
                print(
                    "[balance torque debug] "
                    f"step={step_id} env={env_id} "
                    f"pitch={float(self.pitch_angle[env_id].item()):+.3f} "
                    f"hint_alpha={hint_alpha:.3f} "
                    f"hint_on={bool(hint_trigger_mask is not None and hint_trigger_mask[env_id, 0].item())} "
                    f"L0={self.L0[env_id].detach().cpu().tolist()} "
                    f"gas_force={gas_force[env_id].detach().cpu().tolist()} "
                    f"gas_spring_torques={gas_torques[env_id].detach().cpu().tolist()} "
                    f"pos_ctrl_active_motor_torques={self.active_motor_torques[env_id].detach().cpu().tolist()} "
                    f"total_applied_torques={total_torques[env_id].detach().cpu().tolist()}"
                )

        return total_torques

    def _reward_torques(self):
        """Penalize only active motor torque, excluding passive gas-spring torque."""
        return torch.sum(torch.square(self._get_training_visible_torques()), dim=1)

    def _reward_power(self):
        """Penalize only active motor power, excluding passive gas-spring torque."""
        return torch.sum(torch.abs(self._get_training_visible_torques() * self.dof_vel), dim=1)

    def _reward_torque_limits(self):
        """Penalize active motor torques near limit, excluding passive gas-spring torque."""
        return torch.sum(
            (
                torch.abs(self._get_training_visible_torques())
                - self.torque_limits * self.cfg.rewards.soft_torque_limit
            ).clip(min=0.0),
            dim=1,
        )

    def _reward_upward(self):
        """鼓励机体保持直立：projected_gravity·z 接近 -1 (robot_lab 标准)"""
        return torch.square(1.0 + self.projected_gravity[:, 2])

    def check_termination(self):
        """只有超时才终止，允许从任何姿态恢复（参考 robot_lab）"""
        # 禁用所有失败条件
        self.fail_buf[:] = 0

        # 只有超时才重启
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        # 平地任务不需要边界检查
        self.edge_reset_buf[:] = 0

        # 最终重置条件：只有超时
        self.reset_buf = self.time_out_buf.clone()

    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        # Get balance_reset config
        cfg = self.cfg.balance_reset
        # 随机位置偏移
        self.root_states[env_ids, 0] += torch_rand_float(
            cfg.x_pos_offset[0], cfg.x_pos_offset[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.root_states[env_ids, 1] += torch_rand_float(
            cfg.y_pos_offset[0], cfg.y_pos_offset[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.root_states[env_ids, 2] += torch_rand_float(
            cfg.z_pos_offset[0], cfg.z_pos_offset[1], (len(env_ids), 1), device=self.device
        ).squeeze()

        # 随机姿态（欧拉角 -> 四元数）
        roll = torch_rand_float(cfg.roll[0], cfg.roll[1], (len(env_ids), 1), device=self.device)
        pitch = torch_rand_float(cfg.pitch[0], cfg.pitch[1], (len(env_ids), 1), device=self.device)
        yaw = torch_rand_float(cfg.yaw[0], cfg.yaw[1], (len(env_ids), 1), device=self.device)
                # 转换为四元数
        quat = quat_from_euler_xyz(roll, pitch, yaw).squeeze(1)  # [N, 1, 4] -> [N, 4]
        self.root_states[env_ids, 3:7] = quat

        # 随机速度
        self.root_states[env_ids, 7] = torch_rand_float(
            cfg.lin_vel_x[0], cfg.lin_vel_x[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.root_states[env_ids, 8] = torch_rand_float(
            cfg.lin_vel_y[0], cfg.lin_vel_y[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.root_states[env_ids, 9] = torch_rand_float(
            cfg.lin_vel_z[0], cfg.lin_vel_z[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.root_states[env_ids, 10] = torch_rand_float(
            cfg.ang_vel_roll[0], cfg.ang_vel_roll[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.root_states[env_ids, 11] = torch_rand_float(
            cfg.ang_vel_pitch[0], cfg.ang_vel_pitch[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.root_states[env_ids, 12] = torch_rand_float(
            cfg.ang_vel_yaw[0], cfg.ang_vel_yaw[1], (len(env_ids), 1), device=self.device
        ).squeeze()

        # 应用到仿真器

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )
