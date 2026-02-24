# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float, quat_from_euler_xyz

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_vmc_balance_config import WheelLeggedVMCBalanceCfg


class LeggedRobotVMCBalance(LeggedRobotVMC):
    """
    Balance task: recover from any initial pose to Flat task initial condition
    Goal: reach a state where Flat policy can take over

    启动流程（模拟真实情况）：
    1. 断电状态：腿最长（lf1/rf1 在 lower_limit，气弹簧撑开）
    2. 自由落体：以不同姿态落下
    3. 等待稳定：机器人静止后才开始控制
    4. 开始学习：从稳定状态恢复到 Flat 姿态
    """

    def __init__(
        self, cfg: WheelLeggedVMCBalanceCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # Flat 任务的目标状态（交接点）
        self.flat_target_height = 0.20  # m，匹配 Flat 任务的实际初始高度
        self.flat_angle_threshold = 0.1  # rad，约 5.7°
        self.flat_vel_threshold = 0.2    # m/s 或 rad/s

        # 用于存储计算的角度和角速度
        self.pitch_angle = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.roll_angle = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.pitch_vel = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.roll_vel = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)

        # 启动流程控制
        self.ready_for_control = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.settling_time = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.settling_threshold = 0.2  # 稳定时间阈值（秒）- 更快进入控制
        self.vel_stable_threshold = 0.3  # 速度稳定阈值（m/s 或 rad/s）

        # 调试日志
        import os
        self.debug_log_path = "/tmp/balance_debug.log"
        if os.path.exists(self.debug_log_path):
            os.remove(self.debug_log_path)

    def post_physics_step(self):
        """在每个物理步之后计算 pitch 和 roll，并检查是否稳定"""
        super().post_physics_step()

        # 计算角度
        self.pitch_angle = torch.atan2(
            self.projected_gravity[:, 1],
            -self.projected_gravity[:, 2]
        )
        self.roll_angle = torch.atan2(
            self.projected_gravity[:, 0],
            -self.projected_gravity[:, 2]
        )

        # 提取角速度
        self.roll_vel = self.base_ang_vel[:, 0]
        self.pitch_vel = self.base_ang_vel[:, 1]

        # 检查机器人是否稳定（速度接近 0）
        lin_vel_norm = torch.norm(self.base_lin_vel, dim=1)
        ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)
        is_stable = (lin_vel_norm < self.vel_stable_threshold) & (ang_vel_norm < self.vel_stable_threshold)

        # 更新稳定时间
        self.settling_time = torch.where(
            is_stable,
            self.settling_time + self.dt,
            torch.zeros_like(self.settling_time)
        )

        # 稳定时间超过阈值后，允许控制
        self.ready_for_control = self.settling_time >= self.settling_threshold

        # 添加调试信息到 TensorBoard
        self.extras["ready_ratio"] = self.ready_for_control.float().mean()
        self.extras["settling_time_mean"] = self.settling_time.mean()
        self.extras["lin_vel_norm_mean"] = lin_vel_norm.mean()
        self.extras["ang_vel_norm_mean"] = ang_vel_norm.mean()
        self.extras["pitch_angle_abs_mean"] = torch.abs(self.pitch_angle).mean()
        self.extras["roll_angle_abs_mean"] = torch.abs(self.roll_angle).mean()

        # 添加腿部角度信息
        theta_world = self.theta0 + self.pitch_angle.unsqueeze(1)
        self.extras["theta_world_abs_mean"] = torch.abs(theta_world).mean()

        # 调试信息（每 100 步打印一次）
        if self.common_step_counter % 100 == 0:
            ready_ratio = self.ready_for_control.float().mean().item()
            avg_lin_vel = lin_vel_norm.mean().item()
            avg_ang_vel = ang_vel_norm.mean().item()
            avg_settling = self.settling_time.mean().item()
            avg_pitch = torch.abs(self.pitch_angle).mean().item() * 57.3
            avg_roll = torch.abs(self.roll_angle).mean().item() * 57.3

            with open(self.debug_log_path, 'a', encoding='utf-8') as f:
                f.write(f"Step {self.common_step_counter}: Ready={ready_ratio:.2%}, LinVel={avg_lin_vel:.3f}, AngVel={avg_ang_vel:.3f}, Settling={avg_settling:.2f}s, Pitch={avg_pitch:.1f}deg, Roll={avg_roll:.1f}deg\n")

    def compute_proprioception_observations(self):
        """自定义本体观测：添加线速度和高度，不暴露 ready 标志"""
        obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.theta0 * self.obs_scales.dof_pos,
                self.theta0_dot * self.obs_scales.dof_vel,
                self.L0 * self.obs_scales.l0,
                self.L0_dot * self.obs_scales.l0_dot,
                self.dof_pos[:, [2, 5]] * self.obs_scales.dof_pos,
                self.dof_vel[:, [2, 5]] * self.obs_scales.dof_vel,
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_height.unsqueeze(1) * self.obs_scales.height_measurements,
                self.actions,
            ),
            dim=-1,
        )
        return obs_buf

    def compute_observations(self):
        """计算观测，保持与 reset 时一致的维度"""
        self.obs_buf = self.compute_proprioception_observations()

        # privileged obs 维度在配置里已更新
        if self.num_privileged_obs is not None:
            heights = (
                torch.clip(
                    self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights,
                    -1,
                    1.0,
                )
                * self.obs_scales.height_measurements
            )
            self.privileged_obs_buf = torch.cat(
                (
                    self.base_lin_vel * self.obs_scales.lin_vel,
                    self.obs_buf,
                    self.last_actions[:, :, 0],
                    self.last_actions[:, :, 1],
                    self.dof_acc * self.obs_scales.dof_acc,
                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                    self.dof_vel * self.obs_scales.dof_vel,
                    heights,
                    self.torques * self.obs_scales.torque,
                    (self.base_mass - self.base_mass.mean()).view(self.num_envs, 1),
                    self.base_com,
                    self.default_dof_pos - self.raw_default_dof_pos,
                ),
                dim=-1,
            )

        return self.obs_buf

    def _compute_torques(self, actions):
        """计算力矩，在稳定前强制腿伸长"""
        # 调用父类计算正常力矩
        torques = super()._compute_torques(actions)

        # 在稳定前，覆盖力矩：小腿负扭矩伸长，其他关节零扭矩
        if not self.ready_for_control.all():
            # 找到未稳定的环境
            not_ready = ~self.ready_for_control

            # 找到小腿关节索引（lf1, rf1）
            lf1_idx = self.dof_names.index("lf1_Joint") if "lf1_Joint" in self.dof_names else 1
            rf1_idx = self.dof_names.index("rf1_Joint") if "rf1_Joint" in self.dof_names else 4

            # 调试：记录扭矩
            if self.common_step_counter % 100 == 0 and not_ready.any():
                first_not_ready = torch.where(not_ready)[0][0]
                with open(self.debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"  Before: {torques[first_not_ready].cpu().numpy()}\n")

            # 所有关节零扭矩
            torques[not_ready, :] = 0.0

            # 小腿关节给大的负扭矩（模拟气弹簧撑开）
            torques[not_ready, lf1_idx] = -50.0
            torques[not_ready, rf1_idx] = -50.0

            if self.common_step_counter % 100 == 0 and not_ready.any():
                with open(self.debug_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"  After: {torques[first_not_ready].cpu().numpy()}\n")

        return torques

    def _reward_pitch_angle(self):
        """惩罚 pitch 角度偏差 - 使用指数衰减避免大角度时惩罚爆炸"""
        # exp(-|angle|): angle=0时reward=1, angle=1.57(90°)时reward=0.21
        # 配合负权重使用，角度越小奖励越高
        return torch.exp(-torch.abs(self.pitch_angle))

    def _reward_roll_angle(self):
        """惩罚 roll 角度偏差 - 使用指数衰减避免大角度时惩罚爆炸"""
        return torch.exp(-torch.abs(self.roll_angle))

    def _reward_pitch_vel(self):
        """惩罚 pitch 角速度"""
        return torch.square(self.pitch_vel)

    def _reward_roll_vel(self):
        """惩罚 roll 角速度"""
        return torch.square(self.roll_vel)

    def _reward_upward(self):
        """保持机体竖直：projected_gravity·z 接近 -1"""
        return torch.square(1.0 + self.projected_gravity[:, 2])

    def _reward_leg_angle_zero(self):
        """惩罚腿部摆角在世界坐标系下偏离垂直 - 使用指数衰减

        theta0 是本体坐标系下的腿部前后摆角，只和 pitch 有关
        世界坐标系下的腿部摆角 = 本体坐标系摆角 + pitch 角
        """
        # 世界坐标系下的腿部摆角（左右腿相同，都只加 pitch）
        theta_world = self.theta0 + self.pitch_angle.unsqueeze(1)

        # 使用指数衰减而不是平方，避免大角度时惩罚爆炸
        return torch.sum(torch.exp(-torch.abs(theta_world)), dim=1)

    def _reward_reach_flat_target(self):
        """奖励达到 Flat 初始条件（两阶段切换的目标）"""
        # 检查高度
        height_error = torch.abs(self.base_height - self.flat_target_height)
        height_ok = height_error < 0.05  # ±5cm

        # 检查姿态
        pitch_ok = torch.abs(self.pitch_angle) < self.flat_angle_threshold
        roll_ok = torch.abs(self.roll_angle) < self.flat_angle_threshold

        # 检查速度
        lin_vel_norm = torch.norm(self.base_lin_vel, dim=1)
        ang_vel_norm = torch.norm(self.base_ang_vel, dim=1)
        lin_vel_ok = lin_vel_norm < self.flat_vel_threshold
        ang_vel_ok = ang_vel_norm < self.flat_vel_threshold

        # 所有条件都满足时给予大奖励
        all_ok = height_ok & pitch_ok & roll_ok & lin_vel_ok & ang_vel_ok
        return all_ok.float()

    def _reward_upright_bonus(self):
        """直立奖励 - 基于实际角度"""
        angle_threshold = 0.1
        vel_threshold = 0.5

        angle_ok = (torch.abs(self.pitch_angle) < angle_threshold) & (
            torch.abs(self.roll_angle) < angle_threshold
        )
        vel_ok = (torch.abs(self.pitch_vel) < vel_threshold) & (
            torch.abs(self.roll_vel) < vel_threshold
        )
        lin_vel_ok = torch.norm(self.base_lin_vel, dim=1) < 0.5

        return (angle_ok & vel_ok & lin_vel_ok).float()

    def _reward_ang_vel_yaw(self):
        """惩罚 yaw 角速度"""
        return torch.square(self.base_ang_vel[:, 2])

    def _reward_base_lin_vel_xy(self):
        """惩罚 xy 方向线速度"""
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)

    def check_termination(self):
        """检查是否需要重置：只有超时才重启"""
        # 禁用所有失败条件
        self.fail_buf[:] = 0

        # 只有超时才重启（1 分钟 = 60 秒）
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        # 重置缓冲区
        self.reset_buf = self.time_out_buf.clone()
        self.reset_buf[torch.where(self.time_out_buf)[0]] = 1

    def _reset_dofs(self, env_ids):
        """重置 DOF，设置腿最长状态（断电）"""
        # 调用父类方法设置基本状态
        super()._reset_dofs(env_ids)

        # 设置腿最长状态（lf1/rf1 在 lower_limit）
        # 找到 lf1 和 rf1 的索引
        lf1_idx = self.dof_names.index("lf1_Joint") if "lf1_Joint" in self.dof_names else 1
        rf1_idx = self.dof_names.index("rf1_Joint") if "rf1_Joint" in self.dof_names else 4

        # 设置为 lower_limit（腿最长，气弹簧撑开）
        self.dof_pos[env_ids, lf1_idx] = self.dof_pos_limits[lf1_idx, 0]
        self.dof_pos[env_ids, rf1_idx] = self.dof_pos_limits[rf1_idx, 0]

        # 速度设为 0
        self.dof_vel[env_ids, lf1_idx] = 0.0
        self.dof_vel[env_ids, rf1_idx] = 0.0

        # 将修改后的关节状态写回仿真
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        """重置根状态，应用大范围随机姿态"""
        # 先调用父类设置基本位置
        super()._reset_root_states(env_ids)

        # 应用 balance_reset 配置的随机姿态
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

    def reset_idx(self, env_ids):
        """重置环境"""
        # 重置稳定性跟踪
        self.ready_for_control[env_ids] = False
        self.settling_time[env_ids] = 0.0

        # 调用父类 reset（会调用 _reset_dofs）
        super().reset_idx(env_ids)
