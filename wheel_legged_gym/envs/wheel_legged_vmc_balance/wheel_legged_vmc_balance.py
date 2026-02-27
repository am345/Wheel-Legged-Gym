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

    def post_physics_step(self):
        """计算pitch角度"""
        super().post_physics_step()

        # 计算pitch角度（从projected_gravity）
        self.pitch_angle = torch.atan2(
            self.projected_gravity[:, 1],
            -self.projected_gravity[:, 2]
        )

    def _compute_torques(self, actions):
        """VMC控制计算扭矩（已移除pitch角度提示）"""
        # 调用父类计算正常力矩
        torques = super()._compute_torques(actions)

        # 注释掉pitch角度提示，让机器人通过奖励函数学习平衡
        # # 检查pitch角度是否超过阈值（20度 = 0.349 rad）
        # pitch_threshold = 0.349  # 20度
        # large_pitch = torch.abs(self.pitch_angle) > pitch_threshold
        #
        # if large_pitch.any():
        #     # 找到小腿关节索引（lf1, rf1）
        #     lf1_idx = self.dof_names.index("lf1_Joint") if "lf1_Joint" in self.dof_names else 1
        #     rf1_idx = self.dof_names.index("rf1_Joint") if "rf1_Joint" in self.dof_names else 4
        #
        #     # 对pitch角度过大的环境，给小腿关节施加负扭矩（伸长腿）
        #     torques[large_pitch, lf1_idx] = -30.0
        #     torques[large_pitch, rf1_idx] = -30.0

        return torques

    def _reward_upward(self):
        """鼓励机体保持直立：projected_gravity·z 接近 -1 (robot_lab 标准)"""
        return torch.square(1.0 + self.projected_gravity[:, 2])

    def _reward_leg_extension_when_fallen(self):
        """倒地时鼓励腿部伸展"""
        # 检测倒地状态：|projected_gravity.z| < 0.7 表示倾斜超过45度
        is_fallen = torch.abs(self.projected_gravity[:, 2]) < 0.7

        # 目标腿长（伸展位置以产生扭矩）
        target_L0 = 0.30  # 接近最大伸展

        # 奖励腿部伸展到目标长度
        leg_extension_error = torch.mean(torch.square(self.L0 - target_L0), dim=1)
        reward = torch.exp(-leg_extension_error / 0.01)

        # 仅在倒地时应用
        return reward * is_fallen.float()

    def _reward_recovery_angular_velocity(self):
        """奖励朝向直立方向的角速度"""
        # 根据projected_gravity确定需要的旋转方向
        target_ang_vel = torch.zeros_like(self.base_ang_vel)

        # 如果前倾（gravity.y > 0），需要负的pitch角速度
        target_ang_vel[:, 1] = -self.projected_gravity[:, 1] * 3.0
        # 如果侧倾（gravity.x != 0），需要修正的roll角速度
        target_ang_vel[:, 0] = -self.projected_gravity[:, 0] * 3.0

        # 奖励朝恢复方向的角速度
        ang_vel_error = torch.sum(torch.square(
            self.base_ang_vel[:, :2] - target_ang_vel[:, :2]
        ), dim=1)

        # 仅在明显倾斜时应用
        is_tilted = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1) > 0.1

        return torch.exp(-ang_vel_error / 2.0) * is_tilted.float()

    def _reward_uprightness_shaped(self):
        """平滑的直立度奖励，在所有姿态下都有梯度"""
        # projected_gravity.z 范围从 -1（直立）到 +1（倒立）
        # 映射到 0（倒立）到 1（直立）
        uprightness = (-self.projected_gravity[:, 2] + 1.0) / 2.0

        # 使用指数塑形在所有位置提供平滑梯度
        return torch.exp(uprightness * 2.0) - 1.0  # 范围：0 到 ~6.4

    def _reward_orientation_refined(self):
        """仅在接近直立时强力惩罚姿态偏差（精细调整）"""
        orientation_error = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

        # 仅在基本直立时应用强惩罚
        is_nearly_upright = self.projected_gravity[:, 2] < -0.7

        return orientation_error * is_nearly_upright.float()

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
