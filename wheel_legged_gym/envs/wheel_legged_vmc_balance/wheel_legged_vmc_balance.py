# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch

from wheel_legged_gym.envs.wheel_legged_vmc.wheel_legged_vmc import LeggedRobotVMC
from .wheel_legged_vmc_balance_config import WheelLeggedVMCBalanceCfg


class LeggedRobotVMCBalance(LeggedRobotVMC):
    """
    最小化的平衡任务：
    - 从直立姿态开始
    - 只需要保持平衡
    - 移除所有复杂逻辑
    """

    def __init__(
        self, cfg: WheelLeggedVMCBalanceCfg, sim_params, physics_engine, sim_device, headless
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    # 移除所有自定义的 reset、step、compute_torques 等方法
    # 直接使用父类 LeggedRobotVMC 的标准实现

    # 只保留必要的奖励函数
    def _reward_upright_bonus(self):
        """奖励保持直立"""
        grav_ok = (torch.abs(self.projected_gravity[:, 0]) < 0.05) & (
            torch.abs(self.projected_gravity[:, 1]) < 0.05
        )
        lin_ok = torch.norm(self.base_lin_vel, dim=1) < 0.3
        ang_ok = torch.norm(self.base_ang_vel, dim=1) < 0.3
        return (grav_ok & lin_ok & ang_ok).float()
