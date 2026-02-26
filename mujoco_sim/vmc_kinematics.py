"""
VMC (Virtual Model Control) 运动学模块

实现：
- 虚拟腿正运动学（L0, theta0）
- 有限差分速度（L0_dot, theta0_dot）
- 与训练端一致的 VMC Jacobian 映射（F/T -> 关节力矩）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class VMCState:
    L0: np.ndarray
    theta0: np.ndarray
    L0_dot: np.ndarray
    theta0_dot: np.ndarray


class VMCKinematics:
    """虚拟腿运动学与VMC力矩映射（numpy版）"""

    def __init__(self, l1: float = 0.167, l2: float = 0.200, offset: float = 0.0):
        self.l1 = float(l1)
        self.l2 = float(l2)
        self.offset = float(offset)

    @staticmethod
    def _as_float_array(x) -> np.ndarray:
        return np.asarray(x, dtype=np.float64)

    def forward_kinematics(self, theta1, theta2):
        """
        正运动学：从关节角度计算虚拟腿参数（与 wheel_legged_vmc.py 一致）

        Returns:
            L0, theta0
        """
        theta1 = self._as_float_array(theta1)
        theta2 = self._as_float_array(theta2)

        end_x = self.l1 * np.cos(theta1) - self.l2 * np.sin(theta1 + theta2)
        end_y = self.l1 * np.sin(theta1) + self.l2 * np.cos(theta1 + theta2)

        L0 = np.sqrt(end_x**2 + end_y**2)
        theta0 = np.arctan2(end_x, end_y)
        return L0, theta0

    def compute_velocities(self, theta1, theta2, theta1_dot, theta2_dot, dt: float = 0.001):
        """
        有限差分速度（与训练端 leg_post_physics_step 的实现方式一致）
        """
        theta1 = self._as_float_array(theta1)
        theta2 = self._as_float_array(theta2)
        theta1_dot = self._as_float_array(theta1_dot)
        theta2_dot = self._as_float_array(theta2_dot)
        dt = float(dt)

        L0, theta0 = self.forward_kinematics(theta1, theta2)
        L0_next, theta0_next = self.forward_kinematics(
            theta1 + theta1_dot * dt, theta2 + theta2_dot * dt
        )
        L0_dot = (L0_next - L0) / dt
        theta0_dot = (theta0_next - theta0) / dt
        return L0_dot, theta0_dot

    def compute_state(
        self, theta1, theta2, theta1_dot, theta2_dot, velocity_fd_dt: float = 0.001
    ) -> VMCState:
        L0, theta0 = self.forward_kinematics(theta1, theta2)
        L0_dot, theta0_dot = self.compute_velocities(
            theta1, theta2, theta1_dot, theta2_dot, dt=velocity_fd_dt
        )
        return VMCState(L0=L0, theta0=theta0, L0_dot=L0_dot, theta0_dot=theta0_dot)

    def vmc_jacobian(self, theta1, theta2, L0=None) -> Dict[str, np.ndarray]:
        """
        计算与训练端 VMC() 一致的 Jacobian 项。

        Returns:
            dict with J11, J12, J21, J22 (same semantics as training code)
        """
        theta1 = self._as_float_array(theta1)
        theta2 = self._as_float_array(theta2)
        if L0 is None:
            L0, _ = self.forward_kinematics(theta1, theta2)
        else:
            L0 = self._as_float_array(L0)

        eps = 1e-9
        L0_safe = np.where(np.abs(L0) < eps, eps, L0)

        dx_dtheta1 = -self.l1 * np.sin(theta1) - self.l2 * np.cos(theta1 + theta2)
        dy_dtheta1 = self.l1 * np.cos(theta1) - self.l2 * np.sin(theta1 + theta2)
        dx_dtheta2 = -self.l2 * np.cos(theta1 + theta2)
        dy_dtheta2 = -self.l2 * np.sin(theta1 + theta2)

        dL0_dx = L0_safe**-1 * (
            self.l1 * np.cos(theta1) - self.l2 * np.sin(theta1 + theta2)
        )
        dL0_dy = L0_safe**-1 * (
            self.l1 * np.sin(theta1) + self.l2 * np.cos(theta1 + theta2)
        )
        dphi_dx = L0_safe**-2 * (
            self.l1 * np.sin(theta1) + self.l2 * np.cos(theta1 + theta2)
        )
        dphi_dy = -L0_safe**-2 * (
            self.l1 * np.cos(theta1) - self.l2 * np.sin(theta1 + theta2)
        )

        J11 = dL0_dx * dx_dtheta1 + dL0_dy * dy_dtheta1
        J12 = dL0_dx * dx_dtheta2 + dL0_dy * dy_dtheta2
        J21 = dphi_dx * dx_dtheta1 + dphi_dy * dy_dtheta1
        J22 = dphi_dx * dx_dtheta2 + dphi_dy * dy_dtheta2

        return {"J11": J11, "J12": J12, "J21": J21, "J22": J22}

    def map_virtual_to_joint_torques(self, F, T, theta1, theta2, L0=None):
        """
        VMC 映射：虚拟力/虚拟腿力矩 -> 两个关节力矩

        Args:
            F: 虚拟腿轴向力
            T: 虚拟腿角度方向力矩
            theta1/theta2: 两关节角度
            L0: 可选，若已计算可直接传入

        Returns:
            (T1, T2)
        """
        F = self._as_float_array(F)
        T = self._as_float_array(T)
        jac = self.vmc_jacobian(theta1, theta2, L0=L0)

        T1 = F * jac["J11"] + T * jac["J21"]
        T2 = F * jac["J12"] + T * jac["J22"]
        return T1, T2

    def batch_leg_state_from_dofs(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        velocity_fd_dt: float = 0.001,
    ) -> Tuple[np.ndarray, np.ndarray, VMCState]:
        """
        从 6DoF 向量提取左右腿状态（与 wheel_legged_vmc 环境的索引规则一致）

        Returns:
            theta1(2), theta2(2), VMCState(每项 shape=(2,))
        """
        dof_pos = self._as_float_array(dof_pos)
        dof_vel = self._as_float_array(dof_vel)

        theta1 = np.array([dof_pos[0], dof_pos[3]], dtype=np.float64)
        theta2 = np.array([dof_pos[1], dof_pos[4]], dtype=np.float64)
        theta1_dot = np.array([dof_vel[0], dof_vel[3]], dtype=np.float64)
        theta2_dot = np.array([dof_vel[1], dof_vel[4]], dtype=np.float64)

        state = self.compute_state(
            theta1=theta1,
            theta2=theta2,
            theta1_dot=theta1_dot,
            theta2_dot=theta2_dot,
            velocity_fd_dt=velocity_fd_dt,
        )
        return theta1, theta2, state
