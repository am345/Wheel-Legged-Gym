"""
观测计算模块
计算27维观测空间，匹配IsaacGym的wheel_legged_vmc环境
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from .vmc_kinematics import VMCKinematics
from .control_config import BalanceVMCControlConfig, get_balance_vmc_control_config


class ObservationComputer:
    """计算观测，匹配IsaacGym格式"""

    def __init__(self, control_cfg: BalanceVMCControlConfig | None = None):
        """初始化观测计算器"""
        self.cfg = control_cfg if control_cfg is not None else get_balance_vmc_control_config()
        # 观测缩放系数（来自配置文件）
        self.obs_scales = {
            'ang_vel': self.cfg.obs_scales_ang_vel,
            'dof_pos': self.cfg.obs_scales_dof_pos,
            'dof_vel': self.cfg.obs_scales_dof_vel,
            'l0': self.cfg.obs_scales_l0,
            'l0_dot': self.cfg.obs_scales_l0_dot,
        }

        # 指令缩放系数
        self.commands_scale = self.cfg.commands_scale_np.copy()

        # VMC运动学
        self.vmc = VMCKinematics(l1=self.cfg.l1, l2=self.cfg.l2, offset=self.cfg.offset)

        # 观测裁剪
        self.clip_obs = self.cfg.clip_observations

    def compute(self, data, last_actions):
        """
        计算27维观测

        来自 wheel_legged_vmc.py:282-299

        Args:
            data: MuJoCo data对象
            last_actions: 上一步动作 (6,)

        Returns:
            obs: 27维观测向量
        """
        obs = []

        # 1. 基座角速度（机体坐标系）(3)
        base_quat = data.qpos[3:7]  # [w, x, y, z]
        base_ang_vel_world = data.qvel[3:6]
        base_ang_vel_body = self._rotate_inverse(base_quat, base_ang_vel_world)
        obs.extend(base_ang_vel_body * self.obs_scales['ang_vel'])

        # 2. 投影重力 (3)
        gravity_vec = np.array([0, 0, -1])
        projected_gravity = self._rotate_inverse(base_quat, gravity_vec)
        obs.extend(projected_gravity)

        # 3. 指令 (3): [lin_vel_x=0, ang_vel_yaw=0, height=0.24]
        commands = self.cfg.command_np
        obs.extend(commands * self.commands_scale)

        # 4-7. 虚拟腿运动学 (8)
        dof_pos = data.qpos[7:13]  # 6个关节位置
        dof_vel = data.qvel[6:12]  # 6个关节速度

        # 左腿
        theta1_l = dof_pos[0]  # lf0_Joint
        theta2_l = dof_pos[1]  # lf1_Joint
        theta1_dot_l = dof_vel[0]
        theta2_dot_l = dof_vel[1]
        L0_l, theta0_l = self.vmc.forward_kinematics(theta1_l, theta2_l)
        L0_dot_l, theta0_dot_l = self.vmc.compute_velocities(
            theta1_l, theta2_l, theta1_dot_l, theta2_dot_l
        )

        # 右腿
        theta1_r = dof_pos[3]  # rf0_Joint
        theta2_r = dof_pos[4]  # rf1_Joint
        theta1_dot_r = dof_vel[3]
        theta2_dot_r = dof_vel[4]
        L0_r, theta0_r = self.vmc.forward_kinematics(theta1_r, theta2_r)
        L0_dot_r, theta0_dot_r = self.vmc.compute_velocities(
            theta1_r, theta2_r, theta1_dot_r, theta2_dot_r
        )

        # theta0 (2)
        obs.extend([theta0_l * self.obs_scales['dof_pos'],
                   theta0_r * self.obs_scales['dof_pos']])

        # theta0_dot (2)
        obs.extend([theta0_dot_l * self.obs_scales['dof_vel'],
                   theta0_dot_r * self.obs_scales['dof_vel']])

        # L0 (2)
        obs.extend([L0_l * self.obs_scales['l0'],
                   L0_r * self.obs_scales['l0']])

        # L0_dot (2)
        obs.extend([L0_dot_l * self.obs_scales['l0_dot'],
                   L0_dot_r * self.obs_scales['l0_dot']])

        # 8. 轮子位置 (2)
        wheel_pos = dof_pos[[2, 5]]  # l_wheel, r_wheel
        obs.extend(wheel_pos * self.obs_scales['dof_pos'])

        # 9. 轮子速度 (2)
        wheel_vel = dof_vel[[2, 5]]
        obs.extend(wheel_vel * self.obs_scales['dof_vel'])

        # 10. 上一步动作 (6)
        obs.extend(last_actions)

        # 转换为numpy数组并裁剪
        obs = np.array(obs, dtype=np.float32)
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)

        return obs

    def compute_from_components(
        self,
        *,
        base_quat_wxyz,
        base_ang_vel_world,
        dof_pos,
        dof_vel,
        action_obs,
        commands=None,
        vmc_state: dict | None = None,
    ):
        """
        使用外部已对齐状态量拼接观测（用于 vmc_balance_exact 模式）。

        Args:
            base_quat_wxyz: [w,x,y,z]
            base_ang_vel_world: world frame ang vel (3,)
            dof_pos: (6,)
            dof_vel: (6,)
            action_obs: 当前控制步动作（训练环境 compute_observations 使用当前 self.actions）
            commands: optional (3,), default cfg.command
            vmc_state: optional dict with keys theta0/theta0_dot/L0/L0_dot (shape=(2,))
        """
        obs = []
        base_quat = np.asarray(base_quat_wxyz, dtype=np.float64)
        base_ang_vel_world = np.asarray(base_ang_vel_world, dtype=np.float64)
        dof_pos = np.asarray(dof_pos, dtype=np.float64)
        dof_vel = np.asarray(dof_vel, dtype=np.float64)
        action_obs = np.asarray(action_obs, dtype=np.float64)

        base_ang_vel_body = self._rotate_inverse(base_quat, base_ang_vel_world)
        obs.extend(base_ang_vel_body * self.obs_scales["ang_vel"])

        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        projected_gravity = self._rotate_inverse(base_quat, gravity_vec)
        obs.extend(projected_gravity)

        if commands is None:
            commands = self.cfg.command_np
        commands = np.asarray(commands, dtype=np.float64)
        obs.extend(commands * self.commands_scale)

        if vmc_state is None:
            theta1 = dof_pos[[0, 3]]
            theta2 = dof_pos[[1, 4]]
            theta1_dot = dof_vel[[0, 3]]
            theta2_dot = dof_vel[[1, 4]]
            L0, theta0 = self.vmc.forward_kinematics(theta1, theta2)
            L0_dot, theta0_dot = self.vmc.compute_velocities(theta1, theta2, theta1_dot, theta2_dot)
        else:
            theta0 = np.asarray(vmc_state["theta0"], dtype=np.float64)
            theta0_dot = np.asarray(vmc_state["theta0_dot"], dtype=np.float64)
            L0 = np.asarray(vmc_state["L0"], dtype=np.float64)
            L0_dot = np.asarray(vmc_state["L0_dot"], dtype=np.float64)

        obs.extend(theta0 * self.obs_scales["dof_pos"])
        obs.extend(theta0_dot * self.obs_scales["dof_vel"])
        obs.extend(L0 * self.obs_scales["l0"])
        obs.extend(L0_dot * self.obs_scales["l0_dot"])
        obs.extend(dof_pos[[2, 5]] * self.obs_scales["dof_pos"])
        obs.extend(dof_vel[[2, 5]] * self.obs_scales["dof_vel"])
        obs.extend(action_obs)

        obs = np.array(obs, dtype=np.float32)
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        return obs

    def _rotate_inverse(self, quat, vec):
        """
        用四元数的逆旋转向量

        Args:
            quat: 四元数 [w, x, y, z] (MuJoCo格式)
            vec: 向量 (3,)

        Returns:
            rotated_vec: 旋转后的向量 (3,)
        """
        # MuJoCo使用[w,x,y,z]，scipy使用[x,y,z,w]
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot = Rotation.from_quat(quat_scipy)
        return rot.inv().apply(vec)
