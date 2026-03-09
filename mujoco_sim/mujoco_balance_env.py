"""
MuJoCo balance/fzqver environment for sim2sim verification.

Supported controller modes:
- simplified_joint_pd
- vmc_balance_exact

Supported tasks:
- wheel_legged_vmc_balance
- wheel_legged_fzqver
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from .control_config import (
    BalanceResetRanges,
    BalanceVMCControlConfig,
    FzqverSim2SimProfile,
    get_balance_reset_profile,
    get_balance_vmc_control_config,
    get_mujoco_demo_tuning_profile,
)
from .domain_randomizer import MuJoCoDomainRandomizer
from .observation_computer import ObservationComputer
from .vmc_kinematics import VMCKinematics


class MuJoCoBalanceEnv:
    """MuJoCo environment used by sim2sim verification scripts."""

    SUPPORTED_CONTROLLER_MODES = ("simplified_joint_pd", "vmc_balance_exact")
    SUPPORTED_DOMAIN_RAND_MODES = ("off", "train_ranges")
    SUPPORTED_TASKS = (
        "wheel_legged_vmc_balance",
        "wheel_legged_fzqver",
    )

    def __init__(
        self,
        model_path: str,
        render: bool = False,
        seed: Optional[int] = None,
        controller_mode: str = "simplified_joint_pd",
        domain_rand_mode: str = "off",
        control_cfg: Optional[BalanceVMCControlConfig] = None,
        fidelity_level: Optional[str] = None,
        mujoco_tuning_profile: str = "exact_baseline",
        task: str = "wheel_legged_vmc_balance",
    ):
        if controller_mode not in self.SUPPORTED_CONTROLLER_MODES:
            raise ValueError(
                f"Unsupported controller_mode={controller_mode}. "
                f"Expected one of {self.SUPPORTED_CONTROLLER_MODES}."
            )
        if domain_rand_mode not in self.SUPPORTED_DOMAIN_RAND_MODES:
            raise ValueError(
                f"Unsupported domain_rand_mode={domain_rand_mode}. "
                f"Expected one of {self.SUPPORTED_DOMAIN_RAND_MODES}."
            )
        if task not in self.SUPPORTED_TASKS:
            raise ValueError(f"Unsupported task={task}. Expected one of {self.SUPPORTED_TASKS}.")

        self.model_path = model_path
        self.seed = seed
        self.np_random = np.random.default_rng(seed)
        self.task = task
        self.action_dim = 6
        self.controller_mode = controller_mode
        self.implemented_controller_mode = controller_mode
        self.domain_rand_mode = domain_rand_mode
        self.render_enabled = bool(render)
        self.viewer = None
        self.wait_mode = "n/a"
        self.fidelity_level = (
            fidelity_level
            if fidelity_level is not None
            else ("fidelity" if "fidelity" in str(model_path).lower() else "simple")
        )

        self.cfg = control_cfg if control_cfg is not None else get_balance_vmc_control_config()
        self.fzqver_profile: FzqverSim2SimProfile = self.cfg.fzqver_profile
        if self.task == "wheel_legged_fzqver":
            self.cfg.enable_gas_spring = bool(self.fzqver_profile.enable_gas_spring)
            self.cfg.gas_spring_k = float(self.fzqver_profile.gas_spring_k)
            self.cfg.gas_spring_b = float(self.fzqver_profile.gas_spring_b)

        self.mujoco_tuning_profile = get_mujoco_demo_tuning_profile(mujoco_tuning_profile)
        self.mujoco_tuning_profile_name = self.mujoco_tuning_profile.name

        self.vmc = VMCKinematics(l1=self.cfg.l1, l2=self.cfg.l2, offset=self.cfg.offset)
        self.obs_computer = ObservationComputer(control_cfg=self.cfg)

        print(f"Loading MuJoCo model from: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.sim_dt = float(self.model.opt.timestep)
        self.decimation = int(self.cfg.control_decimation)
        self.dt = float(self.cfg.control_dt)
        if not np.isclose(self.sim_dt, self.cfg.sim_dt):
            print(
                f"[WARN] MuJoCo model timestep={self.sim_dt} differs from cfg.sim_dt={self.cfg.sim_dt}. "
                "sim2sim fidelity may degrade."
            )

        # Joint/actuator mapping must match training order.
        self.joint_names = [
            "lf0_Joint",
            "lf1_Joint",
            "l_wheel_Joint",
            "rf0_Joint",
            "rf1_Joint",
            "r_wheel_Joint",
        ]
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        if any(jid < 0 for jid in self.joint_ids):
            missing = [name for name, jid in zip(self.joint_names, self.joint_ids) if jid < 0]
            raise ValueError(f"Missing expected joints in MJCF model: {missing}")
        self.joint_qpos_addrs = [int(self.model.jnt_qposadr[jid]) for jid in self.joint_ids]
        self.joint_dof_addrs = [int(self.model.jnt_dofadr[jid]) for jid in self.joint_ids]
        if hasattr(self.model, "jnt_limited"):
            self.joint_limited = np.array(
                [bool(int(self.model.jnt_limited[jid])) for jid in self.joint_ids], dtype=bool
            )
        else:
            self.joint_limited = np.array(
                [
                    bool(np.isfinite(self.model.jnt_range[jid]).all())
                    and float(self.model.jnt_range[jid, 1] - self.model.jnt_range[jid, 0]) > 0
                    for jid in self.joint_ids
                ],
                dtype=bool,
            )
        self.joint_range = np.array(
            [self.model.jnt_range[jid].copy() for jid in self.joint_ids], dtype=np.float64
        )

        self.actuator_names = [
            "lf0_act",
            "lf1_act",
            "l_wheel_act",
            "rf0_act",
            "rf1_act",
            "r_wheel_act",
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.actuator_names
        ]
        if any(aid < 0 for aid in self.actuator_ids):
            missing = [name for name, aid in zip(self.actuator_names, self.actuator_ids) if aid < 0]
            raise ValueError(f"Missing expected actuators in MJCF model: {missing}")

        # Bodies/geoms for contacts + domain rand.
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.l_wheel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_Link")
        self.r_wheel_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_Link")
        self.floor_geom_ids = []
        self.robot_geom_ids = []
        self.wheel_geom_ids = []
        self.geom_names = []
        for gid in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
            self.geom_names.append(name)
            body_id = int(self.model.geom_bodyid[gid])
            if name == "floor":
                self.floor_geom_ids.append(gid)
            else:
                self.robot_geom_ids.append(gid)
            if "wheel" in name.lower() or body_id in (self.l_wheel_body_id, self.r_wheel_body_id):
                self.wheel_geom_ids.append(gid)

        # Base control/randomization state.
        self.base_default_dof_pos = self.cfg.default_dof_pos_np.copy()
        self.base_torque_limits = self.cfg.torque_limits_np.copy()
        self.base_torques_scale = np.ones(6, dtype=np.float64)
        self.base_simple_p_gains = self.cfg.simple_p_gains_np.copy()
        self.base_simple_d_gains = self.cfg.simple_d_gains_np.copy()
        self.base_theta_kp = np.full(2, self.cfg.kp_theta, dtype=np.float64)
        self.base_theta_kd = np.full(2, self.cfg.kd_theta, dtype=np.float64)
        self.base_l0_kp = np.full(2, self.cfg.kp_l0, dtype=np.float64)
        self.base_l0_kd = np.full(2, self.cfg.kd_l0, dtype=np.float64)
        self.base_wheel_kd = np.full(2, self.cfg.wheel_kd, dtype=np.float64)
        self.base_feedforward_force = float(self.cfg.feedforward_force)
        self.base_balance_hint_pitch_threshold_rad = float(self.cfg.balance_hint_pitch_threshold_rad)
        self.base_balance_hint_knee_torque = float(self.cfg.balance_hint_knee_torque)
        self.base_enable_balance_hint = bool(self.cfg.enable_balance_hint)
        # fzqver should not use balance hint by default.
        if self.task == "wheel_legged_fzqver":
            self.base_enable_balance_hint = False

        self.current_default_dof_pos = self.base_default_dof_pos.copy()
        self.current_torque_limits = self.base_torque_limits.copy()
        self.current_torques_scale = self.base_torques_scale.copy()
        self.current_simple_p_gains = self.base_simple_p_gains.copy()
        self.current_simple_d_gains = self.base_simple_d_gains.copy()
        self.current_theta_kp = self.base_theta_kp.copy()
        self.current_theta_kd = self.base_theta_kd.copy()
        self.current_l0_kp = self.base_l0_kp.copy()
        self.current_l0_kd = self.base_l0_kd.copy()
        self.current_wheel_kd = self.base_wheel_kd.copy()
        self.current_feedforward_force = float(self.base_feedforward_force)
        self.current_balance_hint_pitch_threshold_rad = float(self.base_balance_hint_pitch_threshold_rad)
        self.current_balance_hint_knee_torque = float(self.base_balance_hint_knee_torque)
        self.current_enable_balance_hint = bool(self.base_enable_balance_hint)

        # Domain randomization baseline caches.
        self.baseline_model_params = {
            "body_mass": self.model.body_mass.copy(),
            "body_inertia": self.model.body_inertia.copy(),
            "body_ipos": self.model.body_ipos.copy(),
            "geom_friction": self.model.geom_friction.copy(),
            "geom_solref": self.model.geom_solref.copy(),
            "geom_solimp": self.model.geom_solimp.copy(),
        }
        floor_ref_gid = self.floor_geom_ids[0] if self.floor_geom_ids else 0
        base_fric_scalar = float(self.model.geom_friction[floor_ref_gid, 0])
        base_dampratio = float(self.model.geom_solref[floor_ref_gid, 1])
        base_restitution_proxy = float(np.clip((1.5 - base_dampratio) / 1.4, 0.0, 1.0))
        self.baseline_contact_defaults = {
            "friction_scalar": base_fric_scalar,
            "restitution_scalar": base_restitution_proxy,
        }
        self.domain_randomizer = MuJoCoDomainRandomizer(mode=domain_rand_mode)
        self.current_domain_params: Dict[str, Any] = {"mode": "baseline", "randomized": False}

        # Action delay FIFO (max 10ms in training ranges).
        delay_ms_max = 10.0
        if self.domain_randomizer.cfg.randomize_action_delay:
            delay_ms_max = float(self.domain_randomizer.cfg.delay_ms_range[1])
        self.action_fifo_len = int(np.ceil(delay_ms_max / 1000.0 / self.sim_dt)) + 1
        self.action_delay_max_idx = self.action_fifo_len - 1
        self.action_delay_idx = 0
        self.action_fifo = np.zeros((self.action_fifo_len, self.action_dim), dtype=np.float64)

        # Command state.
        self.current_commands = self.cfg.command_np.copy()
        self.command_resample_interval_steps = max(
            1, int(round(float(self.fzqver_profile.resampling_time_s) / self.dt))
        )
        self.last_command_resample_step = -1

        # Episode/state buffers.
        self.max_episode_steps = 3000
        self.episode_steps = 0
        self.current_action_obs = np.zeros(self.action_dim, dtype=np.float64)
        self.last_applied_action = np.zeros(self.action_dim, dtype=np.float64)
        self.last_ctrl = np.zeros(6, dtype=np.float64)
        self.last_reward = 0.0
        self.last_done = False
        self.last_info: Dict[str, Any] = {}
        self.last_termination_reason = "running"

        self.dof_pos = np.zeros(6, dtype=np.float64)
        self.dof_vel = np.zeros(6, dtype=np.float64)
        self.dof_vel_raw = np.zeros(6, dtype=np.float64)
        self.dof_pos_dot = np.zeros(6, dtype=np.float64)
        self.prev_dof_pos_for_diff = np.zeros(6, dtype=np.float64)
        self.base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.base_ang_vel_world = np.zeros(3, dtype=np.float64)
        self.base_ang_vel_body = np.zeros(3, dtype=np.float64)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        self.pitch_angle = 0.0

        self.theta1 = np.zeros(2, dtype=np.float64)
        self.theta2 = np.zeros(2, dtype=np.float64)
        self.L0 = np.zeros(2, dtype=np.float64)
        self.theta0 = np.zeros(2, dtype=np.float64)
        self.L0_dot = np.zeros(2, dtype=np.float64)
        self.theta0_dot = np.zeros(2, dtype=np.float64)
        self.gas_spring_force = np.zeros(2, dtype=np.float64)
        self.virtual_leg_force_total = np.zeros(2, dtype=np.float64)

        self.last_control_debug: Dict[str, Any] = {}
        self.prompt_torque_triggered_last_step = False
        self.prompt_torque_step_count = 0
        self.torque_saturation_count = 0
        self.action_clip_count = 0

        self.current_reset_profile = "hard_random_balance"

        mujoco.mj_resetData(self.model, self.data)
        self._post_reset_apply_default_state()
        self._resample_commands(force=True)
        self._refresh_state_from_sim(init_from_current=True)
        self._apply_mujoco_tuning_profile()

        print("Environment initialized successfully")
        print(f"  task: {self.task}")
        print(f"  DOFs (qvel): {self.model.nv}")
        print(f"  Control dt: {self.dt}s ({1/self.dt:.0f}Hz)")
        print(f"  Physics dt: {self.sim_dt}s ({1/self.sim_dt:.0f}Hz)")
        print(f"  Decimation: {self.decimation}")
        print(f"  Seed: {self.seed}")
        print(f"  Controller mode: {self.implemented_controller_mode}")
        print(f"  Domain rand mode: {self.domain_rand_mode}")
        print(f"  Fidelity level: {self.fidelity_level}")
        print(f"  MuJoCo tuning profile: {self.mujoco_tuning_profile_name}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(
        self,
        randomize: bool = True,
        domain_randomize: Optional[bool] = None,
        reset_profile: Optional[str] = None,
    ):
        """Reset environment and return observation (27,)."""
        mujoco.mj_resetData(self.model, self.data)

        if domain_randomize is None:
            domain_randomize = self.domain_rand_mode != "off"

        self.domain_randomizer.reset_env_to_baseline(self)
        if domain_randomize and self.domain_randomizer.is_enabled():
            params = self.domain_randomizer.sample(self.np_random, self)
            self.domain_randomizer.apply_to_env(self, params)
        else:
            self.current_domain_params = {
                "mode": "off",
                "randomized": False,
                "restitution_application": "none",
            }

        self._apply_mujoco_tuning_profile()

        self._post_reset_apply_default_state()
        if randomize:
            if self.task == "wheel_legged_fzqver":
                self.current_reset_profile = "fzqver_mixed_random"
                self._apply_fzqver_root_randomization()
            else:
                self.current_reset_profile, ranges = self._resolve_reset_profile(reset_profile)
                self._apply_balance_root_randomization(ranges)
        else:
            self.current_reset_profile = "fixed"

        mujoco.mj_forward(self.model, self.data)

        self.episode_steps = 0
        self.current_action_obs[:] = 0.0
        self.last_applied_action[:] = 0.0
        self.last_ctrl[:] = 0.0
        self.action_fifo[:] = 0.0
        self.last_reward = 0.0
        self.last_done = False
        self.last_info = {}
        self.last_termination_reason = "running"

        self.prompt_torque_triggered_last_step = False
        self.prompt_torque_step_count = 0
        self.torque_saturation_count = 0
        self.action_clip_count = 0

        self._resample_commands(force=True)
        self._refresh_state_from_sim(init_from_current=True)
        obs = self._compute_observation(self.current_action_obs)
        return obs

    def step(self, action):
        """Run one control step (100Hz) with decimated MuJoCo substeps."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"Action shape mismatch: expected ({self.action_dim},), got {action.shape}"
            )

        clipped_action = np.clip(action, -self.cfg.clip_actions, self.cfg.clip_actions)
        if not np.allclose(clipped_action, action):
            self.action_clip_count += 1

        self.current_action_obs[:] = clipped_action
        self.prompt_torque_triggered_last_step = False

        if self.implemented_controller_mode == "simplified_joint_pd":
            ctrl = self._compute_torques_simplified(clipped_action)
            self.last_applied_action[:] = clipped_action
            for _ in range(self.decimation):
                self.data.ctrl[:] = ctrl
                mujoco.mj_step(self.model, self.data)
                if self.render_enabled and self.viewer is not None:
                    self.viewer.sync()
            self.last_ctrl[:] = ctrl
            self._refresh_state_from_sim(init_from_current=False)
        else:
            for _ in range(self.decimation):
                self._refresh_state_from_sim(init_from_current=False)
                self._push_action_fifo(clipped_action)
                delayed_action = self.action_fifo[self.action_delay_idx]
                ctrl = self._compute_torques_vmc_balance_exact(delayed_action)
                self.last_applied_action[:] = delayed_action
                self.last_ctrl[:] = ctrl
                self.data.ctrl[:] = ctrl
                mujoco.mj_step(self.model, self.data)
                if self.render_enabled and self.viewer is not None:
                    self.viewer.sync()
            self._refresh_state_from_sim(init_from_current=False)

        self.episode_steps += 1
        self._maybe_resample_commands()

        obs = self._compute_observation(self.current_action_obs)
        reward = self._compute_reward()

        done = self.episode_steps >= self.max_episode_steps
        self.last_termination_reason = "timeout" if done else "running"

        info = {
            "episode_steps": int(self.episode_steps),
            "base_height": float(self.data.qpos[2]),
            "controller_mode": self.implemented_controller_mode,
            "domain_params": self.current_domain_params,
            "reset_profile": self.current_reset_profile,
            "reset_mode": self.current_reset_profile,
            "mujoco_tuning_profile": self.mujoco_tuning_profile_name,
            "termination_reason": self.last_termination_reason,
            "prompt_torque_triggered": bool(self.prompt_torque_triggered_last_step),
            "pitch_angle": float(self.pitch_angle),
            "task": self.task,
            "current_commands": self.current_commands.copy().tolist(),
            "command_resample_step": int(self.last_command_resample_step),
        }
        self.last_reward = float(reward)
        self.last_done = bool(done)
        self.last_info = info

        return obs, float(reward), bool(done), info

    def render(self):
        if self.render_enabled and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ------------------------------------------------------------------
    # Replay/diagnostics API
    # ------------------------------------------------------------------
    def set_state(
        self,
        *,
        root_pos,
        root_quat_xyzw=None,
        root_quat_wxyz=None,
        root_lin_vel=None,
        root_ang_vel=None,
        dof_pos=None,
        dof_vel=None,
        reset_buffers: bool = True,
    ) -> None:
        if (root_quat_xyzw is None) == (root_quat_wxyz is None):
            raise ValueError("Provide exactly one of root_quat_xyzw or root_quat_wxyz")

        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_lin_vel = (
            np.zeros(3, dtype=np.float64)
            if root_lin_vel is None
            else np.asarray(root_lin_vel, dtype=np.float64)
        )
        root_ang_vel = (
            np.zeros(3, dtype=np.float64)
            if root_ang_vel is None
            else np.asarray(root_ang_vel, dtype=np.float64)
        )
        dof_pos = (
            self.current_default_dof_pos
            if dof_pos is None
            else np.asarray(dof_pos, dtype=np.float64)
        )
        dof_vel = (
            np.zeros(6, dtype=np.float64)
            if dof_vel is None
            else np.asarray(dof_vel, dtype=np.float64)
        )

        if root_quat_wxyz is None:
            qxyzw = np.asarray(root_quat_xyzw, dtype=np.float64)
            root_quat_wxyz = np.array([qxyzw[3], qxyzw[0], qxyzw[1], qxyzw[2]], dtype=np.float64)
        else:
            root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64)

        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat_wxyz
        self.data.qvel[0:3] = root_lin_vel
        self.data.qvel[3:6] = root_ang_vel
        self.data.qpos[self.joint_qpos_addrs] = dof_pos
        self.data.qvel[self.joint_dof_addrs] = dof_vel

        mujoco.mj_forward(self.model, self.data)
        self._refresh_state_from_sim(init_from_current=True)

        if reset_buffers:
            self.current_action_obs[:] = 0.0
            self.last_applied_action[:] = 0.0
            self.last_ctrl[:] = 0.0
            self.action_fifo[:] = 0.0

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "root_pos": self.data.qpos[0:3].copy(),
            "root_quat_wxyz": self.data.qpos[3:7].copy(),
            "root_quat_xyzw": np.array(
                [self.data.qpos[4], self.data.qpos[5], self.data.qpos[6], self.data.qpos[3]],
                dtype=np.float64,
            ),
            "root_lin_vel": self.data.qvel[0:3].copy(),
            "root_ang_vel": self.data.qvel[3:6].copy(),
            "dof_pos": self.dof_pos.copy(),
            "dof_vel": self.dof_vel.copy(),
            "dof_vel_raw": self.dof_vel_raw.copy(),
            "projected_gravity": self.projected_gravity.copy(),
            "L0": self.L0.copy(),
            "theta0": self.theta0.copy(),
            "L0_dot": self.L0_dot.copy(),
            "theta0_dot": self.theta0_dot.copy(),
            "current_commands": self.current_commands.copy(),
        }

    def get_contact_flags(self) -> Dict[str, bool]:
        base_contact = False
        left_wheel_contact = False
        right_wheel_contact = False
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            b1 = int(self.model.geom_bodyid[c.geom1])
            b2 = int(self.model.geom_bodyid[c.geom2])
            if self.base_body_id in (b1, b2):
                base_contact = True
            if self.l_wheel_body_id in (b1, b2):
                left_wheel_contact = True
            if self.r_wheel_body_id in (b1, b2):
                right_wheel_contact = True
        return {
            "base_contact": bool(base_contact),
            "left_wheel_contact": bool(left_wheel_contact),
            "right_wheel_contact": bool(right_wheel_contact),
        }

    def _joint_limit_diagnostics(self, eps: float = 5e-3) -> Dict[str, Any]:
        margins = np.full(6, np.inf, dtype=np.float64)
        flags = np.zeros(6, dtype=bool)
        lower = self.joint_range[:, 0]
        upper = self.joint_range[:, 1]
        if np.any(self.joint_limited):
            pos = self.dof_pos
            dist_lower = pos - lower
            dist_upper = upper - pos
            margins[self.joint_limited] = np.minimum(dist_lower, dist_upper)[self.joint_limited]
            flags[self.joint_limited] = margins[self.joint_limited] <= float(eps)
        return {
            "joint_limit_hit_flags": flags,
            "joint_limit_margin": margins,
            "joint_limit_eps": float(eps),
        }

    def _torque_saturation_flags_from_last_control(self) -> np.ndarray:
        ctrl_dbg = self.last_control_debug or {}
        pre = ctrl_dbg.get("torques_pre_clip")
        if pre is None:
            return np.zeros(6, dtype=bool)
        try:
            arr = np.asarray(pre, dtype=np.float64).reshape(6)
        except Exception:
            return np.zeros(6, dtype=bool)
        return np.abs(arr) > (self.current_torque_limits + 1e-9)

    def get_debug_state(self) -> Dict[str, Any]:
        joint_limit_diag = self._joint_limit_diagnostics()
        torque_sat_flags = self._torque_saturation_flags_from_last_control()
        tilt_deg = float(
            np.degrees(np.arccos(np.clip(-float(self.projected_gravity[2]), -1.0, 1.0)))
        )
        return {
            "task": self.task,
            "controller_mode": self.implemented_controller_mode,
            "fidelity_level": self.fidelity_level,
            "mujoco_tuning_profile": self.mujoco_tuning_profile_name,
            "wait_mode": self.wait_mode,
            "episode_steps": int(self.episode_steps),
            "current_action_obs": self.current_action_obs.copy().tolist(),
            "last_applied_action": self.last_applied_action.copy().tolist(),
            "last_ctrl": self.last_ctrl.copy().tolist(),
            "pitch_angle": float(self.pitch_angle),
            "tilt_deg": tilt_deg,
            "projected_gravity": self.projected_gravity.copy().tolist(),
            "theta0": self.theta0.copy().tolist(),
            "theta0_dot": self.theta0_dot.copy().tolist(),
            "L0": self.L0.copy().tolist(),
            "L0_dot": self.L0_dot.copy().tolist(),
            "dof_pos": self.dof_pos.copy().tolist(),
            "dof_vel": self.dof_vel.copy().tolist(),
            "dof_vel_raw": self.dof_vel_raw.copy().tolist(),
            "current_commands": self.current_commands.copy().tolist(),
            "command_resample_step": int(self.last_command_resample_step),
            "prompt_torque_triggered_last_step": bool(self.prompt_torque_triggered_last_step),
            "balance_hint_active": bool(self.prompt_torque_triggered_last_step),
            "prompt_torque_step_count": int(self.prompt_torque_step_count),
            "torque_saturation_count": int(self.torque_saturation_count),
            "torque_saturation_flags": torque_sat_flags.tolist(),
            "action_clip_count": int(self.action_clip_count),
            "action_delay_idx": int(self.action_delay_idx),
            "reset_profile": self.current_reset_profile,
            "reset_mode": self.current_reset_profile,
            "current_domain_params": self.current_domain_params,
            "joint_limit_hit_flags": joint_limit_diag["joint_limit_hit_flags"].tolist(),
            "joint_limit_margin": joint_limit_diag["joint_limit_margin"].tolist(),
            "joint_limit_eps": joint_limit_diag["joint_limit_eps"],
            "joint_limited": self.joint_limited.tolist(),
            "joint_names": list(self.joint_names),
            "control_debug": self.last_control_debug,
            "contacts": self.get_contact_flags(),
        }

    @staticmethod
    def _geom_type_name(geom_type_code: int) -> str:
        mapping = {
            int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
            int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
            int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        }
        return mapping.get(int(geom_type_code), f"unknown_{int(geom_type_code)}")

    def get_model_diagnostics(self):
        wheel_collision_geom_ids = [
            gid
            for gid in self.wheel_geom_ids
            if int(self.model.geom_contype[gid]) != 0 or int(self.model.geom_conaffinity[gid]) != 0
        ]
        wheel_collision_geom_names = [self.geom_names[gid] for gid in wheel_collision_geom_ids]
        wheel_collision_geom_types = [
            self._geom_type_name(int(self.model.geom_type[gid])) for gid in wheel_collision_geom_ids
        ]

        wheel_collision_mesh_asset_names = []
        wheel_collision_mesh_asset_sources = []
        for gid, gtype in zip(wheel_collision_geom_ids, wheel_collision_geom_types):
            if gtype != "mesh":
                continue
            mesh_id = int(self.model.geom_dataid[gid])
            mesh_name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)
                or f"mesh_{mesh_id}"
            )
            wheel_collision_mesh_asset_names.append(mesh_name)
            if mesh_name.startswith("visual_"):
                wheel_collision_mesh_asset_sources.append("visual_stl")
            elif mesh_name.startswith("generated_"):
                wheel_collision_mesh_asset_sources.append("visual_stl_chunked")
            elif mesh_name.startswith("collision_"):
                wheel_collision_mesh_asset_sources.append("urdf_collision")
            else:
                wheel_collision_mesh_asset_sources.append("unknown")

        wheel_collision_mesh_count = int(sum(t == "mesh" for t in wheel_collision_geom_types))
        wheel_collision_cylinder_count = int(
            sum(t == "cylinder" for t in wheel_collision_geom_types)
        )
        if wheel_collision_geom_ids:
            if wheel_collision_mesh_count == len(wheel_collision_geom_ids):
                wheel_collision_mode_detected = "mesh"
            elif wheel_collision_cylinder_count == len(wheel_collision_geom_ids):
                wheel_collision_mode_detected = "cylinder"
            else:
                wheel_collision_mode_detected = "mixed"
        else:
            wheel_collision_mode_detected = "none"

        if wheel_collision_mesh_asset_sources:
            uniq_sources = sorted(set(wheel_collision_mesh_asset_sources))
            wheel_collision_mesh_asset_source_detected = (
                uniq_sources[0] if len(uniq_sources) == 1 else "mixed"
            )
        else:
            wheel_collision_mesh_asset_source_detected = "none"

        return {
            "task": self.task,
            "joint_names": list(self.joint_names),
            "joint_ids": list(self.joint_ids),
            "joint_qpos_addrs": list(self.joint_qpos_addrs),
            "joint_dof_addrs": list(self.joint_dof_addrs),
            "actuator_names": list(self.actuator_names),
            "actuator_ids": list(self.actuator_ids),
            "ctrl_size": int(self.model.nu),
            "qpos_size": int(self.model.nq),
            "qvel_size": int(self.model.nv),
            "dt": float(self.dt),
            "sim_dt": float(self.sim_dt),
            "decimation": int(self.decimation),
            "controller_mode": self.controller_mode,
            "implemented_controller_mode": self.implemented_controller_mode,
            "domain_rand_mode": self.domain_rand_mode,
            "fidelity_level": self.fidelity_level,
            "mujoco_tuning_profile": self.mujoco_tuning_profile_name,
            "current_reset_profile": self.current_reset_profile,
            "reset_mode": self.current_reset_profile,
            "wait_mode": self.wait_mode,
            "current_commands": self.current_commands.copy().tolist(),
            "command_resample_step": int(self.last_command_resample_step),
            "domain_rand_config": self.domain_randomizer.config_dict(),
            "wheel_collision_geom_names": wheel_collision_geom_names,
            "wheel_collision_geom_types": wheel_collision_geom_types,
            "wheel_collision_mesh_count": wheel_collision_mesh_count,
            "wheel_collision_cylinder_count": wheel_collision_cylinder_count,
            "wheel_collision_mode_detected": wheel_collision_mode_detected,
            "wheel_collision_mesh_asset_names": wheel_collision_mesh_asset_names,
            "wheel_collision_mesh_asset_sources": wheel_collision_mesh_asset_sources,
            "wheel_collision_mesh_asset_source_detected": wheel_collision_mesh_asset_source_detected,
        }

    # ------------------------------------------------------------------
    # Internal helpers: reset/state/commands/observation
    # ------------------------------------------------------------------
    def _post_reset_apply_default_state(self):
        self.data.qpos[0:3] = np.array([0.0, 0.0, 0.30], dtype=np.float64)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[0:6] = 0.0
        self.data.qpos[self.joint_qpos_addrs] = self.current_default_dof_pos
        self.data.qvel[self.joint_dof_addrs] = 0.0

    def _resolve_reset_profile(self, reset_profile: Optional[str]) -> tuple[str, BalanceResetRanges]:
        return get_balance_reset_profile(reset_profile, self.cfg)

    def _apply_balance_root_randomization(self, r: Optional[BalanceResetRanges] = None):
        r = self.cfg.balance_reset if r is None else r

        self.data.qpos[0] += self.np_random.uniform(*r.x_pos_offset)
        self.data.qpos[1] += self.np_random.uniform(*r.y_pos_offset)
        self.data.qpos[2] += self.np_random.uniform(*r.z_pos_offset)

        roll = self.np_random.uniform(*r.roll)
        pitch = self.np_random.uniform(*r.pitch)
        yaw = self.np_random.uniform(*r.yaw)
        self.data.qpos[3:7] = self._euler_to_quat(roll, pitch, yaw)

        self.data.qvel[0] = self.np_random.uniform(*r.lin_vel_x)
        self.data.qvel[1] = self.np_random.uniform(*r.lin_vel_y)
        self.data.qvel[2] = self.np_random.uniform(*r.lin_vel_z)
        self.data.qvel[3] = self.np_random.uniform(*r.ang_vel_roll)
        self.data.qvel[4] = self.np_random.uniform(*r.ang_vel_pitch)
        self.data.qvel[5] = self.np_random.uniform(*r.ang_vel_yaw)

    def _apply_fzqver_root_randomization(self):
        p = self.fzqver_profile
        upright = bool(self.np_random.uniform() < float(p.upright_ratio))

        if upright:
            roll = self.np_random.uniform(*p.upright_roll_pitch)
            pitch = self.np_random.uniform(*p.upright_roll_pitch)
            yaw = self.np_random.uniform(*p.upright_yaw)
            self.data.qpos[2] += self.np_random.uniform(*p.upright_z_offset)
            reset_mode = "upright"
        else:
            roll = self.np_random.uniform(*p.full_pose)
            pitch = self.np_random.uniform(*p.full_pose)
            yaw = self.np_random.uniform(*p.full_pose)
            self.data.qpos[2] += self.np_random.uniform(*p.fallen_z_offset)
            reset_mode = "fallen"

        self.data.qpos[3:7] = self._euler_to_quat(roll, pitch, yaw)

        self.data.qvel[0] = self.np_random.uniform(*p.lin_vel)
        self.data.qvel[1] = self.np_random.uniform(*p.lin_vel)
        self.data.qvel[2] = self.np_random.uniform(*p.lin_vel)
        self.data.qvel[3] = self.np_random.uniform(*p.ang_vel)
        self.data.qvel[4] = self.np_random.uniform(*p.ang_vel)
        self.data.qvel[5] = self.np_random.uniform(*p.ang_vel)

        self.current_reset_profile = f"fzqver_{reset_mode}"

    def _apply_mujoco_tuning_profile(self) -> None:
        """Apply MuJoCo-only tuning on top of current baseline/DR-adjusted parameters."""
        p = self.mujoco_tuning_profile
        self.current_theta_kp[:] = self.current_theta_kp * float(p.theta_kp_scale)
        self.current_theta_kd[:] = self.current_theta_kd * float(p.theta_kd_scale)
        self.current_l0_kp[:] = self.current_l0_kp * float(p.l0_kp_scale)
        self.current_l0_kd[:] = self.current_l0_kd * float(p.l0_kd_scale)
        self.current_wheel_kd[:] = self.current_wheel_kd * float(p.wheel_kd_scale)
        self.current_feedforward_force = float(self.base_feedforward_force) * float(
            p.feedforward_force_scale
        )
        self.current_balance_hint_pitch_threshold_rad = float(
            self.base_balance_hint_pitch_threshold_rad
            if p.balance_hint_pitch_threshold_rad_override is None
            else p.balance_hint_pitch_threshold_rad_override
        )
        self.current_balance_hint_knee_torque = float(
            self.base_balance_hint_knee_torque
            if p.balance_hint_knee_torque_override is None
            else p.balance_hint_knee_torque_override
        )
        self.current_enable_balance_hint = bool(self.base_enable_balance_hint)

    def _resample_commands(self, force: bool = False) -> None:
        if self.task == "wheel_legged_vmc_balance":
            if force or self.last_command_resample_step < 0:
                self.current_commands[:] = self.cfg.command_np
                self.last_command_resample_step = int(self.episode_steps)
            return

        # fzqver command distribution.
        if not force and self.episode_steps > 0:
            if (self.episode_steps % self.command_resample_interval_steps) != 0:
                return

        lin_vel_x = float(self.np_random.uniform(*self.fzqver_profile.lin_vel_x))
        ang_vel_yaw = float(self.np_random.uniform(*self.fzqver_profile.ang_vel_yaw))
        height = float(self.np_random.uniform(*self.fzqver_profile.height))

        if float(self.np_random.uniform()) < float(self.fzqver_profile.stand_env_ratio):
            lin_vel_x = 0.0
            ang_vel_yaw = 0.0
            height = float(self.fzqver_profile.stand_height)

        self.current_commands[:] = np.array([lin_vel_x, ang_vel_yaw, height], dtype=np.float64)
        self.last_command_resample_step = int(self.episode_steps)

    def _maybe_resample_commands(self) -> None:
        self._resample_commands(force=False)

    def _push_action_fifo(self, action: np.ndarray):
        self.action_fifo[1:] = self.action_fifo[:-1]
        self.action_fifo[0] = action

    @staticmethod
    def _wrap_angle_diff(diff: np.ndarray) -> np.ndarray:
        return (diff + np.pi) % (2 * np.pi) - np.pi

    def _refresh_state_from_sim(self, init_from_current: bool = False):
        self.base_quat = self.data.qpos[3:7].copy()  # wxyz
        self.base_ang_vel_world = self.data.qvel[3:6].copy()
        self.base_ang_vel_body = self._rotate_inverse(self.base_quat, self.base_ang_vel_world)
        self.projected_gravity = self._rotate_inverse(self.base_quat, np.array([0.0, 0.0, -1.0]))
        self.pitch_angle = float(np.arctan2(self.projected_gravity[1], -self.projected_gravity[2]))

        self.dof_pos = self.data.qpos[self.joint_qpos_addrs].copy()
        self.dof_vel_raw = self.data.qvel[self.joint_dof_addrs].copy()

        if init_from_current:
            self.prev_dof_pos_for_diff = self.dof_pos.copy()
            self.dof_pos_dot[:] = 0.0
            if self.cfg.dof_vel_use_pos_diff:
                self.dof_vel[:] = 0.0
            else:
                self.dof_vel[:] = self.dof_vel_raw
        else:
            diff = self._wrap_angle_diff(self.dof_pos - self.prev_dof_pos_for_diff)
            self.dof_pos_dot = diff / self.sim_dt
            if self.cfg.dof_vel_use_pos_diff:
                self.dof_vel = self.dof_pos_dot.copy()
            else:
                self.dof_vel = self.dof_vel_raw.copy()
            self.prev_dof_pos_for_diff = self.dof_pos.copy()

        self.theta1, self.theta2, vmc_state = self.vmc.batch_leg_state_from_dofs(
            self.dof_pos,
            self.dof_vel,
            velocity_fd_dt=self.cfg.vmc_velocity_fd_dt,
        )
        self.L0 = np.asarray(vmc_state.L0, dtype=np.float64)
        self.theta0 = np.asarray(vmc_state.theta0, dtype=np.float64)
        self.L0_dot = np.asarray(vmc_state.L0_dot, dtype=np.float64)
        self.theta0_dot = np.asarray(vmc_state.theta0_dot, dtype=np.float64)

    def _compute_observation(self, action_obs: np.ndarray):
        vmc_state = {
            "theta0": self.theta0,
            "theta0_dot": self.theta0_dot,
            "L0": self.L0,
            "L0_dot": self.L0_dot,
        }
        return self.obs_computer.compute_from_components(
            base_quat_wxyz=self.base_quat,
            base_ang_vel_world=self.base_ang_vel_world,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            action_obs=action_obs,
            commands=self.current_commands,
            vmc_state=vmc_state,
        )

    # ------------------------------------------------------------------
    # Internal helpers: control/reward
    # ------------------------------------------------------------------
    def _compute_torques_simplified(self, actions: np.ndarray) -> np.ndarray:
        pos_ref = actions * self.cfg.simple_pos_action_scale + self.current_default_dof_pos
        vel_ref = actions * self.cfg.simple_vel_action_scale
        torques = self.current_simple_p_gains * (pos_ref - self.dof_pos) + self.current_simple_d_gains * (
            vel_ref - self.dof_vel
        )
        torques = torques * self.current_torques_scale
        torques_clipped = np.clip(torques, -self.current_torque_limits, self.current_torque_limits)
        self.torque_saturation_count += int(np.any(np.abs(torques) > self.current_torque_limits + 1e-9))
        self.last_control_debug = {
            "mode": "simplified_joint_pd",
            "pos_ref": pos_ref.tolist(),
            "vel_ref": vel_ref.tolist(),
            "torques_pre_clip": torques.tolist(),
            "torques_post_clip": torques_clipped.tolist(),
        }
        return torques_clipped

    def _compute_torques_vmc_balance_exact(self, delayed_action: np.ndarray) -> np.ndarray:
        theta0_ref = np.array([delayed_action[0], delayed_action[3]], dtype=np.float64) * self.cfg.action_scale_theta
        l0_ref = (
            np.array([delayed_action[1], delayed_action[4]], dtype=np.float64)
            * self.cfg.action_scale_l0
            + self.cfg.l0_offset
        )
        wheel_vel_ref = np.array([delayed_action[2], delayed_action[5]], dtype=np.float64) * self.cfg.action_scale_vel

        torque_leg = self.current_theta_kp * (theta0_ref - self.theta0) - self.current_theta_kd * self.theta0_dot
        force_leg = self.current_l0_kp * (l0_ref - self.L0) - self.current_l0_kd * self.L0_dot
        if self.cfg.enable_gas_spring:
            self.gas_spring_force = self.cfg.gas_spring_k * self.L0 + self.cfg.gas_spring_b
            self.gas_spring_force = np.nan_to_num(
                self.gas_spring_force, nan=0.0, posinf=0.0, neginf=0.0
            )
        else:
            self.gas_spring_force[:] = 0.0
        self.virtual_leg_force_total = (
            force_leg
            + self.current_feedforward_force
            + self.gas_spring_force
        )
        self.virtual_leg_force_total = np.nan_to_num(
            self.virtual_leg_force_total, nan=0.0, posinf=0.0, neginf=0.0
        )
        wheel_vel = self.dof_vel[[2, 5]]
        torque_wheel = self.current_wheel_kd * (wheel_vel_ref - wheel_vel)

        T1, T2 = self.vmc.map_virtual_to_joint_torques(
            F=self.virtual_leg_force_total,
            T=torque_leg,
            theta1=self.theta1,
            theta2=self.theta2,
            L0=self.L0,
        )
        torques = np.array(
            [T1[0], T2[0], torque_wheel[0], T1[1], T2[1], torque_wheel[1]],
            dtype=np.float64,
        )
        torques = torques * self.current_torques_scale

        prompt_triggered = False
        if self.current_enable_balance_hint and abs(self.pitch_angle) > self.current_balance_hint_pitch_threshold_rad:
            for idx in self.cfg.balance_hint_joint_indices:
                torques[idx] = self.current_balance_hint_knee_torque
            prompt_triggered = True
            self.prompt_torque_step_count += 1
        self.prompt_torque_triggered_last_step = prompt_triggered

        torques_pre_clip = torques.copy()
        torques = np.clip(torques, -self.current_torque_limits, self.current_torque_limits)
        self.torque_saturation_count += int(np.any(np.abs(torques_pre_clip) > self.current_torque_limits + 1e-9))

        self.last_control_debug = {
            "mode": "vmc_balance_exact",
            "theta0_ref": theta0_ref.tolist(),
            "l0_ref": l0_ref.tolist(),
            "wheel_vel_ref": wheel_vel_ref.tolist(),
            "theta0": self.theta0.tolist(),
            "theta0_dot": self.theta0_dot.tolist(),
            "L0": self.L0.tolist(),
            "L0_dot": self.L0_dot.tolist(),
            "torque_leg": torque_leg.tolist(),
            "force_leg": force_leg.tolist(),
            "gas_spring_force": self.gas_spring_force.tolist(),
            "virtual_leg_force_total": self.virtual_leg_force_total.tolist(),
            "torque_wheel": torque_wheel.tolist(),
            "T1": np.asarray(T1).tolist(),
            "T2": np.asarray(T2).tolist(),
            "torques_pre_clip": torques_pre_clip.tolist(),
            "torques_post_clip": torques.tolist(),
            "pitch_angle": float(self.pitch_angle),
            "prompt_torque_triggered": bool(prompt_triggered),
            "feedforward_force": float(self.current_feedforward_force),
            "balance_hint_pitch_threshold_rad": float(self.current_balance_hint_pitch_threshold_rad),
            "balance_hint_knee_torque": float(self.current_balance_hint_knee_torque),
            "mujoco_tuning_profile": self.mujoco_tuning_profile_name,
            "task": self.task,
            "current_commands": self.current_commands.copy().tolist(),
        }
        return torques

    def _compute_reward(self):
        # Lightweight monitoring reward.
        upright_reward = -np.abs(self.projected_gravity[2] + 1.0)
        return float(upright_reward)

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _euler_to_quat(roll, pitch, yaw):
        rot = Rotation.from_euler("xyz", [roll, pitch, yaw])
        quat_scipy = rot.as_quat()  # [x, y, z, w]
        return np.array([quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]], dtype=np.float64)

    @staticmethod
    def _rotate_inverse(quat_wxyz, vec_xyz):
        quat = np.asarray(quat_wxyz, dtype=np.float64)
        vec = np.asarray(vec_xyz, dtype=np.float64)
        quat_scipy = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
        rot = Rotation.from_quat(quat_scipy)
        return rot.inv().apply(vec)
