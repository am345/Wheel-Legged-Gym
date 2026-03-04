"""
MuJoCo sim2sim control/randomization configuration.

These defaults mirror the training-side key parameters while staying standalone
(avoid importing IsaacGym configs in MuJoCo runtime scripts).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class BalanceResetRanges:
    x_pos_offset: Tuple[float, float] = (0.0, 0.0)
    y_pos_offset: Tuple[float, float] = (0.0, 0.0)
    z_pos_offset: Tuple[float, float] = (0.0, 0.0)
    roll: Tuple[float, float] = (-3.14, 3.14)
    pitch: Tuple[float, float] = (-3.14, 3.14)
    yaw: Tuple[float, float] = (-3.14, 3.14)
    lin_vel_x: Tuple[float, float] = (-0.5, 0.5)
    lin_vel_y: Tuple[float, float] = (-0.5, 0.5)
    lin_vel_z: Tuple[float, float] = (-0.5, 0.5)
    ang_vel_roll: Tuple[float, float] = (-0.5, 0.5)
    ang_vel_pitch: Tuple[float, float] = (-0.5, 0.5)
    ang_vel_yaw: Tuple[float, float] = (-0.5, 0.5)

    def to_dict(self) -> Dict[str, List[float]]:
        return {k: list(v) for k, v in asdict(self).items()}

    def copy(self) -> "BalanceResetRanges":
        return BalanceResetRanges(**asdict(self))


@dataclass
class FzqverSim2SimProfile:
    """Task profile used by MuJoCo sim2sim when task=wheel_legged_fzqver."""

    lin_vel_x: Tuple[float, float] = (-0.8, 0.8)
    ang_vel_yaw: Tuple[float, float] = (-1.5, 1.5)
    height: Tuple[float, float] = (0.20, 0.24)
    stand_env_ratio: float = 0.35
    stand_height: float = 0.22
    resampling_time_s: float = 5.0
    heading_command: bool = False
    enable_gas_spring: bool = True
    gas_spring_k: float = 188.3447
    gas_spring_b: float = 1.2055
    enable_policy_gas_compensation: bool = True
    policy_gas_comp_sigmoid_scale: float = 1.0

    upright_ratio: float = 0.2
    full_pose: Tuple[float, float] = (-3.14, 3.14)
    upright_roll_pitch: Tuple[float, float] = (-0.25, 0.25)
    upright_yaw: Tuple[float, float] = (-3.14, 3.14)
    fallen_z_offset: Tuple[float, float] = (0.0, 0.08)
    upright_z_offset: Tuple[float, float] = (0.0, 0.03)
    lin_vel: Tuple[float, float] = (-0.5, 0.5)
    ang_vel: Tuple[float, float] = (-0.5, 0.5)

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        for k, v in list(out.items()):
            if isinstance(v, tuple):
                out[k] = list(v)
        return out


@dataclass
class BalanceVMCControlConfig:
    # Core timing
    sim_dt: float = 0.005
    control_decimation: int = 2

    # Robot geometry (cfg.asset)
    l1: float = 0.167
    l2: float = 0.200
    offset: float = 0.0

    # Observation/action clipping
    clip_actions: float = 100.0
    clip_observations: float = 100.0

    # Default command used by balance task
    command: Tuple[float, float, float] = (0.0, 0.0, 0.24)
    commands_scale: Tuple[float, float, float] = (2.0, 0.25, 1.0)

    # Observation scales
    obs_scales_ang_vel: float = 0.25
    obs_scales_dof_pos: float = 1.0
    obs_scales_dof_vel: float = 0.05
    obs_scales_l0: float = 5.0
    obs_scales_l0_dot: float = 0.25

    # Default DOF state / limits
    default_dof_pos: Tuple[float, ...] = (0.4, 0.25, 0.0, 0.4, 0.25, 0.0)
    torque_limits: Tuple[float, ...] = (30.0, 30.0, 2.0, 30.0, 30.0, 2.0)

    # VMC action semantics
    action_scale_theta: float = 3.14
    action_scale_l0: float = 0.10
    action_scale_vel: float = 5.0
    l0_offset: float = 0.22
    feedforward_force: float = 40.0
    enable_gas_spring: bool = False
    gas_spring_k: float = 188.3447
    gas_spring_b: float = 1.2055
    enable_policy_gas_compensation: bool = False
    policy_gas_comp_sigmoid_scale: float = 1.0

    # VMC PD gains
    kp_theta: float = 10.0
    kd_theta: float = 5.0
    kp_l0: float = 800.0
    kd_l0: float = 7.0

    # Wheel velocity damping
    wheel_kd: float = 0.5

    # Simplified baseline controller
    simple_pos_action_scale: float = 0.5
    simple_vel_action_scale: float = 10.0
    simple_p_gains: Tuple[float, ...] = (40.0, 40.0, 0.0, 40.0, 40.0, 0.0)
    simple_d_gains: Tuple[float, ...] = (1.0, 1.0, 0.5, 1.0, 1.0, 0.5)

    # Balance hint logic
    enable_balance_hint: bool = True
    balance_hint_pitch_threshold_rad: float = 0.349
    balance_hint_knee_torque: float = -30.0
    balance_hint_joint_indices: Tuple[int, int] = (1, 4)

    # Misc
    vmc_velocity_fd_dt: float = 0.001
    dof_vel_use_pos_diff: bool = True
    upright_tilt_threshold_deg: float = 10.0

    # Balance reset
    balance_reset: BalanceResetRanges = field(default_factory=BalanceResetRanges)

    # Fzqver sim2sim profile
    fzqver_profile: FzqverSim2SimProfile = field(default_factory=FzqverSim2SimProfile)

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        for k, v in list(out.items()):
            if isinstance(v, tuple):
                out[k] = list(v)
        for nested_key in ("balance_reset", "fzqver_profile"):
            nested = out.get(nested_key)
            if isinstance(nested, dict):
                for k, v in list(nested.items()):
                    if isinstance(v, tuple):
                        nested[k] = list(v)
        return out

    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.control_decimation

    @property
    def default_dof_pos_np(self) -> np.ndarray:
        return np.array(self.default_dof_pos, dtype=np.float64)

    @property
    def torque_limits_np(self) -> np.ndarray:
        return np.array(self.torque_limits, dtype=np.float64)

    @property
    def simple_p_gains_np(self) -> np.ndarray:
        return np.array(self.simple_p_gains, dtype=np.float64)

    @property
    def simple_d_gains_np(self) -> np.ndarray:
        return np.array(self.simple_d_gains, dtype=np.float64)

    @property
    def command_np(self) -> np.ndarray:
        return np.array(self.command, dtype=np.float64)

    @property
    def commands_scale_np(self) -> np.ndarray:
        return np.array(self.commands_scale, dtype=np.float64)


@dataclass
class MuJoCoDemoTuningProfile:
    """MuJoCo-only tuning scales for interactive demos."""

    name: str = "exact_baseline"
    theta_kp_scale: float = 1.0
    theta_kd_scale: float = 1.0
    l0_kp_scale: float = 1.0
    l0_kd_scale: float = 1.0
    wheel_kd_scale: float = 1.0
    feedforward_force_scale: float = 1.0
    balance_hint_pitch_threshold_rad_override: Optional[float] = None
    balance_hint_knee_torque_override: Optional[float] = None
    action_lowpass_alpha: Optional[float] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DomainRandTrainRanges:
    # Contact
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.1, 2.0)
    randomize_restitution: bool = True
    restitution_range: Tuple[float, float] = (0.0, 1.0)

    # Base physical properties
    randomize_base_mass: bool = True
    added_mass_range: Tuple[float, float] = (-2.0, 3.0)
    randomize_inertia: bool = True
    randomize_inertia_range: Tuple[float, float] = (0.8, 1.2)
    randomize_base_com: bool = True
    rand_com_vec: Tuple[float, float, float] = (0.05, 0.05, 0.05)

    # Controller randomization
    randomize_Kp: bool = True
    randomize_Kp_range: Tuple[float, float] = (0.9, 1.1)
    randomize_Kd: bool = True
    randomize_Kd_range: Tuple[float, float] = (0.9, 1.1)
    randomize_motor_torque: bool = True
    randomize_motor_torque_range: Tuple[float, float] = (0.9, 1.1)
    randomize_default_dof_pos: bool = True
    randomize_default_dof_pos_range: Tuple[float, float] = (-0.05, 0.05)
    randomize_action_delay: bool = True
    delay_ms_range: Tuple[float, float] = (0.0, 10.0)

    push_robots: bool = False

    def to_dict(self) -> Dict[str, object]:
        out = asdict(self)
        for k, v in list(out.items()):
            if isinstance(v, tuple):
                out[k] = list(v)
        return out


def get_balance_vmc_control_config() -> BalanceVMCControlConfig:
    return BalanceVMCControlConfig()


def get_fzqver_sim2sim_profile() -> FzqverSim2SimProfile:
    return FzqverSim2SimProfile()


def get_domain_rand_train_ranges() -> DomainRandTrainRanges:
    return DomainRandTrainRanges()


def get_balance_reset_profile(
    profile_name: Optional[str],
    base_cfg: Optional[BalanceVMCControlConfig] = None,
) -> Tuple[str, BalanceResetRanges]:
    """Return a named reset profile and corresponding ranges."""
    cfg = base_cfg if base_cfg is not None else get_balance_vmc_control_config()
    raw = "default" if profile_name is None else str(profile_name).strip()
    key = raw.lower()

    if key in ("", "default", "hard_random_balance", "random_balance"):
        return ("hard_random_balance", cfg.balance_reset.copy())

    if key in ("nominal", "nominal_demo"):
        deg = np.deg2rad
        return (
            "nominal_demo",
            BalanceResetRanges(
                x_pos_offset=(0.0, 0.0),
                y_pos_offset=(0.0, 0.0),
                z_pos_offset=(-0.015, 0.015),
                roll=(-float(deg(8.0)), float(deg(8.0))),
                pitch=(-float(deg(8.0)), float(deg(8.0))),
                yaw=(-float(deg(10.0)), float(deg(10.0))),
                lin_vel_x=(-0.10, 0.10),
                lin_vel_y=(-0.10, 0.10),
                lin_vel_z=(-0.05, 0.05),
                ang_vel_roll=(-0.20, 0.20),
                ang_vel_pitch=(-0.20, 0.20),
                ang_vel_yaw=(-0.20, 0.20),
            ),
        )

    raise ValueError(
        f"Unsupported reset profile: {profile_name}. "
        "Expected one of {default, nominal_demo, nominal, hard_random_balance, random_balance}."
    )


def get_mujoco_demo_tuning_profile(profile_name: str = "exact_baseline") -> MuJoCoDemoTuningProfile:
    """Return MuJoCo-only tuning profile."""
    key = str(profile_name).strip().lower()
    if key in ("exact_baseline", "exact", "none"):
        return MuJoCoDemoTuningProfile(name="exact_baseline")
    if key in ("demo_tuned", "tuned", "mujoco_demo"):
        return MuJoCoDemoTuningProfile(
            name="demo_tuned",
            theta_kp_scale=1.10,
            theta_kd_scale=1.05,
            l0_kp_scale=1.08,
            l0_kd_scale=1.10,
            wheel_kd_scale=1.25,
            feedforward_force_scale=1.05,
            balance_hint_pitch_threshold_rad_override=float(np.deg2rad(25.0)),
            balance_hint_knee_torque_override=-26.0,
            action_lowpass_alpha=None,
        )
    raise ValueError(
        f"Unsupported mujoco tuning profile: {profile_name}. "
        "Expected one of {exact_baseline, demo_tuned}."
    )
