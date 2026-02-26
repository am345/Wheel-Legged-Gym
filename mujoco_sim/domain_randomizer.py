"""
MuJoCo sim2sim domain randomization (episode-level)

目标：尽量对齐训练侧的随机化分布，同时保持 MuJoCo 实现可控/可诊断。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

import numpy as np
import mujoco

from .control_config import DomainRandTrainRanges, get_domain_rand_train_ranges


class MuJoCoDomainRandomizer:
    """
    MuJoCo episode-level domain randomizer.

    约定：
    - `env` 需要暴露 baseline caches 与若干命名字段（见 apply_to_env）
    - restitution 在 MuJoCo 中使用 `geom_solref[...,1]` 作为阻尼比代理做近似映射
    """

    def __init__(
        self,
        mode: str = "off",
        config: DomainRandTrainRanges | None = None,
    ):
        if mode not in ("off", "train_ranges"):
            raise ValueError(f"Unsupported domain randomization mode: {mode}")
        self.mode = mode
        self.cfg = config if config is not None else get_domain_rand_train_ranges()

    def is_enabled(self) -> bool:
        return self.mode != "off"

    def sample(self, rng: np.random.Generator, env) -> Dict[str, Any]:
        """
        采样一组 episode 参数。若 mode=off，返回 baseline 标记。
        """
        if not self.is_enabled():
            return {
                "mode": "off",
                "seed": int(env.seed) if env.seed is not None else None,
                "randomized": False,
                "restitution_application": "none",
            }

        cfg = self.cfg
        sample: Dict[str, Any] = {
            "mode": self.mode,
            "randomized": True,
        }

        # Contact properties (single scalar for this episode)
        if cfg.randomize_friction:
            friction = float(rng.uniform(*cfg.friction_range))
        else:
            friction = float(env.baseline_contact_defaults["friction_scalar"])

        if cfg.randomize_restitution:
            restitution = float(rng.uniform(*cfg.restitution_range))
        else:
            restitution = float(env.baseline_contact_defaults["restitution_scalar"])

        sample["contact"] = {
            "friction": friction,
            "restitution": restitution,
        }

        # Base mass / inertia / COM
        base_mass_delta = (
            float(rng.uniform(*cfg.added_mass_range)) if cfg.randomize_base_mass else 0.0
        )
        inertia_scale = (
            float(rng.uniform(*cfg.randomize_inertia_range))
            if cfg.randomize_inertia
            else 1.0
        )
        if cfg.randomize_base_com:
            cx, cy, cz = cfg.rand_com_vec
            base_com_offset = np.array(
                [
                    rng.uniform(-cx, cx),
                    rng.uniform(-cy, cy),
                    rng.uniform(-cz, cz),
                ],
                dtype=np.float64,
            )
        else:
            base_com_offset = np.zeros(3, dtype=np.float64)

        sample["base_model"] = {
            "base_mass_delta": base_mass_delta,
            "inertia_scale": inertia_scale,
            "base_com_offset": base_com_offset.tolist(),
        }

        # Controller randomization
        def _scale_array(shape, enabled, value_range, default=1.0):
            if not enabled:
                return np.full(shape, default, dtype=np.float64)
            lo, hi = value_range
            return rng.uniform(lo, hi, size=shape).astype(np.float64)

        # training侧 randomize_default_dof_pos 是每个DoF独立采样
        if cfg.randomize_default_dof_pos:
            dof_pos_offset = rng.uniform(
                cfg.randomize_default_dof_pos_range[0],
                cfg.randomize_default_dof_pos_range[1],
                size=(6,),
            ).astype(np.float64)
        else:
            dof_pos_offset = np.zeros(6, dtype=np.float64)

        if cfg.randomize_action_delay:
            delay_steps = int(
                round(
                    rng.uniform(cfg.delay_ms_range[0], cfg.delay_ms_range[1]) / 1000.0 / env.sim_dt
                )
            )
            delay_steps = int(np.clip(delay_steps, 0, env.action_delay_max_idx))
        else:
            delay_steps = 0

        sample["controller"] = {
            "p_gains_scale": _scale_array((6,), cfg.randomize_Kp, cfg.randomize_Kp_range).tolist(),
            "d_gains_scale": _scale_array((6,), cfg.randomize_Kd, cfg.randomize_Kd_range).tolist(),
            "theta_kp_scale": _scale_array((2,), cfg.randomize_Kp, cfg.randomize_Kp_range).tolist(),
            "theta_kd_scale": _scale_array((2,), cfg.randomize_Kd, cfg.randomize_Kd_range).tolist(),
            "l0_kp_scale": _scale_array((2,), cfg.randomize_Kp, cfg.randomize_Kp_range).tolist(),
            "l0_kd_scale": _scale_array((2,), cfg.randomize_Kd, cfg.randomize_Kd_range).tolist(),
            "torques_scale": _scale_array(
                (6,), cfg.randomize_motor_torque, cfg.randomize_motor_torque_range
            ).tolist(),
            "default_dof_pos_offset": dof_pos_offset.tolist(),
            "action_delay_idx": delay_steps,
        }

        sample["restitution_application"] = "geom_solref_dampratio_proxy"
        return sample

    def reset_env_to_baseline(self, env) -> None:
        """
        恢复模型和控制器到 baseline（每个 episode reset 前先调用）。
        """
        model = env.model
        data = env.data

        model.body_mass[:] = env.baseline_model_params["body_mass"]
        model.body_inertia[:] = env.baseline_model_params["body_inertia"]
        model.body_ipos[:] = env.baseline_model_params["body_ipos"]
        model.geom_friction[:] = env.baseline_model_params["geom_friction"]
        model.geom_solref[:] = env.baseline_model_params["geom_solref"]
        model.geom_solimp[:] = env.baseline_model_params["geom_solimp"]

        env.current_default_dof_pos[:] = env.base_default_dof_pos
        env.current_torque_limits[:] = env.base_torque_limits
        env.current_torques_scale[:] = env.base_torques_scale
        env.current_simple_p_gains[:] = env.base_simple_p_gains
        env.current_simple_d_gains[:] = env.base_simple_d_gains
        env.current_theta_kp[:] = env.base_theta_kp
        env.current_theta_kd[:] = env.base_theta_kd
        env.current_l0_kp[:] = env.base_l0_kp
        env.current_l0_kd[:] = env.base_l0_kd
        env.current_wheel_kd[:] = env.base_wheel_kd
        env.action_delay_idx = 0
        env.current_domain_params = {
            "mode": "baseline",
            "randomized": False,
            "restitution_application": "none",
        }

        mujoco.mj_setConst(model, data)

    def apply_to_env(self, env, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        将采样结果应用到 MuJoCo 模型与控制器。

        Returns:
            实际应用后的参数（含可能的裁剪/代理映射信息），可直接写入结果JSON
        """
        if not sample.get("randomized", False):
            self.reset_env_to_baseline(env)
            env.current_domain_params = sample
            return sample

        # 先回 baseline 再应用，避免参数累乘漂移
        self.reset_env_to_baseline(env)

        model = env.model
        data = env.data

        applied = {
            "mode": sample["mode"],
            "randomized": True,
            "contact": dict(sample["contact"]),
            "base_model": dict(sample["base_model"]),
            "controller": dict(sample["controller"]),
            "restitution_application": sample.get("restitution_application", "unknown"),
        }

        # 1) Contact parameters
        contact = sample["contact"]
        friction = float(contact["friction"])
        restitution = float(contact["restitution"])

        # Apply friction to selected geoms (floor + robot)
        target_geom_ids = env.floor_geom_ids + env.robot_geom_ids
        for gid in target_geom_ids:
            model.geom_friction[gid, 0] = friction
            # Keep the remaining coefficients at baseline values for stability

        # Approximate restitution via contact damping ratio proxy in geom_solref[:,1]
        # Lower damping ratio -> more bounce. Clamp to keep simulation stable.
        # baseline geom_solref[:,1] usually 1.0
        dampratio = float(np.clip(1.5 - 1.4 * restitution, 0.05, 1.5))
        for gid in target_geom_ids:
            model.geom_solref[gid, 1] = dampratio
        applied["contact"]["restitution_dampratio_proxy"] = dampratio

        # 2) Base mass / inertia / COM
        base_body_id = env.base_body_id
        base_mass0 = env.baseline_model_params["body_mass"][base_body_id]
        base_inertia0 = env.baseline_model_params["body_inertia"][base_body_id]
        base_ipos0 = env.baseline_model_params["body_ipos"][base_body_id]

        base_mass_delta = float(sample["base_model"]["base_mass_delta"])
        inertia_scale = float(sample["base_model"]["inertia_scale"])
        base_com_offset = np.asarray(sample["base_model"]["base_com_offset"], dtype=np.float64)

        model.body_mass[base_body_id] = max(1e-6, base_mass0 + base_mass_delta)
        model.body_inertia[base_body_id] = np.maximum(1e-9, base_inertia0 * inertia_scale)
        model.body_ipos[base_body_id] = base_ipos0 + base_com_offset

        # 3) Controller parameters
        c = sample["controller"]
        env.current_simple_p_gains[:] = env.base_simple_p_gains * np.asarray(c["p_gains_scale"])
        env.current_simple_d_gains[:] = env.base_simple_d_gains * np.asarray(c["d_gains_scale"])
        env.current_theta_kp[:] = env.base_theta_kp * np.asarray(c["theta_kp_scale"])
        env.current_theta_kd[:] = env.base_theta_kd * np.asarray(c["theta_kd_scale"])
        env.current_l0_kp[:] = env.base_l0_kp * np.asarray(c["l0_kp_scale"])
        env.current_l0_kd[:] = env.base_l0_kd * np.asarray(c["l0_kd_scale"])
        env.current_torques_scale[:] = env.base_torques_scale * np.asarray(c["torques_scale"])
        env.current_default_dof_pos[:] = env.base_default_dof_pos + np.asarray(c["default_dof_pos_offset"])
        env.action_delay_idx = int(c["action_delay_idx"])

        mujoco.mj_setConst(model, data)
        env.current_domain_params = applied
        return applied

    def config_dict(self) -> Dict[str, Any]:
        return asdict(self.cfg)

