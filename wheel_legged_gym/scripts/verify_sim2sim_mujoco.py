#!/usr/bin/env python3
"""
MuJoCo sim2sim 验证脚本（wheel_legged_vmc_balance）

支持两类用途：
1. 长时 rollout 评估（真实 sim2sim 主评估）
2. 可选的 IsaacGym 参考轨迹对齐回放（短序列工程对齐）
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from mujoco_sim.alignment_utils import replay_reference_rollout_in_mujoco
from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv
from mujoco_sim.policy_loader import PolicyLoader

SUPPORTED_TASKS = (
    "wheel_legged_vmc_balance",
    "wheel_legged_fzqver",
)
DEFAULT_TASK = "wheel_legged_vmc_balance"
DEFAULT_SIMPLE_MODEL = "resources/robots/serialleg/mjcf/serialleg_simple.xml"
DEFAULT_FIDELITY_MODEL = "resources/robots/serialleg/mjcf/serialleg_fidelity.xml"
DEFAULT_CHECKPOINT = "logs/wheel_legged_vmc_balance/Feb23_15-01-26_/model_900.pt"
DEFAULT_CONTROLLER_MODE = "vmc_balance_exact"
DEFAULT_DOMAIN_RAND_MODE = "train_ranges"

EXPECTED_JOINT_NAMES = [
    "lf0_Joint",
    "lf1_Joint",
    "l_wheel_Joint",
    "rf0_Joint",
    "rf1_Joint",
    "r_wheel_Joint",
]
EXPECTED_ACTUATOR_NAMES = [
    "lf0_act",
    "lf1_act",
    "l_wheel_act",
    "rf0_act",
    "rf1_act",
    "r_wheel_act",
]


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def set_global_seed(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_seed_application(seed: int) -> Dict[str, Any]:
    """快速验证 numpy/torch seed 生效。"""
    np.random.seed(seed)
    np_probe_1 = float(np.random.uniform())
    np.random.seed(seed)
    np_probe_2 = float(np.random.uniform())

    torch.manual_seed(seed)
    torch_probe_1 = float(torch.rand(1).item())
    torch.manual_seed(seed)
    torch_probe_2 = float(torch.rand(1).item())

    if not np.isclose(np_probe_1, np_probe_2):
        raise RuntimeError("NumPy seed reproducibility check failed.")
    if not np.isclose(torch_probe_1, torch_probe_2):
        raise RuntimeError("Torch seed reproducibility check failed.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    return {
        "seed": int(seed),
        "numpy_probe": np_probe_1,
        "torch_probe": torch_probe_1,
        "torch_initial_seed": int(torch.initial_seed()),
    }


def flatten_numeric(prefix: str, value: Any, out: Dict[str, List[float]]) -> None:
    """Flatten nested numeric values for aggregate param coverage summary."""
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float, np.integer, np.floating)):
        out.setdefault(prefix, []).append(float(value))
        return
    if isinstance(value, dict):
        for k, v in value.items():
            next_prefix = f"{prefix}.{k}" if prefix else str(k)
            flatten_numeric(next_prefix, v, out)
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        for i, v in enumerate(list(value)):
            next_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            flatten_numeric(next_prefix, v, out)


def summarize_episode_params(episodes: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    flattened: Dict[str, List[float]] = {}
    for ep in episodes:
        params = ep.get("episode_params")
        if params is None:
            continue
        flatten_numeric("", params, flattened)
    summary: Dict[str, Dict[str, float]] = {}
    for key, vals in flattened.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            continue
        summary[key] = {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
        }
    return summary


class RolloutMetrics:
    """收集 episode 指标并生成标准化 JSON 结果。"""

    def __init__(self, control_dt: float):
        self.control_dt = float(control_dt)
        self.episode_data: List[Dict[str, Any]] = []
        self._reset_current()

    def _reset_current(self) -> None:
        self.current = {
            "heights": [],
            "tilts": [],
            "angular_velocities": [],
            "upright_time": 0,
            "max_tilt": 0.0,
        }

    def update(self, env: MuJoCoBalanceEnv) -> None:
        height = float(env.data.qpos[2])
        tilt_angle_deg = float(
            np.degrees(np.arccos(np.clip(-float(env.projected_gravity[2]), -1.0, 1.0)))
        )
        angvel_norm = float(np.linalg.norm(env.base_ang_vel_body))

        self.current["heights"].append(height)
        self.current["tilts"].append(tilt_angle_deg)
        self.current["angular_velocities"].append(angvel_norm)

        if tilt_angle_deg < float(env.cfg.upright_tilt_threshold_deg):
            self.current["upright_time"] += 1
        self.current["max_tilt"] = max(self.current["max_tilt"], tilt_angle_deg)

    def log_episode(
        self,
        *,
        episode_num: int,
        steps: int,
        reward: float,
        env: MuJoCoBalanceEnv,
        termination_reason: str,
        record_episode_params: bool,
    ) -> None:
        heights = np.asarray(self.current["heights"], dtype=np.float64)
        tilts = np.asarray(self.current["tilts"], dtype=np.float64)
        angvels = np.asarray(self.current["angular_velocities"], dtype=np.float64)
        steps = int(steps)

        if steps <= 0 or heights.size == 0:
            control_updates = 0
            ep = {
                "episode": int(episode_num),
                "steps": steps,
                "reward": float(reward),
                "mean_height": None,
                "std_height": None,
                "upright_ratio": 0.0,
                "max_tilt": None,
                "mean_tilt": None,
                "mean_angvel": None,
                "survival_time_seconds": 0.0,
                "termination_reason": str(termination_reason),
                "fall": bool(termination_reason not in ("timeout", "max_steps", "running")),
                "prompt_torque_steps": int(env.prompt_torque_step_count),
                "prompt_torque_trigger_ratio": 0.0,
                "torque_saturation_steps": int(env.torque_saturation_count),
                "torque_saturation_rate": 0.0,
                "action_clip_steps": int(env.action_clip_count),
                "action_clip_rate": 0.0,
                "episode_params": deepcopy(to_jsonable(env.current_domain_params))
                if record_episode_params
                else None,
                "controller_debug_summary": self._controller_debug_summary(env),
            }
        else:
            # vmc mode recomputes torque every substep; simplified mode once per control step
            if env.implemented_controller_mode == "vmc_balance_exact":
                control_updates = max(1, steps * int(env.decimation))
            else:
                control_updates = max(1, steps)
            ep = {
                "episode": int(episode_num),
                "steps": steps,
                "reward": float(reward),
                "mean_height": float(np.mean(heights)),
                "std_height": float(np.std(heights)),
                "upright_ratio": float(self.current["upright_time"] / steps),
                "max_tilt": float(self.current["max_tilt"]),
                "mean_tilt": float(np.mean(tilts)),
                "mean_angvel": float(np.mean(angvels)),
                "survival_time_seconds": float(steps * self.control_dt),
                "termination_reason": str(termination_reason),
                "fall": bool(termination_reason not in ("timeout", "max_steps")),
                "prompt_torque_steps": int(env.prompt_torque_step_count),
                "prompt_torque_trigger_ratio": float(env.prompt_torque_step_count / control_updates),
                "torque_saturation_steps": int(env.torque_saturation_count),
                "torque_saturation_rate": float(env.torque_saturation_count / control_updates),
                "action_clip_steps": int(env.action_clip_count),
                "action_clip_rate": float(env.action_clip_count / max(1, steps)),
                "episode_params": deepcopy(to_jsonable(env.current_domain_params))
                if record_episode_params
                else None,
                "controller_debug_summary": self._controller_debug_summary(env),
            }

        self.episode_data.append(ep)
        self._reset_current()

    @staticmethod
    def _controller_debug_summary(env: MuJoCoBalanceEnv) -> Dict[str, Any]:
        dbg = env.get_debug_state()
        ctrl_dbg = dbg.get("control_debug") or {}
        out = {
            "mode": dbg.get("controller_mode"),
            "mujoco_tuning_profile": dbg.get("mujoco_tuning_profile"),
            "reset_profile": dbg.get("reset_profile"),
            "action_delay_idx": dbg.get("action_delay_idx"),
            "prompt_torque_step_count": dbg.get("prompt_torque_step_count"),
            "balance_hint_active": dbg.get("balance_hint_active"),
            "torque_saturation_count": dbg.get("torque_saturation_count"),
            "torque_saturation_flags": dbg.get("torque_saturation_flags"),
            "action_clip_count": dbg.get("action_clip_count"),
            "final_pitch_angle": dbg.get("pitch_angle"),
            "final_tilt_deg": dbg.get("tilt_deg"),
            "joint_limit_hit_flags": dbg.get("joint_limit_hit_flags"),
            "joint_limit_margin": dbg.get("joint_limit_margin"),
            "contacts": dbg.get("contacts"),
            "last_ctrl": dbg.get("last_ctrl"),
        }
        # Keep VMC intermediate signals concise but useful
        for key in (
            "theta0_ref",
            "l0_ref",
            "wheel_vel_ref",
            "theta0",
            "L0",
            "torque_leg",
            "force_leg",
            "torque_wheel",
            "prompt_torque_triggered",
        ):
            if key in ctrl_dbg:
                out[key] = ctrl_dbg[key]
        return to_jsonable(out)

    def compute_aggregate(self) -> Dict[str, Any]:
        if not self.episode_data:
            return {
                "num_episodes": 0,
                "mean_steps": None,
                "std_steps": None,
                "mean_reward": None,
                "std_reward": None,
                "mean_upright_ratio": None,
                "mean_tilt": None,
                "mean_max_tilt": None,
                "mean_angvel": None,
                "mean_base_height": None,
                "std_base_height": None,
                "survival_time_seconds": None,
                "fall_rate": None,
                "mean_prompt_torque_trigger_ratio": None,
                "mean_torque_saturation_rate": None,
                "mean_action_clip_rate": None,
                "failure_modes": {},
                "episode_params_summary": {},
            }

        def vals(key: str) -> np.ndarray:
            return np.asarray([e[key] for e in self.episode_data if e.get(key) is not None], dtype=np.float64)

        steps = vals("steps")
        rewards = vals("reward")
        upright = vals("upright_ratio")
        mean_tilts = vals("mean_tilt")
        max_tilts = vals("max_tilt")
        angvel = vals("mean_angvel")
        mean_heights = vals("mean_height")
        survival = vals("survival_time_seconds")
        prompt_ratio = vals("prompt_torque_trigger_ratio")
        sat_ratio = vals("torque_saturation_rate")
        clip_ratio = vals("action_clip_rate")
        falls = np.asarray([1.0 if bool(e.get("fall")) else 0.0 for e in self.episode_data], dtype=np.float64)

        failure_modes: Dict[str, int] = {}
        for e in self.episode_data:
            r = str(e.get("termination_reason", "unknown"))
            failure_modes[r] = failure_modes.get(r, 0) + 1

        return {
            "num_episodes": int(len(self.episode_data)),
            "mean_steps": float(np.mean(steps)) if steps.size else None,
            "std_steps": float(np.std(steps)) if steps.size else None,
            "mean_reward": float(np.mean(rewards)) if rewards.size else None,
            "std_reward": float(np.std(rewards)) if rewards.size else None,
            "mean_upright_ratio": float(np.mean(upright)) if upright.size else None,
            "mean_tilt": float(np.mean(mean_tilts)) if mean_tilts.size else None,
            "mean_max_tilt": float(np.mean(max_tilts)) if max_tilts.size else None,
            "mean_angvel": float(np.mean(angvel)) if angvel.size else None,
            "mean_base_height": float(np.mean(mean_heights)) if mean_heights.size else None,
            "std_base_height": float(np.std(mean_heights)) if mean_heights.size else None,
            "survival_time_seconds": float(np.mean(survival)) if survival.size else None,
            "fall_rate": float(np.mean(falls)) if falls.size else None,
            "mean_prompt_torque_trigger_ratio": float(np.mean(prompt_ratio)) if prompt_ratio.size else None,
            "mean_torque_saturation_rate": float(np.mean(sat_ratio)) if sat_ratio.size else None,
            "mean_action_clip_rate": float(np.mean(clip_ratio)) if clip_ratio.size else None,
            "failure_modes": failure_modes,
            "episode_params_summary": summarize_episode_params(self.episode_data),
        }

    def print_summary(self) -> Dict[str, Any]:
        agg = self.compute_aggregate()
        if agg["num_episodes"] == 0:
            return agg

        print("\n" + "=" * 72)
        print("MuJoCo Sim2Sim 验证结果")
        print("=" * 72)
        print(f"Episodes: {agg['num_episodes']}")
        print(f"平均步数: {agg['mean_steps']:.1f} (std={agg['std_steps']:.1f})")
        print(f"平均生存时间: {agg['survival_time_seconds']:.2f}s")
        print(f"平均奖励: {agg['mean_reward']:.3f} (std={agg['std_reward']:.3f})")
        print(f"平均直立率: {agg['mean_upright_ratio'] * 100:.1f}%")
        print(f"平均倾角: {agg['mean_tilt']:.2f} deg")
        print(f"平均最大倾角: {agg['mean_max_tilt']:.2f} deg")
        print(f"平均角速度范数: {agg['mean_angvel']:.3f} rad/s")
        print(f"跌倒率: {agg['fall_rate'] * 100:.1f}%")
        print(
            "控制统计: "
            f"torque_sat={agg['mean_torque_saturation_rate']:.3f}, "
            f"action_clip={agg['mean_action_clip_rate']:.3f}, "
            f"prompt={agg['mean_prompt_torque_trigger_ratio']:.3f}"
        )
        print(f"终止原因统计: {agg['failure_modes']}")
        print("=" * 72)
        return agg

    def save_results(self, filepath: Path, metadata: Dict[str, Any], alignment: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "metadata": to_jsonable(metadata),
            "aggregate": to_jsonable(self.compute_aggregate()),
            "episodes": to_jsonable(self.episode_data),
        }
        if alignment is not None:
            payload["alignment"] = to_jsonable(alignment)
        filepath = Path(filepath)
        if filepath.parent != Path(""):
            filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {filepath}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="真实 MuJoCo sim2sim 验证（wheel_legged_vmc_balance）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="任务名")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="单个 checkpoint 路径（与 --checkpoint-list 二选一）",
    )
    parser.add_argument(
        "--checkpoint-list",
        type=str,
        default=None,
        help="文本文件：每行一个 checkpoint 路径（支持 # 注释）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_FIDELITY_MODEL,
        help="MuJoCo MJCF 模型路径（推荐 fidelity 模型）",
    )
    parser.add_argument("--episodes", type=int, default=10, help="每个 checkpoint 的 episode 数")
    parser.add_argument("--max_steps", type=int, default=3000, help="每个 episode 最大控制步数")
    parser.add_argument("--render", action="store_true", help="可视化（仅单 checkpoint 推荐）")
    parser.add_argument("--device", type=str, default="cpu", help="策略推理设备")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument(
        "--output",
        type=str,
        default="mujoco_sim2sim_results.json",
        help="结果 JSON 输出路径",
    )
    parser.add_argument("--label", type=str, default=None, help="实验标签")
    parser.add_argument(
        "--controller-mode",
        type=str,
        default=DEFAULT_CONTROLLER_MODE,
        choices=list(MuJoCoBalanceEnv.SUPPORTED_CONTROLLER_MODES),
        help="MuJoCo 控制模式",
    )
    parser.add_argument(
        "--mujoco-tuning-profile",
        type=str,
        default="exact_baseline",
        choices=["exact_baseline", "demo_tuned"],
        help="MuJoCo-only tuning profile (exact_baseline for real line, demo_tuned for demo line).",
    )
    parser.add_argument(
        "--domain-rand-mode",
        type=str,
        default=DEFAULT_DOMAIN_RAND_MODE,
        choices=list(MuJoCoBalanceEnv.SUPPORTED_DOMAIN_RAND_MODES),
        help="MuJoCo 域随机化模式",
    )
    parser.add_argument(
        "--reset-profile",
        type=str,
        default="default",
        choices=["default", "nominal_demo", "hard_random_balance", "nominal", "random_balance"],
        help="Reset profile for environment resets (default keeps existing behavior).",
    )
    parser.add_argument(
        "--eval-fall-tilt-deg",
        type=float,
        default=60.0,
        help="评估侧跌倒判定倾角阈值（度）；<=0 表示禁用脚本层提前终止",
    )
    parser.add_argument(
        "--compare-isaac",
        action="store_true",
        help="运行结束后执行 Isaac 参考轨迹的 MuJoCo 对齐回放（短序列工程对齐）",
    )
    parser.add_argument(
        "--reference-rollout",
        type=str,
        default=None,
        help="Isaac 参考轨迹 .npz 路径（--compare-isaac 时必填）",
    )

    rr_group = parser.add_mutually_exclusive_group()
    rr_group.add_argument(
        "--record-episode-params",
        dest="record_episode_params",
        action="store_true",
        help="在结果中记录每个 episode 的域随机化参数样本",
    )
    rr_group.add_argument(
        "--no-record-episode-params",
        dest="record_episode_params",
        action="store_false",
        help="不记录每个 episode 的随机化参数样本（减小结果文件体积）",
    )
    parser.set_defaults(record_episode_params=True)

    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--randomize-reset",
        dest="randomize_reset",
        action="store_true",
        help="启用 balance 风格随机初始化",
    )
    reset_group.add_argument(
        "--no-randomize-reset",
        dest="randomize_reset",
        action="store_false",
        help="关闭随机初始化（便于复现/调试）",
    )
    parser.set_defaults(randomize_reset=True)

    args = parser.parse_args()
    if args.task not in SUPPORTED_TASKS:
        parser.error(f"Unsupported task '{args.task}'. Expected one of {SUPPORTED_TASKS}.")
    if args.episodes <= 0:
        parser.error("--episodes must be > 0")
    if args.max_steps <= 0:
        parser.error("--max_steps must be > 0")
    if args.checkpoint_list and args.render:
        parser.error("--render is only supported for single checkpoint runs (no --checkpoint-list)")
    if args.checkpoint_list and args.compare_isaac:
        parser.error("--compare-isaac currently supports single checkpoint only")
    if args.compare_isaac and not args.reference_rollout:
        parser.error("--compare-isaac requires --reference-rollout")
    return args


def load_checkpoint_list(list_path: Path) -> List[Path]:
    lines = list_path.read_text(encoding="utf-8").splitlines()
    checkpoints: List[Path] = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        checkpoints.append(resolve_path(s))
    if not checkpoints:
        raise ValueError(f"No checkpoint entries found in list file: {list_path}")
    return checkpoints


def print_config_summary(args: argparse.Namespace, model_path: Path, checkpoints: List[Path]) -> None:
    print("\n" + "=" * 72)
    print("MuJoCo Sim2Sim 验证配置")
    print("=" * 72)
    print(f"task:                {args.task}")
    print(f"model:               {model_path}")
    print(f"episodes/checkpoint: {args.episodes}")
    print(f"max_steps:           {args.max_steps}")
    print(f"device:              {args.device}")
    print(f"seed:                {args.seed}")
    print(f"randomize_reset:     {args.randomize_reset}")
    print(f"domain_rand_mode:    {args.domain_rand_mode}")
    print(f"controller_mode:     {args.controller_mode}")
    print(f"mujoco_tuning:       {args.mujoco_tuning_profile}")
    print(f"reset_profile:       {args.reset_profile}")
    print(f"eval_fall_tilt_deg:  {args.eval_fall_tilt_deg}")
    print(f"render:              {args.render}")
    print(f"output:              {args.output}")
    print(f"label:               {args.label}")
    print(f"checkpoint_count:    {len(checkpoints)}")
    if len(checkpoints) == 1:
        print(f"checkpoint:          {checkpoints[0]}")
    else:
        print(f"checkpoint_list:     {args.checkpoint_list}")
        print("first checkpoints:")
        for cp in checkpoints[:5]:
            print(f"  - {cp}")
    print("=" * 72)

    is_simple = "simple" in model_path.name.lower()
    if args.controller_mode == "simplified_joint_pd":
        print(
            "[NOTE] 当前为近似验证：MuJoCo 使用 simplified_joint_pd，"
            "与训练中的 VMC 控制链路不一致。"
        )
    elif is_simple:
        print(
            "[NOTE] 控制链路已使用 vmc_balance_exact，但 MJCF 仍是 simple 模型；"
            "结果可用于对比，不宜视为高保真最终结论。"
        )
    else:
        print(
            "[NOTE] 已启用 vmc_balance_exact + fidelity MJCF。仍存在 PhysX/MuJoCo 接触模型差异，"
            "建议同时查看 Isaac 对齐指标和长时 rollout 指标。"
        )


def run_preflight_checks(
    env: MuJoCoBalanceEnv,
    policy: PolicyLoader,
    *,
    randomize_reset: bool,
    reset_profile: str,
    seed_diag: Dict[str, Any],
) -> Dict[str, Any]:
    print("\n开始 Preflight 检查...")
    diag = env.get_model_diagnostics()
    errors: List[str] = []
    expected_obs_shape = (int(policy.spec.num_obs),)
    expected_action_shape = (int(policy.spec.num_actions),)

    if diag["joint_names"] != EXPECTED_JOINT_NAMES:
        errors.append(f"Joint names mismatch: expected {EXPECTED_JOINT_NAMES}, got {diag['joint_names']}")
    if diag["actuator_names"] != EXPECTED_ACTUATOR_NAMES:
        errors.append(
            f"Actuator names mismatch: expected {EXPECTED_ACTUATOR_NAMES}, got {diag['actuator_names']}"
        )
    if diag["ctrl_size"] != 6:
        errors.append(f"Actuator count mismatch: expected 6, got {diag['ctrl_size']}")
    if diag["joint_qpos_addrs"] != sorted(diag["joint_qpos_addrs"]):
        errors.append(f"Joint qpos address order mismatch: {diag['joint_qpos_addrs']}")
    if diag["joint_dof_addrs"] != sorted(diag["joint_dof_addrs"]):
        errors.append(f"Joint dof address order mismatch: {diag['joint_dof_addrs']}")
    if diag["actuator_ids"] != sorted(diag["actuator_ids"]):
        errors.append(f"Actuator id order mismatch: {diag['actuator_ids']}")

    if not np.isclose(diag["dt"], 0.01):
        errors.append(f"Control dt mismatch: expected 0.01, got {diag['dt']}")
    if not np.isclose(diag["sim_dt"], 0.005):
        errors.append(f"Physics dt mismatch: expected 0.005, got {diag['sim_dt']}")
    if int(diag["decimation"]) != 2:
        errors.append(f"Decimation mismatch: expected 2, got {diag['decimation']}")

    if env.seed is None or int(env.seed) != int(seed_diag["seed"]):
        errors.append(f"Environment seed mismatch: env.seed={env.seed}, expected={seed_diag['seed']}")
    if int(seed_diag["torch_initial_seed"]) != int(seed_diag["seed"]):
        errors.append(
            "Torch seed check failed: "
            f"torch_initial_seed={seed_diag['torch_initial_seed']} expected={seed_diag['seed']}"
        )
    if int(env.action_dim) != int(policy.spec.num_actions):
        errors.append(
            "Env action_dim mismatch with policy num_actions: "
            f"env={env.action_dim}, policy={policy.spec.num_actions}"
        )

    obs = env.reset(randomize=randomize_reset, domain_randomize=False, reset_profile=reset_profile)
    if obs.shape != expected_obs_shape:
        errors.append(
            f"Observation shape mismatch: expected {expected_obs_shape}, got {obs.shape}"
        )
    if not np.all(np.isfinite(obs)):
        errors.append("Observation contains NaN/Inf during preflight")

    policy.reset(obs)
    action = policy.get_action(obs)
    if action.shape != expected_action_shape:
        errors.append(
            f"Action shape mismatch: expected {expected_action_shape}, got {action.shape}"
        )
    if not np.all(np.isfinite(action)):
        errors.append("Action contains NaN/Inf during preflight")

    obs2, rew2, done2, info2 = env.step(action)
    if obs2.shape != expected_obs_shape:
        errors.append(
            f"Post-step observation shape mismatch: expected {expected_obs_shape}, got {obs2.shape}"
        )
    if not np.isfinite(float(rew2)):
        errors.append("Reward is NaN/Inf during preflight step")
    if not np.all(np.isfinite(env.last_ctrl)):
        errors.append("Control output contains NaN/Inf during preflight")

    diag = env.get_model_diagnostics()
    print("Preflight 诊断:")
    print(f"  joint_qpos_addrs: {diag['joint_qpos_addrs']}")
    print(f"  joint_dof_addrs:  {diag['joint_dof_addrs']}")
    print(f"  actuator_ids:     {diag['actuator_ids']}")
    print(f"  obs_shape:        {obs.shape}")
    print(f"  action_shape:     {action.shape}")
    print(f"  ctrl_shape:       {env.last_ctrl.shape}")
    print(
        f"  timing:           dt={diag['dt']}, sim_dt={diag['sim_dt']}, decimation={diag['decimation']}"
    )
    print(
        "  seed_check:       "
        f"numpy_probe={seed_diag['numpy_probe']:.6f}, torch_probe={seed_diag['torch_probe']:.6f}"
    )
    print(
        f"  controller/fidelity/domain: {diag['implemented_controller_mode']} / "
        f"{diag['fidelity_level']} / {diag['domain_rand_mode']}"
    )
    print(
        f"  reset/tuning:      {diag.get('current_reset_profile')} / "
        f"{diag.get('mujoco_tuning_profile')}"
    )
    print(
        "  wheel_collision:  "
        f"mode={diag.get('wheel_collision_mode_detected')} "
        f"(mesh={diag.get('wheel_collision_mesh_count')}, "
        f"cylinder={diag.get('wheel_collision_cylinder_count')})"
    )
    print(
        "  wheel_mesh_src:   "
        f"{diag.get('wheel_collision_mesh_asset_source_detected')} "
        f"{diag.get('wheel_collision_mesh_asset_names', [])}"
    )
    if diag.get("wheel_collision_geom_names"):
        print(f"  wheel_geoms:      {diag['wheel_collision_geom_names']}")

    if (
        diag.get("fidelity_level") == "fidelity"
        and diag.get("implemented_controller_mode") == "vmc_balance_exact"
        and diag.get("wheel_collision_mode_detected") in {"cylinder", "mixed"}
    ):
        print(
            "[WARN] fidelity MJCF 检测到轮子碰撞并非纯 mesh（cylinder/mixed）。"
            "结果可能包含碰撞体简化带来的偏差。"
        )

    if errors:
        raise RuntimeError("Preflight checks failed:\n" + "\n".join(f"- {e}" for e in errors))

    print("Preflight 检查通过。")
    return {
        "joint_qpos_addrs": diag["joint_qpos_addrs"],
        "joint_dof_addrs": diag["joint_dof_addrs"],
        "actuator_ids": diag["actuator_ids"],
        "obs_shape": list(obs.shape),
        "action_shape": list(action.shape),
        "ctrl_shape": list(env.last_ctrl.shape),
        "timing": {"dt": diag["dt"], "sim_dt": diag["sim_dt"], "decimation": diag["decimation"]},
        "seed_check": seed_diag,
        "env_diagnostics": to_jsonable(diag),
        "reset_profile": str(reset_profile),
        "collision_representation": {
            "wheel_collision_mode_detected": diag.get("wheel_collision_mode_detected"),
            "wheel_collision_geom_names": diag.get("wheel_collision_geom_names"),
            "wheel_collision_geom_types": diag.get("wheel_collision_geom_types"),
            "wheel_collision_mesh_count": diag.get("wheel_collision_mesh_count"),
            "wheel_collision_cylinder_count": diag.get("wheel_collision_cylinder_count"),
            "wheel_collision_mesh_asset_names": diag.get("wheel_collision_mesh_asset_names"),
            "wheel_collision_mesh_asset_sources": diag.get("wheel_collision_mesh_asset_sources"),
            "wheel_collision_mesh_asset_source_detected": diag.get(
                "wheel_collision_mesh_asset_source_detected"
            ),
        },
        "preflight_step": {
            "reward": float(rew2),
            "done": bool(done2),
            "termination_reason": info2.get("termination_reason"),
        },
    }


def build_metadata(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    model_path: Path,
    env: MuJoCoBalanceEnv,
    policy: PolicyLoader,
    preflight_info: Dict[str, Any],
) -> Dict[str, Any]:
    approximate_validation = bool(
        env.implemented_controller_mode != "vmc_balance_exact" or env.fidelity_level != "fidelity"
    )
    notes = []
    if env.implemented_controller_mode == "vmc_balance_exact":
        notes.append("MuJoCo controller uses training-aligned VMC balance control path.")
    else:
        notes.append("MuJoCo controller uses simplified_joint_pd baseline (approximate).")
    if env.fidelity_level == "fidelity":
        collision_info = preflight_info.get("collision_representation", {})
        detected = collision_info.get("wheel_collision_mode_detected", "unknown")
        notes.append(
            "MJCF uses fidelity model generated from URDF "
            f"(wheel collision mode detected: {detected})."
        )
    else:
        notes.append("MJCF uses simplified model; dynamics/contact fidelity is limited.")
    notes.append("PhysX and MuJoCo contact dynamics differ; use alignment metrics and rollout metrics jointly.")

    code_paths = [
        "wheel_legged_gym/scripts/verify_sim2sim_mujoco.py",
        "mujoco_sim/mujoco_balance_env.py",
        "mujoco_sim/observation_computer.py",
        "mujoco_sim/policy_loader.py",
        "mujoco_sim/domain_randomizer.py",
        "mujoco_sim/control_config.py",
        "mujoco_sim/vmc_kinematics.py",
        str(Path(args.model)),
    ]

    metadata: Dict[str, Any] = {
        "task": args.task,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "model_path": str(model_path.resolve()),
        "controller_mode": env.implemented_controller_mode,
        "controller_mode_requested": args.controller_mode,
        "mujoco_tuning_profile": args.mujoco_tuning_profile,
        "fidelity_level": env.fidelity_level,
        "domain_rand_mode": args.domain_rand_mode,
        "reset_profile": args.reset_profile,
        "episodes_requested": int(args.episodes),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "randomize_reset": bool(args.randomize_reset),
        "device": str(args.device),
        "render": bool(args.render),
        "wait_mode": "n/a",
        "record_episode_params": bool(args.record_episode_params),
        "eval_fall_tilt_deg": float(args.eval_fall_tilt_deg),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "approximate_validation": approximate_validation,
        "notes": notes,
        "policy_checkpoint_iter": policy.checkpoint_iter,
        "policy_spec": policy.spec.to_dict(),
        "env_timing": {
            "dt": float(env.dt),
            "sim_dt": float(env.sim_dt),
            "decimation": int(env.decimation),
        },
        "config_snapshot": {
            "control_cfg": env.cfg.to_dict() if hasattr(env.cfg, "to_dict") else to_jsonable(env.cfg),
            "domain_rand_cfg": env.domain_randomizer.config_dict(),
        },
        "collision_representation": preflight_info.get("collision_representation"),
        "code_paths": code_paths,
        "preflight": preflight_info,
    }
    if args.label:
        metadata["label"] = args.label
    if args.compare_isaac and args.reference_rollout:
        metadata["reference_rollout"] = str(resolve_path(args.reference_rollout).resolve())
    return metadata


def evaluate_single_checkpoint(
    *,
    args: argparse.Namespace,
    checkpoint_path: Path,
    model_path: Path,
    seed_diag: Dict[str, Any],
) -> Dict[str, Any]:
    env: Optional[MuJoCoBalanceEnv] = None
    policy: Optional[PolicyLoader] = None
    try:
        env = MuJoCoBalanceEnv(
            str(model_path),
            render=args.render,
            seed=args.seed,
            controller_mode=args.controller_mode,
            domain_rand_mode=args.domain_rand_mode,
            mujoco_tuning_profile=args.mujoco_tuning_profile,
            task=args.task,
        )
        env.max_episode_steps = int(args.max_steps)

        policy = PolicyLoader(
            str(checkpoint_path),
            device=args.device,
            policy_spec=None,
            task=args.task,
        )

        preflight_info = run_preflight_checks(
            env=env,
            policy=policy,
            randomize_reset=args.randomize_reset,
            reset_profile=args.reset_profile,
            seed_diag=seed_diag,
        )

        # Re-seed after preflight to make episode randomization sequence reproducible.
        set_global_seed(args.seed, args.device)
        env.np_random = np.random.default_rng(args.seed)
        policy.reset(obs)

        metrics = RolloutMetrics(control_dt=env.dt)
        if args.render:
            env.render()

        print(f"\n开始运行 {args.episodes} 个 episodes（checkpoint={checkpoint_path.name}）...")
        print("-" * 72)

        for episode in range(args.episodes):
            obs = env.reset(
                randomize=args.randomize_reset,
                domain_randomize=(args.domain_rand_mode != "off"),
                reset_profile=args.reset_profile,
            )
            policy.reset(obs)

            episode_reward = 0.0
            episode_steps = 0
            termination_reason = "max_steps"
            terminated = False

            for _step in range(args.max_steps):
                action = policy.get_action(obs)
                obs, reward, done, info = env.step(action)

                metrics.update(env)
                episode_reward += float(reward)
                episode_steps += 1

                if not np.all(np.isfinite(obs)) or not np.isfinite(float(reward)):
                    termination_reason = "nan_detected"
                    terminated = True
                    break

                if args.eval_fall_tilt_deg > 0:
                    tilt_deg = float(
                        np.degrees(np.arccos(np.clip(-float(env.projected_gravity[2]), -1.0, 1.0)))
                    )
                    if tilt_deg > float(args.eval_fall_tilt_deg):
                        termination_reason = "fall_tilt"
                        terminated = True
                        break

                if done:
                    termination_reason = str(info.get("termination_reason", "timeout"))
                    terminated = True
                    break

            if not terminated and episode_steps >= args.max_steps:
                termination_reason = "max_steps"

            metrics.log_episode(
                episode_num=episode,
                steps=episode_steps,
                reward=episode_reward,
                env=env,
                termination_reason=termination_reason,
                record_episode_params=args.record_episode_params,
            )
            last_ep = metrics.episode_data[-1]
            print(
                f"Episode {episode + 1}/{args.episodes} | steps={episode_steps:4d} | "
                f"reward={episode_reward:8.3f} | upright={last_ep['upright_ratio'] * 100:6.2f}% | "
                f"term={termination_reason}"
            )

        aggregate = metrics.print_summary()
        metadata = build_metadata(
            args=args,
            checkpoint_path=checkpoint_path,
            model_path=model_path,
            env=env,
            policy=policy,
            preflight_info=preflight_info,
        )

        alignment = None
        if args.compare_isaac:
            ref_path = resolve_path(args.reference_rollout)
            print("\n开始 Isaac 参考轨迹 MuJoCo 对齐回放...")
            alignment = replay_reference_rollout_in_mujoco(
                reference_rollout_path=ref_path,
                model_path=model_path,
                controller_mode=args.controller_mode,
                domain_rand_mode="off",
                mujoco_tuning_profile=args.mujoco_tuning_profile,
                seed=args.seed,
            )
            print(
                "对齐摘要: "
                f"obs_rmse_total={alignment['summary']['obs_rmse_total']}, "
                f"torque_rmse_total={alignment['summary']['torque_rmse_total']}"
            )

        return {
            "metadata": metadata,
            "aggregate": aggregate,
            "episodes": deepcopy(metrics.episode_data),
            **({"alignment": alignment} if alignment is not None else {}),
        }
    finally:
        if env is not None:
            env.close()


def save_payload(payload: Dict[str, Any], output_path: Path) -> None:
    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_path}")


def run_checkpoint_sweep(
    *,
    args: argparse.Namespace,
    checkpoints: List[Path],
    model_path: Path,
    seed_diag: Dict[str, Any],
    output_path: Path,
) -> None:
    runs: List[Dict[str, Any]] = []
    for idx, cp in enumerate(checkpoints):
        print("\n" + "#" * 72)
        print(f"Checkpoint {idx + 1}/{len(checkpoints)}: {cp}")
        print("#" * 72)
        result = evaluate_single_checkpoint(
            args=args,
            checkpoint_path=cp,
            model_path=model_path,
            seed_diag=seed_diag,
        )
        runs.append(result)

    def key_metric(run: Dict[str, Any], key: str) -> float:
        v = run.get("aggregate", {}).get(key)
        return float("-inf") if v is None else float(v)

    comparison = []
    for run in runs:
        meta = run["metadata"]
        agg = run["aggregate"]
        comparison.append(
            {
                "checkpoint_path": meta["checkpoint_path"],
                "checkpoint_iter": meta.get("policy_checkpoint_iter"),
                "mean_steps": agg.get("mean_steps"),
                "survival_time_seconds": agg.get("survival_time_seconds"),
                "mean_upright_ratio": agg.get("mean_upright_ratio"),
                "mean_tilt": agg.get("mean_tilt"),
                "fall_rate": agg.get("fall_rate"),
            }
        )

    comparison_sorted = sorted(
        comparison,
        key=lambda r: (
            -1 if r["mean_upright_ratio"] is None else -float(r["mean_upright_ratio"]),
            float("inf") if r["mean_tilt"] is None else float(r["mean_tilt"]),
        ),
    )

    print("\n" + "=" * 72)
    print("Checkpoint Sweep 汇总（按 mean_upright_ratio 排序）")
    print("=" * 72)
    for i, row in enumerate(comparison_sorted, 1):
        cp_name = Path(row["checkpoint_path"]).name
        print(
            f"{i:2d}. {cp_name:<20} iter={row['checkpoint_iter']:<8} "
            f"upright={row['mean_upright_ratio'] if row['mean_upright_ratio'] is not None else 'None'} "
            f"tilt={row['mean_tilt'] if row['mean_tilt'] is not None else 'None'} "
            f"fall_rate={row['fall_rate'] if row['fall_rate'] is not None else 'None'}"
        )
    print("=" * 72)

    sweep_payload = {
        "metadata": {
            "mode": "checkpoint_sweep",
            "task": args.task,
            "model_path": str(model_path.resolve()),
            "controller_mode": args.controller_mode,
            "domain_rand_mode": args.domain_rand_mode,
            "episodes_per_checkpoint": int(args.episodes),
            "max_steps": int(args.max_steps),
            "seed": int(args.seed),
            "randomize_reset": bool(args.randomize_reset),
            "device": str(args.device),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "checkpoint_count": len(checkpoints),
            "checkpoint_list_path": str(resolve_path(args.checkpoint_list).resolve()) if args.checkpoint_list else None,
            "seed_check": seed_diag,
        },
        "comparison": comparison_sorted,
        "runs": runs,
    }
    save_payload(sweep_payload, output_path)


def main() -> None:
    args = parse_args()

    model_path = resolve_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")

    if args.checkpoint_list:
        ckpt_list_path = resolve_path(args.checkpoint_list)
        if not ckpt_list_path.exists():
            raise FileNotFoundError(f"Checkpoint list file not found: {ckpt_list_path}")
        checkpoints = load_checkpoint_list(ckpt_list_path)
    else:
        checkpoints = [resolve_path(args.checkpoint)]

    missing = [cp for cp in checkpoints if not cp.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n" + "\n".join(str(p) for p in missing))

    set_global_seed(args.seed, args.device)
    seed_diag = verify_seed_application(args.seed)

    print_config_summary(args, model_path, checkpoints)
    output_path = Path(args.output).expanduser()

    if len(checkpoints) == 1:
        result = evaluate_single_checkpoint(
            args=args,
            checkpoint_path=checkpoints[0],
            model_path=model_path,
            seed_diag=seed_diag,
        )
        save_payload(result, output_path)
        print("\n验证完成！")
        return

    run_checkpoint_sweep(
        args=args,
        checkpoints=checkpoints,
        model_path=model_path,
        seed_diag=seed_diag,
        output_path=output_path,
    )
    print("\nCheckpoint sweep 完成！")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
