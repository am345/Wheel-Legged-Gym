#!/usr/bin/env python3
"""MuJoCo interactive/headless policy runner for wheel_legged_fzqver."""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv
from mujoco_sim.policy_loader import PolicyLoader


SUPPORTED_TASKS = ("wheel_legged_fzqver", "wheel_legged_fzqver_comp8")
DEFAULT_TASK = "wheel_legged_fzqver"
DEFAULT_MODEL = "resources/robots/serialleg/mjcf/serialleg_fidelity.xml"
DEFAULT_CHECKPOINT_ROOT_BY_TASK = {
    "wheel_legged_fzqver": "logs/wheel_legged_fzqver",
    "wheel_legged_fzqver_comp8": "logs/wheel_legged_fzqver_comp8",
}
ESC_KEY = 256  # GLFW_KEY_ESCAPE


def _find_latest_checkpoint(checkpoint_root: Path) -> Tuple[Path, Path]:
    if not checkpoint_root.exists():
        raise FileNotFoundError(
            f"Checkpoint root not found: {checkpoint_root}. "
            "Pass --checkpoint explicitly or verify logs path."
        )
    if not checkpoint_root.is_dir():
        raise NotADirectoryError(f"Checkpoint root is not a directory: {checkpoint_root}")

    runs = [p for p in checkpoint_root.iterdir() if p.is_dir() and p.name != "exported"]
    if not runs:
        raise FileNotFoundError(
            f"No run directories found under {checkpoint_root}. Pass --checkpoint explicitly."
        )
    runs.sort(key=lambda p: p.name)
    run_dir = runs[-1]

    model_files = []
    for p in run_dir.glob("model_*.pt"):
        m = re.match(r"model_(\d+)\.pt$", p.name)
        if m:
            model_files.append((int(m.group(1)), p))
    if not model_files:
        raise FileNotFoundError(
            f"No model_*.pt checkpoints found in latest run: {run_dir}. "
            "Pass --checkpoint explicitly."
        )
    model_files.sort(key=lambda x: x[0])
    return run_dir, model_files[-1][1]


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        ckpt = Path(args.checkpoint).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"--checkpoint path does not exist: {ckpt}")
        print(f"[Checkpoint] Using explicit checkpoint: {ckpt}")
        return ckpt

    checkpoint_root = args.checkpoint_root
    if checkpoint_root is None:
        checkpoint_root = DEFAULT_CHECKPOINT_ROOT_BY_TASK.get(
            str(args.task),
            DEFAULT_CHECKPOINT_ROOT_BY_TASK[DEFAULT_TASK],
        )

    root = Path(checkpoint_root).expanduser().resolve()
    run_dir, ckpt = _find_latest_checkpoint(root)
    print(f"[Checkpoint] Auto-discovered latest run: {run_dir}")
    print(f"[Checkpoint] Auto-discovered latest checkpoint: {ckpt}")
    return ckpt


def _set_gravity(env: MuJoCoBalanceEnv, gravity_on: bool) -> None:
    if gravity_on:
        env.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    else:
        env.model.opt.gravity[:] = 0.0


def _compute_tilt_deg(projected_gravity: np.ndarray) -> float:
    proj_g = np.asarray(projected_gravity, dtype=np.float64).reshape(3)
    return float(np.degrees(np.arccos(np.clip(-proj_g[2], -1.0, 1.0))))


def _step_wait_zero_torque(
    env: MuJoCoBalanceEnv,
    obs: np.ndarray,
) -> Tuple[np.ndarray, float, Dict[str, object], bool]:
    env.current_action_obs[:] = 0.0
    env.prompt_torque_triggered_last_step = False
    env.last_ctrl[:] = 0.0
    env.last_applied_action[:] = 0.0
    env.last_control_debug = {
        "mode": "wait_zero_torque",
        "torques_pre_clip": [0.0] * 6,
        "torques_post_clip": [0.0] * 6,
    }

    for _ in range(int(env.decimation)):
        env.data.ctrl[:] = 0.0
        mujoco.mj_step(env.model, env.data)
        if env.render_enabled and env.viewer is not None:
            env.viewer.sync()

    env._refresh_state_from_sim(init_from_current=False)
    env._maybe_resample_commands()
    obs = env._compute_observation(env.current_action_obs)
    reward = env._compute_reward()
    info = {
        "episode_steps": int(env.episode_steps),
        "base_height": float(env.data.qpos[2]),
        "controller_mode": "wait_zero_torque",
        "domain_params": env.current_domain_params,
        "termination_reason": "waiting_for_start",
        "prompt_torque_triggered": False,
        "pitch_angle": float(env.pitch_angle),
        "task": env.task,
        "current_commands": env.current_commands.copy().tolist(),
        "command_resample_step": int(env.last_command_resample_step),
    }
    env.last_reward = float(reward)
    env.last_done = False
    env.last_info = info
    return obs, float(reward), info, False


def _step_wait_zero_action(
    env: MuJoCoBalanceEnv,
    obs: np.ndarray,
) -> Tuple[np.ndarray, float, Dict[str, object], bool]:
    obs, reward, done, info = env.step(np.zeros(env.action_dim, dtype=np.float32))
    if done:
        info = dict(info)
        info["termination_reason"] = "waiting_for_start"
    return obs, float(reward), info, bool(done)


def _step_wait_freeze(
    env: MuJoCoBalanceEnv,
    obs: np.ndarray,
) -> Tuple[np.ndarray, float, Dict[str, object], bool]:
    if env.render_enabled and env.viewer is not None:
        env.viewer.sync()
    reward = float(getattr(env, "last_reward", 0.0))
    info = {
        "episode_steps": int(env.episode_steps),
        "base_height": float(env.data.qpos[2]),
        "controller_mode": "wait_freeze",
        "domain_params": env.current_domain_params,
        "termination_reason": "waiting_for_start",
        "prompt_torque_triggered": False,
        "pitch_angle": float(env.pitch_angle),
        "task": env.task,
        "current_commands": env.current_commands.copy().tolist(),
        "command_resample_step": int(env.last_command_resample_step),
    }
    return obs, reward, info, False


@dataclass
class PolicyKeyboardState:
    started: bool = False
    paused: bool = False
    show_stats: bool = True
    gravity_on: bool = True
    exit_requested: bool = False
    reset_requested: bool = False
    debug_requested: bool = False
    start_requested: bool = False


class KeyboardController:
    """Thread-safe keyboard state for MuJoCo viewer key_callback."""

    def __init__(self, *, gravity_on: bool, start_enabled: bool, show_stats: bool = True):
        self._lock = threading.Lock()
        self._state = PolicyKeyboardState(
            started=bool(start_enabled),
            paused=False,
            show_stats=bool(show_stats),
            gravity_on=bool(gravity_on),
            start_requested=bool(start_enabled),
        )

    def force_stop(self, reason: str = "") -> None:
        with self._lock:
            self._state.started = False
            self._state.start_requested = False
        if reason:
            print(f"[STATE] Policy control disabled ({reason})")
        else:
            print("[STATE] Policy control disabled")

    def mark_reset_complete(self) -> None:
        with self._lock:
            self._state.started = False
            self._state.start_requested = False

    def request_start(self) -> None:
        with self._lock:
            self._state.started = True
            self._state.start_requested = True

    def key_callback(self, keycode: int) -> None:
        with self._lock:
            if int(keycode) == ESC_KEY:
                self._state.exit_requested = True
                print("[KEY] EXIT requested")
                return
            if not (0 <= int(keycode) < 256):
                return
            ch = chr(int(keycode)).lower()
            self._apply_key(ch)

    def _apply_key(self, ch: str) -> None:
        s = self._state
        if ch == "c":
            if not s.started:
                s.started = True
                s.start_requested = True
                print("[KEY] START requested (policy will be enabled)")
            else:
                print("[KEY] Policy already enabled")
            return
        if ch == "r":
            s.reset_requested = True
            s.started = False
            s.start_requested = False
            print("[KEY] RESET requested (policy disabled)")
            return
        if ch == "p":
            s.paused = not s.paused
            print(f"[KEY] {'PAUSED' if s.paused else 'RESUMED'}")
            return
        if ch == "s":
            s.show_stats = not s.show_stats
            print(f"[KEY] stats {'ON' if s.show_stats else 'OFF'}")
            return
        if ch == "g":
            s.gravity_on = not s.gravity_on
            print(f"[KEY] gravity {'ON' if s.gravity_on else 'OFF'}")
            return
        if ch == "n":
            s.debug_requested = True
            return
        if ch == "h":
            _print_help()
            return

    def consume(self) -> Dict[str, object]:
        with self._lock:
            out = {
                "started": bool(self._state.started),
                "paused": bool(self._state.paused),
                "show_stats": bool(self._state.show_stats),
                "gravity_on": bool(self._state.gravity_on),
                "exit_requested": bool(self._state.exit_requested),
                "reset_requested": bool(self._state.reset_requested),
                "debug_requested": bool(self._state.debug_requested),
                "start_requested": bool(self._state.start_requested),
            }
            self._state.reset_requested = False
            self._state.debug_requested = False
            self._state.start_requested = False
            return out


@dataclass
class RunStats:
    rewards: List[float]
    tilts_deg: List[float]
    heights: List[float]
    commands: List[List[float]]
    started_flags: List[bool]
    term_reasons: List[str]
    steps: int = 0
    policy_steps: int = 0
    nan_detected: bool = False


def _new_stats() -> RunStats:
    return RunStats(
        rewards=[],
        tilts_deg=[],
        heights=[],
        commands=[],
        started_flags=[],
        term_reasons=[],
        steps=0,
        policy_steps=0,
        nan_detected=False,
    )


def _record_stats(stats: RunStats, env: MuJoCoBalanceEnv, reward: float, started: bool) -> None:
    stats.steps += 1
    if started:
        stats.policy_steps += 1
    stats.rewards.append(float(reward))
    stats.heights.append(float(env.data.qpos[2]))
    stats.tilts_deg.append(_compute_tilt_deg(np.asarray(env.projected_gravity, dtype=np.float64)))
    stats.commands.append(np.asarray(env.current_commands, dtype=np.float64).tolist())
    stats.started_flags.append(bool(started))


def _summarize_stats(stats: RunStats) -> Dict[str, object]:
    cmd = np.asarray(stats.commands, dtype=np.float64) if stats.commands else np.zeros((0, 3))
    rewards = np.asarray(stats.rewards, dtype=np.float64) if stats.rewards else np.zeros((0,))
    tilts = np.asarray(stats.tilts_deg, dtype=np.float64) if stats.tilts_deg else np.zeros((0,))
    heights = np.asarray(stats.heights, dtype=np.float64) if stats.heights else np.zeros((0,))
    stand_mask = (
        np.isclose(cmd[:, 0], 0.0) & np.isclose(cmd[:, 1], 0.0) & np.isclose(cmd[:, 2], 0.22, atol=1e-3)
    ) if cmd.size else np.zeros((0,), dtype=bool)

    def _stat(arr: np.ndarray) -> Dict[str, Optional[float]]:
        if arr.size == 0:
            return {"mean": None, "min": None, "max": None, "std": None}
        return {
            "mean": float(np.mean(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    return {
        "steps": int(stats.steps),
        "policy_steps": int(stats.policy_steps),
        "nan_detected": bool(stats.nan_detected),
        "reward": _stat(rewards),
        "tilt_deg": _stat(tilts),
        "height": _stat(heights),
        "commands": {
            "lin_vel_x": _stat(cmd[:, 0]) if cmd.size else _stat(np.zeros((0,))),
            "ang_vel_yaw": _stat(cmd[:, 1]) if cmd.size else _stat(np.zeros((0,))),
            "height": _stat(cmd[:, 2]) if cmd.size else _stat(np.zeros((0,))),
            "stand_ratio": float(np.mean(stand_mask)) if stand_mask.size else None,
        },
        "termination_reasons": stats.term_reasons,
    }


def _print_help() -> None:
    print("\nKeyboard controls:")
    print("  C   : Enable policy control (START)")
    print("  R   : Random/fixed reset (per CLI), disable policy")
    print("  P   : Pause/resume stepping")
    print("  S   : Toggle terminal stats")
    print("  G   : Toggle gravity on/off")
    print("  N   : Print detailed debug snapshot")
    print("  H   : Print this help")
    print("  ESC : Exit")
    print("")


def _print_banner(args: argparse.Namespace, checkpoint_path: Path, env: MuJoCoBalanceEnv) -> None:
    print("\n" + "=" * 88)
    print("MuJoCo Policy Fzqver Interactive Demo")
    print("=" * 88)
    print(f"task:                 {args.task}")
    print(f"checkpoint:           {checkpoint_path}")
    print(f"model:                {Path(args.model).expanduser().resolve()}")
    print(f"controller_mode:      {args.controller_mode}")
    print(f"domain_rand_mode:     {args.domain_rand_mode}")
    print(f"randomize_reset:      {args.randomize_reset}")
    print(f"domain_randomize_rst: {args.domain_randomize_reset}")
    print(f"gravity:              {args.gravity}")
    print(f"start_enabled:        {args.start_enabled}")
    print(f"start_on_reset:       {args.start_on_reset}")
    print(f"wait_mode:            {args.wait_mode}")
    print(f"fall_tilt_deg:        {args.eval_fall_tilt_deg}")
    print(f"no_script_fall_stop:  {args.no_script_fall_stop}")
    print(f"seed/device:          {args.seed} / {args.device}")
    print(f"fidelity_level:       {env.fidelity_level}")
    print(f"headless:             {args.headless}")
    print(f"json_output:          {args.json_output}")
    print("=" * 88)
    _print_help()
    print("Policy is loaded but disabled initially (unless --start-enabled). Press C to start.\n")


def _print_step_stats(
    *,
    total_steps: int,
    reward: float,
    info: Dict[str, object],
    dbg: Dict[str, object],
    state: Dict[str, object],
    termination_reason: Optional[str],
) -> None:
    tilt_deg = float(dbg.get("tilt_deg", _compute_tilt_deg(np.asarray(dbg["projected_gravity"], dtype=float))))
    current_commands = np.round(np.asarray(dbg.get("current_commands", [0.0, 0.0, 0.0]), dtype=float), 4).tolist()
    print(
        f"step={total_steps:05d} "
        f"reward={reward:+.4f} "
        f"started={state['started']} paused={state['paused']} gravity={'on' if state['gravity_on'] else 'off'} "
        f"pitch={dbg['pitch_angle']:+.3f} tilt={tilt_deg:6.2f}deg "
        f"cmd={current_commands} "
        f"ctrl={np.round(np.asarray(dbg['last_ctrl'], dtype=float), 3).tolist()} "
        f"contacts={dbg['contacts']} "
        f"term={termination_reason or info.get('termination_reason', 'running')}"
    )


def _print_debug_snapshot(
    env: MuJoCoBalanceEnv,
    *,
    total_steps: int,
    reward: float,
    info: Dict[str, object],
    state: Dict[str, object],
) -> None:
    dbg = env.get_debug_state()
    tilt_deg = float(dbg.get("tilt_deg", _compute_tilt_deg(np.asarray(dbg["projected_gravity"], dtype=float))))
    print("\n" + "-" * 88)
    print(f"Debug snapshot @ step {total_steps}")
    print(
        f"state={{started={state['started']}, paused={state['paused']}, gravity_on={state['gravity_on']}}} "
        f"reward={reward:+.6f} tilt={tilt_deg:.3f}deg"
    )
    print(f"info={info}")
    print(f"pitch_angle={dbg['pitch_angle']:+.6f}")
    print(f"current_commands={np.round(np.asarray(dbg.get('current_commands', []), dtype=float), 6).tolist()}")
    print(f"projected_gravity={np.round(np.asarray(dbg['projected_gravity'], dtype=float), 6).tolist()}")
    print(
        f"L0={np.round(np.asarray(dbg['L0'], dtype=float), 6).tolist()} "
        f"L0_dot={np.round(np.asarray(dbg['L0_dot'], dtype=float), 6).tolist()}"
    )
    print(
        f"theta0={np.round(np.asarray(dbg['theta0'], dtype=float), 6).tolist()} "
        f"theta0_dot={np.round(np.asarray(dbg['theta0_dot'], dtype=float), 6).tolist()}"
    )
    print(f"last_ctrl={np.round(np.asarray(dbg['last_ctrl'], dtype=float), 6).tolist()}")
    print(f"last_applied_action={np.round(np.asarray(dbg['last_applied_action'], dtype=float), 6).tolist()}")
    print(f"contacts={dbg['contacts']}")
    print(f"control_debug={dbg.get('control_debug')}")
    print("-" * 88 + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo interactive/headless policy demo for wheel_legged_fzqver tasks."
    )
    parser.add_argument(
        "--task",
        default=DEFAULT_TASK,
        choices=list(SUPPORTED_TASKS),
        help=f"Task name (default: {DEFAULT_TASK})",
    )
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (.pt).")
    parser.add_argument(
        "--checkpoint-root",
        default=None,
        help=(
            "Checkpoint root for auto-discovery. "
            "If omitted, uses task-specific default under logs/."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"MJCF path (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--controller-mode",
        default="vmc_balance_exact",
        choices=["vmc_balance_exact", "simplified_joint_pd"],
    )
    parser.add_argument(
        "--domain-rand-mode",
        default="off",
        choices=["off", "train_ranges"],
        help="Env DR mode. Recommended run order: off then train_ranges.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")

    reset_group = parser.add_mutually_exclusive_group()
    reset_group.add_argument(
        "--randomize-reset",
        dest="randomize_reset",
        action="store_true",
        help="Enable random reset (default).",
    )
    reset_group.add_argument(
        "--fixed-reset",
        dest="randomize_reset",
        action="store_false",
        help="Disable random reset (fixed initial state).",
    )
    parser.set_defaults(randomize_reset=True)

    parser.add_argument(
        "--domain-randomize-reset",
        action="store_true",
        help="Apply episode DR on reset (default off).",
    )
    parser.add_argument(
        "--wait-mode",
        default="zero_torque",
        choices=["zero_action", "zero_torque", "freeze"],
        help="Behavior before pressing C.",
    )
    parser.add_argument("--gravity", choices=["on", "off"], default="on")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument(
        "--eval-fall-tilt-deg",
        type=float,
        default=0.0,
        help="Script-level fall stop threshold in deg. <=0 disables this stop condition.",
    )
    parser.add_argument(
        "--no-script-fall-stop",
        action="store_true",
        help="Disable script-level fall_tilt termination (NaN/timeout checks remain).",
    )
    parser.add_argument("--auto-reset-on-done", action="store_true")
    parser.add_argument("--start-enabled", action="store_true")
    parser.add_argument(
        "--start-on-reset",
        action="store_true",
        help="Automatically request START after each reset.",
    )
    parser.add_argument("--max-steps", type=int, default=0, help="0 means unlimited in interactive mode.")
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument("--json-output", default=None, help="Optional JSON output path.")
    parser.add_argument("--show-left-ui", action="store_true", default=True)
    parser.add_argument("--hide-left-ui", dest="show_left_ui", action="store_false")
    parser.add_argument("--show-right-ui", action="store_true", default=True)
    parser.add_argument("--hide-right-ui", dest="show_right_ui", action="store_false")
    return parser.parse_args()


def _dump_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_loop(
    *,
    args: argparse.Namespace,
    env: MuJoCoBalanceEnv,
    policy: PolicyLoader,
    kb: Optional[KeyboardController],
    stats: RunStats,
) -> Tuple[int, float, Dict[str, object], Optional[str]]:
    total_steps = 0
    last_reward = 0.0
    last_info: Dict[str, object] = {}
    episode_done = False
    episode_term_reason: Optional[str] = None

    started = bool(args.start_enabled)
    paused = False
    gravity_on = args.gravity == "on"
    show_stats = True

    if bool(args.start_on_reset) and not started:
        started = True

    while True:
        if kb is not None:
            state = kb.consume()
            if bool(state["exit_requested"]):
                print("[INFO] Exit requested by keyboard.")
                break
            started = bool(state["started"])
            paused = bool(state["paused"])
            gravity_on = bool(state["gravity_on"])
            show_stats = bool(state["show_stats"])
            if bool(state["debug_requested"]):
                _print_debug_snapshot(
                    env,
                    total_steps=total_steps,
                    reward=last_reward,
                    info=last_info,
                    state=state,
                )
            reset_requested = bool(state["reset_requested"])
            start_requested = bool(state["start_requested"])
        else:
            state = {
                "started": started,
                "paused": paused,
                "gravity_on": gravity_on,
                "show_stats": show_stats,
            }
            reset_requested = False
            start_requested = started

        _set_gravity(env, gravity_on)

        if reset_requested:
            obs = env.reset(
                randomize=bool(args.randomize_reset),
                domain_randomize=bool(args.domain_randomize_reset),
            )
            policy.reset(initial_obs=obs, history_init="repeat_obs")
            if kb is not None:
                kb.mark_reset_complete()
                if bool(args.start_on_reset):
                    kb.request_start()
            started = bool(args.start_on_reset)
            episode_done = False
            episode_term_reason = None
            print(
                "[INFO] Environment reset complete "
                f"(randomize_reset={args.randomize_reset}, domain_randomize_reset={args.domain_randomize_reset})."
            )
            if not np.all(np.isfinite(obs)):
                print("[WARN] Observation contains NaN/Inf immediately after reset.")

        if start_requested and episode_done:
            if kb is not None:
                kb.force_stop("episode_done_waiting_reset")
            started = False
            print("[WARN] Episode already terminated. Press R to reset before pressing C.")

        if paused:
            if env.viewer is not None:
                env.wait_mode = "paused"
                env.viewer.sync()
            if float(args.sleep) > 0:
                time.sleep(max(float(args.sleep), 0.01))
            if args.headless and int(args.max_steps) > 0 and total_steps >= int(args.max_steps):
                break
            continue

        if total_steps == 0 and "obs" not in locals():
            obs = env.reset(
                randomize=bool(args.randomize_reset),
                domain_randomize=bool(args.domain_randomize_reset),
            )
            policy.reset(initial_obs=obs, history_init="repeat_obs")

        term_reason: Optional[str] = None
        done_env = False

        if started and not episode_done:
            env.wait_mode = "policy"
            action = policy.get_action(obs)
            if not np.all(np.isfinite(action)):
                episode_done = True
                episode_term_reason = "nan_detected_action"
                stats.nan_detected = True
                if kb is not None:
                    kb.force_stop(episode_term_reason)
                started = False
                print("[ERROR] Policy action contains NaN/Inf.")
                if args.headless:
                    break
                continue

            obs, reward, done_env, info = env.step(action)
            last_reward = float(reward)
            last_info = info
            total_steps += 1
            dbg = env.get_debug_state()
            _record_stats(stats, env, last_reward, started=True)

            if (not np.all(np.isfinite(obs))) or (not np.isfinite(last_reward)) or (
                not np.all(np.isfinite(env.last_ctrl))
            ):
                term_reason = "nan_detected"
                stats.nan_detected = True
            else:
                tilt_deg = _compute_tilt_deg(np.asarray(env.projected_gravity, dtype=float))
                if (
                    (not bool(args.no_script_fall_stop))
                    and float(args.eval_fall_tilt_deg) > 0
                    and tilt_deg > float(args.eval_fall_tilt_deg)
                ):
                    term_reason = "fall_tilt"
                elif bool(done_env):
                    term_reason = str(info.get("termination_reason", "timeout"))
        else:
            if str(args.wait_mode) == "zero_action":
                env.wait_mode = "zero_action"
                obs, reward, info, done_wait = _step_wait_zero_action(env, obs)
            elif str(args.wait_mode) == "zero_torque":
                env.wait_mode = "zero_torque"
                obs, reward, info, done_wait = _step_wait_zero_torque(env, obs)
            else:
                env.wait_mode = "freeze"
                obs, reward, info, done_wait = _step_wait_freeze(env, obs)

            last_reward = float(reward)
            last_info = info
            total_steps += 1
            dbg = env.get_debug_state()
            _record_stats(stats, env, last_reward, started=False)
            if done_wait and not args.headless:
                print("[INFO] Wait mode reached env done; press R to reset.")

        if show_stats and int(args.print_every) > 0 and (total_steps % int(args.print_every) == 0):
            _print_step_stats(
                total_steps=total_steps,
                reward=last_reward,
                info=last_info,
                dbg=dbg,
                state=state,
                termination_reason=term_reason,
            )

        if term_reason is not None:
            episode_done = True
            episode_term_reason = term_reason
            stats.term_reasons.append(str(term_reason))
            if kb is not None:
                kb.force_stop(term_reason)
            started = False
            print(
                f"[INFO] Episode terminated at step {total_steps} "
                f"(reason={term_reason})."
            )
            if bool(args.auto_reset_on_done):
                obs = env.reset(
                    randomize=bool(args.randomize_reset),
                    domain_randomize=bool(args.domain_randomize_reset),
                )
                policy.reset(initial_obs=obs, history_init="repeat_obs")
                if kb is not None:
                    kb.mark_reset_complete()
                    if bool(args.start_on_reset):
                        kb.request_start()
                started = bool(args.start_on_reset)
                episode_done = False
                episode_term_reason = None
                print("[INFO] Auto-reset complete.")
            elif args.headless:
                break

        if int(args.max_steps) > 0 and total_steps >= int(args.max_steps):
            print(f"[INFO] Reached --max-steps={args.max_steps}.")
            break

        if float(args.sleep) > 0:
            time.sleep(float(args.sleep))

        if args.headless and episode_done and not bool(args.auto_reset_on_done):
            break

    return total_steps, last_reward, last_info, episode_term_reason


def main() -> int:
    args = _parse_args()
    if args.task not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task '{args.task}'. Supported tasks: {SUPPORTED_TASKS}."
        )

    if args.headless and int(args.max_steps) <= 0:
        raise ValueError("--headless requires --max-steps > 0 to guarantee bounded execution.")

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MJCF model not found: {model_path}")

    checkpoint_path = _resolve_checkpoint(args)

    env = MuJoCoBalanceEnv(
        model_path=str(model_path),
        render=not bool(args.headless),
        seed=int(args.seed),
        controller_mode=str(args.controller_mode),
        domain_rand_mode=str(args.domain_rand_mode),
        task=str(args.task),
    )
    if int(args.max_steps) > 0:
        env.max_episode_steps = int(args.max_steps)
    else:
        env.max_episode_steps = 10**9

    policy = PolicyLoader(
        str(checkpoint_path),
        device=str(args.device),
        policy_spec=None,
        task=str(args.task),
    )

    kb: Optional[KeyboardController]
    if args.headless:
        kb = None
    else:
        kb = KeyboardController(
            gravity_on=(args.gravity == "on"),
            start_enabled=bool(args.start_enabled),
            show_stats=True,
        )

    _set_gravity(env, gravity_on=(args.gravity == "on"))
    print(f"model diagnostics={env.get_model_diagnostics()}")
    _print_banner(args, checkpoint_path, env)

    stats = _new_stats()

    try:
        if args.headless:
            total_steps, last_reward, last_info, last_term_reason = _run_loop(
                args=args,
                env=env,
                policy=policy,
                kb=None,
                stats=stats,
            )
        else:
            with mujoco.viewer.launch_passive(
                env.model,
                env.data,
                key_callback=kb.key_callback if kb is not None else None,
                show_left_ui=bool(args.show_left_ui),
                show_right_ui=bool(args.show_right_ui),
            ) as viewer:
                env.viewer = viewer
                total_steps, last_reward, last_info, last_term_reason = _run_loop(
                    args=args,
                    env=env,
                    policy=policy,
                    kb=kb,
                    stats=stats,
                )
    finally:
        env.close()

    summary = {
        "config": {
            "task": args.task,
            "checkpoint": str(checkpoint_path),
            "model": str(model_path),
            "controller_mode": args.controller_mode,
            "domain_rand_mode": args.domain_rand_mode,
            "seed": int(args.seed),
            "device": args.device,
            "randomize_reset": bool(args.randomize_reset),
            "domain_randomize_reset": bool(args.domain_randomize_reset),
            "wait_mode": args.wait_mode,
            "start_enabled": bool(args.start_enabled),
            "start_on_reset": bool(args.start_on_reset),
            "headless": bool(args.headless),
            "max_steps": int(args.max_steps),
            "eval_fall_tilt_deg": float(args.eval_fall_tilt_deg),
            "no_script_fall_stop": bool(args.no_script_fall_stop),
        },
        "summary": _summarize_stats(stats),
        "final": {
            "total_steps": int(total_steps),
            "last_reward": float(last_reward),
            "last_term_reason": last_term_reason,
            "last_info": last_info,
        },
    }

    print("\nFinal summary:")
    print(f"  total_steps={total_steps}")
    print(f"  policy_steps={summary['summary']['policy_steps']}")
    print(f"  last_reward={last_reward:+.6f}")
    print(f"  last_term_reason={last_term_reason}")
    print(f"  command_stand_ratio={summary['summary']['commands']['stand_ratio']}")

    if args.json_output:
        out_path = Path(args.json_output).expanduser().resolve()
        _dump_json(out_path, summary)
        print(f"  json_output={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
