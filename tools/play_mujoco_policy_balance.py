#!/usr/bin/env python3
"""MuJoCo policy interactive demo for wheel_legged_vmc_balance.

Behavior is intentionally similar to wheel_legged_gym/scripts/play_balance.py:
- random reset first (by default)
- policy is loaded but disabled initially
- press C to enable policy control and let the network recover balance
- press R to random-reset and disable policy again
"""

from __future__ import annotations

import argparse
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import mujoco
import mujoco.viewer
import numpy as np

from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv
from mujoco_sim.policy_loader import PolicyLoader


DEFAULT_TASK = "wheel_legged_vmc_balance"
DEFAULT_MODEL = "resources/robots/serialleg/mjcf/serialleg_fidelity.xml"
DEFAULT_CHECKPOINT_ROOT = "logs/wheel_legged_vmc_balance"
ESC_KEY = 256  # GLFW_KEY_ESCAPE


def _find_latest_checkpoint(checkpoint_root: Path) -> Tuple[Path, Path]:
    """Return (run_dir, checkpoint_path) using latest run + max model_*.pt index."""
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

    root = Path(args.checkpoint_root).expanduser().resolve()
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


def _step_wait_no_control(env: MuJoCoBalanceEnv, obs: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, object]]:
    """Advance MuJoCo physics without controller output (zero torques).

    This keeps physical behavior intuitive before pressing C: gravity/contact evolve
    normally, but no policy/controller torque is applied.
    """
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
    obs = env._compute_observation(env.current_action_obs)
    reward = env._compute_reward()
    info = {
        "episode_steps": int(env.episode_steps),  # unchanged while waiting
        "base_height": float(env.data.qpos[2]),
        "controller_mode": "wait_zero_torque",
        "domain_params": env.current_domain_params,
        "termination_reason": "waiting_for_start",
        "prompt_torque_triggered": False,
        "pitch_angle": float(env.pitch_angle),
    }
    env.last_reward = float(reward)
    env.last_done = False
    env.last_info = info
    return obs, float(reward), info


def _step_wait_zero_action(env: MuJoCoBalanceEnv, obs: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, object]]:
    """Advance physics through the normal controller path using zero action (Isaac play-like semantics)."""
    obs, reward, done, info = env.step(np.zeros(6, dtype=np.float32))
    # Interactive waiting mode should not latch env timeout semantics at the script level.
    if done:
        info = dict(info)
        info["termination_reason"] = "waiting_for_start"
    return obs, float(reward), info


def _step_wait_freeze(env: MuJoCoBalanceEnv, obs: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, object]]:
    """Do not advance physics; just keep viewer responsive (debug-only mode)."""
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
    }
    return obs, reward, info


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


def _flagged_joint_names(dbg: Dict[str, object], flag_key: str) -> list[str]:
    names = list(dbg.get("joint_names") or [])
    flags = list(dbg.get(flag_key) or [])
    out = []
    for i, f in enumerate(flags):
        if bool(f):
            out.append(names[i] if i < len(names) else f"joint[{i}]")
    return out


def _print_banner(args: argparse.Namespace, checkpoint_path: Path, env: MuJoCoBalanceEnv) -> None:
    print("\n" + "=" * 80)
    print("MuJoCo Policy Balance Interactive Demo")
    print("=" * 80)
    print(f"task:                 {args.task}")
    print(f"checkpoint:           {checkpoint_path}")
    print(f"model:                {Path(args.model).resolve()}")
    print(f"controller_mode:      {args.controller_mode}")
    print(f"domain_rand_mode:     {args.domain_rand_mode}")
    print(f"randomize_reset:      {args.randomize_reset}")
    print(f"domain_randomize_rst: {args.domain_randomize_reset}")
    print(f"gravity:              {args.gravity}")
    print(f"start_enabled:        {args.start_enabled}")
    print(f"start_on_reset:       {args.start_on_reset}")
    print(f"wait_mode:            {args.wait_mode}")
    print(f"reset_profile:        {args.reset_profile}")
    print(f"fall_tilt_deg:        {args.eval_fall_tilt_deg}")
    print(f"no_script_fall_stop:  {args.no_script_fall_stop}")
    print(f"mujoco_tuning_profile:{args.mujoco_tuning_profile}")
    print(f"seed/device:          {args.seed} / {args.device}")
    print(f"fidelity_level:       {env.fidelity_level}")
    print("=" * 80)
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
    show_diagnosis_compact: bool = True,
) -> None:
    tilt_deg = float(dbg.get("tilt_deg", _compute_tilt_deg(np.asarray(dbg["projected_gravity"], dtype=float))))
    sat_joints = _flagged_joint_names(dbg, "torque_saturation_flags")
    limit_joints = _flagged_joint_names(dbg, "joint_limit_hit_flags")
    hint_active = bool(dbg.get("balance_hint_active", dbg.get("prompt_torque_triggered_last_step", False)))
    diag_extra = ""
    if show_diagnosis_compact:
        diag_extra = f" hint={hint_active} sat={sat_joints or '-'} limit={limit_joints or '-'}"
    print(
        f"step={total_steps:05d} "
        f"reward={reward:+.4f} "
        f"wait={dbg.get('wait_mode', 'n/a')} "
        f"started={state['started']} paused={state['paused']} gravity={'on' if state['gravity_on'] else 'off'} "
        f"pitch={dbg['pitch_angle']:+.3f} tilt={tilt_deg:6.2f}deg "
        f"L0={np.round(np.asarray(dbg['L0'], dtype=float), 4).tolist()} "
        f"theta0={np.round(np.asarray(dbg['theta0'], dtype=float), 4).tolist()} "
        f"ctrl={np.round(np.asarray(dbg['last_ctrl'], dtype=float), 3).tolist()} "
        f"{diag_extra}"
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
    print("\n" + "-" * 80)
    print(f"Debug snapshot @ step {total_steps}")
    print(
        f"state={{started={state['started']}, paused={state['paused']}, gravity_on={state['gravity_on']}}} "
        f"reward={reward:+.6f} tilt={tilt_deg:.3f}deg"
    )
    print(f"info={info}")
    print(f"pitch_angle={dbg['pitch_angle']:+.6f}")
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
    print(
        f"sat_flags={dbg.get('torque_saturation_flags')} "
        f"limit_flags={dbg.get('joint_limit_hit_flags')} "
        f"limit_margin={np.round(np.asarray(dbg.get('joint_limit_margin', []), dtype=float), 6).tolist() if dbg.get('joint_limit_margin') is not None else None}"
    )
    print(
        f"hint_active={dbg.get('balance_hint_active')} "
        f"prompt_steps={dbg.get('prompt_torque_step_count')} "
        f"torque_sat_count={dbg.get('torque_saturation_count')}"
    )
    print(f"contacts={dbg['contacts']}")
    print(f"control_debug={dbg.get('control_debug')}")
    print("-" * 80 + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo interactive policy demo (play_balance-like) for wheel_legged_vmc_balance."
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help=f"Task name (default: {DEFAULT_TASK})")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (.pt).")
    parser.add_argument(
        "--checkpoint-root",
        default=DEFAULT_CHECKPOINT_ROOT,
        help=f"Checkpoint root for auto-discovery (default: {DEFAULT_CHECKPOINT_ROOT})",
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
        help="Env DR mode. Interactive demo default is off.",
    )
    parser.add_argument(
        "--mujoco-tuning-profile",
        default="exact_baseline",
        choices=["exact_baseline", "demo_tuned"],
        help="MuJoCo-only tuning profile (exact_baseline for real line, demo_tuned for demo line).",
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
        help="Apply episode DR on reset (default off; usually keep off for demos).",
    )
    parser.add_argument(
        "--reset-profile",
        default="nominal_demo",
        choices=["nominal_demo", "hard_random_balance", "nominal", "random_balance", "default"],
        help="Balance reset profile (demo default is nominal_demo).",
    )
    parser.add_argument(
        "--wait-mode",
        default="zero_action",
        choices=["zero_action", "zero_torque", "freeze"],
        help="Behavior before pressing C (zero_action matches Isaac play_balance semantics better).",
    )
    parser.add_argument("--gravity", choices=["on", "off"], default="on")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument("--eval-fall-tilt-deg", type=float, default=60.0)
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
    diag_group = parser.add_mutually_exclusive_group()
    diag_group.add_argument(
        "--show-diagnosis-compact",
        dest="show_diagnosis_compact",
        action="store_true",
        help="Show compact limit/saturation/hint diagnostics in step prints (default).",
    )
    diag_group.add_argument(
        "--hide-diagnosis-compact",
        dest="show_diagnosis_compact",
        action="store_false",
        help="Hide compact diagnostic columns in step prints.",
    )
    parser.set_defaults(show_diagnosis_compact=True)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--show-left-ui", action="store_true", default=True)
    parser.add_argument("--hide-left-ui", dest="show_left_ui", action="store_false")
    parser.add_argument("--show-right-ui", action="store_true", default=True)
    parser.add_argument("--hide-right-ui", dest="show_right_ui", action="store_false")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.task != DEFAULT_TASK:
        raise ValueError(
            f"Unsupported task '{args.task}'. This script currently supports only '{DEFAULT_TASK}'."
        )

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MJCF model not found: {model_path}")

    checkpoint_path = _resolve_checkpoint(args)

    env = MuJoCoBalanceEnv(
        model_path=str(model_path),
        render=True,
        seed=int(args.seed),
        controller_mode=str(args.controller_mode),
        domain_rand_mode=str(args.domain_rand_mode),
        mujoco_tuning_profile=str(args.mujoco_tuning_profile),
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

    kb = KeyboardController(
        gravity_on=(args.gravity == "on"),
        start_enabled=bool(args.start_enabled),
        show_stats=True,
    )

    obs = env.reset(
        randomize=bool(args.randomize_reset),
        domain_randomize=bool(args.domain_randomize_reset),
        reset_profile=str(args.reset_profile),
    )
    policy.reset()
    _set_gravity(env, gravity_on=(args.gravity == "on"))
    env.wait_mode = str(args.wait_mode)
    if bool(args.start_on_reset) and not bool(args.start_enabled):
        kb.request_start()

    if obs.shape != (27,):
        raise RuntimeError(f"Unexpected observation shape after reset: {obs.shape}")
    if not np.all(np.isfinite(obs)):
        raise RuntimeError("Observation contains NaN/Inf after initial reset")

    print(f"obs.shape={tuple(obs.shape)}")
    print(f"model diagnostics={env.get_model_diagnostics()}")
    _print_banner(args, checkpoint_path, env)
    print(
        "[INFO] Policy starts DISABLED. "
        f"Wait mode='{args.wait_mode}' (physics behavior before C). Press C to start.\n"
    )

    total_steps = 0
    last_reward = 0.0
    last_info: Dict[str, object] = {}
    episode_done = False
    episode_term_reason: Optional[str] = None

    with mujoco.viewer.launch_passive(
        env.model,
        env.data,
        key_callback=kb.key_callback,
        show_left_ui=bool(args.show_left_ui),
        show_right_ui=bool(args.show_right_ui),
    ) as viewer:
        env.viewer = viewer

        while viewer.is_running():
            state = kb.consume()
            if bool(state["exit_requested"]):
                print("[INFO] Exit requested by keyboard.")
                break

            _set_gravity(env, bool(state["gravity_on"]))

            if bool(state["reset_requested"]):
                obs = env.reset(
                    randomize=bool(args.randomize_reset),
                    domain_randomize=bool(args.domain_randomize_reset),
                    reset_profile=str(args.reset_profile),
                )
                policy.reset()
                kb.mark_reset_complete()
                _set_gravity(env, bool(state["gravity_on"]))
                env.wait_mode = str(args.wait_mode)
                episode_done = False
                episode_term_reason = None
                print(
                    "[INFO] Environment reset complete "
                    f"(randomize_reset={args.randomize_reset}, "
                    f"domain_randomize_reset={args.domain_randomize_reset}, "
                    f"reset_profile={args.reset_profile}). "
                    "Press C to start policy."
                )
                if bool(args.start_on_reset):
                    kb.request_start()
                    print("[INFO] START auto-requested after reset.")
                if not np.all(np.isfinite(obs)):
                    print("[WARN] Observation contains NaN/Inf immediately after reset.")

            if bool(state["start_requested"]):
                if episode_done:
                    kb.force_stop("episode_done_waiting_reset")
                    print("[WARN] Episode already terminated. Press R to reset before pressing C.")
                else:
                    policy.reset()
                    print("[INFO] Policy enabled (history reset).")

            if bool(state["debug_requested"]):
                _print_debug_snapshot(
                    env,
                    total_steps=total_steps,
                    reward=last_reward,
                    info=last_info,
                    state=state,
                )

            if bool(state["paused"]):
                env.wait_mode = "paused"
                viewer.sync()
                time.sleep(max(float(args.sleep), 0.01))
                continue

            term_reason: Optional[str] = None
            done_env = False

            if bool(state["started"]) and not episode_done:
                env.wait_mode = "policy"
                action = policy.get_action(obs)
                if not np.all(np.isfinite(action)):
                    episode_done = True
                    episode_term_reason = "nan_detected_action"
                    kb.force_stop(episode_term_reason)
                    print("[ERROR] Policy action contains NaN/Inf. Press R to reset.")
                    viewer.sync()
                    time.sleep(max(float(args.sleep), 0.01))
                    continue

                obs, reward, done_env, info = env.step(action)
                last_reward = float(reward)
                last_info = info
                total_steps += 1
                dbg = env.get_debug_state()

                if (not np.all(np.isfinite(obs))) or (not np.isfinite(last_reward)) or (
                    not np.all(np.isfinite(env.last_ctrl))
                ):
                    term_reason = "nan_detected"
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
                # Before pressing C (or after a latched terminal), advance according to configured wait mode.
                if str(args.wait_mode) == "zero_action":
                    env.wait_mode = "zero_action"
                    obs, reward, info = _step_wait_zero_action(env, obs)
                elif str(args.wait_mode) == "zero_torque":
                    env.wait_mode = "zero_torque"
                    obs, reward, info = _step_wait_no_control(env, obs)
                else:
                    env.wait_mode = "freeze"
                    obs, reward, info = _step_wait_freeze(env, obs)
                last_reward = float(reward)
                last_info = info
                total_steps += 1
                dbg = env.get_debug_state()

            if bool(state["show_stats"]) and int(args.print_every) > 0 and (total_steps % int(args.print_every) == 0):
                _print_step_stats(
                    total_steps=total_steps,
                    reward=last_reward,
                    info=info,
                    dbg=dbg,
                    state=state,
                    termination_reason=term_reason,
                    show_diagnosis_compact=bool(args.show_diagnosis_compact),
                )

            if term_reason is not None:
                episode_done = True
                episode_term_reason = term_reason
                kb.force_stop(term_reason)
                print(
                    f"[INFO] Episode terminated at step {total_steps} "
                    f"(reason={term_reason}). Press R to reset."
                )
                if bool(args.auto_reset_on_done):
                    obs = env.reset(
                        randomize=bool(args.randomize_reset),
                        domain_randomize=bool(args.domain_randomize_reset),
                        reset_profile=str(args.reset_profile),
                    )
                    policy.reset()
                    kb.mark_reset_complete()
                    _set_gravity(env, bool(state["gravity_on"]))
                    env.wait_mode = str(args.wait_mode)
                    episode_done = False
                    episode_term_reason = None
                    print("[INFO] Auto-reset complete (policy remains disabled; press C to start).")
                    if bool(args.start_on_reset):
                        kb.request_start()
                        print("[INFO] START auto-requested after auto-reset.")

            if float(args.sleep) > 0:
                time.sleep(float(args.sleep))

    print("\nFinal summary:")
    print(f"  total_steps={total_steps}")
    print(f"  last_reward={last_reward:+.6f}")
    print(f"  last_term_reason={episode_term_reason}")
    print(f"  last_info={last_info}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
