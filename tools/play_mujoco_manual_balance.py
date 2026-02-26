#!/usr/bin/env python3
"""Manual keyboard control for MuJoCo wheel_legged_vmc_balance (no RL policy).

This script is intended to feel similar to ``play_balance.py``:
- viewer-based keyboard interaction
- start/pause/reset/exit controls
- real-time status printouts

Control is manual action injection into MuJoCoBalanceEnv action space (6D):
  [left_theta, left_l0, left_wheel_vel, right_theta, right_l0, right_wheel_vel]
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import mujoco.viewer
import numpy as np

from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv


DEFAULT_MODEL = "resources/robots/serialleg/mjcf/serialleg_fidelity.xml"
ESC_KEY = 256  # GLFW_KEY_ESCAPE


@dataclass
class ManualActionConfig:
    theta_step: float = 0.05
    l0_step: float = 0.05
    wheel_step: float = 0.10
    steer_step: float = 0.10
    action_clip_soft: float = 2.0


class ManualKeyboardState:
    """Thread-safe state shared between MuJoCo key callback and main loop."""

    def __init__(self, cfg: ManualActionConfig, gravity_on: bool, show_stats: bool = True):
        self._lock = threading.Lock()
        self.cfg = cfg
        self.gravity_on = gravity_on
        self.show_stats = show_stats
        self.started = False
        self.paused = False
        self.exit_requested = False
        self.reset_requested = False
        self.print_debug_requested = False
        self.action = np.zeros(6, dtype=np.float64)

    def _clip(self):
        lim = float(self.cfg.action_clip_soft)
        self.action[:] = np.clip(self.action, -lim, lim)

    def _print_action(self, prefix: str = ""):
        a = self.action
        print(
            f"{prefix}action="
            f"[L(theta={a[0]:+.3f}, l0={a[1]:+.3f}, wheel={a[2]:+.3f}), "
            f"R(theta={a[3]:+.3f}, l0={a[4]:+.3f}, wheel={a[5]:+.3f})]"
        )

    def _apply_key(self, ch: str):
        ch = ch.lower()
        if ch == "c":
            self.started = not self.started
            print(f"[KEY] control {'ENABLED' if self.started else 'DISABLED'}")
            return
        if ch == "p":
            self.paused = not self.paused
            print(f"[KEY] {'PAUSED' if self.paused else 'RESUMED'}")
            return
        if ch == "r":
            self.reset_requested = True
            self.started = False
            print("[KEY] RESET requested (control disabled)")
            return
        if ch == "m":
            self.show_stats = not self.show_stats
            print(f"[KEY] stats {'ON' if self.show_stats else 'OFF'}")
            return
        if ch == "g":
            self.gravity_on = not self.gravity_on
            print(f"[KEY] gravity {'ON' if self.gravity_on else 'OFF'}")
            return
        if ch == "z":
            self.action[:] = 0.0
            print("[KEY] action reset to zeros")
            self._print_action(prefix="      ")
            return
        if ch == "n":
            self.print_debug_requested = True
            return
        if ch == "h":
            self._print_help()
            return

        changed = False
        # Symmetric theta0 reference
        if ch == "j":
            self.action[[0, 3]] -= self.cfg.theta_step
            changed = True
        elif ch == "l":
            self.action[[0, 3]] += self.cfg.theta_step
            changed = True
        # Symmetric l0 reference
        elif ch == "i":
            self.action[[1, 4]] += self.cfg.l0_step
            changed = True
        elif ch == "k":
            self.action[[1, 4]] -= self.cfg.l0_step
            changed = True
        # Forward/back wheel velocity
        elif ch == "w":
            self.action[[2, 5]] += self.cfg.wheel_step
            changed = True
        elif ch == "s":
            self.action[[2, 5]] -= self.cfg.wheel_step
            changed = True
        # Differential steer (left/right wheel opposite)
        elif ch == "a":
            self.action[2] -= self.cfg.steer_step
            self.action[5] += self.cfg.steer_step
            changed = True
        elif ch == "d":
            self.action[2] += self.cfg.steer_step
            self.action[5] -= self.cfg.steer_step
            changed = True
        # Left/right independent theta trim
        elif ch == "q":
            self.action[0] += self.cfg.theta_step
            self.action[3] -= self.cfg.theta_step
            changed = True
        elif ch == "e":
            self.action[0] -= self.cfg.theta_step
            self.action[3] += self.cfg.theta_step
            changed = True

        if changed:
            self._clip()
            self._print_action(prefix="[KEY] ")

    def key_callback(self, keycode: int):
        with self._lock:
            if int(keycode) == ESC_KEY:
                self.exit_requested = True
                print("[KEY] EXIT requested")
                return
            if 0 <= int(keycode) < 256:
                self._apply_key(chr(int(keycode)))

    def consume(self) -> Dict[str, object]:
        """Return a snapshot of state and clear one-shot flags."""
        with self._lock:
            out = {
                "started": self.started,
                "paused": self.paused,
                "exit_requested": self.exit_requested,
                "reset_requested": self.reset_requested,
                "print_debug_requested": self.print_debug_requested,
                "gravity_on": self.gravity_on,
                "show_stats": self.show_stats,
                "action": self.action.copy(),
            }
            self.reset_requested = False
            self.print_debug_requested = False
            return out

    @staticmethod
    def _print_help():
        print("\nKeyboard controls:")
        print("  C : Enable/disable manual control")
        print("  R : Reset environment (also disables control)")
        print("  P : Pause/resume stepping")
        print("  M : Toggle terminal stats")
        print("  G : Toggle gravity on/off")
        print("  Z : Zero manual action")
        print("  N : Print detailed debug snapshot now")
        print("  H : Print this help")
        print("  ESC : Exit")
        print("")
        print("Manual action mapping (vmc_balance_exact action space):")
        print("  J/L : theta0 ref -/+ (both legs)")
        print("  I/K : l0 ref + /- (both legs)")
        print("  W/S : wheel velocity ref + /- (both wheels)")
        print("  A/D : differential wheel steer (left-/right+, left+/right-)")
        print("  Q/E : differential theta trim")
        print("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive manual keyboard control for MuJoCo wheel_legged_vmc_balance."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"MJCF path (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--controller-mode",
        default="vmc_balance_exact",
        choices=["vmc_balance_exact", "simplified_joint_pd"],
        help="Control mode in MuJoCoBalanceEnv (default: vmc_balance_exact)",
    )
    parser.add_argument(
        "--domain-rand-mode",
        default="off",
        choices=["off", "train_ranges"],
        help="Episode-level DR mode (default: off; manual control usually wants off).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gravity", choices=["on", "off"], default="on")
    parser.add_argument("--randomize-reset", action="store_true", help="Enable balance reset randomization.")
    parser.add_argument(
        "--domain-randomize-reset",
        action="store_true",
        help="Apply domain randomization on reset (only meaningful with --domain-rand-mode train_ranges).",
    )
    parser.add_argument("--print-every", type=int, default=20, help="Stats print interval in control steps.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional total control-step cap. 0 means no practical limit.",
    )
    parser.add_argument("--sleep", type=float, default=0.01, help="Wall-clock sleep per step (default 0.01s).")
    parser.add_argument("--theta-step", type=float, default=0.05)
    parser.add_argument("--l0-step", type=float, default=0.05)
    parser.add_argument("--wheel-step", type=float, default=0.10)
    parser.add_argument("--steer-step", type=float, default=0.10)
    parser.add_argument(
        "--action-clip-soft",
        type=float,
        default=2.0,
        help="Soft clip for manual action values to avoid runaway key repeats.",
    )
    return parser.parse_args()


def _set_gravity(env: MuJoCoBalanceEnv, gravity_on: bool) -> None:
    if gravity_on:
        env.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    else:
        env.model.opt.gravity[:] = 0.0


def _print_status_banner() -> None:
    print("\n" + "=" * 72)
    print("MuJoCo Manual Balance Control")
    print("=" * 72)
    ManualKeyboardState._print_help()


def _print_step_stats(step_idx: int, reward: float, info: dict, dbg: dict, state: Dict[str, object]) -> None:
    action = np.asarray(state["action"], dtype=float)
    print(
        f"step={step_idx:05d} reward={reward:+.4f} "
        f"started={state['started']} paused={state['paused']} gravity={'on' if state['gravity_on'] else 'off'} "
        f"pitch={dbg['pitch_angle']:+.3f} "
        f"L0={np.round(np.asarray(dbg['L0']), 4).tolist()} "
        f"theta0={np.round(np.asarray(dbg['theta0']), 4).tolist()} "
        f"ctrl={np.round(np.asarray(dbg['last_ctrl']), 3).tolist()} "
        f"action={np.round(action, 3).tolist()} "
        f"contacts={dbg['contacts']}"
    )


def _print_debug_snapshot(env: MuJoCoBalanceEnv, step_idx: int, reward: float, info: dict, state: Dict[str, object]) -> None:
    dbg = env.get_debug_state()
    print("\n" + "-" * 72)
    print(f"Debug snapshot @ step {step_idx}")
    print(f"reward={reward:+.6f} info={info}")
    print(f"state={{started={state['started']}, paused={state['paused']}, gravity_on={state['gravity_on']}}}")
    print(f"manual_action={np.round(np.asarray(state['action']), 4).tolist()}")
    print(f"pitch_angle={dbg['pitch_angle']:+.6f}")
    print(f"projected_gravity={np.round(np.asarray(dbg['projected_gravity']), 6).tolist()}")
    print(f"L0={np.round(np.asarray(dbg['L0']), 6).tolist()} L0_dot={np.round(np.asarray(dbg['L0_dot']), 6).tolist()}")
    print(
        f"theta0={np.round(np.asarray(dbg['theta0']), 6).tolist()} "
        f"theta0_dot={np.round(np.asarray(dbg['theta0_dot']), 6).tolist()}"
    )
    print(f"last_ctrl={np.round(np.asarray(dbg['last_ctrl']), 6).tolist()}")
    print(f"contacts={dbg['contacts']}")
    print(f"control_debug={dbg.get('control_debug')}")
    print("-" * 72 + "\n")


def main() -> int:
    args = parse_args()

    cfg = ManualActionConfig(
        theta_step=float(args.theta_step),
        l0_step=float(args.l0_step),
        wheel_step=float(args.wheel_step),
        steer_step=float(args.steer_step),
        action_clip_soft=float(args.action_clip_soft),
    )
    kb = ManualKeyboardState(cfg=cfg, gravity_on=(args.gravity == "on"))

    env = MuJoCoBalanceEnv(
        model_path=str(Path(args.model)),
        render=True,
        seed=int(args.seed),
        controller_mode=str(args.controller_mode),
        domain_rand_mode=str(args.domain_rand_mode),
    )
    if int(args.max_steps) > 0:
        env.max_episode_steps = int(args.max_steps)
    else:
        env.max_episode_steps = 10**9

    obs = env.reset(
        randomize=bool(args.randomize_reset),
        domain_randomize=bool(args.domain_randomize_reset),
    )
    _set_gravity(env, kb.gravity_on)
    print(f"obs.shape={tuple(obs.shape)}")
    print(f"model diagnostics={env.get_model_diagnostics()}")
    _print_status_banner()

    total_steps = 0
    last_reward = 0.0
    last_info: Dict[str, object] = {}
    done = False

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=kb.key_callback) as viewer:
        env.viewer = viewer
        while viewer.is_running():
            state = kb.consume()
            if state["exit_requested"]:
                print("[INFO] Exit requested by keyboard.")
                break

            _set_gravity(env, bool(state["gravity_on"]))

            if state["reset_requested"]:
                obs = env.reset(
                    randomize=bool(args.randomize_reset),
                    domain_randomize=bool(args.domain_randomize_reset),
                )
                _set_gravity(env, bool(state["gravity_on"]))
                done = False
                total_steps = 0 if total_steps < 0 else total_steps
                print("[INFO] Environment reset complete.")

            if state["print_debug_requested"]:
                _print_debug_snapshot(env, total_steps, last_reward, last_info, state)

            if state["paused"]:
                viewer.sync()
                time.sleep(max(float(args.sleep), 0.01))
                continue

            if done:
                # No auto-reset; keep viewer responsive and wait for user reset.
                viewer.sync()
                time.sleep(max(float(args.sleep), 0.01))
                continue

            action = np.asarray(state["action"], dtype=np.float64)
            if not state["started"]:
                action = np.zeros(6, dtype=np.float64)

            obs, reward, done, info = env.step(action)
            last_reward = float(reward)
            last_info = info
            total_steps += 1

            if bool(state["show_stats"]) and int(args.print_every) > 0 and (total_steps % int(args.print_every) == 0):
                _print_step_stats(total_steps, last_reward, info, env.get_debug_state(), state)

            if done:
                print(
                    f"[INFO] done=True at step {total_steps} "
                    f"(termination_reason={info.get('termination_reason')}). Press R to reset."
                )

            if float(args.sleep) > 0:
                time.sleep(float(args.sleep))

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

