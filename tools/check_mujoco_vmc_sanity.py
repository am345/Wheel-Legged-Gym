#!/usr/bin/env python3
"""Sanity-check MuJoCo VMC control/observation pipeline without loading an RL policy."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np

from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv


DEFAULT_MODEL = "resources/robots/serialleg/mjcf/serialleg_fidelity.xml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MuJoCoBalanceEnv in vmc_balance_exact mode without an RL policy and "
            "print sanity diagnostics for control/observation/state chains."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"MJCF path (default: {DEFAULT_MODEL})")
    parser.add_argument("--controller-mode", default="vmc_balance_exact")
    parser.add_argument("--domain-rand-mode", default="off", choices=["off", "train_ranges"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--render", action="store_true", help="Open MuJoCo viewer during rollout.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="Wall-clock sleep per control step when --render is enabled (default: 0.01).",
    )
    parser.add_argument(
        "--gravity",
        choices=["on", "off"],
        default="off",
        help="Gravity mode for sanity run (default: off).",
    )
    parser.add_argument(
        "--action-mode",
        choices=["zero", "constant"],
        default="zero",
        help="Action source used for the sanity rollout.",
    )
    parser.add_argument(
        "--action",
        type=float,
        nargs=6,
        metavar=("A0", "A1", "A2", "A3", "A4", "A5"),
        help="Constant action values when --action-mode constant.",
    )
    parser.add_argument(
        "--randomize-reset",
        action="store_true",
        help="Enable balance reset randomization (default: off for deterministic sanity).",
    )
    parser.add_argument(
        "--domain-randomize-reset",
        action="store_true",
        help="Enable episode-level domain randomization on reset (default: off).",
    )
    parser.add_argument(
        "--expect-wheel-collision-mode",
        default="mesh",
        choices=["mesh", "cylinder", "mixed", "none", "skip"],
        help="Expected detected wheel collision mode (default: mesh; use skip to disable).",
    )
    parser.add_argument(
        "--expect-wheel-mesh-source",
        default="visual_stl,visual_stl_chunked",
        help=(
            "Comma-separated acceptable wheel mesh sources in diagnostics "
            "(default: visual_stl,visual_stl_chunked). Use 'skip' to disable."
        ),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON path to save summary diagnostics and final debug state.",
    )
    return parser.parse_args()


def _finite_arrays(debug_state: dict) -> List[str]:
    checks = {
        "L0": np.asarray(debug_state.get("L0", []), dtype=float),
        "theta0": np.asarray(debug_state.get("theta0", []), dtype=float),
        "theta0_dot": np.asarray(debug_state.get("theta0_dot", []), dtype=float),
        "L0_dot": np.asarray(debug_state.get("L0_dot", []), dtype=float),
        "projected_gravity": np.asarray(debug_state.get("projected_gravity", []), dtype=float),
        "last_ctrl": np.asarray(debug_state.get("last_ctrl", []), dtype=float),
    }
    bad = []
    for key, arr in checks.items():
        if not np.all(np.isfinite(arr)):
            bad.append(key)
    return bad


def _build_action(args: argparse.Namespace) -> np.ndarray:
    if args.action_mode == "zero":
        return np.zeros(6, dtype=np.float64)
    if args.action is None:
        raise ValueError("--action-mode constant requires --action with 6 floats.")
    return np.asarray(args.action, dtype=np.float64).reshape(6)


def _print_diag(diag: dict) -> None:
    print("Model diagnostics:")
    print(f"  controller_mode: {diag.get('implemented_controller_mode')}")
    print(f"  fidelity_level: {diag.get('fidelity_level')}")
    print(f"  timing: dt={diag.get('dt')} sim_dt={diag.get('sim_dt')} decimation={diag.get('decimation')}")
    print(f"  joint_names: {diag.get('joint_names')}")
    print(f"  actuator_names: {diag.get('actuator_names')}")
    print(
        "  wheel_collision: "
        f"mode={diag.get('wheel_collision_mode_detected')} "
        f"(mesh={diag.get('wheel_collision_mesh_count')}, "
        f"cylinder={diag.get('wheel_collision_cylinder_count')})"
    )
    print(
        "  wheel_mesh_src: "
        f"{diag.get('wheel_collision_mesh_asset_source_detected')} "
        f"{diag.get('wheel_collision_mesh_asset_names')}"
    )
    if diag.get("wheel_collision_geom_names"):
        print(f"  wheel_geoms: {diag.get('wheel_collision_geom_names')}")


def _check_expected_collision(diag: dict, args: argparse.Namespace) -> List[str]:
    errors: List[str] = []
    expected_mode = args.expect_wheel_collision_mode
    if expected_mode != "skip":
        actual_mode = diag.get("wheel_collision_mode_detected")
        if actual_mode != expected_mode:
            errors.append(
                f"wheel_collision_mode_detected mismatch: expected {expected_mode}, got {actual_mode}"
            )

    expected_src_cfg = args.expect_wheel_mesh_source.strip()
    if expected_src_cfg and expected_src_cfg.lower() != "skip":
        actual_src = diag.get("wheel_collision_mesh_asset_source_detected")
        allowed = {x.strip() for x in expected_src_cfg.split(",") if x.strip()}
        if actual_src not in allowed:
            errors.append(
                "wheel_collision_mesh_asset_source_detected mismatch: "
                f"expected one of {sorted(allowed)}, got {actual_src}"
            )
    return errors


def _print_step_debug(step_idx: int, reward: float, debug_state: dict) -> None:
    print(
        f"step={step_idx:03d} reward={reward:.4f} "
        f"pitch={debug_state['pitch_angle']:.4f} "
        f"prompt={debug_state['prompt_torque_triggered_last_step']} "
        f"sat={debug_state['torque_saturation_count']} "
        f"L0={debug_state['L0']} theta0={debug_state['theta0']} "
        f"ctrl={debug_state['last_ctrl']}"
    )


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def main() -> int:
    args = _parse_args()
    action = _build_action(args)

    env = None
    failures: List[str] = []
    try:
        env = MuJoCoBalanceEnv(
            model_path=str(Path(args.model)),
            render=bool(args.render),
            seed=int(args.seed),
            controller_mode=str(args.controller_mode),
            domain_rand_mode=str(args.domain_rand_mode),
        )

        obs = env.reset(
            randomize=bool(args.randomize_reset),
            domain_randomize=bool(args.domain_randomize_reset),
        )
        if args.gravity == "off":
            env.model.opt.gravity[:] = 0.0
        else:
            env.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])

        if args.render:
            env.render()

        print("Sanity run config:")
        print(
            f"  model={Path(args.model).resolve()} controller_mode={args.controller_mode} "
            f"domain_rand_mode={args.domain_rand_mode} gravity={args.gravity}"
        )
        print(
            f"  steps={args.steps} render={args.render} seed={args.seed} "
            f"randomize_reset={args.randomize_reset} domain_randomize_reset={args.domain_randomize_reset}"
        )
        print(f"  action_mode={args.action_mode} action={action.tolist()}")
        print(f"  obs.shape={tuple(obs.shape)}")

        if tuple(obs.shape) != (27,):
            failures.append(f"obs.shape mismatch: expected (27,), got {tuple(obs.shape)}")
        if not np.all(np.isfinite(obs)):
            failures.append("obs contains NaN/Inf after reset")

        diag = env.get_model_diagnostics()
        _print_diag(diag)
        failures.extend(_check_expected_collision(diag, args))

        last_reward = 0.0
        last_info = {}
        for i in range(args.steps):
            obs, reward, done, info = env.step(action)
            last_reward = float(reward)
            last_info = info

            if not np.all(np.isfinite(obs)):
                failures.append(f"obs contains NaN/Inf at step {i + 1}")
                break
            if not np.isfinite(last_reward):
                failures.append(f"reward is NaN/Inf at step {i + 1}")
                break
            if not np.all(np.isfinite(env.last_ctrl)):
                failures.append(f"last_ctrl contains NaN/Inf at step {i + 1}")
                break

            if args.print_every > 0 and ((i + 1) % args.print_every == 0 or (i + 1) == args.steps):
                _print_step_debug(i + 1, last_reward, env.get_debug_state())

            if args.render and args.sleep > 0:
                time.sleep(args.sleep)

            if done:
                print(f"[INFO] Environment returned done=True at step {i + 1} ({info.get('termination_reason')})")
                break

        debug_state = env.get_debug_state()
        bad_debug_keys = _finite_arrays(debug_state)
        if bad_debug_keys:
            failures.append(f"debug_state contains NaN/Inf in: {bad_debug_keys}")

        print("Final checks:")
        print(f"  last_reward={last_reward:.6f}")
        print(f"  last_info={last_info}")
        print(f"  all_finite={not bad_debug_keys}")
        print(f"  final_pitch={debug_state['pitch_angle']:.6f}")
        print(f"  final_L0={debug_state['L0']}")
        print(f"  final_theta0={debug_state['theta0']}")
        print(f"  final_ctrl={debug_state['last_ctrl']}")
        print(f"  contacts={debug_state['contacts']}")

        if args.json_output is not None:
            payload = {
                "config": {
                    "model": str(Path(args.model).resolve()),
                    "controller_mode": args.controller_mode,
                    "domain_rand_mode": args.domain_rand_mode,
                    "seed": int(args.seed),
                    "steps": int(args.steps),
                    "render": bool(args.render),
                    "gravity": args.gravity,
                    "action_mode": args.action_mode,
                    "action": action.tolist(),
                    "randomize_reset": bool(args.randomize_reset),
                    "domain_randomize_reset": bool(args.domain_randomize_reset),
                },
                "obs_shape": list(obs.shape),
                "model_diagnostics": _jsonable(diag),
                "final_info": _jsonable(last_info),
                "final_debug_state": _jsonable(debug_state),
                "failures": failures,
                "passed": not failures,
            }
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved sanity summary JSON: {args.json_output}")

        if failures:
            print("\nSanity check FAILED:")
            for failure in failures:
                print(f"  - {failure}")
            return 1

        print("\nSanity check PASSED.")
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    sys.exit(main())
