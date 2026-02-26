#!/usr/bin/env python3
"""Open an MJCF model in the MuJoCo viewer without RL policy logic."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


DEFAULT_MODEL = "resources/robots/serialleg/mjcf/serialleg_fidelity.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View an MJCF robot model in MuJoCo.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(DEFAULT_MODEL),
        help=f"MJCF model path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Step the simulation in the viewer. Default is static display (forward only).",
    )
    parser.add_argument(
        "--gravity",
        choices=["on", "off"],
        default="off",
        help="Turn gravity on/off (default: off for easier inspection).",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="When --simulate is enabled, sync steps to wall-clock time.",
    )
    parser.add_argument(
        "--zero-joints",
        action="store_true",
        help="Set actuated joints to zero after reset for a clean pose preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MJCF file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    if args.gravity == "off":
        model.opt.gravity[:] = 0.0

    if args.zero_joints and model.nq >= 13:
        # Free joint (7) + 6 actuated joints layout used by serialleg MJCF.
        data.qpos[7:13] = 0.0

    mujoco.mj_forward(model, data)

    print(f"[MJCF Viewer] model={model_path}")
    print(f"[MJCF Viewer] gravity={args.gravity} simulate={args.simulate}")
    print(f"[MJCF Viewer] nq={model.nq} nv={model.nv} nu={model.nu}")
    print("Close the MuJoCo viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_t = time.perf_counter()
        while viewer.is_running():
            if args.simulate:
                if args.real_time:
                    now = time.perf_counter()
                    target_dt = model.opt.timestep
                    elapsed = now - last_t
                    if elapsed < target_dt:
                        time.sleep(target_dt - elapsed)
                    last_t = time.perf_counter()
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
                time.sleep(1.0 / 60.0)
            viewer.sync()


if __name__ == "__main__":
    main()
