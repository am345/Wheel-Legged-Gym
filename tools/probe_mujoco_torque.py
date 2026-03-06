#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv
from mujoco_sim.policy_loader import PolicyLoader
from wheel_legged_gym.rsl_rl import env


def parse_ctrl(text: str) -> np.ndarray:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 6:
        raise argparse.ArgumentTypeError("manual-ctrl must have 6 values: lf0,lf1,lw,rf0,rf1,rw")
    return np.asarray(vals, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Probe ctrl/qfrc_actuator/qfrc_constraint/qacc in MuJoCo.")
    parser.add_argument("--mode", choices=["manual", "policy"], default="manual")
    parser.add_argument("--model", default="resources/robots/serialleg/mjcf/serialleg_fidelity.xml")
    parser.add_argument("--task", default="wheel_legged_fzqver")
    parser.add_argument(
        "--controller-mode",
        default="vmc_balance_exact",
        choices=["vmc_balance_exact", "simplified_joint_pd"],
    )
    parser.add_argument("--domain-rand-mode", default="off", choices=["off", "train_ranges"])
    parser.add_argument("--checkpoint", default="logs/wheel_legged_fzqver/Feb28_13-51-29_/model_5000.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--domain-randomize-reset", action="store_true")
    parser.add_argument(
        "--manual-ctrl",
        type=parse_ctrl,
        default=np.array([30.0, 30.0, -2.0, 30.0, 30.0, -2.0], dtype=np.float64),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model).expanduser().resolve()
    env = MuJoCoBalanceEnv(
        model_path=str(model_path),
        render=bool(args.render),
        seed=int(args.seed),
        controller_mode=str(args.controller_mode),
        domain_rand_mode=str(args.domain_rand_mode),
        task=str(args.task),
    )

    obs = env.reset(
        randomize=bool(args.randomize_reset),
        domain_randomize=bool(args.domain_randomize_reset),
    )

    policy = None
    if args.mode == "policy":
        ckpt = Path(args.checkpoint).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        policy = PolicyLoader(checkpoint_path=str(ckpt), task=str(args.task), device="cpu")
        policy.reset(initial_obs=obs, history_init="repeat_obs")

    for t in range(1, int(args.steps) + 1):
        if args.mode == "policy":
            action = policy.get_action(obs)
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset(
                    randomize=bool(args.randomize_reset),
                    domain_randomize=bool(args.domain_randomize_reset),
                )
                policy.reset(initial_obs=obs, history_init="repeat_obs")
        else:
            env.data.ctrl[:] = args.manual_ctrl
            for _ in range(env.decimation):
                mujoco.mj_step(env.model, env.data)
            env._refresh_state_from_sim(init_from_current=False)
            if env.viewer is not None:
                env.viewer.sync()

        if t % int(args.print_every) == 0:
            ids = env.joint_dof_addrs
            qa = env.data.qfrc_actuator[ids]
            qc = env.data.qfrc_constraint[ids]
            qacc = env.data.qacc[ids]
            qvel = env.data.qvel[ids]

            g = np.asarray(env.projected_gravity, dtype=float)
            tilt_deg = float(np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0))))

            print(f"\nstep={t} tilt={tilt_deg:.2f}deg base_z={env.data.qpos[2]:.4f}")
            print("ctrl            =", np.round(env.data.ctrl, 3).tolist())
            print("qfrc_actuator   =", np.round(qa, 3).tolist())
            print("qfrc_constraint =", np.round(qc, 3).tolist())
            print("qacc            =", np.round(qacc, 3).tolist())
            print("qvel            =", np.round(qvel, 3).tolist())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

if args.render:
    env.render()
