#!/usr/bin/env python3
"""
Collect IsaacGym reference rollout for sim2sim alignment.

Important: import order matters in this environment.
We import `isaacgym` before `torch` to avoid runtime conflicts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple

# IMPORTANT: isaacgym must be imported before torch in this env.
from isaacgym import gymapi, gymtorch  # noqa: E402
import torch  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import wheel_legged_gym.envs  # noqa: E402,F401 - task registration side effects
from wheel_legged_gym.utils.task_registry import task_registry  # noqa: E402
from mujoco_sim.control_config import get_balance_vmc_control_config  # noqa: E402
from mujoco_sim.policy_loader import PolicyLoader  # noqa: E402

SUPPORTED_TASK = "wheel_legged_vmc_balance"
DEFAULT_CHECKPOINT = "logs/wheel_legged_vmc_balance/Feb23_15-01-26_/model_900.pt"


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect IsaacGym reference rollout for MuJoCo alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", type=str, default=SUPPORTED_TASK)
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-device", type=str, default="cpu", help="Policy inference device")
    parser.add_argument("--sim-device", type=str, default="cuda:0", help="Isaac sim device")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless")
    parser.add_argument(
        "--scenario",
        type=str,
        default="deterministic_nominal",
        choices=["deterministic_nominal", "contact_nominal", "balance_randomized"],
    )
    parser.add_argument(
        "--domain-rand-mode",
        type=str,
        default="off",
        choices=["off", "train_ranges"],
        help="Isaac-side domain randomization for reference collection",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="isaac_reference_rollout.npz",
        help="Output npz path",
    )
    parser.add_argument(
        "--height-offset",
        type=float,
        default=0.0,
        help="Optional additional root height offset applied after reset (debug only)",
    )
    args = parser.parse_args()
    if args.task != SUPPORTED_TASK:
        parser.error(f"Only {SUPPORTED_TASK} is currently supported")
    if args.steps <= 0:
        parser.error("--steps must be > 0")
    return args


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _zero_range(range_like) -> Tuple[float, float]:
    return (0.0, 0.0)


def configure_env_cfg_for_reference(env_cfg, *, scenario: str, domain_rand_mode: str) -> None:
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = 120.0
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.push_robots = False

    # Domain randomization toggles
    dr = env_cfg.domain_rand
    dr_enabled = domain_rand_mode == "train_ranges"
    for attr in (
        "randomize_friction",
        "randomize_restitution",
        "randomize_base_mass",
        "randomize_inertia",
        "randomize_base_com",
        "randomize_Kp",
        "randomize_Kd",
        "randomize_motor_torque",
        "randomize_default_dof_pos",
        "randomize_action_delay",
    ):
        if hasattr(dr, attr):
            setattr(dr, attr, bool(dr_enabled))

    # Scenario-specific reset randomness
    balance_random = scenario == "balance_randomized"
    if hasattr(env_cfg, "balance_reset") and not balance_random:
        br = env_cfg.balance_reset
        for attr in (
            "x_pos_offset",
            "y_pos_offset",
            "z_pos_offset",
            "roll",
            "pitch",
            "yaw",
            "lin_vel_x",
            "lin_vel_y",
            "lin_vel_z",
            "ang_vel_roll",
            "ang_vel_pitch",
            "ang_vel_yaw",
        ):
            if hasattr(br, attr):
                setattr(br, attr, [0.0, 0.0])

    # "contact_nominal" still keeps deterministic reset but we label it separately.


def build_task_args(seed: int, sim_device: str, headless: bool) -> SimpleNamespace:
    use_gpu = not sim_device.startswith("cpu")
    device_type = "cpu" if sim_device.startswith("cpu") else "cuda"
    return SimpleNamespace(
        seed=seed,
        num_envs=1,
        physics_engine=gymapi.SIM_PHYSX,
        device=device_type,
        use_gpu=use_gpu,
        subscenes=0,
        use_gpu_pipeline=use_gpu,
        num_threads=0,
        sim_device=sim_device,
        headless=headless,
    )


def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def build_contact_index_map(env) -> Dict[str, int]:
    env0 = env.envs[0]
    actor0 = env.actor_handles[0]
    gym = env.gym
    out = {}
    for key, body_name in (
        ("base_contact", "base_link"),
        ("left_wheel_contact", "l_wheel_Link"),
        ("right_wheel_contact", "r_wheel_Link"),
    ):
        try:
            idx = gym.find_actor_rigid_body_handle(env0, actor0, body_name)
        except Exception:
            idx = -1
        out[key] = int(idx)
    return out


def get_contact_flags(env, index_map: Dict[str, int], force_threshold: float = 1e-3) -> np.ndarray:
    cf = env.contact_forces[0]  # [num_bodies, 3]
    out = []
    for key in ("base_contact", "left_wheel_contact", "right_wheel_contact"):
        idx = index_map.get(key, -1)
        if idx < 0 or idx >= cf.shape[0]:
            out.append(False)
            continue
        out.append(bool(torch.norm(cf[idx]).item() > force_threshold))
    return np.asarray(out, dtype=np.bool_)


def get_current_obs(env) -> np.ndarray:
    obs = env.obs_buf[0]
    return tensor_to_numpy(obs).astype(np.float32)


def get_root_state(env) -> np.ndarray:
    return tensor_to_numpy(env.root_states[0]).astype(np.float32)


def get_init_state(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    root = get_root_state(env)
    dof_pos = tensor_to_numpy(env.dof_pos[0]).astype(np.float32)
    dof_vel = tensor_to_numpy(env.dof_vel[0]).astype(np.float32)
    return root, dof_pos, dof_vel


def maybe_apply_height_offset(env, offset: float) -> None:
    if abs(offset) < 1e-12:
        return
    # Directly nudge root state in Isaac for debugging scenarios.
    env.root_states[0, 2] += float(offset)
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_path(args.checkpoint)
    output_path = Path(args.output).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    set_global_seed(args.seed)

    print("\n=== Isaac Reference Rollout Config ===")
    print(f"task: {args.task}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"steps: {args.steps}")
    print(f"seed: {args.seed}")
    print(f"policy_device: {args.policy_device}")
    print(f"sim_device: {args.sim_device}")
    print(f"scenario: {args.scenario}")
    print(f"domain_rand_mode: {args.domain_rand_mode}")
    print(f"output: {output_path}")

    env_cfg, _train_cfg = task_registry.get_cfgs(name=args.task)
    configure_env_cfg_for_reference(
        env_cfg,
        scenario=args.scenario,
        domain_rand_mode=args.domain_rand_mode,
    )

    task_args = build_task_args(seed=args.seed, sim_device=args.sim_device, headless=args.headless)
    env, env_cfg = task_registry.make_env(name=args.task, args=task_args, env_cfg=env_cfg)

    # BaseTask.reset() does one zero-action step and returns fresh observations.
    env.reset()
    if abs(args.height_offset) > 0:
        maybe_apply_height_offset(env, args.height_offset)
        # One zero step to refresh tensors consistently after state poke.
        zero = torch.zeros((env.num_envs, env.num_actions), device=env.device)
        env.step(zero)
    obs, _obs_history = env.get_observations()

    policy = PolicyLoader(str(checkpoint_path), device=args.policy_device, task=args.task)
    policy.reset()

    contact_index_map = build_contact_index_map(env)
    init_root_state, init_dof_pos, init_dof_vel = get_init_state(env)

    # Trajectory buffers
    actions = []
    obs_list = []
    torques = []
    root_pos = []
    root_quat_xyzw = []
    root_lin_vel = []
    root_ang_vel = []
    dof_pos = []
    dof_vel = []
    projected_gravity = []
    L0 = []
    theta0 = []
    theta0_ref = []
    l0_ref = []
    force_leg = []
    torque_leg = []
    qpos_like = []
    qvel_like = []
    contacts = []
    done_flags = []

    print("\nCollecting rollout...")
    ctrl_cfg = get_balance_vmc_control_config()
    for t in range(args.steps):
        obs_np = tensor_to_numpy(obs[0]).astype(np.float32)
        action_np = policy.get_action(obs_np).astype(np.float32)
        action_t = torch.from_numpy(action_np).to(env.device).unsqueeze(0)

        step_out = env.step(action_t)
        obs = step_out[0]
        resets = step_out[3]

        root = get_root_state(env)
        dof_p = tensor_to_numpy(env.dof_pos[0]).astype(np.float32)
        dof_v = tensor_to_numpy(env.dof_vel[0]).astype(np.float32)
        qpos_like_step = np.concatenate([root[0:7], dof_p], axis=0)
        qvel_like_step = np.concatenate([root[7:13], dof_v], axis=0)

        actions.append(action_np)
        obs_list.append(get_current_obs(env))
        torques.append(tensor_to_numpy(env.torques[0]).astype(np.float32))
        root_pos.append(root[0:3].copy())
        root_quat_xyzw.append(root[3:7].copy())
        root_lin_vel.append(root[7:10].copy())
        root_ang_vel.append(root[10:13].copy())
        dof_pos.append(dof_p)
        dof_vel.append(dof_v)
        projected_gravity.append(tensor_to_numpy(env.projected_gravity[0]).astype(np.float32))
        L0.append(tensor_to_numpy(env.L0[0]).astype(np.float32))
        theta0.append(tensor_to_numpy(env.theta0[0]).astype(np.float32))
        theta0_ref.append(
            np.asarray(
                [
                    action_np[0] * ctrl_cfg.action_scale_theta,
                    action_np[3] * ctrl_cfg.action_scale_theta,
                ],
                dtype=np.float32,
            )
        )
        l0_ref.append(
            np.asarray(
                [
                    action_np[1] * ctrl_cfg.action_scale_l0 + ctrl_cfg.l0_offset,
                    action_np[4] * ctrl_cfg.action_scale_l0 + ctrl_cfg.l0_offset,
                ],
                dtype=np.float32,
            )
        )
        if hasattr(env, "force_leg"):
            force_leg.append(tensor_to_numpy(env.force_leg[0]).astype(np.float32))
        else:
            force_leg.append(np.full((2,), np.nan, dtype=np.float32))
        if hasattr(env, "torque_leg"):
            torque_leg.append(tensor_to_numpy(env.torque_leg[0]).astype(np.float32))
        else:
            torque_leg.append(np.full((2,), np.nan, dtype=np.float32))
        qpos_like.append(qpos_like_step.astype(np.float32))
        qvel_like.append(qvel_like_step.astype(np.float32))
        contacts.append(get_contact_flags(env, contact_index_map))
        done_flags.append(bool(resets[0].item()))

        if done_flags[-1]:
            print(f"[WARN] Isaac env reset triggered at step {t}; stopping collection early.")
            break

    actions_arr = np.asarray(actions, dtype=np.float32)
    obs_arr = np.asarray(obs_list, dtype=np.float32)
    torques_arr = np.asarray(torques, dtype=np.float32)
    root_pos_arr = np.asarray(root_pos, dtype=np.float32)
    root_quat_xyzw_arr = np.asarray(root_quat_xyzw, dtype=np.float32)
    root_lin_vel_arr = np.asarray(root_lin_vel, dtype=np.float32)
    root_ang_vel_arr = np.asarray(root_ang_vel, dtype=np.float32)
    dof_pos_arr = np.asarray(dof_pos, dtype=np.float32)
    dof_vel_arr = np.asarray(dof_vel, dtype=np.float32)
    projected_gravity_arr = np.asarray(projected_gravity, dtype=np.float32)
    L0_arr = np.asarray(L0, dtype=np.float32)
    theta0_arr = np.asarray(theta0, dtype=np.float32)
    theta0_ref_arr = np.asarray(theta0_ref, dtype=np.float32)
    l0_ref_arr = np.asarray(l0_ref, dtype=np.float32)
    force_leg_arr = np.asarray(force_leg, dtype=np.float32)
    torque_leg_arr = np.asarray(torque_leg, dtype=np.float32)
    qpos_like_arr = np.asarray(qpos_like, dtype=np.float32)
    qvel_like_arr = np.asarray(qvel_like, dtype=np.float32)
    contacts_arr = np.asarray(contacts, dtype=np.bool_)
    done_flags_arr = np.asarray(done_flags, dtype=np.bool_)

    metadata = {
        "task": args.task,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "seed": int(args.seed),
        "scenario": args.scenario,
        "domain_rand_mode": args.domain_rand_mode,
        "policy_device": args.policy_device,
        "sim_device": args.sim_device,
        "num_steps_requested": int(args.steps),
        "num_steps_collected": int(actions_arr.shape[0]),
        "controller_mode": "isaac_training_env_vmc_balance",
        "obs_dim": 27,
        "action_dim": 6,
        "notes": [
            "Collected from IsaacGym training environment using PolicyLoader on checkpoint .pt",
            "Quaternion stored as xyzw in root_state arrays",
            "contacts are boolean flags inferred from net contact force norms",
        ],
    }

    if output_path.parent != Path(""):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        metadata_json=np.array(json.dumps(metadata), dtype=object),
        init_root_state=init_root_state,
        init_dof_pos=init_dof_pos,
        init_dof_vel=init_dof_vel,
        actions=actions_arr,
        obs=obs_arr,
        torques=torques_arr,
        root_pos=root_pos_arr,
        root_quat_xyzw=root_quat_xyzw_arr,
        root_lin_vel=root_lin_vel_arr,
        root_ang_vel=root_ang_vel_arr,
        dof_pos=dof_pos_arr,
        dof_vel=dof_vel_arr,
        projected_gravity=projected_gravity_arr,
        L0=L0_arr,
        theta0=theta0_arr,
        theta0_ref=theta0_ref_arr,
        l0_ref=l0_ref_arr,
        force_leg=force_leg_arr,
        torque_leg=torque_leg_arr,
        qpos_like=qpos_like_arr,
        qvel_like=qvel_like_arr,
        contacts=contacts_arr,
        done_flags=done_flags_arr,
    )

    print("\nCollection finished.")
    print(f"Collected steps: {actions_arr.shape[0]}")
    print(f"Output saved: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
