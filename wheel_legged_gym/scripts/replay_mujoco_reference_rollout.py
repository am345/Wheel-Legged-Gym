#!/usr/bin/env python3
"""Replay an Isaac reference rollout in MuJoCo and report alignment metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from mujoco_sim.alignment_utils import replay_reference_rollout_in_mujoco
from mujoco_sim.mujoco_balance_env import MuJoCoBalanceEnv

DEFAULT_MODEL = "resources/robots/serialleg/mjcf/serialleg_fidelity.xml"


def resolve_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay Isaac reference rollout in MuJoCo for sim2sim alignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-rollout", type=str, required=True, help="Isaac reference .npz")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="MuJoCo MJCF path")
    parser.add_argument(
        "--controller-mode",
        type=str,
        default="vmc_balance_exact",
        choices=list(MuJoCoBalanceEnv.SUPPORTED_CONTROLLER_MODES),
    )
    parser.add_argument(
        "--domain-rand-mode",
        type=str,
        default="off",
        choices=list(MuJoCoBalanceEnv.SUPPORTED_DOMAIN_RAND_MODES),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=str, default="mujoco_alignment_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ref_path = resolve_path(args.reference_rollout)
    model_path = resolve_path(args.model)
    out_path = Path(args.output).expanduser()

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference rollout not found: {ref_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found: {model_path}")

    report = replay_reference_rollout_in_mujoco(
        reference_rollout_path=ref_path,
        model_path=model_path,
        controller_mode=args.controller_mode,
        domain_rand_mode=args.domain_rand_mode,
        seed=args.seed,
        max_steps=args.max_steps,
    )

    if out_path.parent != Path(""):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n=== MuJoCo Alignment Replay Summary ===")
    print(f"reference: {ref_path}")
    print(f"model:     {model_path}")
    print(f"steps:     {report['metadata']['num_steps_replayed']}")
    print(f"obs_rmse_total:    {report['summary']['obs_rmse_total']}")
    print(f"torque_rmse_total: {report['summary']['torque_rmse_total']}")
    print(f"output:    {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
