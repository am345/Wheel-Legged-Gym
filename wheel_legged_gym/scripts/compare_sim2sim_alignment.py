#!/usr/bin/env python3
"""Aggregate multiple MuJoCo alignment JSON reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare multiple sim2sim alignment reports",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("reports", nargs="+", help="Alignment report JSON files")
    p.add_argument("--output", type=str, default=None, help="Optional aggregated JSON output")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for rp in args.reports:
        path = Path(rp).expanduser()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows.append(
            {
                "path": str(path),
                "reference": data.get("metadata", {}).get("reference_rollout_path"),
                "model": data.get("metadata", {}).get("model_path"),
                "controller_mode": data.get("metadata", {}).get("controller_mode"),
                "seed": data.get("metadata", {}).get("seed"),
                "steps": data.get("metadata", {}).get("num_steps_replayed"),
                "obs_rmse_total": data.get("summary", {}).get("obs_rmse_total"),
                "torque_rmse_total": data.get("summary", {}).get("torque_rmse_total"),
                "state_rmse": data.get("state_rmse", {}),
                "contact_alignment": data.get("contact_alignment", {}),
            }
        )

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            float("inf") if r["obs_rmse_total"] is None else float(r["obs_rmse_total"]),
            float("inf") if r["torque_rmse_total"] is None else float(r["torque_rmse_total"]),
        ),
    )

    print("\n=== Sim2Sim Alignment Comparison ===")
    for i, r in enumerate(rows_sorted, 1):
        print(
            f"{i:2d}. {Path(r['path']).name}: obs_rmse={r['obs_rmse_total']} "
            f"torque_rmse={r['torque_rmse_total']} steps={r['steps']} ctrl={r['controller_mode']}"
        )

    if args.output:
        out = Path(args.output).expanduser()
        if out.parent != Path(""):
            out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"reports": rows_sorted}, f, indent=2, ensure_ascii=False)
        print(f"\nAggregated report saved: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
