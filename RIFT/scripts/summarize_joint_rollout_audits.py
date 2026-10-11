#!/usr/bin/env python
"""Summarize persisted Joint-RIFT rollout-composition audits."""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(values):
    # Call sites intentionally pass generators so audit files are streamed into
    # this helper.  NumPy cannot cast a generator directly to a numeric array.
    values = np.asarray(list(values), dtype=np.float64)
    if not values.size:
        return {"n": 0, "mean": None, "p50": None, "min": None, "max": None}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_dir", type=Path)
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args()

    def episode_number(path: Path) -> int:
        return int(path.stem.rsplit("=", 1)[1])

    paths = sorted(
        args.audit_dir.glob("joint_rollout_audit_episode=*.json"),
        key=episode_number,
    )
    if not paths:
        raise SystemExit(f"No rollout audits found in {args.audit_dir}")
    audits = [json.loads(path.read_text()) for path in paths]
    report = {
        "audit_files": [str(path) for path in paths],
        "training_updates": len(audits),
        "carla_episodes": [int(audit["carla_episode"]) for audit in audits],
        "states_per_update": summarize(audit["states"] for audit in audits),
        "high_value_states_per_update": summarize(
            audit["high_value_states"] for audit in audits
        ),
        "high_value_rate_per_update": summarize(
            audit["high_value_rate"] for audit in audits
        ),
        "high_value_eligible_states_per_update": summarize(
            audit.get("high_value_eligible_states", 0) for audit in audits
        ),
        "high_value_eligible_rate_per_update": summarize(
            audit.get("high_value_eligible_rate", 0.0) for audit in audits
        ),
        "state_weight_mean_per_update": summarize(
            audit["state_weight"]["mean"] for audit in audits
        ),
        "diagnostic_means_per_update": {
            key: summarize(
                audit["diagnostics"][key]["mean"]
                for audit in audits
                if key in audit.get("diagnostics", {})
            )
            for key in sorted(
                {
                    key
                    for audit in audits
                    for key in audit.get("diagnostics", {})
                }
            )
        },
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
