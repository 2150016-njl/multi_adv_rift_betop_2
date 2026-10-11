#!/usr/bin/env python
"""Summarize provenance-rich Joint-RIFT oracle diagnostics.

This reports evaluator/candidate-set upper-bound evidence only.  It does not
turn an oracle-gated run into a deployable-policy result.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def _load_records(path: Path) -> List[Dict]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"No diagnostics in {path}")
    return records


def _is_high_value(record: Dict, args: argparse.Namespace) -> bool:
    return (
        float(record.get("oracle_joint_gain", 0.0)) >= args.min_joint_gain
        and float(record.get("nonfactorizable_reward_ratio", 0.0))
        >= args.min_nonfactorizable_ratio
        and (
            not args.require_both_necessary
            or float(record.get("oracle_both_necessary", 0.0)) >= 0.5
        )
        and float(record.get("oracle_pair_collision", 1.0)) <= 0.0
        and float(record.get("pair_collision_rate", 1.0))
        <= args.max_pair_collision_rate
    )


def _summary(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--min_joint_gain", type=float, default=10.0)
    parser.add_argument("--min_nonfactorizable_ratio", type=float, default=0.01)
    parser.add_argument("--max_pair_collision_rate", type=float, default=0.05)
    parser.add_argument("--require_both_necessary", action="store_true")
    parser.add_argument("--output_json", type=Path)
    parser.add_argument(
        "--output_high_value_jsonl",
        type=Path,
        help="Optional exact state-level subset retained for replay/inspection.",
    )
    args = parser.parse_args()

    records = _load_records(args.jsonl)
    high_value = [record for record in records if _is_high_value(record, args)]
    selected_modes = Counter(record.get("oracle_eval_mode", "legacy") for record in records)
    action_modes = Counter(
        record.get("joint_eval_action_mode", "legacy") for record in records
    )
    gate_failures = Counter(
        failure
        for record in records
        for failure in record.get("oracle_gate_failures", [])
    )
    route_groups = defaultdict(list)
    for record in high_value:
        route_key = (
            record.get("route_id", "unknown"),
            record.get("town_name", "unknown"),
            record.get("repetition_index", -1),
        )
        route_groups[route_key].append(record)

    metrics = (
        "oracle_joint_gain",
        "oracle_policy_gap",
        "nonfactorizable_reward_ratio",
        "both_necessary_rate",
        "pair_collision_rate",
        "oracle_nll_1",
        "oracle_nll_2",
    )
    report = {
        "states": len(records),
        "high_value_states": len(high_value),
        "high_value_rate": len(high_value) / len(records),
        "oracle_gate_pass_states": sum(
            bool(record.get("oracle_gate_pass", False)) for record in records
        ),
        "oracle_intervention_states": sum(
            bool(record.get("oracle_intervention", False)) for record in records
        ),
        # These equality counts are intentionally separate from action-mode
        # counts: a policy can choose exactly the same pair as independent or
        # oracle, so they are not mutually exclusive execution categories.
        "execution_matches_policy_states": sum(
            bool(record.get("executed_is_policy", False)) for record in records
        ),
        "execution_matches_independent_states": sum(
            bool(record.get("executed_is_independent", False)) for record in records
        ),
        "execution_matches_oracle_states": sum(
            bool(record.get("executed_is_oracle", False)) for record in records
        ),
        "oracle_eval_modes": dict(selected_modes),
        "joint_eval_action_modes": dict(action_modes),
        "oracle_gate_failures": dict(gate_failures),
        "metrics": {
            metric: _summary(record[metric] for record in records if metric in record)
            for metric in metrics
        },
        "high_value_route_clusters": [
            {
                "route_id": route_id,
                "town_name": town_name,
                "repetition_index": repetition_index,
                "states": len(group),
                "first_time_step": min(record.get("time_step", -1) for record in group),
                "last_time_step": max(record.get("time_step", -1) for record in group),
                "max_oracle_joint_gain": max(
                    float(record["oracle_joint_gain"]) for record in group
                ),
                "pair_ids": sorted({tuple(record["pair_ids"]) for record in group}),
            }
            for (route_id, town_name, repetition_index), group in route_groups.items()
        ],
    }
    report["high_value_route_clusters"].sort(
        key=lambda item: item["max_oracle_joint_gain"], reverse=True
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.output_high_value_jsonl:
        args.output_high_value_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_high_value_jsonl.open("w", encoding="utf-8") as stream:
            for record in high_value:
                stream.write(json.dumps(record, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
