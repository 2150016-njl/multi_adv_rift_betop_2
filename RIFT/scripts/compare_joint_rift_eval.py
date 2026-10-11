#!/usr/bin/env python
"""Compare matched learned-joint and independent-candidate CARLA evaluations."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def load_result(path: Path) -> Dict:
    payload = json.loads(path.read_text())
    checkpoint = payload.get("_checkpoint", payload)
    global_record = checkpoint.get("global_record", payload.get("global_record", {}))
    records = checkpoint.get("records", payload.get("records", []))
    if not global_record or not records:
        raise SystemExit(f"Expected completed simulation results with route records: {path}")
    return {"path": str(path), "global": global_record, "records": records}


def numeric_summary(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def record_metric(record: Dict, metric: str) -> float:
    if metric.startswith("score_"):
        return float(record.get("scores", {}).get(metric, 0.0))
    return float(record.get("meta", {}).get(metric, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args()

    policy = load_result(args.policy)
    independent = load_result(args.independent)
    policy_routes = {record["route_id"]: record for record in policy["records"]}
    independent_routes = {
        record["route_id"]: record for record in independent["records"]
    }
    common_ids = sorted(set(policy_routes) & set(independent_routes))
    if not common_ids:
        raise SystemExit("No common route_id values; evaluations are not matched")

    metrics = (
        "score_composed",
        "score_route",
        "score_penalty",
        "cbv_collision_count",
        "cbv_count",
        "duration_game",
        "duration_system",
    )
    route_deltas = []
    for route_id in common_ids:
        policy_record = policy_routes[route_id]
        independent_record = independent_routes[route_id]
        route_deltas.append(
            {
                "route_id": route_id,
                "policy_status": policy_record.get("status", "Unknown"),
                "independent_status": independent_record.get("status", "Unknown"),
                **{
                    metric: record_metric(policy_record, metric)
                    - record_metric(independent_record, metric)
                    for metric in metrics
                },
            }
        )

    global_metrics = {}
    for metric in ("score_composed", "score_route", "score_penalty"):
        policy_value = float(policy["global"].get("scores_mean", {}).get(metric, 0.0))
        independent_value = float(
            independent["global"].get("scores_mean", {}).get(metric, 0.0)
        )
        global_metrics[metric] = {
            "policy": policy_value,
            "independent": independent_value,
            "policy_minus_independent": policy_value - independent_value,
        }

    report = {
        "policy_result": policy["path"],
        "independent_result": independent["path"],
        "matched_routes": len(common_ids),
        "policy_only_routes": sorted(set(policy_routes) - set(independent_routes)),
        "independent_only_routes": sorted(set(independent_routes) - set(policy_routes)),
        "global_metrics": global_metrics,
        "route_delta_summary": {
            metric: numeric_summary(row[metric] for row in route_deltas)
            for metric in metrics
        },
        "score_composed_wins_ties_losses": {
            "policy_wins": sum(row["score_composed"] > 1e-9 for row in route_deltas),
            "ties": sum(abs(row["score_composed"]) <= 1e-9 for row in route_deltas),
            "policy_losses": sum(row["score_composed"] < -1e-9 for row in route_deltas),
        },
        "status_transitions": dict(
            Counter(
                f"{row['independent_status']} -> {row['policy_status']}"
                for row in route_deltas
            )
        ),
        "per_route_deltas": route_deltas,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
