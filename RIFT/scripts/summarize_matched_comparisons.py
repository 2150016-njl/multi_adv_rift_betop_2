#!/usr/bin/env python
"""Aggregate matched Joint-RIFT policy-vs-independent comparisons by seed."""

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


METRICS = (
    "score_composed",
    "score_route",
    "score_penalty",
    "cbv_collision_count",
    "cbv_count",
    "duration_game",
    "duration_system",
)


def summary(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std_sample": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "positive": int((values > 1e-9).sum()),
        "zero": int((np.abs(values) <= 1e-9).sum()),
        "negative": int((values < -1e-9).sum()),
    }


def load(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"Cannot read comparison JSON {path}: {error}") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("comparisons", nargs="*", type=Path)
    parser.add_argument(
        "--input_glob",
        action="append",
        default=[],
        help="Quoted filesystem glob; may be supplied more than once.",
    )
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args()

    paths = {path.resolve() for path in args.comparisons}
    for pattern in args.input_glob:
        paths.update(Path(path).resolve() for path in glob.glob(pattern))
    paths = sorted(paths)
    if not paths:
        raise SystemExit("Provide comparison JSON paths or --input_glob")

    seed_rows: List[Dict] = []
    route_rows = defaultdict(list)
    status_transitions = Counter()
    for path in paths:
        report = load(path)
        global_metrics = report.get("global_metrics", {})
        if not all(metric in global_metrics for metric in METRICS[:3]):
            raise SystemExit(f"Missing global score metrics in {path}")
        seed_rows.append(
            {
                "comparison_json": str(path),
                **{
                    metric: float(global_metrics[metric]["policy_minus_independent"])
                    for metric in METRICS[:3]
                },
            }
        )
        per_route = report.get("per_route_deltas", [])
        if not per_route:
            raise SystemExit(f"No matched route rows in {path}")
        seed_rows[-1].update(
            {
                "total_cbv_collision_count": float(
                    sum(row["cbv_collision_count"] for row in per_route)
                ),
                "total_cbv_count": float(sum(row["cbv_count"] for row in per_route)),
                "mean_duration_game": float(
                    np.mean([row["duration_game"] for row in per_route])
                ),
                "mean_duration_system": float(
                    np.mean([row["duration_system"] for row in per_route])
                ),
            }
        )
        for row in per_route:
            route_rows[row["route_id"]].append(row)
            status_transitions[
                f"{row['independent_status']} -> {row['policy_status']}"
            ] += 1

    report = {
        "comparison_files": [str(path) for path in paths],
        "seed_level_global_deltas": {
            metric: summary(row[metric] for row in seed_rows)
            for metric in METRICS[:3]
        },
        "seed_level_safety_and_cost_deltas": {
            metric: summary(row[metric] for row in seed_rows)
            for metric in (
                "total_cbv_collision_count",
                "total_cbv_count",
                "mean_duration_game",
                "mean_duration_system",
            )
        },
        "per_seed_global_deltas": seed_rows,
        "route_level_descriptive_deltas": {
            route_id: {
                metric: summary(float(row[metric]) for row in rows)
                for metric in METRICS
            }
            for route_id, rows in sorted(route_rows.items())
        },
        "status_transitions": dict(status_transitions),
        "audit_note": (
            "Primary inference unit is the seed-level global matched delta. "
            "Route rows are descriptive because routes within one seed are correlated."
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
