"""Summarize executed cooperation diagnostics by run and route, not pooled frames."""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


def summarize(paths):
    report = []
    for path in paths:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        groups = defaultdict(list)
        for row in records:
            groups[str(row["route_id"])].append(row)
        keys = ("coop_opportunity", "coop_has_feasible", "coop_feasible_rate", "coop_gain",
                "coop_executed_feasible", "coop_executed_gain", "coop_nominal_both_rate")
        if not records or any(key not in r for r in records for key in keys):
            raise ValueError("missing cooperation/executed diagnostics in " + str(path))
        per_route = {route: {"joint_states": len(rows), **{k: float(np.mean([r[k] for r in rows])) for k in keys}}
                     for route, rows in groups.items()}
        report.append({"path": str(path), "joint_states": len(records), "routes": len(groups),
                       "route_equal_mean": {k: float(np.mean([r[k] for r in per_route.values()])) for k in keys},
                       "per_route": per_route,
                       "warning": "Descriptive route-balanced means. Different policies encounter different states; pair rollout outcomes by route/repetition for effect inference. No frame-level significance claim."})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.inputs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
