#!/usr/bin/env python
"""Summarize the paired Uniform-versus-HV4 Joint-RIFT ablation by seed."""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


SCORES = ("score_composed", "score_route", "score_penalty")
SEED_PATTERN = re.compile(r"seed(\d+)")


def summarize(values: Iterable[float]) -> Dict[str, float]:
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std_sample": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def seed_from_path(path: Path) -> int:
    match = SEED_PATTERN.search(path.name)
    if not match:
        raise SystemExit(f"Cannot infer seed from comparison filename: {path}")
    return int(match.group(1))


def load_by_seed(paths: Iterable[Path]) -> Dict[int, Dict]:
    reports = {}
    for path in paths:
        seed = seed_from_path(path)
        if seed in reports:
            raise SystemExit(f"Duplicate comparison for seed {seed}: {path}")
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"Cannot read {path}: {error}") from error
        metrics = report.get("global_metrics", {})
        if not all(score in metrics for score in SCORES):
            raise SystemExit(f"Missing score metrics in {path}")
        reports[seed] = {"path": str(path), "metrics": metrics}
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform", type=Path, nargs="+", required=True)
    parser.add_argument("--hv4", type=Path, nargs="+", required=True)
    parser.add_argument("--output_json", type=Path)
    args = parser.parse_args()

    uniform = load_by_seed(args.uniform)
    hv4 = load_by_seed(args.hv4)
    if set(uniform) != set(hv4):
        raise SystemExit(
            f"Seed mismatch: uniform={sorted(uniform)}, hv4={sorted(hv4)}"
        )

    rows = []
    for seed in sorted(uniform):
        row = {"seed": seed, "uniform_comparison": uniform[seed]["path"],
               "hv4_comparison": hv4[seed]["path"]}
        for score in SCORES:
            uniform_delta = float(
                uniform[seed]["metrics"][score]["policy_minus_independent"]
            )
            hv4_delta = float(hv4[seed]["metrics"][score]["policy_minus_independent"])
            row[f"uniform_{score}_delta"] = uniform_delta
            row[f"hv4_{score}_delta"] = hv4_delta
            row[f"hv4_minus_uniform_{score}_interaction"] = hv4_delta - uniform_delta
        rows.append(row)

    report = {
        "per_seed": rows,
        "uniform_policy_minus_independent": {
            score: summarize(row[f"uniform_{score}_delta"] for row in rows)
            for score in SCORES
        },
        "hv4_policy_minus_independent": {
            score: summarize(row[f"hv4_{score}_delta"] for row in rows)
            for score in SCORES
        },
        "primary_difference_in_differences": {
            score: summarize(
                row[f"hv4_minus_uniform_{score}_interaction"] for row in rows
            )
            for score in SCORES
        },
        "audit_note": (
            "Primary effect: same-seed [HV4(policy-independent) minus "
            "Uniform(policy-independent)]. A positive score_composed interaction "
            "supports the high-value reweighting ablation."
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
