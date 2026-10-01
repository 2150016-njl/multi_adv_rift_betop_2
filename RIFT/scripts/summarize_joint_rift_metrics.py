#!/usr/bin/env python
"""Summarize JSONL metrics emitted by audited Joint-RIFT evaluation."""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_KEYS = [
    "oracle_joint_gain",
    "oracle_policy_gap",
    "independent_same_as_oracle_rate",
    "joint_same_as_oracle_rate",
    "oracle_rank_1",
    "oracle_rank_2",
    "oracle_nll_1",
    "oracle_nll_2",
    "oracle_both_necessary",
    "both_necessary_rate",
    "nonfactorizable_reward_ratio",
    "pair_collision_rate",
    "reward_scale_p12_std",
    "reward_scale_coalition_std",
    "reward_scale_unary_pressure_std",
    "reward_scale_unary_return_std",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.jsonl.read_text().splitlines() if line.strip()]
    if not records:
        raise SystemExit("No Joint-RIFT metric records found")
    print(f"states={len(records)}")
    for key in DEFAULT_KEYS:
        values = np.asarray([r[key] for r in records if key in r], dtype=np.float64)
        if not values.size:
            continue
        print(
            f"{key}: mean={values.mean():.6f} std={values.std():.6f} "
            f"p50={np.median(values):.6f} p90={np.quantile(values, 0.9):.6f}"
        )


if __name__ == "__main__":
    main()
