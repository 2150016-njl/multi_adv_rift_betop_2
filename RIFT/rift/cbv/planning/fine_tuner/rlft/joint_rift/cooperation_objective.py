"""Fixed-nominal coverage objective; no response model or causal claim.

Unary returns define a quality budget, rather than overwhelming the ranking.
The support remains the original Cartesian product; rejected pairs receive a
bounded bad score, so collection, training and inference use the same action
space. Feasibility is a training target, NOT an inference safety guarantee.
"""

import numpy as np


def cooperation_objective(evaluation, independent, config=None):
    cfg = config or {}
    valid = np.asarray(evaluation["valid_mask"], dtype=bool)
    p12 = np.asarray(evaluation["p12"], dtype=np.float64)
    cg = np.asarray(evaluation["coalition_gain"], dtype=np.float64)
    u1 = np.asarray(evaluation["unary_return_1"], dtype=np.float64)
    u2 = np.asarray(evaluation["unary_return_2"], dtype=np.float64)
    if valid.ndim != 2 or p12.shape != valid.shape or cg.shape != valid.shape:
        raise ValueError("cooperation matrices must share [G1,G2]")
    if u1.shape != (valid.shape[0],) or u2.shape != (valid.shape[1],):
        raise ValueError("unary shapes do not match candidate axes")
    if not valid.any() or not valid[independent]:
        raise ValueError("independent pair must be valid")
    if not all(np.isfinite(v).all() for v in (p12[valid], cg[valid], u1, u2)):
        raise ValueError("nonfinite evaluator values")
    budget = float(cfg.get("unary_budget", 0.0))
    coalition_weight = float(cfg.get("coalition_weight", 1.0))
    gain_min = float(cfg.get("min_gain", 0.1))
    scale_floor = float(cfg.get("interaction_scale_floor", 0.1))
    if budget < 0 or coalition_weight < 0 or gain_min < 0 or scale_floor <= 0:
        raise ValueError("invalid cooperation settings")
    hard = valid & ~np.asarray(evaluation["pair_collision"], dtype=bool)
    for name, enabled in (("collision", cfg.get("require_unary_collision_free", True)),
                          ("offroad", cfg.get("require_offroad_free", True)),
                          ("uncomfortable", cfg.get("require_comfortable", False))):
        if enabled:
            for axis in (1, 2):
                key = "unary_" + name + "_" + str(axis)
                if key not in evaluation:
                    raise ValueError("missing " + key + "; recollect schema-v2 cache or explicitly disable this constraint")
                bad = np.asarray(evaluation[key], dtype=bool)
                if bad.shape != (valid.shape[axis - 1],):
                    raise ValueError("incorrect quality flag shape: " + key)
                hard &= ~bad[:, None] if axis == 1 else ~bad[None, :]
    feasible = (hard & (u1[:, None] >= u1[independent[0]] - budget)
                & (u2[None, :] >= u2[independent[1]] - budget))
    if cfg.get("require_both", False):
        feasible &= cg > float(cfg.get("both_epsilon", .1))
    interaction = p12 + coalition_weight * cg
    target = np.zeros_like(interaction)
    target[valid] = -1.0
    # Empty feasibility is explicit: bounded binary safety/quality recovery,
    # no synthetic "cooperative oracle" and no silent budget relaxation.
    if feasible.any():
        low = float(interaction[feasible].min())
        span = float(interaction[feasible].max() - low)
        target[feasible] = 1.0 + (interaction[feasible] - low) / max(span, scale_floor)
        flat = int(np.where(feasible, interaction, -np.inf).argmax())
        oracle = tuple(int(x) for x in np.unravel_index(flat, valid.shape))
        gain = float(interaction[oracle] - interaction[independent])
    else:
        oracle, gain = None, 0.0
    values = target[valid]
    advantage = np.zeros_like(target)
    advantage[valid] = (values - values.mean()) / (values.std() + 1e-5)
    both = feasible & (cg > float(cfg.get("both_epsilon", .1)))
    metrics = {
        "coop_feasible_rate": float(feasible[valid].mean()),
        "coop_hard_safe_rate": float(hard[valid].mean()),
        "coop_has_feasible": float(feasible.any()),
        "coop_gain": gain,
        "coop_opportunity": float(feasible.any() and gain >= gain_min),
        "coop_nominal_both_rate": float(both[valid].mean()),
        "coop_oracle_nominal_both": float(both[oracle]) if oracle is not None else 0.,
        "coop_independent_feasible": float(feasible[independent]),
    }
    return {"target_score": target, "advantage": advantage, "feasible_mask": feasible,
            "interaction": interaction, "oracle": oracle, "metrics": metrics}
