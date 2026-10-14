"""Teacher decisions separating quality repair, unary interaction and joint value.

No causal Ego-response claim. These decisions use the evaluator and must never
be reported as learned policy results. All ties prefer the reference action.
"""
import numpy as np


def stable_best(score, mask, reference, prior, tolerance=1e-8):
    if not np.any(mask):
        return None
    maximum = float(np.max(score[mask]))
    tied = mask & (score >= maximum - tolerance)
    if tied[reference]:
        return reference
    flat = int(np.where(tied, prior, -np.inf).argmax())
    return tuple(int(x) for x in np.unravel_index(flat, mask.shape))


def decision_teacher(evaluation, independent, base_logits, quality_config=None, config=None):
    # Import is intentionally local so this module remains source-loadable
    # without importing the simulator package.
    cfg, qcfg = config or {}, quality_config or {}
    d = evaluation
    valid = np.asarray(d.get("valid_mask", d.get("valid_pair_mask")), dtype=bool)
    prior = np.asarray(base_logits, dtype=float)
    if prior.shape != valid.shape or not valid[independent] or not np.isfinite(prior[valid]).all():
        raise ValueError("valid independent action and finite matching prior required")
    budget = float(qcfg.get("unary_budget", 0.))
    min_gain = float(cfg.get("min_gain", .05))
    min_joint_gain = float(cfg.get("min_joint_gain", .01))
    min_member = float(cfg.get("min_member_pressure", .01))
    min_coalition = float(cfg.get("min_coalition", .01))
    pw, cw = float(cfg.get("pressure_weight", 1.)), float(cfg.get("coalition_weight", 1.))
    if min(budget, min_gain, min_joint_gain, min_member, min_coalition, pw, cw) < 0:
        raise ValueError("teacher thresholds and weights must be nonnegative")
    feasible = valid & ~np.asarray(d["pair_collision"], dtype=bool)
    for name, enabled in (("collision", qcfg.get("require_unary_collision_free", True)),
                          ("offroad", qcfg.get("require_offroad_free", True)),
                          ("uncomfortable", qcfg.get("require_comfortable", False))):
        if enabled:
            for axis in (1, 2):
                collision_label = cfg.get("nominal_conflict_policy", "reject")
                if collision_label not in {"reject", "label_only"}:
                    raise ValueError("nominal_conflict_policy must be reject or label_only")
                key = "unary_" + name + "_" + str(axis)
                if name == "collision" and collision_label == "label_only":
                    key = "unary_collision_background_" + str(axis)
                    if key not in d:
                        raise ValueError("split Ego/background collision labels absent; recollect V6 cache")
                bad = np.asarray(d[key], dtype=bool)
                feasible &= ~bad[:, None] if axis == 1 else ~bad[None, :]
    u1, u2 = np.asarray(d["unary_return_1"]), np.asarray(d["unary_return_2"])
    feasible &= (u1[:, None] >= u1[independent[0]] - budget) & (u2[None, :] >= u2[independent[1]] - budget)
    p1, p2 = np.asarray(d["p1"]), np.asarray(d["p2"])
    cg, p12 = np.asarray(d["coalition_gain"]), np.asarray(d["p12"])
    score = pw * p12 + cw * cg
    if not all(np.isfinite(x).all() for x in (u1, u2, p1, p2, score)):
        raise ValueError("nonfinite teacher values")
    quality = stable_best(prior, feasible, independent, prior) or independent
    additive = stable_best(p1[:, None] + p2[None, :], feasible, quality, prior) or quality
    maximum = stable_best(score, feasible, quality, prior) or quality
    interaction = maximum if feasible.any() and score[maximum] - score[quality] >= min_gain else quality
    bilateral = feasible & (p1[:, None] >= min_member) & (p2[None, :] >= min_member) & (cg >= min_coalition)
    joint_maximum = stable_best(score, bilateral, quality, prior)
    joint_support = bool(joint_maximum is not None
        and score[joint_maximum] - score[quality] >= min_gain
        and score[joint_maximum] - score[additive] >= min_joint_gain)
    joint = joint_maximum if joint_support else quality
    return {"feasible_mask": feasible, "score": score,
            "quality": quality, "factorized": additive, "interaction": interaction, "joint": joint,
            "metrics": {"decision_has_feasible": float(feasible.any()),
                "decision_quality_repair": float(quality != independent),
                "decision_interaction_support": float(interaction != quality),
                "decision_joint_support": float(joint_support),
                "decision_bilateral_support": float(bilateral.any()),
                "decision_quality_score": float(score[quality]),
                "decision_factorized_score": float(score[additive]),
                "decision_joint_increment": float(score[joint] - score[additive]) if joint_support else 0.}}


def verified_decision(teacher, proposal, evaluation, config=None):
    """Model-based execution: certify a student proposal or use the teacher.

Guarantees concern this fixed-nominal evaluator only. Empty feasible support is
explicit; no prediction of real-world safety or long-horizon return is made.
"""
    cfg = config or {}
    kind = cfg.get("verification_target", "interaction")
    if kind not in {"interaction", "joint"}:
        raise ValueError("verification_target must be interaction or joint")
    f, score, quality = teacher["feasible_mask"], teacher["score"], teacher["quality"]
    accepted = bool(f[proposal] and score[proposal] - score[quality] >= float(cfg.get("min_gain", .05)))
    if kind == "joint":
        d = evaluation
        accepted = bool(accepted and teacher["metrics"]["decision_joint_support"]
            and score[proposal] - score[teacher["factorized"]] >= float(cfg.get("min_joint_gain", .01))
            and d["p1"][proposal[0]] >= float(cfg.get("min_member_pressure", .01))
            and d["p2"][proposal[1]] >= float(cfg.get("min_member_pressure", .01))
            and d["coalition_gain"][proposal] >= float(cfg.get("min_coalition", .01)))
    selected = proposal if accepted else teacher[kind]
    return selected, {"decision_used_verifier": 1., "decision_student_accepted": float(accepted),
                      "decision_verifier_changed_proposal": float(selected != proposal)}
