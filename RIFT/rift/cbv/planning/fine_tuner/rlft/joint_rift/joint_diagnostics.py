"""Pure NumPy diagnostics used to verify that Joint-RIFT learns synergy."""

from typing import Dict, Tuple

import numpy as np


def masked_additive_residual_ratio(values: np.ndarray, valid_mask: np.ndarray) -> float:
    """Fraction of variance that cannot be explained by row+column effects."""
    values = np.asarray(values, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if values.shape != valid_mask.shape or values.ndim != 2:
        raise ValueError("values and valid_mask must share shape [G1, G2]")
    valid_rows = valid_mask.any(axis=1)
    valid_cols = valid_mask.any(axis=0)
    sub_values = values[np.ix_(valid_rows, valid_cols)]
    sub_mask = valid_mask[np.ix_(valid_rows, valid_cols)]
    if not sub_mask.all():
        # Joint-RIFT masks are an outer product in the current design.  Refuse
        # irregular masks instead of reporting a misleading decomposition.
        raise ValueError("additive diagnostic requires an outer-product valid mask")
    if sub_values.size < 2:
        return 0.0
    grand_mean = sub_values.mean()
    additive = (
        sub_values.mean(axis=1, keepdims=True)
        + sub_values.mean(axis=0, keepdims=True)
        - grand_mean
    )
    residual = sub_values - additive
    total_var = float(sub_values.var())
    if total_var < 1e-12:
        return 0.0
    return float(residual.var() / total_var)


def rank_and_nll(logits: np.ndarray, index: int) -> Tuple[int, float]:
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not 0 <= index < logits.size:
        raise IndexError("candidate index out of range")
    rank = int(1 + np.count_nonzero(logits > logits[index]))
    shifted = logits - logits.max()
    logsumexp = float(np.log(np.exp(shifted).sum()) + logits.max())
    nll = float(logsumexp - logits[index])
    return rank, nll


def select_oracle_pair(joint_q: np.ndarray, valid_pair_mask: np.ndarray) -> Tuple[int, int]:
    """Return the evaluator-oracle argmax over valid joint candidates only."""
    joint_q = np.asarray(joint_q, dtype=np.float64)
    valid_pair_mask = np.asarray(valid_pair_mask, dtype=bool)
    if joint_q.shape != valid_pair_mask.shape or joint_q.ndim != 2:
        raise ValueError("joint_q and valid_pair_mask must share shape [G1, G2]")
    if not valid_pair_mask.any():
        raise ValueError("at least one valid pair is required")
    flat_index = int(np.argmax(np.where(valid_pair_mask, joint_q, -np.inf)))
    return tuple(int(v) for v in np.unravel_index(flat_index, joint_q.shape))


def oracle_gate_decision(
    diagnostics: Dict[str, float],
    min_joint_gain: float,
    min_nonfactorizable_ratio: float,
    require_both_necessary: bool,
    max_pair_collision_rate: float,
) -> Tuple[bool, Tuple[str, ...]]:
    """Return an auditable, safety-first oracle intervention decision."""
    if min_joint_gain < 0 or min_nonfactorizable_ratio < 0:
        raise ValueError("oracle gate minima must be non-negative")
    if not 0 <= max_pair_collision_rate <= 1:
        raise ValueError("max_pair_collision_rate must be in [0, 1]")

    failures = []
    if diagnostics['oracle_joint_gain'] < min_joint_gain:
        failures.append('joint_gain')
    if diagnostics['nonfactorizable_reward_ratio'] < min_nonfactorizable_ratio:
        failures.append('nonfactorizable_ratio')
    if require_both_necessary and diagnostics['oracle_both_necessary'] < 0.5:
        failures.append('both_necessary')
    if diagnostics['oracle_pair_collision'] > 0:
        failures.append('oracle_pair_collision')
    if diagnostics['pair_collision_rate'] > max_pair_collision_rate:
        failures.append('candidate_pair_collision_rate')
    return not failures, tuple(failures)


def build_feasibility_diagnostics(
    joint_q: np.ndarray,
    valid_pair_mask: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p12: np.ndarray,
    coalition_gain: np.ndarray,
    pair_collision: np.ndarray,
    logits_1: np.ndarray,
    logits_2: np.ndarray,
    unary_return_1: np.ndarray,
    unary_return_2: np.ndarray,
    selected_indices: Tuple[int, int],
    independent_indices: Tuple[int, int],
    both_necessary_epsilon: float = 0.1,
) -> Dict[str, float]:
    joint_q = np.asarray(joint_q, dtype=np.float64)
    valid_pair_mask = np.asarray(valid_pair_mask, dtype=bool)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p12 = np.asarray(p12, dtype=np.float64)
    coalition_gain = np.asarray(coalition_gain, dtype=np.float64)
    pair_collision = np.asarray(pair_collision, dtype=bool)
    unary_return_1 = np.asarray(unary_return_1, dtype=np.float64)
    unary_return_2 = np.asarray(unary_return_2, dtype=np.float64)
    if not valid_pair_mask.any():
        raise ValueError("at least one valid pair is required")

    oracle_indices = select_oracle_pair(joint_q, valid_pair_mask)
    oracle_q = float(joint_q[oracle_indices])
    selected_q = float(joint_q[selected_indices])
    independent_q = float(joint_q[independent_indices])

    marginal_1 = p12 - p2[None, :]
    marginal_2 = p12 - p1[:, None]
    both_necessary = (
        valid_pair_mask
        & (marginal_1 > both_necessary_epsilon)
        & (marginal_2 > both_necessary_epsilon)
    )
    rank_1, nll_1 = rank_and_nll(logits_1, oracle_indices[0])
    rank_2, nll_2 = rank_and_nll(logits_2, oracle_indices[1])
    unary_pressure_sum = p1[:, None] + p2[None, :]
    unary_return_sum = unary_return_1[:, None] + unary_return_2[None, :]

    return {
        "oracle_joint_gain": oracle_q - independent_q,
        "oracle_policy_gap": oracle_q - selected_q,
        "oracle_q": oracle_q,
        "selected_q": selected_q,
        "independent_q": independent_q,
        "independent_same_as_joint_rate": float(selected_indices == independent_indices),
        "independent_same_as_oracle_rate": float(independent_indices == oracle_indices),
        "joint_same_as_oracle_rate": float(selected_indices == oracle_indices),
        "oracle_rank_1": float(rank_1),
        "oracle_rank_2": float(rank_2),
        "oracle_nll_1": nll_1,
        "oracle_nll_2": nll_2,
        "oracle_marginal_1": float(marginal_1[oracle_indices]),
        "oracle_marginal_2": float(marginal_2[oracle_indices]),
        "oracle_both_necessary": float(both_necessary[oracle_indices]),
        "oracle_index_1": float(oracle_indices[0]),
        "oracle_index_2": float(oracle_indices[1]),
        "oracle_pair_collision": float(pair_collision[oracle_indices]),
        "policy_pair_collision": float(pair_collision[selected_indices]),
        "both_necessary_rate": float(both_necessary[valid_pair_mask].mean()),
        "nonfactorizable_reward_ratio": masked_additive_residual_ratio(joint_q, valid_pair_mask),
        "pair_collision_rate": float(pair_collision[valid_pair_mask].mean()),
        "reward_scale_p12_std": float(p12[valid_pair_mask].std()),
        "reward_scale_coalition_std": float(coalition_gain[valid_pair_mask].std()),
        "reward_scale_unary_pressure_std": float(unary_pressure_sum[valid_pair_mask].std()),
        "reward_scale_unary_return_std": float(unary_return_sum[valid_pair_mask].std()),
    }
