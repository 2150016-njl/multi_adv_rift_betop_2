"""Pair relevance heuristic using shared conflict region and arrival proximity.

This is a collection proposal, not a proof that cooperative trajectories exist.
An existing member is retained; active pairs are not switched mid-segment.
"""
import itertools
import math


def select_conflict_pair(records, active_ids=()):
    active = set(active_ids)
    if len(active) >= 2:
        return []
    pool = [r for r in records if all(math.isfinite(float(r[k])) for k in ("ego_eta", "cbv_eta", "conflict_x", "conflict_y"))]
    best, best_score = None, -1.
    for a, b in itertools.combinations(sorted(pool, key=lambda r: r["id"]), 2):
        if not active.issubset({a["id"], b["id"]}):
            continue
        def relevance(r):
            return math.exp(-abs(r["ego_eta"] - r["cbv_eta"]) / 2.5) * math.exp(-max(r["cbv_eta"], 0.) / 6.)
        region_gap = math.hypot(a["conflict_x"] - b["conflict_x"], a["conflict_y"] - b["conflict_y"])
        value = min(relevance(a), relevance(b)) * math.exp(-region_gap / 25.)
        if value > best_score:
            best_score, best = value, (a["id"], b["id"])
    return [i for i in best if i not in active] if best is not None else []
