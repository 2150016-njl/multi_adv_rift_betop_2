"""Deterministic lifecycle management for a two-CBV Joint-RIFT segment."""

from typing import Dict, Iterable, Optional, Tuple


PairIds = Tuple[int, int]


class PairManager:
    """Lock one sorted pair of active CBVs for each environment.

    A pair starts when at least two active CBVs are observed.  It remains
    unchanged while both members remain active.  If either member disappears
    (including done, destruction, or reaching its goal), that segment ends;
    the next call with two valid CBVs starts a new segment.  Sorting is only a
    deterministic storage convention, not a behavioural role assignment.
    """

    def __init__(self) -> None:
        self.pair_ids: Dict[int, PairIds] = {}

    def update(self, env_id: int, active_cbv_ids: Iterable[int]) -> Optional[PairIds]:
        """Synchronize one environment and return its current valid pair."""
        active_ids = set(active_cbv_ids)
        current_pair = self.pair_ids.get(env_id)

        if current_pair is not None and set(current_pair).issubset(active_ids):
            return current_pair

        # A missing member ends the old segment before any new pair is chosen.
        self.pair_ids.pop(env_id, None)
        if len(active_ids) < 2:
            return None

        pair = tuple(sorted(active_ids)[:2])
        self.pair_ids[env_id] = pair
        return pair

    def get_pair(self, env_id: int) -> Optional[PairIds]:
        """Return the locked pair for ``env_id``, if a segment is active."""
        return self.pair_ids.get(env_id)

    def clear(self, env_id: int) -> None:
        """End the current segment for one environment."""
        self.pair_ids.pop(env_id, None)

    def reset(self) -> None:
        """End all segments, for example when a CARLA episode is rebuilt."""
        self.pair_ids.clear()

