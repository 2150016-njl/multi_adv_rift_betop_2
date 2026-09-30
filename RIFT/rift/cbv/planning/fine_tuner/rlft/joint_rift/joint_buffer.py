"""State-level rollout storage for Joint-RIFT.

Unlike ``CBVRolloutBuffer``, this buffer does not wait for individual CBV
episodes to end.  Each record already contains a complete joint candidate
state and its state-local joint advantage matrix.
"""

from collections import deque
from typing import Any, Dict, List, Sequence, Union

import numpy as np

from rift.gym_carla.buffer.base_buffer import BaseBuffer


class JointRolloutBuffer(BaseBuffer):
    """Fixed-capacity buffer of fully formed two-CBV joint policy states."""

    name = "JointRolloutBuffer"
    state_key = "joint_rollout_states"

    def __init__(self, num_scenario, mode, cbv_config, logger=None):
        super().__init__(num_scenario, mode, logger)
        if mode != "train_cbv":
            raise ValueError("JointRolloutBuffer is only valid for train_cbv mode")
        self.buffer_capacity = int(cbv_config.get("buffer_capacity", 4096))
        if self.buffer_capacity < 1:
            raise ValueError("buffer_capacity must be positive")
        self.reset_buffer()

    def reset_buffer(self):
        self.buffer_pos = 0
        self.buffer_full = False
        self.buffer_data = deque(maxlen=self.buffer_capacity)
        self.collection_stage = None
        self.collection_checkpoint = None
        self.temp_buffer = None  # Explicitly absent: joint states need no GAE flush.

    def store(self, data_dict: Dict[str, Any]) -> None:
        """Append all non-empty joint states from one vector-environment tick."""
        states = data_dict.get(self.state_key)
        if states is None:
            return
        if isinstance(states, dict):
            states = [states]
        if not isinstance(states, Sequence):
            raise TypeError(f"{self.state_key} must be a state dict or a sequence of state dicts")

        for state in states:
            if state is None:
                continue
            if self.buffer_full:
                break
            self._validate_state(state)
            self._register_collection_provenance(state)
            self.buffer_data.append(state)
            self.buffer_pos += 1
            self.buffer_full = self.buffer_pos >= self.buffer_capacity

    def assert_on_policy_collection(self, stage: int, checkpoint: str) -> None:
        """Reject stale states, in particular a Stage-1 buffer in Stage 2."""
        if not self.buffer_full:
            raise RuntimeError("Joint-RIFT buffer must be full before training")
        if self.collection_stage != int(stage):
            raise RuntimeError(
                f"Joint-RIFT Stage {stage} requires a freshly collected Stage {stage} buffer; "
                f"got Stage {self.collection_stage}"
            )
        if self.collection_checkpoint != str(checkpoint):
            raise RuntimeError(
                "Joint-RIFT buffer was collected under a different checkpoint; recollect on-policy states"
            )

    def _register_collection_provenance(self, state: Dict[str, Any]) -> None:
        stage = int(state["collection_stage"])
        checkpoint = str(state["collection_checkpoint"])
        if self.collection_stage is None:
            self.collection_stage = stage
            self.collection_checkpoint = checkpoint
            return
        if stage != self.collection_stage or checkpoint != self.collection_checkpoint:
            raise RuntimeError("Joint-RIFT buffer cannot mix policy stages or checkpoint versions")

    def sample(self, idx: Union[int, Sequence[int]]):
        """Fetch one state or a list of states by deterministic index."""
        if isinstance(idx, (list, tuple, np.ndarray)):
            indices = list(idx)
            if not all(0 <= index < self.buffer_pos for index in indices):
                raise IndexError("joint rollout sample index is out of range")
            return [self.buffer_data[index] for index in indices]
        if not 0 <= idx < self.buffer_pos:
            raise IndexError("joint rollout sample index is out of range")
        return self.buffer_data[idx]

    def get_key_data(self, key: str) -> List[Any]:
        """Return one field from all stored states for diagnostics only."""
        if not self.buffer_full:
            raise RuntimeError("joint rollout buffer must be full before reading all data")
        return [state[key] for state in self.buffer_data]

    @staticmethod
    def _validate_state(state: Dict[str, Any]) -> None:
        required = {
            "feature_1",
            "feature_2",
            "old_joint_logits",
            "joint_advantage",
            "valid_mask_1",
            "valid_mask_2",
            "pair_ids",
            "frame_1",
            "frame_2",
            "collection_stage",
            "collection_checkpoint",
            "diagnostics",
        }
        if not isinstance(state, dict):
            raise TypeError("each joint rollout state must be a dictionary")
        missing = required - set(state)
        if missing:
            raise KeyError(f"joint rollout state is missing fields: {sorted(missing)}")

        logits = np.asarray(state["old_joint_logits"])
        advantage = np.asarray(state["joint_advantage"])
        valid_1 = np.asarray(state["valid_mask_1"], dtype=np.bool_)
        valid_2 = np.asarray(state["valid_mask_2"], dtype=np.bool_)
        if logits.ndim != 2 or advantage.shape != logits.shape:
            raise ValueError("old_joint_logits and joint_advantage must share shape [G1, G2]")
        if valid_1.shape != (logits.shape[0],) or valid_2.shape != (logits.shape[1],):
            raise ValueError("candidate valid masks must match old_joint_logits dimensions")
        if not valid_1.any() or not valid_2.any():
            raise ValueError("a joint rollout state requires at least one valid candidate per CBV")
        valid_pair_mask = valid_1[:, None] & valid_2[None, :]
        if not np.isfinite(logits[valid_pair_mask]).all():
            raise ValueError("valid old_joint_logits must be finite")
        if not np.isfinite(advantage[valid_pair_mask]).all():
            raise ValueError("valid joint_advantage values must be finite")
        pair_ids = tuple(state["pair_ids"])
        if len(pair_ids) != 2 or pair_ids[0] == pair_ids[1]:
            raise ValueError("pair_ids must contain two distinct CBV IDs")
        if int(state["collection_stage"]) not in (1, 2):
            raise ValueError("collection_stage must be 1 or 2")
        if not str(state["collection_checkpoint"]):
            raise ValueError("collection_checkpoint must identify the policy checkpoint")
        diagnostic_keys = {
            "p12",
            "coalition_gain",
            "pair_collision",
            "unary_return_1",
            "unary_return_2",
            "independent_same_as_joint_rate",
            "oracle_gap_if_debug",
        }
        diagnostics = state["diagnostics"]
        if not isinstance(diagnostics, dict) or diagnostic_keys - set(diagnostics):
            raise KeyError("joint rollout diagnostics are incomplete for required training logs")
        for frame_key in ("frame_1", "frame_2"):
            frame = state[frame_key]
            if not isinstance(frame, dict):
                raise TypeError(f"{frame_key} must be a frame-context dictionary")
            if set(("source_origin", "source_heading", "ego_origin", "ego_heading")) - set(frame):
                raise KeyError(f"{frame_key} is missing required geometry fields")
            if np.asarray(frame["source_origin"]).shape != (2,) or np.asarray(frame["ego_origin"]).shape != (2,):
                raise ValueError(f"{frame_key} origins must have shape [2]")
