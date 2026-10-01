"""Audited Joint-RIFT policy.

This layer preserves the established Joint-RIFT candidate/scorer/trainer code
while enforcing the paper-critical invariants found during the implementation
audit:

1. Stage 1 bootstraps from a trained single-CBV RIFT checkpoint, never IL Pluto.
2. Stage 2 bootstraps from a Stage-1 Joint-RIFT checkpoint in a fresh rollout run.
3. Pair/fallback decisions are made per vector environment rather than globally.
4. Joint rewards use temporally weighted conflict events and expose feasibility
   diagnostics needed to verify non-factorizable coordination.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from rift.cbv.planning.fine_tuner.rlft.joint_rift.audited_joint_traj_evaluator import (
    AuditedJointTrajEvaluator,
)
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_diagnostics import (
    build_feasibility_diagnostics,
)
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_action import select_joint_pair
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_rift_pluto import (
    JointCandidateBatch,
    JointRIFTPluto,
)
from rift.cbv.planning.fine_tuner.rlft.rift_pluto.rift_pluto import RIFTPluto
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


class AuditedJointRIFTPluto(JointRIFTPluto):
    """Production Joint-RIFT policy with strict bootstrap and diagnostics."""

    name = "joint_rift_audited"

    def __init__(self, config, logger):
        super().__init__(config, logger)
        reward_config = config.get("reward", {})
        joint_config = config.get("joint", {})
        self.eval_joint_diagnostics = bool(config.get("eval_joint_diagnostics", True))
        self.eval_metrics_path = Path(logger.output_dir) / "joint_rift_eval_metrics.jsonl"
        self.joint_traj_evaluator = AuditedJointTrajEvaluator(
            dt=self._step_interval,
            num_frames=int(config.get("joint_horizon", 40)),
            gamma=float(config.get("gamma", 0.98)),
            lambda_pressure=float(reward_config.get("lambda_pressure", 1.0)),
            lambda_coalition=float(reward_config.get("lambda_coalition", 1.0)),
            lambda_pair_collision=float(reward_config.get("lambda_pair_collision", 20.0)),
            pair_collision_chunk_size=int(
                reward_config.get("pair_collision_chunk_size", 32)
            ),
            arrival_time_tau=float(reward_config.get("arrival_time_tau", 1.0)),
            both_necessary_epsilon=float(
                reward_config.get("both_necessary_epsilon", 0.1)
            ),
        )

    def load_model(self, resume=True):
        """Load own checkpoint or a stage-appropriate adversarial bootstrap.

        Stage 1 requires a previously trained RIFT checkpoint.  Stage 2 uses a
        separate model directory and bootstraps from the latest Stage-1 joint
        checkpoint.  This keeps each stage's CARLA rollout/data-loader schedule
        fresh instead of accidentally resuming at the end of the previous one.
        """
        own_dir = self.model_path / self.load_agent_info
        own_checkpoints = list(own_dir.glob("*.ckpt"))
        if resume and own_checkpoints:
            super().load_model(resume=True)
            self.logger.log(
                ">> Joint-RIFT resume: using latest checkpoint from current stage",
                "green",
            )
            return

        stage = int(self.config.get("joint", {}).get("stage", 1))
        bootstrap = self._resolve_bootstrap_checkpoint(stage)
        self.config["ckpt_path"] = bootstrap.as_posix()
        super().load_model(resume=resume)
        self.logger.log(
            f">> Joint-RIFT Stage {stage} bootstrap: {bootstrap.as_posix()}",
            "green",
        )

    def _resolve_bootstrap_checkpoint(self, stage: int) -> Path:
        root = Path(self.config["ROOT_DIR"])
        if stage == 1:
            explicit = self.config.get("base_rift_ckpt_path")
            if explicit:
                checkpoint = self._resolve_path(root, explicit)
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Configured base_rift_ckpt_path does not exist: {checkpoint}"
                    )
                return checkpoint
            model_root = self._resolve_path(
                root,
                self.config.get(
                    "base_rift_model_path",
                    "rift/cbv/planning/model_ckpt/rift_pluto",
                ),
            )
            checkpoint = self._latest_checkpoint(model_root / self.load_agent_info)
            if checkpoint is None:
                raise RuntimeError(
                    "Joint-RIFT Stage 1 requires a trained RIFT prior. "
                    f"No checkpoint was found in {model_root / self.load_agent_info}. "
                    "Train rift_pluto first with the same ego/cbv-recognition/seed."
                )
            return checkpoint

        if stage == 2:
            explicit = self.config.get("stage1_ckpt_path")
            if explicit:
                checkpoint = self._resolve_path(root, explicit)
                if not checkpoint.is_file():
                    raise FileNotFoundError(
                        f"Configured stage1_ckpt_path does not exist: {checkpoint}"
                    )
                return checkpoint
            stage1_root = self._resolve_path(
                root,
                self.config.get(
                    "stage1_model_path",
                    "rift/cbv/planning/model_ckpt/joint_rift_pluto",
                ),
            )
            checkpoint = self._latest_checkpoint(stage1_root / self.load_agent_info)
            if checkpoint is None:
                raise RuntimeError(
                    "Joint-RIFT Stage 2 requires a Stage-1 Joint-RIFT checkpoint. "
                    f"No checkpoint was found in {stage1_root / self.load_agent_info}."
                )
            return checkpoint
        raise ValueError("Joint-RIFT stage must be 1 or 2")

    @staticmethod
    def _resolve_path(root: Path, value) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    @staticmethod
    def _latest_checkpoint(directory: Path) -> Optional[Path]:
        checkpoints = list(directory.glob("*.ckpt"))
        if not checkpoints:
            return None

        def sort_key(path: Path):
            match = re.search(r"carla_episode=(\d+)", path.stem)
            episode = int(match.group(1)) if match else -1
            return episode, path.stat().st_mtime

        return max(checkpoints, key=sort_key)

    def get_action(
        self,
        CBVs_obs_list,
        infos,
        deterministic=False,
        ego_nominal_trajectories=None,
    ):
        """Select Joint-RIFT actions per environment, with local fallback only."""
        collection_stage = None
        collection_checkpoint = None
        if self.mode == "train":
            collection_stage = self._training_stage()
            if not self.checkpoint:
                raise RuntimeError(
                    "Joint-RIFT must load a checkpoint before collecting on-policy states"
                )
            collection_checkpoint = str(self.checkpoint)

        CBVs_actions = [{} for _ in range(self.num_scenario)]
        joint_rollout_states = [None for _ in range(self.num_scenario)]

        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            env_id = info["env_id"]
            pair_ids = self.pair_manager.update(env_id, CBVs_obs.keys())
            if pair_ids is None or set(CBVs_obs) != set(pair_ids):
                self.last_joint_candidate_batches.pop(env_id, None)
                self.last_joint_score_outputs.pop(env_id, None)
                fallback = RIFTPluto.get_action(
                    self,
                    [CBVs_obs],
                    [info],
                    deterministic=deterministic,
                    ego_nominal_trajectories=ego_nominal_trajectories,
                )
                CBVs_actions[env_id] = fallback["CBVs_actions"][env_id]
                continue

            nominal = (
                ego_nominal_trajectories.get(env_id)
                if ego_nominal_trajectories is not None
                else None
            )
            if self.mode == "train" and nominal is None:
                raise RuntimeError(
                    f"Joint-RIFT training env {env_id} is missing the PDM nominal trajectory"
                )

            with torch.no_grad():
                candidate_batch = self.forward_pair_candidates(
                    env_id, CBVs_obs, pair_ids, nominal
                )
                score_output = self.score_candidate_batch(candidate_batch)
            self.last_joint_candidate_batches[env_id] = candidate_batch
            self.last_joint_score_outputs[env_id] = score_output

            selected_indices = self.select_joint_pair(score_output)
            for candidate_set, selected_idx in zip(
                candidate_batch.candidates, selected_indices
            ):
                cbv_state = CarlaDataProvider.get_history_state(
                    CarlaDataProvider.get_actor_by_id(candidate_set.cbv_id)
                )[-1]
                global_trajectory = candidate_set.global_trajectories[selected_idx]
                local_trajectory = self._global_to_local(
                    global_trajectory.cpu().numpy(), cbv_state
                )
                CBVs_actions[env_id][candidate_set.cbv_id] = self.get_control(
                    env_id=env_id,
                    CBV_id=candidate_set.cbv_id,
                    center_state=cbv_state,
                    local_trajectory=local_trajectory,
                )

            if self.mode == "train":
                joint_evaluation = self.evaluate_joint_candidate_batch(
                    env_id, candidate_batch
                )
                independent_indices = select_joint_pair(
                    score_output.base_logits[0], score_output.valid_pair_mask[0]
                )
                joint_rollout_states[env_id] = self._build_joint_rollout_state(
                    CBVs_obs,
                    candidate_batch,
                    score_output,
                    joint_evaluation,
                    collection_stage=collection_stage,
                    collection_checkpoint=collection_checkpoint,
                    selected_indices=selected_indices,
                    independent_indices=independent_indices,
                )
            elif self.eval_joint_diagnostics and nominal is not None:
                joint_evaluation = self.evaluate_joint_candidate_batch(
                    env_id, candidate_batch
                )
                independent_indices = select_joint_pair(
                    score_output.base_logits[0], score_output.valid_pair_mask[0]
                )
                diagnostics = self._compute_feasibility_diagnostics(
                    candidate_batch,
                    joint_evaluation,
                    selected_indices,
                    independent_indices,
                )
                self._append_eval_diagnostics(env_id, candidate_batch, diagnostics)

        self._clean_CBVs(infos, CBVs_obs_list)
        output = {"CBVs_actions": CBVs_actions}
        if self.mode == "train":
            output["joint_rollout_states"] = joint_rollout_states
        return output

    def _compute_feasibility_diagnostics(
        self,
        candidate_batch: JointCandidateBatch,
        joint_evaluation: Dict,
        selected_indices: Tuple[int, int],
        independent_indices: Tuple[int, int],
    ) -> Dict[str, float]:
        candidate_1, candidate_2 = candidate_batch.candidates
        return build_feasibility_diagnostics(
            joint_q=joint_evaluation["joint_q"],
            valid_pair_mask=joint_evaluation["valid_mask"],
            p1=joint_evaluation["p1"],
            p2=joint_evaluation["p2"],
            p12=joint_evaluation["p12"],
            coalition_gain=joint_evaluation["coalition_gain"],
            pair_collision=joint_evaluation["pair_collision"],
            logits_1=candidate_1.logits.detach().cpu().numpy(),
            logits_2=candidate_2.logits.detach().cpu().numpy(),
            unary_return_1=joint_evaluation["unary_return_1"],
            unary_return_2=joint_evaluation["unary_return_2"],
            selected_indices=selected_indices,
            independent_indices=independent_indices,
            both_necessary_epsilon=self.joint_traj_evaluator.both_necessary_epsilon,
        )

    def _append_eval_diagnostics(
        self,
        env_id: int,
        candidate_batch: JointCandidateBatch,
        diagnostics: Dict[str, float],
    ) -> None:
        record = {
            "env_id": int(env_id),
            "pair_ids": [int(v) for v in candidate_batch.pair_ids],
            **{key: float(value) for key, value in diagnostics.items()},
        }
        self.eval_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.eval_metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def _build_joint_rollout_state(
        self,
        CBVs_obs: Dict,
        candidate_batch: JointCandidateBatch,
        score_output,
        joint_evaluation: Dict,
        collection_stage: int,
        collection_checkpoint: str,
        selected_indices: Tuple[int, int],
        independent_indices: Tuple[int, int],
    ) -> Dict:
        state = JointRIFTPluto._build_joint_rollout_state(
            CBVs_obs,
            candidate_batch,
            score_output,
            joint_evaluation,
            collection_stage,
            collection_checkpoint,
            selected_indices,
            independent_indices,
        )
        diagnostics = self._compute_feasibility_diagnostics(
            candidate_batch,
            joint_evaluation,
            selected_indices,
            independent_indices,
        )
        state["diagnostics"].update(diagnostics)
        # Keep the legacy key for existing dashboards while giving it the same
        # audited value as the explicit oracle-policy gap.
        state["diagnostics"]["oracle_gap_if_debug"] = diagnostics[
            "oracle_policy_gap"
        ]
        return state
