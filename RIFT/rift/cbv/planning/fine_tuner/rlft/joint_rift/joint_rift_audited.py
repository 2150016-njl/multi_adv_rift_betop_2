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
    oracle_gate_decision,
    select_oracle_pair,
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
        self.oracle_eval_mode = str(config.get("oracle_eval_mode", "diagnose"))
        if self.oracle_eval_mode not in {"disabled", "diagnose", "oracle_gated"}:
            raise ValueError(
                "oracle_eval_mode must be one of: disabled, diagnose, oracle_gated"
            )
        self.joint_eval_action_mode = str(
            config.get("joint_eval_action_mode", "policy")
        )
        if self.joint_eval_action_mode not in {
            "policy",
            "independent",
            "oracle_gated",
        }:
            raise ValueError(
                "joint_eval_action_mode must be one of: policy, independent, oracle_gated"
            )
        if (
            self.joint_eval_action_mode != "policy"
            and self.oracle_eval_mode == "disabled"
        ):
            raise ValueError(
                "independent/oracle_gated evaluation requires oracle diagnostics"
            )
        self.oracle_gate = {
            "min_joint_gain": float(config.get("oracle_min_joint_gain", 10.0)),
            "min_nonfactorizable_ratio": float(
                config.get("oracle_min_nonfactorizable_ratio", 0.01)
            ),
            "require_both_necessary": bool(
                config.get("oracle_require_both_necessary", True)
            ),
            "max_pair_collision_rate": float(
                config.get("oracle_max_pair_collision_rate", 0.05)
            ),
        }
        reweight_config = config.get("high_value_reweight", {})
        self.high_value_reweight = {
            "enabled": bool(reweight_config.get("enabled", False)),
            "min_joint_gain": float(reweight_config.get("min_joint_gain", 10.0)),
            "min_nonfactorizable_ratio": float(
                reweight_config.get("min_nonfactorizable_ratio", 0.01)
            ),
            "max_pair_collision_rate": float(
                reweight_config.get("max_pair_collision_rate", 0.05)
            ),
            "weight": float(reweight_config.get("weight", 1.0)),
        }
        if self.high_value_reweight["weight"] < 1.0:
            raise ValueError("high_value_reweight.weight must be at least one")
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
        explicit_joint = self.config.get("joint_ckpt_path")
        if resume and explicit_joint:
            checkpoint = self._resolve_path(Path(self.config["ROOT_DIR"]), explicit_joint)
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"Configured joint_ckpt_path does not exist: {checkpoint}"
                )
            self.checkpoint = checkpoint.as_posix()
            self.continue_episode = 0
            self.current_epoch = 0
            self.pluto_model.load_state_dict(
                self.load_infer_checkpoint(self.checkpoint, device_name=self.device)
            )
            self._load_joint_head_checkpoint(self.checkpoint)
            self.logger.log(
                f">> Joint-RIFT explicit evaluation checkpoint: {checkpoint.name}",
                "green",
            )
            return

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

            policy_indices = self.select_joint_pair(score_output)
            selected_indices = policy_indices
            eval_diagnostics = None
            independent_indices = None
            oracle_indices = None
            executed_q = None
            oracle_gate_pass = False
            oracle_gate_failures = ()

            # The oracle is strictly evaluation-only.  It is an upper bound of
            # the evaluator/candidate set, never a training target or a result
            # that can be compared as a deployable policy.
            if (
                self.mode != "train"
                and self.eval_joint_diagnostics
                and self.oracle_eval_mode != "disabled"
                and nominal is not None
            ):
                joint_evaluation = self.evaluate_joint_candidate_batch(
                    env_id, candidate_batch
                )
                independent_indices = select_joint_pair(
                    score_output.base_logits[0], score_output.valid_pair_mask[0]
                )
                eval_diagnostics = self._compute_feasibility_diagnostics(
                    candidate_batch,
                    joint_evaluation,
                    policy_indices,
                    independent_indices,
                )
                oracle_indices = select_oracle_pair(
                    joint_evaluation["joint_q"], joint_evaluation["valid_mask"]
                )
                oracle_gate_pass, oracle_gate_failures = oracle_gate_decision(
                    eval_diagnostics, **self.oracle_gate
                )
                if self.joint_eval_action_mode == "independent":
                    selected_indices = independent_indices
                elif (
                    self.joint_eval_action_mode == "oracle_gated"
                    and oracle_gate_pass
                ):
                    selected_indices = oracle_indices
                executed_q = float(joint_evaluation["joint_q"][selected_indices])

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
            elif eval_diagnostics is not None:
                self._append_eval_diagnostics(
                    info,
                    candidate_batch,
                    eval_diagnostics,
                    policy_indices=policy_indices,
                    independent_indices=independent_indices,
                    oracle_indices=oracle_indices,
                    executed_indices=selected_indices,
                    executed_q=executed_q,
                    oracle_gate_pass=oracle_gate_pass,
                    oracle_gate_failures=oracle_gate_failures,
                )

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
        info: Dict,
        candidate_batch: JointCandidateBatch,
        diagnostics: Dict[str, float],
        policy_indices: Tuple[int, int],
        independent_indices: Tuple[int, int],
        oracle_indices: Tuple[int, int],
        executed_indices: Tuple[int, int],
        executed_q: float,
        oracle_gate_pass: bool,
        oracle_gate_failures: Tuple[str, ...],
    ) -> None:
        record = {
            "diagnostics_schema_version": 2,
            "policy_name": self.name,
            "env_id": int(info["env_id"]),
            "route_id": str(info.get("route_id", "")),
            "route_name": str(info.get("route_name", "")),
            "town_name": str(info.get("town_name", "")),
            "repetition_index": int(info.get("repetition_index", -1)),
            "route_data_id": int(info.get("route_data_id", -1)),
            "scenario_index": int(info.get("scenario_index", -1)),
            "time_step": int(info.get("time_step", -1)),
            "pair_ids": [int(v) for v in candidate_batch.pair_ids],
            "candidate_count_1": int(candidate_batch.candidates[0].logits.numel()),
            "candidate_count_2": int(candidate_batch.candidates[1].logits.numel()),
            "policy_indices": [int(v) for v in policy_indices],
            "independent_indices": [int(v) for v in independent_indices],
            "oracle_indices": [int(v) for v in oracle_indices],
            "executed_indices": [int(v) for v in executed_indices],
            "executed_q": float(executed_q),
            "executed_is_policy": bool(executed_indices == policy_indices),
            "executed_is_independent": bool(
                executed_indices == independent_indices
            ),
            "executed_is_oracle": bool(executed_indices == oracle_indices),
            "oracle_intervention": bool(
                self.joint_eval_action_mode == "oracle_gated" and oracle_gate_pass
            ),
            "oracle_eval_mode": self.oracle_eval_mode,
            "joint_eval_action_mode": self.joint_eval_action_mode,
            "policy_checkpoint": str(self.checkpoint or ""),
            "oracle_gate_pass": bool(oracle_gate_pass),
            "oracle_gate_failures": list(oracle_gate_failures),
            "oracle_gate": self.oracle_gate,
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
        # Record eligibility independently from whether this particular run
        # applies the reweight.  This makes the uniform control auditable and
        # distinguishes a lack of eligible states from an intentionally
        # disabled reweighting switch.
        state["diagnostics"]["high_value_eligible"] = float(
            self._is_high_value_state(diagnostics)
        )
        state["state_weight"] = self._high_value_state_weight(diagnostics)
        # Keep the legacy key for existing dashboards while giving it the same
        # audited value as the explicit oracle-policy gap.
        state["diagnostics"]["oracle_gap_if_debug"] = diagnostics[
            "oracle_policy_gap"
        ]
        return state

    def _high_value_state_weight(self, diagnostics: Dict[str, float]) -> float:
        """Upweight only safe relaxed-high-value collection states.

        This deliberately does not require ``both_necessary``: the current
        audit found too few such states for a viable Stage-1 buffer.  The
        selection remains an explicit training ablation and is never used to
        label an oracle execution result.
        """
        cfg = self.high_value_reweight
        if not cfg["enabled"]:
            return 1.0
        return cfg["weight"] if self._is_high_value_state(diagnostics) else 1.0

    def _is_high_value_state(self, diagnostics: Dict[str, float]) -> bool:
        """Return the config-defined safe relaxed-high-value eligibility."""
        cfg = self.high_value_reweight
        return bool(
            diagnostics["oracle_joint_gain"] >= cfg["min_joint_gain"]
            and diagnostics["nonfactorizable_reward_ratio"]
            >= cfg["min_nonfactorizable_ratio"]
            and diagnostics["oracle_pair_collision"] <= 0.0
            and diagnostics["pair_collision_rate"] <= cfg["max_pair_collision_rate"]
        )
