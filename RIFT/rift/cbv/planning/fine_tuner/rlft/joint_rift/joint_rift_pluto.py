"""Joint-RIFT policy: pair collection plus conservative residual scoring.

Pair action selection stays factorized until Step 11.  Step 8 nevertheless
computes and retains the zero-initialized residual score, so the exact same
full candidate groups are ready for the later joint argmax and RL stages.
"""

import gc
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import wandb

from rift.cbv.planning.fine_tuner.rlft.rift_pluto.rift_pluto import RIFTPluto
from rift.cbv.planning.fine_tuner.training_builder import TrainingEngine, build_training_engine
from rift.cbv.planning.fine_tuner.rlft.joint_rift.geometry import (
    carla_global_to_right_handed_trajectory,
    global_to_common_frame_trajectory,
    local_to_global_trajectory,
    to_common_frame_trajectory,
)
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_score_head import (
    JointResidualScorer,
    JointScoreOutput,
)
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_action import select_joint_pair
from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_traj_evaluator import (
    JointTrajEvaluator,
)
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature
from rift.scenario.tools.carla_data_provider import CarlaDataProvider


@dataclass
class JointCandidateSet:
    """All valid candidates of one member of an active two-CBV pair."""

    cbv_id: int
    raw_trajectories: torch.Tensor       # [G, T, 6]
    local_trajectories: torch.Tensor     # [G, T, 3]
    global_trajectories: torch.Tensor    # [G, T, 3]
    common_trajectories: torch.Tensor    # [G, T, 3]
    logits: torch.Tensor                 # [G]
    candidate_features: torch.Tensor     # [G, D]
    reference_line_positions: torch.Tensor     # [G, P, 2], candidate-aligned
    reference_line_orientations: torch.Tensor  # [G, P], candidate-aligned
    reference_line_valid_mask: torch.Tensor    # [G, P], candidate-aligned
    source_origin: np.ndarray                  # [2], right-handed global rear axle
    source_heading: float
    ego_origin: np.ndarray                     # [2], right-handed global rear axle
    ego_heading: float


@dataclass
class JointCandidateBatch:
    """Two candidate groups and one PDM nominal plan in global/common frames."""

    pair_ids: Tuple[int, int]
    candidates: Tuple[JointCandidateSet, JointCandidateSet]
    ego_nominal_common: Optional[np.ndarray]
    ego_nominal_global: Optional[np.ndarray]  # RIFT right-handed global frame


class JointRIFTPluto(RIFTPluto):
    """Collect full candidate groups for a fixed pair without top-K pruning."""

    name = 'joint_rift_pluto'

    def __init__(self, config, logger):
        super().__init__(config, logger)
        policy_training = config.get('training', {})
        for policy_key, trainer_key in (
            ('lr_joint', 'lr'),
            ('lr_pi', 'pi_lr'),
            ('train_pi_head', 'train_pi_head'),
        ):
            if policy_key in policy_training:
                self.cfg.joint_training[trainer_key] = policy_training[policy_key]
        self.initial_joint_lr = float(self.cfg.joint_training.lr)
        self.initial_pi_lr = float(self.cfg.joint_training.pi_lr)
        joint_config = config.get('joint', {})
        self.joint_scorer = JointResidualScorer(
            candidate_dim=self.pluto_model.dim,
            representation_dim=int(joint_config.get('representation_dim', 128)),
            pair_dim=int(joint_config.get('latent_dim', 64)),
            lambda_delta=float(joint_config.get('lambda_delta', 1.0)),
            trajectory_frames=int(config.get('joint_horizon', 40)),
            trajectory_stride=int(joint_config.get('traj_sample_interval', 5)),
        ).to(self.device)
        reward_config = config.get('reward', {})
        self.joint_traj_evaluator = JointTrajEvaluator(
            dt=self._step_interval,
            num_frames=int(config.get('joint_horizon', 40)),
            gamma=float(config.get('gamma', 0.98)),
            lambda_pressure=float(reward_config.get('lambda_pressure', 1.0)),
            lambda_coalition=float(reward_config.get('lambda_coalition', 1.0)),
            lambda_pair_collision=float(reward_config.get('lambda_pair_collision', 1.0)),
        )
        self.last_joint_candidate_batches: Dict[int, JointCandidateBatch] = {}
        self.last_joint_score_outputs: Dict[int, JointScoreOutput] = {}

    def train(self, e_i):
        """Run validated Stage 1 or Stage 2 Joint-RIFT PPO training."""
        stage = self._training_stage()
        self.buffer.assert_on_policy_collection(stage, self.checkpoint)
        self.logger.log(f'>> Starting Joint-RIFT Stage {stage} fine-tuning...', color='yellow')
        dir_path = self.model_path / self.load_agent_info
        decay = self.cfg.cl_lr_decay ** self.current_epoch
        self.cfg.joint_training.lr = max(
            self.initial_joint_lr * decay, self.cfg.min_lr
        )
        self.cfg.joint_training.pi_lr = max(self.initial_pi_lr * decay, self.cfg.min_lr)
        self.cfg.lr = self.cfg.joint_training.lr

        training_engine: TrainingEngine = build_training_engine(
            cfg=self.cfg,
            dir_path=dir_path,
            carla_episode=e_i,
            torch_module_wrapper=self.train_model,
            buffer=self.buffer,
            joint_head=self.joint_scorer,
        )
        # A Pluto checkpoint has no ``joint_head`` entries on the first round;
        # strict=False deliberately retains the zero-initialised residual then.
        training_engine.model.load_state_dict(
            self.load_train_checkpoint(self.checkpoint, device_name=self.device), strict=False
        )
        training_engine.datamodule.preprocess_buffer()
        training_engine.trainer.fit(
            model=training_engine.model, datamodule=training_engine.datamodule
        )
        if wandb.run is not None:
            wandb.finish()

        self.update_training_ckpt()
        self.pluto_model.load_state_dict(
            self.load_infer_checkpoint(self.checkpoint, device_name=self.device)
        )
        self._load_joint_head_checkpoint(self.checkpoint)

        del training_engine
        gc.collect()
        torch.cuda.empty_cache()
        self.buffer.reset_buffer()
        self.logger.log(f'>> Finishing Joint-RIFT Stage {stage} fine-tuning...', color='yellow')

    def _training_stage(self) -> int:
        """Validate the only two supported training regimes from the design."""
        stage = int(self.config.get('joint', {}).get('stage', 1))
        train_pi_head = bool(self.cfg.joint_training.train_pi_head)
        if stage not in (1, 2):
            raise ValueError('Joint-RIFT stage must be 1 or 2')
        if stage == 1 and train_pi_head:
            raise ValueError('Stage 1 must keep planning_decoder.pi_head frozen')
        if stage == 2 and not train_pi_head:
            raise ValueError('Stage 2 requires training.train_pi_head: true')
        return stage

    def load_model(self, resume=True):
        """Load both Pluto and the residual scorer from a joint checkpoint."""
        super().load_model(resume=resume)
        self._load_joint_head_checkpoint(self.checkpoint)

    def load_infer_checkpoint(self, checkpoint: str, device_name: str):
        """Return only Pluto weights; joint-head weights have a separate owner."""
        state_dict = self.load_train_checkpoint(checkpoint, device_name=device_name)
        model_state = {
            key[len('model.'):]: value
            for key, value in state_dict.items()
            if key.startswith('model.')
        }
        if model_state:
            return model_state
        # Preserve compatibility with a raw pretrained Pluto state dictionary.
        return super().load_infer_checkpoint(checkpoint, device_name=device_name)

    def _load_joint_head_checkpoint(self, checkpoint: Optional[str]) -> None:
        """Restore residual weights when available; pretrained Pluto has none."""
        if not checkpoint:
            return
        state_dict = self.load_train_checkpoint(checkpoint, device_name=self.device)
        joint_state = {
            key[len('joint_head.'):]: value
            for key, value in state_dict.items()
            if key.startswith('joint_head.')
        }
        if joint_state:
            self.joint_scorer.load_state_dict(joint_state, strict=True)

    def get_action(
        self,
        CBVs_obs_list,
        infos,
        deterministic=False,
        ego_nominal_trajectories=None,
    ):
        collection_stage = None
        collection_checkpoint = None
        if self.mode == 'train':
            collection_stage = self._training_stage()
            if not self.checkpoint:
                raise RuntimeError('Joint-RIFT must load a checkpoint before collecting on-policy states')
            collection_checkpoint = str(self.checkpoint)
        pairs_by_env = {}
        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            env_id = info['env_id']
            pair_ids = self.pair_manager.update(env_id, CBVs_obs.keys())
            if pair_ids is None or set(CBVs_obs) != set(pair_ids):
                # get_action must return one coherent action batch.  Do not
                # partially use joint decisions for some environments and the
                # legacy path for others in the same CARLA tick.
                self.last_joint_candidate_batches.clear()
                self.last_joint_score_outputs.clear()
                return super().get_action(
                    CBVs_obs_list,
                    infos,
                    deterministic=deterministic,
                    ego_nominal_trajectories=ego_nominal_trajectories,
                )
            pairs_by_env[env_id] = pair_ids

        CBVs_actions = [{} for _ in range(self.num_scenario)]
        joint_rollout_states = [None for _ in range(self.num_scenario)]
        for info, CBVs_obs in zip(infos, CBVs_obs_list):
            env_id = info['env_id']
            pair_ids = pairs_by_env[env_id]

            nominal = (
                ego_nominal_trajectories.get(env_id)
                if ego_nominal_trajectories is not None
                else None
            )
            with torch.no_grad():
                candidate_batch = self.forward_pair_candidates(
                    env_id, CBVs_obs, pair_ids, nominal
                )
                score_output = self.score_candidate_batch(candidate_batch)
            self.last_joint_candidate_batches[env_id] = candidate_batch
            self.last_joint_score_outputs[env_id] = score_output

            selected_indices = self.select_joint_pair(score_output)
            for candidate_set, selected_idx in zip(candidate_batch.candidates, selected_indices):
                CBV_state = CarlaDataProvider.get_history_state(
                    CarlaDataProvider.get_actor_by_id(candidate_set.cbv_id)
                )[-1]
                global_trajectory = candidate_set.global_trajectories[selected_idx]
                local_trajectory = self._global_to_local(
                    global_trajectory.cpu().numpy(), CBV_state
                )
                CBVs_actions[env_id][candidate_set.cbv_id] = self.get_control(
                    env_id=env_id,
                    CBV_id=candidate_set.cbv_id,
                    center_state=CBV_state,
                    local_trajectory=local_trajectory,
                )

            if self.mode == 'train':
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

        self._clean_CBVs(infos, CBVs_obs_list)
        output = {'CBVs_actions': CBVs_actions}
        if self.mode == 'train':
            output['joint_rollout_states'] = joint_rollout_states
        return output

    def forward_pair_candidates(
        self,
        env_id: int,
        CBVs_obs: Dict,
        pair_ids: Tuple[int, int],
        ego_nominal_trajectory: Optional[np.ndarray] = None,
    ) -> JointCandidateBatch:
        """Forward both pair members and retain every valid ``R x 12`` mode."""
        if tuple(sorted(pair_ids)) != pair_ids:
            raise ValueError('pair_ids must use PairManager actor-ID ordering')
        if set(pair_ids) - set(CBVs_obs):
            raise ValueError('pair_ids must refer to active CBV observations')

        ordered_obs = [CBVs_obs[cbv_id] for cbv_id in pair_ids]
        pluto_feature_data = PlutoFeature.collate(
            [cbv_obs['raw_pluto_feature'] for cbv_obs in ordered_obs]
        ).to_device(self.device).data
        pluto_output = self.pluto_model(pluto_feature_data)

        ego = CarlaDataProvider.get_ego_vehicle_by_env_id(env_id)
        ego_state = CarlaDataProvider.get_history_state(ego)[-1]
        candidate_sets = tuple(
            self._extract_candidate_set(
                index, cbv_id, pluto_feature_data, pluto_output, ego_state
            )
            for index, cbv_id in enumerate(pair_ids)
        )
        ego_nominal_global = (
            carla_global_to_right_handed_trajectory(ego_nominal_trajectory)
            if ego_nominal_trajectory is not None
            else None
        )
        ego_nominal_common = (
            global_to_common_frame_trajectory(ego_nominal_global, ego_state)
            if ego_nominal_global is not None
            else None
        )
        return JointCandidateBatch(
            pair_ids,
            candidate_sets,
            ego_nominal_common,
            ego_nominal_global,
        )

    def _extract_candidate_set(
        self,
        index: int,
        cbv_id: int,
        pluto_feature_data,
        pluto_output,
        ego_state,
    ) -> JointCandidateSet:
        reference_valid_mask = pluto_feature_data['reference_line']['valid_mask'][index].any(-1)
        if not reference_valid_mask.any():
            raise RuntimeError(f'CBV {cbv_id} has no valid reference-line candidates')

        CBV = CarlaDataProvider.get_actor_by_id(cbv_id)
        cbv_state = CarlaDataProvider.get_history_state(CBV)[-1]
        raw_trajectories = pluto_output['trajectory'][index][reference_valid_mask]
        local_trajectories = pluto_output['candidate_trajectories'][index][reference_valid_mask]
        logits = pluto_output['probability'][index][reference_valid_mask]
        candidate_features = pluto_output['candidate_feature'][index][reference_valid_mask]
        reference_line_positions = pluto_feature_data['reference_line']['position'][index][reference_valid_mask]
        reference_line_orientations = pluto_feature_data['reference_line']['orientation'][index][reference_valid_mask]
        reference_line_valid_mask = pluto_feature_data['reference_line']['valid_mask'][index][reference_valid_mask]

        valid_r, num_modes = logits.shape
        raw_trajectories = raw_trajectories.reshape(valid_r * num_modes, *raw_trajectories.shape[2:])
        local_trajectories = local_trajectories.reshape(valid_r * num_modes, *local_trajectories.shape[2:])
        logits = logits.reshape(valid_r * num_modes)
        candidate_features = candidate_features.reshape(valid_r * num_modes, -1)
        reference_line_positions = reference_line_positions.repeat_interleave(num_modes, dim=0)
        reference_line_orientations = reference_line_orientations.repeat_interleave(num_modes, dim=0)
        reference_line_valid_mask = reference_line_valid_mask.repeat_interleave(num_modes, dim=0)

        global_trajectories = local_to_global_trajectory(local_trajectories, cbv_state)
        common_trajectories = to_common_frame_trajectory(
            local_trajectories, cbv_state, ego_state
        )
        return JointCandidateSet(
            cbv_id=cbv_id,
            raw_trajectories=raw_trajectories,
            local_trajectories=local_trajectories,
            global_trajectories=global_trajectories,
            common_trajectories=common_trajectories,
            logits=logits,
            candidate_features=candidate_features,
            reference_line_positions=reference_line_positions,
            reference_line_orientations=reference_line_orientations,
            reference_line_valid_mask=reference_line_valid_mask,
            source_origin=np.asarray(cbv_state.rear_axle.array[:2], dtype=np.float32),
            source_heading=float(cbv_state.rear_axle.heading),
            ego_origin=np.asarray(ego_state.rear_axle.array[:2], dtype=np.float32),
            ego_heading=float(ego_state.rear_axle.heading),
        )

    def get_last_joint_candidate_batch(self, env_id: int) -> Optional[JointCandidateBatch]:
        """Expose the latest full candidate groups for diagnostics and training."""
        return self.last_joint_candidate_batches.get(env_id)

    def get_last_joint_score_output(self, env_id: int) -> Optional[JointScoreOutput]:
        """Expose Step 8 scores without changing the currently executed action."""
        return self.last_joint_score_outputs.get(env_id)

    def score_candidate_batch(self, candidate_batch: JointCandidateBatch) -> JointScoreOutput:
        """Score one unpadded two-CBV candidate batch using the shared head.

        Candidate extraction removes invalid reference lines before this point,
        hence every candidate in each group is valid and no mask is required.
        The leading batch dimension keeps this API identical to future rollout
        and trainer tensors.
        """
        candidates_1, candidates_2 = candidate_batch.candidates
        return self.joint_scorer(
            candidates_1.candidate_features.unsqueeze(0),
            candidates_1.common_trajectories.unsqueeze(0),
            candidates_1.raw_trajectories.unsqueeze(0),
            candidates_1.logits.unsqueeze(0),
            candidates_2.candidate_features.unsqueeze(0),
            candidates_2.common_trajectories.unsqueeze(0),
            candidates_2.raw_trajectories.unsqueeze(0),
            candidates_2.logits.unsqueeze(0),
        )

    @staticmethod
    def select_joint_pair(score_output: JointScoreOutput) -> Tuple[int, int]:
        """Select one valid ``(i, j)`` from a single state's joint logit matrix."""
        if score_output.joint_logits.shape[0] != 1:
            raise ValueError("action selection expects a single joint state")
        joint_logits = score_output.joint_logits[0]
        valid_mask = score_output.valid_pair_mask[0]
        return select_joint_pair(joint_logits, valid_mask)

    def evaluate_joint_candidate_batch(
        self,
        env_id: int,
        candidate_batch: Optional[JointCandidateBatch] = None,
    ):
        """Compute Step 9--10 diagnostics for a cached or supplied pair batch.

        This is intentionally an explicit evaluator API: collection continues
        to use the independent action path until Step 11, while Step 12 can
        call this method to create joint advantages for its new rollout buffer.
        """
        candidate_batch = candidate_batch or self.get_last_joint_candidate_batch(env_id)
        if candidate_batch is None:
            raise ValueError(f'no joint candidate batch is available for env {env_id}')
        if candidate_batch.ego_nominal_global is None:
            raise ValueError('Step 10 requires the PDM nominal trajectory from Step 1')

        ego = CarlaDataProvider.get_ego_vehicle_by_env_id(env_id)
        ego_history_states = CarlaDataProvider.get_history_state(ego)
        candidate_1, candidate_2 = candidate_batch.candidates
        unary_1 = self._joint_unary_evaluator_kwargs(
            ego.id, candidate_1, candidate_2.cbv_id,
            candidate_batch.ego_nominal_global, ego_history_states,
        )
        unary_2 = self._joint_unary_evaluator_kwargs(
            ego.id, candidate_2, candidate_1.cbv_id,
            candidate_batch.ego_nominal_global, ego_history_states,
        )
        return self.joint_traj_evaluator.evaluate_joint(unary_1, unary_2)

    @staticmethod
    def _build_joint_rollout_state(
        CBVs_obs: Dict,
        candidate_batch: JointCandidateBatch,
        score_output: JointScoreOutput,
        joint_evaluation: Dict,
        collection_stage: int,
        collection_checkpoint: str,
        selected_indices: Tuple[int, int],
        independent_indices: Tuple[int, int],
    ) -> Dict:
        """Serialize one complete Step 12 state without per-CBV done semantics."""
        candidate_1, candidate_2 = candidate_batch.candidates
        valid_mask_1 = score_output.valid_pair_mask[0].any(dim=-1).detach().cpu().numpy()
        valid_mask_2 = score_output.valid_pair_mask[0].any(dim=-2).detach().cpu().numpy()
        diagnostics = {
            key: joint_evaluation[key]
            for key in (
                'p12', 'coalition_gain', 'pair_collision', 'p1', 'p2',
                'unary_return_1', 'unary_return_2',
            )
            if key in joint_evaluation
        }
        diagnostics['independent_same_as_joint_rate'] = float(
            selected_indices == independent_indices
        )
        joint_q = np.asarray(joint_evaluation['joint_q'])
        valid_pair_mask = np.asarray(joint_evaluation['valid_mask'], dtype=bool)
        diagnostics['oracle_gap_if_debug'] = float(
            joint_q[valid_pair_mask].max() - joint_q[selected_indices]
        )
        return {
            'feature_1': CBVs_obs[candidate_1.cbv_id]['raw_pluto_feature'],
            'feature_2': CBVs_obs[candidate_2.cbv_id]['raw_pluto_feature'],
            'old_joint_logits': score_output.joint_logits[0].detach().cpu().numpy(),
            'joint_advantage': joint_evaluation['joint_advantage'],
            'valid_mask_1': valid_mask_1,
            'valid_mask_2': valid_mask_2,
            'pair_ids': candidate_batch.pair_ids,
            'frame_1': JointRIFTPluto._frame_context(candidate_1),
            'frame_2': JointRIFTPluto._frame_context(candidate_2),
            'collection_stage': int(collection_stage),
            'collection_checkpoint': str(collection_checkpoint),
            'diagnostics': diagnostics,
        }

    @staticmethod
    def _frame_context(candidate_set: JointCandidateSet) -> Dict:
        """Serialize immutable local-to-common geometry for trainer re-forward."""
        return {
            'source_origin': candidate_set.source_origin.copy(),
            'source_heading': candidate_set.source_heading,
            'ego_origin': candidate_set.ego_origin.copy(),
            'ego_heading': candidate_set.ego_heading,
        }

    def _joint_unary_evaluator_kwargs(
        self,
        ego_id: int,
        candidate_set: JointCandidateSet,
        other_controlled_cbv_id: int,
        ego_nominal_global: np.ndarray,
        ego_history_states,
    ) -> Dict:
        """Prepare a unary rollout while excluding the other controlled CBV."""
        center_actor = CarlaDataProvider.get_actor_by_id(candidate_set.cbv_id)
        excluded_actor_ids = {ego_id, candidate_set.cbv_id, other_controlled_cbv_id}
        background_actors = [
            actor
            for actor in CarlaDataProvider.get_CBV_nearby_agents(ego_id, candidate_set.cbv_id)
            if actor.id not in excluded_actor_ids
        ]
        return {
            'center_history_states': CarlaDataProvider.get_history_state(center_actor),
            'raw_trajectories': candidate_set.raw_trajectories,
            'reference_line_positions': candidate_set.reference_line_positions,
            'reference_line_orientations': candidate_set.reference_line_orientations,
            'reference_line_valid_mask': candidate_set.reference_line_valid_mask,
            'background_actors': background_actors,
            'ego_nominal_trajectory': ego_nominal_global,
            'ego_history_states': ego_history_states,
        }

