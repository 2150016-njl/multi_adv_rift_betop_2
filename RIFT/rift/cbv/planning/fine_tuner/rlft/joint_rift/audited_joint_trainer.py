"""Joint-RIFT trainer with feasibility and reward-scale diagnostics."""

import torch

from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_trainer import (
    JointRIFTLightningTrainer,
)


class AuditedJointRIFTLightningTrainer(JointRIFTLightningTrainer):
    """Adds paper-facing diagnostics without changing the PPO objective."""

    def _log_collection_diagnostics(self, batch):
        super()._log_collection_diagnostics(batch)
        name_map = {
            "oracle_joint_gain": "feasibility/oracle_joint_gain",
            "oracle_policy_gap": "selection/oracle_policy_gap",
            "independent_same_as_oracle_rate": "selection/independent_same_as_oracle_rate",
            "joint_same_as_oracle_rate": "selection/joint_same_as_oracle_rate",
            "oracle_rank_1": "feasibility/oracle_rank_1",
            "oracle_rank_2": "feasibility/oracle_rank_2",
            "oracle_nll_1": "feasibility/oracle_nll_1",
            "oracle_nll_2": "feasibility/oracle_nll_2",
            "oracle_marginal_1": "feasibility/oracle_marginal_1",
            "oracle_marginal_2": "feasibility/oracle_marginal_2",
            "oracle_both_necessary": "feasibility/oracle_both_necessary",
            "both_necessary_rate": "feasibility/both_necessary_rate",
            "nonfactorizable_reward_ratio": "feasibility/nonfactorizable_reward_ratio",
            "reward_scale_p12_std": "reward_scale/P12_std",
            "reward_scale_coalition_std": "reward_scale/coalition_std",
            "reward_scale_unary_pressure_std": "reward_scale/unary_pressure_std",
        }
        diagnostics = batch.get("diagnostics", [])
        for diagnostic_key, log_name in name_map.items():
            values = [
                torch.as_tensor(
                    diagnostic[diagnostic_key], dtype=torch.float32, device=self.device
                ).mean()
                for diagnostic in diagnostics
                if diagnostic_key in diagnostic
            ]
            if values:
                self.log(
                    log_name,
                    torch.stack(values).mean(),
                    on_step=False,
                    on_epoch=True,
                )
