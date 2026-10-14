"""Stage-1, state-balanced PPO trainer for the Joint-RIFT residual head."""

from typing import Dict, Tuple

import lightning as L
import torch
from torch import Tensor, nn

from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_score_head import JointResidualScorer
from rift.cbv.planning.pluto.optim.warmup_cos_lr import WarmupCosLR


from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_losses import (
    masked_joint_log_softmax as masked_joint_log_softmax, joint_dual_clip_loss, soft_target_loss,
)
__all__ = ["JointRIFTLightningTrainer", "masked_joint_log_softmax", "joint_dual_clip_loss"]


class JointRIFTLightningTrainer(L.LightningModule):
    """Joint-RIFT PPO with a frozen generator and an optional shared pi head."""

    def __init__(
        self,
        model: nn.Module,
        joint_head: JointResidualScorer,
        lr: float,
        cl_lr_decay: float,
        weight_decay: float,
        epochs: int,
        warmup_epochs: int,
        frame_rate: int,
        trainable_layers=None,
        joint_lr: float = None,
        pi_lr: float = None,
        train_pi_head: bool = False,
        clip_ratio: float = 0.2,
        dual_clip: float = 3.0,
        pair_weighting: str = "uniform",
        auxiliary_weight: float = 0.0,
        teacher_temperature: float = 0.25,
        **_unused,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("model", "joint_head"))
        self.model = model
        self.joint_head = joint_head
        self.lr = float(lr)
        self.joint_lr = float(joint_lr if joint_lr is not None else lr)
        self.pi_lr = float(pi_lr if pi_lr is not None else self.joint_lr)
        self.train_pi_head = bool(train_pi_head)
        self.cl_lr_decay = float(cl_lr_decay)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.frame_rate = int(frame_rate)
        self.clip_ratio = float(clip_ratio)
        self.dual_clip = float(dual_clip)
        self.pair_weighting = pair_weighting
        self.auxiliary_weight = float(auxiliary_weight)
        self.teacher_temperature = float(teacher_temperature)
        if self.auxiliary_weight < 0 or self.teacher_temperature <= 0:
            raise ValueError("invalid auxiliary loss settings")
        if trainable_layers:
            raise ValueError(
                "Joint-RIFT controls trainability internally; trainable_layers must remain empty"
            )

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        pi_head = self._get_pi_head()
        if self.train_pi_head:
            for parameter in pi_head.parameters():
                parameter.requires_grad = True
        for parameter in self.joint_head.parameters():
            parameter.requires_grad = True

    def _get_pi_head(self) -> nn.Module:
        """Return the one shared Pluto scorer and reject broader unfreezing."""
        try:
            pi_head = self.model.planning_decoder.pi_head
        except AttributeError as error:
            raise AttributeError(
                "Joint-RIFT Stage 2 requires model.planning_decoder.pi_head"
            ) from error
        if not isinstance(pi_head, nn.Module):
            raise TypeError("planning_decoder.pi_head must be a torch module")
        return pi_head

    def forward(self, features):
        return self.model(features)

    def training_step(self, batch: Dict, batch_idx: int) -> Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: Dict, batch_idx: int) -> Tensor:
        return self._step(batch, "val")

    def _step(self, batch: Dict, prefix: str) -> Tensor:
        new_joint_logits, score_output = self._forward_joint_logits(batch)
        loss, metrics = joint_dual_clip_loss(
            new_joint_logits,
            batch["old_joint_logits"],
            batch["joint_advantage"],
            batch["joint_valid_mask"],
            state_weight=batch.get("state_weight"),
            clip_ratio=self.clip_ratio,
            dual_clip=self.dual_clip,
            pair_weighting=self.pair_weighting,
        )
        if self.auxiliary_weight:
            if "cooperation_score" not in batch:
                raise ValueError("auxiliary loss requires cooperation target cache")
            active = batch["cooperation_feasible"].flatten(1).any(-1)
            auxiliary = new_joint_logits[batch["joint_valid_mask"]].sum() * 0.
            if active.any():
                auxiliary = soft_target_loss(new_joint_logits[active], batch["cooperation_score"][active],
                                             batch["joint_valid_mask"][active], self.teacher_temperature,
                                             batch["state_weight"][active])
            loss = loss + self.auxiliary_weight * auxiliary
            self.log(f"joint/{prefix}_auxiliary_loss", auxiliary, on_step=False, on_epoch=True)
        if "cooperation_score" in batch:
            target = batch["cooperation_score"].masked_fill(~batch["joint_valid_mask"], -torch.inf).flatten(1)
            chosen = new_joint_logits.flatten(1).argmax(-1)
            regret = target.max(-1).values - target.gather(1, chosen[:, None]).squeeze(1)
            feasible = batch["cooperation_feasible"].flatten(1).gather(1, chosen[:, None]).float().mean()
            self.log(f"joint/{prefix}_target_regret", regret.mean(), on_step=False, on_epoch=True)
            self.log(f"joint/{prefix}_selected_feasible", feasible, on_step=False, on_epoch=True)
        self.log(f"joint/{prefix}_loss", loss, on_step=False, on_epoch=True, prog_bar=prefix == "train")
        if prefix == "train":
            self.log("joint/loss", loss, on_step=False, on_epoch=True)
        for name, value in metrics.items():
            self.log(f"joint/{prefix}_{name}", value, on_step=False, on_epoch=True)
            if prefix == "train":
                self.log(f"joint/{name}", value, on_step=False, on_epoch=True)
        high_value_fraction = (batch["state_weight"] > 1.0).float().mean()
        self.log(
            f"joint/{prefix}_high_value_state_fraction",
            high_value_fraction,
            on_step=False,
            on_epoch=True,
        )
        if prefix == "train":
            self.log(
                "joint/high_value_state_fraction",
                high_value_fraction,
                on_step=False,
                on_epoch=True,
            )

        valid_mask = batch["joint_valid_mask"]
        residual_std = _masked_pair_std(score_output.delta, valid_mask).mean().detach()
        base_std = score_output.base_std.mean().detach()
        residual_ratio = score_output.residual_scale.mean().detach()
        for name, value in {
            "residual_std": residual_std,
            "base_std": base_std,
            "residual_base_ratio": residual_ratio,
        }.items():
            self.log(f"joint/{prefix}_{name}", value, on_step=False, on_epoch=True)
            if prefix == "train":
                self.log(f"joint/{name}", value, on_step=False, on_epoch=True)
        if prefix == "train":
            self._log_collection_diagnostics(batch)
        return loss

    def _log_collection_diagnostics(self, batch: Dict) -> None:
        """Log reward/selection evidence collected with each on-policy state."""
        name_map = {
            "unary_return_1": "reward/unary_1",
            "unary_return_2": "reward/unary_2",
            "p12": "reward/P12",
            "coalition_gain": "reward/coalition_gain",
            "pair_collision": "reward/pair_collision_rate",
            "independent_same_as_joint_rate": "selection/independent_same_as_joint_rate",
            "oracle_gap_if_debug": "selection/oracle_gap_if_debug",
        }
        diagnostics = batch.get("diagnostics", [])
        for diagnostic_key, log_name in name_map.items():
            values = [
                torch.as_tensor(diagnostic[diagnostic_key], dtype=torch.float32, device=self.device).mean()
                for diagnostic in diagnostics
                if diagnostic_key in diagnostic
            ]
            if values:
                self.log(log_name, torch.stack(values).mean(), on_step=False, on_epoch=True)

    def _forward_joint_logits(self, batch: Dict) -> Tuple[Tensor, object]:
        # Keep the encoder and all trajectory heads in eval mode.  Stage 2
        # enables autograd only for the shared pi_head, so q/trajectory identity
        # remains fixed while the factorised base logits can adapt on-policy.
        self.model.eval()
        needs_pi_gradient = self.train_pi_head and self.training
        with torch.set_grad_enabled(needs_pi_gradient):
            output_1 = self.model(batch["feature_1"].data)
            output_2 = self.model(batch["feature_2"].data)

        group_1 = _flatten_valid_candidates(output_1, batch["feature_1"].data, batch["valid_mask_1"])
        group_2 = _flatten_valid_candidates(output_2, batch["feature_2"].data, batch["valid_mask_2"])
        common_1 = _local_raw_to_common(group_1["raw"], batch["frame_1"])
        common_2 = _local_raw_to_common(group_2["raw"], batch["frame_2"])
        score_output = self.joint_head(
            group_1["feature"], common_1, group_1["raw"], group_1["logits"],
            group_2["feature"], common_2, group_2["raw"], group_2["logits"],
            valid_mask_1=batch["valid_mask_1"],
            valid_mask_2=batch["valid_mask_2"],
        )
        return score_output.joint_logits, score_output

    def configure_optimizers(self):
        if self.joint_lr <= 0 or self.pi_lr <= 0:
            raise ValueError("joint_lr and pi_lr must be positive")
        optimizer_groups = [{
            "params": self.joint_head.parameters(),
            "lr": self.joint_lr,
            "lr_scale": 1.0,
        }]
        if self.train_pi_head:
            optimizer_groups.append({
                "params": self._get_pi_head().parameters(),
                "lr": self.pi_lr,
                "lr_scale": self.pi_lr / self.joint_lr,
            })
        optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=self.weight_decay)
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.joint_lr,
            min_lr=self.joint_lr * self.cl_lr_decay,
            epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
        )
        return [optimizer], [scheduler]

    def on_save_checkpoint(self, checkpoint):
        head = self.joint_head
        checkpoint["scorer_config"] = dict(candidate_dim=head.candidate_projection.in_features,
            representation_dim=head.candidate_projection.out_features, pair_dim=head.pair_projection.out_features,
            geometry_hidden_dim=head.pair_geometry[0].out_features, lambda_delta=head.lambda_delta,
            trajectory_frames=head.trajectory_frames, trajectory_stride=int(head.sample_indices[1] - head.sample_indices[0]) if len(head.sample_indices) > 1 else head.trajectory_frames,
            base_temperature=head.base_temperature, residual_scale_floor=head.residual_scale_floor,
            geometry_horizon=head.geometry_horizon, residual_scale_mode=head.residual_scale_mode,
            fixed_residual_scale=head.fixed_residual_scale)
        checkpoint["cooperation_config"] = getattr(self, "cooperation_config", {})
        checkpoint["reward_config"] = getattr(self, "reward_config", {})


def _flatten_valid_candidates(output: Dict, data: Dict, candidate_mask: Tensor) -> Dict[str, Tensor]:
    """Flatten valid ``R × M`` Pluto outputs into the padded ``[B, G, ...]`` form."""
    reference_valid = data["reference_line"]["valid_mask"].any(dim=-1)
    raw = output["trajectory"]
    local = output["candidate_trajectories"]
    feature = output["candidate_feature"]
    logits = output["probability"]
    batch_size, references, modes = logits.shape
    if candidate_mask.shape[0] != batch_size:
        raise ValueError("candidate valid mask batch size does not match Pluto output")
    flattened_valid = reference_valid.unsqueeze(-1).expand(-1, -1, modes).reshape(batch_size, -1)
    max_group = candidate_mask.shape[1]
    padded = {
        "raw": raw.new_zeros((batch_size, max_group, *raw.shape[3:])),
        "local": local.new_zeros((batch_size, max_group, *local.shape[3:])),
        "feature": feature.new_zeros((batch_size, max_group, feature.shape[-1])),
        "logits": logits.new_zeros((batch_size, max_group)),
    }
    for batch_index in range(batch_size):
        expected_count = int(candidate_mask[batch_index].sum().item())
        valid_count = int(flattened_valid[batch_index].sum().item())
        if valid_count != expected_count:
            raise ValueError(
                "stored candidate mask does not match current valid reference-line candidates; "
                "recollect the on-policy Joint-RIFT buffer"
            )
        for name, value in (
            ("raw", raw),
            ("local", local),
            ("feature", feature),
            ("logits", logits),
        ):
            flattened = value[batch_index].reshape(references * modes, *value.shape[3:])
            padded[name][batch_index, :expected_count] = flattened[flattened_valid[batch_index]]
    return padded


def _local_raw_to_common(raw_trajectory: Tensor, frame: Dict[str, Tensor]) -> Tensor:
    """Vectorised source-local Pluto raw trajectory to Ego common frame."""
    source_origin = frame["source_origin"].to(device=raw_trajectory.device, dtype=raw_trajectory.dtype)
    source_heading = frame["source_heading"].to(device=raw_trajectory.device, dtype=raw_trajectory.dtype)
    ego_origin = frame["ego_origin"].to(device=raw_trajectory.device, dtype=raw_trajectory.dtype)
    ego_heading = frame["ego_heading"].to(device=raw_trajectory.device, dtype=raw_trajectory.dtype)
    local_xy = raw_trajectory[..., :2]
    cos_source, sin_source = source_heading[:, None, None].cos(), source_heading[:, None, None].sin()
    global_x = local_xy[..., 0] * cos_source - local_xy[..., 1] * sin_source + source_origin[:, None, None, 0]
    global_y = local_xy[..., 0] * sin_source + local_xy[..., 1] * cos_source + source_origin[:, None, None, 1]
    delta_x = global_x - ego_origin[:, None, None, 0]
    delta_y = global_y - ego_origin[:, None, None, 1]
    cos_ego, sin_ego = ego_heading[:, None, None].cos(), ego_heading[:, None, None].sin()
    common_x = delta_x * cos_ego + delta_y * sin_ego
    common_y = -delta_x * sin_ego + delta_y * cos_ego
    local_heading = torch.atan2(raw_trajectory[..., 3], raw_trajectory[..., 2])
    common_heading = torch.atan2(
        (local_heading + source_heading[:, None, None] - ego_heading[:, None, None]).sin(),
        (local_heading + source_heading[:, None, None] - ego_heading[:, None, None]).cos(),
    )
    return torch.stack((common_x, common_y, common_heading), dim=-1)


def _masked_pair_std(values: Tensor, valid_mask: Tensor) -> Tensor:
    count = valid_mask.sum(dim=(1, 2)).clamp_min(1)
    total = torch.where(valid_mask, values, torch.zeros_like(values)).sum(dim=(1, 2))
    mean = total / count
    safe_values = torch.where(valid_mask, values, mean[:, None, None])
    return torch.sqrt(((safe_values - mean[:, None, None]).square().sum(dim=(1, 2))) / count)
