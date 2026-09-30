"""Symmetric, conservative residual scorer for a pair of RIFT candidates.

The scorer intentionally operates on the full candidate groups produced by
``JointRIFTPluto``.  It never constructs a learned tensor with shape
``[B, G1, G2, T, D]``: the only pairwise trajectory operation reduces the
per-time-step distance to one scalar before it reaches an MLP.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class JointScoreOutput:
    """Scores and diagnostics for one batch of two candidate groups."""

    base_logits: Tensor          # [B, G1, G2], log p1 + log p2
    raw_delta: Tensor            # [B, G1, G2], zero at initialization
    delta: Tensor                # [B, G1, G2], calibrated residual
    joint_logits: Tensor         # [B, G1, G2], invalid pairs are -inf
    base_std: Tensor             # [B], detached calibration scale
    residual_scale: Tensor       # [B], std(delta) / std(base)
    valid_pair_mask: Tensor      # [B, G1, G2]


class JointResidualScorer(nn.Module):
    """Score candidate pairs with shared encoders and a zero-init residual.

    The candidate representation fuses Pluto's decoder latent ``q`` with
    eight common-frame trajectory samples ``[x, y, heading, speed]`` at
    frames ``0, 5, ..., 35``.  Velocity in Pluto's raw six-channel output is
    represented by its magnitude, which is invariant to the coordinate-frame
    rotation used for the positions and headings.
    """

    def __init__(
        self,
        candidate_dim: int,
        representation_dim: int = 128,
        pair_dim: int = 64,
        geometry_hidden_dim: int = 32,
        lambda_delta: float = 1.0,
        trajectory_frames: int = 40,
        trajectory_stride: int = 5,
    ) -> None:
        super().__init__()
        if trajectory_frames < trajectory_stride:
            raise ValueError("trajectory_frames must include at least one sample")
        if trajectory_frames % trajectory_stride:
            raise ValueError("trajectory_frames must be divisible by trajectory_stride")
        if lambda_delta < 0:
            raise ValueError("lambda_delta must be non-negative")

        sample_indices = torch.arange(0, trajectory_frames, trajectory_stride)
        self.register_buffer("sample_indices", sample_indices, persistent=False)
        self.trajectory_frames = trajectory_frames
        self.lambda_delta = float(lambda_delta)
        sampled_dim = len(sample_indices) * 4

        self.trajectory_encoder = nn.Sequential(
            nn.Linear(sampled_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.ReLU(inplace=True),
            nn.Linear(representation_dim, representation_dim),
        )
        self.candidate_projection = nn.Linear(candidate_dim, representation_dim)
        self.candidate_fusion = nn.Sequential(
            nn.Linear(2 * representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.ReLU(inplace=True),
        )

        # One projection is shared by both CBV groups, which makes the
        # compatibility matrix transpose under a pair-member swap.
        self.pair_projection = nn.Linear(representation_dim, pair_dim, bias=False)
        self.pair_geometry = nn.Sequential(
            nn.Linear(1, geometry_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(geometry_hidden_dim, 1),
        )
        self.residual_projection = nn.Linear(2, 1)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)

    def forward(
        self,
        candidate_features_1: Tensor,
        common_trajectories_1: Tensor,
        raw_trajectories_1: Tensor,
        logits_1: Tensor,
        candidate_features_2: Tensor,
        common_trajectories_2: Tensor,
        raw_trajectories_2: Tensor,
        logits_2: Tensor,
        valid_mask_1: Optional[Tensor] = None,
        valid_mask_2: Optional[Tensor] = None,
    ) -> JointScoreOutput:
        """Return calibrated pair logits for tensors shaped ``[B, G, ...]``."""
        self._validate_candidate_inputs(
            candidate_features_1, common_trajectories_1, raw_trajectories_1, logits_1
        )
        self._validate_candidate_inputs(
            candidate_features_2, common_trajectories_2, raw_trajectories_2, logits_2
        )
        if candidate_features_1.shape[0] != candidate_features_2.shape[0]:
            raise ValueError("both candidate groups must have the same batch size")

        valid_mask_1 = self._normalise_valid_mask(valid_mask_1, logits_1)
        valid_mask_2 = self._normalise_valid_mask(valid_mask_2, logits_2)
        logp_1 = self._masked_log_softmax(logits_1, valid_mask_1)
        logp_2 = self._masked_log_softmax(logits_2, valid_mask_2)
        valid_pair_mask = valid_mask_1.unsqueeze(-1) & valid_mask_2.unsqueeze(-2)
        base_logits = logp_1.unsqueeze(-1) + logp_2.unsqueeze(-2)

        representation_1 = self._encode_candidates(
            candidate_features_1, common_trajectories_1, raw_trajectories_1
        )
        representation_2 = self._encode_candidates(
            candidate_features_2, common_trajectories_2, raw_trajectories_2
        )
        projected_1 = self.pair_projection(representation_1)
        projected_2 = self.pair_projection(representation_2)
        compatibility = torch.einsum("bid,bjd->bij", projected_1, projected_2)
        compatibility = compatibility / math.sqrt(projected_1.shape[-1])

        # This temporary pairwise tensor contains only two coordinates and is
        # immediately reduced across time; no high-dimensional pair MLP is
        # applied to [B, G1, G2, T, D].
        minimum_distance = torch.linalg.vector_norm(
            common_trajectories_1[:, :, None, :, :2]
            - common_trajectories_2[:, None, :, :, :2],
            dim=-1,
        ).amin(dim=-1, keepdim=True)
        geometry_score = self.pair_geometry(minimum_distance).squeeze(-1)
        raw_delta = self.residual_projection(
            torch.stack((compatibility, geometry_score), dim=-1)
        ).squeeze(-1)

        base_std = self._masked_std(base_logits, valid_pair_mask).detach()
        delta = self.lambda_delta * base_std[:, None, None] * torch.tanh(raw_delta)
        joint_logits = torch.where(
            valid_pair_mask,
            base_logits + delta,
            torch.full_like(base_logits, -torch.inf),
        )
        residual_scale = self._masked_std(delta, valid_pair_mask) / base_std.clamp_min(1e-6)
        return JointScoreOutput(
            base_logits=base_logits,
            raw_delta=raw_delta,
            delta=delta,
            joint_logits=joint_logits,
            base_std=base_std,
            residual_scale=residual_scale,
            valid_pair_mask=valid_pair_mask,
        )

    def _encode_candidates(
        self,
        candidate_features: Tensor,
        common_trajectories: Tensor,
        raw_trajectories: Tensor,
    ) -> Tensor:
        sample_indices = self.sample_indices.to(common_trajectories.device)
        common_samples = common_trajectories.index_select(-2, sample_indices)
        speed = torch.linalg.vector_norm(
            raw_trajectories.index_select(-2, sample_indices)[..., 4:6], dim=-1
        )
        trajectory_samples = torch.cat((common_samples[..., :3], speed.unsqueeze(-1)), dim=-1)
        trajectory_embedding = self.trajectory_encoder(trajectory_samples.flatten(-2))
        feature_embedding = self.candidate_projection(candidate_features)
        return self.candidate_fusion(torch.cat((feature_embedding, trajectory_embedding), dim=-1))

    def _validate_candidate_inputs(
        self,
        candidate_features: Tensor,
        common_trajectories: Tensor,
        raw_trajectories: Tensor,
        logits: Tensor,
    ) -> None:
        if candidate_features.ndim != 3 or logits.ndim != 2:
            raise ValueError("candidate features and logits must have shapes [B, G, D] and [B, G]")
        if common_trajectories.ndim != 4 or raw_trajectories.ndim != 4:
            raise ValueError("trajectories must have shapes [B, G, T, C]")
        batch_size, group_size = logits.shape
        if candidate_features.shape[:2] != (batch_size, group_size):
            raise ValueError("candidate feature shape does not match logits")
        if common_trajectories.shape[:2] != (batch_size, group_size):
            raise ValueError("common trajectory shape does not match logits")
        if raw_trajectories.shape[:2] != (batch_size, group_size):
            raise ValueError("raw trajectory shape does not match logits")
        if common_trajectories.shape[-1] < 3 or raw_trajectories.shape[-1] < 6:
            raise ValueError("common trajectories need 3 channels and raw trajectories need 6")
        if common_trajectories.shape[-2] < self.trajectory_frames:
            raise ValueError("candidate trajectories are shorter than the required 40 frames")
        if raw_trajectories.shape[-2] < self.trajectory_frames:
            raise ValueError("raw trajectories are shorter than the required 40 frames")

    @staticmethod
    def _normalise_valid_mask(mask: Optional[Tensor], logits: Tensor) -> Tensor:
        if mask is None:
            mask = torch.ones_like(logits, dtype=torch.bool)
        elif mask.shape != logits.shape:
            raise ValueError("valid mask must have shape [B, G]")
        else:
            mask = mask.to(dtype=torch.bool, device=logits.device)
        if not mask.any(dim=-1).all():
            raise ValueError("every batch element needs at least one valid candidate")
        return mask

    @staticmethod
    def _masked_log_softmax(logits: Tensor, valid_mask: Tensor) -> Tensor:
        return torch.log_softmax(logits.masked_fill(~valid_mask, -torch.inf), dim=-1)

    @staticmethod
    def _masked_std(values: Tensor, valid_mask: Tensor) -> Tensor:
        """Population standard deviation over each batch item's valid entries."""
        valid_count = valid_mask.sum(dim=(-2, -1)).clamp_min(1)
        masked_values = torch.where(valid_mask, values, torch.zeros_like(values))
        mean = masked_values.sum(dim=(-2, -1)) / valid_count
        # Replace invalid -inf logits before subtracting the mean.  Applying
        # torch.where after ``(-inf - mean).square()`` would be finite in the
        # forward result but can still introduce invalid gradients.
        safe_values = torch.where(valid_mask, values, mean[:, None, None])
        squared_error = (safe_values - mean[:, None, None]).square()
        return torch.sqrt(squared_error.sum(dim=(-2, -1)) / valid_count)
