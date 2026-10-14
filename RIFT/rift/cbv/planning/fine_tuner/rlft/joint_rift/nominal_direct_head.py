"""V6 nominal-conditioned direct scoring, without a bounded prior residual.

Quality and interaction are supervised separately. Deployment gates use only
network predictions, never evaluator labels. Predictions are not safety proofs.
"""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class DirectOutput:
    base_logits: torch.Tensor
    joint_logits: torch.Tensor
    raw_delta: torch.Tensor
    delta: torch.Tensor
    base_std: torch.Tensor
    residual_scale: torch.Tensor
    valid_pair_mask: torch.Tensor
    quality_logits: torch.Tensor
    interaction: torch.Tensor
    quality_threshold: float
    min_gain: float


class NominalDirectScorer(nn.Module):
    def __init__(self, candidate_dim=128, hidden_dim=64, trajectory_frames=40,
                 trajectory_stride=5, prior_weight=.05, quality_threshold=.8, min_gain=.05, head_kind="joint"):
        super().__init__()
        if trajectory_frames % trajectory_stride or not 0 < quality_threshold < 1 or min_gain < 0 or prior_weight < 0:
            raise ValueError("invalid direct-head configuration")
        if head_kind not in {"joint", "unary_guarded"}:
            raise ValueError("unknown direct-head kind")
        self.config = dict(head_type="nominal_direct", candidate_dim=candidate_dim, hidden_dim=hidden_dim,
            trajectory_frames=trajectory_frames, trajectory_stride=trajectory_stride, prior_weight=prior_weight,
            quality_threshold=quality_threshold, min_gain=min_gain, head_kind=head_kind)
        self.trajectory_frames = trajectory_frames
        self.register_buffer("sample_indices", torch.arange(0, trajectory_frames, trajectory_stride), persistent=False)
        n = len(self.sample_indices) * 5
        self.nominal_encoder = nn.Sequential(nn.Linear(n, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.candidate_encoder = nn.Sequential(nn.Linear(candidate_dim + 2 * n, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.member_pressure = nn.Linear(hidden_dim, 1)
        self.pair_encoder = nn.Sequential(nn.Linear(6 * hidden_dim + 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.action = nn.Linear(hidden_dim, 1)
        self.unary_action = nn.Linear(hidden_dim, 1)
        self.quality = nn.Linear(hidden_dim, 1)
        self.pressure = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.action.weight)
        nn.init.zeros_(self.action.bias)
        nn.init.zeros_(self.unary_action.weight)
        nn.init.zeros_(self.unary_action.bias)
        nn.init.zeros_(self.quality.weight)
        nn.init.zeros_(self.quality.bias)
        nn.init.zeros_(self.pressure.weight)
        nn.init.zeros_(self.pressure.bias)

    def forward(self, features1, traj1, raw1, logits1, features2, traj2, raw2, logits2,
                ego_nominal, valid_mask_1=None, valid_mask_2=None):
        if ego_nominal.ndim != 3 or ego_nominal.shape[-2] < self.trajectory_frames or ego_nominal.shape[-1] < 4:
            raise ValueError("direct head requires [B,T>=horizon,4] nominal x,y,heading,speed")
        if not all(torch.isfinite(x).all() for x in (features1, traj1, raw1, logits1, features2, traj2, raw2, logits2, ego_nominal)):
            raise ValueError("nonfinite direct-head input")
        indices = self.sample_indices
        ego = ego_nominal.index_select(-2, indices)
        e = torch.cat((ego[..., :2], ego[..., 2:3].sin(), ego[..., 2:3].cos(), ego[..., 3:4]), -1)
        nominal = self.nominal_encoder(e.flatten(-2))
        reps = []
        for features, traj, raw in ((features1, traj1, raw1), (features2, traj2, raw2)):
            t = traj.index_select(-2, indices)
            speed = torch.linalg.vector_norm(raw.index_select(-2, indices)[..., 4:6], dim=-1, keepdim=True)
            own = torch.cat((t[..., :2], t[..., 2:3].sin(), t[..., 2:3].cos(), speed), -1)
            relative = own - e[:, None]
            reps.append(self.candidate_encoder(torch.cat((features, own.flatten(-2), relative.flatten(-2)), -1)))
        a, b = reps[0][:, :, None], reps[1][:, None, :]
        m1 = torch.ones_like(logits1, dtype=torch.bool) if valid_mask_1 is None else valid_mask_1.bool()
        m2 = torch.ones_like(logits2, dtype=torch.bool) if valid_mask_2 is None else valid_mask_2.bool()
        ref1 = logits1.masked_fill(~m1, -torch.inf).argmax(-1)
        ref2 = logits2.masked_fill(~m2, -torch.inf).argmax(-1)
        batch = torch.arange(len(features1), device=features1.device)
        da = a - reps[0][batch, ref1][:, None, None]
        db = b - reps[1][batch, ref2][:, None, None]
        distance = torch.linalg.vector_norm(traj1[:, :, None, :self.trajectory_frames, :2] - traj2[:, None, :, :self.trajectory_frames, :2], dim=-1)
        geometry = torch.stack((distance.amin(-1), distance.mean(-1)), -1)
        context = nominal[:, None, None].expand(-1, a.shape[1], b.shape[2], -1)
        pair = self.pair_encoder(torch.cat((a + b, (a - b).abs(), a * b, context, da + db, (da - db).abs(), geometry), -1))
        valid = m1[:, :, None] & m2[:, None, :]
        if not valid.flatten(1).any(-1).all():
            raise ValueError("each state needs valid candidates")
        base = torch.log_softmax(logits1.masked_fill(~m1, -torch.inf), -1)[:, :, None] + torch.log_softmax(logits2.masked_fill(~m2, -torch.inf), -1)[:, None, :]
        prior = torch.where(valid, base, torch.zeros_like(base))
        centered = prior - base.flatten(1).max(-1).values[:, None, None]
        # Unbounded action logits; prior is a small preference, not an amplitude cap.
        raw = (self.action(pair).squeeze(-1) if self.config["head_kind"] == "joint" else
               self.unary_action(reps[0]).squeeze(-1)[:, :, None] + self.unary_action(reps[1]).squeeze(-1)[:, None, :])
        scores = (raw + self.config["prior_weight"] * centered).masked_fill(~valid, -torch.inf)
        quality = self.quality(pair).squeeze(-1)
        interaction = self.pressure(pair).squeeze(-1)
        delta = torch.where(valid, scores - prior, torch.zeros_like(scores))
        count = valid.sum((1, 2)).clamp_min(1)
        mean = prior.sum((1, 2)) / count
        std = torch.sqrt(torch.where(valid, (prior - mean[:, None, None]).square(), torch.zeros_like(prior)).sum((1, 2)) / count)
        delta_mean = delta.sum((1, 2)) / count
        delta_std = torch.sqrt(torch.where(valid, (delta - delta_mean[:, None, None]).square(), torch.zeros_like(delta)).sum((1, 2)) / count)
        result = DirectOutput(base, scores, raw, delta, std, delta_std / std.clamp_min(1e-6), valid,
                              quality, interaction, self.config["quality_threshold"], self.config["min_gain"])
        result.member_pressure_1 = self.member_pressure(reps[0]).squeeze(-1)
        result.member_pressure_2 = self.member_pressure(reps[1]).squeeze(-1)
        return result


def select_direct(output):
    if output.joint_logits.shape[0] != 1:
        raise ValueError("selection requires one state")
    shape = output.joint_logits.shape[1:]
    baseline_flat = int(output.base_logits[0].argmax())
    baseline = (baseline_flat // shape[1], baseline_flat % shape[1])
    eligible = output.valid_pair_mask[0] & (output.quality_logits[0].sigmoid() >= output.quality_threshold)
    if not eligible.any():
        return baseline
    flat = int(output.joint_logits[0].masked_fill(~eligible, -torch.inf).argmax())
    selected = (flat // shape[1], flat % shape[1])
    if eligible[baseline] and (output.interaction[0][selected] - output.interaction[0][baseline] < output.min_gain):
        return baseline
    return selected
