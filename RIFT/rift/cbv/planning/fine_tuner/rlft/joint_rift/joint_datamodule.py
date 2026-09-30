"""Padding-aware Lightning data module for complete Joint-RIFT states."""

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split

from rift.cbv.planning.fine_tuner.rlft.joint_rift.joint_buffer import JointRolloutBuffer
from rift.cbv.planning.pluto.feature_builder.pluto_feature import PlutoFeature


class JointRIFTCollate:
    """Pad two independent candidate axes and preserve their outer product mask."""

    def __init__(self, feature_collate: Callable = PlutoFeature.collate):
        self.feature_collate = feature_collate

    def __call__(self, batch: List[Dict]) -> Dict:
        if not batch:
            raise ValueError("cannot collate an empty Joint-RIFT batch")
        group_sizes_1 = [len(np.asarray(sample["valid_mask_1"])) for sample in batch]
        group_sizes_2 = [len(np.asarray(sample["valid_mask_2"])) for sample in batch]
        max_group_1, max_group_2 = max(group_sizes_1), max(group_sizes_2)
        batch_size = len(batch)

        old_joint_logits = torch.full(
            (batch_size, max_group_1, max_group_2), -torch.inf, dtype=torch.float32
        )
        joint_advantage = torch.zeros(
            (batch_size, max_group_1, max_group_2), dtype=torch.float32
        )
        valid_mask_1 = torch.zeros((batch_size, max_group_1), dtype=torch.bool)
        valid_mask_2 = torch.zeros((batch_size, max_group_2), dtype=torch.bool)
        pair_ids = []

        for batch_index, sample in enumerate(batch):
            logits = torch.as_tensor(sample["old_joint_logits"], dtype=torch.float32)
            advantage = torch.as_tensor(sample["joint_advantage"], dtype=torch.float32)
            mask_1 = torch.as_tensor(sample["valid_mask_1"], dtype=torch.bool)
            mask_2 = torch.as_tensor(sample["valid_mask_2"], dtype=torch.bool)
            if logits.ndim != 2 or advantage.shape != logits.shape:
                raise ValueError("each joint matrix must have shape [G1, G2]")
            group_1, group_2 = logits.shape
            if mask_1.shape != (group_1,) or mask_2.shape != (group_2,):
                raise ValueError("candidate masks must match each sample's joint matrix")
            old_joint_logits[batch_index, :group_1, :group_2] = logits
            joint_advantage[batch_index, :group_1, :group_2] = advantage
            valid_mask_1[batch_index, :group_1] = mask_1
            valid_mask_2[batch_index, :group_2] = mask_2
            pair_ids.append(tuple(sample["pair_ids"]))

        joint_valid_mask = valid_mask_1[:, :, None] & valid_mask_2[:, None, :]
        old_joint_logits = old_joint_logits.masked_fill(~joint_valid_mask, -torch.inf)
        joint_advantage = joint_advantage.masked_fill(~joint_valid_mask, 0.0)
        return {
            "feature_1": self.feature_collate([sample["feature_1"] for sample in batch]),
            "feature_2": self.feature_collate([sample["feature_2"] for sample in batch]),
            "old_joint_logits": old_joint_logits,
            "joint_old_logits": old_joint_logits,
            "joint_advantage": joint_advantage,
            "valid_mask_1": valid_mask_1,
            "valid_mask_2": valid_mask_2,
            "joint_valid_mask": joint_valid_mask,
            "pair_ids": pair_ids,
            "frame_1": _collate_frame_context(batch, "frame_1"),
            "frame_2": _collate_frame_context(batch, "frame_2"),
            "diagnostics": [sample.get("diagnostics", {}) for sample in batch],
        }


class JointRIFTDataset(Dataset):
    def __init__(self, buffer: JointRolloutBuffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer.sample(index)


class JointRIFTDataModule(LightningDataModule):
    """90/10 state-level split with conservative joint-matrix batch sizes."""

    def __init__(self, cfg: DictConfig, buffer: JointRolloutBuffer):
        super().__init__()
        self.cfg = cfg
        self.buffer = buffer
        self.train_batch_size = int(cfg.train_batch_size)
        self.val_batch_size = int(cfg.val_batch_size)
        self.shuffle = bool(cfg.shuffle)
        self.num_workers = int(cfg.num_workers)
        self.pin_memory = bool(cfg.pin_memory)
        self.persistent_workers = bool(cfg.persistent_workers) and self.num_workers > 0
        self.train_ratio = float(cfg.train_ratio)
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be strictly between zero and one")

    def setup(self, stage: Optional[str] = None):
        if not self.buffer.buffer_full:
            raise RuntimeError("Joint-RIFT buffer must be full before training")
        if stage not in (None, "fit"):
            raise ValueError(f"Joint-RIFT only supports fit, got {stage}")
        dataset = JointRIFTDataset(self.buffer)
        if len(dataset) < 2:
            raise RuntimeError("Joint-RIFT needs at least two states for a train/validation split")
        train_size = int(len(dataset) * self.train_ratio)
        train_size = min(max(train_size, 1), len(dataset) - 1)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    def preprocess_buffer(self):
        """Joint advantages are computed at collection time; no GAE phase exists."""

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            collate_fn=JointRIFTCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=JointRIFTCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
        )


def _collate_frame_context(batch: List[Dict], key: str) -> Dict[str, torch.Tensor]:
    """Stack fixed-size frame transforms needed to rebuild common geometry."""
    contexts = [sample[key] for sample in batch]
    required = ("source_origin", "source_heading", "ego_origin", "ego_heading")
    if any(set(required) - set(context) for context in contexts):
        raise KeyError(f"{key} is missing geometry fields")
    return {
        "source_origin": torch.as_tensor(
            np.stack([context["source_origin"] for context in contexts]), dtype=torch.float32
        ),
        "source_heading": torch.as_tensor(
            [context["source_heading"] for context in contexts], dtype=torch.float32
        ),
        "ego_origin": torch.as_tensor(
            np.stack([context["ego_origin"] for context in contexts]), dtype=torch.float32
        ),
        "ego_heading": torch.as_tensor(
            [context["ego_heading"] for context in contexts], dtype=torch.float32
        ),
    }
