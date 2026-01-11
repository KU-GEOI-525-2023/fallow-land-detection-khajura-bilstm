from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class LandCoverDataset(Dataset):
    """Time-series land cover dataset for multi-year Sentinel-2 observations."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        split: str = "train",
        features_file: str | Path = "X_train.npy",
        labels_file: str | Path = "y_train.npy",
        val_split: float = 0.2,
        random_seed: int = 42,
        context_length: int | None = None,
        split_by: str = "auto",
        normalize: bool = False,
        augment: bool = False,
        max_samples_per_class: int | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got {split}")

        self.normalize = normalize
        self.augment = augment
        base_dir = Path(data_dir) if data_dir else Path.cwd()
        features_path = base_dir / features_file
        labels_path = base_dir / labels_file

        try:
            features = np.load(features_path, allow_pickle=False).astype(np.float32)
            labels = np.load(labels_path, allow_pickle=False).astype(np.int64)
            # Consolidation keeps older exports usable without re-extraction.
            labels[labels == 5] = 4
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Data files not found: {features_path}, {labels_path}"
            ) from exc

        if features.ndim != 3:
            raise ValueError(f"Expected (N, T, F), got {features.shape}")

        features, labels = self._balance_classes(
            features, labels, random_seed, max_samples_per_class
        )
        features = self._adjust_sequence_length(features, context_length)

        groups_path = base_dir / "sample_sources.npy"
        use_group_split = split_by == "group" or (
            split_by == "auto" and groups_path.exists()
        )

        if use_group_split:
            train_idx, val_idx = self._get_group_indices(
                features, labels, groups_path, val_split, random_seed
            )
        else:
            indices = np.arange(len(features))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_split,
                random_state=random_seed,
                stratify=labels,
            )

        self.indices = train_idx if split == "train" else val_idx
        self.features = features[self.indices]
        self.targets = labels[self.indices]

        if self.normalize:
            # Stats always from train split of THIS features array
            train_feats = features[train_idx]
            self.mean = train_feats.reshape(-1, train_feats.shape[-1]).mean(axis=0)
            self.std = train_feats.reshape(-1, train_feats.shape[-1]).std(axis=0)
            self.features = (self.features - self.mean) / (self.std + 1e-8)

    @staticmethod
    def _balance_classes(
        features: np.ndarray,
        labels: np.ndarray,
        random_seed: int,
        max_samples: int | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if max_samples is None:
            return features, labels

        rng = np.random.default_rng(random_seed)
        class_indices = [
            rng.choice(
                np.flatnonzero(labels == cls),
                min(max_samples, int((labels == cls).sum())),
                replace=False,
            )
            for cls in np.unique(labels)
        ]
        indices = np.sort(np.concatenate(class_indices))
        return features[indices], labels[indices]

    @staticmethod
    def _adjust_sequence_length(
        features: np.ndarray, context_length: int | None
    ) -> np.ndarray:
        if context_length is None:
            return features

        seq_len = features.shape[1]
        if seq_len > context_length:
            return features[:, -context_length:, :]

        if seq_len < context_length:
            n, _, f = features.shape
            padded = np.zeros((n, context_length, f), dtype=features.dtype)
            padded[:, -seq_len:, :] = features
            return padded

        return features

    @staticmethod
    def _get_group_indices(
        features: np.ndarray,
        labels: np.ndarray,
        groups_path: Path,
        val_split: float,
        random_seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            groups = np.load(groups_path, allow_pickle=False)
            if groups.shape[0] != features.shape[0]:
                raise ValueError(
                    f"Groups length {groups.shape[0]} != features {features.shape[0]}"
                )

            gss = GroupShuffleSplit(
                test_size=val_split, n_splits=1, random_state=random_seed
            )
            train_idx, val_idx = next(
                gss.split(np.arange(len(features)), labels, groups)
            )
            return train_idx, val_idx
        except Exception:
            indices = np.arange(len(features))
            return train_test_split(
                indices, test_size=val_split, random_state=random_seed, stratify=labels
            )

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features[index].copy()
        if self.augment:
            if np.random.random() > 0.5:
                # Noise scale is tuned to avoid overpowering small classes.
                x += np.random.normal(0, 0.015, x.shape).astype(np.float32)

            if np.random.random() > 0.5:
                scaling_factor = np.random.uniform(0.98, 1.02)
                x *= scaling_factor

            if np.random.random() > 0.8:
                t_len = x.shape[0]
                mask_len = np.random.randint(1, max(2, t_len // 20))
                start = np.random.randint(0, t_len - mask_len)
                x[start : start + mask_len, :] = 0

        return torch.from_numpy(x), torch.tensor(self.targets[index], dtype=torch.long)


def build_datasets(cfg: DictConfig | dict) -> tuple[LandCoverDataset, LandCoverDataset]:
    """Create train and validation datasets from config."""
    cfg = OmegaConf.create(cfg)
    data_cfg = cfg.get("data", cfg)

    context_length = None
    for config in [cfg.get("model"), cfg, data_cfg]:
        if config and (clen := config.get("context_length")) is not None:
            context_length = int(clen)
            break

    dataset_kwargs = dict(
        data_dir=data_cfg.get("data_dir"),
        features_file=data_cfg.get("features_file", "X_train.npy"),
        labels_file=data_cfg.get("labels_file", "y_train.npy"),
        val_split=data_cfg.get("val_split", 0.2),
        random_seed=data_cfg.get("random_seed", 42),
        context_length=context_length,
        split_by=data_cfg.get("split_by", "auto"),
        normalize=data_cfg.get("normalize", False),
        max_samples_per_class=data_cfg.get("max_samples_per_class"),
    )

    return (
        LandCoverDataset(**dataset_kwargs, split="train", augment=True),
        LandCoverDataset(**dataset_kwargs, split="val", augment=False),
    )


def get_dataloader(
    data_dir: str | Path | None = None,
    batch_size: int = 32,
    split: str = "train",
    shuffle: bool = True,
    val_split: float = 0.2,
    random_seed: int = 42,
    context_length: int | None = None,
) -> DataLoader:
    """Create a single dataloader (backwards-compatible API)."""
    dataset = LandCoverDataset(
        data_dir=data_dir,
        split=split,
        val_split=val_split,
        random_seed=random_seed,
        context_length=context_length,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_dataloaders(cfg: DictConfig | dict) -> tuple[DataLoader, DataLoader]:
    """Return train and validation dataloaders from Hydra config."""
    cfg = OmegaConf.create(cfg)
    train_ds, val_ds = build_datasets(cfg)

    data_cfg = cfg.get("data", cfg)
    train_cfg = cfg.get("train", cfg)

    loader_kwargs = dict(
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=data_cfg.get("pin_memory", True) and torch.cuda.is_available(),
    )

    use_weighted = train_cfg.get("use_weighted_sampler", False)
    sampler = _build_weighted_sampler(train_ds, cfg) if use_weighted else None

    return (
        DataLoader(
            train_ds, shuffle=not use_weighted, sampler=sampler, **loader_kwargs
        ),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
    )


def compute_class_weights(
    labels_path: str | Path,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Compute inverse-frequency class weights."""
    labels = np.load(labels_path, allow_pickle=False)
    # Ensure consistent ID mapping for consolidated classes
    labels[labels == 5] = 4
    num_classes = num_classes or 5
    counts = np.bincount(labels, minlength=num_classes)
    total = counts.sum()
    weights = np.where(counts > 0, total / (num_classes * counts), 0.0)
    return torch.tensor(weights, dtype=torch.float32)


def _build_weighted_sampler(
    dataset: LandCoverDataset, cfg: DictConfig
) -> WeightedRandomSampler:
    labels = np.array(dataset.targets, dtype=np.int64)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)

    data_cfg = cfg.get("data", cfg)
    train_cfg = cfg.get("train", cfg)
    power = float(train_cfg.get("sampler_power", data_cfg.get("sampler_power", 1.0)))

    weights = np.where(counts > 0, counts ** (-power), 0.0)
    return WeightedRandomSampler(weights[labels], len(dataset), replacement=True)
