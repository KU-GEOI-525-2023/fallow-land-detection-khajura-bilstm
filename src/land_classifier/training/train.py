from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from land_classifier.data import build_dataloaders, compute_class_weights
from land_classifier.models import build_model
from land_classifier.utils import ensure_dir, get_logger, set_seed

log = get_logger(__name__)


@dataclass(slots=True)
class EvaluationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    conf_matrix: np.ndarray | None = None
    per_class_precision: np.ndarray | None = None
    per_class_recall: np.ndarray | None = None


class EarlyStopping:
    """Stop training when validation metric plateaus."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: float | None = None
        self.early_stop = False

    def __call__(self, score: float) -> None:
        if self.best_score is None:
            self.best_score = score
            return

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance by down-weighting easy examples."""

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none", weight=self.weight
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()


def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EvaluationMetrics:
    """Compute validation metrics on a dataset."""
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, labels).item()
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(1, len(loader))
    acc = accuracy_score(targets, predictions)
    per_cls_prec, per_cls_rec, _, _ = precision_recall_fscore_support(
        targets, predictions, average=None, zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        targets, predictions, average="weighted", zero_division=0
    )

    return EvaluationMetrics(
        loss=avg_loss,
        accuracy=acc,
        precision=prec_w,
        recall=rec_w,
        f1=f1_w,
        conf_matrix=confusion_matrix(targets, predictions),
        per_class_precision=per_cls_prec,
        per_class_recall=per_cls_rec,
    )


def _save_training_plots(
    ckpt_dir: Path,
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
    val_f1s: list[float],
    val_precs: list[float],
    val_recs: list[float],
    cm: np.ndarray | None,
    prec_per_cls: np.ndarray | None,
    rec_per_cls: np.ndarray | None,
    lrs: list[float] | None,
    class_names: list[str],
) -> None:
    """Save training curves and metrics visualizations."""
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(ckpt_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, val_accs, label="accuracy")
    plt.plot(epochs, val_f1s, label="f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(ckpt_dir / "metrics_curve.png")
    plt.close()

    if cm is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.colorbar()
        labels = (
            class_names
            if len(class_names) == cm.shape[0]
            else [f"c{i}" for i in range(cm.shape[0])]
        )
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(np.arange(len(labels)), labels)
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha="center", va="center", color="black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(ckpt_dir / "confusion_matrix.png")
        plt.close()

    if prec_per_cls is not None and rec_per_cls is not None:
        labels = (
            class_names
            if len(class_names) == len(prec_per_cls)
            else [f"c{i}" for i in range(prec_per_cls.shape[0])]
        )
        x = np.arange(len(labels))
        width = 0.35
        plt.figure(figsize=(8, 4))
        plt.bar(x - width / 2, prec_per_cls, width, label="precision")
        plt.bar(x + width / 2, rec_per_cls, width, label="recall")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis="y")
        plt.tight_layout()
        plt.savefig(ckpt_dir / "per_class_pr.png")
        plt.close()

    if lrs:
        plt.figure(figsize=(8, 3))
        plt.plot(epochs, lrs)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.grid()
        plt.tight_layout()
        plt.savefig(ckpt_dir / "lr_curve.png")
        plt.close()


@hydra.main(config_path="../../../configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info("Hydra directory: %s", Path().resolve())
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(int(cfg.project.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    class_names = [
        str(name)
        for name in cfg.data.get(
            "class_names", ["Abandoned", "Active", "Fallow", "Vegetation"]
        )
    ]
    ckpt_dir = Path(ensure_dir(cfg.train.checkpoint_dir))

    train_loader, val_loader = build_dataloaders(cfg)

    if cfg.data.get("normalize"):
        ds = train_loader.dataset
        if hasattr(ds, "mean") and hasattr(ds, "std"):
            log.info("Normalization: mean=%s std=%s", ds.mean.tolist(), ds.std.tolist())
            stats = {"mean": ds.mean, "std": ds.std}
            torch.save(stats, ckpt_dir / "normalization_stats.pt")

    model = build_model(cfg.model).to(device)

    weight_tensor = None
    if cw := cfg.train.get("class_weights"):
        if cw == "auto":
            labels_path = Path(cfg.data.data_dir) / cfg.data.labels_file
            weight_tensor = compute_class_weights(labels_path, cfg.model.num_classes)

            boost = cfg.train.get("focal_boost_active_fallow", 1.0)
            if boost > 1.0 and weight_tensor.size(0) > 2:
                weight_tensor[1] *= boost * 0.9
                weight_tensor[2] *= boost

                weight_tensor = torch.clamp(weight_tensor, max=3.5)
                log.info("Boosted and Capped weights: %s", weight_tensor.tolist())
            else:
                log.info("Class weights: %s", weight_tensor.tolist())
        elif isinstance(cw, (list, tuple)):
            weight_tensor = torch.tensor(cw, dtype=torch.float32)

    if cfg.train.get("use_focal", False):
        criterion = FocalLoss(
            weight=weight_tensor.to(device) if weight_tensor is not None else None,
            gamma=cfg.train.get("focal_gamma", 2.0),
        )
        log.info("Using Focal Loss (gamma=%s)", cfg.train.focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(
            weight=weight_tensor.to(device) if weight_tensor is not None else None,
            label_smoothing=cfg.train.get("label_smoothing", 0.0),
        )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.get("weight_decay", 1e-4),
    )

    scheduler = None
    if sch_type := cfg.train.get("scheduler"):
        if sch_type == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=cfg.train.get("scheduler_factor", 0.5),
                patience=cfg.train.get("scheduler_patience", 10),
                min_lr=1e-6,
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.train.epochs, eta_min=1e-6
            )

    best_val_f1 = 0.0
    early_stopping = EarlyStopping(
        patience=cfg.train.get("patience", 10),
        min_delta=cfg.train.get("min_delta", 0.0),
    )

    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    val_precs: list[float] = []
    val_recs: list[float] = []
    val_f1s: list[float] = []
    lrs: list[float] = []

    last_cm: np.ndarray | None = None
    last_prec_cls: np.ndarray | None = None
    last_rec_cls: np.ndarray | None = None

    clip_norm = cfg.train.get("clip_norm", 0.0)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(batch_loss=f"{loss.item():.4f}")

        train_loss = epoch_loss / max(1, len(train_loader))
        metrics = evaluate(model, val_loader, criterion, device)

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(metrics.f1)
            else:
                scheduler.step()

        log.info(
            "Epoch %03d/%d | Loss (T/V): %.4f / %.4f | Val F1: %.4f | Acc: %.4f",
            epoch,
            cfg.train.epochs,
            train_loss,
            metrics.loss,
            metrics.f1,
            metrics.accuracy,
        )

        epochs.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(metrics.loss)
        val_accs.append(metrics.accuracy)
        val_precs.append(metrics.precision)
        val_recs.append(metrics.recall)
        val_f1s.append(metrics.f1)
        lrs.append(optimizer.param_groups[0]["lr"])

        last_cm = metrics.conf_matrix
        last_prec_cls = metrics.per_class_precision
        last_rec_cls = metrics.per_class_recall

        if metrics.f1 > best_val_f1:
            best_val_f1 = metrics.f1
            torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
            log.info("Best model saved (f1=%.4f)", best_val_f1)

        early_stopping(metrics.f1)
        if early_stopping.early_stop:
            log.info("Early stopping at epoch %d", epoch)
            break

        if epoch % cfg.train.save_every == 0:
            torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch}.pt")

    try:
        metrics_csv = ckpt_dir / "training_metrics.csv"
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                ]
            )
            for e, tl, vl, acc, prec, rec, f1 in zip(
                epochs,
                train_losses,
                val_losses,
                val_accs,
                val_precs,
                val_recs,
                val_f1s,
            ):
                writer.writerow([e, tl, vl, acc, prec, rec, f1])

        _save_training_plots(
            ckpt_dir,
            epochs,
            train_losses,
            val_losses,
            val_accs,
            val_f1s,
            val_precs,
            val_recs,
            last_cm,
            last_prec_cls,
            last_rec_cls,
            lrs,
            class_names,
        )
        log.info("Metrics saved to %s", metrics_csv)
    except Exception as exc:
        log.warning("Failed to save plots: %s", exc)

    log.info("Training complete")


if __name__ == "__main__":
    main()
