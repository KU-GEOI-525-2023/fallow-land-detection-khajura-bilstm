from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs/baselines/rule_based.yaml"


def _require(value: object, name: str) -> object:
    if value is None:
        raise ValueError(f"Missing required config value: {name}")
    return value


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> DictConfig:
    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    return cfg


def evaluate_rules(cfg: DictConfig | dict, eval_mode: str | None = None) -> None:
    cfg = OmegaConf.create(cfg)
    eval_cfg = cfg.get("evaluation", cfg)

    data_dir = _resolve_path(eval_cfg.get("data_dir", "data/processed"))
    output_dir = _resolve_path(eval_cfg.get("output_dir", "outputs"))
    x_path = _resolve_path(eval_cfg.get("x_file", data_dir / "X_train.npy"))
    y_path = _resolve_path(eval_cfg.get("y_file", data_dir / "y_train.npy"))
    groups_path = _resolve_path(
        eval_cfg.get("groups_file", data_dir / "sample_sources.npy")
    )
    class_names = list(eval_cfg.get("class_names", []))
    if not class_names:
        raise ValueError("evaluation.class_names must be defined in the config.")

    thresholds_cfg = eval_cfg.get("thresholds", {})
    threshold = float(thresholds_cfg.get("peak", 0.6))
    veg_mean = float(thresholds_cfg.get("veg_mean", 0.5))
    other_mean = float(thresholds_cfg.get("other_mean", 0.25))

    ndvi_index = int(eval_cfg.get("ndvi_index", 6))
    year_steps = int(eval_cfg.get("year_steps", 18))
    eval_mode = eval_mode or eval_cfg.get("eval_mode", "val")
    random_seed = int(eval_cfg.get("random_seed", 42))
    val_split = float(eval_cfg.get("val_split", 0.2))
    outputs_cfg = eval_cfg.get("outputs", {})

    try:
        X = np.load(x_path)
        y_true = np.load(y_path)
    except FileNotFoundError:
        print(f"Error: Data files not found in {DATA_DIR}")
        return

    y_true[y_true == 5] = 4
    print(f"Total samples in pool: {len(y_true)}")

    if eval_mode == "val":
        try:
            groups = np.load(groups_path)
            gss = GroupShuffleSplit(
                test_size=val_split, n_splits=1, random_state=random_seed
            )
            _, val_idx = next(gss.split(X, y_true, groups))
            indices = val_idx
            print(f"Mode: Validation Split (N={len(indices)})")
        except FileNotFoundError:
            print(
                "Warning: sample_sources.npy not found. Evaluated on full dataset instead."
            )
            indices = np.arange(len(y_true))
    else:
        indices = np.arange(len(y_true))
        print(f"Mode: Full Dataset (N={len(indices)})")

    X = X[indices]
    y_true = y_true[indices]

    ndvi = X[:, :, ndvi_index]
    total_steps = ndvi.shape[1]
    if total_steps <= year_steps * 2:
        raise ValueError(
            f"Not enough timesteps ({total_steps}) for year_steps={year_steps}."
        )
    y1_max = np.max(ndvi[:, 0:year_steps], axis=1)
    y2_max = np.max(ndvi[:, year_steps : year_steps * 2], axis=1)
    y3_max = np.max(ndvi[:, year_steps * 2 :], axis=1)
    mean_ndvi = np.mean(ndvi, axis=1)

    y_pred = np.empty_like(y_true)

    active_mask = y2_max > threshold
    fallow_mask = ~active_mask & ((y1_max > threshold) | (y3_max > threshold))
    remaining_mask = ~(active_mask | fallow_mask)

    veg_mask = remaining_mask & (mean_ndvi > veg_mean)
    other_mask = remaining_mask & (mean_ndvi < other_mean)
    abandoned_mask = remaining_mask & ~(veg_mask | other_mask)

    y_pred[active_mask] = 1
    y_pred[fallow_mask] = 2
    y_pred[veg_mask] = 3
    y_pred[other_mask] = 4
    y_pred[abandoned_mask] = 0

    print("\n" + "=" * 50)
    print(f"RULE-BASED EVALUATION ({eval_mode.upper()})")
    print("=" * 50)
    labels = list(range(len(class_names)))
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            labels=labels,
            zero_division=0,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
    )
    plt.title(f"Rule-Based Confusion Matrix ({eval_mode.upper()})")
    cm_filename = outputs_cfg.get("cm_filename", "rule_based_cm_{mode}.png").format(
        mode=eval_mode
    )
    plt.savefig(output_dir / cm_filename)
    plt.close()

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    metrics_df = pd.DataFrame(report_dict).transpose().iloc[:5]
    metrics_df[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(12, 6))
    plt.title(f"Rule-Based Performance ({eval_mode.upper()})")
    plt.tight_layout()
    metrics_filename = outputs_cfg.get(
        "metrics_filename", "rule_based_metrics_{mode}.png"
    ).format(mode=eval_mode)
    plt.savefig(output_dir / metrics_filename)
    plt.close()

    accuracy = float((y_pred == y_true).sum() / len(y_true))
    print(f"Overall Accuracy: {accuracy:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule-based classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to rule-based config YAML.",
    )
    parser.add_argument("--mode", choices=["val", "full"], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_rules(load_config(args.config), eval_mode=args.mode)
