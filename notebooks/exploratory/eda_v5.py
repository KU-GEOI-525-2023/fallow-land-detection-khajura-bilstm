"""Exploratory data analysis for expanded land cover time series data (5 classes - Consolidated)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs/eda/v5.yaml"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA v5 with a config file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to EDA v5 config YAML.",
    )
    return parser.parse_args()


def get_date_labels(
    num_steps: int, start_year: int = 2022, step_days: int = 20
) -> list[pd.Timestamp]:
    start_date = pd.to_datetime(f"{start_year}-01-01")
    dates = [start_date + pd.Timedelta(days=i * step_days) for i in range(num_steps)]
    return dates


def plot_mean_timeseries(
    X: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    output_path: Path,
    feature_names: list[str],
    reverse_class_map: dict[int, str],
    start_year: int,
    step_days: int,
    title_template: str,
    figsize: tuple[int, int],
) -> None:
    plt.figure(figsize=figsize)
    feat_idx = feature_names.index(feature_name)
    num_steps = X.shape[1]
    dates = get_date_labels(num_steps, start_year=start_year, step_days=step_days)

    for c_idx, c_name in reverse_class_map.items():
        mask = y == c_idx
        if np.any(mask):
            class_data = X[mask, :, feat_idx]
            mean_vals = np.mean(class_data, axis=0)
            std_vals = np.std(class_data, axis=0)

            plt.plot(dates, mean_vals, label=c_name, linewidth=2)
            plt.fill_between(
                dates, mean_vals - std_vals, mean_vals + std_vals, alpha=0.1
            )

    plt.title(title_template.format(feature=feature_name.upper()), fontsize=15)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel(feature_name.upper(), fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


def plot_sample_trajectories(
    X: np.ndarray,
    y: np.ndarray,
    feature_name: str,
    output_path: Path,
    feature_names: list[str],
    class_names: list[str],
    reverse_class_map: dict[int, str],
    start_year: int,
    step_days: int,
    num_samples_per_class: int,
    title_template: str,
    figsize_width: int,
    figsize_height_per_class: int,
) -> None:
    feat_idx = feature_names.index(feature_name)
    num_steps = X.shape[1]
    dates = get_date_labels(num_steps, start_year=start_year, step_days=step_days)
    fig, axes = plt.subplots(
        len(class_names),
        1,
        figsize=(figsize_width, figsize_height_per_class * len(class_names)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for i, (c_idx, c_name) in enumerate(reverse_class_map.items()):
        mask = y == c_idx
        if not np.any(mask):
            continue

        indices = np.where(mask)[0]
        selected_indices = np.random.choice(
            indices, min(num_samples_per_class, len(indices)), replace=False
        )

        for idx in selected_indices:
            axes[i].plot(dates, X[idx, :, feat_idx], alpha=0.7)

        axes[i].set_title(
            title_template.format(class_name=c_name, feature=feature_name.upper())
        )
        axes[i].set_ylabel(feature_name.upper())
        axes[i].grid(True, alpha=0.2)

        axes[i].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
        axes[i].xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())

    plt.xlabel("Year")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved samples plot: {output_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})
    plots_cfg = cfg.get("plots", {})
    date_cfg = cfg.get("date", {})

    output_file_x = _resolve_path(
        _require(paths_cfg.get("output_file_x"), "paths.output_file_x")
    )
    output_file_y = _resolve_path(
        _require(paths_cfg.get("output_file_y"), "paths.output_file_y")
    )
    output_dir = _resolve_path(
        _require(paths_cfg.get("output_dir"), "paths.output_dir")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = list(cfg.get("class_names", []))
    if not class_names:
        raise ValueError("class_names must be defined in the config.")
    reverse_class_map = {i: name for i, name in enumerate(class_names)}

    feature_names = list(cfg.get("feature_names", []))
    indices = list(cfg.get("indices", []))
    start_year = int(date_cfg.get("start_year", 2022))
    step_days = int(date_cfg.get("step_days", 20))

    print("--- Starting Advanced EDA v5 ---")

    if not output_file_x.exists() or not output_file_y.exists():
        print("Data files not found. Extraction failed or hasn't run.")
        return

    X = np.load(output_file_x)
    y = np.load(output_file_y)

    print(f"Data Loaded: X={X.shape}, y={y.shape}")

    # 1. Class Distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = {reverse_class_map.get(u, str(u)): count for u, count in zip(unique, counts)}
    print("\nClass Distribution:")
    for name, count in dist.items():
        print(f"  {name:12}: {count}")

    class_dist_cfg = plots_cfg.get("class_distribution", {})
    plt.figure(figsize=tuple(class_dist_cfg.get("figsize", (10, 6))))
    sns.barplot(
        x=list(dist.keys()),
        y=list(dist.values()),
        palette=class_dist_cfg.get("palette", "magma"),
    )
    plt.title(class_dist_cfg.get("title", "Class Distribution"))
    plt.ylabel("Pixel Count")
    class_dist_path = output_dir / class_dist_cfg.get(
        "filename", "class_distribution.png"
    )
    plt.savefig(class_dist_path)
    plt.close()

    # 2. Key Indices Over Time
    mean_cfg = plots_cfg.get("mean_timeseries", {})
    samples_cfg = plots_cfg.get("samples", {})
    for idx_feat in indices:
        mean_output = output_dir / mean_cfg.get(
            "filename_template", "mean_{feature}.png"
        ).format(feature=idx_feat)
        plot_mean_timeseries(
            X,
            y,
            idx_feat,
            mean_output,
            feature_names=feature_names,
            reverse_class_map=reverse_class_map,
            start_year=start_year,
            step_days=step_days,
            title_template=mean_cfg.get(
                "title_template", "Mean {feature} Time Series by Class"
            ),
            figsize=tuple(mean_cfg.get("figsize", (14, 7))),
        )

        samples_output = output_dir / samples_cfg.get(
            "filename_template", "samples_{feature}.png"
        ).format(feature=idx_feat)
        plot_sample_trajectories(
            X,
            y,
            idx_feat,
            samples_output,
            feature_names=feature_names,
            class_names=class_names,
            reverse_class_map=reverse_class_map,
            start_year=start_year,
            step_days=step_days,
            num_samples_per_class=int(samples_cfg.get("num_samples_per_class", 3)),
            title_template=samples_cfg.get(
                "title_template",
                "Class: {class} - Sample {feature} Trajectories",
            ),
            figsize_width=int(samples_cfg.get("figsize_width", 15)),
            figsize_height_per_class=int(
                samples_cfg.get("figsize_height_per_class", 4)
            ),
        )

    # 3. Boxplot of indices distribution (averaging over time)
    print("\nCalculating summary stats for boxplots...")
    data_list = []
    for i, c_name in reverse_class_map.items():
        mask = y == i
        if not np.any(mask):
            continue

        # Take mean across temporal axis for each index
        for feat in ["ndvi", "evi", "bsi"]:
            feat_idx = feature_names.index(feat)
            means = X[mask, :, feat_idx].mean(axis=1)
            for m in means:
                data_list.append({"Class": c_name, "Index": feat.upper(), "Value": m})

    df_box = pd.DataFrame(data_list)
    indices_boxplot_cfg = plots_cfg.get("indices_boxplot", {})
    plt.figure(figsize=tuple(indices_boxplot_cfg.get("figsize", (15, 8))))
    sns.boxplot(x="Index", y="Value", hue="Class", data=df_box)
    plt.title(
        indices_boxplot_cfg.get(
            "title", "Distribution of Temporal Mean Indices by Class"
        )
    )
    indices_boxplot_path = output_dir / indices_boxplot_cfg.get(
        "filename", "indices_boxplot.png"
    )
    plt.savefig(indices_boxplot_path)
    plt.close()

    # 4. Correlation between indices
    print("\nGenerating correlation heatmap (on temporal means)...")
    summary_features = []
    for feat in feature_names:
        summary_features.append(X[:, :, feature_names.index(feat)].mean(axis=1))

    df_corr = pd.DataFrame(np.array(summary_features).T, columns=feature_names)
    feature_corr_cfg = plots_cfg.get("feature_correlation", {})
    plt.figure(figsize=tuple(feature_corr_cfg.get("figsize", (10, 8))))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(
        feature_corr_cfg.get(
            "title", "Feature Correlation Matrix (Temporal Means)"
        )
    )
    feature_corr_path = output_dir / feature_corr_cfg.get(
        "filename", "feature_correlation.png"
    )
    plt.savefig(feature_corr_path)
    plt.close()

    print(f"\nEDA Complete. All results saved to {output_dir}")


if __name__ == "__main__":
    main()
