from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from land_classifier.visualization.config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    require,
    resolve_path,
)


def plot_land_distribution(
    csv_path: Path,
    output_png: Path,
    *,
    class_column: str,
    area_column: str,
    percent_column: str,
    bar_title: str,
    pie_title: str,
    cmap: str,
    pie_start_angle: float,
    explode: float,
    figsize: tuple[int, int],
    dpi: int,
) -> None:
    input_path = csv_path
    output_path = output_png

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
        return

    df = df.sort_values(by=area_column, ascending=False)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(df)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    bars = ax1.bar(df[class_column], df[area_column], color=colors)
    ax1.set_title(bar_title, fontsize=14, fontweight="bold")
    ax1.set_ylabel("Area (Ha)")
    ax1.set_xlabel("Land Status")

    for bar, row in zip(bars, df.itertuples(index=False)):
        height = bar.get_height()
        row_dict = row._asdict()
        ax1.annotate(
            f"{row_dict[area_column]:.1f} ha\n({row_dict[percent_column]:.1f}%)",
            (bar.get_x() + bar.get_width() / 2.0, height),
            ha="center",
            va="center",
            xytext=(0, 9),
            textcoords="offset points",
            fontsize=10,
        )

    ax2.pie(
        df[area_column],
        labels=df[class_column],
        autopct="%1.1f%%",
        startangle=pie_start_angle,
        colors=colors,
        explode=[explode] * len(df),
    )
    ax2.set_title(pie_title, fontsize=14, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    print(f"Visualization saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot land distribution summaries.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to visualization config YAML.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    land_cfg = cfg.get("land_distribution", {})

    input_csv = resolve_path(
        require(land_cfg.get("input_csv"), "land_distribution.input_csv")
    )
    output_png = resolve_path(
        require(land_cfg.get("output_png"), "land_distribution.output_png")
    )

    plot_land_distribution(
        input_csv,
        output_png,
        class_column=str(land_cfg.get("class_column", "Class_Name")),
        area_column=str(land_cfg.get("area_column", "Area_Ha")),
        percent_column=str(land_cfg.get("percent_column", "Percentage")),
        bar_title=str(
            land_cfg.get("bar_title", "Land Distribution by Area (Hectares)")
        ),
        pie_title=str(
            land_cfg.get("pie_title", "Proportional Land Distribution")
        ),
        cmap=str(land_cfg.get("cmap", "viridis")),
        pie_start_angle=float(land_cfg.get("pie_start_angle", 140)),
        explode=float(land_cfg.get("explode", 0.05)),
        figsize=tuple(land_cfg.get("figsize", (16, 7))),
        dpi=int(land_cfg.get("dpi", 100)),
    )


if __name__ == "__main__":
    main()
