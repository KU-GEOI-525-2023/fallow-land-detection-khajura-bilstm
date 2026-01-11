from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch

from omegaconf import DictConfig

from land_classifier.visualization.config import (
    DEFAULT_CONFIG_PATH,
    load_class_schema,
    load_config,
    require,
    resolve_path,
)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _valid_mask(arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
    if nodata is None:
        return np.ones(arr.shape, dtype=bool)
    if isinstance(nodata, float) and np.isnan(nodata):
        return ~np.isnan(arr)
    return arr != nodata


class ClassificationPlotter:
    def __init__(
        self,
        class_info: dict[int, tuple[str, str]],
        nodata: float | int | None = None,
        nodata_color: str = "#F2F2F2",
    ) -> None:
        if not class_info:
            raise ValueError("class_info must be provided.")
        self.class_info = class_info
        self.nodata = nodata
        self.nodata_color = nodata_color

    def _load_raster(self, path: Path) -> tuple[np.ndarray, float | int | None]:
        with rasterio.open(path) as src:
            arr = src.read(1)
            nodata = self.nodata if self.nodata is not None else src.nodata

        if np.issubdtype(arr.dtype, np.floating):
            arr = np.rint(arr).astype(np.int16)
        return arr, nodata

    def _class_percentages(
        self, arr: np.ndarray, nodata: float | int | None
    ) -> dict[int, float]:
        mask = _valid_mask(arr, nodata)
        total = int(mask.sum())
        if total == 0:
            return {class_id: 0.0 for class_id in self.class_info}
        return {
            class_id: int(np.count_nonzero((arr == class_id) & mask)) / total * 100.0
            for class_id in sorted(self.class_info)
        }

    def _render_rgb(self, arr: np.ndarray, nodata: float | int | None) -> np.ndarray:
        h, w = arr.shape
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[:, :] = _hex_to_rgb(self.nodata_color)

        for class_id, (_, color) in self.class_info.items():
            rgb[arr == class_id] = _hex_to_rgb(color)

        if nodata is not None:
            mask = _valid_mask(arr, nodata)
            rgb[~mask] = _hex_to_rgb(self.nodata_color)
        return rgb

    def _add_legend(
        self, ax: plt.Axes, stats: dict[int, float], title: str = "Area share"
    ) -> None:
        handles = [
            Patch(
                facecolor=color,
                edgecolor="none",
                label=f"{label} {stats.get(class_id, 0.0):.1f}%",
            )
            for class_id, (label, color) in sorted(self.class_info.items())
        ]
        ax.legend(
            handles=handles,
            title=title,
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            framealpha=0.9,
            fontsize=9,
            title_fontsize=9,
        )

    def plot_single(
        self,
        raster_path: Path,
        output_png: Path,
        title: str | None = None,
        dpi: int = 150,
    ) -> None:
        arr, nodata = self._load_raster(raster_path)
        fig, ax = plt.subplots(1, 1, figsize=(7, 7), constrained_layout=True)
        ax.imshow(self._render_rgb(arr, nodata))
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=12)
        self._add_legend(ax, self._class_percentages(arr, nodata))

        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=dpi)
        plt.close(fig)


def build_arg_parser(
    cfg: DictConfig, parents: list[argparse.ArgumentParser] | None = None
) -> argparse.ArgumentParser:
    viz_cfg = cfg.get("classification_plot", cfg.get("visualize_comparison", {}))
    class_schemas = cfg.get("class_schemas", {})
    schema_choices = sorted(class_schemas.keys()) if class_schemas else None
    default_schema = viz_cfg.get(
        "schema",
        viz_cfg.get("default_schema", schema_choices[0] if schema_choices else "project5"),
    )

    raster_value = (
        viz_cfg.get("raster_path")
        or viz_cfg.get("deep_learning_path")
        or viz_cfg.get("rule_based_path")
    )
    raster_path = resolve_path(require(raster_value, "classification_plot.raster_path"))
    output_path = resolve_path(
        require(viz_cfg.get("output_path"), "classification_plot.output_path")
    )

    parser = argparse.ArgumentParser(
        description="Render a single classification map PNG.",
        parents=parents or [],
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=raster_path,
        help="Classification GeoTIFF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=output_path,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--schema",
        choices=schema_choices,
        default=default_schema,
        help="Class schema used by rasters.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=viz_cfg.get("title", "Land classification map"),
        help="Optional figure title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=int(viz_cfg.get("dpi", 150)),
        help="PNG DPI.",
    )
    parser.add_argument(
        "--nodata",
        type=float,
        default=None,
        help="Override nodata value if rasters do not define it.",
    )
    parser.add_argument(
        "--nodata-color",
        type=str,
        default=viz_cfg.get("nodata_color", "#F2F2F2"),
        help="Hex color for nodata pixels.",
    )
    return parser


def main() -> None:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to visualization config YAML.",
    )
    base_args, remaining = base_parser.parse_known_args()
    cfg = load_config(base_args.config)

    args = build_arg_parser(cfg, parents=[base_parser]).parse_args(remaining)
    schema = load_class_schema(cfg, args.schema)
    class_info = {key: (val["label"], val["color"]) for key, val in schema.items()}
    plotter = ClassificationPlotter(
        class_info=class_info, nodata=args.nodata, nodata_color=args.nodata_color
    )
    plotter.plot_single(
        raster_path=args.raster,
        output_png=args.output,
        title=args.title,
        dpi=args.dpi,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
