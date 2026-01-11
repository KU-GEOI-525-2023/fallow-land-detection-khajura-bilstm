from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "configs/visualization/defaults.yaml"


def require(value: object, name: str) -> object:
    if value is None:
        raise ValueError(f"Missing required config value: {name}")
    return value


def load_config(config_path: Path | None = None) -> DictConfig:
    resolved_path = config_path or DEFAULT_CONFIG_PATH
    if not resolved_path.exists():
        raise FileNotFoundError(f"Visualization config not found: {resolved_path}")
    cfg = OmegaConf.load(resolved_path)
    OmegaConf.resolve(cfg)
    return cfg


def resolve_path(value: str | Path, root: Path | None = None) -> Path:
    root = root or ROOT
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_class_schema(
    cfg: DictConfig, name: str
) -> dict[int, dict[str, str]]:
    schemas = cfg.get("class_schemas", {})
    schema = schemas.get(name)
    if not schema:
        raise KeyError(f"Class schema not found: {name}")
    return {
        int(class_id): {"label": entry["label"], "color": entry["color"]}
        for class_id, entry in schema.items()
    }
