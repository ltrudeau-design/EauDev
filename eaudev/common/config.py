"""EauDev configuration load/save."""

from pathlib import Path

import yaml

from eaudev.common.config_model import EauDevConfig
from eaudev.common.exceptions import EauDevError


def load_config(path: Path | str) -> EauDevConfig:
    """Load YAML config, creating defaults if missing."""
    config_path = Path(path).expanduser()
    if not config_path.exists():
        config = EauDevConfig()
        config_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(config, config_path)
        return config
    text = config_path.read_text()
    if not text.strip():
        config = EauDevConfig()
        save_config(config, config_path)
        return config
    raw = yaml.safe_load(text) or {}
    if "version" in raw and raw["version"] != 1:
        raise EauDevError(
            f"config.yml version {raw['version']} is not supported by this version of EauDev. "
            f"Expected version 1. Please update your config or reinstall EauDev."
        )
    config = EauDevConfig.model_validate(raw)
    # Expand tilde in paths
    config.sessions.persistence_dir = str(Path(config.sessions.persistence_dir).expanduser())
    return config


def save_config(config: EauDevConfig, path: Path | str) -> None:
    """Save config as YAML."""
    config_path = Path(path).expanduser()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config.model_dump(), sort_keys=False, allow_unicode=True)
    )
