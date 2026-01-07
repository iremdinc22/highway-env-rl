from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import yaml


def load_env_config(env_id: str, config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        all_cfg = yaml.safe_load(f)

    default_cfg = all_cfg.get("default", {})
    env_cfg = all_cfg.get(env_id, {})

    merged = default_cfg.copy()
    merged.update(env_cfg)
    return merged
