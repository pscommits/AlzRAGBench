"""Config loading with `inherits:` support.

Every tunable lives in configs/*.yaml. Source code reads config; it never
hardcodes a path, model id, or threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config(dict):
    """dict with attribute access, so cfg.models.embedding.hf_id works."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(f"No config key '{key}'") from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Override wins on scalars; dicts merge recursively."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Config:
    """Load a YAML config, resolving a single `inherits:` parent chain."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f) or {}

    parent_path = raw.pop("inherits", None)
    if parent_path:
        parent = load_config(parent_path)
        raw = _deep_merge(parent, raw)

    return Config(raw)


def save_config(cfg: Config, path: str | Path) -> None:
    """Write the *resolved* config next to results, so any metrics file
    always records exactly which settings produced it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(dict(cfg), f, default_flow_style=False, sort_keys=False)
