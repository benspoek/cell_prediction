"""Configuration and reproducibility utilities without model imports."""
from __future__ import annotations

import hashlib
import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected a schema_version: 1 benchmark mapping")
    for section in ("paths", "data", "evaluation", "models", "embeddings", "cluster"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    return config


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path if path.is_absolute() else root / path


def compute_config(config: dict) -> dict:
    """Exclude the changing allocation counter from reproducibility decisions."""
    result = copy.deepcopy(config)
    result.get("cluster", {}).pop("remaining_gpu_hours", None)
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)
