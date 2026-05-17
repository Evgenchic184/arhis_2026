from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_params(path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "params.yaml"
    with open(resolved_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
