from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_params(path: str | Path = "params.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
