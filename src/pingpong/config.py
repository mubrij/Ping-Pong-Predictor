from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    required = [
        "sportybet", "collector", "storage", "features",
        "model", "diagnostics", "runtime", "api", "dashboard"
    ]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    return cfg
