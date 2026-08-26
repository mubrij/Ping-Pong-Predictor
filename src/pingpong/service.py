from __future__ import annotations

from pathlib import Path
from typing import Any
from .config import load_config
from .storage import connect, load_rounds, count_rounds, latest_round
from .diagnostics import compute_diagnostics
from .modeling import load_bundle, predict_next

def get_state(config_path: str = "config.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    con = connect(cfg["storage"]["sqlite_path"])
    model_path = Path(cfg["runtime"]["model_path"])

    result: dict[str, Any] = {
        "round_count": count_rounds(con),
        "latest_round": latest_round(con),
        "model_exists": model_path.exists(),
    }

    if model_path.exists():
        rounds = load_rounds(con)
        bundle = load_bundle(model_path)
        result["prediction"] = predict_next(bundle, rounds)
    else:
        result["prediction"] = None

    return result

def get_diagnostics(config_path: str = "config.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    con = connect(cfg["storage"]["sqlite_path"])
    rounds = load_rounds(con)
    return compute_diagnostics(rounds, cfg)
