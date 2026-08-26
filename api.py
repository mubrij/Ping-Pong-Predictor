#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import math
from pathlib import Path
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from src.pingpong.config import load_config
from src.pingpong.storage import connect, load_rounds, count_rounds, latest_round
from src.pingpong.modeling import load_bundle, predict_next
from src.pingpong.diagnostics import compute_diagnostics
from src.pingpong.transformer_model import load_transformer_bundle, predict_transformer_next

CONFIG_PATH = "config.yaml"


app = FastAPI(
    title="Ping Pong Advanced Research Predictor",
    version="0.2.0",
)

def _json_safe(value):
    """Convert NaN/Inf and numpy-like scalars to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    # bool must be checked before int because bool subclasses int.
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None

    # numpy / torch / pandas scalar types often expose .item().
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except Exception:
            pass

    return value

def _cfg():
    return load_config(CONFIG_PATH)

@app.get("/health")
def health():
    cfg = _cfg()
    con = connect(cfg["storage"]["sqlite_path"])
    model_path = Path(cfg["runtime"]["model_path"])
    return {
        "ok": True,
        "rounds": count_rounds(con),
        "latest_round": latest_round(con),
        "model_exists": model_path.exists(),
        "transformer_exists": Path(cfg["transformer"]["model_path"]).exists(),
    }

@app.get("/history")
def history(limit: int = 300):
    cfg = _cfg()
    con = connect(cfg["storage"]["sqlite_path"])
    limit = min(max(int(limit), 1), 5000)
    df = load_rounds(con, limit=limit)
    return {
        "rows": df.to_dict(orient="records"),
        "count": len(df),
    }

@app.get("/prediction")
def prediction():
    cfg = _cfg()
    model_path = Path(cfg["runtime"]["model_path"])
    if not model_path.exists():
        raise HTTPException(503, "Model has not been trained yet.")

    con = connect(cfg["storage"]["sqlite_path"])
    rounds = load_rounds(con)
    bundle = load_bundle(model_path)

    try:
        return _json_safe(predict_next(bundle, rounds))
    except Exception as e:
        raise HTTPException(503, str(e))

@app.get("/diagnostics")
def diagnostics():
    cfg = _cfg()
    con = connect(cfg["storage"]["sqlite_path"])
    rounds = load_rounds(con)
    return _json_safe(compute_diagnostics(rounds, cfg))

@app.get("/model/metrics")
def model_metrics():
    cfg = _cfg()
    model_path = Path(cfg["runtime"]["model_path"])
    if not model_path.exists():
        raise HTTPException(503, "Model has not been trained yet.")

    bundle = load_bundle(model_path)
    return _json_safe({
        "rounds_seen": bundle["rounds_seen"],
        "training_rows": bundle["training_rows"],
        "thresholds": bundle["thresholds"],
        "reports": bundle["reports"],
    })

@app.websocket("/ws")
async def websocket_updates(ws: WebSocket):
    await ws.accept()
    last_round_count = -1

    try:
        while True:
            cfg = _cfg()
            con = connect(cfg["storage"]["sqlite_path"])
            n = count_rounds(con)

            if n != last_round_count:
                payload = {
                    "round_count": n,
                    "latest_round": latest_round(con),
                    "prediction": None,
                }

                model_path = Path(cfg["runtime"]["model_path"])
                if model_path.exists():
                    try:
                        rounds = load_rounds(con)
                        bundle = load_bundle(model_path)
                        payload["prediction"] = predict_next(bundle, rounds)
                    except Exception as e:
                        payload["prediction_error"] = str(e)

                await ws.send_json(_json_safe(payload))
                last_round_count = n

            await asyncio.sleep(float(cfg["runtime"]["prediction_poll_seconds"]))

    except WebSocketDisconnect:
        return


@app.get("/transformer/prediction")
def transformer_prediction():
    cfg = _cfg()
    model_path = Path(cfg["transformer"]["model_path"])
    if not model_path.exists():
        raise HTTPException(503, "Transformer model has not been trained yet.")

    con = connect(cfg["storage"]["sqlite_path"])
    rounds = load_rounds(con)
    bundle, model, device = load_transformer_bundle(
        model_path,
        device_name=cfg["transformer"]["device"],
    )
    return _json_safe(predict_transformer_next(
        bundle,
        model,
        device,
        rounds,
    ))

@app.get("/transformer/metrics")
def transformer_metrics():
    cfg = _cfg()
    model_path = Path(cfg["transformer"]["model_path"])
    if not model_path.exists():
        raise HTTPException(503, "Transformer model has not been trained yet.")

    bundle, _, _ = load_transformer_bundle(
        model_path,
        device_name=cfg["transformer"]["device"],
    )
    return _json_safe({
        "rounds_seen": bundle["rounds_seen"],
        "sequence_samples": bundle["sequence_samples"],
        "edge_thresholds": bundle["edge_thresholds"],
        "reports": bundle["reports"],
        "training_history": bundle["training_history"],
        "device_used": bundle["device_used"],
    })
