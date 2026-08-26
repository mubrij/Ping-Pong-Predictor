from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6

class TemporalMultiplierTransformer(nn.Module):
    """
    Causal sequence classifier over log multipliers.

    Input:
        [batch, sequence_length]

    Output:
        one logit per multiplier threshold
    """

    def __init__(
        self,
        *,
        sequence_length: int,
        n_thresholds: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sequence_length = int(sequence_length)

        self.input_projection = nn.Linear(1, d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, d_model)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_thresholds),
        )

        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        h = self.input_projection(x)
        h = h + self.position_embedding[:, : h.size(1), :]
        h = self.encoder(h)

        # Last token summarizes all previous sequence positions.
        h = self.norm(h[:, -1, :])
        return self.head(h)

def _device_from_config(name: str) -> torch.device:
    name = str(name).lower().strip()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)

def _make_sequences(
    multipliers: np.ndarray,
    thresholds: list[float],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(multipliers, dtype=np.float64)
    logm = np.log(np.clip(m, 1.0, None))

    n = len(m) - sequence_length
    if n <= 0:
        raise ValueError("Not enough rows to create transformer sequences")

    X = np.empty((n, sequence_length), dtype=np.float32)
    Y = np.empty((n, len(thresholds)), dtype=np.float32)

    for i in range(n):
        target_idx = i + sequence_length
        X[i] = logm[i:target_idx].astype(np.float32)
        Y[i] = np.array(
            [m[target_idx] >= thr for thr in thresholds],
            dtype=np.float32,
        )

    return X, Y

def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=int)
    result = {
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }
    result["auc"] = (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else float("nan")
    )
    return result

def _edge_gate(
    *,
    baseline: dict[str, float],
    model: dict[str, float],
    model_cfg: dict[str, Any],
    n_test: int,
) -> dict[str, Any]:
    relative_ll = (
        baseline["logloss"] - model["logloss"]
    ) / max(baseline["logloss"], 1e-12)

    brier_gain = baseline["brier"] - model["brier"]
    auc = model["auc"]

    checks = {
        "enough_test_samples": n_test >= int(model_cfg["min_test_samples"]),
        "relative_logloss": (
            relative_ll >= float(model_cfg["min_relative_logloss_improvement"])
        ),
        "brier": (
            brier_gain >= float(model_cfg["min_brier_improvement"])
        ),
        "auc": (
            not math.isnan(auc)
            and auc >= float(model_cfg["min_auc"])
        ),
    }

    return {
        "edge_detected": bool(all(checks.values())),
        "checks": checks,
        "relative_logloss_improvement": float(relative_ll),
        "brier_improvement": float(brier_gain),
    }

def _predict_array(
    model: nn.Module,
    X: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    out = []

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(out, axis=0)

def _monotone_probabilities(
    thresholds: list[float],
    probs: np.ndarray,
) -> list[dict[str, float]]:
    order = np.argsort(np.asarray(thresholds, dtype=float))
    running = 1.0
    result = []

    for idx in order:
        thr = float(thresholds[idx])
        p = min(float(probs[idx]), running)
        running = p
        result.append({
            "threshold": thr,
            "probability": p,
        })

    return result

def _summary_from_survival(
    predictions: list[dict[str, float]],
) -> dict[str, float | None]:
    points = sorted(
        [(x["threshold"], x["probability"]) for x in predictions]
    )

    def crossing(target_survival: float):
        prev_t, prev_p = 1.0, 1.0
        for t, p in points:
            if p <= target_survival <= prev_p:
                if p == prev_p:
                    return float(t)
                frac = (target_survival - prev_p) / (p - prev_p)
                return float(prev_t + frac * (t - prev_t))
            prev_t, prev_p = t, p
        return None

    return {
        "estimated_p50_multiplier": crossing(0.50),
        "estimated_p75_multiplier": crossing(0.25),
        "estimated_p90_multiplier": crossing(0.10),
    }

def train_transformer_bundle(
    rounds: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    tc = cfg["transformer"]
    mc = cfg["model"]

    thresholds = [float(x) for x in mc["thresholds"]]
    seq_len = int(tc["sequence_length"])

    if len(rounds) < int(tc["min_rounds"]):
        raise ValueError(
            f"Transformer needs at least {tc['min_rounds']} rounds; "
            f"have {len(rounds)}"
        )

    X, Y = _make_sequences(
        rounds["multiplier"].astype(float).to_numpy(),
        thresholds,
        seq_len,
    )

    n = len(X)
    train_end = int(n * float(mc["train_fraction"]))
    val_end = int(
        n * (float(mc["train_fraction"]) + float(mc["validation_fraction"]))
    )

    if not (0 < train_end < val_end < n):
        raise ValueError("Invalid chronological split for Transformer")

    # Normalization fitted strictly on training sequences.
    train_values = X[:train_end].reshape(-1)
    mean = float(train_values.mean())
    std = float(train_values.std())
    std = max(std, 1e-6)

    Xn = ((X - mean) / std).astype(np.float32)

    X_train, Y_train = Xn[:train_end], Y[:train_end]
    X_val, Y_val = Xn[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = Xn[val_end:], Y[val_end:]

    device = _device_from_config(tc["device"])

    model_kwargs = {
        "sequence_length": seq_len,
        "n_thresholds": len(thresholds),
        "d_model": int(tc["d_model"]),
        "nhead": int(tc["nhead"]),
        "num_layers": int(tc["num_layers"]),
        "dim_feedforward": int(tc["dim_feedforward"]),
        "dropout": float(tc["dropout"]),
    }

    model = TemporalMultiplierTransformer(**model_kwargs).to(device)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
        ),
        batch_size=int(tc["batch_size"]),
        shuffle=True,
    )

    val_X_t = torch.from_numpy(X_val).to(device)
    val_Y_t = torch.from_numpy(Y_val).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(tc["learning_rate"]),
        weight_decay=float(tc["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    patience_left = int(tc["patience"])
    history = []

    for epoch in range(1, int(tc["epochs"]) + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            val_logits = model(val_X_t)
            val_loss = float(
                criterion(val_logits, val_Y_t).detach().cpu()
            )

        history.append({
            "epoch": epoch,
            "train_bce": float(np.mean(losses)),
            "validation_bce": val_loss,
        })

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            patience_left = int(tc["patience"])
        else:
            patience_left -= 1

        if patience_left <= 0:
            break

    model.load_state_dict(best_state)

    p_test = _predict_array(
        model,
        X_test,
        batch_size=int(tc["batch_size"]),
        device=device,
    )

    reports = {}
    edge_thresholds = []

    for j, thr in enumerate(thresholds):
        y_train_j = Y_train[:, j].astype(int)
        y_test_j = Y_test[:, j].astype(int)

        base_p = float(np.mean(y_train_j))
        baseline_probs = np.full(len(y_test_j), base_p, dtype=float)

        baseline = _metrics(y_test_j, baseline_probs)
        test_metrics = _metrics(y_test_j, p_test[:, j])
        gate = _edge_gate(
            baseline=baseline,
            model=test_metrics,
            model_cfg=mc,
            n_test=len(y_test_j),
        )

        if gate["edge_detected"]:
            edge_thresholds.append(float(thr))

        reports[f"{thr:.2f}"] = {
            "threshold": float(thr),
            "historical_positive_rate": float(np.mean(Y[:, j])),
            "baseline_test": baseline,
            "transformer_test": test_metrics,
            **gate,
        }

    return {
        "schema_version": 1,
        "state_dict": {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
        },
        "model_kwargs": model_kwargs,
        "thresholds": thresholds,
        "normalization": {
            "log_mean": mean,
            "log_std": std,
        },
        "training_history": history,
        "reports": reports,
        "rounds_seen": int(len(rounds)),
        "sequence_samples": int(n),
        "edge_thresholds": edge_thresholds,
        "device_used": str(device),
    }

def save_transformer_bundle(
    bundle: dict[str, Any],
    path: str | Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)

def load_transformer_bundle(
    path: str | Path,
    *,
    device_name: str = "auto",
):
    device = _device_from_config(device_name)
    bundle = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    model = TemporalMultiplierTransformer(
        **bundle["model_kwargs"]
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    return bundle, model, device

def predict_transformer_next(
    bundle: dict[str, Any],
    model: nn.Module,
    device: torch.device,
    rounds: pd.DataFrame,
) -> dict[str, Any]:
    seq_len = int(bundle["model_kwargs"]["sequence_length"])
    m = rounds["multiplier"].astype(float).to_numpy()

    if len(m) < seq_len:
        raise ValueError(
            f"Need at least {seq_len} rounds for Transformer inference"
        )

    seq = np.log(
        np.clip(m[-seq_len:], 1.0, None)
    ).astype(np.float32)

    mean = float(bundle["normalization"]["log_mean"])
    std = float(bundle["normalization"]["log_std"])
    seq = ((seq - mean) / max(std, 1e-6))[None, :]

    with torch.no_grad():
        logits = model(
            torch.from_numpy(seq).to(device)
        )
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    survival = _monotone_probabilities(
        bundle["thresholds"],
        probs,
    )

    report_by_thr = {
        float(v["threshold"]): v
        for v in bundle["reports"].values()
    }

    enriched = []
    for item in survival:
        rep = report_by_thr[item["threshold"]]
        enriched.append({
            **item,
            "edge_detected_for_threshold": bool(rep["edge_detected"]),
            "heldout_auc": rep["transformer_test"]["auc"],
            "heldout_logloss": rep["transformer_test"]["logloss"],
            "baseline_logloss": rep["baseline_test"]["logloss"],
            "relative_logloss_improvement": rep["relative_logloss_improvement"],
            "brier_improvement": rep["brier_improvement"],
        })

    return {
        "status": (
            "MODEL_HAS_HELD_OUT_SIGNAL"
            if bundle["edge_thresholds"]
            else "NO_PREDICTIVE_EDGE"
        ),
        "rounds_available": int(len(rounds)),
        "rounds_model_trained_on": int(bundle["rounds_seen"]),
        "edge_thresholds": bundle["edge_thresholds"],
        "multiplier_summary": _summary_from_survival(enriched),
        "predictions": enriched,
        "model_type": "temporal_transformer",
        "warning": (
            "Sequence-model probabilities are research estimates only. "
            "The Transformer must beat the historical-frequency baseline on "
            "chronologically later data before its outputs count as evidence."
        ),
    }
