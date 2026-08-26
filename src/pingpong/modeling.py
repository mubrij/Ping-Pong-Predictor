from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import (
    make_training_frame,
    make_next_feature_row,
    threshold_name,
)

EPS = 1e-6

def _clip(p):
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)

def metric_bundle(y, p):
    p = _clip(p)
    out = {
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }
    out["auc"] = (
        float(roc_auc_score(y, p))
        if len(np.unique(y)) == 2
        else float("nan")
    )
    return out

def _constant_probs(y_train, n):
    p = float(np.mean(y_train))
    return np.full(n, np.clip(p, EPS, 1 - EPS), dtype=float)

def _base_candidates(cfg: dict[str, Any]):
    enabled = cfg["model"]["candidates"]
    cal_splits = int(cfg["model"]["calibration_splits"])
    candidates = {}

    if enabled.get("logistic", True):
        candidates["logistic"] = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=4000,
                C=0.5,
                solver="lbfgs",
            )),
        ])

    if enabled.get("hist_gradient_boosting", True):
        hgb = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_depth=4,
            max_iter=250,
            min_samples_leaf=30,
            l2_regularization=1.5,
            random_state=42,
        )
        candidates["hist_gradient_boosting"] = CalibratedClassifierCV(
            estimator=hgb,
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=cal_splits),
        )

    if enabled.get("extra_trees", True):
        et = ExtraTreesClassifier(
            n_estimators=350,
            max_depth=10,
            min_samples_leaf=12,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        )
        candidates["extra_trees"] = CalibratedClassifierCV(
            estimator=et,
            method="sigmoid",
            cv=TimeSeriesSplit(n_splits=cal_splits),
        )

    if not candidates:
        raise ValueError("No models enabled in config.yaml")
    return candidates

def _split(n, train_fraction, validation_fraction):
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + validation_fraction))
    if not (0 < train_end < val_end < n):
        raise ValueError("Invalid chronological split")
    return train_end, val_end

def _ensemble_weights(
    val_scores: dict[str, dict[str, float]],
    temperature: float,
) -> dict[str, float]:
    names = list(val_scores)
    losses = np.array([val_scores[n]["logloss"] for n in names], dtype=float)
    centered = losses - np.min(losses)
    temp = max(float(temperature), 1e-6)
    raw = np.exp(-centered / temp)
    raw /= raw.sum()
    return {name: float(w) for name, w in zip(names, raw)}

def _weighted_prediction(
    models: dict[str, Any],
    weights: dict[str, float],
    X: pd.DataFrame,
) -> np.ndarray:
    p = np.zeros(len(X), dtype=float)
    for name, model in models.items():
        p += float(weights[name]) * model.predict_proba(X)[:, 1]
    return _clip(p)

def _edge_gate(
    *,
    baseline: dict[str, float],
    model: dict[str, float],
    cfg: dict[str, Any],
    n_test: int,
) -> dict[str, Any]:
    mc = cfg["model"]
    rel_ll = (
        baseline["logloss"] - model["logloss"]
    ) / max(baseline["logloss"], 1e-12)
    brier_improvement = baseline["brier"] - model["brier"]
    auc = model["auc"]

    checks = {
        "enough_test_samples": n_test >= int(mc["min_test_samples"]),
        "relative_logloss": rel_ll >= float(mc["min_relative_logloss_improvement"]),
        "brier": brier_improvement >= float(mc["min_brier_improvement"]),
        "auc": (not math.isnan(auc)) and auc >= float(mc["min_auc"]),
    }
    return {
        "edge_detected": bool(all(checks.values())),
        "checks": checks,
        "relative_logloss_improvement": float(rel_ll),
        "brier_improvement": float(brier_improvement),
    }

def _monotone_survival(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Enforce P(M>=higher threshold) <= P(M>=lower threshold)
    by a simple cumulative minimum over ascending thresholds.
    """
    predictions = sorted(predictions, key=lambda x: x["threshold"])
    running = 1.0
    fixed = []
    for item in predictions:
        p = min(float(item["probability"]), running)
        running = p
        out = dict(item)
        out["probability"] = p
        fixed.append(out)
    return fixed

def _estimate_multiplier_summary(predictions: list[dict[str, Any]]) -> dict[str, float | None]:
    """
    Convert threshold-survival probabilities into coarse distribution summaries.
    This is intentionally approximate because the model predicts threshold events,
    not an exact crash point.
    """
    pts = sorted(
        [(float(x["threshold"]), float(x["probability"])) for x in predictions],
        key=lambda z: z[0]
    )

    def crossing(target_survival: float):
        # For quantile q, survival target is 1-q.
        prev_t, prev_p = 1.0, 1.0
        for t, p in pts:
            if p <= target_survival <= prev_p:
                if prev_p == p:
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

def train_bundle(rounds: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    fc = cfg["features"]
    mc = cfg["model"]
    thresholds = [float(x) for x in mc["thresholds"]]

    frame, feature_cols = make_training_frame(
        rounds,
        max_lag=int(fc["max_lag"]),
        rolling_windows=[int(x) for x in fc["rolling_windows"]],
        thresholds=thresholds,
        distribution_thresholds=[float(x) for x in fc["distribution_thresholds"]],
    )

    if len(frame) < int(mc["min_rounds"]):
        raise ValueError(
            f"Need at least {mc['min_rounds']} usable training rows; have {len(frame)}"
        )

    train_end, val_end = _split(
        len(frame),
        float(mc["train_fraction"]),
        float(mc["validation_fraction"]),
    )

    X = frame[feature_cols]
    result = {
        "schema_version": 2,
        "feature_cols": feature_cols,
        "thresholds": thresholds,
        "rounds_seen": int(len(rounds)),
        "training_rows": int(len(frame)),
        "models": {},
        "weights": {},
        "reports": {},
        "feature_config": fc,
        "model_config": mc,
    }

    for thr in thresholds:
        key = threshold_name(thr)
        y = frame[f"y_{key}"].astype(int).to_numpy()

        X_train, y_train = X.iloc[:train_end], y[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y[val_end:]

        if len(np.unique(y_train)) < 2:
            raise ValueError(f"{thr:.2f}x target has one training class only")

        baseline_val = metric_bundle(
            y_val, _constant_probs(y_train, len(y_val))
        )
        baseline_test = metric_bundle(
            y_test, _constant_probs(y_train, len(y_test))
        )

        val_scores = {}
        validation_models = {}
        for name, model in _base_candidates(cfg).items():
            model.fit(X_train, y_train)
            pv = model.predict_proba(X_val)[:, 1]
            val_scores[name] = metric_bundle(y_val, pv)
            validation_models[name] = model

        weights = _ensemble_weights(
            val_scores,
            temperature=float(mc["ensemble_temperature"]),
        )

        # Refit each component on train+validation.
        fitted = {}
        for name, model in _base_candidates(cfg).items():
            model.fit(X.iloc[:val_end], y[:val_end])
            fitted[name] = model

        p_test = _weighted_prediction(fitted, weights, X_test)
        test_metrics = metric_bundle(y_test, p_test)
        gate = _edge_gate(
            baseline=baseline_test,
            model=test_metrics,
            cfg=cfg,
            n_test=len(y_test),
        )

        # Final models on all currently available historical rows.
        final_models = {}
        for name, model in _base_candidates(cfg).items():
            model.fit(X, y)
            final_models[name] = model

        result["models"][key] = final_models
        result["weights"][key] = weights
        result["reports"][key] = {
            "threshold": float(thr),
            "historical_positive_rate": float(y.mean()),
            "baseline_validation": baseline_val,
            "candidate_validation": val_scores,
            "ensemble_weights": weights,
            "baseline_test": baseline_test,
            "ensemble_test": test_metrics,
            **gate,
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "n_test": int(len(y_test)),
        }

    return result

def save_bundle(bundle: dict[str, Any], path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    public = {k: v for k, v in bundle.items() if k != "models"}
    with path.with_suffix(".metrics.json").open("w", encoding="utf-8") as f:
        json.dump(public, f, indent=2, allow_nan=True)

def load_bundle(path: str | Path):
    return joblib.load(path)

def predict_next(
    bundle: dict[str, Any],
    rounds: pd.DataFrame,
) -> dict[str, Any]:
    fc = bundle["feature_config"]
    Xnext = make_next_feature_row(
        rounds,
        max_lag=int(fc["max_lag"]),
        rolling_windows=[int(x) for x in fc["rolling_windows"]],
        distribution_thresholds=[float(x) for x in fc["distribution_thresholds"]],
    )
    Xnext = Xnext[bundle["feature_cols"]]

    outputs = []
    any_edge = False

    for thr in bundle["thresholds"]:
        key = threshold_name(thr)
        models = bundle["models"][key]
        weights = bundle["weights"][key]

        p = 0.0
        component_probs = {}
        for name, model in models.items():
            cp = float(model.predict_proba(Xnext)[:, 1][0])
            component_probs[name] = cp
            p += weights[name] * cp

        rep = bundle["reports"][key]
        edge = bool(rep["edge_detected"])
        any_edge = any_edge or edge

        outputs.append({
            "threshold": float(thr),
            "probability": float(np.clip(p, EPS, 1 - EPS)),
            "edge_detected_for_threshold": edge,
            "component_probabilities": component_probs,
            "ensemble_weights": weights,
            "heldout_auc": rep["ensemble_test"]["auc"],
            "heldout_logloss": rep["ensemble_test"]["logloss"],
            "baseline_logloss": rep["baseline_test"]["logloss"],
            "relative_logloss_improvement": rep["relative_logloss_improvement"],
            "brier_improvement": rep["brier_improvement"],
        })

    outputs = _monotone_survival(outputs)
    summary = _estimate_multiplier_summary(outputs)

    edge_thresholds = [
        x["threshold"] for x in outputs if x["edge_detected_for_threshold"]
    ]

    return {
        "status": (
            "MODEL_HAS_HELD_OUT_SIGNAL"
            if any_edge else
            "NO_PREDICTIVE_EDGE"
        ),
        "rounds_available": int(len(rounds)),
        "rounds_model_trained_on": int(bundle["rounds_seen"]),
        "edge_thresholds": edge_thresholds,
        "multiplier_summary": summary,
        "predictions": outputs,
        "warning": (
            "Research probability estimates only. The published Ping Pong rules "
            "state that the final multiplier is generated before each round by "
            "a provably-fair random-number generator. A model that fails the "
            "held-out gates should be treated as having no predictive signal."
        ),
    }
