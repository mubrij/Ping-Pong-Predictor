from __future__ import annotations

import numpy as np
import pandas as pd

def threshold_name(thr: float) -> str:
    return f"{float(thr):.2f}".replace(".", "_")

def _safe_log(s: pd.Series) -> pd.Series:
    return np.log(np.clip(s.astype(float), 1.0, None))

def make_training_frame(
    rounds: pd.DataFrame,
    *,
    max_lag: int,
    rolling_windows: list[int],
    thresholds: list[float],
    distribution_thresholds: list[float],
) -> tuple[pd.DataFrame, list[str]]:
    if "multiplier" not in rounds.columns:
        raise ValueError("rounds must contain multiplier")

    m = rounds["multiplier"].astype(float).reset_index(drop=True)
    logm = _safe_log(m)
    past_m = m.shift(1)
    past_log = logm.shift(1)

    features: dict[str, pd.Series | np.ndarray] = {}

    # Exact lags available before target round t.
    for lag in range(1, max_lag + 1):
        features[f"log_lag_{lag}"] = logm.shift(lag)

    # Rolling shape and exceedance rates, using past rounds only.
    for w in rolling_windows:
        rlog = past_log.rolling(w)
        features[f"log_mean_{w}"] = rlog.mean()
        features[f"log_std_{w}"] = rlog.std()
        features[f"log_min_{w}"] = rlog.min()
        features[f"log_max_{w}"] = rlog.max()
        features[f"log_q25_{w}"] = rlog.quantile(0.25)
        features[f"log_q50_{w}"] = rlog.quantile(0.50)
        features[f"log_q75_{w}"] = rlog.quantile(0.75)

        for thr in distribution_thresholds:
            nm = threshold_name(thr)
            features[f"frac_ge_{nm}_{w}"] = (
                (past_m >= thr).astype(float).rolling(w).mean()
            )

    # Streak ending at t-1.
    for thr in (1.20, 1.50, 2.00, 3.00):
        flags = (past_m < thr).fillna(False).to_numpy()
        streak = np.zeros(len(flags), dtype=float)
        run = 0
        for i, flag in enumerate(flags):
            run = run + 1 if bool(flag) else 0
            streak[i] = run
        features[f"streak_below_{threshold_name(thr)}"] = streak

    # Number of rounds since the latest prior exceedance. This is calculated
    # BEFORE observing multiplier[t], so no extra shift is needed.
    for thr in (2.0, 3.0, 5.0, 10.0):
        flags = (m >= thr).to_numpy()
        since = np.full(len(flags), np.nan, dtype=float)
        last_hit = None

        for i in range(len(flags)):
            if last_hit is None:
                since[i] = float(i)
            else:
                since[i] = float(i - last_hit)

            if flags[i]:
                last_hit = i

        # At row i, the value above still counts i itself if i is a hit.
        # Recompute strictly from the prior prefix for exact alignment.
        strict_since = np.full(len(flags), np.nan, dtype=float)
        last_prior = None
        for i in range(len(flags)):
            if last_prior is None:
                strict_since[i] = float(i)
            else:
                strict_since[i] = float(i - last_prior)
            if flags[i]:
                last_prior = i

        features[f"rounds_since_ge_{threshold_name(thr)}"] = strict_since

    labels = {
        f"y_{threshold_name(thr)}": (m >= thr).astype(int)
        for thr in thresholds
    }

    out = pd.concat(
        [
            pd.DataFrame({"multiplier": m}),
            pd.DataFrame(features),
            pd.DataFrame(labels),
        ],
        axis=1,
    )

    label_cols = list(labels)
    feature_cols = [
        c for c in out.columns
        if c not in {"multiplier", *label_cols}
    ]

    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    return out, feature_cols

def make_next_feature_row(
    rounds: pd.DataFrame,
    *,
    max_lag: int,
    rolling_windows: list[int],
    distribution_thresholds: list[float],
) -> pd.DataFrame:
    m = rounds["multiplier"].astype(float).reset_index(drop=True)
    need = max(max_lag, max(rolling_windows))
    if len(m) < need:
        raise ValueError(f"Need at least {need} rounds; have {len(m)}")

    logm = _safe_log(m)
    row: dict[str, float] = {}

    for lag in range(1, max_lag + 1):
        row[f"log_lag_{lag}"] = float(logm.iloc[-lag])

    for w in rolling_windows:
        tail_log = logm.iloc[-w:]
        tail_m = m.iloc[-w:]
        row[f"log_mean_{w}"] = float(tail_log.mean())
        row[f"log_std_{w}"] = float(tail_log.std())
        row[f"log_min_{w}"] = float(tail_log.min())
        row[f"log_max_{w}"] = float(tail_log.max())
        row[f"log_q25_{w}"] = float(tail_log.quantile(0.25))
        row[f"log_q50_{w}"] = float(tail_log.quantile(0.50))
        row[f"log_q75_{w}"] = float(tail_log.quantile(0.75))

        for thr in distribution_thresholds:
            nm = threshold_name(thr)
            row[f"frac_ge_{nm}_{w}"] = float((tail_m >= thr).mean())

    for thr in (1.20, 1.50, 2.00, 3.00):
        run = 0
        for x in reversed(m.tolist()):
            if x < thr:
                run += 1
            else:
                break
        row[f"streak_below_{threshold_name(thr)}"] = float(run)

    for thr in (2.0, 3.0, 5.0, 10.0):
        distance = len(m)
        for i, x in enumerate(reversed(m.tolist()), start=1):
            if x >= thr:
                distance = i
                break
        row[f"rounds_since_ge_{threshold_name(thr)}"] = float(distance)

    return pd.DataFrame([row])
