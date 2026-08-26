from __future__ import annotations

from typing import Any
import math
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import mutual_info_score

try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
except Exception:
    acorr_ljungbox = None

def _runs_test_binary(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=int)
    if len(x) < 20 or len(np.unique(x)) < 2:
        return {"z": float("nan"), "p_value": float("nan")}

    n1 = int((x == 1).sum())
    n0 = int((x == 0).sum())
    runs = 1 + int(np.sum(x[1:] != x[:-1]))

    expected = 1 + (2 * n1 * n0) / (n1 + n0)
    var = (
        2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)
        / (((n1 + n0) ** 2) * (n1 + n0 - 1))
    )
    if var <= 0:
        return {"z": float("nan"), "p_value": float("nan")}

    z = (runs - expected) / math.sqrt(var)
    # Normal approximation without importing another stats object.
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"z": float(z), "p_value": float(p)}

def _discrete_mi(a: np.ndarray, b: np.ndarray, bins: int) -> float:
    edges_a = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    edges_b = np.unique(np.quantile(b, np.linspace(0, 1, bins + 1)))
    if len(edges_a) < 3 or len(edges_b) < 3:
        return 0.0

    aa = np.digitize(a, edges_a[1:-1], right=True)
    bb = np.digitize(b, edges_b[1:-1], right=True)
    return float(mutual_info_score(aa, bb))

def compute_diagnostics(
    rounds: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    if rounds.empty:
        return {"status": "NO_DATA"}

    dc = cfg["diagnostics"]
    m = rounds["multiplier"].astype(float).to_numpy()
    logm = np.log(np.clip(m, 1.0, None))
    n = len(logm)

    max_lag = min(int(dc["autocorr_lags"]), max(1, n - 2))
    acf = []
    for lag in range(1, max_lag + 1):
        a = logm[:-lag]
        b = logm[lag:]
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
        acf.append({"lag": lag, "corr": corr})

    ljung = []
    if acorr_ljungbox is not None and n > max_lag + 10:
        try:
            lags = sorted(set([1, min(5, max_lag), min(10, max_lag), max_lag]))
            lags = [x for x in lags if x >= 1]
            lb = acorr_ljungbox(logm, lags=lags, return_df=True)
            for lag, row in lb.iterrows():
                ljung.append({
                    "lag": int(lag),
                    "stat": float(row["lb_stat"]),
                    "p_value": float(row["lb_pvalue"]),
                })
        except Exception:
            pass

    binary_2x = (m >= 2.0).astype(int)
    runs = _runs_test_binary(binary_2x)

    mi = []
    bins = int(dc["mutual_information_bins"])
    for lag in range(1, min(10, max_lag) + 1):
        if n > lag + 50:
            mi.append({
                "lag": lag,
                "mutual_information": _discrete_mi(
                    logm[:-lag], logm[lag:], bins=bins
                ),
            })

    ref_w = int(dc["drift_reference_window"])
    recent_w = int(dc["drift_recent_window"])
    drift = {
        "available": False,
        "ks_stat": float("nan"),
        "p_value": float("nan"),
    }
    if n >= ref_w + recent_w:
        reference = logm[-(ref_w + recent_w):-recent_w]
        recent = logm[-recent_w:]
        ks = ks_2samp(reference, recent)
        drift = {
            "available": True,
            "ks_stat": float(ks.statistic),
            "p_value": float(ks.pvalue),
            "drift_flag": bool(ks.pvalue < 0.01),
        }

    quantiles = {
        "p10": float(np.quantile(m, 0.10)),
        "p25": float(np.quantile(m, 0.25)),
        "p50": float(np.quantile(m, 0.50)),
        "p75": float(np.quantile(m, 0.75)),
        "p90": float(np.quantile(m, 0.90)),
        "p95": float(np.quantile(m, 0.95)),
        "p99": float(np.quantile(m, 0.99)),
    }

    threshold_rates = {}
    for thr in (1.2, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0):
        threshold_rates[f"{thr:.2f}"] = float(np.mean(m >= thr))

    return {
        "status": "OK",
        "rounds": int(n),
        "mean_multiplier": float(np.mean(m)),
        "median_multiplier": float(np.median(m)),
        "max_multiplier": float(np.max(m)),
        "quantiles": quantiles,
        "threshold_rates": threshold_rates,
        "autocorrelation": acf,
        "ljung_box": ljung,
        "runs_test_ge_2x": runs,
        "mutual_information": mi,
        "drift": drift,
    }
