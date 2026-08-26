#!/usr/bin/env python3
from __future__ import annotations

import argparse
from src.pingpong.config import load_config
from src.pingpong.storage import connect, load_rounds
from src.pingpong.modeling import train_bundle, save_bundle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rounds = load_rounds(connect(cfg["storage"]["sqlite_path"]))
    print(f"Loaded {len(rounds):,} completed rounds.")

    bundle = train_bundle(rounds, cfg)
    save_bundle(bundle, cfg["runtime"]["model_path"])

    print(f"Saved model: {cfg['runtime']['model_path']}")
    print("\nHeld-out results")
    print("-" * 110)
    for rep in bundle["reports"].values():
        ens = rep["ensemble_test"]
        base = rep["baseline_test"]
        print(
            f"{rep['threshold']:>6.2f}x | "
            f"AUC={ens['auc']:.4f} | "
            f"LL={ens['logloss']:.5f} vs {base['logloss']:.5f} | "
            f"Brier={ens['brier']:.5f} vs {base['brier']:.5f} | "
            f"LL gain={rep['relative_logloss_improvement']*100:>7.3f}% | "
            f"EDGE={rep['edge_detected']}"
        )

if __name__ == "__main__":
    main()
