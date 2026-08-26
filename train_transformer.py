#!/usr/bin/env python3
from __future__ import annotations

import argparse

from src.pingpong.config import load_config
from src.pingpong.storage import connect, load_rounds
from src.pingpong.transformer_model import (
    train_transformer_bundle,
    save_transformer_bundle,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rounds = load_rounds(connect(cfg["storage"]["sqlite_path"]))

    print(f"Loaded {len(rounds):,} completed rounds.")
    bundle = train_transformer_bundle(rounds, cfg)
    save_transformer_bundle(
        bundle,
        cfg["transformer"]["model_path"],
    )

    print("Saved Transformer:", cfg["transformer"]["model_path"])
    print("Device used:", bundle["device_used"])
    print("Edge thresholds:", bundle["edge_thresholds"] or "NONE")

    for rep in bundle["reports"].values():
        m = rep["transformer_test"]
        b = rep["baseline_test"]
        print(
            f"{rep['threshold']:>6.2f}x | "
            f"AUC={m['auc']:.4f} | "
            f"LL={m['logloss']:.5f} vs {b['logloss']:.5f} | "
            f"gain={rep['relative_logloss_improvement']*100:.3f}% | "
            f"EDGE={rep['edge_detected']}"
        )

if __name__ == "__main__":
    main()
