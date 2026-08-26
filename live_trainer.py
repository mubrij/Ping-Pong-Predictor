#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.pingpong.config import load_config
from src.pingpong.storage import connect, load_rounds, count_rounds
from src.pingpong.modeling import train_bundle, save_bundle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    con = connect(cfg["storage"]["sqlite_path"])
    model_path = Path(cfg["runtime"]["model_path"])

    min_rounds = int(cfg["model"]["min_rounds"])
    retrain_every = int(cfg["runtime"]["retrain_every_new_rounds"])
    poll = float(cfg["runtime"]["prediction_poll_seconds"])

    last_trained_count = 0

    while True:
        n = count_rounds(con)

        should_train = (
            n >= min_rounds and (
                not model_path.exists()
                or n - last_trained_count >= retrain_every
            )
        )

        if should_train:
            rounds = load_rounds(con)
            print(f"Training on {len(rounds):,} completed rounds...")
            try:
                bundle = train_bundle(rounds, cfg)
                save_bundle(bundle, model_path)
                last_trained_count = n
                edges = [
                    r["threshold"]
                    for r in bundle["reports"].values()
                    if r["edge_detected"]
                ]
                print("Training complete. Edge thresholds:", edges or "NONE")
            except Exception as e:
                print("Training error:", e)

        time.sleep(poll)

if __name__ == "__main__":
    main()
