#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import numpy as np

from src.pingpong.config import load_config
from src.pingpong.storage import connect, insert_round

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config)
    con = connect(cfg["storage"]["sqlite_path"])
    rng = np.random.default_rng(args.seed)

    start = datetime.now(timezone.utc) - timedelta(seconds=args.n * 10)

    # Independent heavy-tailed synthetic multipliers.
    # This tests software behavior only; it does not model SportyBet's RNG.
    u = np.clip(rng.random(args.n), 1e-8, 1 - 1e-8)
    multipliers = np.maximum(
        1.0,
        np.floor((1.0 / u) * 100.0) / 100.0,
    )

    added = 0
    for i, mult in enumerate(multipliers):
        added += int(insert_round(
            con,
            round_id=f"synthetic-{args.seed}-{i}",
            observed_at=(start + timedelta(seconds=i * 10)).isoformat(),
            multiplier=float(mult),
            source="synthetic",
        ))

    print(f"Added {added} synthetic rounds.")

if __name__ == "__main__":
    main()
