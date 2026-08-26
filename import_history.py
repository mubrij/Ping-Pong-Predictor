#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import pandas as pd

from src.pingpong.config import load_config
from src.pingpong.storage import connect, insert_round

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--multiplier-col", default="multiplier")
    ap.add_argument("--round-id-col", default="round_id")
    ap.add_argument("--timestamp-col", default="timestamp")
    args = ap.parse_args()

    cfg = load_config(args.config)
    con = connect(cfg["storage"]["sqlite_path"])
    df = pd.read_csv(args.csv)

    added = 0
    for i, row in df.iterrows():
        rid = row.get(args.round_id_col)
        ts = row.get(args.timestamp_col)
        if pd.isna(rid):
            rid = f"csv-{i}"
        if pd.isna(ts):
            ts = datetime.now(timezone.utc).isoformat()

        added += int(insert_round(
            con,
            round_id=str(rid),
            observed_at=str(ts),
            multiplier=float(row[args.multiplier_col]),
            source="csv",
        ))

    print(f"Imported {added} new round(s).")

if __name__ == "__main__":
    main()
