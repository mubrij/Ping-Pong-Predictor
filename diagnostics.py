#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from src.pingpong.config import load_config
from src.pingpong.storage import connect, load_rounds
from src.pingpong.diagnostics import compute_diagnostics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rounds = load_rounds(connect(cfg["storage"]["sqlite_path"]))
    print(json.dumps(compute_diagnostics(rounds, cfg), indent=2, allow_nan=True))

if __name__ == "__main__":
    main()
