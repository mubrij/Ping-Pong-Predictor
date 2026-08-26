#!/usr/bin/env python3
"""
Starts the non-browser services:
  1. live_trainer.py
  2. FastAPI
  3. Streamlit dashboard

Run collector.py separately because its browser may require manual login/history
navigation and should remain visible/read-only.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

commands = [
    [sys.executable, "live_trainer.py"],
    [
        sys.executable, "-m", "uvicorn", "api:app",
        "--host", "127.0.0.1", "--port", "8010",
    ],
    [
        sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
        "--server.address", "0.0.0.0",
        "--server.port", "8501",
    ],
]

processes = []

try:
    for cmd in commands:
        print("Starting:", " ".join(cmd))
        processes.append(subprocess.Popen(cmd, cwd=ROOT))
        time.sleep(1)

    print("\nStack started.")
    print("Dashboard: http://127.0.0.1:8501")
    print("API:       http://127.0.0.1:8010")
    print("Run collector.py separately in another terminal.")
    print("Press Ctrl+C to stop this stack.\n")

    while True:
        for p in processes:
            if p.poll() is not None:
                raise RuntimeError(
                    f"Child process exited with code {p.returncode}"
                )
        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
