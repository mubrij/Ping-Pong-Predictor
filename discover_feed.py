#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path
from typing import Any
from playwright.async_api import async_playwright
from src.pingpong.config import load_config

MULT_KEYS = {
    "multiplier", "coefficient", "coef", "crash", "crashpoint",
    "crash_point", "finalmultiplier", "final_multiplier",
    "exitcoefficient", "exit_coefficient", "result",
}
X_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[xX×]\s*$")

def walk(obj: Any, path: str = "$"):
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if str(k).lower() in MULT_KEYS:
                yield p, v
            yield from walk(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:150]):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(obj, str) and X_RE.match(obj):
        yield path, obj

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seconds", type=int, default=300)
    args = ap.parse_args()

    cfg = load_config(args.config)
    profile = Path(cfg["collector"]["user_data_dir"])
    profile.mkdir(parents=True, exist_ok=True)
    seen = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=str(profile),
            headless=False,
        )
        page = browser.pages[0] if browser.pages else await browser.new_page()

        async def on_response(resp):
            if "json" not in (resp.headers.get("content-type") or "").lower():
                return
            try:
                payload = await resp.json()
            except Exception:
                return

            hits = list(walk(payload))
            if not hits:
                return

            signature = (resp.url, tuple(path for path, _ in hits[:15]))
            if signature in seen:
                return
            seen.add(signature)

            print("\n" + "=" * 100)
            print("CANDIDATE RESPONSE:", resp.url)
            for path, value in hits[:40]:
                print(f"  {path}: {repr(value)[:140]}")

        page.on("response", on_response)

        print("Opening Ping Pong in a normal Chromium browser.")
        print(
            "Authenticate manually if required and open the visible history/fairness "
            "panel. The script only observes responses and does not place bets."
        )
        await page.goto(cfg["sportybet"]["url"], wait_until="domcontentloaded")
        await asyncio.sleep(args.seconds)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
