#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from playwright.async_api import async_playwright

from src.pingpong.config import load_config
from src.pingpong.storage import connect, insert_round

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def get_path(obj: Any, path: str) -> Any:
    if not path:
        return obj
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            raise KeyError(path)
        cur = cur[part]
    return cur

def field(item: Any, name: str):
    if not name or not isinstance(item, dict):
        return None
    return item.get(name)

def parse_websocket_json(payload: Any) -> Any | None:
    """Extract JSON from a plain or STOMP-framed WebSocket message."""
    if not isinstance(payload, str):
        return None
    start, end = payload.find("{"), payload.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(payload[start:end + 1])
    except json.JSONDecodeError:
        return None

def parse_multiplier(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        x = float(v)
        return x if x >= 1.0 else None

    s = str(v).strip().lower().replace("×", "x")
    if s.endswith("x"):
        s = s[:-1].strip()
    try:
        x = float(s)
        return x if x >= 1.0 else None
    except ValueError:
        return None

async def attach_network_listener(page, cfg, con):
    c = cfg["collector"]
    needle = c["feed_url_contains"].strip()
    if not needle:
        raise ValueError(
            "collector.feed_url_contains is empty. Run discover_feed.py first."
        )

    async def on_response(resp):
        if needle not in resp.url:
            return
        if "json" not in (resp.headers.get("content-type") or "").lower():
            return

        try:
            payload = await resp.json()
            items = get_path(payload, c["json_items_path"].strip())
        except Exception as e:
            print("Feed parse error:", e)
            return

        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            print("json_items_path did not resolve to a list/dict.")
            return

        added = 0
        for item in items:
            mult = parse_multiplier(field(item, c["multiplier_field"].strip()))
            if mult is None:
                continue

            rid = field(item, c["round_id_field"].strip())
            ts = field(item, c["timestamp_field"].strip()) or now_iso()

            added += int(insert_round(
                con,
                round_id=str(rid) if rid is not None else None,
                observed_at=str(ts),
                multiplier=mult,
                source=f"network:{resp.url}",
                raw_json=json.dumps(item, ensure_ascii=False)[:20000],
            ))

        if added:
            print(f"Captured {added} new completed round(s).")

    page.on("response", on_response)

async def attach_websocket_listener(page, cfg, con):
    c = cfg["collector"]
    needle = c.get("websocket_url_contains", "").strip()
    expected = c.get("websocket_message_type", "ROUND_COEFFICIENT").strip()
    if not needle:
        raise ValueError("collector.websocket_url_contains is empty")

    def on_websocket(ws):
        if needle not in ws.url:
            return
        print(f"Observing completed-round WebSocket: {ws.url}")

        def on_frame(payload):
            message = parse_websocket_json(payload)
            if not isinstance(message, dict) or message.get("messageType") != expected:
                return
            try:
                item = get_path(message, c.get("json_items_path", "").strip())
            except KeyError:
                return
            mult = parse_multiplier(field(item, c["multiplier_field"].strip()))
            if mult is None:
                return
            rid = field(item, c["round_id_field"].strip())
            ts = field(item, c["timestamp_field"].strip()) or now_iso()
            if insert_round(
                con, round_id=str(rid) if rid is not None else None,
                observed_at=str(ts), multiplier=mult,
                source=f"websocket:{ws.url}",
                raw_json=json.dumps(message, ensure_ascii=False)[:20000],
            ):
                print(f"Captured completed round {rid}: {mult:.2f}x")

        ws.on("framereceived", on_frame)

    page.on("websocket", on_websocket)

async def dom_loop(page, cfg, con):
    c = cfg["collector"]
    item_selector = c["dom_history_item_selector"].strip()
    multiplier_selector = c["dom_multiplier_selector"].strip()
    rid_attr = c["dom_round_id_attribute"].strip()

    if not item_selector or not multiplier_selector:
        raise ValueError("DOM mode requires configured history selectors.")

    while True:
        try:
            items = page.locator(item_selector)
            added = 0
            for i in range(await items.count()):
                item = items.nth(i)
                raw = await item.locator(multiplier_selector).inner_text()
                mult = parse_multiplier(raw)
                if mult is None:
                    continue
                rid = await item.get_attribute(rid_attr) if rid_attr else None
                added += int(insert_round(
                    con,
                    round_id=rid,
                    observed_at=now_iso(),
                    multiplier=mult,
                    source="dom",
                ))
            if added:
                print(f"Captured {added} new DOM round(s).")
        except Exception as e:
            print("DOM polling error:", e)

        await asyncio.sleep(1.0)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    con = connect(cfg["storage"]["sqlite_path"])
    profile = Path(cfg["collector"]["user_data_dir"])
    profile.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=str(profile),
            headless=bool(cfg["collector"].get("headless", False)),
        )
        page = browser.pages[0] if browser.pages else await browser.new_page()
        mode = cfg["collector"]["mode"].lower().strip()

        if mode == "network":
            await attach_network_listener(page, cfg, con)
        elif mode == "websocket":
            await attach_websocket_listener(page, cfg, con)

        print(
            "Read-only collector started. Authenticate manually if needed. "
            "No bet/cash-out automation is included."
        )
        await page.goto(cfg["sportybet"]["url"], wait_until="domcontentloaded")

        if mode == "dom":
            await dom_loop(page, cfg, con)
        elif mode in {"network", "websocket"}:
            while True:
                await asyncio.sleep(3600)
        else:
            raise ValueError("collector.mode must be 'network', 'websocket', or 'dom'")

if __name__ == "__main__":
    asyncio.run(main())
