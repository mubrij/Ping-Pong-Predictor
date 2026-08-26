from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any
import pandas as pd

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS rounds (
    round_key TEXT PRIMARY KEY,
    round_id TEXT,
    observed_at TEXT NOT NULL,
    multiplier REAL NOT NULL CHECK(multiplier >= 1.0),
    source TEXT NOT NULL,
    raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_rounds_observed_at
ON rounds(observed_at);

CREATE TABLE IF NOT EXISTS artifacts (
    name TEXT PRIMARY KEY,
    updated_at TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""

def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    con.executescript(SCHEMA)
    return con

def _round_key(
    round_id: str | None,
    observed_at: str,
    multiplier: float,
    source: str,
) -> str:
    if round_id:
        return f"id:{round_id}"
    raw = f"{observed_at}|{multiplier:.8f}|{source}"
    return "sha:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()

def insert_round(
    con: sqlite3.Connection,
    *,
    round_id: str | None,
    observed_at: str,
    multiplier: float,
    source: str,
    raw_json: str | None = None,
) -> bool:
    multiplier = float(multiplier)
    if not (multiplier >= 1.0):
        return False

    key = _round_key(round_id, observed_at, multiplier, source)
    cur = con.execute(
        """
        INSERT OR IGNORE INTO rounds
        (round_key, round_id, observed_at, multiplier, source, raw_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (key, round_id, observed_at, multiplier, source, raw_json),
    )
    con.commit()
    return cur.rowcount == 1

def load_rounds(
    con: sqlite3.Connection,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    if limit is None:
        sql = """
        SELECT round_key, round_id, observed_at, multiplier, source
        FROM rounds
        ORDER BY datetime(observed_at), rowid
        """
        df = pd.read_sql_query(sql, con)
    else:
        sql = """
        SELECT round_key, round_id, observed_at, multiplier, source
        FROM (
            SELECT rowid, round_key, round_id, observed_at, multiplier, source
            FROM rounds
            ORDER BY datetime(observed_at) DESC, rowid DESC
            LIMIT ?
        )
        ORDER BY datetime(observed_at), rowid
        """
        df = pd.read_sql_query(sql, con, params=(int(limit),))

    if not df.empty:
        df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce")
        df = df.dropna(subset=["multiplier"]).reset_index(drop=True)
    return df

def count_rounds(con: sqlite3.Connection) -> int:
    return int(con.execute("SELECT COUNT(*) FROM rounds").fetchone()[0])

def latest_round(con: sqlite3.Connection) -> dict[str, Any] | None:
    row = con.execute(
        """
        SELECT round_id, observed_at, multiplier, source
        FROM rounds
        ORDER BY datetime(observed_at) DESC, rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        return None
    return {
        "round_id": row[0],
        "observed_at": row[1],
        "multiplier": float(row[2]),
        "source": row[3],
    }
