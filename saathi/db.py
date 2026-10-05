"""Tiny SQLite layer. Everything Saathi knows lives in one local file."""
import json
import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    created INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    emb TEXT,
    gen_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    doc_id INTEGER NOT NULL,
    qhash TEXT NOT NULL UNIQUE,
    question TEXT NOT NULL,
    options TEXT NOT NULL,
    answer_index INTEGER NOT NULL,
    explanation TEXT NOT NULL,
    ease REAL NOT NULL DEFAULT 2.5,
    interval_days REAL NOT NULL DEFAULT 0,
    reps INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0,
    due INTEGER NOT NULL,
    created INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER NOT NULL,
    ts INTEGER NOT NULL,
    chosen INTEGER NOT NULL,
    correct INTEGER NOT NULL,
    confident INTEGER NOT NULL,
    tag TEXT,
    why TEXT
);
CREATE INDEX IF NOT EXISTS idx_cards_due ON cards(due);
CREATE INDEX IF NOT EXISTS idx_attempts_ts ON attempts(ts);
"""


class DB:
    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def rows(self, sql, args=()):
        with self._lock:
            cur = self.conn.execute(sql, args)
            return [dict(r) for r in cur.fetchall()]

    def row(self, sql, args=()):
        out = self.rows(sql, args)
        return out[0] if out else None

    def run(self, sql, args=()):
        """Execute a write; returns lastrowid."""
        with self._lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # -- settings -----------------------------------------------------
    def get_setting(self, key, default=None):
        r = self.row("SELECT value FROM settings WHERE key = ?", (key,))
        return json.loads(r["value"]) if r else default

    def set_setting(self, key, value):
        self.run(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value, ensure_ascii=False)),
        )
