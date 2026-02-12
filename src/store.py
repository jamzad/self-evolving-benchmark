import sqlite3
from pathlib import Path

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS questions (
  question_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  domain TEXT NOT NULL,
  category TEXT NOT NULL,
  difficulty INTEGER NOT NULL,
  prompt TEXT NOT NULL,
  prompt_hash TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  run_at TEXT NOT NULL,
  base_url TEXT NOT NULL,
  solve_model TEXT NOT NULL,
  judge_model TEXT NOT NULL,
  n_questions INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS results (
  result_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  question_id TEXT NOT NULL,
  answer TEXT NOT NULL,
  judge_json TEXT NOT NULL,
  score REAL NOT NULL,
  confidence REAL,
  latency_ms INTEGER,
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
  FOREIGN KEY(question_id) REFERENCES questions(question_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
"""

def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

def _add_column_if_missing(con: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    cols = {r["name"] for r in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
        con.commit()

def init_db(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA_SQL)
    con.commit()

    # Migration-style additions (safe on existing DBs)
    _add_column_if_missing(con, "runs", "batch_mean", "REAL")
    _add_column_if_missing(con, "runs", "ema_after", "REAL")
    _add_column_if_missing(con, "runs", "target_difficulty", "INTEGER")
