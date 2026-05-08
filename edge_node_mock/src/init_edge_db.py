"""Initialize the Phase 1 edge-node SQLite database."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mock_common.config import load_config
from mock_common.sqlite_vectors import create_vector_storage, set_metadata


def ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_sql: str) -> None:
    """Add a column to an existing table when a Phase 1 schema evolves."""

    columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name});").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_sql};")


def init_edge_db(config_path: str | Path, *, reset: bool = False) -> Path:
    config = load_config(config_path)
    db_path = Path(str(config["edge_db_path"]))
    embedding_dim = int(config["embedding_dim"])

    if reset and db_path.exists():
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    retained_audio_dir = Path(str(config["retained_audio_dir"]))
    retained_audio_dir.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS buffer_events (
                buffer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_uuid TEXT NOT NULL UNIQUE,
                device_id TEXT NOT NULL,
                source_file TEXT,
                file_buffer_index INTEGER,
                timestamp_utc TEXT NOT NULL,
                inference_buffer_seconds REAL NOT NULL DEFAULT 15.0,
                perch_window_seconds REAL NOT NULL DEFAULT 5.0,
                perch_frame_count INTEGER NOT NULL DEFAULT 3,
                audio_saved INTEGER NOT NULL DEFAULT 0 CHECK(audio_saved IN (0, 1)),
                retention_reason TEXT NOT NULL CHECK(retention_reason IN ('bio_hit', 'dropped')),
                max_nz_bird_common_name TEXT,
                max_nz_bird_scientific_name TEXT,
                max_nz_bird_logit REAL,
                max_perch_label TEXT,
                max_perch_logit REAL,
                excluded_label_scores TEXT,
                nz_bird_logits TEXT,
                gate_mode TEXT NOT NULL,
                gate_threshold REAL NOT NULL,
                gate_trigger_count INTEGER NOT NULL DEFAULT 0,
                retained_clip_count INTEGER NOT NULL DEFAULT 0,
                margin_gate_scores TEXT NOT NULL,
                sync_status TEXT NOT NULL DEFAULT 'pending' CHECK(sync_status IN ('pending', 'in_flight', 'synced', 'failed')),
                created_at_utc TEXT NOT NULL,
                synced_at_utc TEXT
            );
            """
        )
        ensure_column(conn, "buffer_events", "event_uuid", "event_uuid TEXT")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_buffer_events_event_uuid ON buffer_events(event_uuid);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_buffer_events_sync_status ON buffer_events(sync_status);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_buffer_events_timestamp ON buffer_events(timestamp_utc);")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retained_audio_clips (
                clip_id INTEGER PRIMARY KEY AUTOINCREMENT,
                buffer_id INTEGER NOT NULL,
                retention_index INTEGER NOT NULL,
                retention_reason TEXT NOT NULL CHECK(retention_reason IN ('bio_hit')),
                filepath TEXT NOT NULL,
                start_segment_index INTEGER NOT NULL,
                end_segment_index INTEGER NOT NULL,
                start_offset_s REAL NOT NULL,
                end_offset_s REAL NOT NULL,
                duration_s REAL NOT NULL,
                triggered_frame_count INTEGER NOT NULL,
                created_at_utc TEXT NOT NULL,
                FOREIGN KEY(buffer_id) REFERENCES buffer_events(buffer_id) ON DELETE CASCADE,
                UNIQUE(buffer_id, retention_index)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retained_audio_clips_buffer_id ON retained_audio_clips(buffer_id);"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_segments (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                buffer_id INTEGER NOT NULL,
                segment_index INTEGER NOT NULL,
                FOREIGN KEY(buffer_id) REFERENCES buffer_events(buffer_id) ON DELETE CASCADE,
                UNIQUE(buffer_id, segment_index)
            );
            """
        )
        vector_storage = create_vector_storage(
            conn,
            table_name="perch_vectors",
            id_column="embedding_id",
            embedding_dim=embedding_dim,
            fallback_table_name="perch_vector_blobs",
        )
        set_metadata(conn, "embedding_dim", embedding_dim)
        set_metadata(conn, "vector_storage_mode", vector_storage.mode)
        set_metadata(conn, "vector_table", vector_storage.table_name)
        set_metadata(conn, "schema_version", 3)
        set_metadata(conn, "phase1_revision", "margin_variable_retention_v1")
        set_metadata(conn, "gate_mode", str(config.get("bio_gate_mode", "nz_bird_margin")))
        set_metadata(conn, "gate_threshold", str(config.get("bio_margin_threshold", "0.55")))
        conn.commit()

    return db_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="edge_node_mock/config/edge_config.example.yaml")
    parser.add_argument("--reset", action="store_true", help="Delete the configured database before initializing it.")
    args = parser.parse_args(argv)

    db_path = init_edge_db(args.config, reset=args.reset)
    print(f"Initialized edge database: {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
