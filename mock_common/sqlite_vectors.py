"""SQLite vector table setup helpers for Phase 1."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class VectorStorage:
    table_name: str
    mode: str


def create_vector_storage(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    id_column: str,
    embedding_dim: int,
    fallback_table_name: str,
) -> VectorStorage:
    """Create sqlite-vec storage when available, otherwise create a BLOB table."""

    try:
        import sqlite_vec  # type: ignore

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {table_name}
            USING vec0(
                {id_column} INTEGER PRIMARY KEY,
                embedding float[{embedding_dim}]
            );
            """
        )
        return VectorStorage(table_name=table_name, mode="sqlite_vec")
    except Exception:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {fallback_table_name} (
                {id_column} INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            );
            """
        )
        return VectorStorage(table_name=fallback_table_name, mode="blob")


def set_metadata(conn: sqlite3.Connection, key: str, value: object) -> None:
    conn.execute(
        """
        INSERT INTO schema_metadata(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value;
        """,
        (key, str(value)),
    )
