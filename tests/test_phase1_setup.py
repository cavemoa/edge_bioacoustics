from __future__ import annotations

import sqlite3
import struct
import tempfile
import unittest
import wave
from pathlib import Path

from central_hub_mock.src.init_master_db import init_master_db
from edge_node_mock.src.audio_smoke_test import run_smoke_check
from edge_node_mock.src.init_edge_db import init_edge_db


class Phase1SetupTest(unittest.TestCase):
    def test_audio_smoke_check_accepts_configured_wav(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wav_path = root / "night_01" / "fixture.wav"
            wav_path.parent.mkdir()
            with wave.open(str(wav_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(32000)
                frames = struct.pack("<" + "h" * 3200, *([1024] * 3200))
                wav.writeframes(frames)

            config_path = root / "edge_config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "raw_audio_mount: " + str(root),
                        'raw_audio_glob: "**/*.wav"',
                        "source_sample_rates:",
                        "  - 32000",
                    ]
                ),
                encoding="utf-8",
            )

            report = run_smoke_check(config_path)

        self.assertEqual(report["sample_rate"], 32000)
        self.assertEqual(report["channels"], 1)
        self.assertEqual(report["mono_dtype"], "float32")
        self.assertGreater(report["peak_amplitude"], 0.0)

    def test_edge_and_master_schemas_initialize_and_accept_dummy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            edge_config = root / "edge_config.yaml"
            hub_config = root / "hub_config.yaml"
            edge_db = root / "edge.sqlite"
            master_db = root / "master.sqlite"
            edge_config.write_text(
                "\n".join(
                    [
                        "edge_db_path: " + str(edge_db),
                        "retained_audio_dir: " + str(root / "retained_audio"),
                        "embedding_dim: 1536",
                    ]
                ),
                encoding="utf-8",
            )
            hub_config.write_text(
                "\n".join(
                    [
                        "master_db_path: " + str(master_db),
                        "embedding_dim: 1536",
                    ]
                ),
                encoding="utf-8",
            )

            init_edge_db(edge_config)
            init_master_db(hub_config)

            self._assert_edge_schema(edge_db)
            self._assert_master_schema(master_db)

    def _assert_edge_schema(self, db_path: Path) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON;")
            tables = self._tables(conn)
            self.assertTrue({"buffer_events", "embedding_segments", "schema_metadata"}.issubset(tables))
            self.assertTrue({"perch_vectors", "perch_vector_blobs"} & tables)
            self.assertEqual(self._metadata(conn, "embedding_dim"), "1536")

            conn.execute(
                """
                INSERT INTO buffer_events(
                    device_id, timestamp_utc, audio_saved, retention_reason,
                    created_at_utc
                )
                VALUES ('pi_01', '2026-01-01T00:00:00Z', 0, 'dropped', '2026-01-01T00:00:00Z');
                """
            )
            buffer_id = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
            for index in range(3):
                conn.execute(
                    "INSERT INTO embedding_segments(buffer_id, segment_index) VALUES (?, ?);",
                    (buffer_id, index),
                )
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM embedding_segments WHERE buffer_id = ?;", (buffer_id,)).fetchone()[0],
                3,
            )
            conn.execute("DELETE FROM buffer_events WHERE buffer_id = ?;", (buffer_id,))
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM embedding_segments;").fetchone()[0], 0)

    def _assert_master_schema(self, db_path: Path) -> None:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON;")
            tables = self._tables(conn)
            expected = {
                "ingestion_batches",
                "hub_buffer_events",
                "hub_embedding_segments",
                "health_metrics",
                "schema_metadata",
            }
            self.assertTrue(expected.issubset(tables))
            self.assertTrue({"hub_perch_vectors", "hub_perch_vector_blobs"} & tables)
            self.assertEqual(self._metadata(conn, "embedding_dim"), "1536")
            self.assertEqual(conn.execute("PRAGMA journal_mode;").fetchone()[0].lower(), "wal")

            conn.execute(
                """
                INSERT INTO ingestion_batches(
                    device_id, sent_at_utc, received_at_utc, payload_bytes,
                    detection_count, status
                )
                VALUES ('pi_01', '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z', 128, 1, 'accepted');
                """
            )
            batch_id = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
            conn.execute(
                """
                INSERT INTO hub_buffer_events(
                    device_id, source_buffer_id, batch_id, timestamp_utc,
                    audio_saved, retention_reason, received_at_utc
                )
                VALUES ('pi_01', 7, ?, '2026-01-01T00:00:00Z', 0, 'dropped', '2026-01-01T00:00:01Z');
                """,
                (batch_id,),
            )
            hub_buffer_id = conn.execute("SELECT last_insert_rowid();").fetchone()[0]
            for index in range(3):
                conn.execute(
                    "INSERT INTO hub_embedding_segments(hub_buffer_id, source_embedding_id, segment_index) VALUES (?, ?, ?);",
                    (hub_buffer_id, index + 10, index),
                )
            self.assertEqual(
                conn.execute(
                    "SELECT COUNT(*) FROM hub_embedding_segments WHERE hub_buffer_id = ?;",
                    (hub_buffer_id,),
                ).fetchone()[0],
                3,
            )
            conn.execute("DELETE FROM ingestion_batches WHERE batch_id = ?;", (batch_id,))
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM hub_buffer_events;").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM hub_embedding_segments;").fetchone()[0], 0)

    @staticmethod
    def _tables(conn: sqlite3.Connection) -> set[str]:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table');"
            ).fetchall()
        }

    @staticmethod
    def _metadata(conn: sqlite3.Connection, key: str) -> str:
        return conn.execute("SELECT value FROM schema_metadata WHERE key = ?;", (key,)).fetchone()[0]


if __name__ == "__main__":
    unittest.main()
