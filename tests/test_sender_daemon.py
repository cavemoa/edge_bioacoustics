from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import requests

from edge_node_mock.src.bio_capture_loop import active_vector_storage, load_sqlite_vec
from edge_node_mock.src.init_edge_db import init_edge_db
from edge_node_mock.src.sender_daemon import (
    load_pending_detections,
    mark_buffers_synced,
    send_pending_batch,
)


class SenderDaemonTest(unittest.TestCase):
    def test_load_pending_detections_includes_three_binary_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, db_path = self._create_edge_db(Path(tmp), buffer_count=1)

            detections = load_pending_detections(_config(db_path), limit=1)
            config_exists = config_path.exists()

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["buffer_id"], 1)
        self.assertEqual(len(detections[0]["embedding_segments"]), 3)
        self.assertEqual(len(detections[0]["embedding_segments"][0]["embedding"]), 1536 * 4)
        self.assertTrue(config_exists)

    def test_dry_run_does_not_update_sync_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, db_path = self._create_edge_db(Path(tmp), buffer_count=2)

            result = send_pending_batch(config_path, limit=1, dry_run=True)
            statuses = self._sync_statuses(db_path)

        self.assertEqual(result.status, "dry_run")
        self.assertEqual(result.pending_count, 2)
        self.assertEqual(result.detection_count, 1)
        self.assertGreater(result.payload_bytes, 0)
        self.assertEqual(statuses, ["pending", "pending"])

    def test_successful_send_marks_only_accepted_buffer_ids_synced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, db_path = self._create_edge_db(Path(tmp), buffer_count=2)
            response = Mock(status_code=200)
            response.json.return_value = {"status": "ok", "accepted_buffer_ids": [1]}

            with patch("edge_node_mock.src.sender_daemon.requests.post", return_value=response) as post:
                result = send_pending_batch(config_path, limit=2)
            statuses = self._sync_statuses(db_path)

        self.assertEqual(result.status, "sent")
        self.assertEqual(result.accepted_buffer_ids, [1])
        self.assertEqual(statuses, ["synced", "pending"])
        self.assertEqual(post.call_args.kwargs["headers"]["X-API-Key"], "test-key")

    def test_network_failure_leaves_rows_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, db_path = self._create_edge_db(Path(tmp), buffer_count=1)

            with patch(
                "edge_node_mock.src.sender_daemon.requests.post",
                side_effect=requests.ConnectionError("offline"),
            ):
                result = send_pending_batch(config_path, limit=1)
            statuses = self._sync_statuses(db_path)

        self.assertEqual(result.status, "failed")
        self.assertEqual(statuses, ["pending"])

    def test_mark_buffers_synced_ignores_unaccepted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, db_path = self._create_edge_db(Path(tmp), buffer_count=2)

            mark_buffers_synced(_config(db_path), [2])
            statuses = self._sync_statuses(db_path)

        self.assertEqual(statuses, ["pending", "synced"])

    def _create_edge_db(self, root: Path, *, buffer_count: int) -> tuple[Path, Path]:
        db_path = root / "edge.sqlite"
        config_path = root / "edge_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    "device_id: pi_01",
                    f"edge_db_path: {db_path}",
                    f"retained_audio_dir: {root / 'retained_audio'}",
                    "raw_audio_mount: .",
                    'raw_audio_glob: "*.wav"',
                    "embedding_dim: 1536",
                    "hub_ingest_url: http://127.0.0.1:8000/ingest_batch",
                    "api_key: test-key",
                    "telemetry_defaults:",
                    "  cpu_temp_c: 45.0",
                    "  battery_voltage: 12.4",
                    "  solar_amps: 0.0",
                ]
            ),
            encoding="utf-8",
        )
        init_edge_db(config_path, reset=True)
        self._insert_pending_rows(db_path, buffer_count)
        return config_path, db_path

    def _insert_pending_rows(self, db_path: Path, buffer_count: int) -> None:
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        embedding = np.zeros(1536, dtype=np.float32).tobytes()
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON;")
            vector_table, vector_mode = active_vector_storage(conn)
            if vector_mode == "sqlite_vec":
                load_sqlite_vec(conn)
            with conn:
                for index in range(buffer_count):
                    cursor = conn.execute(
                        """
                        INSERT INTO buffer_events(
                            device_id, timestamp_utc, audio_saved, retention_reason,
                            filepath, max_bio_label, max_bio_logit, noise_logits,
                            max_perch_label, max_perch_logit, nz_bird_logits,
                            sync_status, created_at_utc
                        )
                        VALUES ('pi_01', ?, 1, 'bio_hit', 'example.flac',
                                'Animal', 3.5, '[]', 'Animal', 3.5, '[]',
                                'pending', ?);
                        """,
                        (now, now),
                    )
                    buffer_id = int(cursor.lastrowid)
                    for segment_index in range(3):
                        cursor = conn.execute(
                            "INSERT INTO embedding_segments(buffer_id, segment_index) VALUES (?, ?);",
                            (buffer_id, segment_index),
                        )
                        embedding_id = int(cursor.lastrowid)
                        conn.execute(
                            f"INSERT INTO {vector_table}(embedding_id, embedding) VALUES (?, ?);",
                            (embedding_id, embedding),
                        )

    def _sync_statuses(self, db_path: Path) -> list[str]:
        with sqlite3.connect(db_path) as conn:
            return [
                row[0]
                for row in conn.execute(
                    "SELECT sync_status FROM buffer_events ORDER BY buffer_id;"
                ).fetchall()
            ]


def _config(db_path: Path) -> dict:
    return {
        "device_id": "pi_01",
        "edge_db_path": str(db_path),
        "hub_ingest_url": "http://127.0.0.1:8000/ingest_batch",
        "api_key": "test-key",
        "telemetry_defaults": {
            "cpu_temp_c": 45.0,
            "battery_voltage": 12.4,
            "solar_amps": 0.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
