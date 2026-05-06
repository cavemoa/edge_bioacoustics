from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import msgpack
import numpy as np
from fastapi.testclient import TestClient

from central_hub_mock.src.ingestion_api import create_app, load_sqlite_vec


class IngestionApiTest(unittest.TestCase):
    def test_valid_msgpack_payload_is_authenticated_validated_and_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, db_path = self._write_config(Path(tmp))
            client = TestClient(create_app(config_path))
            payload = _payload(buffer_ids=[11])

            response = client.post(
                "/ingest_batch",
                content=msgpack.packb(payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "ok", "accepted_buffer_ids": [11]})
            counts = self._db_counts(db_path)

        self.assertEqual(counts["ingestion_batches"], 1)
        self.assertEqual(counts["hub_buffer_events"], 1)
        self.assertEqual(counts["hub_embedding_segments"], 3)
        self.assertEqual(counts["health_metrics"], 1)
        self.assertEqual(counts["vectors"], 3)
        self.assertEqual(counts["journal_mode"], "wal")

    def test_bad_api_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, _ = self._write_config(Path(tmp))
            client = TestClient(create_app(config_path))

            response = client.post(
                "/ingest_batch",
                content=msgpack.packb(_payload(buffer_ids=[1]), use_bin_type=True),
                headers={"X-API-Key": "wrong-key"},
            )

        self.assertEqual(response.status_code, 403)

    def test_malformed_payload_returns_validation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, _ = self._write_config(Path(tmp))
            client = TestClient(create_app(config_path))

            bad_msgpack = client.post(
                "/ingest_batch",
                content=b"not-msgpack",
                headers={"X-API-Key": "test-key"},
            )
            missing_fields = client.post(
                "/ingest_batch",
                content=msgpack.packb({"device_id": "pi_01"}, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )
            bad_embedding = _payload(buffer_ids=[1])
            bad_embedding["detections"][0]["embedding_segments"][0]["embedding"] = b"short"
            bad_embedding_response = client.post(
                "/ingest_batch",
                content=msgpack.packb(bad_embedding, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )

        self.assertEqual(bad_msgpack.status_code, 422)
        self.assertEqual(missing_fields.status_code, 422)
        self.assertEqual(bad_embedding_response.status_code, 422)

    def test_device_allowlist_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, _ = self._write_config(Path(tmp))
            client = TestClient(create_app(config_path))
            payload = _payload(buffer_ids=[1])
            payload["device_id"] = "pi_99"

            response = client.post(
                "/ingest_batch",
                content=msgpack.packb(payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )

        self.assertEqual(response.status_code, 403)

    def _write_config(self, root: Path) -> tuple[Path, Path]:
        db_path = root / "master.sqlite"
        config_path = root / "hub_config.yaml"
        config_path.write_text(
            "\n".join(
                [
                    f"master_db_path: {db_path}",
                    "api_key: test-key",
                    "allowed_device_ids:",
                    "  - pi_01",
                    "watchdog_stale_minutes: 75",
                    "embedding_dim: 1536",
                ]
            ),
            encoding="utf-8",
        )
        return config_path, db_path

    def _db_counts(self, db_path: Path) -> dict[str, int | str]:
        with sqlite3.connect(db_path) as conn:
            metadata = dict(conn.execute("SELECT key, value FROM schema_metadata;").fetchall())
            vector_table = metadata["vector_table"]
            if vector_table == "hub_perch_vectors":
                load_sqlite_vec(conn)
            return {
                "ingestion_batches": conn.execute("SELECT COUNT(*) FROM ingestion_batches;").fetchone()[0],
                "hub_buffer_events": conn.execute("SELECT COUNT(*) FROM hub_buffer_events;").fetchone()[0],
                "hub_embedding_segments": conn.execute("SELECT COUNT(*) FROM hub_embedding_segments;").fetchone()[0],
                "health_metrics": conn.execute("SELECT COUNT(*) FROM health_metrics;").fetchone()[0],
                "vectors": conn.execute(f"SELECT COUNT(*) FROM {vector_table};").fetchone()[0],
                "journal_mode": conn.execute("PRAGMA journal_mode;").fetchone()[0].lower(),
            }


def _payload(*, buffer_ids: list[int]) -> dict:
    now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return {
        "device_id": "pi_01",
        "sent_at_utc": now,
        "detections": [_detection(buffer_id, now) for buffer_id in buffer_ids],
        "telemetry": {
            "timestamp_utc": now,
            "cpu_temp_c": 45.0,
            "cpu_load_pct": 12.5,
            "disk_free_gb": 128.0,
            "battery_voltage": 12.4,
            "solar_amps": 0.0,
        },
    }


def _detection(buffer_id: int, timestamp_utc: str) -> dict:
    embedding = np.zeros(1536, dtype=np.float32).tobytes()
    return {
        "buffer_id": buffer_id,
        "timestamp_utc": timestamp_utc,
        "audio_saved": 1,
        "retention_reason": "bio_hit",
        "filepath": "edge_node_mock/data/retained_audio/example.flac",
        "max_bio_label": "Animal",
        "max_bio_logit": 4.25,
        "noise_logits": "[]",
        "max_perch_label": "Animal",
        "max_perch_logit": 4.25,
        "nz_bird_logits": "[]",
        "embedding_segments": [
            {"embedding_id": buffer_id * 10 + index, "segment_index": index, "embedding": embedding}
            for index in range(3)
        ],
    }


if __name__ == "__main__":
    unittest.main()
