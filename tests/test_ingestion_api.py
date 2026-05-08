from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

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
        self.assertEqual(counts["hub_retained_audio_clips"], 1)
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
            unexpected_field = _payload(buffer_ids=[2])
            unexpected_field["detections"][0]["filepath"] = "legacy.flac"
            unexpected_field_response = client.post(
                "/ingest_batch",
                content=msgpack.packb(unexpected_field, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )
            invalid_uuid = _payload(buffer_ids=[3])
            invalid_uuid["detections"][0]["event_uuid"] = "not-a-uuid"
            invalid_uuid_response = client.post(
                "/ingest_batch",
                content=msgpack.packb(invalid_uuid, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )

        self.assertEqual(bad_msgpack.status_code, 422)
        self.assertEqual(missing_fields.status_code, 422)
        self.assertEqual(bad_embedding_response.status_code, 422)
        self.assertEqual(unexpected_field_response.status_code, 422)
        self.assertEqual(invalid_uuid_response.status_code, 422)

    def test_bad_retained_clip_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, _ = self._write_config(Path(tmp))
            client = TestClient(create_app(config_path))

            payload = _payload(buffer_ids=[1])
            payload["detections"][0]["retained_clip_count"] = 2
            count_mismatch = client.post(
                "/ingest_batch",
                content=msgpack.packb(payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )

            payload = _payload(buffer_ids=[2])
            payload["detections"][0]["audio_saved"] = 0
            audio_mismatch = client.post(
                "/ingest_batch",
                content=msgpack.packb(payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )

            payload = _payload(buffer_ids=[3])
            payload["detections"][0]["retained_audio_clips"][0]["end_segment_index"] = 4
            bad_span = client.post(
                "/ingest_batch",
                content=msgpack.packb(payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )

            payload = _payload(buffer_ids=[4])
            payload["detections"][0]["retained_audio_clips"][0]["duration_s"] = 6.0
            bad_duration = client.post(
                "/ingest_batch",
                content=msgpack.packb(payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )

        self.assertEqual(count_mismatch.status_code, 422)
        self.assertEqual(audio_mismatch.status_code, 422)
        self.assertEqual(bad_span.status_code, 422)
        self.assertEqual(bad_duration.status_code, 422)

    def test_source_buffer_id_can_repeat_when_event_uuid_differs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, db_path = self._write_config(Path(tmp))
            client = TestClient(create_app(config_path))
            first_payload = _payload(buffer_ids=[7])
            second_payload = _payload(buffer_ids=[7])

            first = client.post(
                "/ingest_batch",
                content=msgpack.packb(first_payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )
            second = client.post(
                "/ingest_batch",
                content=msgpack.packb(second_payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )
            counts = self._db_counts(db_path)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(counts["hub_buffer_events"], 2)

    def test_duplicate_event_uuid_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path, _ = self._write_config(Path(tmp))
            client = TestClient(create_app(config_path))
            payload = _payload(buffer_ids=[7])

            first = client.post(
                "/ingest_batch",
                content=msgpack.packb(payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )
            duplicate = client.post(
                "/ingest_batch",
                content=msgpack.packb(payload, use_bin_type=True),
                headers={"X-API-Key": "test-key"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(duplicate.status_code, 409)

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
                "hub_retained_audio_clips": conn.execute(
                    "SELECT COUNT(*) FROM hub_retained_audio_clips;"
                ).fetchone()[0],
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
        "event_uuid": str(uuid4()),
        "source_file": "fixture.wav",
        "file_buffer_index": buffer_id,
        "timestamp_utc": timestamp_utc,
        "inference_buffer_seconds": 15.0,
        "perch_window_seconds": 5.0,
        "perch_frame_count": 3,
        "audio_saved": 1,
        "retention_reason": "bio_hit",
        "max_nz_bird_common_name": "Nova Bird",
        "max_nz_bird_scientific_name": "Aves nova",
        "max_nz_bird_logit": 4.25,
        "max_perch_label": "Aves nova",
        "max_perch_logit": 4.25,
        "excluded_label_scores": "[]",
        "nz_bird_logits": "[]",
        "gate_mode": "nz_bird_margin",
        "gate_threshold": 0.55,
        "gate_trigger_count": 1,
        "retained_clip_count": 1,
        "margin_gate_scores": "[]",
        "retained_audio_clips": [
            {
                "source_clip_id": buffer_id * 100,
                "retention_index": 1,
                "retention_reason": "bio_hit",
                "filepath": "edge_node_mock/data/retained_audio/example_seg0-0_5s.flac",
                "start_segment_index": 0,
                "end_segment_index": 0,
                "start_offset_s": 0.0,
                "end_offset_s": 5.0,
                "duration_s": 5.0,
                "triggered_frame_count": 1,
            }
        ],
        "embedding_segments": [
            {"embedding_id": buffer_id * 10 + index, "segment_index": index, "embedding": embedding}
            for index in range(3)
        ],
    }


if __name__ == "__main__":
    unittest.main()
