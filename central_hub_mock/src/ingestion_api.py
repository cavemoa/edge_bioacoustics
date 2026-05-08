"""FastAPI MessagePack ingestion service for the Phase 1 central hub mock."""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgpack
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from central_hub_mock.src.init_master_db import init_master_db
from mock_common.config import load_config


DEFAULT_CONFIG_PATH = REPO_ROOT / "central_hub_mock" / "config" / "hub_config.example.yaml"


class EmbeddingSegmentPayload(BaseModel):
    source_embedding_id: int | None = None
    embedding_id: int | None = None
    segment_index: int
    embedding: bytes

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def require_source_embedding_id(self) -> "EmbeddingSegmentPayload":
        if self.source_embedding_id is None and self.embedding_id is None:
            raise ValueError("Each embedding segment must include source_embedding_id or embedding_id")
        return self

    @property
    def normalized_source_embedding_id(self) -> int | None:
        return self.source_embedding_id if self.source_embedding_id is not None else self.embedding_id


class RetainedAudioClipPayload(BaseModel):
    source_clip_id: int | None = None
    retention_index: int
    retention_reason: str
    filepath: str
    start_segment_index: int
    end_segment_index: int
    start_offset_s: float
    end_offset_s: float
    duration_s: float
    triggered_frame_count: int

    model_config = ConfigDict(extra="forbid")

    @field_validator("retention_reason")
    @classmethod
    def validate_retention_reason(cls, value: str) -> str:
        allowed = {"bio_hit", "validation_sample"}
        if value not in allowed:
            raise ValueError(f"retention_reason must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def validate_clip_span(self) -> "RetainedAudioClipPayload":
        if self.start_segment_index < 0 or self.end_segment_index > 2:
            raise ValueError("clip segment indexes must be between 0 and 2")
        if self.end_segment_index < self.start_segment_index:
            raise ValueError("end_segment_index must be >= start_segment_index")
        if self.triggered_frame_count != self.end_segment_index - self.start_segment_index + 1:
            raise ValueError("triggered_frame_count must match the retained segment span")
        if self.duration_s not in {5.0, 10.0, 15.0}:
            raise ValueError("duration_s must be 5, 10, or 15 seconds")
        return self


class DetectionPayload(BaseModel):
    buffer_id: int
    timestamp_utc: str
    audio_saved: int | bool
    retention_reason: str
    filepath: str | None = None
    max_bio_label: str | None = None
    max_bio_logit: float | None = None
    noise_logits: str | None = None
    max_perch_label: str | None = None
    max_perch_logit: float | None = None
    nz_bird_logits: str | None = None
    gate_mode: str | None = None
    gate_threshold: float | None = None
    gate_trigger_count: int = 0
    retained_clip_count: int = 0
    margin_gate_scores: str | None = None
    retained_audio_clips: list[RetainedAudioClipPayload] = Field(default_factory=list)
    embedding_segments: list[EmbeddingSegmentPayload] = Field(min_length=3, max_length=3)

    model_config = ConfigDict(extra="forbid")

    @field_validator("audio_saved")
    @classmethod
    def normalize_audio_saved(cls, value: int | bool) -> int:
        return int(bool(value))

    @field_validator("retention_reason")
    @classmethod
    def validate_retention_reason(cls, value: str) -> str:
        allowed = {"bio_hit", "validation_sample", "dropped"}
        if value not in allowed:
            raise ValueError(f"retention_reason must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def require_three_distinct_segments(self) -> "DetectionPayload":
        indexes = [segment.segment_index for segment in self.embedding_segments]
        if sorted(indexes) != [0, 1, 2]:
            raise ValueError("embedding_segments must contain segment_index values 0, 1, and 2")
        if self.retained_clip_count != len(self.retained_audio_clips):
            raise ValueError("retained_clip_count must equal len(retained_audio_clips)")
        if int(self.audio_saved) != int(self.retained_clip_count > 0):
            raise ValueError("audio_saved must equal retained_clip_count > 0")
        return self


class TelemetryPayload(BaseModel):
    timestamp_utc: str
    cpu_temp_c: float | None = None
    cpu_load_pct: float | None = None
    disk_free_gb: float | None = None
    battery_voltage: float | None = None
    solar_amps: float | None = None

    model_config = ConfigDict(extra="forbid")


class IngestBatchPayload(BaseModel):
    device_id: str
    sent_at_utc: str
    detections: list[DetectionPayload]
    telemetry: TelemetryPayload

    model_config = ConfigDict(extra="forbid")


def create_app(config_path: str | Path | None = None) -> FastAPI:
    """Create a configured FastAPI app.

    Tests call this factory directly with a temporary config. Normal `uvicorn`
    usage relies on the module-level `app`, which reads `HUB_CONFIG`.
    """

    resolved_config_path = Path(config_path or os.environ.get("HUB_CONFIG", DEFAULT_CONFIG_PATH))
    config = load_config(resolved_config_path)
    init_master_db(resolved_config_path)

    api = FastAPI(title="Edge Bioacoustics Phase 1 Hub Ingestion API")
    api.state.config_path = resolved_config_path
    api.state.config = config

    @api.post("/ingest_batch")
    async def ingest_batch(request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        if x_api_key != api.state.config["api_key"]:
            raise HTTPException(status_code=403, detail="Invalid API key")

        body = await request.body()
        try:
            decoded = msgpack.unpackb(body, raw=False)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid MessagePack payload: {exc}") from exc

        try:
            payload = IngestBatchPayload.model_validate(decoded)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        allowed_device_ids = set(api.state.config.get("allowed_device_ids", []))
        if allowed_device_ids and payload.device_id not in allowed_device_ids:
            raise HTTPException(status_code=403, detail=f"Device is not allowed: {payload.device_id}")

        embedding_dim = int(api.state.config["embedding_dim"])
        expected_embedding_bytes = embedding_dim * 4
        for detection in payload.detections:
            for segment in detection.embedding_segments:
                if len(segment.embedding) != expected_embedding_bytes:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"Embedding for buffer {detection.buffer_id}, segment {segment.segment_index} "
                            f"has {len(segment.embedding)} bytes; expected {expected_embedding_bytes}"
                        ),
                    )

        try:
            accepted_buffer_ids = insert_batch(
                api.state.config,
                payload,
                payload_bytes=len(body),
                received_at_utc=utc_now_string(),
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code=409, detail=f"Database constraint failed: {exc}") from exc

        return {"status": "ok", "accepted_buffer_ids": accepted_buffer_ids}

    return api


def insert_batch(
    config: dict[str, Any],
    payload: IngestBatchPayload,
    *,
    payload_bytes: int,
    received_at_utc: str,
) -> list[int]:
    db_path = Path(str(config["master_db_path"]))
    accepted_buffer_ids: list[int] = []

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA journal_mode=WAL;")
        vector_table, vector_mode = active_hub_vector_storage(conn)
        if vector_mode == "sqlite_vec":
            load_sqlite_vec(conn)

        with conn:
            cursor = conn.execute(
                """
                INSERT INTO ingestion_batches(
                    device_id, sent_at_utc, received_at_utc, payload_bytes,
                    detection_count, status
                )
                VALUES (?, ?, ?, ?, ?, 'accepted');
                """,
                (
                    payload.device_id,
                    payload.sent_at_utc,
                    received_at_utc,
                    payload_bytes,
                    len(payload.detections),
                ),
            )
            batch_id = int(cursor.lastrowid)

            for detection in payload.detections:
                cursor = conn.execute(
                    """
                    INSERT INTO hub_buffer_events(
                        device_id, source_buffer_id, batch_id, timestamp_utc,
                        audio_saved, retention_reason, filepath, max_bio_label,
                        max_bio_logit, noise_logits, max_perch_label,
                        max_perch_logit, nz_bird_logits, gate_mode, gate_threshold,
                        gate_trigger_count, retained_clip_count, margin_gate_scores,
                        received_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        payload.device_id,
                        detection.buffer_id,
                        batch_id,
                        detection.timestamp_utc,
                        int(detection.audio_saved),
                        detection.retention_reason,
                        detection.filepath,
                        detection.max_bio_label,
                        detection.max_bio_logit,
                        detection.noise_logits,
                        detection.max_perch_label,
                        detection.max_perch_logit,
                        detection.nz_bird_logits,
                        detection.gate_mode,
                        detection.gate_threshold,
                        detection.gate_trigger_count,
                        detection.retained_clip_count,
                        detection.margin_gate_scores,
                        received_at_utc,
                    ),
                )
                hub_buffer_id = int(cursor.lastrowid)

                for clip in detection.retained_audio_clips:
                    conn.execute(
                        """
                        INSERT INTO hub_retained_audio_clips(
                            hub_buffer_id, source_clip_id, retention_index, retention_reason,
                            filepath, start_segment_index, end_segment_index, start_offset_s,
                            end_offset_s, duration_s, triggered_frame_count, received_at_utc
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            hub_buffer_id,
                            clip.source_clip_id,
                            clip.retention_index,
                            clip.retention_reason,
                            clip.filepath,
                            clip.start_segment_index,
                            clip.end_segment_index,
                            clip.start_offset_s,
                            clip.end_offset_s,
                            clip.duration_s,
                            clip.triggered_frame_count,
                            received_at_utc,
                        ),
                    )

                for segment in detection.embedding_segments:
                    cursor = conn.execute(
                        """
                        INSERT INTO hub_embedding_segments(
                            hub_buffer_id, source_embedding_id, segment_index
                        )
                        VALUES (?, ?, ?);
                        """,
                        (
                            hub_buffer_id,
                            segment.normalized_source_embedding_id,
                            segment.segment_index,
                        ),
                    )
                    hub_embedding_id = int(cursor.lastrowid)
                    conn.execute(
                        f"INSERT INTO {vector_table}(hub_embedding_id, embedding) VALUES (?, ?);",
                        (hub_embedding_id, segment.embedding),
                    )

                accepted_buffer_ids.append(detection.buffer_id)

            conn.execute(
                """
                INSERT INTO health_metrics(
                    device_id, timestamp_utc, received_at_utc, cpu_temp_c,
                    cpu_load_pct, disk_free_gb, battery_voltage, solar_amps
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    payload.device_id,
                    payload.telemetry.timestamp_utc,
                    received_at_utc,
                    payload.telemetry.cpu_temp_c,
                    payload.telemetry.cpu_load_pct,
                    payload.telemetry.disk_free_gb,
                    payload.telemetry.battery_voltage,
                    payload.telemetry.solar_amps,
                ),
            )

    return accepted_buffer_ids


def active_hub_vector_storage(conn: sqlite3.Connection) -> tuple[str, str]:
    metadata = dict(conn.execute("SELECT key, value FROM schema_metadata;").fetchall())
    table = metadata.get("vector_table")
    mode = metadata.get("vector_storage_mode", "blob")
    if table:
        return table, mode
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'virtual table');"
        )
    }
    if "hub_perch_vectors" in tables:
        return "hub_perch_vectors", "sqlite_vec"
    return "hub_perch_vector_blobs", "blob"


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec  # type: ignore

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def utc_now_string() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


app = create_app()
