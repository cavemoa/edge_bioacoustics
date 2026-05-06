"""Mock edge sender daemon for Phase 1 MessagePack sync."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgpack
import psutil
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_node_mock.src.bio_capture_loop import active_vector_storage, load_sqlite_vec
from mock_common.config import load_config


@dataclass(frozen=True)
class SenderResult:
    status: str
    pending_count: int
    detection_count: int
    payload_bytes: int
    accepted_buffer_ids: list[int]


def utc_now_string() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def load_pending_detections(config: dict[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load pending edge rows with their three embedding segments and vectors."""

    db_path = Path(str(config["edge_db_path"]))
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON;")
        vector_table, vector_mode = active_vector_storage(conn)
        if vector_mode == "sqlite_vec":
            load_sqlite_vec(conn)

        sql = """
            SELECT *
            FROM buffer_events
            WHERE sync_status = 'pending'
            ORDER BY buffer_id
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)

        detections = []
        for row in conn.execute(sql, params).fetchall():
            buffer_id = int(row["buffer_id"])
            segments = conn.execute(
                f"""
                SELECT es.embedding_id, es.segment_index, v.embedding
                FROM embedding_segments AS es
                JOIN {vector_table} AS v ON v.embedding_id = es.embedding_id
                WHERE es.buffer_id = ?
                ORDER BY es.segment_index;
                """,
                (buffer_id,),
            ).fetchall()
            if len(segments) != 3:
                raise ValueError(f"Buffer {buffer_id} has {len(segments)} embedding segments, expected 3")

            detections.append(
                {
                    "buffer_id": buffer_id,
                    "timestamp_utc": row["timestamp_utc"],
                    "audio_saved": int(row["audio_saved"]),
                    "retention_reason": row["retention_reason"],
                    "filepath": row["filepath"],
                    "max_bio_label": row["max_bio_label"],
                    "max_bio_logit": row["max_bio_logit"],
                    "noise_logits": row["noise_logits"],
                    "max_perch_label": row["max_perch_label"],
                    "max_perch_logit": row["max_perch_logit"],
                    "nz_bird_logits": row["nz_bird_logits"],
                    "embedding_segments": [
                        {
                            "embedding_id": int(segment["embedding_id"]),
                            "segment_index": int(segment["segment_index"]),
                            "embedding": bytes(segment["embedding"]),
                        }
                        for segment in segments
                    ],
                }
            )

    return detections


def gather_telemetry(config: dict[str, Any]) -> dict[str, float | str | None]:
    """Gather desktop-safe mock telemetry for Phase 1."""

    disk_usage = psutil.disk_usage(Path(str(config["edge_db_path"])).parent)
    defaults = config.get("telemetry_defaults", {}) or {}
    return {
        "timestamp_utc": utc_now_string(),
        "cpu_temp_c": float(defaults.get("cpu_temp_c", 45.0)),
        "cpu_load_pct": float(psutil.cpu_percent(interval=0.0)),
        "disk_free_gb": round(disk_usage.free / (1024**3), 3),
        "battery_voltage": float(defaults.get("battery_voltage", 12.4)),
        "solar_amps": float(defaults.get("solar_amps", 0.0)),
    }


def build_payload(config: dict[str, Any], detections: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "device_id": config["device_id"],
        "sent_at_utc": utc_now_string(),
        "detections": detections,
        "telemetry": gather_telemetry(config),
    }


def pack_payload(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, use_bin_type=True)


def mark_buffers_synced(config: dict[str, Any], accepted_buffer_ids: list[int]) -> None:
    if not accepted_buffer_ids:
        return

    db_path = Path(str(config["edge_db_path"]))
    synced_at = utc_now_string()
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        with conn:
            for buffer_id in accepted_buffer_ids:
                conn.execute(
                    """
                    UPDATE buffer_events
                    SET sync_status = 'synced', synced_at_utc = ?
                    WHERE buffer_id = ? AND sync_status = 'pending';
                    """,
                    (synced_at, buffer_id),
                )


def send_pending_batch(
    config_path: str | Path,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    timeout_seconds: float = 30.0,
) -> SenderResult:
    config = load_config(config_path)
    pending_count = count_pending(config)
    detections = load_pending_detections(config, limit=limit)
    payload = build_payload(config, detections)
    packed = pack_payload(payload)

    if dry_run:
        return SenderResult(
            status="dry_run",
            pending_count=pending_count,
            detection_count=len(detections),
            payload_bytes=len(packed),
            accepted_buffer_ids=[],
        )

    if not detections:
        return SenderResult(
            status="no_pending",
            pending_count=pending_count,
            detection_count=0,
            payload_bytes=len(packed),
            accepted_buffer_ids=[],
        )

    try:
        response = requests.post(
            str(config["hub_ingest_url"]),
            data=packed,
            headers={"X-API-Key": str(config["api_key"]), "Content-Type": "application/msgpack"},
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        print(f"Sender failed before hub acknowledgement: {exc}", file=sys.stderr)
        return SenderResult(
            status="failed",
            pending_count=pending_count,
            detection_count=len(detections),
            payload_bytes=len(packed),
            accepted_buffer_ids=[],
        )

    if response.status_code != 200:
        print(f"Hub rejected payload with HTTP {response.status_code}: {response.text}", file=sys.stderr)
        return SenderResult(
            status="failed",
            pending_count=pending_count,
            detection_count=len(detections),
            payload_bytes=len(packed),
            accepted_buffer_ids=[],
        )

    data = response.json()
    accepted_buffer_ids = [int(buffer_id) for buffer_id in data.get("accepted_buffer_ids", [])]
    mark_buffers_synced(config, accepted_buffer_ids)
    return SenderResult(
        status="sent",
        pending_count=pending_count,
        detection_count=len(detections),
        payload_bytes=len(packed),
        accepted_buffer_ids=accepted_buffer_ids,
    )


def count_pending(config: dict[str, Any]) -> int:
    with sqlite3.connect(Path(str(config["edge_db_path"]))) as conn:
        return int(conn.execute("SELECT COUNT(*) FROM buffer_events WHERE sync_status = 'pending';").fetchone()[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="edge_node_mock/config/edge_config.example.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Send at most this many pending buffers.")
    parser.add_argument("--dry-run", action="store_true", help="Build and summarize the payload without sending it.")
    args = parser.parse_args(argv)

    result = send_pending_batch(args.config, limit=args.limit, dry_run=args.dry_run)
    print(f"status: {result.status}")
    print(f"pending_count: {result.pending_count}")
    print(f"detection_count: {result.detection_count}")
    print(f"payload_bytes: {result.payload_bytes}")
    print(f"accepted_buffer_ids: {result.accepted_buffer_ids}")
    return 0 if result.status in {"sent", "dry_run", "no_pending"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
