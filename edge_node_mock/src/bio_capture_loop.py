"""Mock edge capture loop: audio fixtures, Perch inference, gating, and DB writes."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mock_common.config import load_config

from edge_node_mock.src.init_edge_db import init_edge_db
from edge_node_mock.src.inspect_perch_model import (
    _load_model,
    _pick_output,
    _serving_signature,
    load_nz_bird_labels,
    load_perch_labels,
    make_perch_windows,
)


@dataclass(frozen=True)
class AudioBuffer:
    source_file: Path
    file_buffer_index: int
    timestamp_utc: datetime
    source_audio: Any
    source_sample_rate: int
    perch_audio: Any
    perch_sample_rate: int


@dataclass(frozen=True)
class FrameScores:
    segment_index: int
    max_noise_label: str | None
    max_noise_logit: float | None
    max_bio_label: str | None
    max_bio_logit: float | None
    max_perch_label: str
    max_perch_logit: float
    top_nz_birds: list[dict[str, Any]]


@dataclass(frozen=True)
class BufferDecision:
    retention_reason: str
    audio_saved: bool
    filepath: str | None
    max_bio_label: str | None
    max_bio_logit: float | None
    max_perch_label: str
    max_perch_logit: float
    noise_logits: str
    nz_bird_logits: str


def iter_audio_buffers(
    config: dict[str, Any],
    *,
    seconds: float = 15.0,
    include_partial: bool = False,
) -> Iterator[AudioBuffer]:
    """Yield 15-second mono buffers from the configured raw audio stream."""

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    raw_audio_root = Path(str(config["raw_audio_mount"]))
    raw_audio_glob = str(config.get("raw_audio_glob", "*.wav"))
    perch_sample_rate = int(config.get("perch_sample_rate", 32000))
    audio_files = sorted(path for path in raw_audio_root.glob(raw_audio_glob) if path.is_file())
    if not audio_files:
        raise FileNotFoundError(f"No audio files found at {raw_audio_root}/{raw_audio_glob}")

    for audio_file in audio_files:
        info = sf.info(audio_file)
        source_sample_rate = int(info.samplerate)
        block_size = int(round(source_sample_rate * seconds))
        base_timestamp = timestamp_from_filename(audio_file)

        for file_buffer_index, block in enumerate(
            sf.blocks(audio_file, blocksize=block_size, dtype="float32", always_2d=True)
        ):
            if len(block) < block_size:
                if not include_partial:
                    continue
                block = np.pad(block, ((0, block_size - len(block)), (0, 0)))
            source_mono = np.mean(block, axis=1, dtype=np.float32)
            if source_sample_rate == perch_sample_rate:
                perch_audio = source_mono
            elif source_sample_rate == 48000 and perch_sample_rate == 32000:
                perch_audio = resample_poly(source_mono, up=2, down=3).astype(np.float32)
            else:
                raise ValueError(
                    f"Unsupported sample-rate conversion: {source_sample_rate} Hz to {perch_sample_rate} Hz"
                )

            expected_samples = int(round(perch_sample_rate * seconds))
            if len(perch_audio) != expected_samples:
                perch_audio = _fit_length(perch_audio, expected_samples)

            yield AudioBuffer(
                source_file=audio_file,
                file_buffer_index=file_buffer_index,
                timestamp_utc=base_timestamp + timedelta(seconds=seconds * file_buffer_index),
                source_audio=source_mono,
                source_sample_rate=source_sample_rate,
                perch_audio=perch_audio,
                perch_sample_rate=perch_sample_rate,
            )


def timestamp_from_filename(path: Path) -> datetime:
    """Parse AR4-style YYYYMMDD_HHMMSS filenames as UTC for Phase 1 mock rows."""

    try:
        return datetime.strptime(path.stem, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return datetime.now(UTC)


def _fit_length(audio: Any, expected_samples: int) -> Any:
    import numpy as np

    if len(audio) < expected_samples:
        return np.pad(audio, (0, expected_samples - len(audio))).astype(np.float32)
    return np.asarray(audio[:expected_samples], dtype=np.float32)


def build_label_index(labels: list[str], configured_labels: list[str], *, group_name: str) -> dict[str, int]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    missing = [label for label in configured_labels if label not in label_to_index]
    if missing:
        raise ValueError(f"{group_name} contains labels not present in Perch labels: {missing}")
    return {label: label_to_index[label] for label in configured_labels}


def run_perch_inference(model: Any, perch_audio: Any, sample_rate: int) -> tuple[Any, Any]:
    """Return Perch logits and embeddings for three 5-second windows."""

    import tensorflow as tf

    windows = make_perch_windows(perch_audio, sample_rate)
    signature = _serving_signature(model)
    outputs = signature(inputs=tf.convert_to_tensor(windows, dtype=tf.float32))
    logits = _pick_output(outputs, ("label", "logits")).numpy()
    embeddings = _pick_output(outputs, ("embedding", "embeddings")).numpy()
    return logits, embeddings


def score_frames(
    logits: Any,
    *,
    perch_labels: list[str],
    noise_label_indexes: dict[str, int],
    bio_label_indexes: dict[str, int],
    nz_label_indexes: dict[int, Any],
) -> list[FrameScores]:
    import numpy as np

    frame_scores: list[FrameScores] = []
    nz_indexes = list(nz_label_indexes)
    for segment_index, frame_logits in enumerate(logits):
        max_perch_index = int(np.argmax(frame_logits))
        max_noise_label, max_noise_logit = _max_configured_label(frame_logits, noise_label_indexes)
        max_bio_label, max_bio_logit = _max_configured_label(frame_logits, bio_label_indexes)
        frame_scores.append(
            FrameScores(
                segment_index=segment_index,
                max_noise_label=max_noise_label,
                max_noise_logit=max_noise_logit,
                max_bio_label=max_bio_label,
                max_bio_logit=max_bio_logit,
                max_perch_label=perch_labels[max_perch_index],
                max_perch_logit=float(frame_logits[max_perch_index]),
                top_nz_birds=_top_nz_birds(frame_logits, nz_indexes, nz_label_indexes),
            )
        )
    return frame_scores


def _max_configured_label(frame_logits: Any, label_indexes: dict[str, int]) -> tuple[str | None, float | None]:
    if not label_indexes:
        return None, None
    best_label = max(label_indexes, key=lambda label: float(frame_logits[label_indexes[label]]))
    return best_label, float(frame_logits[label_indexes[best_label]])


def _top_nz_birds(frame_logits: Any, nz_indexes: list[int], nz_label_map: dict[int, Any]) -> list[dict[str, Any]]:
    scored = sorted(nz_indexes, key=lambda index: float(frame_logits[index]), reverse=True)[:3]
    return [
        {
            "perch_label_number": index,
            "common_name": nz_label_map[index].common_name,
            "scientific_name": nz_label_map[index].scientific_name,
            "logit": float(frame_logits[index]),
        }
        for index in scored
    ]


def decide_buffer(
    frame_scores: list[FrameScores],
    *,
    bio_threshold: float,
    noise_threshold: float,
    validation_sample_interval: int,
    dropped_buffer_count: int,
) -> BufferDecision:
    max_bio_frame = max(
        frame_scores,
        key=lambda score: float("-inf") if score.max_bio_logit is None else score.max_bio_logit,
    )
    max_perch_frame = max(frame_scores, key=lambda score: score.max_perch_logit)
    noise_payload = [
        {
            "segment_index": score.segment_index,
            "max_noise_label": score.max_noise_label,
            "max_noise_logit": score.max_noise_logit,
            "noise_dominates": bool(
                score.max_noise_logit is not None
                and score.max_noise_logit >= noise_threshold
                and (score.max_bio_logit is None or score.max_noise_logit > score.max_bio_logit)
            ),
        }
        for score in frame_scores
    ]
    nz_payload = [
        {"segment_index": score.segment_index, "top_3": score.top_nz_birds}
        for score in frame_scores
    ]

    is_bio_hit = max_bio_frame.max_bio_logit is not None and max_bio_frame.max_bio_logit >= bio_threshold
    if is_bio_hit:
        retention_reason = "bio_hit"
    elif validation_sample_interval > 0 and (dropped_buffer_count + 1) % validation_sample_interval == 0:
        retention_reason = "validation_sample"
    else:
        retention_reason = "dropped"

    return BufferDecision(
        retention_reason=retention_reason,
        audio_saved=retention_reason in {"bio_hit", "validation_sample"},
        filepath=None,
        max_bio_label=max_bio_frame.max_bio_label,
        max_bio_logit=max_bio_frame.max_bio_logit,
        max_perch_label=max_perch_frame.max_perch_label,
        max_perch_logit=max_perch_frame.max_perch_logit,
        noise_logits=json.dumps(noise_payload, separators=(",", ":")),
        nz_bird_logits=json.dumps(nz_payload, separators=(",", ":")),
    )


def save_retained_audio(buffer: AudioBuffer, retained_audio_dir: Path, device_id: str, reason: str) -> str:
    import soundfile as sf

    retained_audio_dir.mkdir(parents=True, exist_ok=True)
    timestamp = buffer.timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
    path = retained_audio_dir / f"{device_id}_{timestamp}_{reason}_{buffer.source_file.stem}_{buffer.file_buffer_index:03d}.flac"
    sf.write(path, buffer.source_audio, buffer.source_sample_rate, format="FLAC")
    return str(path)


def insert_buffer_event(
    conn: sqlite3.Connection,
    *,
    config: dict[str, Any],
    buffer: AudioBuffer,
    decision: BufferDecision,
    embeddings: Any,
) -> int:
    import numpy as np

    created_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    timestamp = buffer.timestamp_utc.isoformat().replace("+00:00", "Z")
    cursor = conn.execute(
        """
        INSERT INTO buffer_events(
            device_id, timestamp_utc, audio_saved, retention_reason, filepath,
            max_bio_label, max_bio_logit, noise_logits, max_perch_label,
            max_perch_logit, nz_bird_logits, sync_status, created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?);
        """,
        (
            config["device_id"],
            timestamp,
            int(decision.audio_saved),
            decision.retention_reason,
            decision.filepath,
            decision.max_bio_label,
            decision.max_bio_logit,
            decision.noise_logits,
            decision.max_perch_label,
            decision.max_perch_logit,
            decision.nz_bird_logits,
            created_at,
        ),
    )
    buffer_id = int(cursor.lastrowid)
    vector_table, vector_mode = active_vector_storage(conn)
    if vector_mode == "sqlite_vec":
        load_sqlite_vec(conn)
    for segment_index, embedding in enumerate(embeddings):
        cursor = conn.execute(
            "INSERT INTO embedding_segments(buffer_id, segment_index) VALUES (?, ?);",
            (buffer_id, segment_index),
        )
        embedding_id = int(cursor.lastrowid)
        embedding_blob = np.asarray(embedding, dtype=np.float32).tobytes()
        conn.execute(
            f"INSERT INTO {vector_table}(embedding_id, embedding) VALUES (?, ?);",
            (embedding_id, embedding_blob),
        )
    return buffer_id


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec  # type: ignore

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def active_vector_storage(conn: sqlite3.Connection) -> tuple[str, str]:
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
    if "perch_vectors" in tables:
        return "perch_vectors", "sqlite_vec"
    return "perch_vector_blobs", "blob"


def run_capture_loop(config_path: str | Path, *, iterations: int | None = None) -> dict[str, int]:
    config = load_config(config_path)
    db_path = init_edge_db(config_path)
    perch_labels = load_perch_labels(config["perch_label_path"])
    nz_labels = load_nz_bird_labels(config["nz_bird_label_path"], perch_labels)
    noise_label_indexes = build_label_index(
        perch_labels,
        list(config.get("noise_labels", [])),
        group_name="noise_labels",
    )
    bio_label_indexes = build_label_index(
        perch_labels,
        list(config.get("biological_labels", [])),
        group_name="biological_labels",
    )
    model, _, _ = _load_model(config)

    retained_audio_dir = Path(str(config["retained_audio_dir"]))
    processed = 0
    retained = 0
    bio_hits = 0
    validation_samples = 0
    dropped_buffer_count = 0

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        for buffer in iter_audio_buffers(config):
            if iterations is not None and processed >= iterations:
                break
            logits, embeddings = run_perch_inference(model, buffer.perch_audio, buffer.perch_sample_rate)
            frame_scores = score_frames(
                logits,
                perch_labels=perch_labels,
                noise_label_indexes=noise_label_indexes,
                bio_label_indexes=bio_label_indexes,
                nz_label_indexes=nz_labels,
            )
            decision = decide_buffer(
                frame_scores,
                bio_threshold=float(config["bio_threshold"]),
                noise_threshold=float(config["noise_threshold"]),
                validation_sample_interval=int(config["validation_sample_interval"]),
                dropped_buffer_count=dropped_buffer_count,
            )
            if decision.retention_reason == "dropped":
                dropped_buffer_count += 1
            elif decision.retention_reason == "bio_hit":
                bio_hits += 1
            elif decision.retention_reason == "validation_sample":
                validation_samples += 1

            if decision.audio_saved:
                filepath = save_retained_audio(
                    buffer,
                    retained_audio_dir,
                    str(config["device_id"]),
                    decision.retention_reason,
                )
                decision = BufferDecision(**{**decision.__dict__, "filepath": filepath})
                retained += 1

            insert_buffer_event(conn, config=config, buffer=buffer, decision=decision, embeddings=embeddings)
            conn.commit()
            processed += 1
            print(
                f"buffer={processed} source={buffer.source_file.name} "
                f"segment={buffer.file_buffer_index} decision={decision.retention_reason} "
                f"max_bio={decision.max_bio_label}:{decision.max_bio_logit:.3f}"
            )

    return {
        "processed": processed,
        "retained": retained,
        "bio_hits": bio_hits,
        "validation_samples": validation_samples,
        "dropped": processed - bio_hits - validation_samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="edge_node_mock/config/edge_config.example.yaml")
    parser.add_argument("--iterations", type=int, default=None, help="Process only this many 15-second buffers.")
    args = parser.parse_args(argv)

    summary = run_capture_loop(args.config, iterations=args.iterations)
    print("Capture loop complete")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
