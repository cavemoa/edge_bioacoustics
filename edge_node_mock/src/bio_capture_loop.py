"""Mock edge capture loop: audio fixtures, Perch inference, gating, and DB writes."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mock_common.config import load_config

from edge_node_mock.src.gate_config import validate_margin_gate_config
from edge_node_mock.src.gate_logic import (
    FrameScores,
    RetentionClip,
    build_margin_label_indexes,
    build_variable_retention_buffers,
    margin_gate_scores_payload,
    retained_clip_filename,
    score_frames,
)
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
class BufferDecision:
    retention_reason: str
    audio_saved: bool
    max_nz_bird_common_name: str | None
    max_nz_bird_scientific_name: str | None
    max_nz_bird_logit: float | None
    max_perch_label: str
    max_perch_logit: float
    excluded_label_scores: str
    nz_bird_logits: str
    gate_mode: str = "nz_bird_margin"
    gate_threshold: float | None = None
    gate_trigger_count: int = 0
    retained_clip_count: int = 0
    margin_gate_scores: str = "[]"
    retained_clips: list[RetentionClip] = field(default_factory=list)


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


def run_perch_inference(model: Any, perch_audio: Any, sample_rate: int) -> tuple[Any, Any]:
    """Return Perch logits and embeddings for three 5-second windows."""

    import tensorflow as tf

    windows = make_perch_windows(perch_audio, sample_rate)
    signature = _serving_signature(model)
    outputs = signature(inputs=tf.convert_to_tensor(windows, dtype=tf.float32))
    logits = _pick_output(outputs, ("label", "logits")).numpy()
    embeddings = _pick_output(outputs, ("embedding", "embeddings")).numpy()
    return logits, embeddings


def decide_buffer(
    frame_scores: list[FrameScores],
    *,
    gate_threshold: float = 0.55,
    gate_mode: str = "nz_bird_margin",
    max_variable_buffer_frames: int = 3,
    perch_window_seconds: float = 5.0,
) -> BufferDecision:
    max_nz_frame = max(
        frame_scores,
        key=lambda score: float("-inf") if score.top_nz_logit is None else score.top_nz_logit,
    )
    max_perch_frame = max(frame_scores, key=lambda score: score.max_perch_logit)
    noise_payload = [
        {
            "segment_index": score.segment_index,
            "max_noise_label": score.max_noise_label,
            "max_noise_logit": score.max_noise_logit,
            "top_excluded_label": getattr(score, "top_excluded_label", score.max_noise_label),
            "top_excluded_logit": getattr(score, "top_excluded_logit", score.max_noise_logit),
            "noise_dominates": bool(getattr(score, "excluded_top_label_gate", False)),
        }
        for score in frame_scores
    ]
    nz_payload = [
        {"segment_index": score.segment_index, "top_3": score.top_nz_birds}
        for score in frame_scores
    ]

    retained_clips = build_variable_retention_buffers(
        frame_scores,
        max_frames=max_variable_buffer_frames,
        perch_window_seconds=perch_window_seconds,
    )
    trigger_count = sum(1 for score in frame_scores if bool(getattr(score, "frame_bio_gate", False)))
    is_bio_hit = bool(retained_clips)
    if is_bio_hit:
        retention_reason = "bio_hit"
    else:
        retention_reason = "dropped"

    return BufferDecision(
        retention_reason=retention_reason,
        audio_saved=bool(retained_clips),
        max_nz_bird_common_name=max_nz_frame.top_nz_common_name,
        max_nz_bird_scientific_name=max_nz_frame.top_nz_scientific_name,
        max_nz_bird_logit=max_nz_frame.top_nz_logit,
        max_perch_label=max_perch_frame.max_perch_label,
        max_perch_logit=max_perch_frame.max_perch_logit,
        excluded_label_scores=json.dumps(noise_payload, separators=(",", ":")),
        nz_bird_logits=json.dumps(nz_payload, separators=(",", ":")),
        gate_mode=gate_mode,
        gate_threshold=gate_threshold,
        gate_trigger_count=trigger_count,
        retained_clip_count=len(retained_clips),
        margin_gate_scores=json.dumps(margin_gate_scores_payload(frame_scores), separators=(",", ":")),
        retained_clips=retained_clips,
    )


def save_retained_audio(
    buffer: AudioBuffer,
    retained_audio_dir: Path,
    device_id: str,
    retention_clip: RetentionClip,
) -> str:
    import soundfile as sf

    retained_audio_dir.mkdir(parents=True, exist_ok=True)
    start_sample = int(round(retention_clip.start_offset_s * buffer.source_sample_rate))
    end_sample = int(round(retention_clip.end_offset_s * buffer.source_sample_rate))
    audio = buffer.source_audio[start_sample:end_sample]
    filename = retained_clip_filename(
        device_id=device_id,
        timestamp_utc=buffer.timestamp_utc,
        source_stem=buffer.source_file.stem,
        file_buffer_index=buffer.file_buffer_index,
        clip=retention_clip,
    )
    path = retained_audio_dir / filename
    sf.write(path, audio, buffer.source_sample_rate, format="FLAC")
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
    perch_window_seconds = float(config.get("perch_window_seconds", 5.0))
    cursor = conn.execute(
        """
        INSERT INTO buffer_events(
            event_uuid, device_id, source_file, file_buffer_index, timestamp_utc,
            inference_buffer_seconds, perch_window_seconds, perch_frame_count,
            audio_saved, retention_reason, max_nz_bird_common_name,
            max_nz_bird_scientific_name, max_nz_bird_logit, max_perch_label,
            max_perch_logit, excluded_label_scores, nz_bird_logits, gate_mode, gate_threshold,
            gate_trigger_count, retained_clip_count, margin_gate_scores,
            sync_status, created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?);
        """,
        (
            str(uuid4()),
            config["device_id"],
            str(buffer.source_file),
            buffer.file_buffer_index,
            timestamp,
            len(buffer.source_audio) / float(buffer.source_sample_rate),
            perch_window_seconds,
            len(embeddings),
            int(decision.audio_saved),
            decision.retention_reason,
            decision.max_nz_bird_common_name,
            decision.max_nz_bird_scientific_name,
            decision.max_nz_bird_logit,
            decision.max_perch_label,
            decision.max_perch_logit,
            decision.excluded_label_scores,
            decision.nz_bird_logits,
            decision.gate_mode,
            decision.gate_threshold,
            decision.gate_trigger_count,
            decision.retained_clip_count,
            decision.margin_gate_scores,
            created_at,
        ),
    )
    buffer_id = int(cursor.lastrowid)
    for clip in decision.retained_clips:
        if not clip.filepath:
            continue
        conn.execute(
            """
            INSERT INTO retained_audio_clips(
                buffer_id, retention_index, retention_reason, filepath,
                start_segment_index, end_segment_index, start_offset_s, end_offset_s,
                duration_s, triggered_frame_count, created_at_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                buffer_id,
                clip.retention_index,
                clip.retention_reason,
                clip.filepath,
                clip.start_segment_index,
                clip.end_segment_index,
                clip.start_offset_s,
                clip.end_offset_s,
                clip.duration_s,
                clip.triggered_frame_count,
                created_at,
            ),
        )
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
    gate_config = validate_margin_gate_config(config, perch_labels=perch_labels)
    nz_labels = load_nz_bird_labels(config["nz_bird_label_path"], perch_labels)
    excluded_margin_label_indexes = build_margin_label_indexes(
        perch_labels,
        gate_config.excluded_margin_labels,
    )
    model, _, _ = _load_model(config)

    retained_audio_dir = Path(str(config["retained_audio_dir"]))
    processed = 0
    retained = 0
    bio_hits = 0

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        for buffer in iter_audio_buffers(
            config,
            include_partial=bool(config.get("include_partial_final_buffer", False)),
        ):
            if iterations is not None and processed >= iterations:
                break
            logits, embeddings = run_perch_inference(model, buffer.perch_audio, buffer.perch_sample_rate)
            frame_scores = score_frames(
                logits,
                perch_labels=perch_labels,
                nz_label_indexes=nz_labels,
                excluded_margin_label_indexes=excluded_margin_label_indexes,
                bio_margin_threshold=gate_config.bio_margin_threshold,
            )
            decision = decide_buffer(
                frame_scores,
                gate_threshold=gate_config.bio_margin_threshold,
                gate_mode=gate_config.bio_gate_mode,
                max_variable_buffer_frames=gate_config.max_variable_buffer_frames,
                perch_window_seconds=gate_config.perch_window_seconds,
            )
            if decision.retention_reason == "bio_hit":
                bio_hits += 1

            if decision.audio_saved:
                saved_clips: list[RetentionClip] = []
                for clip in decision.retained_clips:
                    filepath = save_retained_audio(
                        buffer,
                        retained_audio_dir,
                        str(config["device_id"]),
                        clip,
                    )
                    saved_clips.append(replace(clip, filepath=filepath))
                decision = replace(
                    decision,
                    retained_clips=saved_clips,
                    retained_clip_count=len(saved_clips),
                    audio_saved=bool(saved_clips),
                )
                retained += len(saved_clips)

            insert_buffer_event(conn, config=config, buffer=buffer, decision=decision, embeddings=embeddings)
            conn.commit()
            processed += 1
            max_nz_logit = (
                f"{decision.max_nz_bird_logit:.3f}" if decision.max_nz_bird_logit is not None else "None"
            )
            print(
                f"buffer={processed} source={buffer.source_file.name} "
                f"segment={buffer.file_buffer_index} decision={decision.retention_reason} "
                f"clips={decision.retained_clip_count} max_nz={decision.max_nz_bird_common_name}:{max_nz_logit}"
            )

    return {
        "processed": processed,
        "retained": retained,
        "bio_hits": bio_hits,
        "dropped": processed - bio_hits,
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
