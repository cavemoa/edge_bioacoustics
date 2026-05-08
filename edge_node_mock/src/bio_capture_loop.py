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
    top_nz_label_number: int | None = None
    top_nz_common_name: str | None = None
    top_nz_scientific_name: str | None = None
    top_nz_perch_label: str | None = None
    top_nz_logit: float | None = None
    top_excluded_label: str | None = None
    top_excluded_logit: float | None = None
    nz_over_excluded_margin: float | None = None
    bio_margin_threshold: float | None = None
    excluded_top_label_gate: bool = False
    margin_gate: bool = False
    frame_bio_gate: bool = False


@dataclass(frozen=True)
class RetentionClip:
    retention_index: int
    retention_reason: str
    filepath: str | None
    start_segment_index: int
    end_segment_index: int
    start_offset_s: float
    end_offset_s: float
    duration_s: float
    triggered_frame_count: int
    clip_id: int | None = None


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


def build_label_index(labels: list[str], configured_labels: list[str], *, group_name: str) -> dict[str, int]:
    label_to_index = {label: index for index, label in enumerate(labels)}
    missing = [label for label in configured_labels if label not in label_to_index]
    if missing:
        raise ValueError(f"{group_name} contains labels not present in Perch labels: {missing}")
    return {label: label_to_index[label] for label in configured_labels}


def build_margin_label_indexes(perch_labels: list[str], excluded_margin_labels: list[str]) -> dict[str, int]:
    return build_label_index(
        perch_labels,
        excluded_margin_labels,
        group_name="excluded_margin_labels",
    )


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
    noise_label_indexes: dict[str, int] | None = None,
    bio_label_indexes: dict[str, int] | None = None,
    nz_label_indexes: dict[int, Any],
    excluded_margin_label_indexes: dict[str, int] | None = None,
    bio_margin_threshold: float = 0.55,
) -> list[FrameScores]:
    import numpy as np

    frame_scores: list[FrameScores] = []
    nz_indexes = list(nz_label_indexes)
    noise_label_indexes = noise_label_indexes or {}
    bio_label_indexes = bio_label_indexes or {}
    excluded_margin_label_indexes = excluded_margin_label_indexes or noise_label_indexes
    for segment_index, frame_logits in enumerate(logits):
        max_perch_index = int(np.argmax(frame_logits))
        max_perch_label = perch_labels[max_perch_index]
        max_perch_logit = float(frame_logits[max_perch_index])
        top_excluded_label, top_excluded_logit = _max_configured_label(
            frame_logits,
            excluded_margin_label_indexes,
        )
        legacy_max_bio_label, legacy_max_bio_logit = _max_configured_label(frame_logits, bio_label_indexes)
        top_nz_index, top_nz = _top_nz_bird(frame_logits, nz_indexes, nz_label_indexes)
        top_nz_logit = float(frame_logits[top_nz_index]) if top_nz_index is not None else None
        top_nz_common_name = getattr(top_nz, "common_name", None) if top_nz is not None else None
        top_nz_scientific_name = getattr(top_nz, "scientific_name", None) if top_nz is not None else None
        top_nz_perch_label = (
            getattr(top_nz, "perch_label", perch_labels[top_nz_index])
            if top_nz_index is not None and top_nz is not None
            else None
        )
        margin = (
            top_nz_logit - top_excluded_logit
            if top_nz_logit is not None and top_excluded_logit is not None
            else None
        )
        excluded_top_label_gate = max_perch_label in excluded_margin_label_indexes
        margin_gate = bool(margin is not None and margin >= bio_margin_threshold)
        frame_bio_gate = bool((not excluded_top_label_gate) and margin_gate)
        max_bio_label = top_nz_common_name or legacy_max_bio_label
        max_bio_logit = top_nz_logit if top_nz_logit is not None else legacy_max_bio_logit
        frame_scores.append(
            FrameScores(
                segment_index=segment_index,
                max_noise_label=top_excluded_label,
                max_noise_logit=top_excluded_logit,
                max_bio_label=max_bio_label,
                max_bio_logit=max_bio_logit,
                max_perch_label=max_perch_label,
                max_perch_logit=max_perch_logit,
                top_nz_birds=_top_nz_birds(frame_logits, nz_indexes, nz_label_indexes),
                top_nz_label_number=top_nz_index,
                top_nz_common_name=top_nz_common_name,
                top_nz_scientific_name=top_nz_scientific_name,
                top_nz_perch_label=top_nz_perch_label,
                top_nz_logit=top_nz_logit,
                top_excluded_label=top_excluded_label,
                top_excluded_logit=top_excluded_logit,
                nz_over_excluded_margin=margin,
                bio_margin_threshold=bio_margin_threshold,
                excluded_top_label_gate=excluded_top_label_gate,
                margin_gate=margin_gate,
                frame_bio_gate=frame_bio_gate,
            )
        )
    return frame_scores


def _max_configured_label(frame_logits: Any, label_indexes: dict[str, int]) -> tuple[str | None, float | None]:
    if not label_indexes:
        return None, None
    best_label = max(label_indexes, key=lambda label: float(frame_logits[label_indexes[label]]))
    return best_label, float(frame_logits[label_indexes[best_label]])


def _top_nz_bird(frame_logits: Any, nz_indexes: list[int], nz_label_map: dict[int, Any]) -> tuple[int | None, Any | None]:
    if not nz_indexes:
        return None, None
    best_index = max(nz_indexes, key=lambda index: float(frame_logits[index]))
    return best_index, nz_label_map[best_index]


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


def margin_gate_scores_payload(frame_scores: list[FrameScores]) -> list[dict[str, Any]]:
    return [
        {
            "segment_index": score.segment_index,
            "top_nz_label_number": getattr(score, "top_nz_label_number", None),
            "top_nz_common_name": getattr(score, "top_nz_common_name", None),
            "top_nz_scientific_name": getattr(score, "top_nz_scientific_name", None),
            "top_nz_perch_label": getattr(score, "top_nz_perch_label", None),
            "top_nz_logit": getattr(score, "top_nz_logit", None),
            "top_excluded_label": getattr(score, "top_excluded_label", score.max_noise_label),
            "top_excluded_logit": getattr(score, "top_excluded_logit", score.max_noise_logit),
            "nz_over_excluded_margin": getattr(score, "nz_over_excluded_margin", None),
            "bio_margin_threshold": getattr(score, "bio_margin_threshold", None),
            "excluded_top_label_gate": getattr(score, "excluded_top_label_gate", False),
            "margin_gate": getattr(score, "margin_gate", False),
            "frame_bio_gate": getattr(score, "frame_bio_gate", False),
        }
        for score in frame_scores
    ]


def build_variable_retention_buffers(
    frame_scores: list[FrameScores],
    *,
    max_frames: int = 3,
    perch_window_seconds: float = 5.0,
) -> list[RetentionClip]:
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    clips: list[RetentionClip] = []
    active_run: list[FrameScores] = []

    def flush_active_run() -> None:
        nonlocal active_run
        if not active_run:
            return
        for chunk_start in range(0, len(active_run), max_frames):
            chunk = active_run[chunk_start : chunk_start + max_frames]
            start_segment = int(chunk[0].segment_index)
            end_segment = int(chunk[-1].segment_index)
            clips.append(
                RetentionClip(
                    retention_index=len(clips) + 1,
                    retention_reason="bio_hit",
                    filepath=None,
                    start_segment_index=start_segment,
                    end_segment_index=end_segment,
                    start_offset_s=start_segment * perch_window_seconds,
                    end_offset_s=(end_segment + 1) * perch_window_seconds,
                    duration_s=(end_segment - start_segment + 1) * perch_window_seconds,
                    triggered_frame_count=len(chunk),
                )
            )
        active_run = []

    for score in frame_scores:
        if bool(getattr(score, "frame_bio_gate", False)):
            active_run.append(score)
        else:
            flush_active_run()
    flush_active_run()
    return clips


def decide_buffer(
    frame_scores: list[FrameScores],
    *,
    bio_threshold: float = 0.55,
    noise_threshold: float = 0.0,
    validation_sample_interval: int,
    dropped_buffer_count: int,
    gate_mode: str = "nz_bird_margin",
    max_variable_buffer_frames: int = 3,
    perch_window_seconds: float = 5.0,
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
    elif (
        gate_mode != "nz_bird_margin"
        and validation_sample_interval > 0
        and (dropped_buffer_count + 1) % validation_sample_interval == 0
    ):
        retention_reason = "validation_sample"
    else:
        retention_reason = "dropped"

    return BufferDecision(
        retention_reason=retention_reason,
        audio_saved=bool(retained_clips) or retention_reason == "validation_sample",
        filepath=None,
        max_bio_label=max_bio_frame.max_bio_label,
        max_bio_logit=max_bio_frame.max_bio_logit,
        max_perch_label=max_perch_frame.max_perch_label,
        max_perch_logit=max_perch_frame.max_perch_logit,
        noise_logits=json.dumps(noise_payload, separators=(",", ":")),
        nz_bird_logits=json.dumps(nz_payload, separators=(",", ":")),
        gate_mode=gate_mode,
        gate_threshold=bio_threshold,
        gate_trigger_count=trigger_count,
        retained_clip_count=len(retained_clips),
        margin_gate_scores=json.dumps(margin_gate_scores_payload(frame_scores), separators=(",", ":")),
        retained_clips=retained_clips,
    )


def save_retained_audio(
    buffer: AudioBuffer,
    retained_audio_dir: Path,
    device_id: str,
    reason: str,
    retention_clip: RetentionClip | None = None,
) -> str:
    import soundfile as sf

    retained_audio_dir.mkdir(parents=True, exist_ok=True)
    timestamp = buffer.timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
    if retention_clip is None:
        audio = buffer.source_audio
        filename = f"{device_id}_{timestamp}_{reason}_{buffer.source_file.stem}_{buffer.file_buffer_index:03d}.flac"
    else:
        start_sample = int(round(retention_clip.start_offset_s * buffer.source_sample_rate))
        end_sample = int(round(retention_clip.end_offset_s * buffer.source_sample_rate))
        audio = buffer.source_audio[start_sample:end_sample]
        duration = f"{retention_clip.duration_s:g}s"
        filename = (
            f"{device_id}_{timestamp}_{reason}_{buffer.source_file.stem}_{buffer.file_buffer_index:03d}"
            f"_seg{retention_clip.start_segment_index}-{retention_clip.end_segment_index}_{duration}.flac"
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
    cursor = conn.execute(
        """
        INSERT INTO buffer_events(
            device_id, timestamp_utc, audio_saved, retention_reason, filepath,
            max_bio_label, max_bio_logit, noise_logits, max_perch_label,
            max_perch_logit, nz_bird_logits, gate_mode, gate_threshold,
            gate_trigger_count, retained_clip_count, margin_gate_scores,
            sync_status, created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?);
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
    excluded_margin_label_indexes = build_margin_label_indexes(
        perch_labels,
        list(config.get("excluded_margin_labels", ["Water", "Train", "Vehicle"])),
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
                noise_label_indexes=noise_label_indexes,
                bio_label_indexes=bio_label_indexes,
                nz_label_indexes=nz_labels,
                excluded_margin_label_indexes=excluded_margin_label_indexes,
                bio_margin_threshold=float(config.get("bio_margin_threshold", config.get("bio_threshold", 0.55))),
            )
            decision = decide_buffer(
                frame_scores,
                bio_threshold=float(config.get("bio_margin_threshold", config.get("bio_threshold", 0.55))),
                noise_threshold=float(config.get("noise_threshold", 0.0)),
                validation_sample_interval=int(config["validation_sample_interval"]),
                dropped_buffer_count=dropped_buffer_count,
                gate_mode=str(config.get("bio_gate_mode", "nz_bird_margin")),
                max_variable_buffer_frames=int(config.get("max_variable_buffer_frames", 3)),
                perch_window_seconds=float(config.get("perch_window_seconds", 5.0)),
            )
            if decision.retention_reason == "dropped":
                dropped_buffer_count += 1
            elif decision.retention_reason == "bio_hit":
                bio_hits += 1
            elif decision.retention_reason == "validation_sample":
                validation_samples += 1

            if decision.audio_saved:
                if decision.retained_clips:
                    saved_clips: list[RetentionClip] = []
                    for clip in decision.retained_clips:
                        filepath = save_retained_audio(
                            buffer,
                            retained_audio_dir,
                            str(config["device_id"]),
                            clip.retention_reason,
                            clip,
                        )
                        saved_clips.append(replace(clip, filepath=filepath))
                    decision = replace(
                        decision,
                        filepath=None,
                        retained_clips=saved_clips,
                        retained_clip_count=len(saved_clips),
                        audio_saved=bool(saved_clips),
                    )
                    retained += len(saved_clips)
                else:
                    filepath = save_retained_audio(
                        buffer,
                        retained_audio_dir,
                        str(config["device_id"]),
                        decision.retention_reason,
                    )
                    decision = replace(decision, filepath=filepath)
                    retained += 1

            insert_buffer_event(conn, config=config, buffer=buffer, decision=decision, embeddings=embeddings)
            conn.commit()
            processed += 1
            max_bio_logit = (
                f"{decision.max_bio_logit:.3f}" if decision.max_bio_logit is not None else "None"
            )
            print(
                f"buffer={processed} source={buffer.source_file.name} "
                f"segment={buffer.file_buffer_index} decision={decision.retention_reason} "
                f"clips={decision.retained_clip_count} max_bio={decision.max_bio_label}:{max_bio_logit}"
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
