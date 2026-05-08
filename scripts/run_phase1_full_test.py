"""Run the Phase 1 mock pipeline over the full multi-night audio fixture.

The script writes a metrics package that can be used to produce
``docs/02_implementation/Phase1_test_report.md`` after the long run completes.
It deliberately writes to ``outputs/`` by default so the six-night rehearsal
does not mutate the normal local edge or hub databases.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterator

import requests
import numpy as np
import yaml
from scipy.signal import stft

try:
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover - exercised only before deps are installed.
    _tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from central_hub_mock.src.init_master_db import init_master_db
from central_hub_mock.src.watchdog_alert import check_watchdog
from edge_node_mock.src.bio_capture_loop import (
    BufferDecision,
    active_vector_storage,
    decide_buffer,
    insert_buffer_event,
    iter_audio_buffers,
    load_sqlite_vec,
    run_perch_inference,
    save_retained_audio,
)
from edge_node_mock.src.gate_config import validate_margin_gate_config
from edge_node_mock.src.gate_logic import build_margin_label_indexes, score_frames
from edge_node_mock.src.init_edge_db import init_edge_db
from edge_node_mock.src.inspect_perch_model import _load_model, load_nz_bird_labels, load_perch_labels
from edge_node_mock.src.sender_daemon import count_pending, send_pending_batch
from mock_common.config import load_config


BUFFER_FIELDNAMES = [
    "global_buffer_index",
    "night",
    "source_file",
    "file_buffer_index",
    "timestamp_utc",
    "source_sample_rate",
    "perch_sample_rate",
    "retention_reason",
    "audio_saved",
    "gate_mode",
    "gate_threshold",
    "gate_trigger_count",
    "retained_clip_count",
    "retained_seconds",
    "retained_clip_durations_json",
    "max_nz_bird_common_name",
    "max_nz_bird_scientific_name",
    "max_nz_bird_logit",
    "max_perch_label",
    "max_perch_logit",
    "excluded_top_label_veto_frame_count",
    "frame_bio_gate_buffer_active",
    "inference_seconds",
    "db_write_seconds",
    "audio_save_seconds",
    "total_buffer_seconds",
]

FRAME_FIELDNAMES = [
    "global_buffer_index",
    "night",
    "source_file",
    "file_buffer_index",
    "timestamp_utc",
    "segment_index",
    "top_excluded_label",
    "top_excluded_logit",
    "excluded_top_label_gate",
    "top_nz_common_name",
    "top_nz_scientific_name",
    "top_nz_logit",
    "nz_over_excluded_margin",
    "bio_margin_threshold",
    "margin_gate",
    "frame_bio_gate",
    "max_perch_label",
    "max_perch_logit",
    "nz_top3_json",
]

GATE_EVENT_FIELDNAMES = [
    "night",
    "source_file",
    "global_buffer_index",
    "file_buffer_index",
    "retention_index",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
]

RETAINED_CLIP_FIELDNAMES = [
    "night",
    "source_file",
    "global_buffer_index",
    "file_buffer_index",
    "retention_index",
    "retention_reason",
    "filepath",
    "start_segment_index",
    "end_segment_index",
    "start_seconds",
    "end_seconds",
    "duration_seconds",
    "triggered_frame_count",
]


@dataclass(frozen=True)
class GatePlotEvent:
    night: str
    source_file: Path
    global_buffer_index: int
    file_buffer_index: int
    retention_index: int
    start_seconds: float
    end_seconds: float


@dataclass
class AudioPlan:
    night: str
    night_dir: Path
    audio_files: list[Path]
    audio_seconds: float
    expected_buffers: int
    source_sample_rates: list[int]


@dataclass
class AggregateMetrics:
    retention_reasons: Counter[str] = field(default_factory=Counter)
    max_nz_bird_candidates: Counter[str] = field(default_factory=Counter)
    retained_nz_bird_candidates: Counter[str] = field(default_factory=Counter)
    max_perch_labels: Counter[str] = field(default_factory=Counter)
    excluded_label_evidence_labels: Counter[str] = field(default_factory=Counter)
    excluded_top_label_veto_labels: Counter[str] = field(default_factory=Counter)
    nz_rank1_species: Counter[str] = field(default_factory=Counter)
    nz_top3_species: Counter[str] = field(default_factory=Counter)
    bio_gate_frame_labels: Counter[str] = field(default_factory=Counter)
    retained_clip_durations: Counter[str] = field(default_factory=Counter)
    nz_species_max_logit: defaultdict[str, float] = field(
        default_factory=lambda: defaultdict(lambda: float("-inf"))
    )
    processed_buffers: int = 0
    frames: int = 0
    retained_audio: int = 0
    retained_seconds: float = 0.0
    bio_gate_buffers: int = 0
    bio_gate_frames: int = 0
    excluded_top_label_veto_buffers: int = 0
    excluded_top_label_veto_frames: int = 0
    inference_seconds: float = 0.0
    db_write_seconds: float = 0.0
    audio_save_seconds: float = 0.0
    buffer_total_seconds: float = 0.0
    inference_samples: list[float] = field(default_factory=list)
    margin_samples: list[float] = field(default_factory=list)


def utc_now_string() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def default_output_dir(now: datetime | None = None) -> Path:
    """Return outputs/phase1_full_test/DDMMYY/run-HHMM using local time."""

    now = now or datetime.now()
    return REPO_ROOT / "outputs" / "phase1_full_test" / now.strftime("%d%m%y") / f"run-{now:%H%M}"


def progress_bar(*args: Any, **kwargs: Any) -> Any:
    if _tqdm is None:
        raise RuntimeError(
            "tqdm is required for the full Phase 1 runner. "
            "Install dependencies with: .venv/bin/python -m pip install -r requirements-dev.txt"
        )
    return _tqdm(*args, **kwargs)


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def discover_night_dirs(raw_audio_root: Path) -> list[Path]:
    """Return sorted nightly directories, or the root itself for a flat fixture."""

    night_dirs = sorted(path for path in raw_audio_root.iterdir() if path.is_dir())
    return night_dirs or [raw_audio_root]


def count_audio_plan(
    night_dir: Path,
    *,
    raw_audio_glob: str,
    buffer_seconds: float,
    include_partial: bool,
) -> AudioPlan:
    import soundfile as sf

    audio_files = sorted(path for path in night_dir.glob(raw_audio_glob) if path.is_file())
    audio_seconds = 0.0
    expected_buffers = 0
    sample_rates: set[int] = set()

    for audio_file in audio_files:
        info = sf.info(audio_file)
        sample_rate = int(info.samplerate)
        sample_rates.add(sample_rate)
        audio_seconds += float(info.frames) / sample_rate
        block_size = int(round(sample_rate * buffer_seconds))
        full_buffers, remainder = divmod(int(info.frames), block_size)
        expected_buffers += full_buffers
        if include_partial and remainder:
            expected_buffers += 1

    return AudioPlan(
        night=night_dir.name,
        night_dir=night_dir,
        audio_files=audio_files,
        audio_seconds=audio_seconds,
        expected_buffers=expected_buffers,
        source_sample_rates=sorted(sample_rates),
    )


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def counter_payload(counter: Counter[str], *, limit: int | None = None) -> list[dict[str, Any]]:
    items = counter.most_common(limit)
    return [{"label": label, "count": count} for label, count in items]


def max_logit_payload(values: dict[str, float], *, limit: int = 30) -> list[dict[str, Any]]:
    finite_items = [(label, value) for label, value in values.items() if value != float("-inf")]
    finite_items.sort(key=lambda item: item[1], reverse=True)
    return [{"label": label, "max_logit": value} for label, value in finite_items[:limit]]


def timing_payload(metrics: AggregateMetrics, *, audio_seconds: float) -> dict[str, Any]:
    samples = metrics.inference_samples
    return {
        "elapsed_buffer_processing_seconds": metrics.buffer_total_seconds,
        "inference_seconds": metrics.inference_seconds,
        "db_write_seconds": metrics.db_write_seconds,
        "audio_save_seconds": metrics.audio_save_seconds,
        "mean_inference_seconds_per_buffer": mean(samples) if samples else None,
        "median_inference_seconds_per_buffer": median(samples) if samples else None,
        "p95_inference_seconds_per_buffer": percentile(samples, 0.95),
        "audio_realtime_factor_for_inference": audio_seconds / metrics.inference_seconds
        if metrics.inference_seconds
        else None,
        "buffers_per_second_wall": metrics.processed_buffers / metrics.buffer_total_seconds
        if metrics.buffer_total_seconds
        else None,
    }


def metrics_summary(metrics: AggregateMetrics, *, audio_seconds: float) -> dict[str, Any]:
    return {
        "processed_buffers": metrics.processed_buffers,
        "frames": metrics.frames,
        "retained_audio": metrics.retained_audio,
        "retained_seconds": metrics.retained_seconds,
        "bio_gate_buffers": metrics.bio_gate_buffers,
        "bio_gate_frames": metrics.bio_gate_frames,
        "excluded_top_label_veto_buffers": metrics.excluded_top_label_veto_buffers,
        "excluded_top_label_veto_frames": metrics.excluded_top_label_veto_frames,
        "retention_reasons": counter_payload(metrics.retention_reasons),
        "max_nz_bird_candidates": counter_payload(metrics.max_nz_bird_candidates, limit=30),
        "retained_nz_bird_candidates": counter_payload(metrics.retained_nz_bird_candidates, limit=30),
        "bio_gate_frame_labels": counter_payload(metrics.bio_gate_frame_labels, limit=30),
        "excluded_top_label_veto_labels": counter_payload(metrics.excluded_top_label_veto_labels, limit=30),
        "retained_clip_durations": counter_payload(metrics.retained_clip_durations),
        "max_perch_labels": counter_payload(metrics.max_perch_labels, limit=30),
        "excluded_label_evidence_labels": counter_payload(metrics.excluded_label_evidence_labels, limit=30),
        "nz_rank1_species": counter_payload(metrics.nz_rank1_species, limit=30),
        "nz_top3_species": counter_payload(metrics.nz_top3_species, limit=30),
        "nz_species_max_logits": max_logit_payload(metrics.nz_species_max_logit),
        "margin_distribution": {
            "count": len(metrics.margin_samples),
            "min": min(metrics.margin_samples) if metrics.margin_samples else None,
            "median": median(metrics.margin_samples) if metrics.margin_samples else None,
            "p95": percentile(metrics.margin_samples, 0.95),
            "max": max(metrics.margin_samples) if metrics.margin_samples else None,
        },
        "timing": timing_payload(metrics, audio_seconds=audio_seconds),
    }


def update_metrics_from_buffer(
    metrics: AggregateMetrics,
    *,
    decision: BufferDecision,
    frame_scores: list[Any],
    inference_seconds: float,
    db_write_seconds: float,
    audio_save_seconds: float,
    total_buffer_seconds: float,
) -> int:
    metrics.processed_buffers += 1
    metrics.frames += len(frame_scores)
    metrics.retention_reasons[decision.retention_reason] += 1
    metrics.max_perch_labels[decision.max_perch_label] += 1
    metrics.inference_seconds += inference_seconds
    metrics.db_write_seconds += db_write_seconds
    metrics.audio_save_seconds += audio_save_seconds
    metrics.buffer_total_seconds += total_buffer_seconds
    metrics.inference_samples.append(inference_seconds)

    if decision.max_nz_bird_common_name:
        metrics.max_nz_bird_candidates[decision.max_nz_bird_common_name] += 1
    if decision.audio_saved:
        metrics.retained_audio += decision.retained_clip_count or 1
    for clip in decision.retained_clips:
        metrics.retained_seconds += clip.duration_s
        metrics.retained_clip_durations[f"{clip.duration_s:g}s"] += 1
    if decision.retention_reason == "bio_hit":
        metrics.bio_gate_buffers += 1
        if decision.max_nz_bird_common_name:
            metrics.retained_nz_bird_candidates[decision.max_nz_bird_common_name] += 1

    excluded_top_label_veto_frame_count = 0
    for score in frame_scores:
        if score.top_excluded_label:
            metrics.excluded_label_evidence_labels[score.top_excluded_label] += 1
        excluded_top_label_veto = is_excluded_top_label_veto(score)
        if excluded_top_label_veto:
            excluded_top_label_veto_frame_count += 1
            metrics.excluded_top_label_veto_frames += 1
        if getattr(score, "excluded_top_label_gate", False) and getattr(score, "top_excluded_label", None):
            metrics.excluded_top_label_veto_labels[score.top_excluded_label] += 1

        if score.top_nz_birds:
            rank1 = score.top_nz_birds[0]
            rank1_label = species_label(rank1)
            metrics.nz_rank1_species[rank1_label] += 1
            for bird in score.top_nz_birds:
                label = species_label(bird)
                metrics.nz_top3_species[label] += 1
                metrics.nz_species_max_logit[label] = max(
                    metrics.nz_species_max_logit[label],
                    float(bird["logit"]),
                )

        if getattr(score, "nz_over_excluded_margin", None) is not None:
            metrics.margin_samples.append(float(score.nz_over_excluded_margin))

        if getattr(score, "frame_bio_gate", False) and score.top_nz_common_name:
            metrics.bio_gate_frames += 1
            metrics.bio_gate_frame_labels[score.top_nz_common_name] += 1

    if excluded_top_label_veto_frame_count:
        metrics.excluded_top_label_veto_buffers += 1
    return excluded_top_label_veto_frame_count


def is_excluded_top_label_veto(score: Any) -> bool:
    return bool(getattr(score, "excluded_top_label_gate", False))


def species_label(bird: dict[str, Any]) -> str:
    return f"{bird['common_name']} ({bird['scientific_name']})"


def write_row(writer: csv.DictWriter, row: dict[str, Any]) -> None:
    writer.writerow({key: row.get(key) for key in writer.fieldnames or []})


def default_plot_style() -> dict[str, Any]:
    return {
        "start_line_color": "lime",
        "end_line_color": "red",
        "clip_bar_color": "lime",
        "line_alpha": 0.75,
        "clip_bar_alpha": 0.75,
        "line_width": 1.6,
        "clip_bar_height_fraction": 0.06,
    }


def resolve_plot_style(config: dict[str, Any]) -> dict[str, Any]:
    style = default_plot_style()
    configured = config.get("plot_style", {}) or {}
    if not isinstance(configured, dict):
        raise ValueError("plot_style must be a mapping")
    style.update(configured)
    style["line_alpha"] = float(style["line_alpha"])
    style["clip_bar_alpha"] = float(style["clip_bar_alpha"])
    style["line_width"] = float(style["line_width"])
    style["clip_bar_height_fraction"] = float(style["clip_bar_height_fraction"])
    return style


def hz_to_mel(hz: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int = 96, fmin: float = 40.0) -> np.ndarray:
    fmax = sample_rate / 2
    mel_points = np.linspace(float(hz_to_mel(fmin)), float(hz_to_mel(fmax)), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_indexes = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for mel_index in range(1, n_mels + 1):
        left, center, right = bin_indexes[mel_index - 1], bin_indexes[mel_index], bin_indexes[mel_index + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for bin_index in range(left, center):
            if 0 <= bin_index < filters.shape[1]:
                filters[mel_index - 1, bin_index] = (bin_index - left) / (center - left)
        for bin_index in range(center, right):
            if 0 <= bin_index < filters.shape[1]:
                filters[mel_index - 1, bin_index] = (right - bin_index) / (right - center)
    return filters


def compute_mel_db(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    n_fft = 2048
    hop_length = 512
    _, _, zxx = stft(
        audio,
        fs=sample_rate,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    power = np.abs(zxx) ** 2
    mel_power = mel_filterbank(sample_rate, n_fft) @ power
    mel_db = 10.0 * np.log10(np.maximum(mel_power, 1e-12))
    mel_db -= mel_db.max() if mel_db.size else 0.0
    return mel_db


def load_plot_audio_minute(audio_file: Path, minute_index: int) -> tuple[np.ndarray, int]:
    import soundfile as sf
    from scipy.signal import resample_poly

    info = sf.info(audio_file)
    source_sample_rate = int(info.samplerate)
    start_frame = int(minute_index * 60 * source_sample_rate)
    frames = int(60 * source_sample_rate)
    block, sample_rate = sf.read(
        audio_file,
        start=start_frame,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    if len(block) == 0:
        return np.zeros(0, dtype=np.float32), source_sample_rate
    if len(block) < frames:
        block = np.pad(block, ((0, frames - len(block)), (0, 0)))
    mono = np.mean(block, axis=1, dtype=np.float32)
    if sample_rate == 48000:
        return resample_poly(mono, up=2, down=3).astype(np.float32), 32000
    return mono.astype(np.float32, copy=False), int(sample_rate)


def plot_gate_events_for_file(
    audio_file: Path,
    events: list[GatePlotEvent],
    *,
    output_path: Path,
    show: bool,
    plot_style: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt
    import soundfile as sf

    info = sf.info(audio_file)
    minute_count = max(1, int(math.ceil(float(info.duration) / 60.0)))
    fig_height = max(3.0, float(minute_count))
    fig, axes = plt.subplots(
        nrows=minute_count,
        ncols=1,
        figsize=(8, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    axes_list = list(axes[:, 0])

    for minute_index, axis in enumerate(axes_list):
        audio, sample_rate = load_plot_audio_minute(audio_file, minute_index)
        if len(audio):
            mel_db = compute_mel_db(audio, sample_rate)
            axis.imshow(
                mel_db,
                origin="lower",
                aspect="auto",
                extent=[0, min(60.0, len(audio) / sample_rate), 0, mel_db.shape[0]],
                cmap="magma",
                vmin=-80,
                vmax=0,
            )
        axis.set_xlim(0, 60)
        axis.set_ylabel(f"{minute_index:02d}m", rotation=0, labelpad=16, va="center")
        axis.tick_params(axis="y", left=False, labelleft=False)
        if minute_index < minute_count - 1:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Seconds within minute")

    for event in events:
        start_minute = int(event.start_seconds // 60)
        end_minute = int(max(event.end_seconds - 1e-9, 0.0) // 60)
        for minute_index in range(start_minute, end_minute + 1):
            if not 0 <= minute_index < minute_count:
                continue
            minute_start = minute_index * 60.0
            minute_end = minute_start + 60.0
            bar_start = max(event.start_seconds, minute_start) - minute_start
            bar_end = min(event.end_seconds, minute_end) - minute_start
            if bar_end <= bar_start:
                continue
            y_height = max(
                1.0,
                axes_list[minute_index].get_ylim()[1] * plot_style["clip_bar_height_fraction"],
            )
            axes_list[minute_index].broken_barh(
                [(bar_start, bar_end - bar_start)],
                (0, y_height),
                facecolors=plot_style["clip_bar_color"],
                alpha=plot_style["clip_bar_alpha"],
                edgecolors="none",
            )
        if 0 <= start_minute < minute_count:
            axes_list[start_minute].axvline(
                event.start_seconds % 60.0,
                color=plot_style["start_line_color"],
                alpha=plot_style["line_alpha"],
                linewidth=plot_style["line_width"],
            )
        if 0 <= end_minute < minute_count:
            axes_list[end_minute].axvline(
                event.end_seconds % 60.0,
                color=plot_style["end_line_color"],
                alpha=plot_style["line_alpha"],
                linewidth=plot_style["line_width"],
            )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=plot_style["start_line_color"],
            alpha=plot_style["line_alpha"],
            linewidth=plot_style["line_width"],
            label="retained clip start",
        ),
        plt.Line2D(
            [0],
            [0],
            color=plot_style["end_line_color"],
            alpha=plot_style["line_alpha"],
            linewidth=plot_style["line_width"],
            label="retained clip end",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=plot_style["clip_bar_color"],
            alpha=plot_style["clip_bar_alpha"],
            label="saved clip span",
        ),
    ]
    fig.suptitle(f"{audio_file.name} bio-gate retained spans", fontsize=12)
    axes_list[0].legend(handles=handles, loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    if show:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)


def write_gate_plots(
    audio_files: list[Path],
    events_by_file: dict[Path, list[GatePlotEvent]],
    *,
    output_dir: Path,
    raw_audio_root: Path,
    show: bool,
    plot_style: dict[str, Any],
) -> list[str]:
    plot_paths: list[str] = []
    plot_root = output_dir / "gate_plots"
    for audio_file, events in progress_bar(
        [(path, events_by_file.get(path, [])) for path in sorted(audio_files, key=str)],
        desc="Writing gate plots",
        unit="file",
    ):
        try:
            relative_parent = audio_file.parent.relative_to(raw_audio_root)
        except ValueError:
            relative_parent = Path(audio_file.parent.name)
        output_path = plot_root / relative_parent / f"{audio_file.stem}_gate_plot.png"
        plot_gate_events_for_file(audio_file, events, output_path=output_path, show=show, plot_style=plot_style)
        plot_paths.append(str(output_path))
    return plot_paths


def database_counts(db_path: Path, *, hub: bool = False) -> dict[str, Any]:
    if not db_path.exists():
        return {"exists": False}

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        meta = dict(conn.execute("SELECT key, value FROM schema_metadata;").fetchall())
        vector_table = str(meta.get("vector_table"))
        if meta.get("vector_storage_mode") == "sqlite_vec":
            load_sqlite_vec(conn)
        if hub:
            return {
                "exists": True,
                "ingestion_batches": conn.execute("SELECT COUNT(*) FROM ingestion_batches;").fetchone()[0],
                "hub_buffer_events": conn.execute("SELECT COUNT(*) FROM hub_buffer_events;").fetchone()[0],
                "hub_retained_audio_clips": conn.execute(
                    "SELECT COUNT(*) FROM hub_retained_audio_clips;"
                ).fetchone()[0],
                "hub_embedding_segments": conn.execute(
                    "SELECT COUNT(*) FROM hub_embedding_segments;"
                ).fetchone()[0],
                vector_table: conn.execute(f"SELECT COUNT(*) FROM {vector_table};").fetchone()[0],
                "health_metrics": conn.execute("SELECT COUNT(*) FROM health_metrics;").fetchone()[0],
                "vector_table": vector_table,
                "vector_storage_mode": meta.get("vector_storage_mode"),
            }
        return {
            "exists": True,
            "buffer_events": conn.execute("SELECT COUNT(*) FROM buffer_events;").fetchone()[0],
            "retained_audio_clips": conn.execute("SELECT COUNT(*) FROM retained_audio_clips;").fetchone()[0],
            "embedding_segments": conn.execute("SELECT COUNT(*) FROM embedding_segments;").fetchone()[0],
            vector_table: conn.execute(f"SELECT COUNT(*) FROM {vector_table};").fetchone()[0],
            "pending": conn.execute("SELECT COUNT(*) FROM buffer_events WHERE sync_status='pending';").fetchone()[0],
            "synced": conn.execute("SELECT COUNT(*) FROM buffer_events WHERE sync_status='synced';").fetchone()[0],
            "vector_table": vector_table,
            "vector_storage_mode": meta.get("vector_storage_mode"),
        }


@contextmanager
def running_hub_api(hub_config_path: Path, *, port: int) -> Iterator[None]:
    env = os.environ.copy()
    env["HUB_CONFIG"] = str(hub_config_path)
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "central_hub_mock.src.ingestion_api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(80):
            try:
                response = requests.get(f"http://127.0.0.1:{port}/docs", timeout=0.25)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                time.sleep(0.25)
        else:
            server.terminate()
            output = ""
            try:
                output = server.communicate(timeout=5)[0]
            except subprocess.TimeoutExpired:
                server.kill()
            raise RuntimeError(f"Hub API did not start on port {port}. Output:\n{output}")
        yield
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


def sync_pending_rows(
    edge_config_path: Path,
    *,
    total_buffers: int,
    batch_size: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    accepted_ids: list[int] = []
    payload_bytes = 0
    batches = 0

    with progress_bar(total=total_buffers, desc="Syncing edge rows to hub", unit="buffer") as progress:
        while True:
            pending = count_pending(load_config(edge_config_path))
            if pending <= 0:
                break
            result = send_pending_batch(edge_config_path, limit=batch_size)
            batches += 1
            payload_bytes += result.payload_bytes
            if result.status != "sent":
                raise RuntimeError(f"Hub sync failed with status={result.status}")
            accepted_ids.extend(result.accepted_buffer_ids)
            progress.update(len(result.accepted_buffer_ids))

    return {
        "enabled": True,
        "batches": batches,
        "accepted_buffer_count": len(accepted_ids),
        "first_accepted_buffer_id": accepted_ids[0] if accepted_ids else None,
        "last_accepted_buffer_id": accepted_ids[-1] if accepted_ids else None,
        "payload_bytes": payload_bytes,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_full_test(args: argparse.Namespace) -> dict[str, Any]:
    started_wall = time.perf_counter()
    run_started_at = utc_now_string()
    if args.skip_audio_save and args.sync_to_hub:
        raise ValueError("--skip-audio-save cannot be combined with --sync-to-hub because hub validation requires retained clip metadata.")

    output_dir = default_output_dir() if args.output_dir is None else resolve_repo_path(args.output_dir)
    existing_outputs = [
        output_dir / "edge_full_test.sqlite",
        output_dir / "hub_full_test.sqlite",
        output_dir / "phase1_full_test_metrics.json",
        output_dir / "buffer_metrics.csv",
        output_dir / "frame_metrics.csv",
        output_dir / "gate_plot_events.csv",
        output_dir / "retained_clips.csv",
    ]
    if not args.reset_output and any(path.exists() for path in existing_outputs):
        raise RuntimeError(
            f"Output directory already contains Phase 1 test artifacts: {output_dir}. "
            "Use --reset-output or choose a different --output-dir."
        )
    if args.reset_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buffer_csv_path = output_dir / "buffer_metrics.csv"
    frame_csv_path = output_dir / "frame_metrics.csv"
    gate_event_csv_path = output_dir / "gate_plot_events.csv"
    retained_clip_csv_path = output_dir / "retained_clips.csv"
    metrics_path = output_dir / "phase1_full_test_metrics.json"

    base_config = load_config(args.edge_config)
    raw_audio_root = resolve_repo_path(args.raw_audio_root or base_config["raw_audio_mount"])
    if not raw_audio_root.exists():
        raise FileNotFoundError(f"Raw audio root does not exist: {raw_audio_root}")

    include_partial = bool(args.include_partial_final_buffer or base_config.get("include_partial_final_buffer", False))
    raw_audio_glob = str(args.raw_audio_glob or base_config.get("raw_audio_glob", "*.wav"))
    buffer_seconds = float(args.buffer_seconds)

    night_dirs = discover_night_dirs(raw_audio_root)
    if args.max_nights is not None:
        night_dirs = night_dirs[: args.max_nights]

    plans = [
        count_audio_plan(
            night_dir,
            raw_audio_glob=raw_audio_glob,
            buffer_seconds=buffer_seconds,
            include_partial=include_partial,
        )
        for night_dir in night_dirs
    ]
    plans = [plan for plan in plans if plan.audio_files]
    if not plans:
        raise FileNotFoundError(f"No audio files found under {raw_audio_root} with glob {raw_audio_glob}")

    edge_config = dict(base_config)
    edge_config.update(
        {
            "edge_db_path": str(output_dir / "edge_full_test.sqlite"),
            "retained_audio_dir": str(output_dir / "retained_audio"),
            "include_partial_final_buffer": include_partial,
        }
    )
    plot_style = resolve_plot_style(edge_config)

    hub_config_path: Path | None = None
    hub_config: dict[str, Any] | None = None
    if args.sync_to_hub:
        hub_config = dict(load_config(args.hub_config))
        hub_config.update(
            {
                "master_db_path": str(output_dir / "hub_full_test.sqlite"),
                "api_key": edge_config["api_key"],
                "allowed_device_ids": [edge_config["device_id"]],
            }
        )
        edge_config["hub_ingest_url"] = f"http://127.0.0.1:{args.hub_port}/ingest_batch"
        hub_config_path = output_dir / "hub_config.full_test.yaml"
        hub_config_path.write_text(yaml.safe_dump(hub_config, sort_keys=False), encoding="utf-8")
        init_master_db(hub_config_path, reset=True)

    edge_config_path = output_dir / "edge_config.full_test.yaml"
    edge_config_path.write_text(yaml.safe_dump(edge_config, sort_keys=False), encoding="utf-8")
    edge_db_path = init_edge_db(edge_config_path, reset=True)

    perch_labels = load_perch_labels(resolve_repo_path(edge_config["perch_label_path"]))
    gate_config = validate_margin_gate_config(edge_config, perch_labels=perch_labels)
    nz_labels = load_nz_bird_labels(resolve_repo_path(edge_config["nz_bird_label_path"]), perch_labels)
    excluded_margin_label_indexes = build_margin_label_indexes(
        perch_labels,
        gate_config.excluded_margin_labels,
    )

    model_started = time.perf_counter()
    model, model_source, model_ref = _load_model(edge_config)
    model_load_seconds = time.perf_counter() - model_started

    totals = AggregateMetrics()
    night_metrics: list[dict[str, Any]] = []
    gate_events_by_file: dict[Path, list[GatePlotEvent]] = defaultdict(list)
    processed_audio_files: set[Path] = set()
    global_buffer_index = 0

    with (
        sqlite3.connect(edge_db_path) as conn,
        buffer_csv_path.open("w", newline="", encoding="utf-8") as buffer_file,
        frame_csv_path.open("w", newline="", encoding="utf-8") as frame_file,
        gate_event_csv_path.open("w", newline="", encoding="utf-8") as gate_event_file,
        retained_clip_csv_path.open("w", newline="", encoding="utf-8") as retained_clip_file,
    ):
        conn.execute("PRAGMA foreign_keys=ON;")
        buffer_writer = csv.DictWriter(buffer_file, fieldnames=BUFFER_FIELDNAMES)
        frame_writer = csv.DictWriter(frame_file, fieldnames=FRAME_FIELDNAMES)
        gate_event_writer = csv.DictWriter(gate_event_file, fieldnames=GATE_EVENT_FIELDNAMES)
        retained_clip_writer = csv.DictWriter(retained_clip_file, fieldnames=RETAINED_CLIP_FIELDNAMES)
        buffer_writer.writeheader()
        frame_writer.writeheader()
        gate_event_writer.writeheader()
        retained_clip_writer.writeheader()

        for plan in progress_bar(plans, desc="Nights", unit="night"):
            night_config = dict(edge_config)
            night_config["raw_audio_mount"] = str(plan.night_dir)
            night_config["raw_audio_glob"] = raw_audio_glob
            night = AggregateMetrics()

            with progress_bar(
                total=plan.expected_buffers,
                desc=f"{plan.night}",
                unit="buffer",
                leave=False,
            ) as progress:
                for buffer in iter_audio_buffers(
                    night_config,
                    seconds=buffer_seconds,
                    include_partial=include_partial,
                ):
                    if args.max_buffers_per_night is not None and night.processed_buffers >= args.max_buffers_per_night:
                        break
                    processed_audio_files.add(buffer.source_file)
                    global_buffer_index += 1
                    buffer_started = time.perf_counter()

                    infer_started = time.perf_counter()
                    logits, embeddings = run_perch_inference(model, buffer.perch_audio, buffer.perch_sample_rate)
                    inference_seconds = time.perf_counter() - infer_started

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
                    raw_retained_clips = list(decision.retained_clips)
                    for clip in raw_retained_clips:
                        start_seconds = buffer.file_buffer_index * buffer_seconds + clip.start_offset_s
                        end_seconds = buffer.file_buffer_index * buffer_seconds + clip.end_offset_s
                        event = GatePlotEvent(
                            night=plan.night,
                            source_file=buffer.source_file,
                            global_buffer_index=global_buffer_index,
                            file_buffer_index=buffer.file_buffer_index,
                            retention_index=clip.retention_index,
                            start_seconds=start_seconds,
                            end_seconds=end_seconds,
                        )
                        gate_events_by_file[buffer.source_file].append(event)
                        write_row(
                            gate_event_writer,
                            {
                                "night": event.night,
                                "source_file": str(event.source_file),
                                "global_buffer_index": event.global_buffer_index,
                                "file_buffer_index": event.file_buffer_index,
                                "retention_index": event.retention_index,
                                "start_seconds": event.start_seconds,
                                "end_seconds": event.end_seconds,
                                "duration_seconds": event.end_seconds - event.start_seconds,
                            },
                        )

                    audio_save_seconds = 0.0
                    if decision.audio_saved and args.skip_audio_save:
                        metrics_decision = decision
                        decision = replace(
                            decision,
                            audio_saved=False,
                            retained_clips=[],
                            retained_clip_count=0,
                        )
                    elif decision.audio_saved:
                        audio_save_started = time.perf_counter()
                        saved_clips = []
                        for clip in decision.retained_clips:
                            filepath = save_retained_audio(
                                buffer,
                                Path(str(edge_config["retained_audio_dir"])),
                                str(edge_config["device_id"]),
                                clip,
                            )
                            saved_clips.append(replace(clip, filepath=filepath))
                        decision = replace(
                            decision,
                            retained_clips=saved_clips,
                            retained_clip_count=len(saved_clips),
                            audio_saved=bool(saved_clips),
                        )
                        metrics_decision = decision
                        audio_save_seconds = time.perf_counter() - audio_save_started
                    else:
                        metrics_decision = decision

                    db_started = time.perf_counter()
                    insert_buffer_event(conn, config=edge_config, buffer=buffer, decision=decision, embeddings=embeddings)
                    conn.commit()
                    db_write_seconds = time.perf_counter() - db_started

                    total_buffer_seconds = time.perf_counter() - buffer_started
                    excluded_top_label_veto_frame_count = update_metrics_from_buffer(
                        night,
                        decision=metrics_decision,
                        frame_scores=frame_scores,
                        inference_seconds=inference_seconds,
                        db_write_seconds=db_write_seconds,
                        audio_save_seconds=audio_save_seconds,
                        total_buffer_seconds=total_buffer_seconds,
                    )
                    update_metrics_from_buffer(
                        totals,
                        decision=metrics_decision,
                        frame_scores=frame_scores,
                        inference_seconds=inference_seconds,
                        db_write_seconds=db_write_seconds,
                        audio_save_seconds=audio_save_seconds,
                        total_buffer_seconds=total_buffer_seconds,
                    )

                    write_row(
                        buffer_writer,
                        {
                            "global_buffer_index": global_buffer_index,
                            "night": plan.night,
                            "source_file": str(buffer.source_file),
                            "file_buffer_index": buffer.file_buffer_index,
                            "timestamp_utc": buffer.timestamp_utc.isoformat().replace("+00:00", "Z"),
                            "source_sample_rate": buffer.source_sample_rate,
                            "perch_sample_rate": buffer.perch_sample_rate,
                            "retention_reason": decision.retention_reason,
                            "audio_saved": int(decision.audio_saved and not args.skip_audio_save),
                            "gate_mode": decision.gate_mode,
                            "gate_threshold": decision.gate_threshold,
                            "gate_trigger_count": decision.gate_trigger_count,
                            "retained_clip_count": metrics_decision.retained_clip_count,
                            "retained_seconds": sum(clip.duration_s for clip in metrics_decision.retained_clips),
                            "retained_clip_durations_json": json.dumps(
                                [clip.duration_s for clip in metrics_decision.retained_clips],
                                separators=(",", ":"),
                            ),
                            "max_nz_bird_common_name": decision.max_nz_bird_common_name,
                            "max_nz_bird_scientific_name": decision.max_nz_bird_scientific_name,
                            "max_nz_bird_logit": decision.max_nz_bird_logit,
                            "max_perch_label": decision.max_perch_label,
                            "max_perch_logit": decision.max_perch_logit,
                            "excluded_top_label_veto_frame_count": excluded_top_label_veto_frame_count,
                            "frame_bio_gate_buffer_active": int(metrics_decision.retention_reason == "bio_hit"),
                            "inference_seconds": inference_seconds,
                            "db_write_seconds": db_write_seconds,
                            "audio_save_seconds": audio_save_seconds,
                            "total_buffer_seconds": total_buffer_seconds,
                        },
                    )

                    for score in frame_scores:
                        write_row(
                            frame_writer,
                            {
                                "global_buffer_index": global_buffer_index,
                                "night": plan.night,
                                "source_file": str(buffer.source_file),
                                "file_buffer_index": buffer.file_buffer_index,
                                "timestamp_utc": buffer.timestamp_utc.isoformat().replace("+00:00", "Z"),
                                "segment_index": score.segment_index,
                                "top_excluded_label": score.top_excluded_label,
                                "top_excluded_logit": score.top_excluded_logit,
                                "excluded_top_label_gate": int(score.excluded_top_label_gate),
                                "top_nz_common_name": score.top_nz_common_name,
                                "top_nz_scientific_name": score.top_nz_scientific_name,
                                "top_nz_logit": score.top_nz_logit,
                                "nz_over_excluded_margin": score.nz_over_excluded_margin,
                                "bio_margin_threshold": score.bio_margin_threshold,
                                "margin_gate": int(score.margin_gate),
                                "frame_bio_gate": int(score.frame_bio_gate),
                                "max_perch_label": score.max_perch_label,
                                "max_perch_logit": score.max_perch_logit,
                                "nz_top3_json": json.dumps(score.top_nz_birds, separators=(",", ":")),
                            },
                        )

                    for clip in metrics_decision.retained_clips:
                        start_seconds = buffer.file_buffer_index * buffer_seconds + clip.start_offset_s
                        end_seconds = buffer.file_buffer_index * buffer_seconds + clip.end_offset_s
                        write_row(
                            retained_clip_writer,
                            {
                                "night": plan.night,
                                "source_file": str(buffer.source_file),
                                "global_buffer_index": global_buffer_index,
                                "file_buffer_index": buffer.file_buffer_index,
                                "retention_index": clip.retention_index,
                                "retention_reason": clip.retention_reason,
                                "filepath": clip.filepath,
                                "start_segment_index": clip.start_segment_index,
                                "end_segment_index": clip.end_segment_index,
                                "start_seconds": start_seconds,
                                "end_seconds": end_seconds,
                                "duration_seconds": clip.duration_s,
                                "triggered_frame_count": clip.triggered_frame_count,
                            },
                        )

                    progress.update(1)
                    progress.set_postfix(
                        bio=night.retention_reasons.get("bio_hit", 0),
                        saved=night.retained_audio,
                        veto=night.excluded_top_label_veto_buffers,
                    )

            night_metrics.append(
                {
                    "night": plan.night,
                    "night_dir": str(plan.night_dir),
                    "audio_file_count": len(plan.audio_files),
                    "audio_seconds": plan.audio_seconds,
                    "expected_buffers": plan.expected_buffers,
                    "source_sample_rates": plan.source_sample_rates,
                    **metrics_summary(night, audio_seconds=plan.audio_seconds),
                }
            )

    gate_plot_paths: list[str] = []
    if not args.no_gate_plots:
        gate_plot_paths = write_gate_plots(
            sorted(processed_audio_files, key=str),
            gate_events_by_file,
            output_dir=output_dir,
            raw_audio_root=raw_audio_root,
            show=bool(args.show_gate_plots),
            plot_style=plot_style,
        )

    sync_metrics: dict[str, Any] = {"enabled": False}
    watchdog_metrics: dict[str, Any] | None = None
    if args.sync_to_hub and hub_config_path is not None:
        with running_hub_api(hub_config_path, port=args.hub_port):
            sync_metrics = sync_pending_rows(
                edge_config_path,
                total_buffers=totals.processed_buffers,
                batch_size=args.sender_batch_size,
            )
        watchdog = check_watchdog(hub_config_path)
        watchdog_metrics = {
            "status": watchdog.status,
            "message": watchdog.message,
            "age_minutes": watchdog.age_minutes,
        }

    report = {
        "schema_version": 2,
        "created_at_utc": utc_now_string(),
        "run_started_at_utc": run_started_at,
        "inputs": {
            "edge_config": str(resolve_repo_path(args.edge_config)),
            "hub_config": str(resolve_repo_path(args.hub_config)) if args.sync_to_hub else None,
            "raw_audio_root": str(raw_audio_root),
            "raw_audio_glob": raw_audio_glob,
            "night_count": len(plans),
            "buffer_seconds": buffer_seconds,
            "include_partial_final_buffer": include_partial,
            "skip_audio_save": args.skip_audio_save,
            "max_nights": args.max_nights,
            "max_buffers_per_night": args.max_buffers_per_night,
        },
        "config_snapshot": {
            "device_id": edge_config["device_id"],
            "bio_gate_mode": gate_config.bio_gate_mode,
            "bio_margin_threshold": gate_config.bio_margin_threshold,
            "excluded_margin_labels": gate_config.excluded_margin_labels,
            "max_variable_buffer_frames": gate_config.max_variable_buffer_frames,
            "perch_window_seconds": gate_config.perch_window_seconds,
            "embedding_dim": edge_config["embedding_dim"],
        },
        "model": {
            "source": model_source,
            "handle_or_path": model_ref,
            "load_seconds": model_load_seconds,
        },
        "outputs": {
            "output_dir": str(output_dir),
            "edge_config": str(edge_config_path),
            "hub_config": str(hub_config_path) if hub_config_path else None,
            "edge_db": str(edge_db_path),
            "hub_db": str(output_dir / "hub_full_test.sqlite") if args.sync_to_hub else None,
            "retained_audio_dir": str(edge_config["retained_audio_dir"]),
            "buffer_metrics_csv": str(buffer_csv_path),
            "frame_metrics_csv": str(frame_csv_path),
            "gate_plot_events_csv": str(gate_event_csv_path),
            "retained_clips_csv": str(retained_clip_csv_path),
            "gate_plot_dir": str(output_dir / "gate_plots"),
            "gate_plot_count": len(gate_plot_paths),
            "gate_plot_paths": gate_plot_paths,
            "metrics_json": str(metrics_path),
        },
        "plot_style": plot_style,
        "totals": {
            "audio_file_count": sum(len(plan.audio_files) for plan in plans),
            "audio_seconds": sum(plan.audio_seconds for plan in plans),
            "expected_buffers": sum(plan.expected_buffers for plan in plans),
            **metrics_summary(totals, audio_seconds=sum(plan.audio_seconds for plan in plans)),
            "wall_seconds": time.perf_counter() - started_wall,
        },
        "nights": night_metrics,
        "sync": sync_metrics,
        "watchdog": watchdog_metrics,
        "database_counts": {
            "edge": database_counts(edge_db_path),
            "hub": database_counts(Path(str(output_dir / "hub_full_test.sqlite")), hub=True)
            if args.sync_to_hub
            else None,
        },
    }

    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nMetrics written to: {metrics_path}")
    print(f"Buffer CSV written to: {buffer_csv_path}")
    print(f"Frame CSV written to: {frame_csv_path}")
    print(f"Gate plot events CSV written to: {gate_event_csv_path}")
    print(f"Retained clips CSV written to: {retained_clip_csv_path}")
    if gate_plot_paths:
        print(f"Gate plots written to: {output_dir / 'gate_plots'}")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-config", default="edge_node_mock/config/edge_config.local.yaml")
    parser.add_argument("--hub-config", default="central_hub_mock/config/hub_config.local.yaml")
    parser.add_argument("--raw-audio-root", default=None, help="Override the raw audio root from the edge config.")
    parser.add_argument("--raw-audio-glob", default=None, help="Override the raw audio glob from the edge config.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the default outputs/phase1_full_test/DDMMYY/run-HHMM directory.",
    )
    parser.add_argument("--reset-output", action="store_true", help="Delete the output directory before running.")
    parser.add_argument("--buffer-seconds", type=float, default=15.0)
    parser.add_argument("--include-partial-final-buffer", action="store_true")
    parser.add_argument("--skip-audio-save", action="store_true", help="Collect retention metrics without writing FLACs.")
    parser.add_argument("--sync-to-hub", action="store_true", help="Start a local hub API and sync all pending rows.")
    parser.add_argument("--no-gate-plots", action="store_true", help="Skip writing stacked mel spectrogram gate plots.")
    parser.add_argument(
        "--show-gate-plots",
        action="store_true",
        help="Display each saved gate plot in an on-screen matplotlib window while writing it.",
    )
    parser.add_argument("--hub-port", type=int, default=8011)
    parser.add_argument("--sender-batch-size", type=int, default=100)
    parser.add_argument("--max-nights", type=int, default=None, help="Short-run debugging limit.")
    parser.add_argument("--max-buffers-per-night", type=int, default=None, help="Short-run debugging limit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_full_test(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
