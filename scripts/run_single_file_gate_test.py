"""Run the margin bio gate over one configured audio file section.

The script is intended for fast, repeatable gate tuning after a promising
notebook experiment. It writes a compact metrics package under ``outputs/``:

* ``summary.json`` for headline statistics
* ``buffer_metrics.csv`` for one row per 15-second inference buffer
* ``frame_metrics.csv`` for one row per 5-second Perch frame
* ``retained_clips.csv`` for proposed variable-length retained clips
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from scipy.signal import resample_poly, stft

try:
    from tqdm.auto import tqdm as _tqdm
except ModuleNotFoundError:  # pragma: no cover - only before deps are installed.
    _tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_node_mock.src.bio_capture_loop import (
    AudioBuffer,
    save_retained_audio,
    timestamp_from_filename,
)
from edge_node_mock.src.gate_config import validate_margin_gate_config
from edge_node_mock.src.gate_logic import (
    build_margin_label_indexes,
    build_variable_retention_buffers,
    score_frames,
)
from edge_node_mock.src.inspect_perch_model import (
    _load_model,
    _pick_output,
    _serving_signature,
    load_nz_bird_labels,
    load_perch_labels,
)
from mock_common.config import load_config


BUFFER_FIELDNAMES = [
    "buffer_index",
    "source_file",
    "file_buffer_index",
    "timestamp_utc",
    "section_start_s",
    "section_end_s",
    "retention_reason",
    "audio_saved",
    "gate_mode",
    "gate_threshold",
    "gate_trigger_count",
    "retained_clip_count",
    "retained_seconds",
    "max_nz_bird_common_name",
    "max_nz_bird_scientific_name",
    "max_nz_bird_logit",
    "max_perch_label",
    "max_perch_logit",
    "inference_seconds",
]

FRAME_FIELDNAMES = [
    "buffer_index",
    "source_file",
    "file_buffer_index",
    "timestamp_utc",
    "segment_index",
    "absolute_start_s",
    "absolute_end_s",
    "max_perch_label",
    "max_perch_logit",
    "top_nz_common_name",
    "top_nz_scientific_name",
    "top_nz_perch_label",
    "top_nz_logit",
    "top_excluded_label",
    "top_excluded_logit",
    "nz_over_excluded_margin",
    "bio_margin_threshold",
    "excluded_top_label_gate",
    "margin_gate",
    "frame_bio_gate",
    "nz_top3_json",
]

CLIP_FIELDNAMES = [
    "buffer_index",
    "retention_index",
    "retention_reason",
    "filepath",
    "start_segment_index",
    "end_segment_index",
    "absolute_start_s",
    "absolute_end_s",
    "duration_s",
    "triggered_frame_count",
]


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def utc_now_string() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def progress_bar(*args: Any, **kwargs: Any) -> Any:
    if _tqdm is None:
        iterable = args[0] if args else kwargs.get("iterable", [])
        return iterable
    return _tqdm(*args, **kwargs)


def default_output_dir(output_root: Path, run_name: str | None = None, now: datetime | None = None) -> Path:
    now = now or datetime.now()
    resolved_run_name = run_name or f"run-{now:%H%M}"
    return output_root / now.strftime("%d%m%y") / resolved_run_name


def default_config_path() -> Path:
    local_path = SCRIPT_DIR / "single_file_gate_test.local.yaml"
    shorthand_path = SCRIPT_DIR / "single_file_gate_test.yaml"
    example_path = SCRIPT_DIR / "single_file_gate_test.example.yaml"
    if local_path.exists():
        return local_path
    if shorthand_path.exists():
        return shorthand_path
    return example_path


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_repo_path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    data["_config_path"] = str(config_path)
    return data


def load_audio_section(
    audio_file: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    perch_sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load one section as source-rate audio and Perch-rate audio."""

    info = sf.info(audio_file)
    source_sample_rate = int(info.samplerate)
    start_frame = int(round(start_seconds * source_sample_rate))
    end_frame = int(round(end_seconds * source_sample_rate))
    section_frames = max(0, end_frame - start_frame)
    if section_frames <= 0:
        raise ValueError("Configured section has no audio")

    block, source_sample_rate = sf.read(
        audio_file,
        start=start_frame,
        frames=section_frames,
        dtype="float32",
        always_2d=True,
    )
    source_mono = np.mean(block, axis=1, dtype=np.float32)
    if source_sample_rate == perch_sample_rate:
        perch_audio = source_mono
    elif source_sample_rate == 48000 and perch_sample_rate == 32000:
        perch_audio = resample_poly(source_mono, up=2, down=3).astype(np.float32)
    else:
        raise ValueError(f"Unsupported sample-rate conversion: {source_sample_rate} Hz to {perch_sample_rate} Hz")
    return source_mono, perch_audio.astype(np.float32, copy=False), int(source_sample_rate), int(perch_sample_rate)


def make_windows(audio: np.ndarray, sample_rate: int, *, window_seconds: float, include_partial: bool) -> np.ndarray:
    window_samples = int(round(sample_rate * window_seconds))
    remainder = len(audio) % window_samples
    if remainder:
        if include_partial:
            audio = np.pad(audio, (0, window_samples - remainder)).astype(np.float32)
        else:
            audio = np.asarray(audio[: len(audio) - remainder], dtype=np.float32)
    if len(audio) == 0:
        raise ValueError("Selected section contains no complete Perch windows")
    return audio.reshape((-1, window_samples)).astype(np.float32, copy=False)


def run_perch_windows(model: Any, windows: np.ndarray, *, batch_size: int | None = 24) -> tuple[np.ndarray, float]:
    import tensorflow as tf

    started = time.perf_counter()
    signature = _serving_signature(model)
    if batch_size is None or batch_size <= 0:
        batch_size = len(windows)

    logits_batches = []
    batch_starts = range(0, len(windows), batch_size)
    for start in progress_bar(batch_starts, desc="Running Perch batches", unit="batch"):
        batch = windows[start : start + batch_size]
        outputs = signature(inputs=tf.convert_to_tensor(batch, dtype=tf.float32))
        logits_batches.append(_pick_output(outputs, ("label", "logits")).numpy())
    return np.concatenate(logits_batches, axis=0), time.perf_counter() - started


def utc_offset(seconds: float) -> Any:
    from datetime import timedelta

    return timedelta(seconds=seconds)


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(np.floor(index))
    upper = int(np.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


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
        raise ValueError("plot_style must be a YAML mapping")
    style.update(configured)
    style["line_alpha"] = float(style["line_alpha"])
    style["clip_bar_alpha"] = float(style["clip_bar_alpha"])
    style["line_width"] = float(style["line_width"])
    style["clip_bar_height_fraction"] = float(style["clip_bar_height_fraction"])
    return style


def resolve_section_end_seconds(value: Any, file_duration_seconds: float) -> float:
    if value is None:
        return float(file_duration_seconds)
    if isinstance(value, str) and not value.strip():
        return float(file_duration_seconds)
    return float(value)


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


def plot_gate_events(
    audio: np.ndarray,
    sample_rate: int,
    clips: list[Any],
    *,
    start_seconds: float,
    output_path: Path,
    show: bool,
    plot_style: dict[str, Any],
) -> None:
    import matplotlib.pyplot as plt

    minute_samples = int(round(60 * sample_rate))
    minute_count = max(1, int(np.ceil(len(audio) / minute_samples)))
    fig_height = max(3.0, float(minute_count))
    fig, axes = plt.subplots(
        nrows=minute_count,
        ncols=1,
        figsize=(8, fig_height),
        squeeze=False,
        constrained_layout=True,
    )
    axes_list = list(axes[:, 0])

    for minute_index in progress_bar(range(minute_count), desc="Writing gate plot", unit="minute"):
        axis = axes_list[minute_index]
        chunk = audio[minute_index * minute_samples : (minute_index + 1) * minute_samples]
        if len(chunk) < minute_samples:
            chunk = np.pad(chunk, (0, minute_samples - len(chunk))).astype(np.float32)
        mel_db = compute_mel_db(chunk, sample_rate)
        axis.imshow(
            mel_db,
            origin="lower",
            aspect="auto",
            extent=[0, 60, 0, mel_db.shape[0]],
            cmap="magma",
            vmin=-80,
            vmax=0,
        )
        row_start = start_seconds + minute_index * 60
        axis.set_xlim(0, 60)
        axis.set_ylabel(f"{row_start:.0f}s", rotation=0, labelpad=18, va="center")
        axis.tick_params(axis="y", left=False, labelleft=False)
        if minute_index < minute_count - 1:
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Seconds within minute")

    for clip in clips:
        start = clip.start_offset_s
        end = clip.end_offset_s
        start_minute = int(start // 60)
        end_minute = int(max(end - 1e-9, 0.0) // 60)
        for minute_index in range(start_minute, end_minute + 1):
            if not 0 <= minute_index < minute_count:
                continue
            minute_start = minute_index * 60.0
            minute_end = minute_start + 60.0
            bar_start = max(start, minute_start) - minute_start
            bar_end = min(end, minute_end) - minute_start
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
                start % 60,
                color=plot_style["start_line_color"],
                alpha=plot_style["line_alpha"],
                linewidth=plot_style["line_width"],
            )
        if 0 <= end_minute < minute_count:
            axes_list[end_minute].axvline(
                end % 60,
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
            label="bio gate start",
        ),
        plt.Line2D(
            [0],
            [0],
            color=plot_style["end_line_color"],
            alpha=plot_style["line_alpha"],
            linewidth=plot_style["line_width"],
            label="bio gate end",
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
    fig.suptitle("Single-file bio-gate retained spans", fontsize=12)
    axes_list[0].legend(handles=handles, loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    if show:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)


def run_gate_test(config_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_yaml_config(config_path)
    edge_config_path = resolve_repo_path(config.get("edge_config_path", "edge_node_mock/config/edge_config.local.yaml"))
    edge_config = load_config(edge_config_path)

    audio_file = resolve_repo_path(config["audio_file"])
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    start_seconds = float(config.get("section_start_seconds", 0.0))
    info = sf.info(audio_file)
    end_seconds = resolve_section_end_seconds(config.get("section_end_seconds"), float(info.duration))
    buffer_seconds = float(config.get("buffer_seconds", 15.0))
    include_partial = bool(config.get("include_partial_final_buffer", False))
    save_audio = bool(config.get("save_retained_audio", False))

    output_root = resolve_repo_path(config.get("output_root", "outputs/single_file_gate_tests"))
    configured_run_name = config.get("run_name")
    run_name = str(configured_run_name) if configured_run_name else None
    output_dir = default_output_dir(output_root, run_name=run_name)
    if output_dir.exists():
        if bool(config.get("overwrite", False)):
            shutil.rmtree(output_dir)
        else:
            suffix = time.strftime("%S")
            output_dir = output_dir.with_name(f"{output_dir.name}-{suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)
    retained_audio_dir = output_dir / "retained_audio"
    if save_audio:
        retained_audio_dir.mkdir(parents=True, exist_ok=True)

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

    buffer_csv_path = output_dir / "buffer_metrics.csv"
    frame_csv_path = output_dir / "frame_metrics.csv"
    clip_csv_path = output_dir / "retained_clips.csv"
    plot_path = output_dir / "gate_plot.png"
    summary_path = output_dir / "summary.json"
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    counters: dict[str, Counter[str]] = {
        "retention_reasons": Counter(),
        "top_nz_common_names": Counter(),
        "top_excluded_labels": Counter(),
        "veto_labels": Counter(),
        "clip_durations": Counter(),
    }
    inference_samples: list[float] = []
    margin_samples: list[float] = []
    retained_seconds = 0.0
    buffers_processed = 0
    frames_processed = 0
    frame_triggers = 0
    buffer_triggers = 0
    retained_clip_count = 0

    source_audio, perch_audio, source_sample_rate, perch_sample_rate = load_audio_section(
        audio_file,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        perch_sample_rate=int(edge_config.get("perch_sample_rate", 32000)),
    )
    perch_window_seconds = gate_config.perch_window_seconds
    windows = make_windows(
        perch_audio,
        perch_sample_rate,
        window_seconds=perch_window_seconds,
        include_partial=include_partial,
    )
    inference_batch_size = int(config.get("inference_batch_size", 24))
    logits, inference_seconds = run_perch_windows(model, windows, batch_size=inference_batch_size)
    inference_samples.append(inference_seconds)

    frame_scores = score_frames(
        logits,
        perch_labels=perch_labels,
        nz_label_indexes=nz_labels,
        excluded_margin_label_indexes=excluded_margin_label_indexes,
        bio_margin_threshold=gate_config.bio_margin_threshold,
    )
    retained_clips = build_variable_retention_buffers(
        frame_scores,
        max_frames=gate_config.max_variable_buffer_frames,
        perch_window_seconds=perch_window_seconds,
    )

    section_timestamp = timestamp_from_filename(audio_file) + utc_offset(start_seconds)
    section_buffer = AudioBuffer(
        source_file=audio_file,
        file_buffer_index=0,
        timestamp_utc=section_timestamp,
        source_audio=source_audio,
        source_sample_rate=source_sample_rate,
        perch_audio=perch_audio,
        perch_sample_rate=perch_sample_rate,
    )
    if save_audio and retained_clips:
        saved_clips = []
        for clip in retained_clips:
            saved_path = save_retained_audio(
                section_buffer,
                retained_audio_dir,
                str(edge_config["device_id"]),
                clip,
            )
            saved_clips.append(replace(clip, filepath=saved_path))
        retained_clips = saved_clips

    if not bool(config.get("no_gate_plot", False)):
        plot_style = resolve_plot_style(config)
        plot_gate_events(
            source_audio,
            source_sample_rate,
            retained_clips,
            start_seconds=start_seconds,
            output_path=plot_path,
            show=bool(config.get("show_gate_plot", True)),
            plot_style=plot_style,
        )
    else:
        plot_style = resolve_plot_style(config)

    frames_processed = len(frame_scores)
    frame_triggers = sum(1 for score in frame_scores if score.frame_bio_gate)
    buffer_seconds = float(config.get("buffer_seconds", 15.0))
    frames_per_buffer = max(1, int(round(buffer_seconds / perch_window_seconds)))
    buffers_processed = int(np.ceil(frames_processed / frames_per_buffer))
    buffer_triggers = 0
    retained_clip_count = len(retained_clips)
    retained_seconds = sum(clip.duration_s for clip in retained_clips)
    clip_by_buffer = {
        buffer_index: [
            clip
            for clip in retained_clips
            if clip.start_segment_index // frames_per_buffer == buffer_index - 1
        ]
        for buffer_index in range(1, buffers_processed + 1)
    }

    with (
        buffer_csv_path.open("w", newline="", encoding="utf-8") as buffer_file,
        frame_csv_path.open("w", newline="", encoding="utf-8") as frame_file,
        clip_csv_path.open("w", newline="", encoding="utf-8") as clip_file,
    ):
        buffer_writer = csv.DictWriter(buffer_file, fieldnames=BUFFER_FIELDNAMES)
        frame_writer = csv.DictWriter(frame_file, fieldnames=FRAME_FIELDNAMES)
        clip_writer = csv.DictWriter(clip_file, fieldnames=CLIP_FIELDNAMES)
        buffer_writer.writeheader()
        frame_writer.writeheader()
        clip_writer.writeheader()

        for buffer_index in range(1, buffers_processed + 1):
            start_frame_index = (buffer_index - 1) * frames_per_buffer
            end_frame_index = min(start_frame_index + frames_per_buffer, frames_processed)
            group = frame_scores[start_frame_index:end_frame_index]
            if not group:
                continue
            buffer_start_s = start_seconds + start_frame_index * perch_window_seconds
            buffer_end_s = start_seconds + end_frame_index * perch_window_seconds
            max_nz_score = max(group, key=lambda score: float("-inf") if score.top_nz_logit is None else score.top_nz_logit)
            max_perch_score = max(group, key=lambda score: score.max_perch_logit)
            group_clips = clip_by_buffer.get(buffer_index, [])
            gate_trigger_count = sum(1 for score in group if score.frame_bio_gate)
            if gate_trigger_count:
                buffer_triggers += 1
            counters["retention_reasons"]["bio_hit" if gate_trigger_count else "dropped"] += 1
            write_row(
                buffer_writer,
                {
                    "buffer_index": buffer_index,
                    "source_file": str(audio_file),
                    "file_buffer_index": buffer_index - 1,
                    "timestamp_utc": (section_timestamp + utc_offset((buffer_index - 1) * buffer_seconds)).isoformat().replace("+00:00", "Z"),
                    "section_start_s": buffer_start_s,
                    "section_end_s": buffer_end_s,
                    "retention_reason": "bio_hit" if gate_trigger_count else "dropped",
                    "audio_saved": int(bool(group_clips)),
                    "gate_mode": gate_config.bio_gate_mode,
                    "gate_threshold": gate_config.bio_margin_threshold,
                    "gate_trigger_count": gate_trigger_count,
                    "retained_clip_count": len(group_clips),
                    "retained_seconds": sum(clip.duration_s for clip in group_clips),
                    "max_nz_bird_common_name": max_nz_score.top_nz_common_name,
                    "max_nz_bird_scientific_name": max_nz_score.top_nz_scientific_name,
                    "max_nz_bird_logit": max_nz_score.top_nz_logit,
                    "max_perch_label": max_perch_score.max_perch_label,
                    "max_perch_logit": max_perch_score.max_perch_logit,
                    "inference_seconds": inference_seconds / buffers_processed if buffers_processed else inference_seconds,
                },
            )

        for score in frame_scores:
            absolute_start = start_seconds + score.segment_index * perch_window_seconds
            absolute_end = absolute_start + perch_window_seconds
            if score.top_nz_common_name:
                counters["top_nz_common_names"][score.top_nz_common_name] += 1
            if score.top_excluded_label:
                counters["top_excluded_labels"][score.top_excluded_label] += 1
            if score.excluded_top_label_gate and score.top_excluded_label:
                counters["veto_labels"][score.top_excluded_label] += 1
            if score.nz_over_excluded_margin is not None:
                margin_samples.append(float(score.nz_over_excluded_margin))

            write_row(
                frame_writer,
                {
                    "buffer_index": score.segment_index // frames_per_buffer + 1,
                    "source_file": str(audio_file),
                    "file_buffer_index": score.segment_index // frames_per_buffer,
                    "timestamp_utc": (section_timestamp + utc_offset(score.segment_index * perch_window_seconds)).isoformat().replace("+00:00", "Z"),
                    "segment_index": score.segment_index,
                    "absolute_start_s": absolute_start,
                    "absolute_end_s": absolute_end,
                    "max_perch_label": score.max_perch_label,
                    "max_perch_logit": score.max_perch_logit,
                    "top_nz_common_name": score.top_nz_common_name,
                    "top_nz_scientific_name": score.top_nz_scientific_name,
                    "top_nz_perch_label": score.top_nz_perch_label,
                    "top_nz_logit": score.top_nz_logit,
                    "top_excluded_label": score.top_excluded_label,
                    "top_excluded_logit": score.top_excluded_logit,
                    "nz_over_excluded_margin": score.nz_over_excluded_margin,
                    "bio_margin_threshold": score.bio_margin_threshold,
                    "excluded_top_label_gate": int(score.excluded_top_label_gate),
                    "margin_gate": int(score.margin_gate),
                    "frame_bio_gate": int(score.frame_bio_gate),
                    "nz_top3_json": json.dumps(score.top_nz_birds, separators=(",", ":")),
                },
            )

        for clip in retained_clips:
            absolute_start = start_seconds + clip.start_offset_s
            absolute_end = start_seconds + clip.end_offset_s
            counters["clip_durations"][f"{clip.duration_s:g}s"] += 1
            write_row(
                clip_writer,
                {
                    "buffer_index": clip.start_segment_index // frames_per_buffer + 1,
                    "retention_index": clip.retention_index,
                    "retention_reason": clip.retention_reason,
                    "filepath": clip.filepath,
                    "start_segment_index": clip.start_segment_index,
                    "end_segment_index": clip.end_segment_index,
                    "absolute_start_s": absolute_start,
                    "absolute_end_s": absolute_end,
                    "duration_s": clip.duration_s,
                    "triggered_frame_count": clip.triggered_frame_count,
                },
            )

    summary = {
        "schema_version": 2,
        "created_at_utc": utc_now_string(),
        "config_path": config["_config_path"],
        "edge_config_path": str(edge_config_path),
        "audio_file": str(audio_file),
        "section_start_seconds": start_seconds,
        "section_end_seconds": end_seconds,
        "section_duration_seconds": end_seconds - start_seconds,
        "buffer_seconds": buffer_seconds,
        "perch_window_count": int(len(windows)),
        "inference_batch_size": inference_batch_size,
        "buffers_processed": buffers_processed,
        "frames_processed": frames_processed,
        "frame_triggers": frame_triggers,
        "frame_trigger_rate": frame_triggers / frames_processed if frames_processed else None,
        "bio_hit_buffers": buffer_triggers,
        "bio_hit_buffer_rate": buffer_triggers / buffers_processed if buffers_processed else None,
        "retained_clip_count": retained_clip_count,
        "retained_seconds": retained_seconds,
        "retained_audio_saved": save_audio,
        "plot_style": plot_style,
        "gate": {
            "mode": gate_config.bio_gate_mode,
            "threshold": gate_config.bio_margin_threshold,
            "excluded_margin_labels": gate_config.excluded_margin_labels,
            "max_variable_buffer_frames": gate_config.max_variable_buffer_frames,
        },
        "model": {
            "source": model_source,
            "handle_or_path": model_ref,
            "load_seconds": model_load_seconds,
        },
        "timing": {
            "elapsed_seconds": time.perf_counter() - started,
            "total_inference_seconds": sum(inference_samples),
            "mean_inference_seconds": mean(inference_samples) if inference_samples else None,
            "median_inference_seconds": median(inference_samples) if inference_samples else None,
            "p95_inference_seconds": percentile(inference_samples, 0.95),
        },
        "margin_distribution": {
            "count": len(margin_samples),
            "min": min(margin_samples) if margin_samples else None,
            "median": median(margin_samples) if margin_samples else None,
            "p95": percentile(margin_samples, 0.95),
            "max": max(margin_samples) if margin_samples else None,
        },
        "counts": {
            name: [{"label": label, "count": count} for label, count in counter.most_common(30)]
            for name, counter in counters.items()
        },
        "outputs": {
            "output_dir": str(output_dir),
            "summary_json": str(summary_path),
            "buffer_metrics_csv": str(buffer_csv_path),
            "frame_metrics_csv": str(frame_csv_path),
            "retained_clips_csv": str(clip_csv_path),
            "gate_plot_png": str(plot_path) if not bool(config.get("no_gate_plot", False)) else None,
            "retained_audio_dir": str(retained_audio_dir) if save_audio else None,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Summary written to: {summary_path}")
    print(f"Frame metrics written to: {frame_csv_path}")
    print(f"Retained clips written to: {clip_csv_path}")
    if not bool(config.get("no_gate_plot", False)):
        print(f"Gate plot written to: {plot_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=default_config_path(),
        help=(
            "YAML config describing the file, section, output directory, and plot settings. "
            "Defaults to scripts/single_file_gate_test.local.yaml when present, then "
            "scripts/single_file_gate_test.yaml, then scripts/single_file_gate_test.example.yaml."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_gate_test(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
