"""Margin bio-gate scoring and variable retained-clip grouping."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class FrameScores:
    segment_index: int
    max_noise_label: str | None
    max_noise_logit: float | None
    max_nz_bird_common_name: str | None
    max_nz_bird_logit: float | None
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

    @property
    def max_bio_label(self) -> str | None:
        return self.max_nz_bird_common_name

    @property
    def max_bio_logit(self) -> float | None:
        return self.max_nz_bird_logit


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


def score_frames(
    logits: Any,
    *,
    perch_labels: list[str],
    nz_label_indexes: dict[int, Any],
    excluded_margin_label_indexes: dict[str, int],
    bio_margin_threshold: float,
) -> list[FrameScores]:
    """Score each Perch frame with the locked Phase 1 NZ-bird margin gate."""

    import numpy as np

    frame_scores: list[FrameScores] = []
    nz_indexes = list(nz_label_indexes)
    for segment_index, frame_logits in enumerate(logits):
        max_perch_index = int(np.argmax(frame_logits))
        max_perch_label = perch_labels[max_perch_index]
        max_perch_logit = float(frame_logits[max_perch_index])
        top_excluded_label, top_excluded_logit = _max_configured_label(
            frame_logits,
            excluded_margin_label_indexes,
        )
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
        frame_scores.append(
            FrameScores(
                segment_index=segment_index,
                max_noise_label=top_excluded_label,
                max_noise_logit=top_excluded_logit,
                max_nz_bird_common_name=top_nz_common_name,
                max_nz_bird_logit=top_nz_logit,
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
    """Group consecutive triggered frames inside one inference buffer."""

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


def retained_clip_filename(
    *,
    device_id: str,
    timestamp_utc: datetime,
    source_stem: str,
    file_buffer_index: int,
    clip: RetentionClip,
) -> str:
    timestamp = timestamp_utc.strftime("%Y%m%dT%H%M%SZ")
    duration = f"{clip.duration_s:g}s"
    return (
        f"{device_id}_{timestamp}_{clip.retention_reason}_{source_stem}_{file_buffer_index:03d}"
        f"_seg{clip.start_segment_index}-{clip.end_segment_index}_{duration}.flac"
    )
