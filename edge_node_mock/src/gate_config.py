"""Validation helpers for the active Phase 1 margin-gate configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MarginGateConfig:
    bio_gate_mode: str
    bio_margin_threshold: float
    excluded_margin_labels: list[str]
    max_variable_buffer_frames: int
    perch_window_seconds: float


def validate_margin_gate_config(
    config: dict[str, Any],
    *,
    perch_labels: list[str] | None = None,
) -> MarginGateConfig:
    """Validate and normalize the active Phase 1 margin-gate config."""

    gate_mode = str(config.get("bio_gate_mode", "")).strip()
    if gate_mode != "nz_bird_margin":
        raise ValueError("bio_gate_mode must be nz_bird_margin")

    try:
        threshold = float(config["bio_margin_threshold"])
    except KeyError as exc:
        raise ValueError("bio_margin_threshold is required") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("bio_margin_threshold must be numeric") from exc

    excluded_labels = config.get("excluded_margin_labels")
    if not isinstance(excluded_labels, list) or not excluded_labels:
        raise ValueError("excluded_margin_labels must be a non-empty list")
    normalized_excluded_labels = [str(label) for label in excluded_labels]

    if perch_labels is not None:
        label_set = set(perch_labels)
        missing = [label for label in normalized_excluded_labels if label not in label_set]
        if missing:
            raise ValueError(f"excluded_margin_labels contains labels not present in Perch labels: {missing}")

    try:
        perch_window_seconds = float(config["perch_window_seconds"])
    except KeyError as exc:
        raise ValueError("perch_window_seconds is required") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("perch_window_seconds must be numeric") from exc
    if perch_window_seconds <= 0:
        raise ValueError("perch_window_seconds must be > 0")

    try:
        max_variable_buffer_frames = int(config["max_variable_buffer_frames"])
    except KeyError as exc:
        raise ValueError("max_variable_buffer_frames is required") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError("max_variable_buffer_frames must be an integer") from exc
    if max_variable_buffer_frames < 1:
        raise ValueError("max_variable_buffer_frames must be >= 1")

    return MarginGateConfig(
        bio_gate_mode=gate_mode,
        bio_margin_threshold=threshold,
        excluded_margin_labels=normalized_excluded_labels,
        max_variable_buffer_frames=max_variable_buffer_frames,
        perch_window_seconds=perch_window_seconds,
    )
