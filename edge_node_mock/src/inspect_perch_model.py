"""Inspect the configured Perch model, labels, and 15-second input contract."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mock_common.config import load_config


@dataclass(frozen=True)
class NzBirdLabel:
    perch_label_number: int
    common_name: str
    scientific_name: str
    perch_label: str


@dataclass(frozen=True)
class PerchInspectionReport:
    model_source: str
    model_handle_or_path: str
    audio_file: str
    sample_rate: int
    buffer_seconds: float
    model_input_shape: list[int | None]
    logits_shape: list[int]
    embeddings_shape: list[int]
    label_count: int
    embedding_dim: int
    expected_embedding_dim: int
    frame_count: int
    window_seconds: float
    nz_bird_label_count: int
    first_nz_bird_labels: list[dict[str, Any]]


def load_perch_labels(path: str | Path) -> list[str]:
    """Load Perch labels as a zero-based numeric index list."""

    label_path = Path(path)
    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if labels and labels[0] == "inat2024_fsd50k":
        labels = labels[1:]
    if not labels:
        raise ValueError(f"No Perch labels found in {label_path}")
    return labels


def load_nz_bird_labels(path: str | Path, perch_labels: list[str]) -> dict[int, NzBirdLabel]:
    """Load the NZ bird subset and verify numeric labels index Perch labels."""

    nz_path = Path(path)
    mapped: dict[int, NzBirdLabel] = {}
    with nz_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"perch_label_number", "common_name", "scientific_name"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"{nz_path} must contain columns: {sorted(required_columns)}")

        for row in reader:
            label_number = int(row["perch_label_number"])
            if label_number < 0 or label_number >= len(perch_labels):
                raise ValueError(
                    f"NZ label {label_number} is outside Perch label range 0..{len(perch_labels) - 1}"
                )
            mapped[label_number] = NzBirdLabel(
                perch_label_number=label_number,
                common_name=row["common_name"],
                scientific_name=row["scientific_name"],
                perch_label=perch_labels[label_number],
            )

    if not mapped:
        raise ValueError(f"No NZ bird labels found in {nz_path}")
    return mapped


def load_audio_buffer(config: dict[str, Any], *, seconds: float = 15.0) -> tuple[Any, int, Path]:
    """Load one configured audio buffer as mono float32 at the Perch sample rate."""

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    raw_audio_root = Path(str(config["raw_audio_mount"]))
    raw_audio_glob = str(config.get("raw_audio_glob", "*.wav"))
    audio_files = sorted(path for path in raw_audio_root.glob(raw_audio_glob) if path.is_file())
    if not audio_files:
        raise FileNotFoundError(f"No audio files found at {raw_audio_root}/{raw_audio_glob}")

    selected_file = audio_files[0]
    info = sf.info(selected_file)
    source_sample_rate = int(info.samplerate)
    frames_to_read = int(round(source_sample_rate * seconds))
    audio, sample_rate = sf.read(
        selected_file,
        frames=frames_to_read,
        dtype="float32",
        always_2d=True,
    )
    mono = np.mean(audio, axis=1, dtype=np.float32)

    target_sample_rate = int(config.get("perch_sample_rate", 32000))
    if sample_rate == target_sample_rate:
        resampled = mono
    elif sample_rate == 48000 and target_sample_rate == 32000:
        resampled = resample_poly(mono, up=2, down=3).astype(np.float32)
    else:
        raise ValueError(f"Unsupported sample-rate conversion: {sample_rate} Hz to {target_sample_rate} Hz")

    expected_samples = int(round(target_sample_rate * seconds))
    if len(resampled) < expected_samples:
        resampled = np.pad(resampled, (0, expected_samples - len(resampled)))
    elif len(resampled) > expected_samples:
        resampled = resampled[:expected_samples]

    return resampled.astype(np.float32, copy=False), target_sample_rate, selected_file


def make_perch_windows(audio: Any, sample_rate: int, *, window_seconds: float = 5.0) -> Any:
    """Split a 15-second mono buffer into three 5-second Perch windows."""

    import numpy as np

    window_samples = int(round(sample_rate * window_seconds))
    if len(audio) % window_samples != 0:
        raise ValueError(f"Audio length {len(audio)} is not divisible by {window_samples} samples")
    windows = np.asarray(audio, dtype=np.float32).reshape((-1, window_samples))
    if windows.shape[0] != 3:
        raise ValueError(f"Expected exactly 3 Perch windows, got {windows.shape[0]}")
    return windows


def inspect_perch_model(config_path: str | Path) -> PerchInspectionReport:
    config = load_config(config_path)
    expected_embedding_dim = int(config["embedding_dim"])
    perch_labels = load_perch_labels(config["perch_label_path"])
    nz_labels = load_nz_bird_labels(config["nz_bird_label_path"], perch_labels)
    audio, sample_rate, audio_file = load_audio_buffer(config)
    windows = make_perch_windows(audio, sample_rate)
    model, model_source, model_ref = _load_model(config)
    signature = _serving_signature(model)
    outputs = signature(inputs=_to_tensor(windows))
    logits = _pick_output(outputs, ("label", "logits"))
    embeddings = _pick_output(outputs, ("embedding", "embeddings"))

    logits_shape = [int(dim) for dim in logits.shape]
    embeddings_shape = [int(dim) for dim in embeddings.shape]
    if logits_shape != [3, len(perch_labels)]:
        raise ValueError(f"Expected logits shape [3, {len(perch_labels)}], got {logits_shape}")
    if embeddings_shape != [3, expected_embedding_dim]:
        raise ValueError(f"Expected embeddings shape [3, {expected_embedding_dim}], got {embeddings_shape}")

    input_shape = _input_shape(signature)
    return PerchInspectionReport(
        model_source=model_source,
        model_handle_or_path=model_ref,
        audio_file=str(audio_file),
        sample_rate=sample_rate,
        buffer_seconds=len(audio) / sample_rate,
        model_input_shape=input_shape,
        logits_shape=logits_shape,
        embeddings_shape=embeddings_shape,
        label_count=len(perch_labels),
        embedding_dim=embeddings_shape[1],
        expected_embedding_dim=expected_embedding_dim,
        frame_count=windows.shape[0],
        window_seconds=windows.shape[1] / sample_rate,
        nz_bird_label_count=len(nz_labels),
        first_nz_bird_labels=[asdict(label) for label in list(nz_labels.values())[:5]],
    )


def _load_model(config: dict[str, Any]) -> tuple[Any, str, str]:
    model_path = config.get("perch_model_path")
    if model_path:
        import tensorflow as tf

        model_ref = str(model_path)
        return tf.saved_model.load(model_ref), "saved_model", model_ref

    model_ref = str(config.get("perch_tfhub_handle") or "").strip()
    if not model_ref:
        raise ValueError("Set either perch_model_path or perch_tfhub_handle in the edge config")

    import tensorflow_hub as hub

    return hub.load(model_ref), "tfhub", model_ref


def _serving_signature(model: Any) -> Any:
    signatures = getattr(model, "signatures", None)
    if not signatures:
        raise ValueError("Loaded model does not expose TensorFlow signatures")
    if "serving_default" in signatures:
        return signatures["serving_default"]
    return next(iter(signatures.values()))


def _to_tensor(value: Any) -> Any:
    import tensorflow as tf

    return tf.convert_to_tensor(value, dtype=tf.float32)


def _pick_output(outputs: dict[str, Any], preferred_keys: tuple[str, ...]) -> Any:
    for key in preferred_keys:
        if key in outputs:
            return outputs[key]
    raise ValueError(f"Model outputs missing one of {preferred_keys}; available keys: {sorted(outputs)}")


def _input_shape(signature: Any) -> list[int | None]:
    _, keyword_specs = signature.structured_input_signature
    input_spec = keyword_specs.get("inputs") or next(iter(keyword_specs.values()))
    return [None if dim is None else int(dim) for dim in input_spec.shape]


def write_report(report: PerchInspectionReport, path: str | Path) -> None:
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="edge_node_mock/config/edge_config.example.yaml")
    parser.add_argument("--report-json", default="docs/02_implementation/perch_model_inspection_report.json")
    args = parser.parse_args(argv)

    try:
        report = inspect_perch_model(args.config)
    except Exception as exc:
        print(f"Perch model inspection failed: {exc}", file=sys.stderr)
        return 1

    write_report(report, args.report_json)
    print("Perch model inspection passed")
    for key, value in asdict(report).items():
        print(f"{key}: {value}")
    print(f"report_json: {args.report_json}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    raise SystemExit(main())
