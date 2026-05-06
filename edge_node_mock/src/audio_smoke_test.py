"""Confirm the configured raw-audio stream can feed the edge mock."""

from __future__ import annotations

import argparse
import math
import struct
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mock_common.config import load_config


def find_audio_files(raw_audio_mount: str, raw_audio_glob: str) -> list[Path]:
    root = Path(raw_audio_mount)
    return sorted(path for path in root.glob(raw_audio_glob) if path.is_file())


def inspect_wav(path: Path) -> dict[str, object]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frame_count = wav.getnframes()
        duration = frame_count / sample_rate
        raw = wav.readframes(frame_count)

    peak = _pcm_peak_as_float(raw, sample_width, channels)
    return {
        "sample_rate": sample_rate,
        "duration_seconds": duration,
        "channels": channels,
        "sample_width_bytes": sample_width,
        "source_dtype": f"pcm_s{sample_width * 8}",
        "mono_dtype": "float32",
        "peak_amplitude": peak,
    }


def _pcm_peak_as_float(raw: bytes, sample_width: int, channels: int) -> float:
    if sample_width != 2:
        raise ValueError(f"Only 16-bit PCM WAV smoke checks are supported; got {sample_width * 8}-bit")

    if not raw:
        return 0.0

    sample_count = len(raw) // sample_width
    samples = struct.unpack(f"<{sample_count}h", raw)
    max_peak = 0.0

    if channels == 1:
        for sample in samples:
            max_peak = max(max_peak, abs(sample) / 32768.0)
        return max_peak

    for frame_start in range(0, len(samples), channels):
        frame = samples[frame_start : frame_start + channels]
        mono_sample = sum(frame) / len(frame)
        max_peak = max(max_peak, abs(mono_sample) / 32768.0)
    return max_peak


def run_smoke_check(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    raw_audio_mount = str(config["raw_audio_mount"])
    raw_audio_glob = str(config.get("raw_audio_glob", "*.wav"))
    audio_files = find_audio_files(raw_audio_mount, raw_audio_glob)
    if not audio_files:
        raise FileNotFoundError(f"No WAV files found at {raw_audio_mount!r} with glob {raw_audio_glob!r}")

    selected_file = audio_files[0]
    info = inspect_wav(selected_file)

    allowed_channels = {1, 2}
    if info["channels"] not in allowed_channels:
        raise ValueError(f"Expected mono or stereo audio, got {info['channels']} channels")

    allowed_sample_rates = set(config.get("source_sample_rates", [48000]))
    if info["sample_rate"] not in allowed_sample_rates:
        rates = ", ".join(str(rate) for rate in sorted(allowed_sample_rates))
        raise ValueError(f"Expected sample rate in {{{rates}}}, got {info['sample_rate']}")

    if not math.isfinite(float(info["peak_amplitude"])):
        raise ValueError("Peak amplitude is not finite")

    return {
        "search_path": f"{raw_audio_mount}/{raw_audio_glob}",
        "selected_file": str(selected_file),
        **info,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="edge_node_mock/config/edge_config.example.yaml")
    args = parser.parse_args(argv)

    try:
        report = run_smoke_check(args.config)
    except Exception as exc:
        print(f"Audio smoke check failed: {exc}", file=sys.stderr)
        return 1

    print("Audio smoke check passed")
    for key, value in report.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
