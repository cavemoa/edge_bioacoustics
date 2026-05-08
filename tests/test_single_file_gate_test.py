from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from datetime import datetime

from scripts.run_single_file_gate_test import (
    default_output_dir,
    load_audio_section,
    make_windows,
    resolve_plot_style,
    resolve_section_end_seconds,
)


class SingleFileGateTestRunnerTests(unittest.TestCase):
    def test_load_audio_section_reads_requested_span(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "20230527_213004.wav"
            sf.write(audio_path, np.ones(32_000 * 20, dtype=np.float32), 32_000)

            source_audio, perch_audio, source_rate, perch_rate = load_audio_section(
                audio_path,
                start_seconds=5.0,
                end_seconds=15.0,
                perch_sample_rate=32_000,
            )

        self.assertEqual(source_rate, 32_000)
        self.assertEqual(perch_rate, 32_000)
        self.assertEqual(len(source_audio), 32_000 * 10)
        self.assertEqual(len(perch_audio), 32_000 * 10)

    def test_make_windows_trims_or_pads_partial_frame(self) -> None:
        audio = np.zeros(32_000 * 12, dtype=np.float32)

        trimmed = make_windows(audio, 32_000, window_seconds=5.0, include_partial=False)
        padded = make_windows(audio, 32_000, window_seconds=5.0, include_partial=True)

        self.assertEqual(trimmed.shape, (2, 160_000))
        self.assertEqual(padded.shape, (3, 160_000))

    def test_default_output_dir_uses_date_and_run_time(self) -> None:
        path = default_output_dir(Path("outputs/single"), now=datetime(2026, 5, 8, 8, 54))

        self.assertEqual(path, Path("outputs/single/080526/run-0854"))

    def test_blank_section_end_uses_file_duration(self) -> None:
        self.assertEqual(resolve_section_end_seconds(None, 123.4), 123.4)
        self.assertEqual(resolve_section_end_seconds("", 123.4), 123.4)
        self.assertEqual(resolve_section_end_seconds("  ", 123.4), 123.4)
        self.assertEqual(resolve_section_end_seconds(77.0, 123.4), 77.0)

    def test_plot_style_merges_defaults_and_coerces_numbers(self) -> None:
        style = resolve_plot_style(
            {
                "plot_style": {
                    "start_line_color": "green",
                    "line_alpha": "0.5",
                    "clip_bar_height_fraction": "0.1",
                }
            }
        )

        self.assertEqual(style["start_line_color"], "green")
        self.assertEqual(style["end_line_color"], "red")
        self.assertAlmostEqual(style["line_alpha"], 0.5)
        self.assertAlmostEqual(style["clip_bar_height_fraction"], 0.1)


if __name__ == "__main__":
    unittest.main()
