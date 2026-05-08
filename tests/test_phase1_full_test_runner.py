from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from datetime import datetime

from scripts.run_phase1_full_test import (
    RETAINED_CLIP_FIELDNAMES,
    count_audio_plan,
    default_output_dir,
    discover_night_dirs,
    percentile,
    resolve_plot_style,
    run_full_test,
)


class Phase1FullTestRunnerTests(unittest.TestCase):
    def test_discover_night_dirs_prefers_sorted_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "20230528").mkdir()
            (root / "20230527").mkdir()
            self.assertEqual(
                [path.name for path in discover_night_dirs(root)],
                ["20230527", "20230528"],
            )

    def test_discover_night_dirs_falls_back_to_flat_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(discover_night_dirs(root), [root])

    def test_count_audio_plan_counts_full_and_partial_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio = np.zeros(32_000 * 16, dtype=np.float32)
            sf.write(root / "20230527_173004.wav", audio, 32_000)

            without_partial = count_audio_plan(
                root,
                raw_audio_glob="*.wav",
                buffer_seconds=15.0,
                include_partial=False,
            )
            with_partial = count_audio_plan(
                root,
                raw_audio_glob="*.wav",
                buffer_seconds=15.0,
                include_partial=True,
            )

            self.assertEqual(without_partial.expected_buffers, 1)
            self.assertEqual(with_partial.expected_buffers, 2)
            self.assertAlmostEqual(with_partial.audio_seconds, 16.0)
            self.assertEqual(with_partial.source_sample_rates, [32_000])

    def test_percentile_interpolates_sorted_values(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.75), 3.25)
        self.assertIsNone(percentile([], 0.95))

    def test_default_output_dir_uses_date_and_run_time(self) -> None:
        path = default_output_dir(datetime(2026, 5, 8, 8, 54))

        self.assertEqual(path.parts[-3:], ("phase1_full_test", "080526", "run-0854"))

    def test_full_runner_exposes_retained_clip_csv_and_plot_style_contract(self) -> None:
        self.assertIn("filepath", RETAINED_CLIP_FIELDNAMES)
        self.assertIn("duration_seconds", RETAINED_CLIP_FIELDNAMES)
        self.assertIn("triggered_frame_count", RETAINED_CLIP_FIELDNAMES)

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

    def test_full_runner_rejects_skip_audio_save_with_hub_sync(self) -> None:
        args = type(
            "Args",
            (),
            {
                "skip_audio_save": True,
                "sync_to_hub": True,
            },
        )()

        with self.assertRaisesRegex(ValueError, "skip-audio-save"):
            run_full_test(args)


if __name__ == "__main__":
    unittest.main()
