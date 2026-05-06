from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from edge_node_mock.src.bio_capture_loop import (
    build_label_index,
    decide_buffer,
    iter_audio_buffers,
    score_frames,
)


class BioCaptureLoopTest(unittest.TestCase):
    def test_configured_label_indexes_are_validated(self) -> None:
        labels = ["Animal", "Rain", "Wild_animals"]

        indexes = build_label_index(labels, ["Rain", "Animal"], group_name="test_labels")

        self.assertEqual(indexes, {"Rain": 1, "Animal": 0})
        with self.assertRaisesRegex(ValueError, "not present"):
            build_label_index(labels, ["Missing"], group_name="test_labels")

    def test_score_and_decision_use_configured_bio_and_noise_labels(self) -> None:
        perch_labels = ["Animal", "Rain", "Wild_animals", "Other"]
        logits = np.array(
            [
                [1.0, 4.0, 0.5, 0.0],
                [2.5, 1.0, 0.2, 0.0],
                [0.3, 0.1, 3.2, 0.0],
            ],
            dtype=np.float32,
        )

        frame_scores = score_frames(
            logits,
            perch_labels=perch_labels,
            noise_label_indexes={"Rain": 1},
            bio_label_indexes={"Animal": 0, "Wild_animals": 2},
            nz_label_indexes={2: _Label("Wild animals", "Wildus animalis")},
        )
        decision = decide_buffer(
            frame_scores,
            bio_threshold=3.0,
            noise_threshold=5.0,
            validation_sample_interval=100,
            dropped_buffer_count=0,
        )

        self.assertEqual(decision.retention_reason, "bio_hit")
        self.assertEqual(decision.max_bio_label, "Wild_animals")
        self.assertAlmostEqual(decision.max_bio_logit, 3.2, places=5)
        self.assertEqual(json.loads(decision.nz_bird_logits)[0]["top_3"][0]["perch_label_number"], 2)

    def test_validation_sample_cadence_applies_to_dropped_buffers(self) -> None:
        frame_scores = [
            _FrameScore(segment_index=0, max_noise_label="Rain", max_noise_logit=6.0, max_bio_label="Animal", max_bio_logit=1.0),
            _FrameScore(segment_index=1, max_noise_label="Rain", max_noise_logit=5.5, max_bio_label="Animal", max_bio_logit=1.2),
            _FrameScore(segment_index=2, max_noise_label="Rain", max_noise_logit=5.1, max_bio_label="Animal", max_bio_logit=1.1),
        ]

        decision = decide_buffer(
            frame_scores,
            bio_threshold=3.0,
            noise_threshold=5.0,
            validation_sample_interval=2,
            dropped_buffer_count=1,
        )

        self.assertEqual(decision.retention_reason, "validation_sample")
        self.assertTrue(decision.audio_saved)

    def test_audio_buffer_iterator_reads_32k_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "20260101_000000.wav"
            sf.write(audio_path, np.zeros(32000 * 30, dtype=np.float32), 32000)
            config = {
                "raw_audio_mount": str(root),
                "raw_audio_glob": "*.wav",
                "perch_sample_rate": 32000,
            }

            buffers = list(iter_audio_buffers(config))

        self.assertEqual(len(buffers), 2)
        self.assertEqual(len(buffers[0].perch_audio), 32000 * 15)
        self.assertEqual(buffers[1].file_buffer_index, 1)

    def test_audio_buffer_iterator_can_pad_partial_final_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audio_path = root / "20260101_000000.wav"
            sf.write(audio_path, np.ones(32000 * 16, dtype=np.float32), 32000)
            config = {
                "raw_audio_mount": str(root),
                "raw_audio_glob": "*.wav",
                "perch_sample_rate": 32000,
            }

            without_partial = list(iter_audio_buffers(config))
            with_partial = list(iter_audio_buffers(config, include_partial=True))

        self.assertEqual(len(without_partial), 1)
        self.assertEqual(len(with_partial), 2)
        self.assertEqual(len(with_partial[1].perch_audio), 32000 * 15)
        self.assertTrue(np.all(with_partial[1].perch_audio[32000:] == 0.0))


class _Label:
    def __init__(self, common_name: str, scientific_name: str) -> None:
        self.common_name = common_name
        self.scientific_name = scientific_name


class _FrameScore:
    def __init__(
        self,
        *,
        segment_index: int,
        max_noise_label: str,
        max_noise_logit: float,
        max_bio_label: str,
        max_bio_logit: float,
    ) -> None:
        self.segment_index = segment_index
        self.max_noise_label = max_noise_label
        self.max_noise_logit = max_noise_logit
        self.max_bio_label = max_bio_label
        self.max_bio_logit = max_bio_logit
        self.max_perch_label = max_noise_label
        self.max_perch_logit = max_noise_logit
        self.top_nz_birds = []


if __name__ == "__main__":
    unittest.main()
