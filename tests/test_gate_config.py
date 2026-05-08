from __future__ import annotations

import unittest

from edge_node_mock.src.gate_config import validate_margin_gate_config


class GateConfigTest(unittest.TestCase):
    def test_valid_margin_gate_config_is_normalized(self) -> None:
        config = {
            "bio_gate_mode": "nz_bird_margin",
            "bio_margin_threshold": "0.55",
            "excluded_margin_labels": ["Water", "Train", "Vehicle"],
            "perch_window_seconds": "5.0",
            "max_variable_buffer_frames": "3",
        }

        gate_config = validate_margin_gate_config(
            config,
            perch_labels=["Water", "Train", "Vehicle", "Other"],
        )

        self.assertEqual(gate_config.bio_gate_mode, "nz_bird_margin")
        self.assertAlmostEqual(gate_config.bio_margin_threshold, 0.55)
        self.assertEqual(gate_config.excluded_margin_labels, ["Water", "Train", "Vehicle"])
        self.assertAlmostEqual(gate_config.perch_window_seconds, 5.0)
        self.assertEqual(gate_config.max_variable_buffer_frames, 3)

    def test_rejects_unsupported_gate_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "bio_gate_mode"):
            validate_margin_gate_config(
                {
                    "bio_gate_mode": "legacy_threshold",
                    "bio_margin_threshold": 0.55,
                    "excluded_margin_labels": ["Water"],
                    "perch_window_seconds": 5.0,
                    "max_variable_buffer_frames": 3,
                }
            )

    def test_rejects_bad_threshold_and_empty_excluded_labels(self) -> None:
        base = {
            "bio_gate_mode": "nz_bird_margin",
            "bio_margin_threshold": 0.55,
            "excluded_margin_labels": ["Water"],
            "perch_window_seconds": 5.0,
            "max_variable_buffer_frames": 3,
        }

        with self.assertRaisesRegex(ValueError, "bio_margin_threshold"):
            validate_margin_gate_config({**base, "bio_margin_threshold": "not-a-number"})
        with self.assertRaisesRegex(ValueError, "excluded_margin_labels"):
            validate_margin_gate_config({**base, "excluded_margin_labels": []})

    def test_rejects_excluded_labels_missing_from_perch_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            validate_margin_gate_config(
                {
                    "bio_gate_mode": "nz_bird_margin",
                    "bio_margin_threshold": 0.55,
                    "excluded_margin_labels": ["Water", "Missing"],
                    "perch_window_seconds": 5.0,
                    "max_variable_buffer_frames": 3,
                },
                perch_labels=["Water"],
            )

    def test_rejects_bad_window_and_buffer_frame_values(self) -> None:
        base = {
            "bio_gate_mode": "nz_bird_margin",
            "bio_margin_threshold": 0.55,
            "excluded_margin_labels": ["Water"],
            "perch_window_seconds": 5.0,
            "max_variable_buffer_frames": 3,
        }

        with self.assertRaisesRegex(ValueError, "perch_window_seconds"):
            validate_margin_gate_config({**base, "perch_window_seconds": 0})
        with self.assertRaisesRegex(ValueError, "max_variable_buffer_frames"):
            validate_margin_gate_config({**base, "max_variable_buffer_frames": 0})


if __name__ == "__main__":
    unittest.main()
