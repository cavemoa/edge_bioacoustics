from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from edge_node_mock.src.inspect_perch_model import (
    load_nz_bird_labels,
    load_perch_labels,
    make_perch_windows,
)


class PerchInspectionTest(unittest.TestCase):
    def test_perch_labels_are_zero_based_after_metadata_row(self) -> None:
        labels = load_perch_labels("labels/perch_label.csv")

        self.assertEqual(len(labels), 14795)
        self.assertEqual(labels[0], "Abavorana luctuosa")
        self.assertIn("Pterodroma gouldi", labels)

    def test_nz_bird_labels_map_numeric_ids_to_perch_labels(self) -> None:
        labels = load_perch_labels("labels/perch_label.csv")
        nz_labels = load_nz_bird_labels("labels/north_island_nz_perch_lablel.csv", labels)

        self.assertEqual(nz_labels[798].common_name, "New Zealand Bellbird")
        self.assertEqual(nz_labels[798].scientific_name, "Anthornis melanura")
        self.assertEqual(nz_labels[798].perch_label, labels[798])

    def test_nz_bird_label_validation_rejects_out_of_range_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad_nz_labels.csv"
            path.write_text(
                "perch_label_number,common_name,scientific_name\n"
                "99,Imaginary Bird,Testus birdus\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "outside Perch label range"):
                load_nz_bird_labels(path, ["label-0"])

    def test_make_perch_windows_splits_15_seconds_into_three_frames(self) -> None:
        audio = np.zeros(32000 * 15, dtype=np.float32)

        windows = make_perch_windows(audio, 32000)

        self.assertEqual(windows.shape, (3, 160000))
        self.assertEqual(windows.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
