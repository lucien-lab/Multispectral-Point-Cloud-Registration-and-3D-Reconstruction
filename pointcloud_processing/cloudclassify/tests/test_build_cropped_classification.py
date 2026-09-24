import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).parents[1]
SOURCE_PATH = PROJECT_DIR.parent / "cropped_point_cloud.txt"


class CroppedClassificationTests(unittest.TestCase):
    def test_real_files_build_a_complete_forty_class_mapping(self):
        from build_cropped_classification import (
            build_classification,
            load_cropped_point_cloud,
        )

        points = load_cropped_point_cloud(SOURCE_PATH)
        labels, mapping = build_classification(points, PROJECT_DIR)

        self.assertEqual(points.shape, (36840, 8))
        self.assertEqual(labels.shape, (36840,))
        self.assertEqual(set(labels), set(range(40)))
        self.assertEqual(len(mapping), 40)
        self.assertEqual(mapping[0]["class_name"], "palette_background")
        self.assertEqual(mapping[24]["class_name"], "palette_color_24")
        self.assertEqual(mapping[25]["class_name"], "bottles")
        self.assertEqual(mapping[39]["class_name"], "trash_bin")
        self.assertEqual(sum(row["point_count"] for row in mapping), len(points))

    def test_writer_preserves_seven_source_columns_and_appends_class_id(self):
        from build_cropped_classification import write_classified_point_cloud

        points = np.array([[2058, 1, 2, 3, 4, 5, 6, 0]], dtype=float)
        labels = np.array([25], dtype=int)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "classified.txt"
            write_classified_point_cloud(output, points, labels)
            with output.open(encoding="utf-8") as handle:
                self.assertEqual(
                    handle.readline().strip(),
                    "point_id,x,y,z,r,g,b,class_id",
                )
            loaded = np.loadtxt(output, delimiter=",", skiprows=1)
        self.assertEqual(loaded.shape, (8,))
        self.assertEqual(int(loaded[-1]), 25)


if __name__ == "__main__":
    unittest.main()
