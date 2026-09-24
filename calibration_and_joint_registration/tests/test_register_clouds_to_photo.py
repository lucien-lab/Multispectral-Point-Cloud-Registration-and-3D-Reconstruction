import csv
from pathlib import Path
import tempfile
import unittest

from register_clouds_to_photo import _load_cloud_csv, _validate_output_root


class JointRegistrationCliValidationTest(unittest.TestCase):
    def test_missing_required_column_names_the_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cloud.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["x_mm", "y_mm", "z_mm", "azimuth_rad", "valid"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "x_mm": "1",
                        "y_mm": "2",
                        "z_mm": "3",
                        "azimuth_rad": "0.1",
                        "valid": "1",
                    }
                )

            with self.assertRaisesRegex(ValueError, "elevation_rad"):
                _load_cloud_csv(path, "left")

    def test_output_root_cannot_be_inside_input_pointcloud_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            pointcloud = Path(directory) / "pointcloud_output"
            pointcloud.mkdir()
            left = pointcloud / "left.csv"
            right = pointcloud / "right.csv"
            left.touch()
            right.touch()

            with self.assertRaisesRegex(ValueError, "input point-cloud directory"):
                _validate_output_root(pointcloud / "joint", left, right)


if __name__ == "__main__":
    unittest.main()
