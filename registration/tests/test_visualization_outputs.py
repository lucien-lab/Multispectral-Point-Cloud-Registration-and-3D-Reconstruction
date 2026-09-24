from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.visualization import (
    SOURCE_COLOR,
    TARGET_COLOR,
    write_coarse_registration_clouds,
    write_registration_clouds,
)


class TrueRgbMergedOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source_perturbed = np.array(
            [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]],
            dtype=np.float64,
        )
        self.source_registered = self.source_perturbed + np.array([0.1, -0.2, 0.3])
        self.target = np.array(
            [[4.0, 5.0, 6.0], [4.5, 5.5, 6.5], [5.0, 6.0, 7.0]],
            dtype=np.float64,
        )
        self.source_rgb = np.array(
            [[0.10, 0.20, 0.30], [0.35, 0.45, 0.55]],
            dtype=np.float64,
        )
        self.target_rgb = np.array(
            [[0.80, 0.15, 0.05], [0.70, 0.25, 0.10], [0.60, 0.35, 0.20]],
            dtype=np.float64,
        )

    def assert_cloud(
        self,
        path: Path,
        expected_points: np.ndarray,
        expected_colors: np.ndarray,
    ) -> None:
        cloud = o3d.io.read_point_cloud(str(path))
        self.assertFalse(cloud.is_empty(), path.name)
        np.testing.assert_allclose(
            np.asarray(cloud.points),
            expected_points,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(cloud.colors),
            expected_colors,
            atol=1.0 / 255.0 + 1e-6,
        )

    def test_fine_writer_preserves_true_rgb_and_diagnostic_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            write_registration_clouds(
                output_dir,
                self.source_perturbed,
                self.source_registered,
                self.target,
                self.source_rgb,
                self.target_rgb,
            )

            merged_points = np.vstack((self.source_registered, self.target))
            initial_points = np.vstack((self.source_perturbed, self.target))
            diagnostic_colors = np.vstack(
                (
                    np.tile(SOURCE_COLOR, (len(self.source_perturbed), 1)),
                    np.tile(TARGET_COLOR, (len(self.target), 1)),
                )
            )
            self.assert_cloud(
                output_dir / "source_target_initial_merged.ply",
                initial_points,
                diagnostic_colors,
            )
            self.assert_cloud(
                output_dir / "source_target_rgb_merged.ply",
                merged_points,
                np.vstack((self.source_rgb, self.target_rgb)),
            )
            self.assert_cloud(
                output_dir / "source_target_merged.ply",
                merged_points,
                diagnostic_colors,
            )

    def test_coarse_writer_preserves_true_rgb_and_diagnostic_colors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            write_coarse_registration_clouds(
                output_dir,
                self.source_registered,
                self.target,
                self.source_rgb,
                self.target_rgb,
            )

            merged_points = np.vstack((self.source_registered, self.target))
            self.assert_cloud(
                output_dir / "source_target_coarse_rgb_merged.ply",
                merged_points,
                np.vstack((self.source_rgb, self.target_rgb)),
            )
            self.assert_cloud(
                output_dir / "source_target_coarse_merged.ply",
                merged_points,
                np.vstack(
                    (
                        np.tile(SOURCE_COLOR, (len(self.source_registered), 1)),
                        np.tile(TARGET_COLOR, (len(self.target), 1)),
                    )
                ),
            )


if __name__ == "__main__":
    unittest.main()
