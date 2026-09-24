from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from visualize_registration import (
    COLOR_MODES,
    coordinate_frame_spec,
    colors_for_mode,
    load_point_cloud,
    resolve_input_file,
    screenshot_path,
    view_state_path,
)


class RegistrationViewerTests(unittest.TestCase):
    def make_ply(self, directory: Path) -> tuple[Path, np.ndarray, np.ndarray]:
        path = directory / "example cloud.ply"
        points = np.array(
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [4.0, 5.0, 8.0]],
            dtype=np.float64,
        )
        colors = np.array(
            [[0.0, 0.75, 0.1], [0.9, 0.1, 0.1], [0.2, 0.3, 0.4]],
            dtype=np.float64,
        )
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        cloud.colors = o3d.utility.Vector3dVector(colors)
        self.assertTrue(o3d.io.write_point_cloud(str(path), cloud))
        return path, points, colors

    def test_cli_file_is_used_without_opening_chooser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = self.make_ply(Path(tmp))

            resolved = resolve_input_file(
                path,
                Path(tmp),
                chooser=lambda _: self.fail("不应调用GUI选择器"),
            )

            self.assertEqual(resolved, path.resolve())

    def test_missing_cli_file_uses_gui_chooser(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = self.make_ply(Path(tmp))

            resolved = resolve_input_file(None, Path(tmp), chooser=lambda _: path)

            self.assertEqual(resolved, path.resolve())

    def test_load_preserves_ply_colors_and_supports_scalar_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, points, colors = self.make_ply(Path(tmp))

            data = load_point_cloud(path)

            np.testing.assert_allclose(data.points, points)
            np.testing.assert_allclose(data.original_colors, colors, atol=1 / 255)
            self.assertEqual(COLOR_MODES, ("original", "height", "distance", "x", "y", "z"))
            np.testing.assert_allclose(colors_for_mode(data, "original"), data.original_colors)
            for mode in COLOR_MODES[1:]:
                mapped = colors_for_mode(data, mode)
                self.assertEqual(mapped.shape, points.shape)
                self.assertTrue(np.all((mapped >= 0.0) & (mapped <= 1.0)))

    def test_view_state_is_unique_to_selected_file(self) -> None:
        path = Path("/tmp/trial_002/source_target_merged.ply")
        self.assertEqual(
            view_state_path(path),
            Path("/tmp/trial_002/.source_target_merged.ply.view.json"),
        )

    def test_coordinate_frame_uses_true_origin_and_screenshot_names_do_not_collide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, _, _ = self.make_ply(Path(tmp))
            data = load_point_cloud(path)

            origin, size = coordinate_frame_spec(data)
            np.testing.assert_array_equal(origin, np.zeros(3))
            self.assertGreater(size, 0.0)
            first = screenshot_path(path, datetime(2026, 7, 26, 19, 0, 0, 1))
            second = screenshot_path(path, datetime(2026, 7, 26, 19, 0, 0, 2))
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, path.parent)


if __name__ == "__main__":
    unittest.main()
