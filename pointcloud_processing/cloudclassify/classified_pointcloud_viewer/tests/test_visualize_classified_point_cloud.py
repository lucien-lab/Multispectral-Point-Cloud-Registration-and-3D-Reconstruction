from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import visualize_classified_point_cloud as viewer


class PointCloudDataTests(unittest.TestCase):
    def write_text_file(self, content: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "cloud.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def test_load_classified_point_cloud_normalizes_rgb(self) -> None:
        path = self.write_text_file(
            "point_id,x,y,z,r,g,b,class_id\n"
            "1,1,2,3,255,128,0,7\n"
        )

        xyz, rgb, class_ids = viewer.load_classified_point_cloud(path)

        np.testing.assert_allclose(xyz, [[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(rgb, [[1.0, 128 / 255, 0.0]])
        np.testing.assert_array_equal(class_ids, [7])

    def test_class_colors_is_stable_and_groups_equal_labels(self) -> None:
        colors = viewer.class_colors(np.array([5, 2, 5]))

        np.testing.assert_allclose(colors[0], colors[2])
        self.assertFalse(np.allclose(colors[0], colors[1]))

    def test_loader_rejects_missing_required_column(self) -> None:
        path = self.write_text_file("x,y,z,r,g,b\n0,0,0,0,0,0\n")

        with self.assertRaisesRegex(ValueError, "class_id"):
            viewer.load_classified_point_cloud(path)


class PointCloudViewerTests(unittest.TestCase):
    def test_viewer_changes_point_size_with_floor_one(self) -> None:
        viewer_state = viewer.PointCloudViewer(
            np.zeros((1, 3)), np.zeros((1, 3)), np.array([0])
        )

        viewer_state.change_point_size(-99)
        self.assertEqual(viewer_state.point_size, 1)
        viewer_state.change_point_size(2)
        self.assertEqual(viewer_state.point_size, 3)

    def test_viewer_switches_between_original_and_class_colors(self) -> None:
        original = np.array([[0.1, 0.2, 0.3]])
        viewer_state = viewer.PointCloudViewer(
            np.zeros((1, 3)), original, np.array([9])
        )

        viewer_state.set_color_mode("class")
        self.assertFalse(np.allclose(viewer_state.active_colors, original))
        viewer_state.set_color_mode("original")
        np.testing.assert_allclose(viewer_state.active_colors, original)


class ViewerStateTests(unittest.TestCase):
    def temp_path(self, name: str) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return Path(directory.name) / name

    def test_state_is_saved_and_loaded_per_resolved_input_path(self) -> None:
        state_path = self.temp_path("viewer-state.json")
        first = self.temp_path("first.txt")
        second = self.temp_path("second.txt")

        viewer.save_viewer_state(
            first, {"color_mode": "class", "point_size": 7}, state_path
        )
        viewer.save_viewer_state(
            second, {"color_mode": "original", "point_size": 2}, state_path
        )

        self.assertEqual(viewer.load_viewer_state(first, state_path)["point_size"], 7)
        self.assertEqual(
            viewer.load_viewer_state(second, state_path)["color_mode"], "original"
        )

    def test_state_uses_the_input_relative_path_as_its_key(self) -> None:
        state_path = self.temp_path("viewer-state.json")
        input_path = Path("examples") / "cloud.txt"

        viewer.save_viewer_state(
            input_path, {"color_mode": "class", "point_size": 4}, state_path
        )

        document = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertIn("examples/cloud.txt", document["files"])
        self.assertFalse(viewer.DEFAULT_STATE_PATH.is_absolute())

    def test_loading_migrates_a_legacy_absolute_path_key(self) -> None:
        state_path = self.temp_path("viewer-state.json")
        input_path = Path("cloud.txt")
        legacy_key = str(Path.cwd() / input_path)
        state_path.write_text(
            json.dumps(
                {
                    "files": {
                        legacy_key: {"color_mode": "class", "point_size": 5}
                    }
                }
            ),
            encoding="utf-8",
        )

        state = viewer.load_viewer_state(input_path, state_path)

        self.assertEqual(state["point_size"], 5)
        document = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertNotIn(legacy_key, document["files"])
        self.assertIn("cloud.txt", document["files"])

    def test_invalid_state_file_falls_back_to_empty_state(self) -> None:
        state_path = self.temp_path("viewer-state.json")
        state_path.write_text("{invalid", encoding="utf-8")

        self.assertEqual(
            viewer.load_viewer_state(self.temp_path("cloud.txt"), state_path), {}
        )

    def test_invalid_state_fields_fall_back_to_empty_state(self) -> None:
        state_path = self.temp_path("viewer-state.json")
        cloud = self.temp_path("cloud.txt")
        state_path.write_text(
            json.dumps(
                {
                    "files": {
                        str(cloud.resolve()): {
                            "color_mode": "unknown",
                            "point_size": 0,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        self.assertEqual(viewer.load_viewer_state(cloud, state_path), {})

    def test_viewer_restores_saved_color_mode_and_point_size(self) -> None:
        state_path = self.temp_path("viewer-state.json")
        cloud = self.temp_path("cloud.txt")
        viewer.save_viewer_state(
            cloud, {"color_mode": "class", "point_size": 6}, state_path
        )

        viewer_state = viewer.PointCloudViewer(
            np.zeros((1, 3)),
            np.array([[0.1, 0.2, 0.3]]),
            np.array([9]),
            input_path=cloud,
            state_path=state_path,
        )

        self.assertEqual(viewer_state.color_mode, "class")
        self.assertEqual(viewer_state.point_size, 6)
        self.assertFalse(
            np.allclose(viewer_state.active_colors, viewer_state.original_colors)
        )

    def test_camera_state_is_converted_to_open3d_parameters(self) -> None:
        import open3d as o3d

        class FakeViewControl:
            def __init__(self) -> None:
                self.parameters = None
                self.allow_arbitrary = None

            def convert_from_pinhole_camera_parameters(self, parameters, allow_arbitrary):
                self.parameters = parameters
                self.allow_arbitrary = allow_arbitrary
                return True

        class FakeVisualizer:
            def __init__(self) -> None:
                self.view_control = FakeViewControl()

            def get_view_control(self):
                return self.view_control

        visualizer = FakeVisualizer()
        camera_state = {
            "intrinsic": {
                "width": 640,
                "height": 480,
                "matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            },
            "extrinsic": np.eye(4).tolist(),
        }

        viewer.PointCloudViewer._restore_camera(visualizer, o3d, camera_state)

        self.assertTrue(visualizer.view_control.allow_arbitrary)
        np.testing.assert_allclose(
            visualizer.view_control.parameters.extrinsic, np.eye(4)
        )


if __name__ == "__main__":
    unittest.main()
