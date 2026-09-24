from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).with_name("visualize_pointcloud.py")


def load_module():
    spec = importlib.util.spec_from_file_location("visualize_pointcloud", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VisualizePointCloudTests(unittest.TestCase):
    def test_side_mode_is_default_and_maps_source_colors(self):
        module = load_module()
        with mock.patch.object(sys, "argv", ["visualize_pointcloud.py"]):
            args = module.parse_args()

        self.assertEqual(args.color, "side")
        self.assertIn("side", module.COLOR_MODES)

        data = {"side_name": np.array(["left", "right", "unknown"])}
        actual = module.colors_for_mode(
            data,
            np.arange(3),
            "side",
            brightness=1.0,
        )
        np.testing.assert_allclose(
            actual,
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                ]
            ),
        )

    def test_rgb_mode_remains_available(self):
        module = load_module()
        with mock.patch.object(
            sys,
            "argv",
            ["visualize_pointcloud.py", "--color", "rgb"],
        ):
            self.assertEqual(module.parse_args().color, "rgb")

        data = {
            "rgb": np.array(
                [
                    [0.2, 0.3, 0.4],
                    [0.8, 0.9, 1.0],
                ]
            )
        }

        actual = module.colors_for_mode(
            data,
            np.array([0, 1]),
            "rgb",
            brightness=1.5,
        )

        np.testing.assert_allclose(
            actual,
            np.array(
                [
                    [0.3, 0.45, 0.6],
                    [1.0, 1.0, 1.0],
                ]
            ),
        )

    def test_scalar_color_modes_remain_available(self):
        module = load_module()
        data = {
            "xyz_mm": np.array(
                [
                    [0.0, 0.0, 100.0],
                    [0.0, 0.0, 200.0],
                    [0.0, 0.0, 300.0],
                ]
            ),
            "range_m": np.array([6.0, 7.0, 8.0]),
            "reflectance": np.arange(18, dtype=float).reshape(3, 6),
        }
        indices = np.arange(3)

        height = module.colors_for_mode(data, indices, "height", 1.0)
        distance = module.colors_for_mode(data, indices, "distance", 1.0)
        self.assertEqual(height.shape, (3, 3))
        self.assertEqual(distance.shape, (3, 3))
        self.assertTrue(np.isfinite(height).all())
        self.assertTrue(np.isfinite(distance).all())

        for wavelength in module.WAVELENGTHS:
            colors = module.colors_for_mode(
                data,
                indices,
                f"rfl-{wavelength}",
                1.0,
            )
            self.assertEqual(colors.shape, (3, 3))
            self.assertTrue(np.isfinite(colors).all())

    def test_color_shortcuts_keep_zero_through_nine_mapping(self):
        module = load_module()
        expected = {
            "0": "side",
            "1": "rgb",
            "2": "height",
            "3": "distance",
            "4": "rfl-495",
            "5": "rfl-696",
            "6": "rfl-600",
            "7": "rfl-803",
            "8": "rfl-545",
            "9": "rfl-642",
        }

        self.assertEqual(
            getattr(module, "KEY_COLOR_MODES", None),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
