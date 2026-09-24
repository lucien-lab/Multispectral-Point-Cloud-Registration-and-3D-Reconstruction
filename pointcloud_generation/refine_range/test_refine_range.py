from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("refine_range.py")


def load_module():
    assert MODULE_PATH.exists(), "refine_range.py 尚未实现"
    spec = importlib.util.spec_from_file_location("refine_range", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RefineRangeTests(unittest.TestCase):
    def test_default_maximum_range_is_10_metres(self):
        module = load_module()

        self.assertEqual(getattr(module, "DEFAULT_MAX_RANGE_M", None), 10.0)

    def test_range_mask_is_closed_and_respects_valid(self):
        module = load_module()
        ranges = np.array([6.29, 6.30, 7.00, 8.00, 8.01, 7.50])
        valid = np.array([True, True, True, True, True, False])

        actual = module.range_mask(ranges, valid, 6.3, 8.0)

        np.testing.assert_array_equal(
            actual,
            np.array([False, True, True, True, False, False]),
        )

    def test_filter_payload_keeps_rows_aligned_and_records_source_scan_index(self):
        module = load_module()
        payload = {
            "xyz_mm": np.arange(18, dtype=float).reshape(6, 3),
            "range_m": np.array([6.29, 6.30, 7.00, 8.00, 8.01, 7.50]),
            "reflectance": np.arange(36, dtype=float).reshape(6, 6),
            "valid": np.array([True, True, True, True, True, False]),
            "wavelengths_nm": np.array([495, 696, 600, 803, 545, 642]),
        }

        filtered = module.filter_payload(payload, 6.3, 8.0)

        np.testing.assert_array_equal(filtered["source_scan_index"], [2, 3, 4])
        np.testing.assert_array_equal(filtered["range_m"], [6.3, 7.0, 8.0])
        np.testing.assert_array_equal(filtered["xyz_mm"], payload["xyz_mm"][1:4])
        np.testing.assert_array_equal(
            filtered["reflectance"],
            payload["reflectance"][1:4],
        )
        np.testing.assert_array_equal(
            filtered["wavelengths_nm"],
            payload["wavelengths_nm"],
        )


if __name__ == "__main__":
    unittest.main()
