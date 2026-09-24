from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.result_reporting import (
    build_method_parameter_rows,
    write_method_parameter_summary,
)


class MethodParameterReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "voxel_sizes_m": [0.20, 0.10, 0.05],
            "max_correspondence_distances_m": [0.80, 0.40, 0.20],
            "max_iterations": [80, 50, 30],
            "evaluation_max_correspondence_m": 0.20,
            "success_rotation_deg": 2.0,
            "success_translation_m": 0.10,
            "lambda_geometric_rgb": 0.968,
            "lambda_geo_spectral": 0.968,
            "lambda_spec_spectral": 0.032,
            "coarse_registration": {
                "enabled": True,
                "candidate_count": 8,
                "ransac_n": 3,
                "max_iterations": 30000,
                "confidence": 0.999,
                "max_rotation_deg": 25.0,
                "max_translation_m": 1.25,
                "methods": {
                    "geometry": {
                        "config_id": "geometry_selected",
                        "voxel_size_m": 0.10,
                        "max_correspondence_m": 0.15,
                        "normal_radius_multiplier": 3.0,
                        "lambda_geo": 0.9,
                        "lambda_attribute": 0.0,
                        "lambda_overlap": 0.1,
                    },
                    "rgb": {
                        "config_id": "rgb_selected",
                        "voxel_size_m": 0.10,
                        "max_correspondence_m": 0.15,
                        "normal_radius_multiplier": 2.0,
                        "attribute_feature_weight": 2.0,
                        "lambda_geo": 0.64,
                        "lambda_attribute": 0.16,
                        "lambda_overlap": 0.2,
                    },
                    "spectral6": {
                        "config_id": "spectral_selected",
                        "voxel_size_m": 0.10,
                        "max_correspondence_m": 0.15,
                        "normal_radius_multiplier": 2.0,
                        "attribute_feature_weight": 2.0,
                        "lambda_geo": 0.56,
                        "lambda_attribute": 0.24,
                        "lambda_overlap": 0.2,
                    },
                },
            },
        }
        self.wavelengths = np.array([495, 696, 600, 803, 545, 642])

    def test_parameter_rows_distinguish_all_three_modalities(self) -> None:
        rows = build_method_parameter_rows(self.config, self.wavelengths)

        self.assertEqual([row["method"] for row in rows], [
            "geometry",
            "rgb",
            "spectral6",
        ])
        self.assertEqual(rows[0]["fine_estimation"], "point_to_plane")
        self.assertEqual(rows[0]["coarse_config_id"], "geometry_selected")
        self.assertEqual(rows[1]["rgb_geometric_weight"], 0.968)
        self.assertAlmostEqual(rows[1]["rgb_weight"], 0.032)
        self.assertEqual(rows[2]["spectral_geometric_weight"], 0.968)
        self.assertEqual(rows[2]["spectral_weight"], 0.032)
        self.assertEqual(
            rows[2]["wavelengths_nm"],
            "[495,696,600,803,545,642]",
        )
        self.assertEqual(rows[0]["overlap_thresholds_m"], "[0.1,0.25]")

    def test_writer_creates_csv_and_markdown_with_three_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            rows = write_method_parameter_summary(
                output_dir,
                self.config,
                self.wavelengths,
            )

            with (output_dir / "method_parameters_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                csv_rows = list(csv.DictReader(handle))
            markdown = (
                output_dir / "method_parameters_summary.md"
            ).read_text(encoding="utf-8")
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(csv_rows), 3)
            self.assertEqual(csv_rows[2]["coarse_modality"], "spectral6")
            self.assertIn("geometry", markdown)
            self.assertIn("RGB Colored ICP", markdown)
            self.assertIn("六波段", markdown)


if __name__ == "__main__":
    unittest.main()
