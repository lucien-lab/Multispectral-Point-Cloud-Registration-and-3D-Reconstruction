from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from tune_coarse_registration import run_tuning


def make_input(path: Path, points: np.ndarray, side: str) -> None:
    shifted = points - points.min(axis=0)
    span = np.maximum(np.ptp(points, axis=0), 1e-9)
    rgb = shifted / span
    x, y, z = points.T
    spectral = np.column_stack(
        [0.3 + 0.02 * (band + 1) * x + 0.01 * y + 0.005 * z for band in range(6)]
    )
    np.savez_compressed(
        path,
        xyz_mm=points * 1000.0,
        rgb=rgb,
        reflectance=spectral,
        wavelengths_nm=np.array([495, 696, 600, 803, 545, 642]),
        source_scan_index=np.arange(1, len(points) + 1),
        valid=np.ones(len(points), dtype=bool),
        side=np.array(side),
    )


class TuneCoarseRegistrationTests(unittest.TestCase):
    def test_small_tuning_run_writes_selected_geometry_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            values = np.linspace(-0.4, 0.4, 6)
            a, b = np.meshgrid(values, values)
            points = np.unique(
                np.vstack(
                    (
                        np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size))),
                        np.column_stack((a.ravel(), np.zeros(a.size), b.ravel())),
                        np.column_stack((np.zeros(a.size), a.ravel(), b.ravel())),
                    )
                ),
                axis=0,
            )
            make_input(processed / "left_input.npz", points, "left")
            make_input(processed / "right_input.npz", points, "right")
            config = {
                "seed": 42,
                "trial_count": 10,
                "max_rotation_deg": 2.0,
                "max_translation_m": 0.02,
                "evaluation_max_correspondence_m": 0.2,
                "success_rotation_deg": 2.0,
                "success_translation_m": 0.1,
                "coarse_registration": {
                    "enabled": True,
                    "feature_radius_multiplier": 5.0,
                    "candidate_count": 0,
                    "ransac_n": 3,
                    "max_iterations": 30000,
                    "confidence": 0.999,
                    "max_rotation_deg": 25.0,
                    "max_translation_m": 1.25,
                    "score_max_distance_m": 0.3,
                    "trim_fraction": 0.7,
                    "min_correspondences": 20,
                    "min_coverage": 0.1,
                    "min_score_improvement": 0.03,
                },
                "data": {
                    "left_input": "data/processed/left_input.npz",
                    "right_input": "data/processed/right_input.npz",
                    "left_prepared": "data/processed/left_points.npz",
                    "right_prepared": "data/processed/right_points.npz",
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            search_config = {
                **config["coarse_registration"],
                "modality": "geometry",
                "config_id": "small_geometry",
                "voxel_size_m": 0.12,
                "max_correspondence_m": 0.18,
                "normal_radius_multiplier": 2.0,
                "lambda_geo": 0.8,
                "lambda_attribute": 0.0,
                "lambda_overlap": 0.2,
            }

            selected = run_tuning(
                config_path,
                methods=("geometry",),
                experiment_root=root,
                search_spaces={"geometry": [search_config]},
                trial_numbers=(1,),
            )

            self.assertEqual(selected["geometry"]["config_id"], "small_geometry")
            output = root / "results" / "coarse_tuning" / "geometry"
            self.assertTrue((output / "search_results.csv").is_file())
            self.assertTrue((output / "selected_config.json").is_file())
            self.assertTrue(
                (root / "results" / "coarse_tuning_summary.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
