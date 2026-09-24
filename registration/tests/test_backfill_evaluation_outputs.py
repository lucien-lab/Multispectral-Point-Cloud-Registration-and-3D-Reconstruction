from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from backfill_evaluation_outputs import backfill_outputs


class BackfillEvaluationOutputsTests(unittest.TestCase):
    def test_backfill_preserves_existing_fields_and_adds_joint_outputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            processed = root / "data" / "processed"
            processed.mkdir(parents=True)
            points = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.1, 0.0],
                    [0.0, 0.0, 0.1],
                ]
            )
            rgb = np.tile(np.array([[0.2, 0.4, 0.6]]), (len(points), 1))
            spectral = np.tile(np.arange(1, 7, dtype=float), (len(points), 1))
            wavelengths = np.array([495, 696, 600, 803, 545, 642])
            for side in ("left", "right"):
                np.savez_compressed(
                    processed / f"{side}_points.npz",
                    xyz_m=points,
                    rgb=rgb,
                    spectral6=spectral,
                    wavelengths_nm=wavelengths,
                )

            config = json.loads(
                (EXPERIMENT_ROOT / "configs" / "default.json").read_text(
                    encoding="utf-8"
                )
            )
            config["data"]["left_prepared"] = (
                "data/processed/left_points.npz"
            )
            config["data"]["right_prepared"] = (
                "data/processed/right_points.npz"
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            identity_payload = json.dumps({"matrix": np.eye(4).tolist()})
            method_dirs = (
                ("geometry", "icp_geometry"),
                ("rgb", "icp_rgb"),
                ("spectral6", "icp_spectral6"),
            )
            for method, folder in method_dirs:
                trial_dir = root / "results" / folder / "trial_001"
                trial_dir.mkdir(parents=True)
                for filename in ("T_perturb.json", "T_gt.json", "T_est.json"):
                    (trial_dir / filename).write_text(
                        identity_payload,
                        encoding="utf-8",
                    )
                (trial_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "method": method,
                            "trial": 1,
                            "status": "ok",
                            "runtime_s": 1.0,
                            "preserved_marker": "keep-me",
                        }
                    ),
                    encoding="utf-8",
                )

            rows = backfill_outputs(config_path, experiment_root=root)

            self.assertEqual(len(rows), 3)
            for method, folder in method_dirs:
                trial_dir = root / "results" / folder / "trial_001"
                metrics = json.loads(
                    (trial_dir / "metrics.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metrics["preserved_marker"], "keep-me")
                self.assertAlmostEqual(metrics["rotation_error_deg"], 0.0)
                self.assertAlmostEqual(metrics["translation_error_m"], 0.0)
                self.assertAlmostEqual(metrics["overlap_100mm_pct"], 100.0)
                self.assertAlmostEqual(metrics["overlap_250mm_pct"], 100.0)
                cloud = o3d.io.read_point_cloud(
                    str(trial_dir / "source_target_initial_merged.ply")
                )
                self.assertEqual(len(cloud.points), 2 * len(points))

            with (root / "results" / "results_summary.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                summary_rows = list(csv.DictReader(handle))
            self.assertEqual(len(summary_rows), 3)
            self.assertTrue(
                (root / "results" / "method_parameters_summary.csv").is_file()
            )
            self.assertTrue(
                (root / "results" / "method_parameters_summary.md").is_file()
            )


if __name__ == "__main__":
    unittest.main()
