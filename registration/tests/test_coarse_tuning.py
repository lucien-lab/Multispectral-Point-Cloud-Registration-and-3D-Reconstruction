from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.coarse_tuning import (
    generate_search_configs,
    select_best_config,
    summarize_search_results,
    write_tuning_outputs,
)


class CoarseTuningTests(unittest.TestCase):
    def test_balanced_search_spaces_have_exact_sizes_and_shared_budget(self) -> None:
        common = {
            "enabled": True,
            "feature_radius_multiplier": 5.0,
            "candidate_count": 8,
            "ransac_n": 3,
            "max_iterations": 30000,
            "confidence": 0.999,
            "max_rotation_deg": 25.0,
            "max_translation_m": 1.25,
            "score_max_distance_m": 0.3,
            "trim_fraction": 0.7,
            "min_correspondences": 30,
            "min_coverage": 0.1,
            "min_score_improvement": 0.03,
        }

        spaces = {
            method: generate_search_configs(method, common)
            for method in ("geometry", "rgb", "spectral6")
        }

        self.assertEqual(len(spaces["geometry"]), 18)
        self.assertEqual(len(spaces["rgb"]), 27)
        self.assertEqual(len(spaces["spectral6"]), 27)
        for configs in spaces.values():
            self.assertEqual(
                len({config["config_id"] for config in configs}),
                len(configs),
            )
            for config in configs:
                self.assertEqual(config["candidate_count"], 8)
                self.assertEqual(config["ransac_n"], 3)
                self.assertEqual(config["max_iterations"], 30000)
                self.assertAlmostEqual(
                    config["lambda_geo"]
                    + config["lambda_attribute"]
                    + config["lambda_overlap"],
                    1.0,
                )

    def test_selection_uses_only_trials_one_to_five(self) -> None:
        rows = []
        for trial in range(1, 6):
            rows.extend(
                [
                    {
                        "config_id": "train_winner",
                        "trial": trial,
                        "status": "ok",
                        "rotation_error_deg": 1.0,
                        "translation_error_m": 0.05,
                        "runtime_s": 2.0,
                    },
                    {
                        "config_id": "heldout_winner",
                        "trial": trial,
                        "status": "ok",
                        "rotation_error_deg": 3.0,
                        "translation_error_m": 0.15,
                        "runtime_s": 1.0,
                    },
                ]
            )
        for trial in range(6, 11):
            rows.extend(
                [
                    {
                        "config_id": "train_winner",
                        "trial": trial,
                        "status": "ok",
                        "rotation_error_deg": 8.0,
                        "translation_error_m": 0.8,
                        "runtime_s": 2.0,
                    },
                    {
                        "config_id": "heldout_winner",
                        "trial": trial,
                        "status": "ok",
                        "rotation_error_deg": 0.1,
                        "translation_error_m": 0.01,
                        "runtime_s": 1.0,
                    },
                ]
            )

        summaries = summarize_search_results(rows)
        selected = select_best_config(summaries)

        self.assertEqual(selected["config_id"], "train_winner")
        self.assertEqual(selected["success_count"], 5)
        self.assertEqual(selected["trial_count"], 5)

    def test_selection_applies_all_declared_tie_breakers(self) -> None:
        summaries = [
            {
                "config_id": "slow",
                "success_count": 4,
                "normalized_error_mean": 1.0,
                "translation_error_m_median": 0.05,
                "runtime_s_mean": 3.0,
            },
            {
                "config_id": "fast_b",
                "success_count": 4,
                "normalized_error_mean": 1.0,
                "translation_error_m_median": 0.05,
                "runtime_s_mean": 2.0,
            },
            {
                "config_id": "fast_a",
                "success_count": 4,
                "normalized_error_mean": 1.0,
                "translation_error_m_median": 0.05,
                "runtime_s_mean": 2.0,
            },
            {
                "config_id": "more_error",
                "success_count": 4,
                "normalized_error_mean": 1.2,
                "translation_error_m_median": 0.03,
                "runtime_s_mean": 1.0,
            },
            {
                "config_id": "fewer_successes",
                "success_count": 3,
                "normalized_error_mean": 0.2,
                "translation_error_m_median": 0.01,
                "runtime_s_mean": 1.0,
            },
        ]

        selected = select_best_config(summaries)

        self.assertEqual(selected["config_id"], "fast_a")

    def test_output_writer_saves_full_selected_config_and_search_rows(self) -> None:
        configs = [
            {
                "config_id": "winner",
                "voxel_size_m": 0.12,
                "lambda_geo": 0.8,
                "lambda_attribute": 0.0,
                "lambda_overlap": 0.2,
            }
        ]
        rows = [
            {
                "method": "geometry",
                "config_id": "winner",
                "trial": trial,
                "status": "ok",
                "rotation_error_deg": 0.5,
                "translation_error_m": 0.04,
                "runtime_s": 1.0,
            }
            for trial in range(1, 6)
        ]

        with tempfile.TemporaryDirectory() as tmp:
            selected = write_tuning_outputs(
                Path(tmp),
                "geometry",
                configs,
                rows,
            )
            selected_payload = json.loads(
                (Path(tmp) / "selected_config.json").read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(selected["config_id"], "winner")
            self.assertEqual(
                selected_payload["config"]["voxel_size_m"],
                0.12,
            )
            self.assertEqual(
                len(
                    (Path(tmp) / "search_results.csv")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ),
                6,
            )
            self.assertTrue((Path(tmp) / "tuning_summary.md").is_file())


if __name__ == "__main__":
    unittest.main()
