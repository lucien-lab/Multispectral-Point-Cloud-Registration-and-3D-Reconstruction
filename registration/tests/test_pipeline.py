from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from run_experiment import (
    _load_tuned_coarse_configs,
    _run_coarse_registration,
    _run_method,
    _summary_rows,
    run_pipeline,
)
from src.coarse_registration import CoarseRegistrationResult
from src.icp_geometry import ICPResult


def make_input(path: Path, points_m: np.ndarray, side: str) -> None:
    x, y, z = points_m.T
    rgb = np.column_stack(
        (
            np.clip(x - x.min(), 0, 1),
            np.clip(y - y.min(), 0, 1),
            np.clip(z - z.min(), 0, 1),
        )
    )
    spectral = np.column_stack(
        [0.2 + (band + 1) * 0.02 * x + 0.01 * y for band in range(6)]
    )
    np.savez_compressed(
        path,
        xyz_mm=points_m * 1000.0,
        rgb=rgb,
        reflectance=spectral,
        wavelengths_nm=np.array([495, 696, 600, 803, 545, 642]),
        source_scan_index=np.arange(1, len(points_m) + 1),
        valid=np.ones(len(points_m), dtype=bool),
        side=np.array(side),
    )


def three_plane_points(grid_size: int = 8) -> np.ndarray:
    values = np.linspace(-0.4, 0.4, grid_size)
    a, b = np.meshgrid(values, values)
    return np.unique(
        np.vstack(
            (
                np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size))),
                np.column_stack((a.ravel(), np.zeros(a.size), b.ravel())),
                np.column_stack((np.zeros(a.size), a.ravel(), b.ravel())),
            )
        ),
        axis=0,
    )


def registration_validation_config() -> dict[str, float | int]:
    return {
        "voxel_size_m": 0.10,
        "max_correspondence_m": 0.30,
        "trim_fraction": 0.70,
        "min_correspondences": 30,
        "min_relative_improvement": 0.005,
        "max_coverage_drop": 0.02,
        "overlap_weight": 0.20,
        "attribute_clip_sigma": 3.0,
    }


def small_pipeline_fixture(
    root: Path,
    coarse_enabled: bool = True,
) -> tuple[Path, Path]:
    (root / "data" / "processed").mkdir(parents=True)
    points = three_plane_points()
    make_input(root / "data" / "processed" / "left_input.npz", points, "left")
    make_input(root / "data" / "processed" / "right_input.npz", points, "right")
    config = {
        "seed": 42,
        "trial_count": 1,
        "max_rotation_deg": 2.0,
        "max_translation_m": 0.02,
        "voxel_sizes_m": [0.12, 0.06],
        "max_correspondence_distances_m": [0.30, 0.12],
        "max_iterations": [30, 20],
        "evaluation_max_correspondence_m": 0.12,
        "lambda_geometric_rgb": 0.968,
        "lambda_geo_spectral": 0.5,
        "lambda_spec_spectral": 0.5,
        "success_rotation_deg": 2.0,
        "success_translation_m": 0.1,
        "registration_validation": registration_validation_config(),
        "coarse_registration": {
            "enabled": coarse_enabled,
            "feature_radius_multiplier": 5.0,
            "candidate_count": 2,
            "ransac_n": 3,
            "max_iterations": 3000,
            "confidence": 0.999,
            "max_rotation_deg": 25.0,
            "max_translation_m": 1.25,
            "score_max_distance_m": 0.30,
            "trim_fraction": 0.70,
            "min_correspondences": 20,
            "min_coverage": 0.10,
            "min_score_improvement": 0.03,
            "methods": {
                "geometry": {
                    "config_id": "test_geometry",
                    "voxel_size_m": 0.12,
                    "max_correspondence_m": 0.18,
                    "normal_radius_multiplier": 2.0,
                    "lambda_geo": 0.80,
                    "lambda_attribute": 0.0,
                    "lambda_overlap": 0.20,
                },
                "rgb": {
                    "config_id": "test_rgb",
                    "voxel_size_m": 0.12,
                    "max_correspondence_m": 0.18,
                    "normal_radius_multiplier": 2.0,
                    "attribute_feature_weight": 1.0,
                    "lambda_geo": 0.48,
                    "lambda_attribute": 0.32,
                    "lambda_overlap": 0.20,
                },
                "spectral6": {
                    "config_id": "test_spectral6",
                    "voxel_size_m": 0.12,
                    "max_correspondence_m": 0.18,
                    "normal_radius_multiplier": 2.0,
                    "attribute_feature_weight": 2.0,
                    "lambda_geo": 0.40,
                    "lambda_attribute": 0.40,
                    "lambda_overlap": 0.20,
                },
            },
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
    return root, config_path


class SharedInitializationTests(unittest.TestCase):
    def test_default_spectral_weights_match_rgb_ablation(self) -> None:
        config = json.loads(
            (EXPERIMENT_ROOT / "configs" / "default.json").read_text(
                encoding="utf-8"
            )
        )
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=float,
        )
        rgb = np.full((4, 3), 0.5)
        spectral = np.full((4, 6), 0.5)
        result = ICPResult(np.eye(4), 1.0, 0.0, [])

        with patch(
            "run_experiment.register_spectral_icp",
            return_value=result,
        ) as fine_registration:
            _run_method(
                "spectral6",
                points,
                points,
                rgb,
                rgb,
                spectral,
                spectral,
                config,
                init=np.eye(4),
            )

        call = fine_registration.call_args
        self.assertAlmostEqual(call.kwargs["lambda_geo"], 0.968)
        self.assertAlmostEqual(call.kwargs["lambda_spec"], 0.032)
        self.assertAlmostEqual(
            call.kwargs["lambda_geo"] + call.kwargs["lambda_spec"],
            1.0,
        )

    def test_run_method_passes_same_init_to_every_fine_registration(self) -> None:
        points = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=float,
        )
        rgb = np.full((4, 3), 0.5)
        spectral = np.full((4, 6), 0.5)
        init = np.eye(4)
        init[:3, 3] = [0.1, -0.2, 0.3]
        config = {
            "voxel_sizes_m": [0.2],
            "max_correspondence_distances_m": [0.4],
            "max_iterations": [5],
            "lambda_geometric_rgb": 0.968,
            "lambda_geo_spectral": 0.5,
            "lambda_spec_spectral": 0.5,
            "registration_validation": registration_validation_config(),
        }
        result = ICPResult(init.copy(), 1.0, 0.0, [])
        targets = {
            "geometry": "run_experiment.register_geometry_icp",
            "rgb": "run_experiment.register_rgb_icp",
            "spectral6": "run_experiment.register_spectral_icp",
        }

        for method, target in targets.items():
            with self.subTest(method=method), patch(
                target,
                return_value=result,
            ) as fine_registration:
                _run_method(
                    method,
                    points,
                    points,
                    rgb,
                    rgb,
                    spectral,
                    spectral,
                    config,
                    init=init,
                )

                np.testing.assert_allclose(
                    fine_registration.call_args.kwargs["init"],
                    init,
                )
                self.assertEqual(
                    fine_registration.call_args.kwargs["validation_config"],
                    registration_validation_config(),
                )


class PipelineTests(unittest.TestCase):
    def test_tuned_configs_override_only_their_matching_method(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "coarse_registration": {
                    "enabled": True,
                    "candidate_count": 8,
                    "methods": {
                        "geometry": {"config_id": "old_geometry"},
                        "rgb": {"config_id": "old_rgb"},
                        "spectral6": {"config_id": "old_spectral"},
                    },
                }
            }
            selected_dir = (
                root / "results" / "coarse_tuning" / "rgb"
            )
            selected_dir.mkdir(parents=True)
            (selected_dir / "selected_config.json").write_text(
                json.dumps(
                    {
                        "method": "rgb",
                        "config": {
                            "config_id": "tuned_rgb",
                            "voxel_size_m": 0.16,
                        },
                    }
                ),
                encoding="utf-8",
            )

            _load_tuned_coarse_configs(config, root)

            methods = config["coarse_registration"]["methods"]
            self.assertEqual(methods["geometry"]["config_id"], "old_geometry")
            self.assertEqual(methods["rgb"]["config_id"], "tuned_rgb")
            self.assertEqual(methods["rgb"]["voxel_size_m"], 0.16)
            self.assertEqual(
                methods["spectral6"]["config_id"],
                "old_spectral",
            )
            self.assertEqual(
                config["coarse_registration"]["candidate_count"],
                8,
            )

    def test_default_config_has_approved_registration_validation(self) -> None:
        config = json.loads(
            (EXPERIMENT_ROOT / "configs" / "default.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            config["registration_validation"],
            registration_validation_config(),
        )

    def test_default_config_enables_real_coarse_registration(self) -> None:
        config = json.loads(
            (EXPERIMENT_ROOT / "configs" / "default.json").read_text(
                encoding="utf-8"
            )
        )
        points = three_plane_points()
        x, y, z = points.T
        spectral = np.column_stack(
            [
                0.3 + (band + 1) * 0.02 * x + 0.01 * y + 0.005 * z
                for band in range(6)
            ]
        )

        result = _run_coarse_registration(
            "geometry",
            points,
            points,
            np.zeros((len(points), 3)),
            np.zeros((len(points), 3)),
            spectral,
            spectral,
            config,
            trial_number=1,
        )

        self.assertTrue(result.parameters["enabled"])
        self.assertGreater(result.fitness, 0.9)
        np.testing.assert_allclose(result.transformation, np.eye(4), atol=1e-6)

    def test_disabled_coarse_registration_records_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, config_path = small_pipeline_fixture(
                Path(tmp),
                coarse_enabled=False,
            )

            run_pipeline(
                config_path,
                method="geometry",
                trial=1,
                experiment_root=root,
            )

            coarse_dir = (
                root
                / "results"
                / "coarse_registration"
                / "geometry"
                / "trial_001"
            )
            transform = json.loads(
                (coarse_dir / "T_coarse.json").read_text(encoding="utf-8")
            )
            np.testing.assert_allclose(transform["matrix"], np.eye(4))
            metrics = json.loads(
                (coarse_dir / "coarse_metrics.json").read_text(encoding="utf-8")
            )
            self.assertFalse(metrics["enabled"])

    def test_each_method_uses_its_own_coarse_registration_per_trial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, config_path = small_pipeline_fixture(Path(tmp))
            transforms: dict[str, np.ndarray] = {}
            coarse_results: dict[str, CoarseRegistrationResult] = {}
            for index, method_name in enumerate(("geometry", "rgb", "spectral6")):
                transform = np.eye(4)
                transform[0, 3] = 0.01 * (index + 1)
                transforms[method_name] = transform
                coarse_results[method_name] = CoarseRegistrationResult(
                    transformation=transform,
                    fitness=0.8,
                    inlier_rmse=0.05,
                    source_downsampled_count=100,
                    target_downsampled_count=100,
                    parameters={
                        "modality": method_name,
                        "config_id": f"test_{method_name}",
                        "seed": 43,
                        "identity_baseline_score": float("inf"),
                        "selected_score": 0.25,
                        "valid_candidate_count": 1,
                    },
                )

            with (
                patch(
                    "run_experiment.register_geometry_coarse",
                    return_value=coarse_results["geometry"],
                ) as geometry_coarse,
                patch(
                    "run_experiment.register_rgb_coarse",
                    return_value=coarse_results["rgb"],
                ) as rgb_coarse,
                patch(
                    "run_experiment.register_spectral_coarse",
                    return_value=coarse_results["spectral6"],
                ) as spectral_coarse,
                patch(
                    "run_experiment._run_method",
                    side_effect=lambda method_name, *args, init, **kwargs: ICPResult(
                        init.copy(), 0.8, 0.05, []
                    ),
                ) as fine_registration,
            ):
                rows = run_pipeline(
                    config_path,
                    method="all",
                    trial=1,
                    experiment_root=root,
                )

            self.assertEqual(geometry_coarse.call_count, 1)
            self.assertEqual(rgb_coarse.call_count, 1)
            self.assertEqual(spectral_coarse.call_count, 1)
            self.assertEqual(fine_registration.call_count, 3)
            for fine_call in fine_registration.call_args_list:
                method_name = fine_call.args[0]
                np.testing.assert_allclose(
                    fine_call.kwargs["init"],
                    transforms[method_name],
                )
            self.assertEqual(len(rows), 3)
            for method_name in transforms:
                coarse_metrics = json.loads(
                    (
                        root
                        / "results"
                        / "coarse_registration"
                        / method_name
                        / "trial_001"
                        / "coarse_metrics.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(coarse_metrics["modality"], method_name)
                self.assertEqual(
                    coarse_metrics["config_id"],
                    f"test_{method_name}",
                )
            self.assertEqual(
                {row["coarse_method"] for row in rows},
                {"geometry", "rgb", "spectral6"},
            )

    def test_one_coarse_failure_does_not_stop_other_methods(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, config_path = small_pipeline_fixture(Path(tmp))
            identity_result = CoarseRegistrationResult(
                transformation=np.eye(4),
                fitness=1.0,
                inlier_rmse=0.0,
                source_downsampled_count=100,
                target_downsampled_count=100,
                parameters={
                    "modality": "ok",
                    "config_id": "ok",
                    "seed": 43,
                    "identity_baseline_score": 0.0,
                    "selected_score": 0.0,
                    "valid_candidate_count": 0,
                },
            )
            fine_result = ICPResult(np.eye(4), 1.0, 0.0, [])

            with (
                patch(
                    "run_experiment.register_geometry_coarse",
                    side_effect=RuntimeError("geometry failed"),
                ),
                patch(
                    "run_experiment.register_rgb_coarse",
                    return_value=identity_result,
                ),
                patch(
                    "run_experiment.register_spectral_coarse",
                    return_value=identity_result,
                ),
                patch(
                    "run_experiment._run_method",
                    return_value=fine_result,
                ) as fine_registration,
            ):
                rows = run_pipeline(
                    config_path,
                    method="all",
                    trial=1,
                    experiment_root=root,
                )

            by_method = {row["method"]: row for row in rows}
            self.assertEqual(by_method["geometry"]["status"], "error")
            self.assertEqual(by_method["rgb"]["status"], "ok")
            self.assertEqual(by_method["spectral6"]["status"], "ok")
            self.assertEqual(fine_registration.call_count, 2)

    def test_raw_data_links_are_repository_relative(self) -> None:
        raw_dir = EXPERIMENT_ROOT / "data" / "raw"
        for filename in ("left.bin", "right.bin", "left_angle.txt", "right_angle.txt"):
            link = raw_dir / filename
            self.assertTrue(link.is_symlink(), filename)
            self.assertFalse(Path(os.readlink(link)).is_absolute(), filename)

    def test_summary_counts_error_trials_as_registration_failures(self) -> None:
        rows = [
            {
                "method": "geometry",
                "status": "ok",
                "success": True,
                "rotation_error_deg": 0.1,
                "translation_error_m": 0.01,
                "fitness": 0.9,
                "inlier_rmse_m": 0.02,
                "runtime_s": 1.0,
            },
            {
                "method": "geometry",
                "status": "error",
                "success": False,
                "rotation_error_deg": None,
                "translation_error_m": None,
                "fitness": None,
                "inlier_rmse_m": None,
                "runtime_s": 0.5,
            },
        ]

        summary = _summary_rows(rows)[0]

        self.assertEqual(summary["trial_count"], 2)
        self.assertEqual(summary["completed_count"], 1)
        self.assertAlmostEqual(summary["success_rate"], 0.5)

    def test_geometry_smoke_run_writes_required_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "processed").mkdir(parents=True)
            values = np.linspace(-0.4, 0.4, 8)
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
            make_input(root / "data" / "processed" / "left_input.npz", points, "left")
            make_input(root / "data" / "processed" / "right_input.npz", points, "right")
            config = {
                "seed": 42,
                "trial_count": 1,
                "max_rotation_deg": 2.0,
                "max_translation_m": 0.02,
                "voxel_sizes_m": [0.12, 0.06],
                "max_correspondence_distances_m": [0.30, 0.12],
                "max_iterations": [30, 20],
                "evaluation_max_correspondence_m": 0.12,
                "lambda_geometric_rgb": 0.968,
                "lambda_geo_spectral": 0.5,
                "lambda_spec_spectral": 0.5,
                "success_rotation_deg": 2.0,
                "success_translation_m": 0.1,
                "registration_validation": registration_validation_config(),
                "data": {
                    "left_input": "data/processed/left_input.npz",
                    "right_input": "data/processed/right_input.npz",
                    "left_prepared": "data/processed/left_points.npz",
                    "right_prepared": "data/processed/right_points.npz",
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            rows = run_pipeline(
                config_path,
                method="geometry",
                trial=1,
                experiment_root=root,
            )

            self.assertEqual(len(rows), 1)
            trial_dir = root / "results" / "icp_geometry" / "trial_001"
            for filename in (
                "T_perturb.json",
                "T_gt.json",
                "T_est.json",
                "metrics.json",
                "convergence.csv",
                "source_perturbed.ply",
                "source_registered.ply",
                "source_target_initial_merged.ply",
                "source_target_merged.ply",
                "source_target_rgb_merged.ply",
                "before_after.png",
            ):
                self.assertTrue((trial_dir / filename).is_file(), filename)

            with (trial_dir / "convergence.csv").open(
                newline="",
                encoding="utf-8",
            ) as handle:
                fieldnames = set(csv.DictReader(handle).fieldnames or [])
            self.assertTrue(
                {
                    "validation_score_before",
                    "validation_score_after",
                    "validation_coverage_before",
                    "validation_coverage_after",
                    "validation_relative_improvement",
                    "accepted",
                    "rollback_reason",
                }.issubset(fieldnames)
            )

            coarse_dir = (
                root
                / "results"
                / "coarse_registration"
                / "geometry"
                / "trial_001"
            )
            for filename in (
                "T_coarse.json",
                "coarse_metrics.json",
                "coarse_candidates.csv",
                "coarse_candidates.json",
                "source_coarse_aligned.ply",
                "source_target_coarse_merged.ply",
                "source_target_coarse_rgb_merged.ply",
            ):
                self.assertTrue((coarse_dir / filename).is_file(), filename)
            coarse_metrics = json.loads(
                (coarse_dir / "coarse_metrics.json").read_text(encoding="utf-8")
            )
            for field in (
                "rotation_error_deg",
                "translation_error_m",
                "fitness",
                "inlier_rmse_m",
                "runtime_s",
                "enabled",
                "selected_candidate_id",
                "fallback_identity",
                "selection_reason",
                "identity_baseline_score",
                "selected_score",
                "candidate_count",
                "valid_candidate_count",
            ):
                self.assertIn(field, coarse_metrics)

            self.assertTrue((root / "results" / "results_summary.csv").is_file())
            self.assertTrue((root / "results" / "method_summary.md").is_file())
            self.assertTrue(
                (root / "results" / "all_trials_method_summary.csv").is_file()
            )
            self.assertTrue(
                (root / "results" / "all_trials_method_summary.md").is_file()
            )
            self.assertTrue(
                (root / "results" / "heldout_method_summary.csv").is_file()
            )
            self.assertTrue(
                (root / "results" / "heldout_method_summary.md").is_file()
            )
            self.assertTrue(
                (root / "results" / "method_parameters_summary.csv").is_file()
            )
            self.assertTrue(
                (root / "results" / "method_parameters_summary.md").is_file()
            )
            self.assertTrue(
                (
                    root
                    / "results"
                    / "coarse_and_fine_error_boxplots.png"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "results"
                    / "coarse_and_fine_error_boxplots.svg"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "results"
                    / "coarse_and_fine_error_boxplots.pdf"
                ).is_file()
            )
            self.assertTrue(
                (
                    root
                    / "results"
                    / "coarse_and_fine_error_boxplots.tiff"
                ).is_file()
            )
            summary_header = (
                root / "results" / "results_summary.csv"
            ).read_text(encoding="utf-8").splitlines()[0]
            for field in (
                "overlap_100mm_pct",
                "overlap_250mm_pct",
                "coarse_method",
                "coarse_config_id",
                "coarse_rotation_error_deg",
                "coarse_translation_error_m",
                "coarse_fitness",
                "coarse_inlier_rmse_m",
                "coarse_runtime_s",
            ):
                self.assertIn(field, summary_header)
            metrics = json.loads(
                (trial_dir / "metrics.json").read_text(encoding="utf-8")
            )
            self.assertIn("overlap_100mm_pct", metrics)
            self.assertIn("overlap_250mm_pct", metrics)
            method_summary_header = (
                root / "results" / "method_summary.csv"
            ).read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("overlap_100mm_pct_mean", method_summary_header)
            self.assertIn("overlap_250mm_pct_mean", method_summary_header)
            self.assertTrue((root / "experiment_log.md").is_file())
            self.assertTrue(any((root / "logs").glob("experiment_*.log")))
            experiment_log = (root / "experiment_log.md").read_text(encoding="utf-8")
            self.assertIn("## 粗配准选参与独立评价", experiment_log)
            self.assertIn("trial 1–5", experiment_log)
            self.assertIn("trial 6–10", experiment_log)
            self.assertIn("T_gt 只用于离线选参与最终评价", experiment_log)
            self.assertIn("## 每次随机扰动", experiment_log)
            self.assertIn("## 逐次结果", experiment_log)


if __name__ == "__main__":
    unittest.main()
