from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from run_spectral_recommended_experiment import (
    _error_row,
    _persist_trial_error,
    _tune_round,
    _write_ablation_summary,
    run_protocol,
)
from src.spectral_calibration import SpectralCalibration
from src.data_prepare import WAVELENGTHS_NM


def prepared_cloud(path: Path, side: str) -> None:
    values = np.linspace(-0.4, 0.4, 10)
    x, y = np.meshgrid(values, values)
    xyz = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    spectral = np.column_stack(
        [
            0.4 + (band + 1) * 0.05 * xyz[:, 0] - 0.03 * xyz[:, 1]
            for band in range(6)
        ]
    )
    rgb = np.clip(
        np.column_stack(
            (
                xyz[:, 0] - xyz[:, 0].min(),
                xyz[:, 1] - xyz[:, 1].min(),
                np.full(len(xyz), 0.4),
            )
        ),
        0.0,
        1.0,
    )
    np.savez_compressed(
        path,
        xyz_m=xyz,
        rgb=rgb,
        spectral6=spectral,
        wavelengths_nm=WAVELENGTHS_NM,
        source_scan_index=np.arange(len(xyz), dtype=np.int32),
        side=np.array(side),
        coordinate_unit=np.array("m"),
    )


def smoke_config() -> dict[str, object]:
    return {
        "seed": 42,
        "trial_count": 20,
        "max_rotation_deg": 0.0,
        "max_translation_m": 0.0,
        "voxel_sizes_m": [0.10],
        "max_correspondence_distances_m": [0.20],
        "max_iterations": [3],
        "evaluation_max_correspondence_m": 0.20,
        "success_rotation_deg": 2.0,
        "success_translation_m": 0.10,
        "lambda_geo_spectral": 0.968,
        "lambda_spec_spectral": 0.032,
        "output_dir": "results/spectral6_recommended_experiment",
        "data": {
            "left_prepared": "data/processed/left_points.npz",
            "right_prepared": "data/processed/right_points.npz",
        },
        "calibration": {
            "max_correspondence_m": 0.10,
            "normal_angle_deg": 30.0,
            "normal_radius_m": 0.20,
            "min_pairs": 50,
            "gain_min": 0.25,
            "gain_max": 4.0,
            "soft_l1_scale": 0.02,
            "weight_floor": 0.05,
            "weight_ceiling": 0.35,
        },
        "robust_registration": {
            "lambda_spec_schedules_round1": [[0.10]],
            "lambda_spec_schedules_round2_extra": [[0.20]],
            "huber_delta_geo": 1.5,
            "huber_delta_spec": 1.5,
            "min_gradient_neighbors": 3,
            "max_gradient_condition": 100.0,
            "normal_angle_deg": 45.0,
            "min_correspondences": 20,
            "trim_fraction": 0.70,
            "gradient_max_nn": 50,
            "gradient_irls_iterations": 5,
        },
        "registration_validation": {
            "voxel_size_m": 0.10,
            "max_correspondence_m": 0.30,
            "trim_fraction": 0.70,
            "min_correspondences": 20,
            "min_relative_improvement": 0.005,
            "max_coverage_drop": 0.02,
            "overlap_weight": 0.20,
            "attribute_clip_sigma": 3.0,
            "normal_angle_deg": 45.0,
            "huber_delta_geo": 1.5,
            "huber_delta_spec": 1.5,
            "gradient_radius_multiplier": 2.5,
            "gradient_max_nn": 50,
            "min_gradient_neighbors": 3,
            "max_gradient_condition": 100.0,
            "gradient_irls_iterations": 5,
        },
        "coarse_registration": {
            "enabled": True,
            "feature_radius_multiplier": 5.0,
            "candidate_count": 0,
            "ransac_n": 3,
            "max_iterations": 10,
            "confidence": 0.99,
            "max_rotation_deg": 25.0,
            "max_translation_m": 1.25,
            "score_max_distance_m": 0.30,
            "trim_fraction": 0.70,
            "min_correspondences": 20,
            "min_coverage": 0.10,
            "min_score_improvement": 0.03,
            "methods": {
                "geometry": {
                    "config_id": "smoke_geometry",
                    "voxel_size_m": 0.10,
                    "max_correspondence_m": 0.20,
                    "normal_radius_multiplier": 2.0,
                    "lambda_geo": 0.80,
                    "lambda_attribute": 0.0,
                    "lambda_overlap": 0.20,
                },
                "spectral6": {
                    "config_id": "smoke_spectral",
                    "voxel_size_m": 0.10,
                    "max_correspondence_m": 0.20,
                    "normal_radius_multiplier": 2.0,
                    "attribute_feature_weight": 1.0,
                    "lambda_geo": 0.56,
                    "lambda_attribute": 0.24,
                    "lambda_overlap": 0.20,
                },
            },
        },
    }


class SpectralRecommendedPipelineTests(unittest.TestCase):
    def test_tuning_reuses_coarse_search_transform_and_penalizes_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "processed"
            data_dir.mkdir(parents=True)
            prepared_cloud(data_dir / "left_points.npz", "left")
            prepared_cloud(data_dir / "right_points.npz", "right")
            from src.data_prepare import load_prepared

            left = load_prepared(data_dir / "left_points.npz")
            right = load_prepared(data_dir / "right_points.npz")
            config = smoke_config()
            calibration = SpectralCalibration(
                wavelengths_nm=WAVELENGTHS_NM.copy(),
                gain=np.ones(6),
                offset=np.zeros(6),
                residual_mad=np.ones(6),
                correlation=np.ones(6),
                band_weights=np.full(6, 1.0 / 6.0),
                pair_count=100,
            )

            def coarse_side_effect(
                *args: object,
                **kwargs: object,
            ) -> types.SimpleNamespace:
                if kwargs["seed"] == 44:
                    raise RuntimeError("trial 2 failed")
                return types.SimpleNamespace(
                    transformation=np.eye(4),
                )

            with (
                mock.patch(
                    "run_spectral_recommended_experiment.generate_search_configs",
                    return_value=[{"config_id": "only"}],
                ),
                mock.patch(
                    "run_spectral_recommended_experiment.register_spectral_coarse",
                    side_effect=coarse_side_effect,
                ) as coarse,
                mock.patch(
                    "run_spectral_recommended_experiment.register_robust_spectral_icp",
                    return_value=types.SimpleNamespace(
                        transformation=np.eye(4),
                    ),
                ) as robust,
            ):
                selected, schedule = _tune_round(
                    root,
                    config,
                    left,
                    right,
                    calibration,
                    round_number=1,
                )
            self.assertEqual(selected["config_id"], "only")
            self.assertEqual(schedule, (0.10,))
            self.assertEqual(coarse.call_count, 5)
            self.assertEqual(robust.call_count, 4)
            rows = (
                root
                / str(config["output_dir"])
                / "round1"
                / "tuning"
                / "schedule_search_results.csv"
            ).read_text(encoding="utf-8")
            self.assertIn("trial 2 failed", rows)

    def test_evaluate_without_trial_uses_frozen_round1_heldout_parameters(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "processed"
            data_dir.mkdir(parents=True)
            prepared_cloud(data_dir / "left_points.npz", "left")
            prepared_cloud(data_dir / "right_points.npz", "right")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(smoke_config()),
                encoding="utf-8",
            )
            frozen = mock.sentinel.calibration

            def result_row(*args: object, **kwargs: object) -> dict[str, object]:
                return {
                    "round": kwargs["round_number"],
                    "variant": kwargs["variant"],
                    "trial": kwargs["trial_number"],
                    "status": "ok",
                    "rotation_error_deg": 0.0,
                    "translation_error_m": 0.0,
                    "overlap_100mm_pct": 100.0,
                    "overlap_250mm_pct": 100.0,
                    "success": True,
                }

            with (
                mock.patch(
                    "run_spectral_recommended_experiment._load_round_calibration",
                    return_value=frozen,
                ) as load_calibration,
                mock.patch(
                    "run_spectral_recommended_experiment._load_round_tuning",
                    return_value=(
                        {"config_id": "frozen_coarse"},
                        (0.02,),
                    ),
                ) as load_tuning,
                mock.patch(
                    "run_spectral_recommended_experiment._fit_round_calibration"
                ) as fit,
                mock.patch(
                    "run_spectral_recommended_experiment._run_variant_trial",
                    side_effect=result_row,
                ) as run_trial,
            ):
                rows = run_protocol(
                    config_path,
                    stage="evaluate",
                    variant="all",
                    experiment_root=root,
                )
            self.assertEqual(len(rows), 20)
            self.assertEqual({row["trial"] for row in rows}, set(range(6, 11)))
            fit.assert_not_called()
            load_calibration.assert_called_once()
            load_tuning.assert_called_once()
            robust_calls = [
                call
                for call in run_trial.call_args_list
                if call.kwargs["variant"] == "robust_full"
            ]
            self.assertTrue(robust_calls)
            self.assertTrue(
                all(
                    call.kwargs["coarse_override"]["config_id"]
                    == "frozen_coarse"
                    and call.kwargs["lambda_spec_schedule"] == (0.02,)
                    for call in robust_calls
                )
            )

    def test_failed_trial_is_persisted_with_error_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _error_row(
                1,
                "robust_full",
                6,
                RuntimeError("boom"),
            )
            _persist_trial_error(root, row)
            output = root / "round1" / "robust_full" / "trial_006"
            self.assertTrue((output / "metrics.json").is_file())
            self.assertIn(
                "RuntimeError: boom",
                (output / "error.txt").read_text(encoding="utf-8"),
            )

    def test_tune_stage_only_calibrates_and_tunes_round_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "processed"
            data_dir.mkdir(parents=True)
            prepared_cloud(data_dir / "left_points.npz", "left")
            prepared_cloud(data_dir / "right_points.npz", "right")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(smoke_config()),
                encoding="utf-8",
            )
            fake_calibration = mock.sentinel.calibration
            with (
                mock.patch(
                    "run_spectral_recommended_experiment._fit_round_calibration",
                    return_value=fake_calibration,
                ) as fit,
                mock.patch(
                    "run_spectral_recommended_experiment._tune_round",
                    return_value=({"config_id": "chosen"}, (0.10,)),
                ) as tune,
                mock.patch(
                    "run_spectral_recommended_experiment._run_variant_trial"
                ) as run_trial,
            ):
                rows = run_protocol(
                    config_path,
                    stage="tune",
                    variant="all",
                    experiment_root=root,
                )
            self.assertEqual(rows, [])
            self.assertEqual(fit.call_args.kwargs["round_number"], 1)
            self.assertEqual(tune.call_args.kwargs["round_number"], 1)
            run_trial.assert_not_called()

    def test_ablation_summary_keeps_failed_trial_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            _write_ablation_summary(
                output,
                [_error_row(1, "robust_full", 6, RuntimeError("boom"))],
            )
            text = (output / "ablation_results.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("robust_full", text)
            self.assertIn("N/A", text)

    def test_single_baseline_trial_writes_isolated_complete_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "processed"
            data_dir.mkdir(parents=True)
            prepared_cloud(data_dir / "left_points.npz", "left")
            prepared_cloud(data_dir / "right_points.npz", "right")
            config = smoke_config()
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            rows = run_protocol(
                config_path,
                stage="evaluate",
                variant="baseline",
                trial=1,
                experiment_root=root,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["variant"], "baseline")
            output = (
                root
                / "results"
                / "spectral6_recommended_experiment"
                / "round1"
                / "baseline"
                / "trial_001"
            )
            for name in (
                "T_perturb.json",
                "T_gt.json",
                "T_coarse.json",
                "T_est.json",
                "metrics.json",
                "convergence.csv",
                "correspondence_diagnostics.csv",
                "source_perturbed.ply",
                "source_coarse_aligned.ply",
                "source_registered.ply",
                "source_target_merged.ply",
                "source_target_rgb_merged.ply",
                "before_after.png",
            ):
                self.assertTrue((output / name).is_file(), name)
            self.assertFalse((root / "results" / "icp_spectral6").exists())

    def test_all_variants_share_perturbation_and_write_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data" / "processed"
            data_dir.mkdir(parents=True)
            prepared_cloud(data_dir / "left_points.npz", "left")
            prepared_cloud(data_dir / "right_points.npz", "right")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(smoke_config()),
                encoding="utf-8",
            )

            rows = run_protocol(
                config_path,
                stage="evaluate",
                variant="all",
                trial=1,
                experiment_root=root,
            )

            self.assertEqual(
                {row["variant"] for row in rows},
                {
                    "baseline",
                    "calibrated",
                    "aligned_validation",
                    "robust_full",
                },
            )
            output_root = (
                root
                / "results"
                / "spectral6_recommended_experiment"
                / "round1"
            )
            perturbations = []
            for variant in (
                "baseline",
                "calibrated",
                "aligned_validation",
                "robust_full",
            ):
                payload = json.loads(
                    (
                        output_root
                        / variant
                        / "trial_001"
                        / "T_perturb.json"
                    ).read_text(encoding="utf-8")
                )
                perturbations.append(payload["transformation"])
            for value in perturbations[1:]:
                np.testing.assert_allclose(value, perturbations[0])
            calibration_dir = output_root / "calibration"
            for name in (
                "calibration.json",
                "calibration_pairs.csv",
                "spectral_fit_before_after.png",
                "calibration_summary.md",
            ):
                self.assertTrue((calibration_dir / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
