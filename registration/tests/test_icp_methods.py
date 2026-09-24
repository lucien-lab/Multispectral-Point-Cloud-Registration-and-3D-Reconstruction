from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.icp_geometry import register_geometry_icp
from src.icp_rgb import register_rgb_icp
from src.icp_spectral import (
    fit_tangent_spectral_gradients,
    register_spectral_icp,
)
from src.coarse_registration import (
    build_geometry_features,
    build_joint_features,
    build_rgb_features,
    build_spectral_features,
    preprocess_fpfh,
    register_fpfh_ransac,
    register_geometry_coarse,
    register_multispectral_coarse,
    register_rgb_coarse,
    register_spectral_coarse,
    score_modality_candidate,
    score_coarse_candidate,
    voxel_reduce_with_spectral,
)
from src.transform import apply_transform, invert_transform, transform_errors
from src.data_prepare import load_prepared
from src.transform import generate_perturbations


def three_plane_cloud(grid_size: int = 12) -> np.ndarray:
    values = np.linspace(-0.5, 0.5, grid_size)
    a, b = np.meshgrid(values, values)
    xy = np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size)))
    xz = np.column_stack((a.ravel(), np.zeros(a.size), b.ravel()))
    yz = np.column_stack((np.zeros(a.size), a.ravel(), b.ravel()))
    return np.unique(np.vstack((xy, xz, yz)), axis=0)


def known_perturbation() -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [2.0, -1.5, 1.0], degrees=True).as_matrix()
    transform[:3, 3] = [0.025, -0.018, 0.02]
    return transform


def attributes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shifted = points - points.min(axis=0)
    span = np.maximum(np.ptp(points, axis=0), 1e-9)
    rgb = np.clip(shifted / span, 0.0, 1.0)
    x, y, z = points.T
    spectral = np.column_stack(
        (
            0.2 + 0.3 * x + 0.1 * y,
            0.4 - 0.2 * x + 0.15 * z,
            0.3 + 0.25 * y - 0.1 * z,
            0.5 + 0.1 * x + 0.2 * z,
            0.25 - 0.15 * x - 0.1 * y,
            0.35 + 0.12 * x - 0.18 * y + 0.08 * z,
        )
    )
    return rgb, spectral


class CoarseRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        points = rng.uniform(-0.6, 0.6, size=(500, 3))
        points[:, 2] += 0.18 * points[:, 0] ** 2 + 0.11 * points[:, 1]
        self.target = points
        self.perturbation = np.eye(4)
        self.perturbation[:3, :3] = Rotation.from_euler(
            "xyz", [8.0, -6.0, 5.0], degrees=True
        ).as_matrix()
        self.perturbation[:3, 3] = [0.16, -0.11, 0.09]
        self.source = apply_transform(self.target, self.perturbation)
        _, self.spectral = attributes(self.target)

    def _multispectral_config(self, candidate_count: int = 4) -> dict[str, object]:
        return {
            "voxel_size_m": 0.08,
            "max_correspondence_m": 0.14,
            "normal_radius_multiplier": 2.0,
            "feature_radius_multiplier": 5.0,
            "spectral_feature_weight": 2.0,
            "candidate_count": candidate_count,
            "ransac_n": 3,
            "max_iterations": 30000,
            "confidence": 0.999,
            "max_rotation_deg": 25.0,
            "max_translation_m": 1.25,
            "score_max_distance_m": 0.20,
            "trim_fraction": 0.70,
            "min_correspondences": 30,
            "min_coverage": 0.10,
            "min_score_improvement": 0.03,
            "lambda_geo": 0.35,
            "lambda_spec": 0.45,
            "lambda_overlap": 0.20,
        }

    def _geometry_config(self, candidate_count: int = 0) -> dict[str, object]:
        config = self._multispectral_config(candidate_count)
        config.update(
            {
                "modality": "geometry",
                "lambda_geo": 0.8,
                "lambda_attribute": 0.0,
                "lambda_overlap": 0.2,
            }
        )
        return config

    def _rgb_config(self, candidate_count: int = 0) -> dict[str, object]:
        config = self._multispectral_config(candidate_count)
        config.update(
            {
                "modality": "rgb",
                "attribute_feature_weight": 1.0,
                "lambda_geo": 0.4,
                "lambda_attribute": 0.4,
                "lambda_overlap": 0.2,
            }
        )
        return config

    def test_method_specific_descriptors_have_expected_dimensions(self) -> None:
        rgb, spectral = attributes(self.target)
        geometry = build_geometry_features(
            self.target,
            self.target,
            config=self._geometry_config(),
        )
        rgb_joint = build_rgb_features(
            self.target,
            self.target,
            rgb,
            rgb,
            config=self._rgb_config(),
        )
        spectral_joint = build_spectral_features(
            self.target,
            self.target,
            spectral,
            spectral,
            config=self._multispectral_config(),
        )

        self.assertEqual(geometry[2].dimension(), 33)
        self.assertEqual(rgb_joint[2].dimension(), 36)
        self.assertEqual(spectral_joint[2].dimension(), 39)

    def test_geometry_score_does_not_accept_or_use_attributes(self) -> None:
        with self.assertRaises(TypeError):
            register_geometry_coarse(
                self.target,
                self.target,
                np.zeros((len(self.target), 3)),
                config=self._geometry_config(),
                seed=123,
            )

        unchanged = score_modality_candidate(
            modality="geometry",
            candidate_id="identity",
            seed=None,
            transformation=np.eye(4),
            source_points=self.target,
            target_points=self.target,
            config=self._geometry_config(),
        )
        self.assertEqual(unchanged.modality, "geometry")
        self.assertEqual(unchanged.attribute_error, 0.0)

    def test_only_matching_modality_attribute_changes_score(self) -> None:
        rgb, spectral = attributes(self.target)
        rgb_config = self._rgb_config()
        spectral_config = self._multispectral_config()
        spectral_config["lambda_attribute"] = spectral_config.pop("lambda_spec")

        rgb_match = score_modality_candidate(
            modality="rgb",
            candidate_id="rgb_match",
            seed=None,
            transformation=np.eye(4),
            source_points=self.target,
            target_points=self.target,
            source_attribute=rgb,
            target_attribute=rgb,
            config=rgb_config,
        )
        rgb_changed = score_modality_candidate(
            modality="rgb",
            candidate_id="rgb_changed",
            seed=None,
            transformation=np.eye(4),
            source_points=self.target,
            target_points=self.target,
            source_attribute=1.0 - rgb,
            target_attribute=rgb,
            config=rgb_config,
        )
        spectral_match = score_modality_candidate(
            modality="spectral6",
            candidate_id="spectral_match",
            seed=None,
            transformation=np.eye(4),
            source_points=self.target,
            target_points=self.target,
            source_attribute=spectral,
            target_attribute=spectral,
            config=spectral_config,
        )
        spectral_changed = score_modality_candidate(
            modality="spectral6",
            candidate_id="spectral_changed",
            seed=None,
            transformation=np.eye(4),
            source_points=self.target,
            target_points=self.target,
            source_attribute=spectral[:, ::-1],
            target_attribute=spectral,
            config=spectral_config,
        )

        self.assertLess(rgb_match.joint_score, rgb_changed.joint_score)
        self.assertLess(
            spectral_match.joint_score,
            spectral_changed.joint_score,
        )
        self.assertIsNotNone(rgb_match.rgb_error)
        self.assertIsNone(rgb_match.spectral_error)
        self.assertIsNone(spectral_match.rgb_error)
        self.assertIsNotNone(spectral_match.spectral_error)

    def test_three_public_coarse_interfaces_fall_back_deterministically(self) -> None:
        rgb, spectral = attributes(self.target)
        results = (
            register_geometry_coarse(
                self.target,
                self.target,
                config=self._geometry_config(),
                seed=123,
            ),
            register_rgb_coarse(
                self.target,
                self.target,
                rgb,
                rgb,
                config=self._rgb_config(),
                seed=123,
            ),
            register_spectral_coarse(
                self.target,
                self.target,
                spectral,
                spectral,
                config=self._multispectral_config(candidate_count=0),
                seed=123,
            ),
        )

        self.assertEqual(
            [result.parameters["modality"] for result in results],
            ["geometry", "rgb", "spectral6"],
        )
        for result in results:
            np.testing.assert_allclose(result.transformation, np.eye(4))
            self.assertTrue(result.fallback_identity)

    def test_candidate_outside_transform_bound_is_rejected(self) -> None:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler(
            "z",
            40.0,
            degrees=True,
        ).as_matrix()

        candidate = score_coarse_candidate(
            candidate_id="out_of_bounds",
            seed=123,
            transformation=transform,
            source_points=self.target,
            target_points=self.target,
            source_spectral6=self.spectral,
            target_spectral6=self.spectral,
            config=self._multispectral_config(),
        )

        self.assertFalse(candidate.accepted)
        self.assertEqual(
            candidate.rejection_reason,
            "rejected_transform_bound",
        )

    def test_multispectral_coarse_explicitly_falls_back_without_candidates(
        self,
    ) -> None:
        result = register_multispectral_coarse(
            self.target,
            self.target,
            self.spectral,
            self.spectral,
            config=self._multispectral_config(candidate_count=0),
            seed=123,
        )

        np.testing.assert_allclose(result.transformation, np.eye(4))
        self.assertTrue(result.fallback_identity)
        self.assertEqual(
            result.selection_reason,
            "no_candidate_improved_baseline",
        )

    def test_multispectral_coarse_recovers_partial_overlap_reproducibly(
        self,
    ) -> None:
        overlap = self.target[:, 0] > -0.35
        source = apply_transform(self.target[overlap], self.perturbation)
        source_spectral = self.spectral[overlap]
        config = self._multispectral_config(candidate_count=6)

        first = register_multispectral_coarse(
            source,
            self.target,
            source_spectral,
            self.spectral,
            config=config,
            seed=123,
        )
        second = register_multispectral_coarse(
            source,
            self.target,
            source_spectral,
            self.spectral,
            config=config,
            seed=123,
        )
        errors = transform_errors(
            first.transformation,
            invert_transform(self.perturbation),
        )

        self.assertLess(errors["rotation_error_deg"], 3.0)
        self.assertLess(errors["translation_error_m"], 0.15)
        self.assertFalse(first.fallback_identity)
        np.testing.assert_allclose(
            first.transformation,
            second.transformation,
            atol=1e-12,
        )

    def test_voxel_reduce_keeps_xyz_spectral_and_counts_aligned(self) -> None:
        xyz = np.array(
            [[0.01, 0.0, 0.0], [0.04, 0.0, 0.0], [0.21, 0.0, 0.0]]
        )
        spectral = np.array(
            [
                [1, 2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7, 8],
                [9, 10, 11, 12, 13, 14],
            ],
            dtype=float,
        )

        points, bands, counts = voxel_reduce_with_spectral(
            xyz,
            spectral,
            voxel_size_m=0.1,
        )

        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_allclose(bands[0], [2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(counts, [2, 1])

    def test_joint_feature_has_33_fpfh_plus_6_spectral_rows(self) -> None:
        _, spectral = attributes(self.target)

        (
            source_cloud,
            target_cloud,
            source_feature,
            target_feature,
            source_spec,
            target_spec,
        ) = build_joint_features(
            self.source,
            self.target,
            spectral,
            spectral,
            voxel_size_m=0.08,
            normal_radius_multiplier=2.0,
            feature_radius_multiplier=5.0,
            spectral_feature_weight=2.0,
        )

        self.assertEqual(
            np.asarray(source_feature.data).shape,
            (39, len(source_cloud.points)),
        )
        self.assertEqual(
            np.asarray(target_feature.data).shape,
            (39, len(target_cloud.points)),
        )
        self.assertEqual(source_spec.shape, (len(source_cloud.points), 6))
        self.assertEqual(target_spec.shape, (len(target_cloud.points), 6))
        self.assertTrue(np.isfinite(np.asarray(source_feature.data)).all())
        self.assertTrue(np.isfinite(np.asarray(target_feature.data)).all())

    def test_preprocess_fpfh_returns_normals_and_33d_features(self) -> None:
        cloud, feature = preprocess_fpfh(
            self.target,
            voxel_size_m=0.08,
            normal_radius_multiplier=2.0,
            feature_radius_multiplier=5.0,
        )

        self.assertGreater(len(cloud.points), 50)
        self.assertEqual(len(cloud.normals), len(cloud.points))
        self.assertEqual(np.asarray(feature.data).shape, (33, len(cloud.points)))
        self.assertTrue(np.isfinite(np.asarray(feature.data)).all())

    def test_preprocess_fpfh_rejects_invalid_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "N×3"):
            preprocess_fpfh(
                np.zeros((8, 2)),
                voxel_size_m=0.08,
                normal_radius_multiplier=2.0,
                feature_radius_multiplier=5.0,
            )

    def _register(self):
        return register_fpfh_ransac(
            self.source,
            self.target,
            voxel_size_m=0.08,
            max_correspondence_m=0.14,
            normal_radius_multiplier=2.0,
            feature_radius_multiplier=5.0,
            ransac_n=3,
            max_iterations=30000,
            confidence=0.999,
            seed=123,
        )

    def test_fpfh_ransac_recovers_known_transform(self) -> None:
        result = self._register()
        errors = transform_errors(
            result.transformation,
            invert_transform(self.perturbation),
        )

        self.assertLess(errors["rotation_error_deg"], 3.0)
        self.assertLess(errors["translation_error_m"], 0.08)
        self.assertGreater(result.fitness, 0.65)

    def test_fpfh_ransac_is_reproducible_for_fixed_seed(self) -> None:
        first = self._register()
        second = self._register()

        np.testing.assert_allclose(
            first.transformation,
            second.transformation,
            atol=1e-12,
        )


class Open3DICPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = three_plane_cloud()
        self.rgb, self.spectral = attributes(self.target)
        self.perturbation = known_perturbation()
        self.source = apply_transform(self.target, self.perturbation)
        self.ground_truth = invert_transform(self.perturbation)
        self.voxels = [0.12, 0.06, 0.03]
        self.distances = [0.30, 0.15, 0.08]
        self.iterations = [60, 40, 30]
        self.validation_config = {
            "voxel_size_m": 0.05,
            "max_correspondence_m": 0.30,
            "trim_fraction": 0.70,
            "min_correspondences": 30,
            "min_relative_improvement": 0.005,
            "max_coverage_drop": 0.02,
            "overlap_weight": 0.20,
            "attribute_clip_sigma": 3.0,
        }

    def assert_good_recovery(self, estimated: np.ndarray, rotation_deg: float, translation_m: float) -> None:
        errors = transform_errors(estimated, self.ground_truth)
        self.assertLess(errors["rotation_error_deg"], rotation_deg)
        self.assertLess(errors["translation_error_m"], translation_m)

    def test_point_to_plane_icp_recovers_known_transform(self) -> None:
        result = register_geometry_icp(
            self.source,
            self.target,
            self.voxels,
            self.distances,
            self.iterations,
        )
        self.assertEqual(len(result.history), 3)
        self.assert_good_recovery(result.transformation, 0.5, 0.01)

    def test_colored_icp_recovers_known_transform(self) -> None:
        result = register_rgb_icp(
            self.source,
            self.target,
            self.rgb,
            self.rgb,
            self.voxels,
            self.distances,
            self.iterations,
            lambda_geometric=0.968,
        )
        self.assertEqual(len(result.history), 3)
        self.assert_good_recovery(result.transformation, 0.5, 0.01)

    def test_geometry_rolls_back_a_worse_level_candidate(self) -> None:
        bad_candidate = np.eye(4)
        bad_candidate[:3, 3] = [0.20, 0.0, 0.0]
        registration = SimpleNamespace(
            transformation=bad_candidate,
            fitness=0.5,
            inlier_rmse=0.1,
        )

        with patch(
            "src.icp_geometry.o3d.pipelines.registration.registration_icp",
            return_value=registration,
        ):
            result = register_geometry_icp(
                self.target,
                self.target,
                [0.10],
                [0.30],
                [5],
                init=np.eye(4),
                validation_config=self.validation_config,
            )

        np.testing.assert_allclose(result.transformation, np.eye(4))
        self.assertFalse(result.history[0]["accepted"])
        self.assertEqual(
            result.history[0]["rollback_reason"],
            "insufficient_score_improvement",
        )
        self.assertEqual(result.history[0]["record_type"], "level_validation")
        self.assertAlmostEqual(result.fitness, 1.0)
        self.assertAlmostEqual(result.inlier_rmse, 0.0)

    def test_rgb_rolls_back_a_worse_level_candidate(self) -> None:
        bad_candidate = np.eye(4)
        bad_candidate[:3, 3] = [0.20, 0.0, 0.0]
        registration = SimpleNamespace(
            transformation=bad_candidate,
            fitness=0.5,
            inlier_rmse=0.1,
        )

        with patch(
            "src.icp_rgb.o3d.pipelines.registration.registration_colored_icp",
            return_value=registration,
        ):
            result = register_rgb_icp(
                self.target,
                self.target,
                self.rgb,
                self.rgb,
                [0.10],
                [0.30],
                [5],
                lambda_geometric=0.968,
                init=np.eye(4),
                validation_config=self.validation_config,
            )

        np.testing.assert_allclose(result.transformation, np.eye(4))
        self.assertFalse(result.history[0]["accepted"])
        self.assertEqual(
            result.history[0]["rollback_reason"],
            "insufficient_score_improvement",
        )
        self.assertEqual(result.history[0]["record_type"], "level_validation")

    def test_geometry_accepts_an_improving_level_candidate(self) -> None:
        registration = SimpleNamespace(
            transformation=self.ground_truth,
            fitness=1.0,
            inlier_rmse=0.0,
        )

        with patch(
            "src.icp_geometry.o3d.pipelines.registration.registration_icp",
            return_value=registration,
        ):
            result = register_geometry_icp(
                self.source,
                self.target,
                [0.10],
                [0.30],
                [5],
                init=np.eye(4),
                validation_config=self.validation_config,
            )

        np.testing.assert_allclose(result.transformation, self.ground_truth)
        self.assertTrue(result.history[0]["accepted"])
        self.assertEqual(result.history[0]["rollback_reason"], "")
        for field in (
            "validation_score_before",
            "validation_score_after",
            "validation_coverage_before",
            "validation_coverage_after",
            "validation_relative_improvement",
        ):
            self.assertIn(field, result.history[0])


class SpectralICPTests(unittest.TestCase):
    def test_tangent_gradient_recovers_linear_spectrum_on_plane(self) -> None:
        values = np.linspace(-0.3, 0.3, 9)
        x, y = np.meshgrid(values, values)
        points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        spectral = np.column_stack(
            [0.5 + (band + 1) * 0.1 * points[:, 0] - 0.05 * points[:, 1] for band in range(6)]
        )

        gradients = fit_tangent_spectral_gradients(
            points,
            normals,
            spectral,
            radius=0.18,
            max_nn=30,
        )

        center = np.argmin(np.linalg.norm(points, axis=1))
        expected = np.column_stack(
            (
                np.arange(1, 7) * 0.1,
                np.full(6, -0.05),
                np.zeros(6),
            )
        )
        np.testing.assert_allclose(gradients[center], expected, atol=1e-4)

    def test_spectral_icp_recovers_known_transform(self) -> None:
        target = three_plane_cloud(grid_size=10)
        _, spectral = attributes(target)
        perturbation = known_perturbation()
        source = apply_transform(target, perturbation)
        ground_truth = invert_transform(perturbation)

        result = register_spectral_icp(
            source,
            target,
            spectral,
            spectral,
            voxel_sizes=[0.12, 0.06, 0.03],
            max_correspondence_distances=[0.30, 0.15, 0.08],
            max_iterations=[40, 30, 20],
            lambda_geo=0.5,
            lambda_spec=0.5,
        )

        self.assertGreater(len(result.history), 0)
        errors = transform_errors(result.transformation, ground_truth)
        self.assertLess(errors["rotation_error_deg"], 1.0)
        self.assertLess(errors["translation_error_m"], 0.02)

    def test_spectral_rolls_back_rejected_level_candidate(self) -> None:
        target = three_plane_cloud(grid_size=10)
        _, spectral = attributes(target)
        perturbation = known_perturbation()
        source = apply_transform(target, perturbation)
        baseline = np.eye(4)
        rejected = SimpleNamespace(
            accepted=False,
            relative_improvement=-0.1,
            rollback_reason="insufficient_score_improvement",
            before=SimpleNamespace(score=0.2, coverage=0.9),
            after=SimpleNamespace(score=0.3, coverage=0.9),
        )
        validation_config = {
            "voxel_size_m": 0.05,
            "max_correspondence_m": 0.30,
            "trim_fraction": 0.70,
            "min_correspondences": 30,
            "min_relative_improvement": 0.005,
            "max_coverage_drop": 0.02,
            "overlap_weight": 0.20,
            "attribute_clip_sigma": 3.0,
        }

        with patch(
            "src.icp_spectral.decide_candidate",
            return_value=rejected,
        ):
            result = register_spectral_icp(
                source,
                target,
                spectral,
                spectral,
                voxel_sizes=[0.10],
                max_correspondence_distances=[0.30],
                max_iterations=[5],
                lambda_geo=0.5,
                lambda_spec=0.5,
                init=baseline,
                validation_config=validation_config,
            )

        np.testing.assert_allclose(result.transformation, baseline)
        level_rows = [
            row
            for row in result.history
            if row["record_type"] == "level_validation"
        ]
        iteration_rows = [
            row for row in result.history if row["record_type"] == "iteration"
        ]
        self.assertEqual(len(level_rows), 1)
        self.assertGreater(len(iteration_rows), 0)
        self.assertFalse(level_rows[0]["accepted"])
        self.assertEqual(
            level_rows[0]["rollback_reason"],
            "insufficient_score_improvement",
        )

    def test_real_trial_one_does_not_collapse_all_correspondences(self) -> None:
        data_dir = EXPERIMENT_ROOT / "data" / "processed"
        if not (data_dir / "left_points.npz").is_file():
            self.skipTest("正式实验快照尚未准备")
        left = load_prepared(data_dir / "left_points.npz")
        right = load_prepared(data_dir / "right_points.npz")
        perturbation = generate_perturbations(10, 42, 10.0, 0.5)[0].matrix
        source = apply_transform(left["xyz_m"], perturbation)

        result = register_spectral_icp(
            source,
            right["xyz_m"],
            left["spectral6"],
            right["spectral6"],
            voxel_sizes=[0.2, 0.1, 0.05],
            max_correspondence_distances=[0.8, 0.4, 0.2],
            max_iterations=[80, 50, 30],
            lambda_geo=0.5,
            lambda_spec=0.5,
        )

        self.assertGreater(result.fitness, 0.10)
        self.assertTrue(np.isfinite(result.transformation).all())


if __name__ == "__main__":
    unittest.main()
