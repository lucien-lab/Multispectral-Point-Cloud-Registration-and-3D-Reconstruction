from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.icp_spectral_robust import (
    build_mutual_correspondences,
    fit_robust_tangent_spectral_gradients,
    register_robust_spectral_icp,
    score_robust_spectral_transform,
)
from src.icp_spectral import register_spectral_icp
from src.coarse_registration import (
    build_spectral_features,
    score_modality_candidate,
)
from src.transform import apply_transform, invert_transform, transform_errors


def plane(side: int = 11, spacing: float = 0.04) -> np.ndarray:
    values = (np.arange(side, dtype=np.float64) - side // 2) * spacing
    x, y = np.meshgrid(values, values)
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))


def spectrum(points: np.ndarray) -> np.ndarray:
    x, y, z = points.T
    return np.column_stack(
        [
            0.4
            + (band + 1) * 0.08 * x
            - 0.04 * y
            + 0.03 * z
            for band in range(6)
        ]
    )


def three_planes(side: int = 11) -> np.ndarray:
    values = np.linspace(-0.5, 0.5, side)
    a, b = np.meshgrid(values, values)
    xy = np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size)))
    xz = np.column_stack((a.ravel(), np.zeros(a.size), b.ravel()))
    yz = np.column_stack((np.zeros(a.size), a.ravel(), b.ravel()))
    return np.unique(np.vstack((xy, xz, yz)), axis=0)


class RobustGradientTests(unittest.TestCase):
    def test_huber_gradient_recovers_linear_field_despite_neighbor_outliers(
        self,
    ) -> None:
        points = plane()
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        spectral = spectrum(points)
        spectral[[12, 25, 97], :] += np.array([2.0, -1.5, 1.0, -2.0, 1.2, -1.0])

        gradients, confidence = fit_robust_tangent_spectral_gradients(
            points,
            normals,
            spectral,
            radius=0.13,
            max_nn=50,
            min_neighbors=8,
            max_condition=100.0,
            huber_delta=1.5,
            irls_iterations=5,
        )

        center = int(np.argmin(np.linalg.norm(points, axis=1)))
        expected = np.column_stack(
            (
                np.arange(1, 7) * 0.08,
                np.full(6, -0.04),
                np.zeros(6),
            )
        )
        np.testing.assert_allclose(gradients[center], expected, atol=0.025)
        self.assertTrue(np.all(confidence[center] > 0.0))

    def test_invalid_local_support_produces_zero_gradient_and_confidence(
        self,
    ) -> None:
        points = plane(side=3)
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))

        gradients, confidence = fit_robust_tangent_spectral_gradients(
            points,
            normals,
            spectrum(points),
            radius=0.05,
            max_nn=50,
            min_neighbors=8,
            max_condition=100.0,
            huber_delta=1.5,
            irls_iterations=5,
        )

        np.testing.assert_array_equal(gradients, np.zeros_like(gradients))
        np.testing.assert_array_equal(confidence, np.zeros_like(confidence))


class RobustCorrespondenceTests(unittest.TestCase):
    def test_correspondences_are_mutual_distance_limited_and_normal_consistent(
        self,
    ) -> None:
        target = plane(side=10)
        source = target + np.array([0.05, 0.0, 0.0])
        source_normals = np.tile([0.0, 0.0, 1.0], (len(source), 1))
        target_normals = source_normals.copy()
        source_normals[0] = [1.0, 0.0, 0.0]
        transform = np.eye(4)
        transform[0, 3] = -0.05

        pairs = build_mutual_correspondences(
            source,
            target,
            source_normals,
            target_normals,
            transform,
            distance_limit=0.01,
            normal_angle_deg=45.0,
        )

        self.assertEqual(len(pairs.source_indices), len(source) - 1)
        self.assertNotIn(0, pairs.source_indices)
        np.testing.assert_allclose(pairs.distances, 0.0, atol=1e-12)


class WeightedCoarseTests(unittest.TestCase):
    def test_post_standardization_channel_weights_control_attribute_score(
        self,
    ) -> None:
        rng = np.random.default_rng(4)
        points = rng.uniform(-0.5, 0.5, size=(180, 3))
        base = np.linspace(-1.0, 1.0, len(points))
        target_spectral = np.column_stack(
            [base + 0.05 * band for band in range(6)]
        )
        weights = np.array([0.35, 0.15, 0.15, 0.10, 0.20, 0.05])
        config = {
            "voxel_size_m": 0.04,
            "max_correspondence_m": 0.10,
            "normal_radius_multiplier": 2.0,
            "feature_radius_multiplier": 5.0,
            "attribute_feature_weight": 1.0,
            "attribute_channel_weights": weights.tolist(),
            "score_max_distance_m": 0.10,
            "trim_fraction": 1.0,
            "min_correspondences": 30,
            "min_coverage": 0.10,
            "lambda_geo": 0.0,
            "lambda_attribute": 0.8,
            "lambda_overlap": 0.2,
            "max_rotation_deg": 25.0,
            "max_translation_m": 1.25,
        }

        def attribute_error(changed_band: int) -> float:
            source_spectral = target_spectral.copy()
            source_spectral[:, changed_band] += 0.2
            (
                source_cloud,
                target_cloud,
                _,
                _,
                source_values,
                target_values,
            ) = build_spectral_features(
                points,
                points,
                source_spectral,
                target_spectral,
                config=config,
            )
            candidate = score_modality_candidate(
                modality="spectral6",
                candidate_id=f"band_{changed_band}",
                seed=1,
                transformation=np.eye(4),
                source_points=np.asarray(source_cloud.points),
                target_points=np.asarray(target_cloud.points),
                source_attribute=source_values,
                target_attribute=target_values,
                config=config,
            )
            return candidate.attribute_error

        high_weight_error = attribute_error(0)
        low_weight_error = attribute_error(5)

        self.assertGreater(high_weight_error, 5.0 * low_weight_error)


class RobustObjectiveTests(unittest.TestCase):
    def test_true_transform_scores_below_identity_with_same_tangent_model(
        self,
    ) -> None:
        target = plane(side=13)
        target_spectral = spectrum(target)
        perturbation = np.eye(4)
        perturbation[:3, 3] = [0.06, -0.03, 0.02]
        source = apply_transform(target, perturbation)
        ground_truth = invert_transform(perturbation)
        config = {
            "voxel_size_m": 0.04,
            "max_correspondence_m": 0.12,
            "normal_angle_deg": 45.0,
            "min_correspondences": 30,
            "trim_fraction": 0.70,
            "overlap_weight": 0.20,
            "huber_delta_geo": 1.5,
            "huber_delta_spec": 1.5,
            "gradient_radius_multiplier": 2.5,
            "gradient_max_nn": 50,
            "min_gradient_neighbors": 8,
            "max_gradient_condition": 100.0,
            "gradient_irls_iterations": 5,
        }

        identity_score = score_robust_spectral_transform(
            source,
            target,
            target_spectral,
            target_spectral,
            np.eye(4),
            band_weights=np.full(6, 1.0 / 6.0),
            lambda_spec=0.10,
            config=config,
        )
        true_score = score_robust_spectral_transform(
            source,
            target,
            target_spectral,
            target_spectral,
            ground_truth,
            band_weights=np.full(6, 1.0 / 6.0),
            lambda_spec=0.10,
            config=config,
        )

        self.assertTrue(identity_score.valid)
        self.assertTrue(true_score.valid)
        self.assertLess(true_score.score, identity_score.score)
        self.assertGreaterEqual(true_score.coverage, identity_score.coverage)


class RobustRegistrationTests(unittest.TestCase):
    def test_recovers_known_transform_and_records_aligned_validation(self) -> None:
        target = three_planes()
        target_spectral = spectrum(target)
        perturbation = np.eye(4)
        perturbation[:3, :3] = Rotation.from_euler(
            "xyz",
            [2.0, -1.5, 1.0],
            degrees=True,
        ).as_matrix()
        perturbation[:3, 3] = [0.025, -0.018, 0.020]
        source = apply_transform(target, perturbation)
        ground_truth = invert_transform(perturbation)
        validation_config = {
            "voxel_size_m": 0.05,
            "max_correspondence_m": 0.30,
            "normal_angle_deg": 45.0,
            "min_correspondences": 30,
            "trim_fraction": 0.70,
            "overlap_weight": 0.20,
            "huber_delta_geo": 1.5,
            "huber_delta_spec": 1.5,
            "gradient_radius_multiplier": 2.5,
            "gradient_max_nn": 50,
            "min_gradient_neighbors": 8,
            "max_gradient_condition": 100.0,
            "gradient_irls_iterations": 5,
            "min_relative_improvement": 0.005,
            "max_coverage_drop": 0.02,
        }

        result = register_robust_spectral_icp(
            source,
            target,
            target_spectral,
            target_spectral,
            voxel_sizes=[0.12, 0.06, 0.03],
            max_correspondence_distances=[0.30, 0.15, 0.08],
            max_iterations=[40, 30, 20],
            band_weights=np.full(6, 1.0 / 6.0),
            lambda_spec_levels=(0.02, 0.05, 0.10),
            validation_config=validation_config,
        )

        errors = transform_errors(result.transformation, ground_truth)
        self.assertLess(errors["rotation_error_deg"], 1.0)
        self.assertLess(errors["translation_error_m"], 0.03)
        level_rows = [
            row
            for row in result.history
            if row["record_type"] == "level_validation"
        ]
        self.assertEqual(len(level_rows), 3)
        self.assertTrue(any(bool(row["accepted"]) for row in level_rows))
        for row in level_rows:
            self.assertIn("validation_spectral_before", row)
            self.assertIn("validation_spectral_after", row)

    def test_insufficient_correspondences_preserve_initial_transform(self) -> None:
        target = plane()
        source = target + np.array([2.0, 0.0, 0.0])

        result = register_robust_spectral_icp(
            source,
            target,
            spectrum(target),
            spectrum(target),
            voxel_sizes=[0.04],
            max_correspondence_distances=[0.01],
            max_iterations=[5],
            band_weights=np.full(6, 1.0 / 6.0),
            lambda_spec_levels=(0.10,),
            min_correspondences=30,
        )

        np.testing.assert_allclose(result.transformation, np.eye(4))
        self.assertTrue(np.isfinite(result.transformation).all())
        self.assertEqual(result.history[0]["rollback_reason"], "insufficient_correspondences")


class AlignedLegacyValidationTests(unittest.TestCase):
    def test_legacy_optimizer_can_use_its_own_tangent_objective_for_validation(
        self,
    ) -> None:
        target = three_planes()
        target_spectral = spectrum(target)
        perturbation = np.eye(4)
        perturbation[:3, :3] = Rotation.from_euler(
            "xyz",
            [1.5, -1.0, 0.8],
            degrees=True,
        ).as_matrix()
        perturbation[:3, 3] = [0.02, -0.015, 0.018]
        source = apply_transform(target, perturbation)
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

        result = register_spectral_icp(
            source,
            target,
            target_spectral,
            target_spectral,
            voxel_sizes=[0.12, 0.06, 0.03],
            max_correspondence_distances=[0.30, 0.15, 0.08],
            max_iterations=[40, 30, 20],
            lambda_geo=0.968,
            lambda_spec=0.032,
            validation_config=validation_config,
            validation_mode="tangent",
        )

        level_rows = [
            row
            for row in result.history
            if row["record_type"] == "level_validation"
        ]
        self.assertEqual(len(level_rows), 3)
        for row in level_rows:
            self.assertIn("validation_spectral_before", row)
            self.assertIn("validation_spectral_after", row)


if __name__ == "__main__":
    unittest.main()
