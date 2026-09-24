from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.registration_validation import (
    build_validation_context,
    decide_candidate,
    score_transform,
)
from src.transform import apply_transform, invert_transform


VALIDATION_CONFIG = {
    "voxel_size_m": 0.05,
    "max_correspondence_m": 0.30,
    "trim_fraction": 0.70,
    "min_correspondences": 30,
    "min_relative_improvement": 0.005,
    "max_coverage_drop": 0.02,
    "overlap_weight": 0.20,
    "attribute_clip_sigma": 3.0,
}


def grid_points() -> np.ndarray:
    values = np.linspace(-0.5, 0.5, 9)
    x, y = np.meshgrid(values, values)
    return np.column_stack((x.ravel(), y.ravel(), 0.1 * x.ravel() ** 2))


class RegistrationValidationTests(unittest.TestCase):
    def test_correct_geometry_transform_scores_below_identity(self) -> None:
        target = grid_points()
        perturbation = np.eye(4)
        perturbation[:3, 3] = [0.18, -0.08, 0.04]
        source = apply_transform(target, perturbation)
        context = build_validation_context(
            source,
            target,
            None,
            None,
            geometry_fraction=1.0,
            attribute_fraction=0.0,
            config=VALIDATION_CONFIG,
        )

        correct = score_transform(context, invert_transform(perturbation))
        identity = score_transform(context, np.eye(4))

        self.assertTrue(correct.valid)
        self.assertLess(correct.score, identity.score)

    def test_matching_rgb_scores_below_permuted_rgb(self) -> None:
        target = grid_points()
        rgb = np.clip(target - target.min(axis=0), 0.0, 1.0)
        matching = build_validation_context(
            target,
            target,
            rgb,
            rgb,
            geometry_fraction=0.5,
            attribute_fraction=0.5,
            config=VALIDATION_CONFIG,
        )
        mismatched = build_validation_context(
            target,
            target,
            np.roll(rgb, 17, axis=0),
            rgb,
            geometry_fraction=0.5,
            attribute_fraction=0.5,
            config=VALIDATION_CONFIG,
        )

        self.assertLess(
            score_transform(matching, np.eye(4)).score,
            score_transform(mismatched, np.eye(4)).score,
        )

    def test_matching_spectral_scores_below_permuted_spectral(self) -> None:
        target = grid_points()
        x, y, z = target.T
        spectral = np.column_stack(
            [0.2 + 0.03 * (band + 1) * x + 0.01 * y + 0.02 * z for band in range(6)]
        )
        matching = build_validation_context(
            target,
            target,
            spectral,
            spectral,
            geometry_fraction=0.5,
            attribute_fraction=0.5,
            config=VALIDATION_CONFIG,
        )
        mismatched = build_validation_context(
            target,
            target,
            np.roll(spectral, 23, axis=0),
            spectral,
            geometry_fraction=0.5,
            attribute_fraction=0.5,
            config=VALIDATION_CONFIG,
        )

        self.assertLess(
            score_transform(matching, np.eye(4)).score,
            score_transform(mismatched, np.eye(4)).score,
        )

    def test_attribute_outlier_is_clipped(self) -> None:
        target = grid_points()
        attributes = np.column_stack((target[:, 0], target[:, 1], target[:, 2]))
        source_attributes = attributes.copy()
        source_attributes[0] = 1e12
        context = build_validation_context(
            target,
            target,
            source_attributes,
            attributes,
            geometry_fraction=0.0,
            attribute_fraction=1.0,
            config={**VALIDATION_CONFIG, "trim_fraction": 1.0},
        )

        score = score_transform(context, np.eye(4))

        self.assertLessEqual(score.attribute_error, 1.0)

    def test_better_candidate_is_accepted(self) -> None:
        target = grid_points()
        perturbation = np.eye(4)
        perturbation[:3, 3] = [0.18, -0.08, 0.04]
        source = apply_transform(target, perturbation)
        context = build_validation_context(
            source,
            target,
            None,
            None,
            geometry_fraction=1.0,
            attribute_fraction=0.0,
            config=VALIDATION_CONFIG,
        )

        decision = decide_candidate(
            context,
            np.eye(4),
            invert_transform(perturbation),
        )

        self.assertTrue(decision.accepted)
        self.assertEqual(decision.rollback_reason, "")
        self.assertGreaterEqual(decision.relative_improvement, 0.005)

    def test_identical_candidate_is_rejected_for_insufficient_improvement(self) -> None:
        target = grid_points()
        context = build_validation_context(
            target,
            target,
            None,
            None,
            geometry_fraction=1.0,
            attribute_fraction=0.0,
            config=VALIDATION_CONFIG,
        )

        decision = decide_candidate(context, np.eye(4), np.eye(4))

        self.assertFalse(decision.accepted)
        self.assertEqual(
            decision.rollback_reason,
            "insufficient_score_improvement",
        )

    def test_nonfinite_candidate_is_rejected(self) -> None:
        target = grid_points()
        context = build_validation_context(
            target,
            target,
            None,
            None,
            geometry_fraction=1.0,
            attribute_fraction=0.0,
            config=VALIDATION_CONFIG,
        )
        candidate = np.eye(4)
        candidate[0, 0] = np.nan

        decision = decide_candidate(context, np.eye(4), candidate)

        self.assertFalse(decision.accepted)
        self.assertEqual(decision.rollback_reason, "nonfinite_transform")

    def test_candidate_without_correspondences_is_rejected(self) -> None:
        target = grid_points()
        context = build_validation_context(
            target,
            target,
            None,
            None,
            geometry_fraction=1.0,
            attribute_fraction=0.0,
            config=VALIDATION_CONFIG,
        )
        candidate = np.eye(4)
        candidate[:3, 3] = 100.0

        decision = decide_candidate(context, np.eye(4), candidate)

        self.assertFalse(decision.accepted)
        self.assertEqual(
            decision.rollback_reason,
            "insufficient_correspondences",
        )

    def test_candidate_with_large_coverage_drop_is_rejected(self) -> None:
        main = np.column_stack(
            (
                np.arange(95, dtype=float) * 2.0,
                np.zeros(95),
                np.zeros(95),
            )
        )
        anchors = np.column_stack(
            (
                1000.0 + np.arange(5, dtype=float) * 2.0,
                np.zeros(5),
                np.zeros(5),
            )
        )
        source = np.vstack((main, anchors))
        target = np.vstack(
            (
                main - np.array([0.20, 0.0, 0.0]),
                anchors + np.array([0.29, 0.0, 0.0]),
            )
        )
        config = {
            **VALIDATION_CONFIG,
            "voxel_size_m": 0.01,
            "min_correspondences": 10,
        }
        context = build_validation_context(
            source,
            target,
            None,
            None,
            geometry_fraction=1.0,
            attribute_fraction=0.0,
            config=config,
        )
        candidate = np.eye(4)
        candidate[:3, 3] = [-0.20, 0.0, 0.0]

        decision = decide_candidate(context, np.eye(4), candidate)

        self.assertGreater(decision.relative_improvement, 0.005)
        self.assertLess(
            decision.after.coverage,
            decision.before.coverage - 0.02,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.rollback_reason, "coverage_drop")

    def test_invalid_baseline_raises(self) -> None:
        target = grid_points()
        context = build_validation_context(
            target,
            target,
            None,
            None,
            geometry_fraction=1.0,
            attribute_fraction=0.0,
            config=VALIDATION_CONFIG,
        )
        invalid = np.eye(4)
        invalid[:3, 3] = 100.0

        with self.assertRaisesRegex(ValueError, "初始变换"):
            decide_candidate(context, invalid, np.eye(4))

    def test_invalid_weight_configuration_raises(self) -> None:
        target = grid_points()
        with self.assertRaises(ValueError):
            build_validation_context(
                target,
                target,
                None,
                None,
                geometry_fraction=0.0,
                attribute_fraction=0.0,
                config=VALIDATION_CONFIG,
            )


if __name__ == "__main__":
    unittest.main()
