from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.data_prepare import WAVELENGTHS_NM, prepare_snapshot
from src.evaluation import (
    bidirectional_overlap_percentages,
    evaluate_registration,
    full_metrics,
)
from src.transform import (
    apply_transform,
    generate_perturbations,
    invert_transform,
    transform_errors,
)


class DataPrepareTests(unittest.TestCase):
    def test_prepare_snapshot_converts_mm_to_m_and_builds_feature12(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source.npz"
            output = tmp_path / "prepared.npz"
            np.savez_compressed(
                source,
                xyz_mm=np.array([[1000.0, 2000.0, 3000.0], [4000.0, 5000.0, 6000.0]]),
                rgb=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                reflectance=np.arange(12, dtype=float).reshape(2, 6),
                wavelengths_nm=WAVELENGTHS_NM,
                source_scan_index=np.array([7, 11]),
                valid=np.ones(2, dtype=bool),
            )

            summary = prepare_snapshot(source, output, side="left")

            with np.load(output) as prepared:
                np.testing.assert_allclose(
                    prepared["xyz_m"],
                    np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                )
                self.assertEqual(prepared["feature12"].shape, (2, 12))
                np.testing.assert_array_equal(prepared["wavelengths_nm"], WAVELENGTHS_NM)
                np.testing.assert_array_equal(prepared["source_scan_index"], [7, 11])
                self.assertEqual(prepared["side"].item(), "left")
            self.assertEqual(summary["point_count"], 2)
            self.assertEqual(summary["coordinate_unit"], "m")


class TransformTests(unittest.TestCase):
    def test_seeded_perturbations_are_reproducible_and_bounded(self) -> None:
        first = generate_perturbations(10, 42, 10.0, 0.5)
        second = generate_perturbations(10, 42, 10.0, 0.5)
        self.assertEqual(len(first), 10)
        for lhs, rhs in zip(first, second, strict=True):
            np.testing.assert_allclose(lhs.matrix, rhs.matrix)
            self.assertTrue(np.all(np.abs(lhs.euler_deg) <= 10.0))
            self.assertTrue(np.all(np.abs(lhs.translation_m) <= 0.5))

    def test_apply_inverse_and_error_metrics(self) -> None:
        perturbation = generate_perturbations(1, 42, 10.0, 0.5)[0].matrix
        points = np.array([[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]])
        moved = apply_transform(points, perturbation)
        recovered = apply_transform(moved, invert_transform(perturbation))
        np.testing.assert_allclose(recovered, points, atol=1e-12)

        errors = transform_errors(invert_transform(perturbation), invert_transform(perturbation))
        self.assertAlmostEqual(errors["rotation_error_deg"], 0.0, places=10)
        self.assertAlmostEqual(errors["translation_error_m"], 0.0, places=10)


class EvaluationTests(unittest.TestCase):
    def test_bidirectional_overlap_uses_both_directions_and_includes_boundary(
        self,
    ) -> None:
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target = np.array([[0.1, 0.0, 0.0], [2.0, 0.0, 0.0]])

        result = bidirectional_overlap_percentages(
            source,
            target,
            thresholds_m=(0.10, 0.25),
        )

        self.assertAlmostEqual(result["overlap_100mm_pct"], 50.0)
        self.assertAlmostEqual(result["overlap_250mm_pct"], 50.0)

    def test_full_metrics_applies_estimated_transform_before_overlap(self) -> None:
        source = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.0, 0.0, 0.1],
            ]
        )
        target = source + np.array([1.0, 0.0, 0.0])
        estimated = np.eye(4)
        estimated[0, 3] = 1.0

        metrics = full_metrics(
            source,
            target,
            estimated,
            estimated,
            max_correspondence_m=0.2,
        )

        self.assertAlmostEqual(metrics["overlap_100mm_pct"], 100.0)
        self.assertAlmostEqual(metrics["overlap_250mm_pct"], 100.0)

    def test_identical_clouds_have_perfect_registration_metrics(self) -> None:
        points = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
        )
        metrics = evaluate_registration(points, points, np.eye(4), 0.2)
        self.assertAlmostEqual(metrics["fitness"], 1.0)
        self.assertAlmostEqual(metrics["inlier_rmse_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
