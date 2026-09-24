from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.data_prepare import WAVELENGTHS_NM
from src.spectral_calibration import (
    CalibrationTrial,
    apply_source_calibration,
    fit_overlap_calibration,
)


def planar_grid(side: int = 12, spacing: float = 0.04) -> np.ndarray:
    axis = np.arange(side, dtype=np.float64) * spacing
    xx, yy = np.meshgrid(axis, axis)
    return np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))


class SpectralCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.points = planar_grid()
        x, y = self.points[:, :2].T
        self.source_spectral = np.column_stack(
            (
                0.20 + 0.30 * x,
                0.15 + 0.20 * y,
                0.10 + 0.15 * (x + y),
                0.30 + 0.10 * x - 0.05 * y,
                0.12 + 0.25 * x,
                0.18 + 0.22 * y,
            )
        )
        self.gain = np.array([1.20, 0.85, 1.50, 0.70, 1.10, 0.95])
        self.offset = np.array([0.01, -0.02, 0.03, 0.04, -0.01, 0.02])
        self.target_spectral = self.source_spectral * self.gain + self.offset
        self.trials = [
            CalibrationTrial(
                trial=1,
                source_to_target=np.eye(4),
            )
        ]
        self.config = {
            "max_correspondence_m": 0.10,
            "normal_angle_deg": 30.0,
            "normal_radius_m": 0.12,
            "min_pairs": 100,
            "gain_min": 0.25,
            "gain_max": 4.0,
            "soft_l1_scale": 0.02,
            "weight_floor": 0.05,
            "weight_ceiling": 0.35,
        }

    def test_fit_recovers_affine_response_and_normalized_band_weights(self) -> None:
        noisy = self.target_spectral.copy()
        rng = np.random.default_rng(9)
        noisy[:, 5] += rng.normal(0.0, 0.08, len(noisy))
        noisy[[3, 17, 81], 0] += 1.5

        result = fit_overlap_calibration(
            self.trials,
            self.points,
            self.points,
            self.source_spectral,
            noisy,
            WAVELENGTHS_NM,
            config=self.config,
        )

        np.testing.assert_allclose(result.gain[:5], self.gain[:5], atol=0.04)
        np.testing.assert_allclose(result.offset[:5], self.offset[:5], atol=0.015)
        self.assertEqual(result.pair_count, len(self.points))
        self.assertAlmostEqual(float(result.band_weights.sum()), 1.0, places=12)
        self.assertTrue(np.all(result.band_weights >= 0.05 - 1e-12))
        self.assertTrue(np.all(result.band_weights <= 0.35 + 1e-12))
        self.assertLess(result.band_weights[5], result.band_weights[1])

    def test_apply_source_calibration_uses_source_to_target_direction(self) -> None:
        result = fit_overlap_calibration(
            self.trials,
            self.points,
            self.points,
            self.source_spectral,
            self.target_spectral,
            WAVELENGTHS_NM,
            config=self.config,
        )

        corrected = apply_source_calibration(self.source_spectral, result)

        np.testing.assert_allclose(corrected, self.target_spectral, atol=1e-8)

    def test_rejects_wrong_wavelength_order_nonfinite_values_and_too_few_pairs(
        self,
    ) -> None:
        wrong_order = WAVELENGTHS_NM[[0, 2, 1, 3, 4, 5]]
        with self.assertRaisesRegex(ValueError, "波长顺序"):
            fit_overlap_calibration(
                self.trials,
                self.points,
                self.points,
                self.source_spectral,
                self.target_spectral,
                wrong_order,
                config=self.config,
            )

        nonfinite = self.source_spectral.copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN|Inf"):
            fit_overlap_calibration(
                self.trials,
                self.points,
                self.points,
                nonfinite,
                self.target_spectral,
                WAVELENGTHS_NM,
                config=self.config,
            )

        few_points = planar_grid(side=9)
        with self.assertRaisesRegex(ValueError, "至少需要100"):
            fit_overlap_calibration(
                [
                    CalibrationTrial(
                        trial=1,
                        source_to_target=np.eye(4),
                    )
                ],
                few_points,
                few_points,
                self.source_spectral[: len(few_points)],
                self.target_spectral[: len(few_points)],
                WAVELENGTHS_NM,
                config=self.config,
            )


if __name__ == "__main__":
    unittest.main()
