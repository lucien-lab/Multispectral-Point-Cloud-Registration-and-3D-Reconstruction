from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.spectral_experiment import (
    ROUND1_SCHEDULES,
    ROUND2_EXTRA_SCHEDULES,
    VARIANTS,
    beats_rgb_reference,
    chamfer_metrics_mm,
    protocol_rounds,
    select_best_schedule,
    trial_split,
)


def metric_row(
    trial: int,
    *,
    rotation: float,
    translation: float,
    overlap100: float,
    overlap250: float,
    success: bool,
    schedule_id: str = "a",
) -> dict[str, object]:
    return {
        "trial": trial,
        "status": "ok",
        "rotation_error_deg": rotation,
        "translation_error_m": translation,
        "overlap_100mm_pct": overlap100,
        "overlap_250mm_pct": overlap250,
        "success": success,
        "runtime_s": 1.0,
        "schedule_id": schedule_id,
    }


class SpectralExperimentProtocolTests(unittest.TestCase):
    def test_protocol_has_four_ablation_variants_and_fixed_schedules(self) -> None:
        self.assertEqual(
            VARIANTS,
            (
                "baseline",
                "calibrated",
                "aligned_validation",
                "robust_full",
            ),
        )
        self.assertEqual(len(ROUND1_SCHEDULES), 4)
        self.assertEqual(len(ROUND2_EXTRA_SCHEDULES), 4)
        self.assertEqual(ROUND1_SCHEDULES[0], (0.02, 0.05, 0.10))
        self.assertEqual(ROUND2_EXTRA_SCHEDULES[-1], (0.02, 0.15, 0.35))

    def test_trial_splits_keep_confirmation_trials_out_of_tuning(self) -> None:
        self.assertEqual(trial_split(1), ("tune", tuple(range(1, 6))))
        self.assertEqual(trial_split(1, evaluation=True), ("heldout", tuple(range(6, 11))))
        self.assertEqual(trial_split(2), ("retune", tuple(range(1, 11))))
        self.assertEqual(
            trial_split(2, evaluation=True),
            ("confirmation", tuple(range(11, 21))),
        )
        self.assertEqual(protocol_rounds(round1_passed=True), (1,))
        self.assertEqual(protocol_rounds(round1_passed=False), (1, 2))

    def test_schedule_selection_ignores_rows_outside_declared_training_trials(
        self,
    ) -> None:
        rows = [
            metric_row(
                trial,
                rotation=0.5 if schedule == "a" else 1.5,
                translation=0.03 if schedule == "a" else 0.08,
                overlap100=40.0,
                overlap250=50.0,
                success=True,
                schedule_id=schedule,
            )
            for schedule in ("a", "b")
            for trial in range(1, 6)
        ]
        rows.extend(
            [
                metric_row(
                    trial,
                    rotation=9.0,
                    translation=0.9,
                    overlap100=10.0,
                    overlap250=20.0,
                    success=False,
                    schedule_id="a",
                )
                for trial in range(6, 11)
            ]
        )

        selected = select_best_schedule(rows, training_trials=range(1, 6))

        self.assertEqual(selected["schedule_id"], "a")
        self.assertEqual(selected["trial_count"], 5)

    def test_joint_rgb_gate_requires_both_pose_metrics_success_and_overlap(
        self,
    ) -> None:
        rgb = [
            metric_row(
                trial,
                rotation=1.0,
                translation=0.12,
                overlap100=38.0,
                overlap250=43.0,
                success=trial <= 7,
            )
            for trial in range(6, 11)
        ]
        spectral = [
            metric_row(
                trial,
                rotation=0.9,
                translation=0.10,
                overlap100=37.2,
                overlap250=42.1,
                success=trial <= 7,
            )
            for trial in range(6, 11)
        ]

        decision = beats_rgb_reference(spectral, rgb, overlap_tolerance_pct=1.0)

        self.assertTrue(decision["passed"])
        worse_overlap = [dict(row, overlap_100mm_pct=36.9) for row in spectral]
        self.assertFalse(
            beats_rgb_reference(
                worse_overlap,
                rgb,
                overlap_tolerance_pct=1.0,
            )["passed"]
        )
        incomplete = spectral[:-1] + [
            {
                **spectral[-1],
                "status": "error",
                "success": False,
            }
        ]
        self.assertFalse(
            beats_rgb_reference(
                incomplete,
                rgb,
                overlap_tolerance_pct=1.0,
            )["passed"]
        )
        all_failed = [
            {
                **row,
                "status": "error",
                "success": False,
            }
            for row in spectral
        ]
        decision = beats_rgb_reference(
            all_failed,
            rgb,
            overlap_tolerance_pct=1.0,
        )
        self.assertFalse(decision["passed"])
        self.assertEqual(decision["reason"], "missing_valid_results")

    def test_chamfer_metrics_are_bidirectional_and_report_millimetres(self) -> None:
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        target = np.array([[0.1, 0.0, 0.0], [1.2, 0.0, 0.0]])

        metrics = chamfer_metrics_mm(source, target, trim_fraction=0.70)

        self.assertAlmostEqual(metrics["global_chamfer_mm"], 150.0)
        self.assertAlmostEqual(metrics["distance_median_mm"], 150.0)
        self.assertAlmostEqual(metrics["trimmed_chamfer_mm"], 100.0)


if __name__ == "__main__":
    unittest.main()
