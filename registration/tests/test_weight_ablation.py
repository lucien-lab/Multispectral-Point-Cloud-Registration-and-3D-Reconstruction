from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from run_spectral_weight_ablation import (
    archive_baseline,
    build_comparison_rows,
    summarize_comparison,
)


RESULT_FILES = (
    "method_summary.csv",
    "method_summary.md",
    "all_trials_method_summary.csv",
    "all_trials_method_summary.md",
    "heldout_method_summary.csv",
    "heldout_method_summary.md",
    "registration_error_boxplots.png",
    "coarse_and_fine_error_boxplots.png",
)


def result_row(
    method: str,
    trial: int,
    *,
    rotation: float | None = None,
    translation: float | None = None,
) -> dict[str, object]:
    return {
        "method": method,
        "trial": trial,
        "status": "ok",
        "rotation_error_deg": rotation if rotation is not None else trial / 10,
        "translation_error_m": (
            translation if translation is not None else trial / 100
        ),
        "fitness": 0.4 + trial / 100,
        "inlier_rmse_m": 0.05 + trial / 1000,
        "success": trial <= 5,
        "runtime_s": 0.1,
        "coarse_method": method,
        "coarse_config_id": (
            "spectral6_v0.10_f2.0_d0.30"
            if method == "spectral6"
            else f"{method}_config"
        ),
        "coarse_rotation_error_deg": 1.0,
        "coarse_translation_error_m": 0.1,
        "coarse_fitness": 0.4,
        "coarse_inlier_rmse_m": 0.06,
        "coarse_runtime_s": 0.2,
        "error_message": "",
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_minimal_formal_results(root: Path) -> list[dict[str, object]]:
    results = root / "results"
    rows = [
        result_row(method, trial)
        for trial in range(1, 11)
        for method in ("geometry", "rgb", "spectral6")
    ]
    write_rows(results / "results_summary.csv", rows)
    for name in RESULT_FILES:
        path = results / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    for trial in range(1, 11):
        fine = results / "icp_spectral6" / f"trial_{trial:03d}"
        coarse = (
            results
            / "coarse_registration"
            / "spectral6"
            / f"trial_{trial:03d}"
        )
        fine.mkdir(parents=True)
        coarse.mkdir(parents=True)
        (fine / "metrics.json").write_text("{}", encoding="utf-8")
        (coarse / "coarse_metrics.json").write_text("{}", encoding="utf-8")
    (root / "experiment_log.md").write_text("baseline", encoding="utf-8")
    return rows


class BaselineArchiveTests(unittest.TestCase):
    def test_archive_baseline_preserves_existing_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_minimal_formal_results(root)

            first = archive_baseline(root)
            original = (first / "results_summary.csv").read_bytes()
            (root / "results" / "results_summary.csv").write_text(
                "changed\n",
                encoding="utf-8",
            )

            second = archive_baseline(root)

            self.assertEqual(second, first)
            self.assertEqual(
                (second / "results_summary.csv").read_bytes(),
                original,
            )


class ComparisonTests(unittest.TestCase):
    def test_build_comparison_rows_aligns_variants_and_weights(self) -> None:
        baseline = [
            result_row(method, trial)
            for trial in range(1, 3)
            for method in ("geometry", "rgb", "spectral6")
        ]
        current = [
            result_row(
                "spectral6",
                trial,
                rotation=1.0 + trial,
                translation=0.2 + trial / 100,
            )
            for trial in range(1, 3)
        ]

        rows = build_comparison_rows(baseline, current)

        self.assertEqual(len(rows), 6)
        self.assertEqual(
            {
                (
                    row["variant"],
                    row["lambda_geo"],
                    row["lambda_attribute"],
                )
                for row in rows
            },
            {
                ("rgb_baseline_0968_0032", 0.968, 0.032),
                ("spectral6_baseline_050_050", 0.5, 0.5),
                ("spectral6_new_0968_0032", 0.968, 0.032),
            },
        )
        self.assertEqual(
            [int(row["trial"]) for row in rows],
            [1, 1, 1, 2, 2, 2],
        )
        new_trial_one = next(
            row
            for row in rows
            if row["variant"] == "spectral6_new_0968_0032"
            and int(row["trial"]) == 1
        )
        self.assertEqual(float(new_trial_one["rotation_error_deg"]), 2.0)

    def test_summarize_comparison_uses_literal_split_statistics(self) -> None:
        rows: list[dict[str, object]] = []
        for trial, rotation, translation, success in (
            (1, 1.0, 0.05, True),
            (2, 3.0, 0.15, False),
            (6, 2.0, 0.10, True),
            (7, 4.0, 0.30, False),
        ):
            rows.append(
                {
                    "variant": "spectral6_new_0968_0032",
                    "trial": trial,
                    "status": "ok",
                    "rotation_error_deg": rotation,
                    "translation_error_m": translation,
                    "fitness": 0.5,
                    "inlier_rmse_m": 0.1,
                    "success": success,
                    "accepted_levels": 1,
                    "rollback_levels": 2,
                }
            )

        summaries = summarize_comparison(rows)
        heldout = next(
            row
            for row in summaries
            if row["split"] == "heldout_6_10"
        )

        self.assertEqual(heldout["trial_count"], 2)
        self.assertEqual(heldout["success_count"], 1)
        self.assertAlmostEqual(float(heldout["success_rate"]), 0.5)
        self.assertAlmostEqual(
            float(heldout["rotation_error_deg_mean"]),
            3.0,
        )
        self.assertAlmostEqual(
            float(heldout["rotation_error_deg_median"]),
            3.0,
        )
        self.assertAlmostEqual(
            float(heldout["translation_error_m_mean"]),
            0.2,
        )
        self.assertAlmostEqual(
            float(heldout["translation_error_m_median"]),
            0.2,
        )
        self.assertEqual(heldout["accepted_levels"], 2)
        self.assertEqual(heldout["rollback_levels"], 4)


if __name__ == "__main__":
    unittest.main()
