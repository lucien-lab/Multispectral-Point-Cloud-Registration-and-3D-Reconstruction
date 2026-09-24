from __future__ import annotations

import json
import math
import re
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from src.three_method_comparison import (
    METHODS,
    METRIC_FIELDS,
    MethodTrialMetrics,
    load_metric,
    metric_payload,
    select_shared_best_trial,
)


def sample_metric(
    method: str = "geometry",
    trial: int = 1,
    rotation_error_deg: float = 1.0,
    translation_error_m: float = 0.10,
    overlap_100mm_pct: float = 10.0,
    overlap_250mm_pct: float = 20.0,
) -> MethodTrialMetrics:
    return MethodTrialMetrics(
        method=method,
        trial=trial,
        rotation_error_deg=rotation_error_deg,
        translation_error_m=translation_error_m,
        overlap_100mm_pct=overlap_100mm_pct,
        overlap_250mm_pct=overlap_250mm_pct,
        global_chamfer_mm=1.0,
        trimmed_chamfer_mm=2.0,
        distance_median_mm=3.0,
        fitness=0.5,
        inlier_rmse_m=0.1,
    )


def make_rows(
    trial6: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    trial7: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> list[MethodTrialMetrics]:
    rows = []
    for trial, values in ((6, trial6), (7, trial7)):
        for method, (rotation, translation) in zip(METHODS, values):
            rows.append(
                sample_metric(
                    method=method,
                    trial=trial,
                    rotation_error_deg=rotation,
                    translation_error_m=translation,
                )
            )
    return rows


def translated(distance: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[0, 3] = distance
    return matrix


def write_transform_dir(
    root: Path, matrix: np.ndarray, transform_key: str | None = None
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in ("T_perturb.json", "T_gt.json", "T_est.json"):
        payload: object = matrix.tolist()
        if transform_key is not None:
            payload = {transform_key: payload}
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    return root


def write_nonempty_ply(path: Path, points: np.ndarray | None = None) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(
            ((0.0, 0.0, 0.0), (0.1, 0.0, 0.0)) if points is None else points
        )
    )
    if not o3d.io.write_point_cloud(str(path), cloud):
        raise RuntimeError(f"无法写入测试点云: {path}")


def write_trial(
    root: Path,
    method: str,
    trial: int,
    rotation_error_deg: float,
    translation_error_m: float,
    include_chamfer: bool = True,
    registered_points: np.ndarray | None = None,
) -> None:
    trial_dir = write_transform_dir(root / f"trial_{trial:03d}", np.eye(4))
    metric = metric_payload(
        sample_metric(
            method=method,
            trial=trial,
            rotation_error_deg=rotation_error_deg,
            translation_error_m=translation_error_m,
        )
    )
    metric["success"] = True
    if not include_chamfer:
        for field in (
            "global_chamfer_mm",
            "trimmed_chamfer_mm",
            "distance_median_mm",
        ):
            metric.pop(field)
    (trial_dir / "metrics.json").write_text(
        json.dumps(metric), encoding="utf-8"
    )
    for name in (
        "source_target_initial_merged.ply",
        "source_target_merged.ply",
        "source_target_rgb_merged.ply",
    ):
        write_nonempty_ply(trial_dir / name)
    write_nonempty_ply(trial_dir / "source_registered.ply", registered_points)


def write_target_snapshot(geometry_root: Path, points: np.ndarray) -> None:
    path = geometry_root / "data" / "processed" / "right_points.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, xyz_m=points)


class ThreeMethodComparisonTests(unittest.TestCase):
    def test_target_snapshot_rejects_nan_or_inf_coordinates(self) -> None:
        from build_three_method_best_trial_report import _load_target_xyz

        with tempfile.TemporaryDirectory() as tmp:
            geometry_root = Path(tmp) / "geometry"
            write_target_snapshot(
                geometry_root,
                np.asarray(((0.0, 0.0, 0.0), (np.nan, 0.0, 0.0))),
            )

            with self.assertRaisesRegex(ValueError, r"right_points\.npz.*NaN或Inf"):
                _load_target_xyz(geometry_root)

    def test_build_report_backfills_missing_chamfer_from_registered_ply(self) -> None:
        from build_three_method_best_trial_report import build_report

        source_points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
        target_points = np.asarray(((0.1, 0.0, 0.0), (1.2, 0.0, 0.0)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "spectral6"
            geometry_root = Path(tmp) / "geometry"
            output = Path(tmp) / "output"
            method_roots = {
                "geometry": geometry_root / "results" / "icp_geometry",
                "rgb": geometry_root / "results" / "icp_rgb",
                "spectral6_robust": (
                    root
                    / "results/spectral6_recommended_experiment/round1/robust_full"
                ),
            }
            write_target_snapshot(geometry_root, target_points)
            for trial in (6, 7, 8, 9, 10):
                for method, method_root in method_roots.items():
                    write_trial(
                        method_root,
                        method,
                        trial,
                        rotation_error_deg=0.5 if trial == 7 else 1.0,
                        translation_error_m=0.05 if trial == 7 else 0.10,
                        include_chamfer=method == "spectral6_robust",
                        registered_points=source_points,
                    )

            build_report(root, geometry_root, output)

            for method in ("geometry", "rgb"):
                metrics = json.loads((output / method / "metrics.json").read_text())
                self.assertAlmostEqual(metrics["global_chamfer_mm"], 150.0)
                self.assertAlmostEqual(metrics["trimmed_chamfer_mm"], 100.0)
                self.assertAlmostEqual(metrics["distance_median_mm"], 150.0)
            robust_metrics = json.loads(
                (output / "spectral6_robust" / "metrics.json").read_text()
            )
            self.assertEqual(robust_metrics["global_chamfer_mm"], 1.0)

    def test_cli_uses_explicit_roots_and_default_output_directory(self) -> None:
        from build_three_method_best_trial_report import main

        with tempfile.TemporaryDirectory() as tmp:
            experiment_root = Path(tmp) / "experiment"
            geometry_root = Path(tmp) / "geometry"
            experiment_root.mkdir()
            geometry_root.mkdir()
            stdout = StringIO()
            with (
                patch("build_three_method_best_trial_report.build_report", return_value=7) as build,
                patch.object(
                    sys,
                    "argv",
                    [
                        "build_three_method_best_trial_report.py",
                        "--experiment-root",
                        str(experiment_root),
                        "--geometry-experiment-root",
                        str(geometry_root),
                    ],
                ),
                patch("sys.stdout", stdout),
            ):
                main()

            output = experiment_root / "results" / "three_method_best_trial_comparison"
            build.assert_called_once_with(
                experiment_root.resolve(), geometry_root.resolve(), output.resolve()
            )
            self.assertEqual(
                stdout.getvalue(),
                f"selected_trial=7\noutput_dir={output.resolve()}\n",
            )

    def test_cli_rejects_missing_default_geometry_root_with_override_hint(self) -> None:
        from build_three_method_best_trial_report import main

        with tempfile.TemporaryDirectory() as tmp:
            experiment_root = Path(tmp) / "worktree" / "2026_07_26实验"
            experiment_root.mkdir(parents=True)
            with patch.object(
                sys,
                "argv",
                [
                    "build_three_method_best_trial_report.py",
                    "--experiment-root",
                    str(experiment_root),
                ],
            ):
                with self.assertRaisesRegex(
                    FileNotFoundError, "--geometry-experiment-root"
                ):
                    main()

    def test_transform_validation_accepts_experiment_matrix_wrappers(self) -> None:
        from src.three_method_comparison import verify_shared_transforms

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verify_shared_transforms(
                {
                    "geometry": write_transform_dir(
                        root / "geometry", np.eye(4), "matrix"
                    ),
                    "rgb": write_transform_dir(root / "rgb", np.eye(4), "matrix"),
                    "spectral6_robust": write_transform_dir(
                        root / "spectral6_robust", np.eye(4), "transformation"
                    ),
                }
            )

    def test_transform_mismatch_is_rejected(self) -> None:
        from src.three_method_comparison import verify_shared_transforms

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "T_perturb"):
                verify_shared_transforms(
                    {
                        "geometry": write_transform_dir(root / "geometry", np.eye(4)),
                        "rgb": write_transform_dir(root / "rgb", translated(0.01)),
                        "spectral6_robust": write_transform_dir(
                            root / "spectral6_robust", np.eye(4)
                        ),
                    }
                )

    def test_build_report_writes_three_complete_method_directories(self) -> None:
        from build_three_method_best_trial_report import build_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "spectral6"
            geometry_root = Path(tmp) / "geometry"
            output = Path(tmp) / "output"
            method_roots = {
                "geometry": geometry_root / "results" / "icp_geometry",
                "rgb": geometry_root / "results" / "icp_rgb",
                "spectral6_robust": (
                    root
                    / "results/spectral6_recommended_experiment/round1/robust_full"
                ),
            }
            for trial in (6, 7, 8, 9, 10):
                for method, method_root in method_roots.items():
                    write_trial(
                        method_root,
                        method,
                        trial,
                        rotation_error_deg=0.5 if trial == 7 else 1.0,
                        translation_error_m=0.05 if trial == 7 else 0.10,
                    )

            selected = build_report(root, geometry_root, output)

            self.assertEqual(selected, 7)
            for method in METHODS:
                method_dir = output / method
                for name in (
                    "initial_merged.ply",
                    "registered_merged.ply",
                    "registered_true_rgb.ply",
                    "metrics.json",
                    "T_perturb.json",
                    "T_gt.json",
                    "T_est.json",
                ):
                    self.assertTrue((method_dir / name).is_file())
            summary = (output / "metrics_summary.csv").read_text(encoding="utf-8")
            self.assertNotIn("success", summary)
            for path in output.rglob("*"):
                if path.suffix in {".csv", ".json", ".md"}:
                    self.assertIsNone(
                        re.search(r"success|成功率", path.read_text(encoding="utf-8"), re.I),
                        path,
                    )

    def test_selects_one_shared_trial_from_three_method_mean_score(self) -> None:
        rows = make_rows(
            trial6=((1.0, 0.10), (1.2, 0.11), (0.8, 0.09)),
            trial7=((0.5, 0.05), (0.6, 0.06), (0.7, 0.07)),
        )

        selected, scores = select_shared_best_trial(rows)

        self.assertEqual(selected, 7)
        self.assertEqual({row["trial"] for row in scores}, {6, 7})

    def test_uses_overlap_then_trial_number_as_deterministic_tie_breakers(self) -> None:
        rows = make_rows(
            trial6=((1.0, 0.10), (1.0, 0.10), (1.0, 0.10)),
            trial7=((1.0, 0.10), (1.0, 0.10), (1.0, 0.10)),
        )
        rows[0] = sample_metric(method="geometry", trial=6, overlap_100mm_pct=11.0)
        rows[1] = sample_metric(method="rgb", trial=6, overlap_100mm_pct=11.0)
        rows[2] = sample_metric(method="spectral6_robust", trial=6, overlap_100mm_pct=11.0)

        selected, scores = select_shared_best_trial(rows)

        self.assertEqual(selected, 6)
        self.assertEqual(scores[0]["trial"], 6)

    def test_uses_250mm_overlap_when_pose_and_100mm_overlap_tie(self) -> None:
        rows = make_rows(
            trial6=((1.0, 0.10), (1.0, 0.10), (1.0, 0.10)),
            trial7=((1.0, 0.10), (1.0, 0.10), (1.0, 0.10)),
        )
        for index, method in enumerate(METHODS):
            rows[index] = sample_metric(
                method=method,
                trial=6,
                overlap_100mm_pct=10.0,
                overlap_250mm_pct=21.0,
            )
            rows[index + 3] = sample_metric(
                method=method,
                trial=7,
                overlap_100mm_pct=10.0,
                overlap_250mm_pct=22.0,
            )

        selected, _ = select_shared_best_trial(rows)

        self.assertEqual(selected, 7)

    def test_uses_smaller_trial_when_all_scores_tie(self) -> None:
        rows = make_rows(
            trial6=((1.0, 0.10), (1.0, 0.10), (1.0, 0.10)),
            trial7=((1.0, 0.10), (1.0, 0.10), (1.0, 0.10)),
        )

        selected, _ = select_shared_best_trial(rows)

        self.assertEqual(selected, 6)

    def test_rejects_trials_outside_fixed_candidate_range(self) -> None:
        rows = [sample_metric(method=method, trial=11) for method in METHODS]

        with self.assertRaises(ValueError):
            select_shared_best_trial(rows)

    def test_rejects_trial_with_missing_declared_method(self) -> None:
        rows = [
            sample_metric(method="geometry", trial=6),
            sample_metric(method="rgb", trial=6),
        ]

        with self.assertRaisesRegex(
            ValueError, "missing required methods: spectral6_robust"
        ):
            select_shared_best_trial(rows)

    def test_rejects_trial_with_extra_method(self) -> None:
        rows = [sample_metric(method=method, trial=6) for method in METHODS]
        rows.append(sample_metric(method="other", trial=6))

        with self.assertRaisesRegex(ValueError, "unsupported methods: other"):
            select_shared_best_trial(rows)

    def test_rejects_trial_with_duplicate_method(self) -> None:
        rows = [
            sample_metric(method="geometry", trial=6),
            sample_metric(method="rgb", trial=6),
            sample_metric(method="rgb", trial=6),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate methods: rgb"):
            select_shared_best_trial(rows)

    def test_metric_payload_excludes_success_fields(self) -> None:
        payload = metric_payload(sample_metric())

        self.assertNotIn("success", payload)
        self.assertNotIn("success_rate", payload)
        self.assertEqual(set(payload), set(METRIC_FIELDS))

    def test_load_metric_ignores_success_and_rejects_missing_or_nonfinite_values(self) -> None:
        payload = metric_payload(sample_metric())
        payload["success"] = True
        with tempfile.TemporaryDirectory() as tmp:
            metric_path = Path(tmp) / "metrics.json"
            metric_path.write_text(json.dumps(payload), encoding="utf-8")
            metric = load_metric(metric_path, "geometry", 9)
            self.assertEqual(metric.method, "geometry")
            self.assertEqual(metric.trial, 9)

            payload.pop("fitness")
            metric_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metric(metric_path, "geometry", 9)

            payload["fitness"] = 0.5
            payload.pop("global_chamfer_mm")
            metric_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metric(metric_path, "geometry", 9)

            payload["global_chamfer_mm"] = 1.0
            payload["fitness"] = math.inf
            metric_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metric(metric_path, "geometry", 9)


if __name__ == "__main__":
    unittest.main()
