from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


METHODS = ("geometry", "rgb", "spectral6_robust")
CANDIDATE_TRIALS = (6, 7, 8, 9, 10)
METRIC_FIELDS = (
    "method",
    "trial",
    "rotation_error_deg",
    "translation_error_m",
    "overlap_100mm_pct",
    "overlap_250mm_pct",
    "global_chamfer_mm",
    "trimmed_chamfer_mm",
    "distance_median_mm",
    "fitness",
    "inlier_rmse_m",
)
_INPUT_METRIC_FIELDS = METRIC_FIELDS[2:]


@dataclass(frozen=True)
class MethodTrialMetrics:
    method: str
    trial: int
    rotation_error_deg: float
    translation_error_m: float
    overlap_100mm_pct: float
    overlap_250mm_pct: float
    global_chamfer_mm: float
    trimmed_chamfer_mm: float
    distance_median_mm: float
    fitness: float
    inlier_rmse_m: float

    @property
    def pose_score(self) -> float:
        return self.rotation_error_deg / 2.0 + self.translation_error_m / 0.10


def load_metric(
    path: Path,
    method: str,
    trial: int,
    missing_metric_values: Mapping[str, float] | None = None,
) -> MethodTrialMetrics:
    """Load only finite, whitelisted metric values from a metrics JSON file."""
    try:
        raw_metric = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load metric file: {path}") from error

    if not isinstance(raw_metric, dict):
        raise ValueError(f"Metric file must contain an object: {path}")

    values: dict[str, float] = {}
    for field in _INPUT_METRIC_FIELDS:
        value = raw_metric.get(field)
        if field not in raw_metric and missing_metric_values is not None:
            value = missing_metric_values.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Missing or invalid metric value {field!r} in {path}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite metric value {field!r} in {path}")
        values[field] = value

    return MethodTrialMetrics(method=method, trial=trial, **values)


def metric_payload(metric: MethodTrialMetrics) -> dict[str, object]:
    """Return the explicit metric whitelist, excluding input status fields."""
    return asdict(metric)


def verify_shared_transforms(method_dirs: Mapping[str, Path]) -> None:
    """Require each declared method to share identical perturbation and GT matrices."""
    missing_methods = tuple(method for method in METHODS if method not in method_dirs)
    if missing_methods:
        raise ValueError(f"Missing transform directories: {', '.join(missing_methods)}")
    extra_methods = tuple(method for method in method_dirs if method not in METHODS)
    if extra_methods:
        raise ValueError(f"Unsupported transform directories: {', '.join(extra_methods)}")

    for transform_name in ("T_perturb.json", "T_gt.json"):
        reference: np.ndarray | None = None
        for method in METHODS:
            path = method_dirs[method] / transform_name
            try:
                raw_matrix = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw_matrix, dict):
                    raw_matrix = raw_matrix.get(
                        "matrix", raw_matrix.get("transformation")
                    )
                matrix = np.asarray(raw_matrix, dtype=float)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Cannot load {transform_name}: {path}") from error
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"Invalid {transform_name}: {path}")
            if reference is None:
                reference = matrix
            elif not np.allclose(matrix, reference, atol=1e-12, rtol=0.0):
                raise ValueError(f"Mismatched {transform_name}: {path}")


def select_shared_best_trial(
    rows: Sequence[MethodTrialMetrics],
) -> tuple[int, list[dict[str, object]]]:
    """Select the best trial shared by all declared registration methods."""
    by_trial: dict[int, list[MethodTrialMetrics]] = {}
    for row in rows:
        if row.trial not in CANDIDATE_TRIALS:
            raise ValueError(
                f"Trial {row.trial} is outside candidate trials: {CANDIDATE_TRIALS}"
            )
        by_trial.setdefault(row.trial, []).append(row)

    scores: list[dict[str, object]] = []
    for trial, trial_rows in by_trial.items():
        methods = tuple(row.method for row in trial_rows)
        unsupported_methods = tuple(
            method for method in methods if method not in METHODS
        )
        if unsupported_methods:
            raise ValueError(
                f"Trial {trial} has unsupported methods: {', '.join(unsupported_methods)}"
            )
        duplicate_methods = tuple(
            method for method in METHODS if methods.count(method) > 1
        )
        if duplicate_methods:
            raise ValueError(
                f"Trial {trial} has duplicate methods: {', '.join(duplicate_methods)}"
            )
        missing_methods = tuple(method for method in METHODS if method not in methods)
        if missing_methods:
            raise ValueError(
                f"Trial {trial} is missing required methods: {', '.join(missing_methods)}"
            )

        mean_pose_score = sum(row.pose_score for row in trial_rows) / len(METHODS)
        mean_overlap_100 = (
            sum(row.overlap_100mm_pct for row in trial_rows) / len(METHODS)
        )
        mean_overlap_250 = (
            sum(row.overlap_250mm_pct for row in trial_rows) / len(METHODS)
        )
        scores.append(
            {
                "trial": trial,
                "mean_pose_score": mean_pose_score,
                "mean_overlap_100mm_pct": mean_overlap_100,
                "mean_overlap_250mm_pct": mean_overlap_250,
            }
        )

    if not scores:
        raise ValueError("At least one complete shared trial is required")

    scores.sort(
        key=lambda score: (
            score["mean_pose_score"],
            -score["mean_overlap_100mm_pct"],
            -score["mean_overlap_250mm_pct"],
            score["trial"],
        )
    )
    return int(scores[0]["trial"]), scores
