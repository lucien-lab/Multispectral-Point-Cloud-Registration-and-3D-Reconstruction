"""六通道独立实验的固定协议、选参与评价纯函数。"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree


VARIANTS = (
    "baseline",
    "calibrated",
    "aligned_validation",
    "robust_full",
)
ROUND1_SCHEDULES = (
    (0.02, 0.05, 0.10),
    (0.03, 0.08, 0.15),
    (0.05, 0.10, 0.20),
    (0.02, 0.10, 0.25),
)
ROUND2_EXTRA_SCHEDULES = (
    (0.01, 0.03, 0.06),
    (0.05, 0.15, 0.30),
    (0.10, 0.20, 0.35),
    (0.02, 0.15, 0.35),
)


def protocol_rounds(*, round1_passed: bool) -> tuple[int, ...]:
    return (1,) if round1_passed else (1, 2)


def trial_split(
    round_number: int,
    *,
    evaluation: bool = False,
) -> tuple[str, tuple[int, ...]]:
    if round_number == 1:
        return (
            ("heldout", tuple(range(6, 11)))
            if evaluation
            else ("tune", tuple(range(1, 6)))
        )
    if round_number == 2:
        return (
            ("confirmation", tuple(range(11, 21)))
            if evaluation
            else ("retune", tuple(range(1, 11)))
        )
    raise ValueError("round_number必须是1或2")


def _valid_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return [
        row
        for row in rows
        if row.get("status") == "ok"
        and np.isfinite(float(row["rotation_error_deg"]))
        and np.isfinite(float(row["translation_error_m"]))
    ]


def select_best_schedule(
    rows: Iterable[dict[str, object]],
    *,
    training_trials: Iterable[int],
) -> dict[str, object]:
    allowed = {int(value) for value in training_trials}
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if int(row["trial"]) in allowed:
            grouped[str(row["schedule_id"])].append(row)
    if not grouped:
        raise ValueError("没有属于指定训练trial的schedule结果")

    summaries: list[dict[str, object]] = []
    for schedule_id, schedule_rows in sorted(grouped.items()):
        success_count = 0
        normalized: list[float] = []
        translations: list[float] = []
        runtimes: list[float] = []
        for row in schedule_rows:
            runtimes.append(float(row.get("runtime_s", 0.0)))
            if row.get("status") != "ok":
                normalized.append(1_000_000.0)
                translations.append(1_000_000.0)
                continue
            rotation = float(row["rotation_error_deg"])
            translation = float(row["translation_error_m"])
            normalized.append(rotation / 2.0 + translation / 0.10)
            translations.append(translation)
            success_count += int(bool(row["success"]))
        summaries.append(
            {
                "schedule_id": schedule_id,
                "trial_count": len(schedule_rows),
                "success_count": success_count,
                "normalized_error_mean": float(np.mean(normalized)),
                "translation_error_m_median": float(
                    np.median(translations)
                ),
                "runtime_s_mean": float(np.mean(runtimes)),
            }
        )
    return min(
        summaries,
        key=lambda item: (
            -int(item["success_count"]),
            float(item["normalized_error_mean"]),
            float(item["translation_error_m_median"]),
            float(item["runtime_s_mean"]),
            str(item["schedule_id"]),
        ),
    )


def _metric_means(rows: Iterable[dict[str, object]]) -> dict[str, float]:
    valid = _valid_rows(rows)
    if not valid:
        raise ValueError("没有可用于RGB门槛判断的有效结果")
    return {
        "rotation_error_deg": float(
            np.mean([float(row["rotation_error_deg"]) for row in valid])
        ),
        "translation_error_m": float(
            np.mean([float(row["translation_error_m"]) for row in valid])
        ),
        "overlap_100mm_pct": float(
            np.mean([float(row["overlap_100mm_pct"]) for row in valid])
        ),
        "overlap_250mm_pct": float(
            np.mean([float(row["overlap_250mm_pct"]) for row in valid])
        ),
        "success_rate": float(
            np.mean([bool(row["success"]) for row in valid])
        ),
    }


def beats_rgb_reference(
    spectral_rows: Iterable[dict[str, object]],
    rgb_rows: Iterable[dict[str, object]],
    *,
    overlap_tolerance_pct: float,
) -> dict[str, object]:
    spectral_input = list(spectral_rows)
    rgb_input = list(rgb_rows)
    spectral_valid = _valid_rows(spectral_input)
    rgb_valid = _valid_rows(rgb_input)
    if not spectral_valid or not rgb_valid:
        return {
            "passed": False,
            "reason": "missing_valid_results",
            "checks": {
                "all_trials_completed": False,
            },
            "spectral_valid_count": len(spectral_valid),
            "rgb_valid_count": len(rgb_valid),
            "spectral_trial_count": len(spectral_input),
            "rgb_trial_count": len(rgb_input),
            "overlap_tolerance_pct": float(overlap_tolerance_pct),
        }
    spectral = _metric_means(spectral_input)
    rgb = _metric_means(rgb_input)
    checks = {
        "all_trials_completed": (
            len(spectral_valid) == len(spectral_input)
            and len(rgb_valid) == len(rgb_input)
            and {
                int(row["trial"])
                for row in spectral_input
            }
            == {int(row["trial"]) for row in rgb_input}
        ),
        "rotation_lower": (
            spectral["rotation_error_deg"] < rgb["rotation_error_deg"]
        ),
        "translation_lower": (
            spectral["translation_error_m"] < rgb["translation_error_m"]
        ),
        "success_not_lower": (
            spectral["success_rate"] >= rgb["success_rate"]
        ),
        "overlap_100_within_tolerance": (
            spectral["overlap_100mm_pct"]
            >= rgb["overlap_100mm_pct"] - overlap_tolerance_pct
        ),
        "overlap_250_within_tolerance": (
            spectral["overlap_250mm_pct"]
            >= rgb["overlap_250mm_pct"] - overlap_tolerance_pct
        ),
    }
    return {
        "passed": bool(all(checks.values())),
        "reason": "" if all(checks.values()) else "joint_gate_failed",
        "checks": checks,
        "spectral": spectral,
        "rgb": rgb,
        "overlap_tolerance_pct": float(overlap_tolerance_pct),
    }


def chamfer_metrics_mm(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    *,
    trim_fraction: float = 0.70,
) -> dict[str, float]:
    source = np.asarray(source_xyz_m, dtype=np.float64)
    target = np.asarray(target_xyz_m, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or target.ndim != 2
        or target.shape[1] != 3
        or len(source) == 0
        or len(target) == 0
    ):
        raise ValueError("Chamfer输入必须是非空N×3点云")
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError("trim_fraction必须位于(0,1]")
    source_to_target = cKDTree(target).query(source, k=1, workers=1)[0]
    target_to_source = cKDTree(source).query(target, k=1, workers=1)[0]
    global_chamfer = 0.5 * (
        float(np.mean(source_to_target))
        + float(np.mean(target_to_source))
    )
    source_keep = max(1, int(trim_fraction * len(source_to_target)))
    target_keep = max(1, int(trim_fraction * len(target_to_source)))
    trimmed_chamfer = 0.5 * (
        float(np.mean(np.partition(source_to_target, source_keep - 1)[:source_keep]))
        + float(np.mean(np.partition(target_to_source, target_keep - 1)[:target_keep]))
    )
    bidirectional = np.concatenate((source_to_target, target_to_source))
    return {
        "global_chamfer_mm": 1000.0 * global_chamfer,
        "trimmed_chamfer_mm": 1000.0 * trimmed_chamfer,
        "distance_median_mm": 1000.0 * float(np.median(bidirectional)),
    }
