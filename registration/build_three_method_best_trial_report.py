from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import open3d as o3d

from src.data_prepare import load_prepared
from src.spectral_experiment import chamfer_metrics_mm
from src.three_method_comparison import (
    CANDIDATE_TRIALS,
    METHODS,
    MethodTrialMetrics,
    load_metric,
    metric_payload,
    select_shared_best_trial,
    verify_shared_transforms,
)


METRICS_SUMMARY_FIELDS = (
    "method",
    "selected_trial",
    "rotation_error_deg",
    "translation_error_m",
    "overlap_100mm_pct",
    "overlap_250mm_pct",
    "global_chamfer_mm",
    "trimmed_chamfer_mm",
    "distance_median_mm",
    "fitness",
    "inlier_rmse_m",
    "pose_score",
)
_PLY_COPIES = (
    ("source_target_initial_merged.ply", "initial_merged.ply"),
    ("source_target_merged.ply", "registered_merged.ply"),
    ("source_target_rgb_merged.ply", "registered_true_rgb.ply"),
)
_CHAMFER_FIELDS = (
    "global_chamfer_mm",
    "trimmed_chamfer_mm",
    "distance_median_mm",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="导出三种方法共享最佳 trial 的自包含点云与指标报告。"
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="鲁棒六通道实验根目录（默认：本脚本所在目录）。",
    )
    parser.add_argument(
        "--geometry-experiment-root",
        type=Path,
        help="geometry/RGB 专属粗配准实验根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="导出目录（默认：<experiment-root>/results/three_method_best_trial_comparison）。",
    )
    return parser.parse_args(argv)


def _method_sources(experiment_root: Path, geometry_root: Path) -> dict[str, Path]:
    return {
        "geometry": geometry_root / "results" / "icp_geometry",
        "rgb": geometry_root / "results" / "icp_rgb",
        "spectral6_robust": (
            experiment_root
            / "results/spectral6_recommended_experiment/round1/robust_full"
        ),
    }


def _trial_dir(method_root: Path, trial: int) -> Path:
    return method_root / f"trial_{trial:03d}"


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_summary(path: Path, rows: list[dict[str, object]]) -> None:
    header = "| " + " | ".join(METRICS_SUMMARY_FIELDS) + " |"
    separator = "| " + " | ".join("---" for _ in METRICS_SUMMARY_FIELDS) + " |"
    body = [
        "| " + " | ".join(str(row[field]) for field in METRICS_SUMMARY_FIELDS) + " |"
        for row in rows
    ]
    path.write_text("\n".join((header, separator, *body, "")), encoding="utf-8")


def _validate_cloud(path: Path) -> None:
    cloud = o3d.io.read_point_cloud(str(path))
    if len(cloud.points) == 0:
        raise ValueError(f"点云为空或无法读取: {path}")


def _missing_chamfer_fields(metric_path: Path) -> tuple[str, ...]:
    try:
        raw_metric = json.loads(metric_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load metric file: {metric_path}") from error
    if not isinstance(raw_metric, dict):
        raise ValueError(f"Metric file must contain an object: {metric_path}")
    return tuple(field for field in _CHAMFER_FIELDS if field not in raw_metric)


def _load_target_xyz(geometry_root: Path) -> np.ndarray:
    target_path = geometry_root / "data" / "processed" / "right_points.npz"
    if not target_path.is_file():
        raise FileNotFoundError(f"Target NPZ is missing: {target_path}")
    try:
        target_xyz = np.asarray(load_prepared(target_path)["xyz_m"], dtype=np.float64)
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"Cannot load target xyz_m from: {target_path}") from error
    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3 or len(target_xyz) == 0:
        raise ValueError(f"Target xyz_m is empty or invalid: {target_path}")
    if not np.isfinite(target_xyz).all():
        raise ValueError(f"Target NPZ {target_path} contains NaN或Inf")
    return target_xyz


def _chamfer_fallback_values(trial_dir: Path, target_xyz_m: np.ndarray) -> dict[str, float]:
    source_path = trial_dir / "source_registered.ply"
    source_cloud = o3d.io.read_point_cloud(str(source_path))
    source_xyz = np.asarray(source_cloud.points, dtype=np.float64)
    if source_xyz.ndim != 2 or source_xyz.shape[1] != 3 or len(source_xyz) == 0:
        raise ValueError(f"Registered PLY is empty or unreadable: {source_path}")
    return chamfer_metrics_mm(source_xyz, target_xyz_m, trim_fraction=0.70)


def _copy_method_output(
    source_dir: Path, output_dir: Path, metric: MethodTrialMetrics
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_name, output_name in _PLY_COPIES:
        destination = output_dir / output_name
        shutil.copy2(source_dir / source_name, destination)
        _validate_cloud(destination)
    for name in ("T_perturb.json", "T_gt.json", "T_est.json"):
        shutil.copy2(source_dir / name, output_dir / name)
    (output_dir / "metrics.json").write_text(
        json.dumps(metric_payload(metric), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _summary_row(metric: MethodTrialMetrics, selected_trial: int) -> dict[str, object]:
    payload = metric_payload(metric)
    return {
        "method": payload["method"],
        "selected_trial": selected_trial,
        **{field: payload[field] for field in METRICS_SUMMARY_FIELDS[2:-1]},
        "pose_score": metric.pose_score,
    }


def _write_selection_explanation(
    path: Path,
    selected_trial: int,
    sources: dict[str, Path],
) -> None:
    source_lines = "\n".join(f"- {method}: `{sources[method]}`" for method in METHODS)
    path.write_text(
        "# 统一最佳 Trial 选择\n\n"
        "候选范围为 trial 6–10。对每种方法计算：\n\n"
        "```text\n"
        "pose_score = rotation_error_deg / 2.0 + translation_error_m / 0.10\n"
        "```\n\n"
        "每个 trial 取三种方法的 pose_score 平均值，取最小者。若相同，依次比较三方法"
        "平均 100 mm 重叠率（高者优先）、平均 250 mm 重叠率（高者优先）和更小的"
        " trial 编号。\n\n"
        f"选择的共享 trial：**{selected_trial}**。\n\n"
        "三种方法的来源目录：\n"
        f"{source_lines}\n\n"
        "阈值判定类字段已按要求省略，未参与本统计或导出指标。\n",
        encoding="utf-8",
    )


def build_report(experiment_root: Path, geometry_root: Path, output_dir: Path) -> int:
    """Export a self-contained comparison report for the shared best trial."""
    sources = _method_sources(experiment_root, geometry_root)
    target_xyz_m: np.ndarray | None = None
    metrics: list[MethodTrialMetrics] = []
    for trial in CANDIDATE_TRIALS:
        for method in METHODS:
            trial_dir = _trial_dir(sources[method], trial)
            metric_path = trial_dir / "metrics.json"
            missing_chamfer = _missing_chamfer_fields(metric_path)
            fallback_values: dict[str, float] | None = None
            if missing_chamfer:
                if target_xyz_m is None:
                    target_xyz_m = _load_target_xyz(geometry_root)
                calculated = _chamfer_fallback_values(trial_dir, target_xyz_m)
                fallback_values = {
                    field: calculated[field] for field in missing_chamfer
                }
            metrics.append(
                load_metric(metric_path, method, trial, fallback_values)
            )
    selected_trial, scores = select_shared_best_trial(metrics)
    selected_dirs = {
        method: _trial_dir(sources[method], selected_trial) for method in METHODS
    }
    verify_shared_transforms(selected_dirs)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_method = {
        metric.method: metric
        for metric in metrics
        if metric.trial == selected_trial
    }
    for method in METHODS:
        _copy_method_output(selected_dirs[method], output_dir / method, metrics_by_method[method])

    score_rows = [
        {**score, "selected": score["trial"] == selected_trial}
        for score in sorted(scores, key=lambda score: int(score["trial"]))
    ]
    _write_csv(
        output_dir / "trial_selection_scores.csv",
        (
            "trial",
            "mean_pose_score",
            "mean_overlap_100mm_pct",
            "mean_overlap_250mm_pct",
            "selected",
        ),
        score_rows,
    )
    summary_rows = [_summary_row(metrics_by_method[method], selected_trial) for method in METHODS]
    _write_csv(output_dir / "metrics_summary.csv", METRICS_SUMMARY_FIELDS, summary_rows)
    _write_markdown_summary(output_dir / "metrics_summary.md", summary_rows)
    _write_selection_explanation(
        output_dir / "selection_explanation.md", selected_trial, sources
    )
    return selected_trial


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    experiment_root = args.experiment_root.resolve()
    geometry_root = (
        args.geometry_experiment_root.resolve()
        if args.geometry_experiment_root is not None
        else (
            experiment_root.parent.parent
            / "method-specific-coarse-registration/2026_07_26实验"
        ).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (experiment_root / "results/three_method_best_trial_comparison").resolve()
    )
    if not geometry_root.is_dir():
        raise FileNotFoundError(
            f"Geometry experiment root does not exist: {geometry_root}. "
            "Provide --geometry-experiment-root to override it."
        )
    selected_trial = build_report(experiment_root, geometry_root, output_dir)
    print(f"selected_trial={selected_trial}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    main()
