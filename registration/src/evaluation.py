"""三种配准方法共用的 Open3D 评价接口。"""

from __future__ import annotations

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .transform import apply_transform, transform_errors


def make_point_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(points, dtype=float)))
    if colors is not None:
        cloud.colors = o3d.utility.Vector3dVector(np.clip(np.asarray(colors, dtype=float), 0, 1))
    return cloud


def evaluate_registration(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    transformation: np.ndarray,
    max_correspondence_m: float,
) -> dict[str, float]:
    source = make_point_cloud(source_xyz_m)
    target = make_point_cloud(target_xyz_m)
    result = o3d.pipelines.registration.evaluate_registration(
        source,
        target,
        max_correspondence_m,
        np.asarray(transformation, dtype=np.float64),
    )
    return {
        "fitness": float(result.fitness),
        "inlier_rmse_m": float(result.inlier_rmse),
    }


def bidirectional_overlap_percentages(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    thresholds_m: tuple[float, ...] = (0.10, 0.25),
) -> dict[str, float]:
    """计算两个点云在各距离阈值下的双向最近邻重叠率。"""
    source = np.asarray(source_xyz_m, dtype=np.float64)
    target = np.asarray(target_xyz_m, dtype=np.float64)
    for name, points in (("source", source), ("target", target)):
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError(f"{name}_xyz_m 必须是非空的 (N,3) 数组")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"{name}_xyz_m 包含非有限值")

    thresholds = tuple(float(value) for value in thresholds_m)
    if not thresholds or any(
        not np.isfinite(value) or value <= 0.0 for value in thresholds
    ):
        raise ValueError("thresholds_m 必须包含正的有限距离阈值")

    source_to_target = cKDTree(target).query(source, k=1, workers=1)[0]
    target_to_source = cKDTree(source).query(target, k=1, workers=1)[0]
    result: dict[str, float] = {}
    for threshold in thresholds:
        threshold_mm = int(round(1000.0 * threshold))
        result[f"overlap_{threshold_mm}mm_pct"] = float(
            100.0
            * 0.5
            * (
                np.mean(source_to_target <= threshold)
                + np.mean(target_to_source <= threshold)
            )
        )
    return result


def full_metrics(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    max_correspondence_m: float,
    success_rotation_deg: float = 2.0,
    success_translation_m: float = 0.1,
) -> dict[str, float | bool]:
    metrics: dict[str, float | bool] = {}
    metrics.update(transform_errors(estimated, ground_truth))
    metrics.update(
        evaluate_registration(
            source_xyz_m,
            target_xyz_m,
            estimated,
            max_correspondence_m,
        )
    )
    metrics.update(
        bidirectional_overlap_percentages(
            apply_transform(source_xyz_m, estimated),
            target_xyz_m,
        )
    )
    metrics["success"] = bool(
        metrics["rotation_error_deg"] <= success_rotation_deg
        and metrics["translation_error_m"] <= success_translation_m
    )
    return metrics
