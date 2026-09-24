"""Open3D 多尺度 RGB Colored ICP。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import open3d as o3d

from .evaluation import evaluate_registration
from .icp_geometry import (
    ICPResult,
    _validate_pyramid,
    downsample_and_estimate_normals,
)
from .registration_validation import (
    ValidationContext,
    build_validation_context,
    decide_candidate,
)


def register_rgb_icp(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    voxel_sizes: list[float],
    max_correspondence_distances: list[float],
    max_iterations: list[int],
    lambda_geometric: float = 0.968,
    init: np.ndarray | None = None,
    validation_config: Mapping[str, object] | None = None,
) -> ICPResult:
    """从粗到细调用 Open3D Colored ICP。"""
    _validate_pyramid(voxel_sizes, max_correspondence_distances, max_iterations)
    transformation = np.eye(4) if init is None else np.asarray(init, dtype=float).copy()
    history: list[dict[str, object]] = []
    validation_context: ValidationContext | None = None
    if validation_config is not None:
        validation_context = build_validation_context(
            source_xyz_m,
            target_xyz_m,
            source_rgb,
            target_rgb,
            geometry_fraction=lambda_geometric,
            attribute_fraction=1.0 - lambda_geometric,
            config=validation_config,
        )
    last_result = None
    estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP(
        lambda_geometric
    )
    for level, (voxel, distance, iterations) in enumerate(
        zip(voxel_sizes, max_correspondence_distances, max_iterations, strict=True), 1
    ):
        source = downsample_and_estimate_normals(source_xyz_m, voxel, source_rgb)
        target = downsample_and_estimate_normals(target_xyz_m, voxel, target_rgb)
        level_start = transformation.copy()
        last_result = o3d.pipelines.registration.registration_colored_icp(
            source,
            target,
            distance,
            transformation,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=iterations,
            ),
        )
        candidate = np.asarray(last_result.transformation).copy()
        row: dict[str, object] = {
            "record_type": "level_validation",
            "level": level,
            "voxel_size_m": voxel,
            "max_correspondence_m": distance,
            "max_iterations": iterations,
            "fitness": float(last_result.fitness),
            "inlier_rmse_m": float(last_result.inlier_rmse),
        }
        if validation_context is None:
            transformation = candidate
        else:
            decision = decide_candidate(
                validation_context,
                level_start,
                candidate,
            )
            transformation = candidate if decision.accepted else level_start
            row.update(
                {
                    "validation_score_before": decision.before.score,
                    "validation_score_after": decision.after.score,
                    "validation_coverage_before": decision.before.coverage,
                    "validation_coverage_after": decision.after.coverage,
                    "validation_relative_improvement": (
                        decision.relative_improvement
                    ),
                    "accepted": decision.accepted,
                    "rollback_reason": decision.rollback_reason,
                }
            )
        history.append(row)
    assert last_result is not None
    if validation_context is not None:
        metrics = evaluate_registration(
            source_xyz_m,
            target_xyz_m,
            transformation,
            max_correspondence_distances[-1],
        )
        fitness = float(metrics["fitness"])
        inlier_rmse = float(metrics["inlier_rmse_m"])
    else:
        fitness = float(last_result.fitness)
        inlier_rmse = float(last_result.inlier_rmse)
    return ICPResult(
        transformation=transformation,
        fitness=fitness,
        inlier_rmse=inlier_rmse,
        history=history,
    )
