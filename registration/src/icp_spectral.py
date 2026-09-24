"""论文式几何 + 六波段切平面残差的多尺度 ICP。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .evaluation import evaluate_registration
from .icp_geometry import ICPResult, _validate_pyramid
from .registration_validation import (
    ValidationContext,
    ValidationScore,
    build_validation_context,
    decide_candidate,
)
from .transform import apply_transform


def _estimate_normals(points: np.ndarray, radius: float) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
    )
    normals = np.asarray(cloud.normals).copy()
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def _voxel_reduce(
    points: np.ndarray,
    attributes: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    reduced_points = np.column_stack(
        [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
    )
    reduced_attributes = np.column_stack(
        [
            np.bincount(inverse, weights=attributes[:, channel]) / counts
            for channel in range(attributes.shape[1])
        ]
    )
    return reduced_points, reduced_attributes


def fit_tangent_spectral_gradients(
    points: np.ndarray,
    normals: np.ndarray,
    spectral: np.ndarray,
    radius: float,
    max_nn: int,
) -> np.ndarray:
    """对每个目标点的6个波段拟合受切平面约束的三维梯度。"""
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    spectral = np.asarray(spectral, dtype=np.float64)
    tree = cKDTree(points)
    gradients = np.zeros((len(points), spectral.shape[1], 3), dtype=np.float64)
    for index, (point, normal) in enumerate(zip(points, normals, strict=True)):
        neighbors = tree.query_ball_point(point, radius)
        if len(neighbors) > max_nn:
            _, local = tree.query(point, k=max_nn)
            neighbors = np.atleast_1d(local).tolist()
        neighbors = [neighbor for neighbor in neighbors if neighbor != index]
        if len(neighbors) < 3:
            continue
        delta = points[neighbors] - point
        tangent = delta - np.outer(delta @ normal, normal)
        if np.linalg.matrix_rank(tangent, tol=1e-10) < 2:
            continue
        values = spectral[neighbors] - spectral[index]
        coefficient, *_ = np.linalg.lstsq(tangent, values, rcond=1e-8)
        gradients[index] = coefficient.T
    return gradients


def _minus_skew(points: np.ndarray) -> np.ndarray:
    result = np.zeros((len(points), 3, 3), dtype=np.float64)
    x, y, z = points.T
    result[:, 0, 1] = z
    result[:, 0, 2] = -y
    result[:, 1, 0] = -z
    result[:, 1, 2] = x
    result[:, 2, 0] = y
    result[:, 2, 1] = -x
    return result


def _se3_exp(twist: np.ndarray) -> np.ndarray:
    omega = twist[:3]
    velocity = twist[3:]
    theta = float(np.linalg.norm(omega))
    rotation = Rotation.from_rotvec(omega).as_matrix()
    omega_hat = -_minus_skew(omega.reshape(1, 3))[0]
    if theta < 1e-8:
        v_matrix = np.eye(3) + 0.5 * omega_hat + (omega_hat @ omega_hat) / 6.0
    else:
        v_matrix = (
            np.eye(3)
            + (1.0 - np.cos(theta)) / theta**2 * omega_hat
            + (theta - np.sin(theta)) / theta**3 * (omega_hat @ omega_hat)
        )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = v_matrix @ velocity
    return transform


def _robust_standardize(
    source_spectral: np.ndarray,
    target_spectral: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(target_spectral, axis=0)
    mad = 1.4826 * np.median(np.abs(target_spectral - median), axis=0)
    fallback = np.std(target_spectral, axis=0)
    scale = np.where(mad > 1e-6, mad, np.maximum(fallback, 1e-6))
    return (source_spectral - median) / scale, (target_spectral - median) / scale


@dataclass(frozen=True)
class TangentValidationContext:
    source_points: np.ndarray
    target_points: np.ndarray
    source_spectral: np.ndarray
    target_spectral: np.ndarray
    target_normals: np.ndarray
    target_gradients: np.ndarray
    target_tree: cKDTree
    max_correspondence_m: float
    trim_fraction: float
    min_correspondences: int
    min_relative_improvement: float
    max_coverage_drop: float
    overlap_weight: float
    attribute_clip_sigma: float
    lambda_geo: float
    lambda_spec: float


def _build_tangent_validation_context(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    source_spectral: np.ndarray,
    target_spectral: np.ndarray,
    lambda_geo: float,
    lambda_spec: float,
    config: Mapping[str, object],
) -> TangentValidationContext:
    voxel = float(config["voxel_size_m"])
    source_points, source_spec = _voxel_reduce(
        np.asarray(source_xyz_m, dtype=np.float64),
        source_spectral,
        voxel,
    )
    target_points, target_spec = _voxel_reduce(
        np.asarray(target_xyz_m, dtype=np.float64),
        target_spectral,
        voxel,
    )
    target_normals = _estimate_normals(target_points, 2.5 * voxel)
    target_gradients = fit_tangent_spectral_gradients(
        target_points,
        target_normals,
        target_spec,
        radius=2.5 * voxel,
        max_nn=50,
    )
    return TangentValidationContext(
        source_points=source_points,
        target_points=target_points,
        source_spectral=source_spec,
        target_spectral=target_spec,
        target_normals=target_normals,
        target_gradients=target_gradients,
        target_tree=cKDTree(target_points),
        max_correspondence_m=float(config["max_correspondence_m"]),
        trim_fraction=float(config["trim_fraction"]),
        min_correspondences=int(config["min_correspondences"]),
        min_relative_improvement=float(config["min_relative_improvement"]),
        max_coverage_drop=float(config["max_coverage_drop"]),
        overlap_weight=float(config["overlap_weight"]),
        attribute_clip_sigma=float(config["attribute_clip_sigma"]),
        lambda_geo=lambda_geo,
        lambda_spec=lambda_spec,
    )


def _score_tangent_validation(
    context: TangentValidationContext,
    transformation: np.ndarray,
) -> ValidationScore:
    transformed = apply_transform(context.source_points, transformation)
    distances, target_indices = context.target_tree.query(
        transformed,
        k=1,
        workers=1,
    )
    inliers = distances <= context.max_correspondence_m
    count = int(np.count_nonzero(inliers))
    coverage = float(np.mean(inliers))
    if count < context.min_correspondences:
        return ValidationScore(
            score=float("inf"),
            coverage=coverage,
            inlier_count=count,
            geometry_error=float("inf"),
            attribute_error=float("inf"),
            valid=False,
            invalid_reason="insufficient_correspondences",
        )
    q = transformed[inliers]
    selected_target = target_indices[inliers]
    p = context.target_points[selected_target]
    normals = context.target_normals[selected_target]
    delta_xyz = q - p
    normal_offset = np.einsum("ij,ij->i", delta_xyz, normals)
    tangent_offset = delta_xyz - normals * normal_offset[:, None]
    spectral_residual = (
        context.target_spectral[selected_target]
        + np.einsum(
            "nki,ni->nk",
            context.target_gradients[selected_target],
            tangent_offset,
        )
        - context.source_spectral[inliers]
    )
    geometry_per_pair = (
        normal_offset / context.max_correspondence_m
    ) ** 2
    clip_squared = context.attribute_clip_sigma**2
    spectral_per_pair = np.mean(
        np.minimum(spectral_residual**2, clip_squared) / clip_squared,
        axis=1,
    )
    pair_score = (
        context.lambda_geo * geometry_per_pair
        + context.lambda_spec * spectral_per_pair
    )
    keep_count = max(
        1,
        int(np.ceil(context.trim_fraction * count)),
    )
    keep = np.argsort(pair_score)[:keep_count]
    geometry_error = float(np.mean(geometry_per_pair[keep]))
    spectral_error = float(np.mean(spectral_per_pair[keep]))
    data_weight = 1.0 - context.overlap_weight
    score = (
        data_weight
        * (
            context.lambda_geo * geometry_error
            + context.lambda_spec * spectral_error
        )
        + context.overlap_weight * (1.0 - coverage)
    )
    return ValidationScore(
        score=float(score),
        coverage=coverage,
        inlier_count=count,
        geometry_error=geometry_error,
        attribute_error=spectral_error,
        valid=True,
        invalid_reason="",
    )


def _tangent_validation_decision(
    context: TangentValidationContext,
    before_transform: np.ndarray,
    after_transform: np.ndarray,
) -> tuple[bool, float, str, ValidationScore, ValidationScore]:
    before = _score_tangent_validation(context, before_transform)
    after = _score_tangent_validation(context, after_transform)
    if not after.valid:
        return False, float("-inf"), after.invalid_reason, before, after
    if not before.valid:
        return True, float("inf"), "", before, after
    if before.coverage - after.coverage > context.max_coverage_drop:
        return False, float("-inf"), "coverage_drop", before, after
    relative = (before.score - after.score) / max(
        abs(before.score),
        1e-12,
    )
    if relative < context.min_relative_improvement:
        return (
            False,
            float(relative),
            "insufficient_score_improvement",
            before,
            after,
        )
    return True, float(relative), "", before, after


def _fixed_pair_objective(
    transformation: np.ndarray,
    source_points: np.ndarray,
    source_spec: np.ndarray,
    source_indices: np.ndarray,
    target_points: np.ndarray,
    target_spec: np.ndarray,
    target_normals: np.ndarray,
    target_gradients: np.ndarray,
    target_indices: np.ndarray,
    distance_limit: float,
    lambda_geo: float,
    lambda_spec: float,
) -> tuple[float, np.ndarray]:
    q = apply_transform(source_points[source_indices], transformation)
    p = target_points[target_indices]
    normals = target_normals[target_indices]
    delta = q - p
    normal_offset = np.einsum("mi,mi->m", delta, normals)
    tangent_offset = delta - normals * normal_offset[:, None]
    spectral_residual = (
        target_spec[target_indices]
        + np.einsum("mki,mi->mk", target_gradients[target_indices], tangent_offset)
        - source_spec[source_indices]
    )
    residual = np.concatenate(
        (
            np.sqrt(lambda_geo) / distance_limit * normal_offset,
            (np.sqrt(lambda_spec / spectral_residual.shape[1]) * spectral_residual).ravel(),
        )
    )
    return float(np.mean(residual**2)), np.linalg.norm(delta, axis=1)


def register_spectral_icp(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    voxel_sizes: list[float],
    max_correspondence_distances: list[float],
    max_iterations: list[int],
    lambda_geo: float = 0.5,
    lambda_spec: float = 0.5,
    init: np.ndarray | None = None,
    validation_config: Mapping[str, object] | None = None,
    validation_mode: str = "direct",
) -> ICPResult:
    """用6通道切平面光谱残差和point-to-plane残差联合估计SE(3)。"""
    _validate_pyramid(voxel_sizes, max_correspondence_distances, max_iterations)
    if not np.isclose(lambda_geo + lambda_spec, 1.0):
        raise ValueError("lambda_geo 与 lambda_spec 之和必须为 1")
    if validation_mode not in {"direct", "tangent"}:
        raise ValueError("validation_mode必须是direct或tangent")
    source_spectral, target_spectral = _robust_standardize(
        np.asarray(source_spectral6, dtype=float),
        np.asarray(target_spectral6, dtype=float),
    )
    transformation = np.eye(4) if init is None else np.asarray(init, dtype=float).copy()
    history: list[dict[str, object]] = []
    validation_context: ValidationContext | None = None
    tangent_validation_context: TangentValidationContext | None = None
    if validation_config is not None and validation_mode == "direct":
        validation_context = build_validation_context(
            source_xyz_m,
            target_xyz_m,
            source_spectral6,
            target_spectral6,
            geometry_fraction=lambda_geo,
            attribute_fraction=lambda_spec,
            config=validation_config,
        )
    elif validation_config is not None:
        tangent_validation_context = _build_tangent_validation_context(
            source_xyz_m,
            target_xyz_m,
            source_spectral,
            target_spectral,
            lambda_geo,
            lambda_spec,
            validation_config,
        )
    final_fitness = 0.0
    final_rmse = float("inf")

    for level, (voxel, distance_limit, iteration_limit) in enumerate(
        zip(voxel_sizes, max_correspondence_distances, max_iterations, strict=True), 1
    ):
        level_start = transformation.copy()
        source_points, source_spec = _voxel_reduce(
            np.asarray(source_xyz_m, dtype=float), source_spectral, voxel
        )
        target_points, target_spec = _voxel_reduce(
            np.asarray(target_xyz_m, dtype=float), target_spectral, voxel
        )
        target_normals = _estimate_normals(target_points, 2.5 * voxel)
        target_gradients = fit_tangent_spectral_gradients(
            target_points,
            target_normals,
            target_spec,
            radius=2.5 * voxel,
            max_nn=50,
        )
        target_tree = cKDTree(target_points)

        for iteration in range(1, iteration_limit + 1):
            transformed = apply_transform(source_points, transformation)
            distances, target_indices = target_tree.query(transformed, k=1)
            inliers = distances <= distance_limit
            if np.count_nonzero(inliers) < 6:
                break
            q = transformed[inliers]
            p = target_points[target_indices[inliers]]
            normals = target_normals[target_indices[inliers]]
            gradients = target_gradients[target_indices[inliers]]
            delta = q - p
            normal_offset = np.einsum("mi,mi->m", delta, normals)
            tangent_offset = delta - normals * normal_offset[:, None]

            spectral_residual = (
                target_spec[target_indices[inliers]]
                + np.einsum("mki,mi->mk", gradients, tangent_offset)
                - source_spec[inliers]
            )
            motion_jacobian = np.concatenate(
                (_minus_skew(q), np.broadcast_to(np.eye(3), (len(q), 3, 3))),
                axis=2,
            )
            projector = (
                np.broadcast_to(np.eye(3), (len(q), 3, 3))
                - normals[:, :, None] * normals[:, None, :]
            )
            tangent_motion = np.einsum("mij,mjk->mik", projector, motion_jacobian)
            spectral_jacobian = np.einsum(
                "mki,mij->mkj", gradients, tangent_motion
            )
            geometry_jacobian = np.einsum(
                "mi,mij->mj", normals, motion_jacobian
            )

            geo_weight = np.sqrt(lambda_geo) / distance_limit
            spec_weight = np.sqrt(lambda_spec / spectral_residual.shape[1])
            residual = np.concatenate(
                (geo_weight * normal_offset, (spec_weight * spectral_residual).ravel())
            )
            jacobian = np.vstack(
                (geo_weight * geometry_jacobian, (spec_weight * spectral_jacobian).reshape(-1, 6))
            )
            normal_matrix = jacobian.T @ jacobian + 1e-8 * np.eye(6)
            rhs = -(jacobian.T @ residual)
            twist = np.linalg.solve(normal_matrix, rhs)

            rotation_norm = np.linalg.norm(twist[:3])
            if rotation_norm > 0.2:
                twist[:3] *= 0.2 / rotation_norm
            translation_norm = np.linalg.norm(twist[3:])
            if translation_norm > 0.5 * distance_limit:
                twist[3:] *= 0.5 * distance_limit / translation_norm

            objective = float(np.mean(residual**2))
            source_indices = np.flatnonzero(inliers)
            pair_target_indices = target_indices[inliers]
            accepted_scale = 0.0
            accepted_objective = objective
            accepted_transform = transformation
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
                candidate = _se3_exp(scale * twist) @ transformation
                candidate_objective, candidate_distances = _fixed_pair_objective(
                    candidate,
                    source_points,
                    source_spec,
                    source_indices,
                    target_points,
                    target_spec,
                    target_normals,
                    target_gradients,
                    pair_target_indices,
                    distance_limit,
                    lambda_geo,
                    lambda_spec,
                )
                retained = float(np.mean(candidate_distances <= distance_limit))
                if (
                    np.isfinite(candidate_objective)
                    and candidate_objective <= objective
                    and retained >= 0.90
                ):
                    accepted_scale = scale
                    accepted_objective = candidate_objective
                    accepted_transform = candidate
                    break
            transformation = accepted_transform
            accepted_twist_norm = float(accepted_scale * np.linalg.norm(twist))

            updated = apply_transform(source_points, transformation)
            updated_distances = target_tree.query(updated, k=1)[0]
            updated_inliers = updated_distances <= distance_limit
            final_fitness = float(np.mean(updated_inliers))
            final_rmse = (
                float(np.sqrt(np.mean(updated_distances[updated_inliers] ** 2)))
                if np.any(updated_inliers)
                else float("inf")
            )
            history.append(
                {
                    "record_type": "iteration",
                    "level": level,
                    "iteration": iteration,
                    "voxel_size_m": voxel,
                    "max_correspondence_m": distance_limit,
                    "inlier_count": int(np.count_nonzero(inliers)),
                    "fitness": final_fitness,
                    "inlier_rmse_m": final_rmse,
                    "objective": objective,
                    "objective_after": accepted_objective,
                    "step_scale": accepted_scale,
                    "twist_norm": accepted_twist_norm,
                }
            )
            if (
                accepted_scale == 0.0
                or accepted_twist_norm < 1e-7
                or abs(objective - accepted_objective) < 1e-10
            ):
                break

        if validation_context is not None:
            candidate = transformation.copy()
            decision = decide_candidate(
                validation_context,
                level_start,
                candidate,
            )
            transformation = candidate if decision.accepted else level_start
            history.append(
                {
                    "record_type": "level_validation",
                    "level": level,
                    "voxel_size_m": voxel,
                    "max_correspondence_m": distance_limit,
                    "max_iterations": iteration_limit,
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
        elif tangent_validation_context is not None:
            candidate = transformation.copy()
            (
                accepted,
                relative,
                reason,
                before,
                after,
            ) = _tangent_validation_decision(
                tangent_validation_context,
                level_start,
                candidate,
            )
            transformation = candidate if accepted else level_start
            history.append(
                {
                    "record_type": "level_validation",
                    "level": level,
                    "voxel_size_m": voxel,
                    "max_correspondence_m": distance_limit,
                    "max_iterations": iteration_limit,
                    "validation_score_before": before.score,
                    "validation_score_after": after.score,
                    "validation_coverage_before": before.coverage,
                    "validation_coverage_after": after.coverage,
                    "validation_spectral_before": before.attribute_error,
                    "validation_spectral_after": after.attribute_error,
                    "validation_relative_improvement": relative,
                    "accepted": accepted,
                    "rollback_reason": reason,
                }
            )

    if validation_context is not None or tangent_validation_context is not None:
        metrics = evaluate_registration(
            source_xyz_m,
            target_xyz_m,
            transformation,
            max_correspondence_distances[-1],
        )
        final_fitness = float(metrics["fitness"])
        final_rmse = float(metrics["inlier_rmse_m"])
    return ICPResult(transformation, final_fitness, final_rmse, history)
