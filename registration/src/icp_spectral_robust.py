"""带鲁棒梯度、双向对应和一致验证的六波段ICP组件。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .evaluation import evaluate_registration
from .icp_geometry import ICPResult, _validate_pyramid
from .icp_spectral import (
    _estimate_normals,
    _minus_skew,
    _robust_standardize,
    _se3_exp,
    _voxel_reduce,
)
from .transform import apply_transform


@dataclass(frozen=True)
class MutualCorrespondences:
    source_indices: np.ndarray
    target_indices: np.ndarray
    distances: np.ndarray


@dataclass(frozen=True)
class RobustSpectralScore:
    score: float
    coverage: float
    inlier_count: int
    geometry_error: float
    spectral_error: float
    valid: bool
    invalid_reason: str


def _validate_band_weights(band_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(band_weights, dtype=np.float64)
    if (
        weights.shape != (6,)
        or not np.isfinite(weights).all()
        or np.any(weights < 0.0)
        or not np.isclose(weights.sum(), 1.0)
    ):
        raise ValueError("band_weights必须是非负且和为1的六元素数组")
    return weights


def _huber_loss(values: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(
        absolute <= delta,
        0.5 * values**2,
        delta * (absolute - 0.5 * delta),
    )


def fit_robust_tangent_spectral_gradients(
    points: np.ndarray,
    normals: np.ndarray,
    spectral: np.ndarray,
    *,
    radius: float,
    max_nn: int,
    min_neighbors: int,
    max_condition: float,
    huber_delta: float,
    irls_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """用Huber-IRLS拟合六波段切平面梯度并返回逐点逐波段置信度。"""
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    spectral = np.asarray(spectral, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points必须是N×3")
    if normals.shape != points.shape:
        raise ValueError("normals必须与points同形")
    if spectral.shape != (len(points), 6):
        raise ValueError("spectral必须是N×6")
    if not (
        np.isfinite(points).all()
        and np.isfinite(normals).all()
        and np.isfinite(spectral).all()
    ):
        raise ValueError("梯度输入包含NaN或Inf")

    gradients = np.zeros((len(points), 6, 3), dtype=np.float64)
    confidence = np.zeros((len(points), 6), dtype=np.float64)
    tree = cKDTree(points)
    for index, (point, normal) in enumerate(zip(points, normals, strict=True)):
        neighbors = tree.query_ball_point(point, radius)
        neighbors = [item for item in neighbors if item != index]
        if len(neighbors) > max_nn:
            _, nearest = tree.query(point, k=max_nn + 1)
            neighbors = [
                int(item)
                for item in np.atleast_1d(nearest)
                if int(item) != index
            ][:max_nn]
        if len(neighbors) < min_neighbors:
            continue
        delta_xyz = points[neighbors] - point
        tangent = delta_xyz - np.outer(delta_xyz @ normal, normal)
        singular = np.linalg.svd(tangent, compute_uv=False)
        if len(singular) < 2 or singular[1] <= 1e-10:
            continue
        condition = float(singular[0] / singular[1])
        if not np.isfinite(condition) or condition > max_condition:
            continue

        values = spectral[neighbors] - spectral[index]
        support_confidence = min(1.0, len(neighbors) / 20.0)
        condition_confidence = min(1.0, 25.0 / max(condition, 1.0))
        for band in range(6):
            coefficients, *_ = np.linalg.lstsq(
                tangent,
                values[:, band],
                rcond=1e-8,
            )
            for _ in range(irls_iterations):
                residual = tangent @ coefficients - values[:, band]
                center = np.median(residual)
                scale = max(
                    1.4826 * np.median(np.abs(residual - center)),
                    1e-6,
                )
                threshold = huber_delta * scale
                weights = np.minimum(
                    1.0,
                    threshold / np.maximum(np.abs(residual), 1e-12),
                )
                root = np.sqrt(weights)
                coefficients, *_ = np.linalg.lstsq(
                    tangent * root[:, None],
                    values[:, band] * root,
                    rcond=1e-8,
                )
            residual = tangent @ coefficients - values[:, band]
            robust_rmse = float(
                np.sqrt(np.median(np.minimum(residual**2, (huber_delta * 0.5) ** 2)))
            )
            gradients[index, band] = coefficients
            confidence[index, band] = (
                support_confidence
                * condition_confidence
                / (1.0 + robust_rmse / 0.5)
            )
    return gradients, confidence


def build_mutual_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_normals: np.ndarray,
    target_normals: np.ndarray,
    transformation: np.ndarray,
    *,
    distance_limit: float,
    normal_angle_deg: float,
) -> MutualCorrespondences:
    """建立带距离和法线约束的双向最近邻对应。"""
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    source_n = np.asarray(source_normals, dtype=np.float64)
    target_n = np.asarray(target_normals, dtype=np.float64)
    transform = np.asarray(transformation, dtype=np.float64)
    if source_n.shape != source.shape or target_n.shape != target.shape:
        raise ValueError("法线必须与对应点云同形")
    moved = apply_transform(source, transform)
    moved_normals = source_n @ transform[:3, :3].T
    distances, target_indices = cKDTree(target).query(moved, k=1, workers=1)
    reverse_indices = cKDTree(moved).query(target, k=1, workers=1)[1]
    source_indices = np.arange(len(source), dtype=np.int64)
    mutual = reverse_indices[target_indices] == source_indices
    cosine = np.abs(
        np.einsum("ij,ij->i", moved_normals, target_n[target_indices])
    )
    valid = (
        mutual
        & (distances <= distance_limit)
        & (cosine >= np.cos(np.radians(normal_angle_deg)))
    )
    selected = np.flatnonzero(valid)
    return MutualCorrespondences(
        source_indices=selected,
        target_indices=target_indices[selected].astype(np.int64),
        distances=distances[selected],
    )


def _invalid_score(reason: str, coverage: float, count: int) -> RobustSpectralScore:
    return RobustSpectralScore(
        score=float("inf"),
        coverage=coverage,
        inlier_count=count,
        geometry_error=float("inf"),
        spectral_error=float("inf"),
        valid=False,
        invalid_reason=reason,
    )


def score_robust_spectral_transform(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    transformation: np.ndarray,
    *,
    band_weights: np.ndarray,
    lambda_spec: float,
    config: Mapping[str, object],
) -> RobustSpectralScore:
    """用与鲁棒ICP相同的切平面模型评价一个固定变换。"""
    weights = _validate_band_weights(band_weights)
    if not 0.0 <= lambda_spec < 1.0:
        raise ValueError("lambda_spec必须位于[0,1)")
    source_standard, target_standard = _robust_standardize(
        np.asarray(source_spectral6, dtype=np.float64),
        np.asarray(target_spectral6, dtype=np.float64),
    )
    voxel = float(config["voxel_size_m"])
    source_points, source_spec = _voxel_reduce(
        np.asarray(source_xyz_m, dtype=np.float64),
        source_standard,
        voxel,
    )
    target_points, target_spec = _voxel_reduce(
        np.asarray(target_xyz_m, dtype=np.float64),
        target_standard,
        voxel,
    )
    radius = float(config["gradient_radius_multiplier"]) * voxel
    source_normals = _estimate_normals(source_points, radius)
    target_normals = _estimate_normals(target_points, radius)
    gradients, gradient_confidence = fit_robust_tangent_spectral_gradients(
        target_points,
        target_normals,
        target_spec,
        radius=radius,
        max_nn=int(config["gradient_max_nn"]),
        min_neighbors=int(config["min_gradient_neighbors"]),
        max_condition=float(config["max_gradient_condition"]),
        huber_delta=float(config["huber_delta_spec"]),
        irls_iterations=int(config["gradient_irls_iterations"]),
    )
    pairs = build_mutual_correspondences(
        source_points,
        target_points,
        source_normals,
        target_normals,
        transformation,
        distance_limit=float(config["max_correspondence_m"]),
        normal_angle_deg=float(config["normal_angle_deg"]),
    )
    coverage = float(len(pairs.source_indices) / max(len(source_points), 1))
    if len(pairs.source_indices) < int(config["min_correspondences"]):
        return _invalid_score(
            "insufficient_correspondences",
            coverage,
            len(pairs.source_indices),
        )

    q = apply_transform(source_points[pairs.source_indices], transformation)
    p = target_points[pairs.target_indices]
    normals = target_normals[pairs.target_indices]
    delta_xyz = q - p
    normal_offset = np.einsum("ij,ij->i", delta_xyz, normals)
    tangent_offset = delta_xyz - normals * normal_offset[:, None]
    spectral_residual = (
        target_spec[pairs.target_indices]
        + np.einsum(
            "nki,ni->nk",
            gradients[pairs.target_indices],
            tangent_offset,
        )
        - source_spec[pairs.source_indices]
    )
    geometry_per_pair = _huber_loss(
        normal_offset / float(config["max_correspondence_m"]),
        float(config["huber_delta_geo"]),
    )
    confidence = gradient_confidence[pairs.target_indices]
    effective = confidence * weights[None, :]
    normalizer = np.maximum(effective.sum(axis=1), 1e-12)
    spectral_per_pair = (
        _huber_loss(spectral_residual, float(config["huber_delta_spec"]))
        * effective
    ).sum(axis=1) / normalizer

    overlap_weight = float(config["overlap_weight"])
    data_weight = 1.0 - overlap_weight
    combined = data_weight * (
        (1.0 - lambda_spec) * geometry_per_pair
        + lambda_spec * spectral_per_pair
    )
    keep_count = max(
        1,
        int(np.ceil(float(config["trim_fraction"]) * len(combined))),
    )
    keep = np.argsort(combined)[:keep_count]
    geometry_error = float(np.mean(geometry_per_pair[keep]))
    spectral_error = float(np.mean(spectral_per_pair[keep]))
    score = float(
        data_weight
        * (
            (1.0 - lambda_spec) * geometry_error
            + lambda_spec * spectral_error
        )
        + overlap_weight * (1.0 - coverage)
    )
    return RobustSpectralScore(
        score=score,
        coverage=coverage,
        inlier_count=len(pairs.source_indices),
        geometry_error=geometry_error,
        spectral_error=spectral_error,
        valid=True,
        invalid_reason="",
    )


def _huber_irls_weights(values: np.ndarray, delta: float) -> np.ndarray:
    return np.minimum(
        1.0,
        delta / np.maximum(np.abs(values), 1e-12),
    )


def _fixed_pair_robust_objective(
    transformation: np.ndarray,
    source_points: np.ndarray,
    source_spec: np.ndarray,
    source_indices: np.ndarray,
    target_points: np.ndarray,
    target_spec: np.ndarray,
    target_indices: np.ndarray,
    target_normals: np.ndarray,
    target_gradients: np.ndarray,
    target_confidence: np.ndarray,
    *,
    distance_limit: float,
    band_weights: np.ndarray,
    lambda_spec: float,
    huber_delta_geo: float,
    huber_delta_spec: float,
    trim_fraction: float,
) -> tuple[float, np.ndarray]:
    q = apply_transform(source_points[source_indices], transformation)
    p = target_points[target_indices]
    normals = target_normals[target_indices]
    delta_xyz = q - p
    normal_offset = np.einsum("ij,ij->i", delta_xyz, normals)
    tangent_offset = delta_xyz - normals * normal_offset[:, None]
    spectral_residual = (
        target_spec[target_indices]
        + np.einsum(
            "nki,ni->nk",
            target_gradients[target_indices],
            tangent_offset,
        )
        - source_spec[source_indices]
    )
    geometry_loss = _huber_loss(
        normal_offset / distance_limit,
        huber_delta_geo,
    )
    effective = target_confidence[target_indices] * band_weights[None, :]
    effective /= np.maximum(effective.sum(axis=1, keepdims=True), 1e-12)
    spectral_loss = (
        _huber_loss(spectral_residual, huber_delta_spec) * effective
    ).sum(axis=1)
    combined = (
        (1.0 - lambda_spec) * geometry_loss
        + lambda_spec * spectral_loss
    )
    keep_count = max(1, int(np.ceil(trim_fraction * len(combined))))
    keep = np.argsort(combined)[:keep_count]
    return (
        float(np.mean(combined[keep])),
        np.linalg.norm(delta_xyz, axis=1),
    )


def _validation_decision(
    before: RobustSpectralScore,
    after: RobustSpectralScore,
    *,
    min_relative_improvement: float,
    max_coverage_drop: float,
) -> tuple[bool, float, str]:
    if not after.valid:
        return False, float("-inf"), after.invalid_reason
    if not before.valid:
        return True, float("inf"), ""
    coverage_drop = before.coverage - after.coverage
    if coverage_drop > max_coverage_drop:
        return False, float("-inf"), "coverage_drop"
    relative = (before.score - after.score) / max(abs(before.score), 1e-12)
    if relative < min_relative_improvement:
        return False, float(relative), "insufficient_score_improvement"
    return True, float(relative), ""


def register_robust_spectral_icp(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    voxel_sizes: list[float],
    max_correspondence_distances: list[float],
    max_iterations: list[int],
    *,
    band_weights: np.ndarray,
    lambda_spec_levels: tuple[float, ...],
    init: np.ndarray | None = None,
    validation_config: Mapping[str, object] | None = None,
    huber_delta_geo: float = 1.5,
    huber_delta_spec: float = 1.5,
    min_gradient_neighbors: int = 8,
    max_gradient_condition: float = 100.0,
    normal_angle_deg: float = 45.0,
    min_correspondences: int = 30,
    trim_fraction: float = 0.70,
    gradient_max_nn: int = 50,
    gradient_irls_iterations: int = 5,
) -> ICPResult:
    """以鲁棒六波段切平面残差和point-to-plane残差联合估计SE(3)。"""
    _validate_pyramid(
        voxel_sizes,
        max_correspondence_distances,
        max_iterations,
    )
    if len(lambda_spec_levels) != len(voxel_sizes):
        raise ValueError("lambda_spec_levels必须与金字塔层数一致")
    if any(not 0.0 <= value < 1.0 for value in lambda_spec_levels):
        raise ValueError("每层lambda_spec必须位于[0,1)")
    weights = _validate_band_weights(band_weights)
    source_standard, target_standard = _robust_standardize(
        np.asarray(source_spectral6, dtype=np.float64),
        np.asarray(target_spectral6, dtype=np.float64),
    )
    transformation = (
        np.eye(4, dtype=np.float64)
        if init is None
        else np.asarray(init, dtype=np.float64).copy()
    )
    history: list[dict[str, object]] = []

    for level, (
        voxel,
        distance_limit,
        iteration_limit,
        lambda_spec,
    ) in enumerate(
        zip(
            voxel_sizes,
            max_correspondence_distances,
            max_iterations,
            lambda_spec_levels,
            strict=True,
        ),
        1,
    ):
        level_start = transformation.copy()
        source_points, source_spec = _voxel_reduce(
            np.asarray(source_xyz_m, dtype=np.float64),
            source_standard,
            voxel,
        )
        target_points, target_spec = _voxel_reduce(
            np.asarray(target_xyz_m, dtype=np.float64),
            target_standard,
            voxel,
        )
        radius = 2.5 * voxel
        source_normals = _estimate_normals(source_points, radius)
        target_normals = _estimate_normals(target_points, radius)
        target_gradients, target_confidence = (
            fit_robust_tangent_spectral_gradients(
                target_points,
                target_normals,
                target_spec,
                radius=radius,
                max_nn=gradient_max_nn,
                min_neighbors=min_gradient_neighbors,
                max_condition=max_gradient_condition,
                huber_delta=huber_delta_spec,
                irls_iterations=gradient_irls_iterations,
            )
        )
        insufficient = False
        for iteration in range(1, iteration_limit + 1):
            pairs = build_mutual_correspondences(
                source_points,
                target_points,
                source_normals,
                target_normals,
                transformation,
                distance_limit=distance_limit,
                normal_angle_deg=normal_angle_deg,
            )
            if len(pairs.source_indices) < min_correspondences:
                insufficient = True
                history.append(
                    {
                        "record_type": "level_validation",
                        "level": level,
                        "iteration": iteration,
                        "accepted": False,
                        "rollback_reason": "insufficient_correspondences",
                        "inlier_count": len(pairs.source_indices),
                    }
                )
                transformation = level_start
                break

            q = apply_transform(
                source_points[pairs.source_indices],
                transformation,
            )
            p = target_points[pairs.target_indices]
            normals = target_normals[pairs.target_indices]
            gradients = target_gradients[pairs.target_indices]
            confidence = target_confidence[pairs.target_indices]
            delta_xyz = q - p
            normal_offset = np.einsum("ij,ij->i", delta_xyz, normals)
            tangent_offset = delta_xyz - normals * normal_offset[:, None]
            spectral_residual = (
                target_spec[pairs.target_indices]
                + np.einsum("nki,ni->nk", gradients, tangent_offset)
                - source_spec[pairs.source_indices]
            )
            motion_jacobian = np.concatenate(
                (
                    _minus_skew(q),
                    np.broadcast_to(np.eye(3), (len(q), 3, 3)),
                ),
                axis=2,
            )
            geometry_jacobian = (
                np.einsum("ni,nij->nj", normals, motion_jacobian)
                / distance_limit
            )
            projector = (
                np.broadcast_to(np.eye(3), (len(q), 3, 3))
                - normals[:, :, None] * normals[:, None, :]
            )
            tangent_motion = np.einsum(
                "nij,njk->nik",
                projector,
                motion_jacobian,
            )
            spectral_jacobian = np.einsum(
                "nki,nij->nkj",
                gradients,
                tangent_motion,
            )

            geometry_normalized = normal_offset / distance_limit
            effective = confidence * weights[None, :]
            effective /= np.maximum(
                effective.sum(axis=1, keepdims=True),
                1e-12,
            )
            geometry_loss = _huber_loss(
                geometry_normalized,
                huber_delta_geo,
            )
            spectral_loss = (
                _huber_loss(spectral_residual, huber_delta_spec)
                * effective
            ).sum(axis=1)
            combined = (
                (1.0 - lambda_spec) * geometry_loss
                + lambda_spec * spectral_loss
            )
            keep_count = max(
                1,
                int(np.ceil(trim_fraction * len(combined))),
            )
            keep = np.argsort(combined)[:keep_count]

            geo_huber = np.sqrt(
                _huber_irls_weights(
                    geometry_normalized[keep],
                    huber_delta_geo,
                )
            )
            geo_factor = np.sqrt(1.0 - lambda_spec) * geo_huber
            geometry_rows = (
                geo_factor[:, None] * geometry_jacobian[keep]
            )
            geometry_values = geo_factor * geometry_normalized[keep]

            spec_huber = np.sqrt(
                _huber_irls_weights(
                    spectral_residual[keep],
                    huber_delta_spec,
                )
            )
            spec_factor = np.sqrt(
                lambda_spec * effective[keep]
            ) * spec_huber
            spectral_rows = (
                spec_factor[:, :, None] * spectral_jacobian[keep]
            ).reshape(-1, 6)
            spectral_values = (
                spec_factor * spectral_residual[keep]
            ).ravel()
            jacobian = np.vstack((geometry_rows, spectral_rows))
            residual = np.concatenate((geometry_values, spectral_values))
            normal_matrix = jacobian.T @ jacobian + 1e-8 * np.eye(6)
            rhs = -(jacobian.T @ residual)
            try:
                twist = np.linalg.solve(normal_matrix, rhs)
            except np.linalg.LinAlgError:
                history.append(
                    {
                        "record_type": "iteration",
                        "level": level,
                        "iteration": iteration,
                        "accepted": False,
                        "rollback_reason": "singular_normal_matrix",
                    }
                )
                break

            rotation_norm = float(np.linalg.norm(twist[:3]))
            if rotation_norm > 0.2:
                twist[:3] *= 0.2 / rotation_norm
            translation_norm = float(np.linalg.norm(twist[3:]))
            if translation_norm > 0.5 * distance_limit:
                twist[3:] *= 0.5 * distance_limit / translation_norm

            objective = float(np.mean(combined[keep]))
            accepted_scale = 0.0
            accepted_objective = objective
            accepted_transform = transformation
            for scale in (
                1.0,
                0.5,
                0.25,
                0.125,
                0.0625,
                0.03125,
                0.015625,
            ):
                candidate = _se3_exp(scale * twist) @ transformation
                candidate_objective, candidate_distances = (
                    _fixed_pair_robust_objective(
                        candidate,
                        source_points,
                        source_spec,
                        pairs.source_indices,
                        target_points,
                        target_spec,
                        pairs.target_indices,
                        target_normals,
                        target_gradients,
                        target_confidence,
                        distance_limit=distance_limit,
                        band_weights=weights,
                        lambda_spec=lambda_spec,
                        huber_delta_geo=huber_delta_geo,
                        huber_delta_spec=huber_delta_spec,
                        trim_fraction=trim_fraction,
                    )
                )
                retained = float(
                    np.mean(candidate_distances <= distance_limit)
                )
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
            history.append(
                {
                    "record_type": "iteration",
                    "level": level,
                    "iteration": iteration,
                    "voxel_size_m": voxel,
                    "max_correspondence_m": distance_limit,
                    "inlier_count": len(pairs.source_indices),
                    "objective": objective,
                    "objective_after": accepted_objective,
                    "step_scale": accepted_scale,
                    "twist_norm": float(
                        accepted_scale * np.linalg.norm(twist)
                    ),
                }
            )
            if (
                accepted_scale == 0.0
                or accepted_scale * np.linalg.norm(twist) < 1e-7
                or abs(objective - accepted_objective) < 1e-10
            ):
                break

        if insufficient:
            continue
        if validation_config is not None:
            before = score_robust_spectral_transform(
                source_xyz_m,
                target_xyz_m,
                source_spectral6,
                target_spectral6,
                level_start,
                band_weights=weights,
                lambda_spec=lambda_spec,
                config=validation_config,
            )
            after = score_robust_spectral_transform(
                source_xyz_m,
                target_xyz_m,
                source_spectral6,
                target_spectral6,
                transformation,
                band_weights=weights,
                lambda_spec=lambda_spec,
                config=validation_config,
            )
            accepted, relative, reason = _validation_decision(
                before,
                after,
                min_relative_improvement=float(
                    validation_config["min_relative_improvement"]
                ),
                max_coverage_drop=float(
                    validation_config["max_coverage_drop"]
                ),
            )
            if not accepted:
                transformation = level_start
            history.append(
                {
                    "record_type": "level_validation",
                    "level": level,
                    "voxel_size_m": voxel,
                    "max_correspondence_m": distance_limit,
                    "accepted": accepted,
                    "rollback_reason": reason,
                    "validation_score_before": before.score,
                    "validation_score_after": after.score,
                    "validation_coverage_before": before.coverage,
                    "validation_coverage_after": after.coverage,
                    "validation_spectral_before": before.spectral_error,
                    "validation_spectral_after": after.spectral_error,
                    "validation_relative_improvement": relative,
                }
            )

    metrics = evaluate_registration(
        source_xyz_m,
        target_xyz_m,
        transformation,
        max_correspondence_distances[-1],
    )
    return ICPResult(
        transformation=transformation,
        fitness=float(metrics["fitness"]),
        inlier_rmse=float(metrics["inlier_rmse_m"]),
        history=history,
    )
