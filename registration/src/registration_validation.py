"""固定验证集上的多模态配准评分与逐层回退决策。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from .transform import apply_transform


@dataclass(frozen=True)
class ValidationScore:
    score: float
    coverage: float
    inlier_count: int
    geometry_error: float
    attribute_error: float
    valid: bool
    invalid_reason: str


@dataclass(frozen=True)
class ValidationDecision:
    accepted: bool
    relative_improvement: float
    rollback_reason: str
    before: ValidationScore
    after: ValidationScore


@dataclass(frozen=True)
class ValidationContext:
    source_xyz: np.ndarray
    target_xyz: np.ndarray
    source_attributes: np.ndarray | None
    target_attributes: np.ndarray | None
    target_tree: cKDTree
    voxel_size_m: float
    max_correspondence_m: float
    trim_fraction: float
    min_correspondences: int
    min_relative_improvement: float
    max_coverage_drop: float
    geometry_weight: float
    attribute_weight: float
    overlap_weight: float
    attribute_clip_sigma: float


_REQUIRED_CONFIG = (
    "voxel_size_m",
    "max_correspondence_m",
    "trim_fraction",
    "min_correspondences",
    "min_relative_improvement",
    "max_coverage_drop",
    "overlap_weight",
    "attribute_clip_sigma",
)


def _validated_points(points: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0:
        raise ValueError(f"{name} 必须是非空 N×3 数组")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    return result


def _validated_attributes(
    attributes: np.ndarray | None,
    point_count: int,
    name: str,
) -> np.ndarray | None:
    if attributes is None:
        return None
    result = np.asarray(attributes, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] != point_count or result.shape[1] == 0:
        raise ValueError(f"{name} 必须是与点数一致的 N×C 数组")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    return result


def _voxel_reduce(
    points: np.ndarray,
    attributes: np.ndarray | None,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    reduced_points = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis]) / counts
            for axis in range(3)
        ]
    )
    if attributes is None:
        return reduced_points, None
    reduced_attributes = np.column_stack(
        [
            np.bincount(inverse, weights=attributes[:, channel]) / counts
            for channel in range(attributes.shape[1])
        ]
    )
    return reduced_points, reduced_attributes


def _standardize_attributes(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(target, axis=0)
    mad = 1.4826 * np.median(np.abs(target - median), axis=0)
    fallback = np.std(target, axis=0)
    scale = np.where(mad > 1e-6, mad, np.maximum(fallback, 1e-6))
    return (source - median) / scale, (target - median) / scale


def _finite_float(config: Mapping[str, object], key: str) -> float:
    value = float(config[key])
    if not np.isfinite(value):
        raise ValueError(f"registration_validation.{key} 必须是有限数")
    return value


def build_validation_context(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_attributes: np.ndarray | None,
    target_attributes: np.ndarray | None,
    *,
    geometry_fraction: float,
    attribute_fraction: float,
    config: Mapping[str, object],
) -> ValidationContext:
    """构造一次方法调用内固定不变的降采样验证集。"""
    missing = [key for key in _REQUIRED_CONFIG if key not in config]
    if missing:
        raise ValueError(f"registration_validation 缺少配置: {missing}")
    source = _validated_points(source_xyz, "source_xyz")
    target = _validated_points(target_xyz, "target_xyz")
    source_attr = _validated_attributes(
        source_attributes,
        len(source),
        "source_attributes",
    )
    target_attr = _validated_attributes(
        target_attributes,
        len(target),
        "target_attributes",
    )
    if (source_attr is None) != (target_attr is None):
        raise ValueError("source_attributes 与 target_attributes 必须同时提供")
    if (
        source_attr is not None
        and target_attr is not None
        and source_attr.shape[1] != target_attr.shape[1]
    ):
        raise ValueError("source_attributes 与 target_attributes 通道数必须一致")

    geometry_fraction = float(geometry_fraction)
    attribute_fraction = float(attribute_fraction)
    if (
        not np.isfinite([geometry_fraction, attribute_fraction]).all()
        or geometry_fraction < 0.0
        or attribute_fraction < 0.0
        or geometry_fraction + attribute_fraction <= 0.0
    ):
        raise ValueError("几何与属性验证权重必须非负且总和大于0")
    if attribute_fraction > 0.0 and source_attr is None:
        raise ValueError("属性验证权重大于0时必须提供属性")

    voxel_size = _finite_float(config, "voxel_size_m")
    max_distance = _finite_float(config, "max_correspondence_m")
    trim_fraction = _finite_float(config, "trim_fraction")
    min_relative_improvement = _finite_float(
        config,
        "min_relative_improvement",
    )
    max_coverage_drop = _finite_float(config, "max_coverage_drop")
    overlap_weight = _finite_float(config, "overlap_weight")
    attribute_clip_sigma = _finite_float(config, "attribute_clip_sigma")
    min_correspondences = int(config["min_correspondences"])
    if voxel_size <= 0.0 or max_distance <= 0.0:
        raise ValueError("验证体素和最大对应距离必须大于0")
    if not 0.0 < trim_fraction <= 1.0:
        raise ValueError("trim_fraction 必须位于 (0,1]")
    if min_correspondences < 1:
        raise ValueError("min_correspondences 必须至少为1")
    if min_relative_improvement < 0.0:
        raise ValueError("min_relative_improvement 不能为负数")
    if not 0.0 <= max_coverage_drop <= 1.0:
        raise ValueError("max_coverage_drop 必须位于 [0,1]")
    if not 0.0 <= overlap_weight < 1.0:
        raise ValueError("overlap_weight 必须位于 [0,1)")
    if attribute_clip_sigma <= 0.0:
        raise ValueError("attribute_clip_sigma 必须大于0")

    source_down, source_attr_down = _voxel_reduce(
        source,
        source_attr,
        voxel_size,
    )
    target_down, target_attr_down = _voxel_reduce(
        target,
        target_attr,
        voxel_size,
    )
    if source_attr_down is not None and target_attr_down is not None:
        source_attr_down, target_attr_down = _standardize_attributes(
            source_attr_down,
            target_attr_down,
        )

    data_weight = 1.0 - overlap_weight
    fraction_sum = geometry_fraction + attribute_fraction
    return ValidationContext(
        source_xyz=source_down,
        target_xyz=target_down,
        source_attributes=source_attr_down,
        target_attributes=target_attr_down,
        target_tree=cKDTree(target_down),
        voxel_size_m=voxel_size,
        max_correspondence_m=max_distance,
        trim_fraction=trim_fraction,
        min_correspondences=min_correspondences,
        min_relative_improvement=min_relative_improvement,
        max_coverage_drop=max_coverage_drop,
        geometry_weight=data_weight * geometry_fraction / fraction_sum,
        attribute_weight=data_weight * attribute_fraction / fraction_sum,
        overlap_weight=overlap_weight,
        attribute_clip_sigma=attribute_clip_sigma,
    )


def _invalid_score(reason: str, coverage: float, count: int) -> ValidationScore:
    return ValidationScore(
        score=float("inf"),
        coverage=coverage,
        inlier_count=count,
        geometry_error=float("inf"),
        attribute_error=float("inf"),
        valid=False,
        invalid_reason=reason,
    )


def score_transform(
    context: ValidationContext,
    transformation: np.ndarray,
) -> ValidationScore:
    """在固定验证集上计算越小越好的鲁棒联合分数。"""
    transform = np.asarray(transformation, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        return _invalid_score("nonfinite_transform", 0.0, 0)
    transformed = apply_transform(context.source_xyz, transform)
    distances, target_indices = context.target_tree.query(transformed, k=1)
    inliers = distances <= context.max_correspondence_m
    inlier_count = int(np.count_nonzero(inliers))
    coverage = float(np.mean(inliers))
    if inlier_count < context.min_correspondences:
        return _invalid_score(
            "insufficient_correspondences",
            coverage,
            inlier_count,
        )

    geometry_per_pair = (
        distances[inliers] / context.max_correspondence_m
    ) ** 2
    if (
        context.attribute_weight > 0.0
        and context.source_attributes is not None
        and context.target_attributes is not None
    ):
        delta = (
            context.source_attributes[inliers]
            - context.target_attributes[target_indices[inliers]]
        )
        clip_squared = context.attribute_clip_sigma**2
        attribute_per_pair = np.mean(
            np.minimum(delta**2, clip_squared) / clip_squared,
            axis=1,
        )
    else:
        attribute_per_pair = np.zeros(inlier_count, dtype=np.float64)

    pair_score = (
        context.geometry_weight * geometry_per_pair
        + context.attribute_weight * attribute_per_pair
    )
    keep_count = max(
        1,
        int(np.ceil(context.trim_fraction * inlier_count)),
    )
    keep = np.argsort(pair_score)[:keep_count]
    geometry_error = float(np.mean(geometry_per_pair[keep]))
    attribute_error = float(np.mean(attribute_per_pair[keep]))
    score = (
        context.geometry_weight * geometry_error
        + context.attribute_weight * attribute_error
        + context.overlap_weight * (1.0 - coverage)
    )
    return ValidationScore(
        score=score,
        coverage=coverage,
        inlier_count=inlier_count,
        geometry_error=geometry_error,
        attribute_error=attribute_error,
        valid=True,
        invalid_reason="",
    )


def decide_candidate(
    context: ValidationContext,
    accepted_transform: np.ndarray,
    candidate_transform: np.ndarray,
) -> ValidationDecision:
    """比较候选与已接受变换，返回接受或回退决定。"""
    before = score_transform(context, accepted_transform)
    if not before.valid:
        raise ValueError(
            f"初始变换无法建立有效验证基线: {before.invalid_reason}"
        )
    after = score_transform(context, candidate_transform)
    relative_improvement = (
        (before.score - after.score) / max(abs(before.score), 1e-12)
        if after.valid
        else float("-inf")
    )
    if not after.valid:
        return ValidationDecision(
            False,
            relative_improvement,
            after.invalid_reason,
            before,
            after,
        )
    if relative_improvement < context.min_relative_improvement:
        return ValidationDecision(
            False,
            relative_improvement,
            "insufficient_score_improvement",
            before,
            after,
        )
    if after.coverage < before.coverage - context.max_coverage_drop:
        return ValidationDecision(
            False,
            relative_improvement,
            "coverage_drop",
            before,
            after,
        )
    return ValidationDecision(
        True,
        relative_improvement,
        "",
        before,
        after,
    )
