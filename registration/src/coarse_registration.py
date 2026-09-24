"""Open3D FPFH + RANSAC 全局粗配准。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .evaluation import evaluate_registration, make_point_cloud
from .transform import apply_transform


@dataclass
class CoarseCandidate:
    candidate_id: str
    seed: int | None
    transformation: np.ndarray
    rotation_magnitude_deg: float
    translation_magnitude_m: float
    coverage: float
    geometry_error: float
    spectral_error: float | None
    overlap_error: float
    joint_score: float
    accepted: bool
    rejection_reason: str
    modality: str = "spectral6"
    attribute_error: float = 0.0
    rgb_error: float | None = None


@dataclass
class CoarseRegistrationResult:
    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    source_downsampled_count: int
    target_downsampled_count: int
    parameters: dict[str, object]
    candidates: list[CoarseCandidate] = field(default_factory=list)
    selected_candidate_id: str = ""
    fallback_identity: bool = False
    selection_reason: str = ""


def _validated_points(
    points: np.ndarray,
    name: str,
    min_points: int = 4,
) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} 必须是 N×3")
    if len(array) < min_points:
        raise ValueError(f"{name} 至少需要{min_points}个点")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    return array


def voxel_reduce_with_spectral(
    xyz: np.ndarray,
    spectral6: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """用同一体素索引聚合坐标、六波段和原始点计数。"""
    points = _validated_points(xyz, "xyz", min_points=1)
    spectral = np.asarray(spectral6, dtype=np.float64)
    if spectral.shape != (len(points), 6):
        raise ValueError("spectral6 必须是与 xyz 对齐的 N×6")
    if not np.isfinite(spectral).all():
        raise ValueError("spectral6 包含 NaN 或 Inf")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m 必须为正数")

    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64)
    divisor = counts.astype(np.float64)
    reduced_points = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis]) / divisor
            for axis in range(3)
        ]
    )
    reduced_spectral = np.column_stack(
        [
            np.bincount(inverse, weights=spectral[:, band]) / divisor
            for band in range(6)
        ]
    )
    return reduced_points, reduced_spectral, counts


def _voxel_reduce_with_attribute(
    xyz: np.ndarray,
    attribute: np.ndarray | None,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """以相同体素索引聚合坐标及可选的逐点属性。"""
    points = _validated_points(xyz, "xyz", min_points=1)
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m 必须为正数")
    values: np.ndarray | None = None
    if attribute is not None:
        values = np.asarray(attribute, dtype=np.float64)
        if values.ndim != 2 or len(values) != len(points):
            raise ValueError("attribute 必须是与 xyz 对齐的 N×C")
        if not np.isfinite(values).all():
            raise ValueError("attribute 包含 NaN 或 Inf")

    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64)
    divisor = counts.astype(np.float64)
    reduced_points = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis]) / divisor
            for axis in range(3)
        ]
    )
    reduced_attribute = None
    if values is not None:
        reduced_attribute = np.column_stack(
            [
                np.bincount(inverse, weights=values[:, channel]) / divisor
                for channel in range(values.shape[1])
            ]
        )
    return reduced_points, reduced_attribute, counts


def _feature_from_array(data: np.ndarray) -> o3d.pipelines.registration.Feature:
    feature = o3d.pipelines.registration.Feature()
    feature.resize(data.shape[0], data.shape[1])
    feature.data = np.asarray(data, dtype=np.float64)
    return feature


def _cloud_with_fpfh(
    points: np.ndarray,
    voxel_size_m: float,
    normal_radius_multiplier: float,
    feature_radius_multiplier: float,
) -> tuple[o3d.geometry.PointCloud, np.ndarray]:
    cloud = make_point_cloud(points)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius_multiplier * voxel_size_m,
            max_nn=50,
        )
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=feature_radius_multiplier * voxel_size_m,
            max_nn=100,
        ),
    )
    return cloud, np.asarray(feature.data, dtype=np.float64).copy()


def _standardize_against_target(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """使用目标数据的median/MAD稳健标准化两侧特征。"""
    center = np.median(target, axis=0)
    mad = 1.4826 * np.median(np.abs(target - center), axis=0)
    scale = np.where(
        mad > 1e-6,
        mad,
        np.maximum(np.std(target, axis=0), 1e-6),
    )
    return (source - center) / scale, (target - center) / scale


def _build_modality_features(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_attribute: np.ndarray | None,
    target_attribute: np.ndarray | None,
    *,
    attribute_channels: int,
    config: dict[str, object],
) -> tuple[
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    o3d.pipelines.registration.Feature,
    o3d.pipelines.registration.Feature,
    np.ndarray | None,
    np.ndarray | None,
]:
    voxel_size = float(config["voxel_size_m"])
    source_points, source_reduced, _ = _voxel_reduce_with_attribute(
        source_xyz,
        source_attribute,
        voxel_size,
    )
    target_points, target_reduced, _ = _voxel_reduce_with_attribute(
        target_xyz,
        target_attribute,
        voxel_size,
    )
    if min(len(source_points), len(target_points)) < 4:
        raise ValueError("体素降采样后每侧至少需要4个点")
    if attribute_channels:
        if source_reduced is None or target_reduced is None:
            raise ValueError("该模态需要逐点属性")
        if (
            source_reduced.shape[1] != attribute_channels
            or target_reduced.shape[1] != attribute_channels
        ):
            raise ValueError(f"属性必须为 N×{attribute_channels}")

    source_cloud, source_fpfh = _cloud_with_fpfh(
        source_points,
        voxel_size,
        float(config["normal_radius_multiplier"]),
        float(config["feature_radius_multiplier"]),
    )
    target_cloud, target_fpfh = _cloud_with_fpfh(
        target_points,
        voxel_size,
        float(config["normal_radius_multiplier"]),
        float(config["feature_radius_multiplier"]),
    )
    fpfh_center = np.mean(target_fpfh, axis=1, keepdims=True)
    fpfh_scale = np.maximum(
        np.std(target_fpfh, axis=1, keepdims=True),
        1e-6,
    )
    source_descriptor = (source_fpfh - fpfh_center) / fpfh_scale
    target_descriptor = (target_fpfh - fpfh_center) / fpfh_scale

    standardized_source = None
    standardized_target = None
    if attribute_channels:
        standardized_source, standardized_target = _standardize_against_target(
            source_reduced,
            target_reduced,
        )
        channel_weights = config.get("attribute_channel_weights")
        if channel_weights is not None:
            channel_weights = np.asarray(
                channel_weights,
                dtype=np.float64,
            )
            if (
                channel_weights.shape != (attribute_channels,)
                or not np.isfinite(channel_weights).all()
                or np.any(channel_weights < 0.0)
                or not np.isclose(channel_weights.sum(), 1.0)
            ):
                raise ValueError(
                    "attribute_channel_weights必须非负、和为1且与通道数一致"
                )
            channel_scale = np.sqrt(
                attribute_channels * channel_weights
            )
            standardized_source = standardized_source * channel_scale
            standardized_target = standardized_target * channel_scale
        weight = float(
            config.get(
                "attribute_feature_weight",
                config.get("spectral_feature_weight", 1.0),
            )
        )
        if weight <= 0:
            raise ValueError("attribute_feature_weight 必须为正数")
        source_descriptor = np.vstack(
            (source_descriptor, weight * standardized_source.T)
        )
        target_descriptor = np.vstack(
            (target_descriptor, weight * standardized_target.T)
        )
    if not (
        np.isfinite(source_descriptor).all()
        and np.isfinite(target_descriptor).all()
    ):
        raise ValueError("粗配准描述子包含 NaN 或 Inf")
    return (
        source_cloud,
        target_cloud,
        _feature_from_array(source_descriptor),
        _feature_from_array(target_descriptor),
        standardized_source,
        standardized_target,
    )


def build_geometry_features(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    config: dict[str, object],
) -> tuple[
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    o3d.pipelines.registration.Feature,
    o3d.pipelines.registration.Feature,
    None,
    None,
]:
    """构造仅含33维标准化FPFH的几何描述子。"""
    return _build_modality_features(
        source_xyz,
        target_xyz,
        None,
        None,
        attribute_channels=0,
        config=config,
    )


def build_rgb_features(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    config: dict[str, object],
) -> tuple[
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    o3d.pipelines.registration.Feature,
    o3d.pipelines.registration.Feature,
    np.ndarray,
    np.ndarray,
]:
    """构造33维FPFH+3维真实RGB描述子。"""
    return _build_modality_features(
        source_xyz,
        target_xyz,
        source_rgb,
        target_rgb,
        attribute_channels=3,
        config=config,
    )


def build_spectral_features(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    *,
    config: dict[str, object],
) -> tuple[
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    o3d.pipelines.registration.Feature,
    o3d.pipelines.registration.Feature,
    np.ndarray,
    np.ndarray,
]:
    """构造33维FPFH+6维真实六波段描述子。"""
    return _build_modality_features(
        source_xyz,
        target_xyz,
        source_spectral6,
        target_spectral6,
        attribute_channels=6,
        config=config,
    )


def build_joint_features(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    voxel_size_m: float,
    normal_radius_multiplier: float,
    feature_radius_multiplier: float,
    spectral_feature_weight: float,
) -> tuple[
    o3d.geometry.PointCloud,
    o3d.geometry.PointCloud,
    o3d.pipelines.registration.Feature,
    o3d.pipelines.registration.Feature,
    np.ndarray,
    np.ndarray,
]:
    """构造坐标与六波段严格对齐的39维联合特征。"""
    if min(
        voxel_size_m,
        normal_radius_multiplier,
        feature_radius_multiplier,
        spectral_feature_weight,
    ) <= 0:
        raise ValueError("联合特征参数必须为正数")

    source_points, source_spec, _ = voxel_reduce_with_spectral(
        source_xyz,
        source_spectral6,
        voxel_size_m,
    )
    target_points, target_spec, _ = voxel_reduce_with_spectral(
        target_xyz,
        target_spectral6,
        voxel_size_m,
    )
    if min(len(source_points), len(target_points)) < 4:
        raise ValueError("联合体素降采样后每侧至少需要4个点")

    source_cloud, source_fpfh = _cloud_with_fpfh(
        source_points,
        voxel_size_m,
        normal_radius_multiplier,
        feature_radius_multiplier,
    )
    target_cloud, target_fpfh = _cloud_with_fpfh(
        target_points,
        voxel_size_m,
        normal_radius_multiplier,
        feature_radius_multiplier,
    )

    fpfh_center = np.mean(target_fpfh, axis=1, keepdims=True)
    fpfh_scale = np.maximum(
        np.std(target_fpfh, axis=1, keepdims=True),
        1e-6,
    )
    source_fpfh = (source_fpfh - fpfh_center) / fpfh_scale
    target_fpfh = (target_fpfh - fpfh_center) / fpfh_scale

    spectral_center = np.median(target_spec, axis=0)
    spectral_mad = 1.4826 * np.median(
        np.abs(target_spec - spectral_center),
        axis=0,
    )
    spectral_scale = np.where(
        spectral_mad > 1e-6,
        spectral_mad,
        np.maximum(np.std(target_spec, axis=0), 1e-6),
    )
    source_spec = (source_spec - spectral_center) / spectral_scale
    target_spec = (target_spec - spectral_center) / spectral_scale

    source_joint = np.vstack(
        (source_fpfh, spectral_feature_weight * source_spec.T)
    )
    target_joint = np.vstack(
        (target_fpfh, spectral_feature_weight * target_spec.T)
    )
    if not (
        np.isfinite(source_joint).all()
        and np.isfinite(target_joint).all()
        and np.isfinite(source_spec).all()
        and np.isfinite(target_spec).all()
    ):
        raise ValueError("联合特征包含 NaN 或 Inf")
    return (
        source_cloud,
        target_cloud,
        _feature_from_array(source_joint),
        _feature_from_array(target_joint),
        source_spec,
        target_spec,
    )


def score_modality_candidate(
    *,
    modality: str,
    candidate_id: str,
    seed: int | None,
    transformation: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    config: dict[str, object],
    source_attribute: np.ndarray | None = None,
    target_attribute: np.ndarray | None = None,
) -> CoarseCandidate:
    """在固定空间对应上计算指定模态的联合候选分数。"""
    if modality not in {"geometry", "rgb", "spectral6"}:
        raise ValueError(f"未知粗配准模态: {modality}")
    transform = np.asarray(transformation, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("候选变换必须是有限的4×4矩阵")
    source = _validated_points(source_points, "source_points")
    target = _validated_points(target_points, "target_points")
    source_values = None
    target_values = None
    expected_channels = {"rgb": 3, "spectral6": 6}.get(modality)
    if expected_channels is not None:
        source_values = np.asarray(source_attribute, dtype=np.float64)
        target_values = np.asarray(target_attribute, dtype=np.float64)
        if source_values.shape != (len(source), expected_channels):
            raise ValueError(
                f"source_attribute 必须是 N×{expected_channels}"
            )
        if target_values.shape != (len(target), expected_channels):
            raise ValueError(
                f"target_attribute 必须是 N×{expected_channels}"
            )
        if not (
            np.isfinite(source_values).all()
            and np.isfinite(target_values).all()
        ):
            raise ValueError("粗配准属性包含 NaN 或 Inf")

    rotation_magnitude = float(
        np.degrees(Rotation.from_matrix(transform[:3, :3]).magnitude())
    )
    translation_magnitude = float(np.linalg.norm(transform[:3, 3]))
    if (
        rotation_magnitude > float(config["max_rotation_deg"])
        or translation_magnitude > float(config["max_translation_m"])
    ):
        return CoarseCandidate(
            candidate_id=candidate_id,
            seed=seed,
            transformation=transform.copy(),
            rotation_magnitude_deg=rotation_magnitude,
            translation_magnitude_m=translation_magnitude,
            coverage=0.0,
            geometry_error=float("inf"),
            spectral_error=(
                float("inf") if modality == "spectral6" else None
            ),
            overlap_error=1.0,
            joint_score=float("inf"),
            accepted=False,
            rejection_reason="rejected_transform_bound",
            modality=modality,
            attribute_error=(
                0.0 if modality == "geometry" else float("inf")
            ),
            rgb_error=float("inf") if modality == "rgb" else None,
        )

    transformed = apply_transform(source, transform)
    distances, target_indices = cKDTree(target).query(transformed, k=1)
    max_distance = float(config["score_max_distance_m"])
    inliers = distances <= max_distance
    inlier_count = int(np.count_nonzero(inliers))
    coverage = float(np.mean(inliers))
    if (
        inlier_count < int(config["min_correspondences"])
        or coverage < float(config["min_coverage"])
    ):
        return CoarseCandidate(
            candidate_id=candidate_id,
            seed=seed,
            transformation=transform.copy(),
            rotation_magnitude_deg=rotation_magnitude,
            translation_magnitude_m=translation_magnitude,
            coverage=coverage,
            geometry_error=float("inf"),
            spectral_error=(
                float("inf") if modality == "spectral6" else None
            ),
            overlap_error=1.0 - coverage,
            joint_score=float("inf"),
            accepted=False,
            rejection_reason="rejected_insufficient_overlap",
            modality=modality,
            attribute_error=(
                0.0 if modality == "geometry" else float("inf")
            ),
            rgb_error=float("inf") if modality == "rgb" else None,
        )

    geometry_per_pair = (distances[inliers] / max_distance) ** 2
    attribute_per_pair = np.zeros_like(geometry_per_pair)
    if source_values is not None and target_values is not None:
        attribute_delta = (
            source_values[inliers] - target_values[target_indices[inliers]]
        )
        attribute_per_pair = np.mean(attribute_delta**2, axis=1)
    lambda_geo = float(config["lambda_geo"])
    lambda_attribute = float(
        config.get("lambda_attribute", config.get("lambda_spec", 0.0))
    )
    lambda_overlap = float(config["lambda_overlap"])
    if not np.isclose(
        lambda_geo + lambda_attribute + lambda_overlap,
        1.0,
    ):
        raise ValueError("粗配准评分权重之和必须为1")
    if modality == "geometry" and not np.isclose(lambda_attribute, 0.0):
        raise ValueError("geometry 粗配准不能使用属性权重")
    pair_score = (
        lambda_geo * geometry_per_pair
        + lambda_attribute * attribute_per_pair
    )
    keep_count = max(
        1,
        int(np.ceil(float(config["trim_fraction"]) * len(pair_score))),
    )
    keep = np.argsort(pair_score)[:keep_count]
    geometry_error = float(np.mean(geometry_per_pair[keep]))
    attribute_error = float(np.mean(attribute_per_pair[keep]))
    overlap_error = 1.0 - coverage
    joint_score = (
        lambda_geo * geometry_error
        + lambda_attribute * attribute_error
        + lambda_overlap * overlap_error
    )
    return CoarseCandidate(
        candidate_id=candidate_id,
        seed=seed,
        transformation=transform.copy(),
        rotation_magnitude_deg=rotation_magnitude,
        translation_magnitude_m=translation_magnitude,
        coverage=coverage,
        geometry_error=geometry_error,
        spectral_error=(
            attribute_error if modality == "spectral6" else None
        ),
        overlap_error=overlap_error,
        joint_score=joint_score,
        accepted=True,
        rejection_reason="",
        modality=modality,
        attribute_error=attribute_error,
        rgb_error=attribute_error if modality == "rgb" else None,
    )


def score_coarse_candidate(
    *,
    candidate_id: str,
    seed: int | None,
    transformation: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    config: dict[str, object],
) -> CoarseCandidate:
    """兼容旧接口：按六波段模态计算候选分数。"""
    return score_modality_candidate(
        modality="spectral6",
        candidate_id=candidate_id,
        seed=seed,
        transformation=transformation,
        source_points=source_points,
        target_points=target_points,
        source_attribute=source_spectral6,
        target_attribute=target_spectral6,
        config=config,
    )


def register_multispectral_coarse(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    *,
    config: dict[str, object],
    seed: int,
) -> CoarseRegistrationResult:
    """生成并选择几何+六波段粗配准候选。"""
    (
        source_cloud,
        target_cloud,
        source_feature,
        target_feature,
        source_spec,
        target_spec,
    ) = build_joint_features(
        source_xyz_m,
        target_xyz_m,
        source_spectral6,
        target_spectral6,
        voxel_size_m=float(config["voxel_size_m"]),
        normal_radius_multiplier=float(config["normal_radius_multiplier"]),
        feature_radius_multiplier=float(config["feature_radius_multiplier"]),
        spectral_feature_weight=float(config["spectral_feature_weight"]),
    )
    source_points = np.asarray(source_cloud.points).copy()
    target_points = np.asarray(target_cloud.points).copy()
    identity = score_coarse_candidate(
        candidate_id="identity_baseline",
        seed=None,
        transformation=np.eye(4),
        source_points=source_points,
        target_points=target_points,
        source_spectral6=source_spec,
        target_spectral6=target_spec,
        config=config,
    )
    candidate_count = int(config["candidate_count"])
    if candidate_count < 0:
        raise ValueError("candidate_count 不能为负数")
    candidates = [identity]
    valid_candidates: list[CoarseCandidate] = []
    for index in range(candidate_count):
        candidate_seed = int(seed) + index
        o3d.utility.random.seed(candidate_seed)
        registration = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_cloud,
                target_cloud,
                source_feature,
                target_feature,
                True,
                float(config["max_correspondence_m"]),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    False
                ),
                int(config["ransac_n"]),
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        float(config["max_correspondence_m"])
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    int(config["max_iterations"]),
                    float(config["confidence"]),
                ),
            )
        )
        candidate = score_coarse_candidate(
            candidate_id=f"ransac_{index + 1:03d}",
            seed=candidate_seed,
            transformation=np.asarray(registration.transformation).copy(),
            source_points=source_points,
            target_points=target_points,
            source_spectral6=source_spec,
            target_spectral6=target_spec,
            config=config,
        )
        if candidate.accepted and not (
            candidate.joint_score
            <= identity.joint_score - float(config["min_score_improvement"])
        ):
            candidate.accepted = False
            candidate.rejection_reason = "rejected_no_score_improvement"
        if candidate.accepted:
            valid_candidates.append(candidate)
        candidates.append(candidate)

    selected = (
        min(valid_candidates, key=lambda item: item.joint_score)
        if valid_candidates
        else identity
    )
    fallback_identity = not valid_candidates
    selection_reason = (
        "best_joint_score"
        if valid_candidates
        else "no_candidate_improved_baseline"
    )
    evaluation = evaluate_registration(
        source_xyz_m,
        target_xyz_m,
        selected.transformation,
        float(config["score_max_distance_m"]),
    )
    parameters: dict[str, float | int | bool] = {
        key: value
        for key, value in config.items()
        if isinstance(value, (bool, int, float))
    }
    parameters["seed"] = seed
    parameters["identity_baseline_score"] = identity.joint_score
    parameters["selected_score"] = selected.joint_score
    parameters["valid_candidate_count"] = len(valid_candidates)
    return CoarseRegistrationResult(
        transformation=selected.transformation.copy(),
        fitness=float(evaluation["fitness"]),
        inlier_rmse=float(evaluation["inlier_rmse_m"]),
        source_downsampled_count=len(source_points),
        target_downsampled_count=len(target_points),
        parameters=parameters,
        candidates=candidates,
        selected_candidate_id=selected.candidate_id,
        fallback_identity=fallback_identity,
        selection_reason=selection_reason,
    )


def _register_modality_coarse(
    modality: str,
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    source_attribute: np.ndarray | None,
    target_attribute: np.ndarray | None,
    *,
    config: dict[str, object],
    seed: int,
) -> CoarseRegistrationResult:
    """执行共用预算、指定描述子与评分属性的粗配准。"""
    builders = {
        "geometry": lambda: build_geometry_features(
            source_xyz_m,
            target_xyz_m,
            config=config,
        ),
        "rgb": lambda: build_rgb_features(
            source_xyz_m,
            target_xyz_m,
            np.asarray(source_attribute),
            np.asarray(target_attribute),
            config=config,
        ),
        "spectral6": lambda: build_spectral_features(
            source_xyz_m,
            target_xyz_m,
            np.asarray(source_attribute),
            np.asarray(target_attribute),
            config=config,
        ),
    }
    if modality not in builders:
        raise ValueError(f"未知粗配准模态: {modality}")
    (
        source_cloud,
        target_cloud,
        source_feature,
        target_feature,
        source_values,
        target_values,
    ) = builders[modality]()
    source_points = np.asarray(source_cloud.points).copy()
    target_points = np.asarray(target_cloud.points).copy()

    identity = score_modality_candidate(
        modality=modality,
        candidate_id="identity_baseline",
        seed=None,
        transformation=np.eye(4),
        source_points=source_points,
        target_points=target_points,
        source_attribute=source_values,
        target_attribute=target_values,
        config=config,
    )
    candidate_count = int(config["candidate_count"])
    if candidate_count < 0:
        raise ValueError("candidate_count 不能为负数")
    candidates = [identity]
    valid_candidates: list[CoarseCandidate] = []
    for index in range(candidate_count):
        candidate_seed = int(seed) + index
        o3d.utility.random.seed(candidate_seed)
        registration = (
            o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_cloud,
                target_cloud,
                source_feature,
                target_feature,
                True,
                float(config["max_correspondence_m"]),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    False
                ),
                int(config["ransac_n"]),
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                        0.9
                    ),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                        float(config["max_correspondence_m"])
                    ),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(
                    int(config["max_iterations"]),
                    float(config["confidence"]),
                ),
            )
        )
        candidate = score_modality_candidate(
            modality=modality,
            candidate_id=f"ransac_{index + 1:03d}",
            seed=candidate_seed,
            transformation=np.asarray(registration.transformation).copy(),
            source_points=source_points,
            target_points=target_points,
            source_attribute=source_values,
            target_attribute=target_values,
            config=config,
        )
        improvement = float(config["min_score_improvement"])
        if candidate.accepted and not (
            candidate.joint_score <= identity.joint_score - improvement
        ):
            candidate.accepted = False
            candidate.rejection_reason = "rejected_no_score_improvement"
        if candidate.accepted:
            valid_candidates.append(candidate)
        candidates.append(candidate)

    selected = (
        min(valid_candidates, key=lambda item: item.joint_score)
        if valid_candidates
        else identity
    )
    evaluation = evaluate_registration(
        source_xyz_m,
        target_xyz_m,
        selected.transformation,
        float(config["score_max_distance_m"]),
    )
    parameters: dict[str, object] = {
        key: value
        for key, value in config.items()
        if isinstance(value, (bool, int, float, str))
    }
    parameters.update(
        {
            "modality": modality,
            "seed": seed,
            "identity_baseline_score": identity.joint_score,
            "selected_score": selected.joint_score,
            "valid_candidate_count": len(valid_candidates),
        }
    )
    return CoarseRegistrationResult(
        transformation=selected.transformation.copy(),
        fitness=float(evaluation["fitness"]),
        inlier_rmse=float(evaluation["inlier_rmse_m"]),
        source_downsampled_count=len(source_points),
        target_downsampled_count=len(target_points),
        parameters=parameters,
        candidates=candidates,
        selected_candidate_id=selected.candidate_id,
        fallback_identity=not valid_candidates,
        selection_reason=(
            "best_joint_score"
            if valid_candidates
            else "no_candidate_improved_baseline"
        ),
    )


def register_geometry_coarse(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    config: dict[str, object],
    seed: int,
) -> CoarseRegistrationResult:
    """仅使用XYZ/FPFH生成并评价粗配准候选。"""
    return _register_modality_coarse(
        "geometry",
        source_xyz,
        target_xyz,
        None,
        None,
        config=config,
        seed=seed,
    )


def register_rgb_coarse(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    config: dict[str, object],
    seed: int,
) -> CoarseRegistrationResult:
    """使用FPFH+真实RGB生成并评价粗配准候选。"""
    return _register_modality_coarse(
        "rgb",
        source_xyz,
        target_xyz,
        source_rgb,
        target_rgb,
        config=config,
        seed=seed,
    )


def register_spectral_coarse(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    *,
    config: dict[str, object],
    seed: int,
) -> CoarseRegistrationResult:
    """使用FPFH+真实六波段生成并评价粗配准候选。"""
    return _register_modality_coarse(
        "spectral6",
        source_xyz,
        target_xyz,
        source_spectral6,
        target_spectral6,
        config=config,
        seed=seed,
    )


def preprocess_fpfh(
    points: np.ndarray,
    voxel_size_m: float,
    normal_radius_multiplier: float,
    feature_radius_multiplier: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """降采样点云、估计法向并计算33维FPFH特征。"""
    points = _validated_points(points, "points")
    if min(voxel_size_m, normal_radius_multiplier, feature_radius_multiplier) <= 0:
        raise ValueError("FPFH 参数必须为正数")

    cloud = make_point_cloud(points).voxel_down_sample(voxel_size_m)
    if len(cloud.points) < 4:
        raise ValueError("体素降采样后至少需要4个点")
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius_multiplier * voxel_size_m,
            max_nn=50,
        )
    )
    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=feature_radius_multiplier * voxel_size_m,
            max_nn=100,
        ),
    )
    return cloud, feature


def register_fpfh_ransac(
    source_xyz_m: np.ndarray,
    target_xyz_m: np.ndarray,
    voxel_size_m: float,
    max_correspondence_m: float,
    normal_radius_multiplier: float,
    feature_radius_multiplier: float,
    ransac_n: int,
    max_iterations: int,
    confidence: float,
    seed: int,
) -> CoarseRegistrationResult:
    """通过FPFH特征匹配和RANSAC估计不依赖初始位姿的粗变换。"""
    if max_correspondence_m <= 0 or ransac_n < 3 or max_iterations <= 0:
        raise ValueError("RANSAC 距离、采样点数和迭代次数无效")
    if not 0.0 < confidence <= 1.0:
        raise ValueError("confidence 必须在 (0, 1] 内")

    source, source_feature = preprocess_fpfh(
        source_xyz_m,
        voxel_size_m,
        normal_radius_multiplier,
        feature_radius_multiplier,
    )
    target, target_feature = preprocess_fpfh(
        target_xyz_m,
        voxel_size_m,
        normal_radius_multiplier,
        feature_radius_multiplier,
    )
    o3d.utility.random.seed(int(seed))
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_feature,
        target_feature,
        True,
        max_correspondence_m,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                max_correspondence_m
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(
            max_iterations,
            confidence,
        ),
    )
    transformation = np.asarray(result.transformation, dtype=np.float64).copy()
    if transformation.shape != (4, 4) or not np.isfinite(transformation).all():
        raise RuntimeError("RANSAC 返回了无效变换")

    parameters: dict[str, float | int | bool] = {
        "voxel_size_m": voxel_size_m,
        "max_correspondence_m": max_correspondence_m,
        "normal_radius_multiplier": normal_radius_multiplier,
        "feature_radius_multiplier": feature_radius_multiplier,
        "ransac_n": ransac_n,
        "max_iterations": max_iterations,
        "confidence": confidence,
        "seed": seed,
    }
    return CoarseRegistrationResult(
        transformation=transformation,
        fitness=float(result.fitness),
        inlier_rmse=float(result.inlier_rmse),
        source_downsampled_count=len(source.points),
        target_downsampled_count=len(target.points),
        parameters=parameters,
    )
