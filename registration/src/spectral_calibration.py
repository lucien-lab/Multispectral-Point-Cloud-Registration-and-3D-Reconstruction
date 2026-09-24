"""左右扫描六波段相对响应校正。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from .data_prepare import WAVELENGTHS_NM
from .transform import apply_transform


@dataclass(frozen=True)
class CalibrationTrial:
    """一个训练trial中由纯几何粗配准给出的原始source到target变换。"""

    trial: int
    source_to_target: np.ndarray


@dataclass(frozen=True)
class SpectralCalibration:
    wavelengths_nm: np.ndarray
    gain: np.ndarray
    offset: np.ndarray
    residual_mad: np.ndarray
    correlation: np.ndarray
    band_weights: np.ndarray
    pair_count: int


@dataclass(frozen=True)
class OverlapCorrespondences:
    source_indices: np.ndarray
    target_indices: np.ndarray


def _points(array: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) == 0:
        raise ValueError(f"{name} 必须是非空N×3数组")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} 包含NaN或Inf")
    return result


def _spectral(array: np.ndarray, count: int, name: str) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64)
    if result.shape != (count, 6):
        raise ValueError(f"{name} 必须是与点数一致的N×6数组")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} 包含NaN或Inf")
    return result


def _estimate_normals(points: np.ndarray, radius: float) -> np.ndarray:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50)
    )
    normals = np.asarray(cloud.normals).copy()
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(lengths, 1e-12)


def _bounded_simplex(
    values: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    if len(values) * lower > 1.0 or len(values) * upper < 1.0:
        raise ValueError("通道权重上下界无法满足总和为1")
    values = np.asarray(values, dtype=np.float64)
    values = values / max(float(values.sum()), 1e-12)
    low = float(np.min(values - upper))
    high = float(np.max(values - lower))
    for _ in range(100):
        middle = 0.5 * (low + high)
        projected = np.clip(values - middle, lower, upper)
        if float(projected.sum()) > 1.0:
            low = middle
        else:
            high = middle
    projected = np.clip(values - 0.5 * (low + high), lower, upper)
    return projected / float(projected.sum())


def _collect_pairs(
    trials: Iterable[CalibrationTrial],
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    max_distance: float,
    normal_angle_deg: float,
    normal_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_normals = _estimate_normals(source_xyz, normal_radius)
    target_normals = _estimate_normals(target_xyz, normal_radius)
    target_tree = cKDTree(target_xyz)
    unique_pairs: set[tuple[int, int]] = set()
    for trial in trials:
        transform = np.asarray(trial.source_to_target, dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("训练trial变换必须是有限4×4矩阵")
        moved = apply_transform(source_xyz, transform)
        moved_normals = source_normals @ transform[:3, :3].T
        distances, target_indices = target_tree.query(moved, k=1, workers=1)
        reverse_indices = cKDTree(moved).query(
            target_xyz,
            k=1,
            workers=1,
        )[1]
        cosine_limit = np.cos(np.radians(normal_angle_deg))
        cosine = np.abs(
            np.einsum("ij,ij->i", moved_normals, target_normals[target_indices])
        )
        for source_index in np.flatnonzero(
            (distances <= max_distance) & (cosine >= cosine_limit)
        ):
            target_index = int(target_indices[source_index])
            if int(reverse_indices[target_index]) == int(source_index):
                unique_pairs.add((int(source_index), target_index))
    if not unique_pairs:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ordered = sorted(unique_pairs)
    return (
        np.fromiter((item[0] for item in ordered), dtype=np.int64),
        np.fromiter((item[1] for item in ordered), dtype=np.int64),
    )


def collect_overlap_correspondences(
    training_trials: Iterable[CalibrationTrial],
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    config: Mapping[str, object],
) -> OverlapCorrespondences:
    source = _points(source_xyz, "source_xyz")
    target = _points(target_xyz, "target_xyz")
    source_indices, target_indices = _collect_pairs(
        list(training_trials),
        source,
        target,
        max_distance=float(config["max_correspondence_m"]),
        normal_angle_deg=float(config["normal_angle_deg"]),
        normal_radius=float(config["normal_radius_m"]),
    )
    return OverlapCorrespondences(source_indices, target_indices)


def fit_overlap_calibration(
    training_trials: Iterable[CalibrationTrial],
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_spectral6: np.ndarray,
    target_spectral6: np.ndarray,
    wavelengths_nm: np.ndarray,
    *,
    config: Mapping[str, object],
) -> SpectralCalibration:
    """仅依据纯几何粗配准后的高置信重叠点拟合source到target响应。"""
    source = _points(source_xyz, "source_xyz")
    target = _points(target_xyz, "target_xyz")
    source_spec = _spectral(source_spectral6, len(source), "source_spectral6")
    target_spec = _spectral(target_spectral6, len(target), "target_spectral6")
    wavelengths = np.asarray(wavelengths_nm, dtype=np.int32)
    if not np.array_equal(wavelengths, WAVELENGTHS_NM):
        raise ValueError(f"波长顺序必须为{WAVELENGTHS_NM.tolist()}")

    pairs = collect_overlap_correspondences(
        list(training_trials),
        source,
        target,
        config=config,
    )
    source_indices = pairs.source_indices
    target_indices = pairs.target_indices
    minimum = int(config["min_pairs"])
    if len(source_indices) < minimum:
        raise ValueError(
            f"校正对应点至少需要{minimum}对，实际只有{len(source_indices)}对"
        )

    x = source_spec[source_indices]
    y = target_spec[target_indices]
    gains = np.empty(6, dtype=np.float64)
    offsets = np.empty(6, dtype=np.float64)
    residual_mad = np.empty(6, dtype=np.float64)
    correlations = np.empty(6, dtype=np.float64)
    for band in range(6):
        design = np.column_stack((x[:, band], np.ones(len(x))))
        initial, *_ = np.linalg.lstsq(design, y[:, band], rcond=None)
        result = least_squares(
            lambda parameters: (
                parameters[0] * x[:, band]
                + parameters[1]
                - y[:, band]
            ),
            x0=np.array(
                [
                    np.clip(
                        initial[0],
                        float(config["gain_min"]),
                        float(config["gain_max"]),
                    ),
                    initial[1],
                ]
            ),
            bounds=(
                [float(config["gain_min"]), -np.inf],
                [float(config["gain_max"]), np.inf],
            ),
            loss="soft_l1",
            f_scale=float(config["soft_l1_scale"]),
        )
        gains[band], offsets[band] = result.x
        residual = gains[band] * x[:, band] + offsets[band] - y[:, band]
        center = np.median(residual)
        residual_mad[band] = max(
            1.4826 * np.median(np.abs(residual - center)),
            1e-6,
        )
        correlation = np.corrcoef(x[:, band], y[:, band])[0, 1]
        correlations[band] = correlation if np.isfinite(correlation) else 0.0

    channel_correlation = np.nan_to_num(
        np.abs(np.corrcoef(y, rowvar=False)),
        nan=0.0,
    )
    redundancy = (channel_correlation.sum(axis=1) - 1.0) / 5.0
    raw_weights = (
        np.maximum(correlations, 0.0) ** 2
        / (residual_mad**2 + 1e-12)
        / (1.0 + redundancy)
    )
    band_weights = _bounded_simplex(
        raw_weights,
        float(config["weight_floor"]),
        float(config["weight_ceiling"]),
    )
    return SpectralCalibration(
        wavelengths_nm=wavelengths.copy(),
        gain=gains,
        offset=offsets,
        residual_mad=residual_mad,
        correlation=correlations,
        band_weights=band_weights,
        pair_count=len(source_indices),
    )


def apply_source_calibration(
    source_spectral6: np.ndarray,
    calibration: SpectralCalibration,
) -> np.ndarray:
    source = np.asarray(source_spectral6, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 6:
        raise ValueError("source_spectral6必须是N×6数组")
    if not np.isfinite(source).all():
        raise ValueError("source_spectral6包含NaN或Inf")
    return source * calibration.gain + calibration.offset
