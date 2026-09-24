"""Numerical primitives for non-destructive cloud and photo registration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.optimize import least_squares
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    sobel,
)
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image


@dataclass(frozen=True)
class RigidRegistrationResult:
    transform: np.ndarray
    initial_metrics: dict[str, float]
    final_metrics: dict[str, float]
    correspondence_count: int
    accepted: bool
    reason: str


@dataclass(frozen=True)
class CameraModel:
    rotation_vector: np.ndarray
    translation_mm: np.ndarray
    focal_px: float
    principal_point_px: np.ndarray
    radial_k1: float = 0.0


@dataclass(frozen=True)
class CameraFitResult:
    camera: CameraModel
    reprojection_median_px: float
    reprojection_rmse_px: float
    positive_depth_fraction: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class CameraRefinementResult:
    camera: CameraModel
    initial_control_median_px: float
    final_control_median_px: float
    positive_depth_fraction: float
    history: tuple[dict[str, float], ...]
    accepted: bool
    reason: str


def _points(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (N, 3) with N > 0")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _transform(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite 4 by 4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("transform must have homogeneous final row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("transform rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("transform rotation must have determinant +1")
    return matrix


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a right-to-left style homogeneous rigid transform."""
    points = _points(points, "points")
    transform = _transform(transform)
    return points @ transform[:3, :3].T + transform[:3, 3]


def symmetric_nearest_metrics(
    source: np.ndarray,
    target: np.ndarray,
    max_distance_mm: float | None = None,
) -> dict[str, float]:
    """Measure bidirectional nearest-neighbour distances without pairing bias."""
    source = _points(source, "source")
    target = _points(target, "target")
    forward = cKDTree(target).query(source, k=1)[0]
    reverse = cKDTree(source).query(target, k=1)[0]
    distances = np.concatenate((forward, reverse))
    if max_distance_mm is not None:
        maximum = float(max_distance_mm)
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("max_distance_mm must be finite and positive")
        selected = distances <= maximum
        measured = distances[selected]
    else:
        maximum = np.inf
        selected = np.ones_like(distances, dtype=bool)
        measured = distances
    if measured.size == 0:
        median = rmse = p75 = p90 = float("inf")
    else:
        median = float(np.median(measured))
        rmse = float(np.sqrt(np.mean(np.square(measured))))
        p75 = float(np.percentile(measured, 75))
        p90 = float(np.percentile(measured, 90))
    return {
        "symmetric_median_mm": median,
        "symmetric_rmse_mm": rmse,
        "symmetric_p75_mm": p75,
        "symmetric_p90_mm": p90,
        "measured_pair_count": int(np.count_nonzero(selected)),
        "measured_fraction": float(np.mean(selected)),
        "source_count": int(source.shape[0]),
        "target_count": int(target.shape[0]),
    }


def _cloud(points: np.ndarray, voxel_size: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(float(voxel_size))
    radius = max(float(voxel_size) * 2.5, 1.0)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    return cloud


def _rotation_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _mutual_correspondence_count(
    source_aligned: np.ndarray, target: np.ndarray, maximum: float
) -> int:
    target_tree = cKDTree(target)
    source_tree = cKDTree(source_aligned)
    distances, target_rows = target_tree.query(source_aligned, k=1)
    _, source_rows = source_tree.query(target, k=1)
    rows = np.arange(source_aligned.shape[0])
    mutual = source_rows[target_rows] == rows
    return int(np.count_nonzero(mutual & (distances <= maximum)))


def register_right_to_left(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
    voxel_sizes_mm: tuple[float, ...] = (150.0, 75.0, 40.0),
    max_distances_mm: tuple[float, ...] = (450.0, 225.0, 120.0),
) -> RigidRegistrationResult:
    """Align a partially overlapping right scan to the left reference scan."""
    left = _points(left_xyz, "left_xyz")
    right = _points(right_xyz, "right_xyz")
    if len(voxel_sizes_mm) != len(max_distances_mm) or not voxel_sizes_mm:
        raise ValueError("voxel and correspondence scales must have equal non-zero length")
    if any(value <= 0 or not np.isfinite(value) for value in (*voxel_sizes_mm, *max_distances_mm)):
        raise ValueError("registration scales must be finite and positive")

    evaluation_distance = float(max_distances_mm[0])
    initial_metrics = symmetric_nearest_metrics(right, left, evaluation_distance)
    current = np.eye(4, dtype=np.float64)
    for voxel_size, max_distance in zip(voxel_sizes_mm, max_distances_mm):
        source = _cloud(right, float(voxel_size))
        target = _cloud(left, float(voxel_size))
        loss = o3d.pipelines.registration.TukeyLoss(k=float(max_distance) * 0.75)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            float(max_distance),
            current,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-7,
                relative_rmse=1e-7,
                max_iteration=80,
            ),
        )
        candidate = np.asarray(result.transformation, dtype=np.float64)
        if np.all(np.isfinite(candidate)):
            current = candidate

    try:
        current = _transform(current).copy()
    except ValueError as error:
        return RigidRegistrationResult(
            np.eye(4), initial_metrics, initial_metrics, 0, False, str(error)
        )
    aligned = apply_transform(right, current)
    final_metrics = symmetric_nearest_metrics(aligned, left, evaluation_distance)
    correspondence_count = _mutual_correspondence_count(
        aligned, left, float(max_distances_mm[-1])
    )
    translation_mm = float(np.linalg.norm(current[:3, 3]))
    rotation_degrees = _rotation_degrees(current[:3, :3])
    reasons = []
    if correspondence_count < 50:
        reasons.append(f"only {correspondence_count} mutual overlap correspondences")
    if not (
        np.isfinite(final_metrics["symmetric_median_mm"])
        and final_metrics["symmetric_median_mm"]
        < initial_metrics["symmetric_median_mm"]
    ):
        reasons.append("held-out symmetric median error did not improve")
    if translation_mm > 1000.0:
        reasons.append(f"translation {translation_mm:.3f} mm exceeds 1000 mm")
    if rotation_degrees > 10.0:
        reasons.append(f"rotation {rotation_degrees:.3f} degrees exceeds 10 degrees")
    accepted = not reasons
    return RigidRegistrationResult(
        transform=current if accepted else np.eye(4),
        initial_metrics=initial_metrics,
        final_metrics=final_metrics,
        correspondence_count=correspondence_count,
        accepted=accepted,
        reason="accepted" if accepted else "; ".join(reasons),
    )


def _camera(camera: CameraModel) -> CameraModel:
    rotation = np.asarray(camera.rotation_vector, dtype=np.float64)
    translation = np.asarray(camera.translation_mm, dtype=np.float64)
    principal = np.asarray(camera.principal_point_px, dtype=np.float64)
    if rotation.shape != (3,) or translation.shape != (3,) or principal.shape != (2,):
        raise ValueError("camera vectors have invalid shape")
    scalars = np.array([camera.focal_px, camera.radial_k1], dtype=np.float64)
    if not np.all(np.isfinite(np.concatenate((rotation, translation, principal, scalars)))):
        raise ValueError("camera parameters must be finite")
    if camera.focal_px <= 0.0:
        raise ValueError("camera focal_px must be positive")
    return CameraModel(
        rotation.copy(),
        translation.copy(),
        float(camera.focal_px),
        principal.copy(),
        float(camera.radial_k1),
    )


def project_camera(
    points_xyz: np.ndarray, camera: CameraModel
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points with a point-cloud-to-camera transform."""
    points = _points(points_xyz, "points_xyz")
    camera = _camera(camera)
    rotation = Rotation.from_rotvec(camera.rotation_vector).as_matrix()
    camera_xyz = points @ rotation.T + camera.translation_mm
    depth = camera_xyz[:, 2]
    safe_depth = np.where(np.abs(depth) < 1e-12, np.copysign(1e-12, depth + 1e-12), depth)
    normalized = camera_xyz[:, :2] / safe_depth[:, None]
    radius_squared = np.sum(np.square(normalized), axis=1)
    normalized *= (1.0 + camera.radial_k1 * radius_squared)[:, None]
    pixels = normalized * camera.focal_px + camera.principal_point_px
    return pixels, depth


def lift_angular_controls(
    azimuth: np.ndarray,
    elevation: np.ndarray,
    xyz: np.ndarray,
    scan_control_uv: np.ndarray,
) -> np.ndarray:
    """Lift normalized angular controls to their nearest measured XYZ rows."""
    azimuth = np.asarray(azimuth, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=np.float64)
    xyz = _points(xyz, "xyz")
    controls = np.asarray(scan_control_uv, dtype=np.float64)
    if azimuth.shape != elevation.shape or azimuth.ndim != 1 or len(azimuth) != len(xyz):
        raise ValueError("azimuth, elevation, and xyz rows must match")
    if controls.ndim != 2 or controls.shape[1] != 2 or controls.shape[0] == 0:
        raise ValueError("scan_control_uv must have shape (N, 2)")
    if not np.all(np.isfinite(np.concatenate((azimuth, elevation, controls.ravel())))):
        raise ValueError("angular inputs must be finite")
    spans = np.array([np.ptp(azimuth), np.ptp(elevation)])
    if np.any(spans <= 0.0):
        raise ValueError("angular inputs must span both dimensions")
    normalized = np.column_stack(
        ((azimuth - azimuth.min()) / spans[0],
         (elevation - elevation.min()) / spans[1])
    )
    rows = cKDTree(normalized).query(controls, k=1)[1]
    return xyz[np.asarray(rows, dtype=np.int64)].copy()


def _normalized_dlt(points_3d: np.ndarray, pixels_2d: np.ndarray) -> np.ndarray:
    center_3d = points_3d.mean(axis=0)
    scale_3d = np.sqrt(3.0) / np.mean(np.linalg.norm(points_3d - center_3d, axis=1))
    world_normalizer = np.eye(4)
    world_normalizer[:3, :3] *= scale_3d
    world_normalizer[:3, 3] = -scale_3d * center_3d
    center_2d = pixels_2d.mean(axis=0)
    scale_2d = np.sqrt(2.0) / np.mean(np.linalg.norm(pixels_2d - center_2d, axis=1))
    image_normalizer = np.array(
        [[scale_2d, 0.0, -scale_2d * center_2d[0]],
         [0.0, scale_2d, -scale_2d * center_2d[1]],
         [0.0, 0.0, 1.0]]
    )
    world_h = np.column_stack((points_3d, np.ones(len(points_3d))))
    world_n = world_h @ world_normalizer.T
    image_h = np.column_stack((pixels_2d, np.ones(len(pixels_2d))))
    image_n = image_h @ image_normalizer.T
    image_n = image_n[:, :2] / image_n[:, 2, None]
    rows = []
    for point, (u, v) in zip(world_n, image_n):
        zeros = np.zeros(4)
        rows.append(np.concatenate((point, zeros, -u * point)))
        rows.append(np.concatenate((zeros, point, -v * point)))
    design = np.asarray(rows)
    if np.linalg.matrix_rank(design) < 11:
        raise ValueError("camera controls are degenerate or coplanar")
    _, _, vh = np.linalg.svd(design)
    normalized_projection = vh[-1].reshape(3, 4)
    return np.linalg.inv(image_normalizer) @ normalized_projection @ world_normalizer


def _camera_residual(
    parameters: np.ndarray,
    points: np.ndarray,
    pixels: np.ndarray,
    principal: np.ndarray,
) -> np.ndarray:
    camera = CameraModel(
        rotation_vector=parameters[:3],
        translation_mm=parameters[3:6],
        focal_px=float(np.exp(parameters[6])),
        principal_point_px=principal,
    )
    projected, depth = project_camera(points, camera)
    residual = projected - pixels
    behind = depth <= 1.0
    if np.any(behind):
        residual[behind] = np.sign(residual[behind] + 1e-9) * (
            5000.0 + np.maximum(0.0, 1.0 - depth[behind, None])
        )
    return residual.ravel()


def initialize_camera(
    points_3d: np.ndarray,
    pixels_2d: np.ndarray,
    image_size: tuple[int, int],
    focal_bounds_px: tuple[float, float] | None = None,
) -> CameraFitResult:
    """Estimate handheld-camera pose and equal pixel focal length without OpenCV."""
    points = _points(points_3d, "points_3d")
    pixels = np.asarray(pixels_2d, dtype=np.float64)
    if pixels.shape != (len(points), 2) or not np.all(np.isfinite(pixels)):
        raise ValueError("pixels_2d must have shape (N, 2) and be finite")
    if len(points) < 6:
        raise ValueError("at least six 3D to 2D controls are required")
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")
    if focal_bounds_px is None:
        focal_bounds_px = (0.5 * width, 4.0 * width)
    low_focal, high_focal = map(float, focal_bounds_px)
    if not (0.0 < low_focal < high_focal and np.isfinite(high_focal)):
        raise ValueError("focal bounds must be finite, positive, and increasing")
    principal = np.array([width / 2.0, height / 2.0])
    projection = _normalized_dlt(points, pixels)
    focal_candidates = np.geomspace(low_focal, high_focal, 9)
    seeds = []
    for sign in (1.0, -1.0):
        signed = projection * sign
        for focal in focal_candidates:
            intrinsic = np.array(
                [[focal, 0.0, principal[0]], [0.0, focal, principal[1]], [0.0, 0.0, 1.0]]
            )
            normalized_projection = np.linalg.inv(intrinsic) @ signed
            approximate_rotation = normalized_projection[:, :3]
            scale = float(np.mean(np.linalg.norm(approximate_rotation, axis=1)))
            if not np.isfinite(scale) or scale <= 1e-12:
                continue
            u, _, vh = np.linalg.svd(approximate_rotation / scale)
            rotation = u @ vh
            if np.linalg.det(rotation) < 0.0:
                u[:, -1] *= -1.0
                rotation = u @ vh
            translation = normalized_projection[:, 3] / scale
            parameters = np.concatenate(
                (Rotation.from_matrix(rotation).as_rotvec(), translation, [np.log(focal)])
            )
            residual = _camera_residual(parameters, points, pixels, principal)
            seeds.append((float(np.median(np.abs(residual))), parameters))
    if not seeds:
        raise ValueError("camera initialization produced no finite candidates")
    seeds.sort(key=lambda item: item[0])
    point_scale = max(float(np.ptp(points, axis=0).max()), 1000.0)
    lower = np.array([-np.pi, -np.pi, -np.pi, -20 * point_scale, -20 * point_scale, -20 * point_scale, np.log(low_focal)])
    upper = np.array([np.pi, np.pi, np.pi, 20 * point_scale, 20 * point_scale, 20 * point_scale, np.log(high_focal)])
    solutions = []
    for _, seed in seeds[:4]:
        seed = np.clip(seed, lower + 1e-9, upper - 1e-9)
        result = least_squares(
            _camera_residual,
            seed,
            bounds=(lower, upper),
            args=(points, pixels, principal),
            loss="soft_l1",
            f_scale=3.0,
            max_nfev=3000,
        )
        camera = CameraModel(
            result.x[:3], result.x[3:6], float(np.exp(result.x[6])), principal
        )
        projected, depth = project_camera(points, camera)
        errors = np.linalg.norm(projected - pixels, axis=1)
        positive = float(np.mean(depth > 1.0))
        score = float(np.median(errors)) + 1000.0 * max(0.0, 0.8 - positive)
        solutions.append((score, camera, errors, positive))
    _, camera, errors, positive = min(solutions, key=lambda item: item[0])
    median = float(np.median(errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    reasons = []
    if positive < 0.8:
        reasons.append(f"positive depth fraction {positive:.3f} is below 0.8")
    if not np.isfinite(median) or median > 30.0:
        reasons.append(f"median reprojection error {median:.3f} px exceeds 30 px")
    return CameraFitResult(
        camera=camera,
        reprojection_median_px=median,
        reprojection_rmse_px=rmse,
        positive_depth_fraction=positive,
        accepted=not reasons,
        reason="accepted" if not reasons else "; ".join(reasons),
    )


def zbuffer_visible(
    projected_xy: np.ndarray,
    depth: np.ndarray,
    image_size: tuple[int, int],
    depth_tolerance_mm: float,
) -> np.ndarray:
    """Return points visible under a nearest-depth pixel Z-buffer."""
    projected = np.asarray(projected_xy, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if projected.ndim != 2 or projected.shape[1] != 2 or depth.shape != (len(projected),):
        raise ValueError("projected_xy and depth shapes must match")
    width, height = int(image_size[0]), int(image_size[1])
    tolerance = float(depth_tolerance_mm)
    if width <= 0 or height <= 0 or tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("image size and depth tolerance are invalid")
    rounded = np.rint(projected).astype(np.int64)
    valid = (
        np.all(np.isfinite(projected), axis=1)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    flat_depth = np.full(width * height, np.inf, dtype=np.float64)
    flat_index = rounded[:, 1] * width + rounded[:, 0]
    np.minimum.at(flat_depth, flat_index[valid], depth[valid])
    visible = np.zeros(len(projected), dtype=bool)
    visible[valid] = depth[valid] <= flat_depth[flat_index[valid]] + tolerance
    return visible


def _resized_photo(photo_srgb: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    photo = np.asarray(photo_srgb, dtype=np.float64)
    if photo.ndim != 3 or photo.shape[2] != 3 or not np.all(np.isfinite(photo)):
        raise ValueError("photo_srgb must be a finite H by W by 3 array")
    if np.any((photo < 0.0) | (photo > 1.0)):
        raise ValueError("photo_srgb must lie in [0, 1]")
    width, height = int(output_size[0]), int(output_size[1])
    if width <= 0 or height <= 0:
        raise ValueError("output_size must be positive")
    image = Image.fromarray(np.rint(photo * 255.0).astype(np.uint8), mode="RGB")
    resized = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float64) / 255.0


def photo_edge_distance(
    photo_srgb: np.ndarray, output_size: tuple[int, int]
) -> np.ndarray:
    """Return Euclidean pixel distance to stable luminance or colour edges."""
    photo = _resized_photo(photo_srgb, output_size)
    luminance = photo @ np.array([0.2126, 0.7152, 0.0722])
    gradients = [np.hypot(sobel(luminance, axis=1), sobel(luminance, axis=0))]
    for channel in range(3):
        gradients.append(
            np.hypot(sobel(photo[:, :, channel], axis=1), sobel(photo[:, :, channel], axis=0))
        )
    strength = np.maximum.reduce(gradients)
    positive = strength[strength > 0.0]
    if positive.size == 0:
        return np.full(strength.shape, float(max(strength.shape)), dtype=np.float64)
    threshold = max(float(np.percentile(positive, 70.0)), 0.03)
    edges = strength >= threshold
    edges = binary_dilation(edges, iterations=1)
    return distance_transform_edt(~edges).astype(np.float64)


def render_cloud_edge_mask(
    points_xyz: np.ndarray,
    camera: CameraModel,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Render a sparse occupancy/depth boundary mask for diagnostics."""
    points = _points(points_xyz, "points_xyz")
    width, height = int(image_size[0]), int(image_size[1])
    projected, depth = project_camera(points, camera)
    visible = zbuffer_visible(projected, depth, (width, height), 30.0)
    rounded = np.rint(projected[visible]).astype(np.int64)
    occupancy = np.zeros((height, width), dtype=bool)
    occupancy[rounded[:, 1], rounded[:, 0]] = True
    occupancy = binary_dilation(occupancy, iterations=1)
    return occupancy & ~binary_erosion(occupancy, iterations=1)


def _bilinear_distance(distance: np.ndarray, xy: np.ndarray, outside: float) -> np.ndarray:
    height, width = distance.shape
    x = xy[:, 0]
    y = xy[:, 1]
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0.0) & (x < width - 1) & (y >= 0.0) & (y < height - 1)
    values = np.full(len(xy), float(outside), dtype=np.float64)
    if not np.any(valid):
        return values
    xv, yv = x[valid], y[valid]
    x0, y0 = np.floor(xv).astype(int), np.floor(yv).astype(int)
    dx, dy = xv - x0, yv - y0
    values[valid] = (
        distance[y0, x0] * (1.0 - dx) * (1.0 - dy)
        + distance[y0, x0 + 1] * dx * (1.0 - dy)
        + distance[y0 + 1, x0] * (1.0 - dx) * dy
        + distance[y0 + 1, x0 + 1] * dx * dy
    )
    return values


def refine_camera_multiscale(
    points_xyz: np.ndarray,
    photo_srgb: np.ndarray,
    controls_3d: np.ndarray,
    controls_2d: np.ndarray,
    initial_camera: CameraModel,
    pyramid_widths: tuple[int, ...] = (500, 1000, 2000, 4000),
) -> CameraRefinementResult:
    """Refine pose/focal length using robust controls plus continuous edge distance."""
    points = _points(points_xyz, "points_xyz")
    controls_3d = _points(controls_3d, "controls_3d")
    controls_2d = np.asarray(controls_2d, dtype=np.float64)
    if controls_2d.shape != (len(controls_3d), 2) or len(controls_3d) < 6:
        raise ValueError("at least six matching 3D and 2D controls are required")
    camera = _camera(initial_camera)
    photo = np.asarray(photo_srgb, dtype=np.float64)
    original_height, original_width = photo.shape[:2]
    initial_pixels, initial_depth = project_camera(controls_3d, camera)
    initial_median = float(np.median(np.linalg.norm(initial_pixels - controls_2d, axis=1)))
    sample_rows = np.linspace(0, len(points) - 1, min(len(points), 1500), dtype=int)
    sample_points = points[sample_rows]
    initial_parameters = np.concatenate(
        (camera.rotation_vector, camera.translation_mm, [np.log(camera.focal_px)])
    )
    current = initial_parameters.copy()
    rotation_margin = np.radians(8.0)
    lower = np.concatenate((initial_parameters[:3] - rotation_margin, initial_parameters[3:6] - 1500.0, [initial_parameters[6] - np.log(2.0)]))
    upper = np.concatenate((initial_parameters[:3] + rotation_margin, initial_parameters[3:6] + 1500.0, [initial_parameters[6] + np.log(2.0)]))
    history = []

    def make_camera(parameters: np.ndarray) -> CameraModel:
        return CameraModel(parameters[:3], parameters[3:6], float(np.exp(parameters[6])), camera.principal_point_px)

    for width in pyramid_widths:
        width = min(int(width), original_width)
        height = max(1, int(round(original_height * width / original_width)))
        distance = photo_edge_distance(photo, (width, height))
        scale = width / original_width

        def residual(parameters: np.ndarray) -> np.ndarray:
            candidate = make_camera(parameters)
            control_pixels, control_depth = project_camera(controls_3d, candidate)
            control_residual = (control_pixels - controls_2d).ravel() / 3.0
            if np.any(control_depth <= 1.0):
                control_residual = control_residual + 500.0 * np.repeat(control_depth <= 1.0, 2)
            projected, depth = project_camera(sample_points, candidate)
            edge_residual = _bilinear_distance(distance, projected * scale, 50.0)
            edge_residual[depth <= 1.0] = 50.0
            edge_residual = np.clip(edge_residual / max(scale, 1e-6), 0.0, 100.0) * 0.03
            pose_prior = np.concatenate(
                ((parameters[:3] - initial_parameters[:3]) / 0.15,
                 (parameters[3:6] - initial_parameters[3:6]) / 750.0,
                 [(parameters[6] - initial_parameters[6]) / 0.35])
            ) * 0.1
            return np.concatenate((control_residual, edge_residual, pose_prior))

        before_camera = make_camera(current)
        before_pixels, _ = project_camera(controls_3d, before_camera)
        before_median = float(np.median(np.linalg.norm(before_pixels - controls_2d, axis=1)))
        result = least_squares(
            residual,
            np.clip(current, lower + 1e-9, upper - 1e-9),
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=2.0,
            max_nfev=300,
        )
        candidate = make_camera(result.x)
        candidate_pixels, candidate_depth = project_camera(controls_3d, candidate)
        candidate_median = float(np.median(np.linalg.norm(candidate_pixels - controls_2d, axis=1)))
        positive = float(np.mean(candidate_depth > 1.0))
        accepted_level = positive >= 0.8 and candidate_median <= before_median + 0.25
        if accepted_level:
            current = result.x
        history.append(
            {
                "width": float(width),
                "control_median_before_px": before_median,
                "control_median_candidate_px": candidate_median,
                "positive_depth_fraction": positive,
                "cost": float(result.cost),
                "accepted": float(accepted_level),
            }
        )
        if width == original_width:
            break
    final_camera = make_camera(current)
    final_pixels, final_depth = project_camera(controls_3d, final_camera)
    final_median = float(np.median(np.linalg.norm(final_pixels - controls_2d, axis=1)))
    positive = float(np.mean(final_depth > 1.0))
    reasons = []
    if positive < 0.8:
        reasons.append(f"positive depth fraction {positive:.3f} is below 0.8")
    if final_median > max(30.0, initial_median + 0.5):
        reasons.append("control reprojection error worsened")
    if not np.all(np.isfinite(current)):
        reasons.append("refined camera contains non-finite values")
    return CameraRefinementResult(
        camera=final_camera,
        initial_control_median_px=initial_median,
        final_control_median_px=final_median,
        positive_depth_fraction=positive,
        history=tuple(history),
        accepted=not reasons,
        reason="accepted" if not reasons else "; ".join(reasons),
    )
