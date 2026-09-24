"""Core calibration constants and numerical color helpers for six-band scans."""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms
from scipy.optimize import differential_evolution, minimize


WAVELENGTHS_NM = np.array([495, 696, 600, 803, 545, 642], dtype=np.int32)
SOURCE_WHITE = np.array(
    [
        124.1064096774194,
        120.2428838709678,
        115.0127903225807,
        166.3828838709677,
        115.8136612903226,
        226.6485516129032,
    ],
    dtype=np.float64,
)
CURRENT_WHITE = np.array(
    [531.2975, 396.4109, 425.0606, 361.2250, 572.4702, 477.3115],
    dtype=np.float64,
)
VISIBLE_INDICES = (0, 1, 2, 4, 5)
CIE_1931_2DEG_XYZ = np.array(
    [
        [0.014700000000, 0.2586000000000, 0.353300000000],
        [0.014790640000, 0.0053430590000, 0.000000000000],
        [1.062200000000, 0.6310000000000, 0.000800000000],
        [0.000008308358, 0.0000030003000, 0.000000000000],
        [0.359700000000, 0.9803000000000, 0.013400000000],
        [0.412098000000, 0.1596464000000, 0.000016200000],
    ],
    dtype=np.float64,
)
XYZ_TO_LINEAR_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ],
    dtype=np.float64,
)
LINEAR_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)
D65_WHITE_XYZ = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
ANCHOR_CLASS_NAMES = ("white", "red", "green", "brown", "black")


@dataclass(frozen=True)
class CalibrationSamples:
    amplitudes: np.ndarray
    target_linear_rgb: np.ndarray
    class_ids: np.ndarray


@dataclass(frozen=True)
class WhiteFitResult:
    white: np.ndarray
    exposure: float
    objective: float
    class_metrics: dict[str, float]


@dataclass(frozen=True)
class MatrixFitResult:
    matrix: np.ndarray
    bias: np.ndarray
    ridge_coefficient: float
    validation_error: float


def restore_amplitudes(reflectance: np.ndarray) -> np.ndarray:
    reflectance = np.asarray(reflectance, dtype=np.float64)
    if reflectance.shape[-1] != 6:
        raise ValueError("reflectance must end with six wavelength channels")
    return reflectance * SOURCE_WHITE


def reflectance_from_white(amplitudes: np.ndarray, white: np.ndarray) -> np.ndarray:
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    white = np.asarray(white, dtype=np.float64)
    if white.shape != (6,) or np.any(~np.isfinite(white)) or np.any(white <= 0):
        raise ValueError("white must contain six finite positive values")
    return amplitudes / white


def spectral_to_linear_srgb(reflectance: np.ndarray) -> np.ndarray:
    reflectance = np.asarray(reflectance, dtype=np.float64)
    visible = np.array(VISIBLE_INDICES, dtype=np.int32)
    normalization = CIE_1931_2DEG_XYZ[visible, 1].sum()
    xyz = reflectance[:, visible] @ CIE_1931_2DEG_XYZ[visible] / normalization
    return np.maximum(xyz @ XYZ_TO_LINEAR_SRGB.T, 0.0)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.maximum(np.asarray(rgb, dtype=np.float64), 0.0)
    encoded = np.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )
    return np.clip(encoded, 0.0, 1.0)


def linear_srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = np.asarray(rgb, dtype=np.float64) @ LINEAR_SRGB_TO_XYZ.T
    ratio = xyz / D65_WHITE_XYZ
    delta = 6.0 / 29.0
    f = np.where(
        ratio > delta**3,
        np.cbrt(ratio),
        ratio / (3.0 * delta**2) + 4.0 / 29.0,
    )
    return np.column_stack(
        (
            116.0 * f[:, 1] - 16.0,
            500.0 * (f[:, 0] - f[:, 1]),
            200.0 * (f[:, 1] - f[:, 2]),
        )
    )


def delta_e76(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64),
        axis=-1,
    )


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        np.power((rgb + 0.055) / 1.055, 2.4),
    )


def _anchor_class_ids(lab: np.ndarray) -> np.ndarray:
    lightness, a_axis, b_axis = lab.T
    chroma = np.hypot(a_axis, b_axis)
    conditions = (
        (lightness >= 70.0) & (chroma <= 12.0),
        (a_axis >= 25.0) & (a_axis >= np.abs(b_axis) * 0.7),
        (a_axis <= -12.0) & (b_axis >= 5.0),
        (lightness >= 15.0)
        & (lightness <= 65.0)
        & (a_axis >= 8.0)
        & (a_axis <= 35.0)
        & (b_axis >= 8.0)
        & (b_axis <= 40.0),
        lightness <= 25.0,
    )
    class_ids = np.full(lab.shape[0], -1, dtype=np.int32)
    for class_id, condition in enumerate(conditions):
        class_ids[(class_ids < 0) & condition] = class_id
    return class_ids


def build_anchor_samples(
    amplitudes: np.ndarray,
    projected_xy: np.ndarray,
    photo_rgb: np.ndarray,
) -> CalibrationSamples:
    """Extract deterministic, texture-filtered, class-balanced photo anchors."""
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    projected_xy = np.asarray(projected_xy, dtype=np.float64)
    photo_rgb = np.asarray(photo_rgb, dtype=np.float64)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != 6:
        raise ValueError("amplitudes must have shape (N, 6)")
    if projected_xy.shape != (amplitudes.shape[0], 2):
        raise ValueError("projected_xy must have shape (N, 2)")
    if photo_rgb.ndim != 3 or photo_rgb.shape[2] != 3:
        raise ValueError("photo_rgb must have shape (height, width, 3)")
    height, width = photo_rgb.shape[:2]
    if height < 9 or width < 9:
        raise ValueError("photo_rgb must be at least 9 by 9 pixels")

    valid = np.all(np.isfinite(amplitudes) & (amplitudes >= 0.0), axis=1)
    valid &= np.all(np.isfinite(projected_xy), axis=1)
    x = projected_xy[:, 0]
    y = projected_xy[:, 1]
    valid &= (x >= 4.0) & (x < width - 4.0)
    valid &= (y >= 4.0) & (y < height - 4.0)
    candidate_rows = np.flatnonzero(valid)

    accepted_rows = []
    sampled_srgb = []
    for row in candidate_rows:
        x0 = int(np.floor(x[row]))
        y0 = int(np.floor(y[row]))
        x1 = x0 + 1
        y1 = y0 + 1
        patch = photo_rgb[y0 - 2 : y0 + 3, x0 - 2 : x0 + 3]
        if patch.shape != (5, 5, 3) or not np.all(np.isfinite(patch)):
            continue
        if np.max(np.std(patch, axis=(0, 1))) > 0.08:
            continue
        dx = x[row] - x0
        dy = y[row] - y0
        sampled = (
            photo_rgb[y0, x0] * (1.0 - dx) * (1.0 - dy)
            + photo_rgb[y0, x1] * dx * (1.0 - dy)
            + photo_rgb[y1, x0] * (1.0 - dx) * dy
            + photo_rgb[y1, x1] * dx * dy
        )
        if not np.all(np.isfinite(sampled)):
            continue
        accepted_rows.append(row)
        sampled_srgb.append(sampled)

    if not accepted_rows:
        raise ValueError("no valid photo anchors remain after filtering")
    accepted_rows = np.asarray(accepted_rows, dtype=np.int64)
    target_linear_rgb = _srgb_to_linear(np.asarray(sampled_srgb, dtype=np.float64))
    class_ids = _anchor_class_ids(linear_srgb_to_lab(target_linear_rgb))

    valid_classes = []
    for class_id in range(len(ANCHOR_CLASS_NAMES)):
        rows = np.flatnonzero(class_ids == class_id)
        if rows.size >= 15:
            valid_classes.append((class_id, rows))
    if len(valid_classes) < 3:
        raise ValueError("at least three photo classes with 15 valid samples are required")

    samples_per_class = min(500, *(rows.size for _, rows in valid_classes))
    rng = np.random.default_rng(20260722)
    selected = []
    selected_class_ids = []
    for class_id, rows in valid_classes:
        chosen = rng.choice(rows, size=samples_per_class, replace=False)
        selected.extend(chosen)
        selected_class_ids.extend([class_id] * samples_per_class)
    selected = np.asarray(selected, dtype=np.int64)
    return CalibrationSamples(
        amplitudes=amplitudes[accepted_rows[selected]].copy(),
        target_linear_rgb=target_linear_rgb[selected].copy(),
        class_ids=np.asarray(selected_class_ids, dtype=np.int32),
    )


def _prepare_calibration_samples(
    samples: CalibrationSamples,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(samples, CalibrationSamples):
        raise TypeError("samples must be a CalibrationSamples instance")
    amplitudes = np.asarray(samples.amplitudes, dtype=np.float64)
    target_linear_rgb = np.asarray(samples.target_linear_rgb, dtype=np.float64)
    class_ids = np.asarray(samples.class_ids)
    if amplitudes.ndim != 2 or amplitudes.shape[1] != 6:
        raise ValueError("sample amplitudes must have shape (N, 6)")
    sample_count = amplitudes.shape[0]
    if target_linear_rgb.shape != (sample_count, 3):
        raise ValueError("sample target_linear_rgb must have shape (N, 3)")
    if class_ids.shape != (sample_count,):
        raise ValueError("sample class_ids must have shape (N,)")
    if sample_count == 0:
        raise ValueError("samples must not be empty")
    if not np.all(np.isfinite(amplitudes)) or np.any(amplitudes < 0.0):
        raise ValueError("sample amplitudes must be finite and non-negative")
    if not np.all(np.isfinite(target_linear_rgb)) or np.any(target_linear_rgb < 0.0):
        raise ValueError("sample target_linear_rgb must be finite and non-negative")
    if not np.issubdtype(class_ids.dtype, np.integer):
        if not np.all(np.isfinite(class_ids)) or not np.all(class_ids == np.floor(class_ids)):
            raise ValueError("sample class_ids must be integers")
    class_ids = class_ids.astype(np.int32, copy=False)
    if np.any((class_ids < 0) | (class_ids >= len(ANCHOR_CLASS_NAMES))):
        raise ValueError("sample class_ids must identify the five photo classes")
    if not np.any(class_ids == 0):
        raise ValueError("samples must include white anchors for exposure estimation")
    target_lab = linear_srgb_to_lab(target_linear_rgb)
    return amplitudes, target_linear_rgb, target_lab, class_ids


def _candidate_metrics(
    log_visible_white: np.ndarray,
    prepared_samples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    center_white: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    log_visible_white = np.asarray(log_visible_white, dtype=np.float64)
    center_white = np.asarray(center_white, dtype=np.float64)
    if log_visible_white.shape != (len(VISIBLE_INDICES),):
        raise ValueError("log_visible_white must contain five values")
    if not np.all(np.isfinite(log_visible_white)):
        raise ValueError("log_visible_white must contain only finite values")
    if center_white.shape != (6,) or not np.all(np.isfinite(center_white)) or np.any(center_white <= 0.0):
        raise ValueError("center_white must contain six finite positive values")

    amplitudes, target_linear_rgb, target_lab, class_ids = prepared_samples
    visible = np.asarray(VISIBLE_INDICES, dtype=np.int32)
    white = CURRENT_WHITE.copy()
    white[visible] = np.exp(log_visible_white)
    predicted_linear_rgb = spectral_to_linear_srgb(
        reflectance_from_white(amplitudes, white)
    )
    luminance_weights = LINEAR_SRGB_TO_XYZ[1]
    white_anchors = class_ids == 0
    target_luminance = np.median(
        target_linear_rgb[white_anchors] @ luminance_weights
    )
    predicted_luminance = np.median(
        predicted_linear_rgb[white_anchors] @ luminance_weights
    )
    if not np.isfinite(target_luminance) or target_luminance <= 0.0:
        raise ValueError("white anchors must have positive target luminance")
    if not np.isfinite(predicted_luminance) or predicted_luminance <= 0.0:
        return np.inf, np.nan, {}
    exposure = float(target_luminance / predicted_luminance)
    predicted_lab = linear_srgb_to_lab(predicted_linear_rgb * exposure)
    color_errors = delta_e76(predicted_lab, target_lab)
    huber_delta = 10.0
    huber_losses = np.where(
        color_errors <= huber_delta,
        0.5 * color_errors**2,
        huber_delta * (color_errors - 0.5 * huber_delta),
    )

    class_losses = []
    class_metrics = {}
    for class_id in np.unique(class_ids):
        in_class = class_ids == class_id
        class_losses.append(float(np.median(huber_losses[in_class])))
        class_metrics[ANCHOR_CLASS_NAMES[class_id]] = float(
            np.median(color_errors[in_class])
        )
    class_loss = float(np.median(class_losses))
    log_center = np.log(center_white[visible])
    regularization = 0.15 * float(np.mean((log_visible_white - log_center) ** 2))
    return class_loss + regularization, exposure, class_metrics


def white_objective(
    log_visible_white: np.ndarray,
    samples: CalibrationSamples,
    center_white: np.ndarray,
) -> float:
    """Return the robust class-balanced color objective for visible white values."""
    prepared_samples = _prepare_calibration_samples(samples)
    objective, _, _ = _candidate_metrics(
        log_visible_white, prepared_samples, center_white
    )
    return float(objective)


def optimize_white(samples: CalibrationSamples) -> WhiteFitResult:
    """Fit visible white-channel ratios while keeping the 803 nm value fixed."""
    prepared_samples = _prepare_calibration_samples(samples)
    visible = np.asarray(VISIBLE_INDICES, dtype=np.int32)
    lower = np.log(CURRENT_WHITE[visible] * 0.15)
    upper = np.log(CURRENT_WHITE[visible] * 5.0)
    bounds = list(zip(lower, upper))

    def objective(log_white: np.ndarray) -> float:
        value, _, _ = _candidate_metrics(log_white, prepared_samples, CURRENT_WHITE)
        return value

    deterministic_whites = (
        CURRENT_WHITE,
        SOURCE_WHITE,
        np.array([473.05, 366.73, 395.53, 322.5, 522.67, 421.58]),
    )
    deterministic_starts = [
        np.clip(np.log(white[visible]), lower, upper)
        for white in deterministic_whites
    ]
    global_result = differential_evolution(
        objective,
        bounds,
        seed=20260722,
        x0=deterministic_starts[0],
        popsize=12,
        maxiter=160,
        tol=1e-7,
        polish=False,
        workers=1,
        updating="immediate",
    )
    candidate_starts = [global_result.x, *deterministic_starts]
    candidates = []
    for start in candidate_starts:
        refined = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        candidates.append((float(refined.fun), np.asarray(refined.x, dtype=np.float64)))
    candidates.append((float(global_result.fun), np.asarray(global_result.x, dtype=np.float64)))
    _, best_log_white = min(candidates, key=lambda candidate: candidate[0])
    final_objective, exposure, class_metrics = _candidate_metrics(
        best_log_white, prepared_samples, CURRENT_WHITE
    )
    white = CURRENT_WHITE.copy()
    white[visible] = np.exp(best_log_white)
    white[3] = 361.2250
    return WhiteFitResult(
        white=white,
        exposure=float(exposure),
        objective=float(final_objective),
        class_metrics=class_metrics,
    )


def _cie_baseline_matrix() -> np.ndarray:
    visible = np.asarray(VISIBLE_INDICES, dtype=np.int32)
    normalization = CIE_1931_2DEG_XYZ[visible, 1].sum()
    return CIE_1931_2DEG_XYZ[visible] @ XYZ_TO_LINEAR_SRGB.T / normalization


def _fit_ridge_matrix(
    visible_reflectance: np.ndarray,
    target_linear_rgb: np.ndarray,
    ridge_coefficient: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack(
        (visible_reflectance, np.ones(visible_reflectance.shape[0]))
    )
    center = np.vstack((_cie_baseline_matrix(), np.zeros((1, 3))))
    penalty = np.diag(
        np.array([ridge_coefficient] * len(VISIBLE_INDICES) + [0.0])
    )
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target_linear_rgb + penalty @ center,
    )
    return coefficients[:-1], coefficients[-1:]


def fit_regularized_matrix(
    samples: CalibrationSamples, current_white: np.ndarray
) -> MatrixFitResult:
    """Fit a visible-band linear RGB matrix with leave-one-class-out ridge CV."""
    amplitudes, target_linear_rgb, _, class_ids = _prepare_calibration_samples(samples)
    current_white = np.asarray(current_white, dtype=np.float64)
    if (
        current_white.shape != (6,)
        or not np.all(np.isfinite(current_white))
        or np.any(current_white <= 0.0)
    ):
        raise ValueError("current_white must contain six finite positive values")

    visible = np.asarray(VISIBLE_INDICES, dtype=np.int32)
    visible_reflectance = amplitudes[:, visible] / current_white[visible]
    ridge_candidates = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    unique_classes = np.unique(class_ids)
    if unique_classes.size < 2:
        raise ValueError("at least two color classes are required for validation")

    validation_errors = []
    for ridge_coefficient in ridge_candidates:
        fold_errors = []
        for class_id in unique_classes:
            held_out = class_ids == class_id
            matrix, bias = _fit_ridge_matrix(
                visible_reflectance[~held_out],
                target_linear_rgb[~held_out],
                ridge_coefficient,
            )
            prediction = visible_reflectance[held_out] @ matrix + bias
            fold_errors.append(float(np.mean(np.abs(prediction - target_linear_rgb[held_out]))))
        validation_errors.append(float(np.mean(fold_errors)))

    best_index = int(np.argmin(validation_errors))
    ridge_coefficient = ridge_candidates[best_index]
    matrix, bias = _fit_ridge_matrix(
        visible_reflectance, target_linear_rgb, ridge_coefficient
    )
    return MatrixFitResult(
        matrix=matrix,
        bias=bias,
        ridge_coefficient=float(ridge_coefficient),
        validation_error=validation_errors[best_index],
    )


def apply_matrix_fit(reflectance: np.ndarray, result: MatrixFitResult) -> np.ndarray:
    """Predict unclipped linear RGB from six-band reflectance with a matrix fit."""
    reflectance = np.asarray(reflectance, dtype=np.float64)
    if reflectance.ndim != 2 or reflectance.shape[1] != 6:
        raise ValueError("reflectance must have shape (N, 6)")
    if not np.all(np.isfinite(reflectance)):
        raise ValueError("reflectance must contain only finite values")
    if not isinstance(result, MatrixFitResult):
        raise TypeError("result must be a MatrixFitResult instance")
    matrix = np.asarray(result.matrix, dtype=np.float64)
    bias = np.asarray(result.bias, dtype=np.float64)
    if matrix.shape != (len(VISIBLE_INDICES), 3) or bias.shape != (1, 3):
        raise ValueError("matrix fit must contain a 5x3 matrix and 1x3 bias")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must contain only finite values")
    if not np.all(np.isfinite(bias)):
        raise ValueError("bias must contain only finite values")
    return reflectance[:, VISIBLE_INDICES] @ matrix + bias


def load_camera_srgb(path: str | Path) -> np.ndarray:
    """Load an ICC-tagged camera photo as normalized float64 sRGB pixels."""
    with Image.open(path) as source:
        profile_bytes = source.info.get("icc_profile")
        if not profile_bytes:
            raise ValueError("camera image must include an embedded ICC profile")
        source_profile = ImageCms.ImageCmsProfile(BytesIO(profile_bytes))
        srgb_profile = ImageCms.createProfile("sRGB")
        rgb = source.convert("RGB")
        converted = ImageCms.profileToProfile(
            rgb, source_profile, srgb_profile, outputMode="RGB"
        )
        return np.asarray(converted, dtype=np.float64) / 255.0


def _point_array(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def normalized_scan_coordinates(
    azimuth: np.ndarray, elevation: np.ndarray
) -> np.ndarray:
    """Normalize matching angular coordinate vectors independently to [0, 1]."""
    azimuth = np.asarray(azimuth, dtype=np.float64)
    elevation = np.asarray(elevation, dtype=np.float64)
    if azimuth.shape != elevation.shape or azimuth.ndim != 1:
        raise ValueError("azimuth and elevation must be matching one-dimensional arrays")
    if azimuth.size == 0 or not (np.all(np.isfinite(azimuth)) and np.all(np.isfinite(elevation))):
        raise ValueError("azimuth and elevation must be non-empty and finite")
    azimuth_span = np.ptp(azimuth)
    elevation_span = np.ptp(elevation)
    if azimuth_span == 0 or elevation_span == 0:
        raise ValueError("azimuth and elevation must each have non-zero span")
    return np.column_stack(
        ((azimuth - azimuth.min()) / azimuth_span,
         (elevation - elevation.min()) / elevation_span)
    )


def project_homogeneous(scan_uv: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Project normalized scan coordinates into photo pixel coordinates."""
    scan_uv = _point_array(scan_uv, "scan_uv")
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 3x3 matrix")
    projected = np.column_stack((scan_uv, np.ones(scan_uv.shape[0]))) @ transform.T
    denominator = projected[:, 2]
    if np.any(np.isclose(denominator, 0.0)):
        raise ValueError("transform maps a point to infinity")
    return projected[:, :2] / denominator[:, None]


def _normalization_transform(points: np.ndarray) -> np.ndarray:
    center = points.mean(axis=0)
    mean_distance = np.linalg.norm(points - center, axis=1).mean()
    if not np.isfinite(mean_distance) or np.isclose(mean_distance, 0.0):
        raise ValueError("point correspondences must not collapse to one point")
    scale = np.sqrt(2.0) / mean_distance
    return np.array(
        [[scale, 0.0, -scale * center[0]],
         [0.0, scale, -scale * center[1]],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def fit_homography(scan_uv: np.ndarray, photo_xy: np.ndarray) -> np.ndarray:
    """Fit a finite scan-to-photo homography with normalized DLT."""
    scan_uv = _point_array(scan_uv, "scan_uv")
    photo_xy = _point_array(photo_xy, "photo_xy")
    if scan_uv.shape[0] != photo_xy.shape[0]:
        raise ValueError("scan_uv and photo_xy must have the same number of points")
    if scan_uv.shape[0] < 4:
        raise ValueError("at least four point correspondences are required")

    scan_normalizer = _normalization_transform(scan_uv)
    photo_normalizer = _normalization_transform(photo_xy)
    scan_normalized = project_homogeneous(scan_uv, scan_normalizer)
    photo_normalized = project_homogeneous(photo_xy, photo_normalizer)
    x, y = scan_normalized.T
    u, v = photo_normalized.T
    zeros = np.zeros_like(x)
    ones = np.ones_like(x)
    design = np.empty((2 * len(scan_uv), 9), dtype=np.float64)
    design[0::2] = np.column_stack((-x, -y, -ones, zeros, zeros, zeros, u * x, u * y, u))
    design[1::2] = np.column_stack((zeros, zeros, zeros, -x, -y, -ones, v * x, v * y, v))
    if np.linalg.matrix_rank(design) < 8:
        raise ValueError("point correspondences produce a singular homography")
    _, _, vh = np.linalg.svd(design)
    normalized_transform = vh[-1].reshape(3, 3)
    transform = np.linalg.inv(photo_normalizer) @ normalized_transform @ scan_normalizer
    scale = transform[2, 2]
    if np.isclose(scale, 0.0):
        scale = np.linalg.norm(transform)
    if not np.isfinite(scale) or np.isclose(scale, 0.0):
        raise ValueError("point correspondences produce a singular homography")
    transform = transform / scale
    if not np.all(np.isfinite(transform)):
        raise ValueError("point correspondences produce a non-finite homography")
    if registration_error(transform, scan_uv, photo_xy) > 30.0:
        raise ValueError("median reprojection error exceeds 30 pixels")
    return transform


def registration_error(
    transform: np.ndarray, scan_uv: np.ndarray, photo_xy: np.ndarray
) -> float:
    """Return the median Euclidean reprojection error in photo pixels."""
    scan_uv = _point_array(scan_uv, "scan_uv")
    photo_xy = _point_array(photo_xy, "photo_xy")
    if scan_uv.shape[0] != photo_xy.shape[0] or scan_uv.shape[0] == 0:
        raise ValueError("scan_uv and photo_xy must have the same non-zero length")
    return float(np.median(np.linalg.norm(project_homogeneous(scan_uv, transform) - photo_xy, axis=1)))
