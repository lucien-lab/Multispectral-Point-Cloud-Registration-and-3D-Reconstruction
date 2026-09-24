#!/usr/bin/env python3
"""Fit and render photo-guided colors without modifying source point clouds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import sys
from typing import Any, TextIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from color_calibration_core import (
    ANCHOR_CLASS_NAMES,
    CURRENT_WHITE,
    SOURCE_WHITE,
    VISIBLE_INDICES,
    WAVELENGTHS_NM,
    CalibrationSamples,
    MatrixFitResult,
    WhiteFitResult,
    apply_matrix_fit,
    build_anchor_samples,
    delta_e76,
    fit_homography,
    fit_regularized_matrix,
    linear_srgb_to_lab,
    linear_to_srgb,
    load_camera_srgb,
    normalized_scan_coordinates,
    optimize_white,
    project_homogeneous,
    reflectance_from_white,
    registration_error,
    restore_amplitudes,
    spectral_to_linear_srgb,
)


SCRIPT_VERSION = "1.0.0"
REFLECTANCE_COLUMNS = [f"rfl_{value}_nm" for value in WAVELENGTHS_NM]
REQUIRED_COLUMNS = [
    "scan_index",
    "x_mm",
    "y_mm",
    "z_mm",
    "azimuth_rad",
    "elevation_rad",
    "valid",
    *REFLECTANCE_COLUMNS,
]
PRESERVED_COLUMNS = [
    "scan_index",
    "x_mm",
    "y_mm",
    "z_mm",
    "azimuth_rad",
    "elevation_rad",
    "valid",
]
RGB_COLUMNS = [
    "rgb_physical_r",
    "rgb_physical_g",
    "rgb_physical_b",
    "rgb_photo_matched_r",
    "rgb_photo_matched_g",
    "rgb_photo_matched_b",
]
WHITE_STATUS = [
    "optimized_from_rgb",
    "optimized_from_rgb",
    "optimized_from_rgb",
    "fixed_unobservable",
    "optimized_from_rgb",
    "optimized_from_rgb",
]
CANDIDATE_WHITE_STATUS = [
    "candidate_optimized_from_rgb",
    "candidate_optimized_from_rgb",
    "candidate_optimized_from_rgb",
    "fixed_unobservable",
    "candidate_optimized_from_rgb",
    "candidate_optimized_from_rgb",
]
MAX_ACCEPTED_WHITE_SPREAD = 0.10
MAX_ACCEPTED_BLACK_LUMINANCE = 0.10
MAX_ACCEPTED_GAMUT_CLIPPING_FRACTION = 0.05
REQUIRED_AUTOMATED_CHECK_NAMES = (
    "minimum_valid_classes",
    "minimum_improved_classes",
    "white_wall_channel_spread",
    "black_region_linear_luminance",
    "white_fit_gamut_clipping_fraction",
    "white_values_positive_finite",
    "fixed_803_nm",
    "geometry_and_row_counts_preserved",
)
FINAL_WHITE_RELATIVE_PATH = "white_fit/final_white_amplitudes.json"


@dataclass(frozen=True)
class CloudData:
    side: str
    path: Path
    fieldnames: list[str]
    rows: list[dict[str, str]]
    xyz: np.ndarray
    azimuth: np.ndarray
    elevation: np.ndarray
    valid: np.ndarray
    source_reflectance: np.ndarray
    amplitudes: np.ndarray


class SafeOutput:
    """Resolve every artifact beneath one root without following child symlinks."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, relative: str | Path, create_parents: bool = False) -> Path:
        relative = Path(relative)
        if relative.is_absolute() or not relative.parts or any(
            part in ("", ".", "..") for part in relative.parts
        ):
            raise ValueError(f"unsafe calibration output path: {relative}")
        if create_parents:
            self.root.mkdir(parents=True, exist_ok=True)
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(f"calibration output root is not a directory: {self.root}")

        current = self.root
        for part in relative.parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ValueError(
                    f"refusing calibration output symlink component: {current}"
                )
            if current.exists() and not current.is_dir():
                raise ValueError(
                    f"calibration output parent is not a directory: {current}"
                )
            if create_parents and not current.exists():
                current.mkdir()
            if current.exists():
                resolved = current.resolve()
                if resolved != self.root and self.root not in resolved.parents:
                    raise ValueError(
                        f"calibration output path escapes output root: {current}"
                    )

        target = current / relative.parts[-1]
        if target.is_symlink():
            raise ValueError(f"refusing calibration output target symlink: {target}")
        if target.exists() and target.is_dir():
            raise ValueError(f"calibration output target is a directory: {target}")
        resolved_target = target.resolve(strict=False)
        if resolved_target != self.root and self.root not in resolved_target.parents:
            raise ValueError(f"calibration output path escapes output root: {target}")
        return target

    def is_file(self, relative: str | Path) -> bool:
        return self._path(relative).is_file()

    def absolute(self, relative: str | Path) -> Path:
        return self._path(relative)

    def open_text(self, relative: str | Path, newline: str | None = None) -> TextIO:
        target = self._path(relative, create_parents=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o666)
        return os.fdopen(descriptor, "w", encoding="utf-8", newline=newline)

    def write_text(self, relative: str | Path, value: str) -> None:
        with self.open_text(relative) as handle:
            handle.write(value)

    def write_text_atomic(self, relative: str | Path, value: str) -> None:
        target = self._path(relative, create_parents=True)
        temporary_relative = Path(relative).with_name(
            f".{target.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
        )
        temporary = self._path(temporary_relative)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(temporary, flags, 0o666)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            self._path(relative)
            os.replace(temporary, target)
        except BaseException:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def write_bytes(self, relative: str | Path, value: bytes) -> None:
        target = self._path(relative, create_parents=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(target, flags, 0o666)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Register a camera photo and produce non-destructive calibration outputs."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-registration", action="store_true")
    modes.add_argument("--fit", action="store_true")
    modes.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--left-csv",
        type=Path,
        default=here / "pointcloud_output" / "left_pointcloud_all.csv",
    )
    parser.add_argument(
        "--right-csv",
        type=Path,
        default=here / "pointcloud_output" / "right_pointcloud_all.csv",
    )
    parser.add_argument("--camera-photo", type=Path, default=here / "camera_show.jpg")
    parser.add_argument(
        "--registration-json", type=Path, default=here / "calibration_registration.json"
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=here / "pointcloud_output" / "metadata.json",
        help="Input point-cloud metadata whose wavelength order must match.",
    )
    parser.add_argument(
        "--output-root", type=Path, default=here / "calibration_output"
    )
    parser.add_argument(
        "--wavelength-order-nm",
        nargs=6,
        type=int,
        default=WAVELENGTHS_NM.tolist(),
    )
    parser.add_argument(
        "--source-white-amplitudes",
        nargs=6,
        type=float,
        default=SOURCE_WHITE.tolist(),
        help="Explicit white array used to generate the source CSV reflectance.",
    )
    parser.add_argument("--overwrite-calibration-output", action="store_true")
    parser.add_argument(
        "--accept-fit",
        action="store_true",
        help=(
            "Explicitly attest visual acceptance; accepted=true is still blocked "
            "unless every automated acceptance check passes."
        ),
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(output: SafeOutput, relative: str | Path, value: Any) -> None:
    output.write_text(
        relative,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _write_fit_state(output: SafeOutput, value: dict[str, Any]) -> None:
    output.write_text_atomic(
        FINAL_WHITE_RELATIVE_PATH,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _fit_in_progress_document() -> dict[str, Any]:
    return {
        "accepted": False,
        "classification": "fit_in_progress",
        "automated_acceptance_passed": False,
        "visual_acceptance_required": True,
        "visual_acceptance_explicitly_approved": False,
        "acceptance_reasons": ["fit_in_progress"],
        "completion_marker": "fit_in_progress",
    }


def _validate_wavelength_order(value: Any, source: str) -> None:
    try:
        order = [int(item) for item in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} wavelength_order_nm must contain six integers") from error
    expected = WAVELENGTHS_NM.tolist()
    if order != expected:
        raise ValueError(
            f"{source} wavelength order mismatch: expected {expected}, received {order}"
        )


def _validate_paths(args: argparse.Namespace) -> Path:
    here = Path(__file__).resolve().parent
    output_root = args.output_root.expanduser().resolve()
    pointcloud_output = (here / "pointcloud_output").resolve()
    if output_root == pointcloud_output or pointcloud_output in output_root.parents:
        raise ValueError("output root must not be pointcloud_output or a directory inside it")
    for input_path in (
        args.left_csv,
        args.right_csv,
        args.camera_photo,
        args.registration_json,
        args.metadata_json,
    ):
        resolved = input_path.expanduser().resolve()
        if resolved == output_root or output_root in resolved.parents:
            raise ValueError(f"input file must not be inside output root: {input_path}")
    if not args.render_only and output_root.exists() and not args.overwrite_calibration_output:
        raise FileExistsError(
            f"calibration output already exists: {output_root}; use "
            "--overwrite-calibration-output to replace calibration artifacts"
        )
    if args.render_only and not output_root.is_dir():
        raise FileNotFoundError(f"render-only requires an existing output root: {output_root}")
    return output_root


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label} JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def _load_cloud(path: Path, side: str, source_white: np.ndarray) -> CloudData:
    if not path.is_file():
        raise FileNotFoundError(f"missing {side} CSV: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{side} CSV missing required columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"{side} CSV contains no data rows")

    def numeric(columns: list[str]) -> np.ndarray:
        try:
            values = np.asarray(
                [[float(row[column]) for column in columns] for row in rows],
                dtype=np.float64,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{side} CSV contains a non-numeric value in: {', '.join(columns)}"
            ) from error
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"{side} CSV contains a non-finite value in: {', '.join(columns)}"
            )
        return values

    xyz = numeric(["x_mm", "y_mm", "z_mm"])
    angles = numeric(["azimuth_rad", "elevation_rad"])
    validity = numeric(["valid"])[:, 0]
    if np.any((validity != 0.0) & (validity != 1.0)):
        raise ValueError(f"{side} CSV valid column must contain only 0 or 1")
    reflectance = numeric(REFLECTANCE_COLUMNS)
    # The source CSVs were generated with the historical SOURCE_WHITE array.
    # Validation above makes that source explicit rather than guessing it.
    amplitudes = restore_amplitudes(reflectance)
    return CloudData(
        side=side,
        path=path.resolve(),
        fieldnames=fieldnames,
        rows=rows,
        xyz=xyz,
        azimuth=angles[:, 0],
        elevation=angles[:, 1],
        valid=validity.astype(bool),
        source_reflectance=reflectance,
        amplitudes=amplitudes,
    )


def _validate_configuration(
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    _validate_wavelength_order(args.wavelength_order_nm, "CLI")
    source_white = np.asarray(args.source_white_amplitudes, dtype=np.float64)
    if (
        source_white.shape != (6,)
        or not np.all(np.isfinite(source_white))
        or np.any(source_white <= 0.0)
    ):
        raise ValueError("source white amplitudes must contain six finite positive values")
    if not np.array_equal(source_white, SOURCE_WHITE):
        raise ValueError(
            "source white amplitudes must explicitly match SOURCE_WHITE for these CSV files"
        )
    registration = _load_json(args.registration_json, "registration")
    _validate_wavelength_order(registration.get("wavelength_order_nm"), "registration")
    if registration.get("orientation") != "same":
        raise ValueError("registration orientation must be 'same'")
    metadata = _load_json(args.metadata_json, "metadata")
    _validate_wavelength_order(metadata.get("wavelength_order_nm"), "metadata")
    return source_white, registration, metadata


def _registration_projection(
    clouds: list[CloudData], registration: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, float]:
    azimuth = np.concatenate([cloud.azimuth for cloud in clouds])
    elevation = np.concatenate([cloud.elevation for cloud in clouds])
    scan_uv = normalized_scan_coordinates(azimuth, elevation)
    scan_control = np.asarray(registration.get("scan_control_points"), dtype=np.float64)
    photo_control = np.asarray(
        registration.get("photo_control_points_px"), dtype=np.float64
    )
    transform = fit_homography(scan_control, photo_control)
    error = registration_error(transform, scan_control, photo_control)
    maximum = float(registration.get("max_median_reprojection_error_px", 30.0))
    if not np.isfinite(maximum) or maximum <= 0.0 or maximum > 30.0:
        raise ValueError(
            "registration max_median_reprojection_error_px must be in (0, 30]"
        )
    if error > maximum:
        raise ValueError(
            f"registration median reprojection error {error:.6g} exceeds {maximum:.6g} pixels"
        )
    return scan_uv, project_homogeneous(scan_uv, transform), error


def _white_array_provenance(name: str, values: np.ndarray) -> dict[str, Any]:
    canonical_value = {
        "name": name,
        "wavelength_order_nm": [int(value) for value in WAVELENGTHS_NM],
        "values": [float(value) for value in values],
    }
    canonical_json = json.dumps(
        canonical_value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "canonical_value": canonical_value,
        "canonical_json_utf8": canonical_json,
        "encoding": "UTF-8 canonical JSON",
        "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
    }


def _provenance(
    args: argparse.Namespace, source_white: np.ndarray
) -> dict[str, Any]:
    script = Path(__file__).resolve()
    paths = {
        "camera_photo": args.camera_photo,
        "left_csv": args.left_csv,
        "right_csv": args.right_csv,
        "registration_json": args.registration_json,
        "script": script,
    }
    return {
        "script_version": SCRIPT_VERSION,
        "wavelength_order_nm": WAVELENGTHS_NM.tolist(),
        "source_white_amplitudes": source_white.tolist(),
        "optimization_center_white": CURRENT_WHITE.tolist(),
        "white_array_provenance": {
            "source_white_amplitudes": _white_array_provenance(
                "source_white_amplitudes", source_white
            ),
            "optimization_center_white": _white_array_provenance(
                "optimization_center_white", CURRENT_WHITE
            ),
        },
        "files": {
            name: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
    }


def _registration_report(
    registration: dict[str, Any], transform: np.ndarray, error: float
) -> dict[str, Any]:
    return {
        **registration,
        "homography_scan_to_photo": transform.tolist(),
        "median_reprojection_error_px": error,
        "accepted_numerically": True,
        "visual_acceptance_required": True,
    }


def _rgb8(linear_rgb: np.ndarray) -> np.ndarray:
    encoded = linear_to_srgb(np.asarray(linear_rgb, dtype=np.float64))
    if not np.all(np.isfinite(encoded)):
        raise ValueError("display RGB contains non-finite values")
    values = np.rint(encoded * 255.0)
    if np.any((values < 0.0) | (values > 255.0)):
        raise ValueError("display RGB falls outside byte range 0..255")
    return values.astype(np.uint8)


def _split(array: np.ndarray, clouds: list[CloudData]) -> list[np.ndarray]:
    sizes = [len(cloud.rows) for cloud in clouds]
    offsets = np.cumsum(sizes)[:-1]
    return list(np.split(array, offsets))


def _write_cloud_csv(
    output: SafeOutput,
    relative: str | Path,
    cloud: CloudData,
    reflectance: np.ndarray,
    physical_rgb8: np.ndarray,
    matched_rgb8: np.ndarray,
) -> None:
    fields = cloud.fieldnames + [name for name in RGB_COLUMNS if name not in cloud.fieldnames]
    with output.open_text(relative, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, source_row in enumerate(cloud.rows):
            row = dict(source_row)
            for column, value in zip(REFLECTANCE_COLUMNS, reflectance[index]):
                row[column] = f"{value:.12g}"
            for column, value in zip(RGB_COLUMNS[:3], physical_rgb8[index]):
                row[column] = str(int(value))
            for column, value in zip(RGB_COLUMNS[3:], matched_rgb8[index]):
                row[column] = str(int(value))
            writer.writerow(row)


def _write_ply(
    output: SafeOutput,
    relative: str | Path,
    cloud: CloudData,
    reflectance: np.ndarray,
    matched_rgb8: np.ndarray,
    scheme: str,
    parameter_relative: str | Path,
) -> None:
    selected = np.flatnonzero(cloud.valid)
    with output.open_text(relative, newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"comment calibration_scheme {scheme}\n")
        handle.write(
            f"comment parameter_file {output.absolute(parameter_relative)}\n"
        )
        handle.write(f"element vertex {selected.size}\n")
        for axis in ("x", "y", "z"):
            handle.write(f"property float {axis}\n")
        for color in ("red", "green", "blue"):
            handle.write(f"property uchar {color}\n")
        handle.write("property int scan_index\n")
        for wavelength in WAVELENGTHS_NM:
            handle.write(f"property float rfl_{int(wavelength)}_nm\n")
        handle.write("end_header\n")
        for index in selected:
            xyz = cloud.xyz[index]
            rgb = matched_rgb8[index]
            fields = [
                *(f"{value:.9g}" for value in xyz),
                *(str(int(value)) for value in rgb),
                str(int(float(cloud.rows[index]["scan_index"]))),
                *(f"{value:.9g}" for value in reflectance[index]),
            ]
            handle.write(" ".join(fields) + "\n")


def _plot_preview(
    output: SafeOutput,
    relative: str | Path,
    clouds: list[CloudData],
    rgb_parts: list[np.ndarray],
    title: str,
    overwrite: bool,
) -> None:
    if output.is_file(relative) and not overwrite:
        return
    figure = plt.figure(figsize=(8, 6), dpi=130)
    axis = figure.add_subplot(111, projection="3d")
    for cloud, rgb in zip(clouds, rgb_parts):
        valid = cloud.valid
        axis.scatter(
            cloud.xyz[valid, 0],
            cloud.xyz[valid, 1],
            cloud.xyz[valid, 2],
            c=rgb[valid] / 255.0,
            s=5,
            depthshade=False,
        )
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(title)
    axis.set_box_aspect((1, 1, 0.5))
    axis.view_init(elev=20, azim=-50)
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    output.write_bytes(relative, buffer.getvalue())


def _plot_angular_projection(
    output: SafeOutput,
    relative: str | Path,
    photo: np.ndarray,
    projected_xy: np.ndarray,
    valid: np.ndarray,
    rgb8: np.ndarray,
    overwrite: bool,
) -> None:
    if output.is_file(relative) and not overwrite:
        return
    figure, axis = plt.subplots(figsize=(10, 6), dpi=130)
    axis.imshow(photo)
    axis.scatter(
        projected_xy[valid, 0],
        projected_xy[valid, 1],
        c=rgb8[valid] / 255.0,
        s=8,
        edgecolors="none",
    )
    axis.set_title("Angular projection on registered camera photo")
    axis.set_axis_off()
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    output.write_bytes(relative, buffer.getvalue())


def _plot_comparison(
    output: SafeOutput,
    relative: str | Path,
    photo: np.ndarray,
    projected_xy: np.ndarray,
    valid: np.ndarray,
    white_rgb8: np.ndarray,
    matrix_rgb8: np.ndarray,
    overwrite: bool,
) -> None:
    if output.is_file(relative) and not overwrite:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=130)
    axes[0].imshow(photo)
    axes[0].set_title("Camera sRGB")
    for axis, colors, title in (
        (axes[1], white_rgb8, "White-divisor fit candidate"),
        (axes[2], matrix_rgb8, "Photo-matched matrix"),
    ):
        axis.imshow(photo * 0.25)
        axis.scatter(
            projected_xy[valid, 0],
            projected_xy[valid, 1],
            c=colors[valid] / 255.0,
            s=8,
            edgecolors="none",
        )
        axis.set_title(title)
    for axis in axes:
        axis.set_axis_off()
    figure.tight_layout()
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    plt.close(figure)
    output.write_bytes(relative, buffer.getvalue())


def _class_statistics(samples: CalibrationSamples) -> dict[str, Any]:
    target_srgb = linear_to_srgb(samples.target_linear_rgb)
    result: dict[str, Any] = {}
    for class_id in np.unique(samples.class_ids):
        selected = samples.class_ids == class_id
        result[ANCHOR_CLASS_NAMES[int(class_id)]] = {
            "pointcloud_sample_count": int(np.sum(selected)),
            "target_camera_srgb_median": np.median(target_srgb[selected], axis=0).tolist(),
        }
    return result


def _fit_metrics(
    samples: CalibrationSamples,
    white_result: WhiteFitResult,
    matrix_result: MatrixFitResult,
) -> dict[str, Any]:
    current_linear = spectral_to_linear_srgb(
        reflectance_from_white(samples.amplitudes, CURRENT_WHITE)
    )
    white_linear = (
        spectral_to_linear_srgb(
            reflectance_from_white(samples.amplitudes, white_result.white)
        )
        * white_result.exposure
    )
    matrix_linear = apply_matrix_fit(
        reflectance_from_white(samples.amplitudes, CURRENT_WHITE), matrix_result
    )
    target_lab = linear_srgb_to_lab(samples.target_linear_rgb)
    predictions = {
        "current_white": current_linear,
        "white_fit": white_linear,
        "matrix_fit": matrix_linear,
    }
    by_class: dict[str, Any] = {}
    scheme_errors: dict[str, list[float]] = {name: [] for name in predictions}
    for class_id in np.unique(samples.class_ids):
        selected = samples.class_ids == class_id
        entry: dict[str, Any] = {
            "sample_count": int(np.sum(selected)),
            "camera_target_srgb": np.median(
                linear_to_srgb(samples.target_linear_rgb[selected]), axis=0
            ).tolist(),
        }
        for name, linear in predictions.items():
            error = float(
                np.median(
                    delta_e76(
                        linear_srgb_to_lab(np.maximum(linear[selected], 0.0)),
                        target_lab[selected],
                    )
                )
            )
            entry[name] = {
                "predicted_srgb": np.median(
                    linear_to_srgb(linear[selected]), axis=0
                ).tolist(),
                "median_delta_e76": error,
            }
            scheme_errors[name].append(error)
        by_class[ANCHOR_CLASS_NAMES[int(class_id)]] = entry
    return {
        "class_metrics": by_class,
        "equal_class_median_delta_e76": {
            name: float(np.median(errors)) for name, errors in scheme_errors.items()
        },
        "gamut_clipping_fraction": {
            name: float(np.mean(np.any((linear < 0.0) | (linear > 1.0), axis=1)))
            for name, linear in predictions.items()
        },
        "white_wall_channel_spread": float(
            np.ptp(np.median(linear_to_srgb(white_linear[samples.class_ids == 0]), axis=0))
        ),
        "black_region_linear_luminance": (
            float(
                np.median(
                    white_linear[samples.class_ids == 4]
                    @ np.array([0.2126729, 0.7151522, 0.0721750])
                )
            )
            if np.any(samples.class_ids == 4)
            else None
        ),
    }


def _automated_acceptance_checks(
    metrics: dict[str, Any],
    white_result: WhiteFitResult,
) -> dict[str, dict[str, Any]]:
    class_metrics = metrics["class_metrics"]
    improved_classes = sum(
        entry["white_fit"]["median_delta_e76"]
        < entry["current_white"]["median_delta_e76"]
        for entry in class_metrics.values()
    )
    white_spread = metrics["white_wall_channel_spread"]
    black_luminance = metrics["black_region_linear_luminance"]
    clipping = metrics["gamut_clipping_fraction"]["white_fit"]
    white = np.asarray(white_result.white, dtype=np.float64)
    checks = {
        "minimum_valid_classes": {
            "passed": len(class_metrics) >= 3,
            "actual": len(class_metrics),
            "minimum": 3,
        },
        "minimum_improved_classes": {
            "passed": improved_classes >= 3,
            "actual": improved_classes,
            "minimum": 3,
        },
        "white_wall_channel_spread": {
            "passed": bool(
                white_spread is not None
                and np.isfinite(white_spread)
                and white_spread <= MAX_ACCEPTED_WHITE_SPREAD
            ),
            "actual": white_spread,
            "maximum": MAX_ACCEPTED_WHITE_SPREAD,
        },
        "black_region_linear_luminance": {
            "passed": bool(
                black_luminance is not None
                and np.isfinite(black_luminance)
                and black_luminance <= MAX_ACCEPTED_BLACK_LUMINANCE
            ),
            "actual": black_luminance,
            "maximum": MAX_ACCEPTED_BLACK_LUMINANCE,
        },
        "white_fit_gamut_clipping_fraction": {
            "passed": bool(
                np.isfinite(clipping)
                and clipping <= MAX_ACCEPTED_GAMUT_CLIPPING_FRACTION
            ),
            "actual": clipping,
            "maximum": MAX_ACCEPTED_GAMUT_CLIPPING_FRACTION,
        },
        "white_values_positive_finite": {
            "passed": bool(
                white.shape == (6,)
                and np.all(np.isfinite(white))
                and np.all(white > 0.0)
            ),
            "actual_count": int(white.size),
            "required_count": 6,
        },
        "fixed_803_nm": {
            "passed": bool(white.shape == (6,) and white[3] == 361.2250),
            "actual": float(white[3]) if white.size > 3 else None,
            "required": 361.2250,
        },
    }
    return checks


def verify_exported_geometry_and_counts(
    clouds: list[CloudData], output: SafeOutput
) -> dict[str, Any]:
    """Verify actual white/matrix CSV invariants against source rows."""
    sides: dict[str, Any] = {}
    passed = True
    for cloud in clouds:
        source_rows = cloud.rows
        source_valid_count = int(np.sum(cloud.valid))
        side_result: dict[str, Any] = {
            "source_rows": len(source_rows),
            "source_valid_count": source_valid_count,
        }
        for scheme in ("white_fit", "matrix_fit"):
            relative = f"{scheme}/{cloud.side}_pointcloud.csv"
            path = output.absolute(relative)
            output_rows: list[dict[str, str]] = []
            fieldnames: list[str] = []
            read_error: str | None = None
            if not path.is_file():
                read_error = f"missing actual output CSV: {path}"
            else:
                try:
                    with path.open(newline="", encoding="utf-8") as handle:
                        reader = csv.DictReader(handle)
                        fieldnames = list(reader.fieldnames or [])
                        output_rows = [dict(row) for row in reader]
                except (OSError, csv.Error) as error:
                    read_error = str(error)

            missing_fields = [
                field for field in PRESERVED_COLUMNS if field not in fieldnames
            ]
            row_count_matches = len(output_rows) == len(source_rows)
            output_valid_count: int | None = None
            valid_values_well_formed = False
            if not missing_fields:
                try:
                    valid_values = [float(row["valid"]) for row in output_rows]
                    valid_values_well_formed = all(
                        value in (0.0, 1.0) for value in valid_values
                    )
                    if valid_values_well_formed:
                        output_valid_count = sum(value == 1.0 for value in valid_values)
                except (KeyError, TypeError, ValueError):
                    valid_values_well_formed = False
            valid_count_matches = (
                valid_values_well_formed
                and output_valid_count == source_valid_count
            )
            preserved_fields_exact = bool(
                not missing_fields
                and row_count_matches
                and all(
                    all(
                        source_row[field] == output_row[field]
                        for field in PRESERVED_COLUMNS
                    )
                    for source_row, output_row in zip(source_rows, output_rows)
                )
            )
            scheme_passed = bool(
                read_error is None
                and row_count_matches
                and valid_count_matches
                and preserved_fields_exact
            )
            passed &= scheme_passed
            side_result[scheme] = {
                "passed": scheme_passed,
                "actual_output_path": str(path),
                "actual_output_rows": len(output_rows),
                "actual_output_valid_count": output_valid_count,
                "row_count_matches": row_count_matches,
                "valid_count_matches": valid_count_matches,
                "preserved_fields_exact": preserved_fields_exact,
                "missing_preserved_fields": missing_fields,
                "read_error": read_error,
            }
        sides[cloud.side] = side_result
    return {
        "passed": bool(passed),
        "verification": "actual_output_csvs",
        "preserved_fields": PRESERVED_COLUMNS,
        "sides": sides,
    }


def require_accepted_white_amplitudes(document: dict[str, Any]) -> np.ndarray:
    """Return a write-back array only from a consistent, explicitly accepted fit."""
    if not isinstance(document, dict):
        raise TypeError("accepted fit document must be a JSON object")
    if document.get("accepted") is not True:
        raise ValueError("accepted must be true before white-array write-back")
    if document.get("automated_acceptance_passed") is not True:
        raise ValueError(
            "automated_acceptance_passed must be true before white-array write-back"
        )
    if document.get("visual_acceptance_explicitly_approved") is not True:
        raise ValueError(
            "visual_acceptance_explicitly_approved must be true before white-array write-back"
        )
    if document.get("classification") != "accepted_photo_fit":
        raise ValueError(
            "classification must be accepted_photo_fit before white-array write-back"
        )
    if document.get("completion_marker") != "fit_outputs_complete":
        raise ValueError(
            "completion_marker must be fit_outputs_complete before white-array write-back"
        )
    checks = document.get("automated_checks")
    if not isinstance(checks, dict) or set(checks) != set(
        REQUIRED_AUTOMATED_CHECK_NAMES
    ):
        raise ValueError(
            "automated_checks must contain the complete required acceptance checks"
        )
    if any(
        not isinstance(check, dict) or check.get("passed") is not True
        for check in checks.values()
    ):
        raise ValueError("automated_checks must all have passed true")
    geometry_check = checks["geometry_and_row_counts_preserved"]
    if geometry_check.get("verification") != "actual_output_csvs":
        raise ValueError(
            "geometry_and_row_counts_preserved verification must be actual_output_csvs"
        )
    if geometry_check.get("preserved_fields") != PRESERVED_COLUMNS:
        raise ValueError("actual output geometry preserved_fields are incomplete")
    sides = geometry_check.get("sides")
    if not isinstance(sides, dict) or set(sides) != {"left", "right"}:
        raise ValueError("actual output geometry must contain left and right sides")
    for side in ("left", "right"):
        side_check = sides[side]
        if not isinstance(side_check, dict):
            raise ValueError(f"actual output geometry {side} evidence must be an object")
        source_rows = side_check.get("source_rows")
        source_valid_count = side_check.get("source_valid_count")
        if (
            type(source_rows) is not int
            or type(source_valid_count) is not int
            or source_rows < 0
            or source_valid_count < 0
            or source_valid_count > source_rows
        ):
            raise ValueError(f"actual output geometry {side} source counts are invalid")
        if not {"white_fit", "matrix_fit"}.issubset(side_check):
            raise ValueError(
                f"actual output geometry {side} must contain white_fit and matrix_fit"
            )
        for scheme in ("white_fit", "matrix_fit"):
            scheme_check = side_check[scheme]
            if not isinstance(scheme_check, dict):
                raise ValueError(
                    f"actual output geometry {side} {scheme} evidence must be an object"
                )
            actual_path = scheme_check.get("actual_output_path")
            required_exact_values = (
                scheme_check.get("passed") is True
                and scheme_check.get("actual_output_rows") == source_rows
                and scheme_check.get("actual_output_valid_count")
                == source_valid_count
                and scheme_check.get("row_count_matches") is True
                and scheme_check.get("valid_count_matches") is True
                and scheme_check.get("preserved_fields_exact") is True
                and scheme_check.get("missing_preserved_fields") == []
                and scheme_check.get("read_error") is None
                and isinstance(actual_path, str)
                and bool(actual_path)
            )
            if not required_exact_values:
                raise ValueError(
                    f"actual output geometry {side} {scheme} evidence is incomplete"
                )
    if document.get("status") != WHITE_STATUS:
        raise ValueError("status must contain the six accepted WHITE_STATUS values")
    if document.get("acceptance_reasons") != []:
        raise ValueError("acceptance_reasons must be empty for an accepted fit")
    _validate_wavelength_order(document.get("wavelength_order_nm"), "accepted fit")
    white = np.asarray(document.get("white_amplitudes"), dtype=np.float64)
    if white.shape != (6,) or not np.all(np.isfinite(white)) or np.any(white <= 0.0):
        raise ValueError("accepted white amplitudes must contain six finite positive values")
    if white[3] != 361.2250:
        raise ValueError("accepted white amplitude at 803 nm must remain 361.2250")
    return white.copy()


def _prepare_inputs(
    args: argparse.Namespace,
    validated_output_root: Path | None = None,
) -> tuple[
    Path,
    np.ndarray,
    dict[str, Any],
    list[CloudData],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    np.ndarray,
]:
    output_root = validated_output_root or _validate_paths(args)
    source_white, registration, _ = _validate_configuration(args)
    clouds = [
        _load_cloud(args.left_csv, "left", source_white),
        _load_cloud(args.right_csv, "right", source_white),
    ]
    scan_uv, projected_xy, error = _registration_projection(clouds, registration)
    transform = fit_homography(
        np.asarray(registration["scan_control_points"], dtype=np.float64),
        np.asarray(registration["photo_control_points_px"], dtype=np.float64),
    )
    photo = load_camera_srgb(args.camera_photo)
    height, width = photo.shape[:2]
    photo_control = np.asarray(registration["photo_control_points_px"], dtype=np.float64)
    if np.any(photo_control[:, 0] < 0.0) or np.any(photo_control[:, 0] > width):
        raise ValueError("registration photo x control points fall outside camera image")
    if np.any(photo_control[:, 1] < 0.0) or np.any(photo_control[:, 1] > height):
        raise ValueError("registration photo y control points fall outside camera image")
    return (
        output_root,
        source_white,
        registration,
        clouds,
        scan_uv,
        projected_xy,
        transform,
        error,
        photo,
    )


def prepare_registration(args: argparse.Namespace) -> None:
    (
        output_root,
        source_white,
        registration,
        clouds,
        _,
        projected_xy,
        transform,
        error,
        photo,
    ) = _prepare_inputs(args)
    output = SafeOutput(output_root)
    amplitudes = np.vstack([cloud.amplitudes for cloud in clouds])
    valid = np.concatenate([cloud.valid for cloud in clouds])
    physical = _rgb8(
        spectral_to_linear_srgb(reflectance_from_white(amplitudes, CURRENT_WHITE))
    )
    _write_json(output, "common/source_provenance.json", _provenance(args, source_white))
    _write_json(
        output,
        "common/registration.json",
        _registration_report(registration, transform, error),
    )
    _plot_angular_projection(
        output,
        "common/angular_projection.png",
        photo,
        projected_xy,
        valid,
        physical,
        True,
    )


def fit(args: argparse.Namespace) -> None:
    output_root = _validate_paths(args)
    output = SafeOutput(output_root)
    existing_output_invalidated = output_root.exists()
    if existing_output_invalidated:
        _write_fit_state(output, _fit_in_progress_document())
    (
        _,
        source_white,
        registration,
        clouds,
        _,
        projected_xy,
        transform,
        error,
        photo,
    ) = _prepare_inputs(args, output_root)
    if not existing_output_invalidated:
        _write_fit_state(output, _fit_in_progress_document())
    amplitudes = np.vstack([cloud.amplitudes for cloud in clouds])
    valid = np.concatenate([cloud.valid for cloud in clouds])
    samples = build_anchor_samples(amplitudes[valid], projected_xy[valid], photo)
    white_result = optimize_white(samples)
    matrix_result = fit_regularized_matrix(samples, CURRENT_WHITE)
    if (
        white_result.white.shape != (6,)
        or not np.all(np.isfinite(white_result.white))
        or np.any(white_result.white <= 0.0)
        or white_result.white[3] != 361.2250
    ):
        raise ValueError("white optimizer produced invalid six-element amplitudes")

    current_reflectance = reflectance_from_white(amplitudes, CURRENT_WHITE)
    white_reflectance = reflectance_from_white(amplitudes, white_result.white)
    physical_rgb8 = _rgb8(spectral_to_linear_srgb(current_reflectance))
    white_rgb8 = _rgb8(
        spectral_to_linear_srgb(white_reflectance) * white_result.exposure
    )
    matrix_rgb8 = _rgb8(apply_matrix_fit(current_reflectance, matrix_result))
    provenance = _provenance(args, source_white)
    white_parameters = {
        "wavelength_order_nm": WAVELENGTHS_NM.tolist(),
        "white_amplitudes": white_result.white.tolist(),
        "exposure": white_result.exposure,
        "objective": white_result.objective,
        "class_median_delta_e76": white_result.class_metrics,
        "visible_optimized_indices": list(VISIBLE_INDICES),
        "fixed_unobservable_index": 3,
    }
    matrix_parameters = {
        "wavelength_order_nm": WAVELENGTHS_NM.tolist(),
        "visible_input_order_nm": [int(WAVELENGTHS_NM[index]) for index in VISIBLE_INDICES],
        "current_white_amplitudes": CURRENT_WHITE.tolist(),
        "matrix": matrix_result.matrix.tolist(),
        "bias": matrix_result.bias.tolist(),
        "ridge_coefficient": matrix_result.ridge_coefficient,
        "validation_error": matrix_result.validation_error,
        "domain": "linear_rgb",
        "rgb_physical_note": "CIE baseline using current white amplitudes",
        "rgb_photo_matched_note": "regularized fitted display matrix",
    }
    metrics = _fit_metrics(samples, white_result, matrix_result)
    automated_checks = _automated_acceptance_checks(metrics, white_result)
    anchor_statistics = {
        "minimum_pointcloud_samples_per_class": 15,
        "balanced_maximum_per_class": 500,
        "sampling_seed": 20260722,
        "classes": _class_statistics(samples),
    }

    _write_json(output, "common/source_provenance.json", provenance)
    _write_json(
        output,
        "common/registration.json",
        _registration_report(registration, transform, error),
    )
    _write_json(output, "common/anchor_statistics.json", anchor_statistics)
    _write_json(output, "white_fit/parameters.json", white_parameters)
    _write_json(output, "matrix_fit/parameters.json", matrix_parameters)
    _write_json(output, "comparison/metrics.json", metrics)

    physical_parts = _split(physical_rgb8, clouds)
    white_rgb_parts = _split(white_rgb8, clouds)
    matrix_rgb_parts = _split(matrix_rgb8, clouds)
    white_reflectance_parts = _split(white_reflectance, clouds)
    current_reflectance_parts = _split(current_reflectance, clouds)
    for index, cloud in enumerate(clouds):
        _write_cloud_csv(
            output,
            f"white_fit/{cloud.side}_pointcloud.csv",
            cloud,
            white_reflectance_parts[index],
            physical_parts[index],
            white_rgb_parts[index],
        )
        _write_ply(
            output,
            f"white_fit/{cloud.side}_pointcloud.ply",
            cloud,
            white_reflectance_parts[index],
            white_rgb_parts[index],
            "photo-fit-candidate",
            "white_fit/parameters.json",
        )
        _write_cloud_csv(
            output,
            f"matrix_fit/{cloud.side}_pointcloud.csv",
            cloud,
            current_reflectance_parts[index],
            physical_parts[index],
            matrix_rgb_parts[index],
        )
        _write_ply(
            output,
            f"matrix_fit/{cloud.side}_pointcloud.ply",
            cloud,
            current_reflectance_parts[index],
            matrix_rgb_parts[index],
            "photo-matched-color-matrix",
            "matrix_fit/parameters.json",
        )
    automated_checks["geometry_and_row_counts_preserved"] = (
        verify_exported_geometry_and_counts(clouds, output)
    )
    automated_acceptance_passed = all(
        check["passed"] is True for check in automated_checks.values()
    )
    visual_acceptance_explicitly_approved = bool(args.accept_fit)
    accepted = bool(
        automated_acceptance_passed and visual_acceptance_explicitly_approved
    )
    acceptance_reasons = [
        f"automated_check_failed:{name}"
        for name, check in automated_checks.items()
        if check["passed"] is not True
    ]
    if not visual_acceptance_explicitly_approved:
        acceptance_reasons.append("visual_acceptance_not_explicitly_approved")
    final_white = {
        "wavelength_order_nm": WAVELENGTHS_NM.tolist(),
        "white_amplitudes": white_result.white.tolist(),
        "status": WHITE_STATUS if accepted else CANDIDATE_WHITE_STATUS,
        "accepted": accepted,
        "classification": (
            "accepted_photo_fit" if accepted else "rejected_photo_fit_candidate"
        ),
        "candidate_classification": "photo_fitted_white_divisor_candidate",
        "automated_checks": automated_checks,
        "automated_acceptance_passed": automated_acceptance_passed,
        "visual_acceptance_required": True,
        "visual_acceptance_explicitly_approved": (
            visual_acceptance_explicitly_approved
        ),
        "acceptance_reasons": acceptance_reasons,
        "completion_marker": "fit_outputs_complete",
        "python_array": f"[{', '.join(repr(float(value)) for value in white_result.white)}]",
        "matlab_row_vector": f"[{', '.join(repr(float(value)) for value in white_result.white)}]",
        "identifiability_note": (
            "RGB identifies visible-channel ratios only; the common absolute scale is "
            "selected by regularization toward optimization_center_white."
        ),
        "sources": provenance,
    }
    _plot_preview(
        output,
        "white_fit/preview.png",
        clouds,
        white_rgb_parts,
        "Photo-fit candidate",
        True,
    )
    _plot_preview(
        output,
        "matrix_fit/preview.png",
        clouds,
        matrix_rgb_parts,
        "Photo-matched color matrix",
        True,
    )
    _plot_angular_projection(
        output,
        "common/angular_projection.png",
        photo,
        projected_xy,
        valid,
        physical_rgb8,
        True,
    )
    _plot_comparison(
        output,
        "comparison/camera_white_matrix.png",
        photo,
        projected_xy,
        valid,
        white_rgb8,
        matrix_rgb8,
        True,
    )
    _write_fit_state(output, final_white)


def _load_output_rgb(output: SafeOutput, relative: str | Path) -> np.ndarray:
    path = output.absolute(relative)
    if not path.is_file():
        raise FileNotFoundError(f"render-only missing calibrated CSV: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [column for column in RGB_COLUMNS if not rows or column not in rows[0]]
    if missing:
        raise ValueError(f"render-only CSV missing RGB columns: {', '.join(missing)}")
    values = np.asarray(
        [[int(row[column]) for column in RGB_COLUMNS[3:]] for row in rows],
        dtype=np.int64,
    )
    if np.any((values < 0) | (values > 255)):
        raise ValueError(f"render-only CSV RGB values must be in 0..255: {path}")
    return values.astype(np.uint8)


def render_only(args: argparse.Namespace) -> None:
    (
        output_root,
        _,
        _,
        clouds,
        _,
        projected_xy,
        _,
        _,
        photo,
    ) = _prepare_inputs(args)
    output = SafeOutput(output_root)
    white_parts = [
        _load_output_rgb(output, f"white_fit/{cloud.side}_pointcloud.csv")
        for cloud in clouds
    ]
    matrix_parts = [
        _load_output_rgb(output, f"matrix_fit/{cloud.side}_pointcloud.csv")
        for cloud in clouds
    ]
    overwrite = bool(args.overwrite_calibration_output)
    _plot_preview(
        output,
        "white_fit/preview.png",
        clouds,
        white_parts,
        "Photo-fit candidate",
        overwrite,
    )
    _plot_preview(
        output,
        "matrix_fit/preview.png",
        clouds,
        matrix_parts,
        "Photo-matched color matrix",
        overwrite,
    )
    valid = np.concatenate([cloud.valid for cloud in clouds])
    _plot_comparison(
        output,
        "comparison/camera_white_matrix.png",
        photo,
        projected_xy,
        valid,
        np.vstack(white_parts),
        np.vstack(matrix_parts),
        overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.accept_fit and not args.fit:
            raise ValueError("--accept-fit may only be used together with --fit")
        if args.prepare_registration:
            prepare_registration(args)
        elif args.fit:
            fit(args)
        else:
            render_only(args)
    except (FileNotFoundError, FileExistsError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"completed calibration mode under: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
