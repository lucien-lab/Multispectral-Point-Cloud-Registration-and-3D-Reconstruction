#!/usr/bin/env python3
"""Register left/right scans and an external phone photo without changing inputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from color_calibration_core import load_camera_srgb
from joint_registration_core import (
    CameraModel,
    apply_transform,
    initialize_camera,
    photo_edge_distance,
    project_camera,
    refine_camera_multiscale,
    register_right_to_left,
    render_cloud_edge_mask,
    zbuffer_visible,
)


REQUIRED_COLUMNS = (
    "x_mm",
    "y_mm",
    "z_mm",
    "azimuth_rad",
    "elevation_rad",
    "valid",
)


@dataclass(frozen=True)
class CloudRows:
    side: str
    path: Path
    xyz_all: np.ndarray
    azimuth_all: np.ndarray
    elevation_all: np.ndarray
    valid_all: np.ndarray

    @property
    def xyz(self) -> np.ndarray:
        return self.xyz_all[self.valid_all]

    @property
    def azimuth(self) -> np.ndarray:
        return self.azimuth_all[self.valid_all]

    @property
    def elevation(self) -> np.ndarray:
        return self.elevation_all[self.valid_all]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_cloud_csv(path: Path, side: str) -> CloudRows:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(f"{side} CSV missing required column {missing[0]}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{side} CSV contains no rows")
    values = np.array(
        [
            [
                float(row["x_mm"]),
                float(row["y_mm"]),
                float(row["z_mm"]),
                float(row["azimuth_rad"]),
                float(row["elevation_rad"]),
                float(row["valid"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{side} CSV contains non-finite required values")
    valid = values[:, 5] == 1.0
    if np.count_nonzero(valid) < 6:
        raise ValueError(f"{side} CSV has fewer than six valid points")
    return CloudRows(
        side=side,
        path=path,
        xyz_all=values[:, :3],
        azimuth_all=values[:, 3],
        elevation_all=values[:, 4],
        valid_all=valid,
    )


def _validate_output_root(output_root: Path, left_csv: Path, right_csv: Path) -> Path:
    output = Path(output_root).resolve()
    input_parents = {Path(left_csv).resolve().parent, Path(right_csv).resolve().parent}
    for parent in input_parents:
        if output == parent or parent in output.parents:
            raise ValueError("output root cannot be inside an input point-cloud directory")
    return output


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"refusing symlink output target {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(6)}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _lift_controls(
    left: CloudRows,
    right: CloudRows,
    right_aligned: np.ndarray,
    scan_controls: np.ndarray,
) -> np.ndarray:
    az_all = np.concatenate((left.azimuth_all, right.azimuth_all))
    el_all = np.concatenate((left.elevation_all, right.elevation_all))
    spans = np.array([np.ptp(az_all), np.ptp(el_all)])
    if np.any(spans <= 0.0):
        raise ValueError("combined angular scan has zero span")
    az_valid = np.concatenate((left.azimuth, right.azimuth))
    el_valid = np.concatenate((left.elevation, right.elevation))
    uv = np.column_stack(
        ((az_valid - az_all.min()) / spans[0], (el_valid - el_all.min()) / spans[1])
    )
    xyz = np.vstack((left.xyz, right_aligned))
    rows = cKDTree(uv).query(scan_controls, k=1)[1]
    return xyz[np.asarray(rows, dtype=np.int64)]


def _sample_photo(
    photo: np.ndarray, projected: np.ndarray, depth: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = photo.shape[:2]
    visible = zbuffer_visible(projected, depth, (width, height), 40.0)
    x, y = projected[:, 0], projected[:, 1]
    inside = visible & (x >= 0.0) & (x < width - 1) & (y >= 0.0) & (y < height - 1)
    colors = np.full((len(projected), 3), 0.45, dtype=np.float64)
    rows = np.flatnonzero(inside)
    if rows.size:
        x0 = np.floor(x[rows]).astype(int)
        y0 = np.floor(y[rows]).astype(int)
        dx = x[rows] - x0
        dy = y[rows] - y0
        colors[rows] = (
            photo[y0, x0] * (1 - dx)[:, None] * (1 - dy)[:, None]
            + photo[y0, x0 + 1] * dx[:, None] * (1 - dy)[:, None]
            + photo[y0 + 1, x0] * (1 - dx)[:, None] * dy[:, None]
            + photo[y0 + 1, x0 + 1] * dx[:, None] * dy[:, None]
        )
    return colors, inside


def _write_ply(
    path: Path,
    xyz: np.ndarray,
    colors: np.ndarray,
    sides: np.ndarray,
    photo_valid: np.ndarray,
) -> None:
    rgb = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        for name in ("x", "y", "z"):
            handle.write(f"property float {name}\n")
        for name in ("red", "green", "blue", "side", "photo_color_valid"):
            handle.write(f"property uchar {name}\n")
        handle.write("end_header\n")
        for point, color, side, valid in zip(xyz, rgb, sides, photo_valid):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{int(side)} {int(valid)}\n"
            )


def _plot_clouds(path: Path, left: np.ndarray, right: np.ndarray, aligned: np.ndarray) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for axis, cloud, title in ((axes[0], right, "Before"), (axes[1], aligned, "After")):
        axis.scatter(left[:, 1], left[:, 2], s=2, c="#2468b4", label="left")
        axis.scatter(cloud[:, 1], cloud[:, 2], s=2, c="#e24a33", label="right")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(title)
        axis.set_xlabel("y (mm)")
        axis.set_ylabel("z (mm)")
    axes[0].legend(markerscale=4)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_overlay(
    path: Path, photo: np.ndarray, projected: np.ndarray, depth: np.ndarray, side: np.ndarray
) -> None:
    height, width = photo.shape[:2]
    inside = (
        (depth > 0.0)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < height)
    )
    figure, axis = plt.subplots(figsize=(13.333, 10), constrained_layout=True)
    axis.imshow(photo)
    axis.scatter(
        projected[inside, 0],
        projected[inside, 1],
        s=5,
        c=np.where(side[inside] == 0, "#00d5ff", "#ffe100"),
        alpha=0.7,
    )
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_axis_off()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_edges(path: Path, photo: np.ndarray, cloud: np.ndarray, camera: CameraModel) -> None:
    width = min(1000, photo.shape[1])
    height = int(round(photo.shape[0] * width / photo.shape[1]))
    distance = photo_edge_distance(photo, (width, height))
    scaled = CameraModel(
        camera.rotation_vector,
        camera.translation_mm,
        camera.focal_px * width / photo.shape[1],
        camera.principal_point_px * width / photo.shape[1],
        camera.radial_k1,
    )
    mask = render_cloud_edge_mask(cloud, scaled, (width, height))
    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    axis.imshow(np.clip(distance, 0, 20), cmap="gray_r")
    overlay = np.zeros((height, width, 4), dtype=float)
    overlay[mask] = [1.0, 0.0, 0.0, 0.85]
    axis.imshow(overlay)
    axis.set_axis_off()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _camera_document(camera: CameraModel) -> dict:
    return {
        "model": "pinhole_equal_focal_with_k1",
        "rotation_vector": camera.rotation_vector.tolist(),
        "translation_mm": camera.translation_mm.tolist(),
        "focal_px": float(camera.focal_px),
        "principal_point_px": camera.principal_point_px.tolist(),
        "radial_k1": float(camera.radial_k1),
    }


def solve(args: argparse.Namespace) -> dict:
    output = _validate_output_root(args.output_root, args.left_csv, args.right_csv)
    if output.exists() and not args.overwrite_output:
        raise ValueError("output root exists; pass --overwrite-output for this generated directory")
    if output.is_symlink():
        raise ValueError("output root must not be a symlink")
    output.mkdir(parents=True, exist_ok=True)
    sources = [Path(args.left_csv), Path(args.right_csv), Path(args.camera_photo), Path(args.registration_json)]
    hashes_before = {str(path.resolve()): _sha256(path) for path in sources}
    left = _load_cloud_csv(args.left_csv, "left")
    right = _load_cloud_csv(args.right_csv, "right")
    photo = load_camera_srgb(args.camera_photo)
    registration = json.loads(Path(args.registration_json).read_text(encoding="utf-8"))
    scan_controls = np.asarray(registration["scan_control_points"], dtype=float)
    photo_controls = np.asarray(registration["photo_control_points_px"], dtype=float)

    rigid = register_right_to_left(left.xyz, right.xyz)
    right_aligned = apply_transform(right.xyz, rigid.transform)
    merged = np.vstack((left.xyz, right_aligned))
    sides = np.concatenate((np.zeros(len(left.xyz), dtype=np.uint8), np.ones(len(right.xyz), dtype=np.uint8)))
    controls_3d = _lift_controls(left, right, right_aligned, scan_controls)
    initial = initialize_camera(
        controls_3d,
        photo_controls,
        image_size=(photo.shape[1], photo.shape[0]),
        focal_bounds_px=(0.5 * photo.shape[1], 4.0 * photo.shape[1]),
    )
    refined = refine_camera_multiscale(
        merged,
        photo,
        controls_3d,
        photo_controls,
        initial.camera,
        pyramid_widths=(500, 1000, 2000, 4000),
    )
    camera = refined.camera
    projected, depth = project_camera(merged, camera)
    colors, photo_valid = _sample_photo(photo, projected, depth)
    positive_fraction = float(np.mean(depth > 0.0))

    control_initial, _ = project_camera(controls_3d, initial.camera)
    control_final, _ = project_camera(controls_3d, camera)
    initial_errors = np.linalg.norm(control_initial - photo_controls, axis=1)
    final_errors = np.linalg.norm(control_final - photo_controls, axis=1)
    region_metrics = []
    for region_id, rows in enumerate(np.array_split(np.arange(len(photo_controls)), 3)):
        region_metrics.append(
            {
                "region": region_id,
                "initial_median_px": float(np.median(initial_errors[rows])),
                "final_median_px": float(np.median(final_errors[rows])),
                "non_worsening": bool(np.median(final_errors[rows]) <= np.median(initial_errors[rows]) + 1.0),
            }
        )
    hashes_after = {str(path.resolve()): _sha256(path) for path in sources}
    checks = {
        "rigid_registration_accepted": bool(rigid.accepted),
        "minimum_overlap_correspondences": bool(rigid.correspondence_count >= 50),
        "camera_initialization_accepted": bool(initial.accepted),
        "camera_refinement_accepted": bool(refined.accepted),
        "control_median_at_most_30_px": bool(refined.final_control_median_px <= 30.0),
        "positive_depth_fraction_at_least_0_8": bool(positive_fraction >= 0.8),
        "three_regions_non_worsening": bool(sum(item["non_worsening"] for item in region_metrics) >= 3),
        "input_hashes_unchanged": hashes_before == hashes_after,
        "output_point_count_preserved": bool(len(merged) == len(left.xyz) + len(right.xyz)),
    }
    accepted = all(checks.values())

    _atomic_json(
        output / "right_to_left_transform.json",
        _jsonable(
            {
                "accepted": rigid.accepted,
                "reason": rigid.reason,
                "transform": rigid.transform,
                "initial_metrics": rigid.initial_metrics,
                "final_metrics": rigid.final_metrics,
                "correspondence_count": rigid.correspondence_count,
            }
        ),
    )
    _atomic_json(output / "camera_model.json", _camera_document(camera))
    with (output / "optimization_history.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(refined.history[0]) if refined.history else ["width"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(refined.history)
    neutral = np.full((len(merged), 3), 0.65)
    _write_ply(output / "left_registered.ply", left.xyz, neutral[: len(left.xyz)], np.zeros(len(left.xyz), dtype=np.uint8), np.zeros(len(left.xyz), dtype=bool))
    _write_ply(output / "right_registered.ply", right_aligned, neutral[: len(right_aligned)], np.ones(len(right_aligned), dtype=np.uint8), np.zeros(len(right_aligned), dtype=bool))
    _write_ply(output / "merged_registered.ply", merged, neutral, sides, np.zeros(len(merged), dtype=bool))
    _write_ply(output / "merged_photo_colored.ply", merged, colors, sides, photo_valid)
    _plot_clouds(output / "cloud_registration_before_after.png", left.xyz, right.xyz, right_aligned)
    _plot_overlay(output / "photo_projection_overlay.png", photo, projected, depth, sides)
    _plot_edges(output / "depth_edge_alignment.png", photo, merged, camera)
    report = _jsonable(
        {
            "accepted": accepted,
            "checks": checks,
            "input_hashes": hashes_before,
            "counts": {"left": len(left.xyz), "right": len(right.xyz), "merged": len(merged), "photo_colored": int(np.count_nonzero(photo_valid))},
            "rigid": {"reason": rigid.reason, "initial_metrics": rigid.initial_metrics, "final_metrics": rigid.final_metrics, "correspondence_count": rigid.correspondence_count},
            "camera_initial": {"reason": initial.reason, "control_median_px": initial.reprojection_median_px, "positive_depth_fraction": initial.positive_depth_fraction, "model": _camera_document(initial.camera)},
            "camera_final": {"reason": refined.reason, "control_median_px": refined.final_control_median_px, "positive_depth_fraction_all_points": positive_fraction, "model": _camera_document(camera)},
            "control_errors_initial_px": initial_errors,
            "control_errors_final_px": final_errors,
            "region_metrics": region_metrics,
        }
    )
    _atomic_json(output / "registration_report.json", report)
    return report


def parse_args(argv=None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--solve", action="store_true")
    modes.add_argument("--render-only", action="store_true")
    parser.add_argument("--left-csv", type=Path, default=here / "pointcloud_output/left_pointcloud_all.csv")
    parser.add_argument("--right-csv", type=Path, default=here / "pointcloud_output/right_pointcloud_all.csv")
    parser.add_argument("--camera-photo", type=Path, default=here / "camera_show.jpg")
    parser.add_argument("--registration-json", type=Path, default=here / "calibration_registration.json")
    parser.add_argument("--output-root", type=Path, default=here / "joint_registration_output")
    parser.add_argument("--overwrite-output", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.render_only:
        report = Path(args.output_root) / "registration_report.json"
        if not report.is_file():
            raise ValueError("render-only requires an existing registration report")
        print(report.read_text(encoding="utf-8"))
        return 0
    report = solve(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
