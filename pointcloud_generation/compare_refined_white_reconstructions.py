#!/usr/bin/env python3
"""Compare non-destructive point-cloud recoloring for refined white references."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from generate_pointcloud import WAVELENGTHS_NM, spectral_rgb


HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "pointcloud_output"
REGISTRATION_DIR = HERE / "joint_registration_output"
OUTPUT_DIR = HERE / "white_candidate_reconstruction_comparison"

CURRENT_WHITE = np.array(
    [531.2975, 396.4109, 425.0606, 361.2250, 572.4702, 477.3115],
    dtype=np.float64,
)
WHITE_SCHEMES = {
    "baseline_current": CURRENT_WHITE,
    "candidate_five_class": np.array(
        [160.1871424121157, 59.46195480036788, 283.7910930694781,
         361.225, 1407.1647404638325, 2386.4709641906925],
        dtype=np.float64,
    ),
    "candidate_white_wall": np.array(
        [85.43235066178009, 122.78490929747464, 1884.3109650143142,
         361.225, 2557.321585357129, 240.8707390983203],
        dtype=np.float64,
    ),
}
LABELS = {
    "baseline_current": "Current white reference",
    "candidate_five_class": "Photo fit: five classes",
    "candidate_white_wall": "Photo fit: white wall only",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def load_source_side(side: str, right_transform: np.ndarray) -> dict[str, np.ndarray]:
    path = INPUT_DIR / f"{side}_pointcloud_all.npz"
    with np.load(path) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    if side == "right":
        result["xyz_mm"] = apply_transform(result["xyz_mm"], right_transform)
    return result


def recolor(source: dict[str, np.ndarray], white: np.ndarray) -> dict[str, np.ndarray]:
    raw_amplitude = source["reflectance"] * CURRENT_WHITE
    reflectance = raw_amplitude / white
    rgb, cie_xyz, _ = spectral_rgb(reflectance)
    result = dict(source)
    result["reflectance"] = reflectance
    result["cie_xyz"] = cie_xyz
    result["rgb"] = rgb
    result["rgb_uint8"] = np.rint(rgb * 255.0).astype(np.uint8)
    return result


def write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **data)


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, reflectance: np.ndarray) -> None:
    rgb8 = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment Refined left-right geometry with experimental white recoloring\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        for wavelength in WAVELENGTHS_NM:
            handle.write(f"property float rfl_{int(wavelength)}_nm\n")
        handle.write("end_header\n")
        for point, color, spectrum in zip(xyz, rgb8, reflectance):
            fields = [
                *(f"{value:.6f}" for value in point),
                *(str(int(value)) for value in color),
                *(f"{value:.9g}" for value in spectrum),
            ]
            handle.write(" ".join(fields) + "\n")


def camera_projection(points: np.ndarray, camera: dict) -> tuple[np.ndarray, np.ndarray]:
    rotation = Rotation.from_rotvec(np.asarray(camera["rotation_vector"], dtype=np.float64)).as_matrix()
    camera_xyz = points @ rotation.T + np.asarray(camera["translation_mm"], dtype=np.float64)
    depth = camera_xyz[:, 2]
    normalized = camera_xyz[:, :2] / np.where(
        np.abs(depth[:, None]) < 1e-12, 1e-12, depth[:, None]
    )
    radius_squared = np.sum(normalized * normalized, axis=1)
    normalized *= (1.0 + float(camera.get("radial_k1", 0.0)) * radius_squared)[:, None]
    pixels = (
        normalized * float(camera["focal_px"])
        + np.asarray(camera["principal_point_px"], dtype=np.float64)
    )
    return pixels, depth


def visible_projection(
    points: np.ndarray, rgb: np.ndarray, camera: dict, image_shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    pixels, depth = camera_projection(points, camera)
    rounded = np.rint(pixels).astype(np.int64)
    inside = (
        (depth > 0.0)
        & (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    )
    indices = np.flatnonzero(inside)
    if indices.size == 0:
        return np.empty((0, 2)), np.empty((0, 3))
    key = rounded[indices, 1] * width + rounded[indices, 0]
    order = np.lexsort((depth[indices], key))
    ordered = indices[order]
    ordered_key = key[order]
    keep = np.r_[True, ordered_key[1:] != ordered_key[:-1]]
    chosen = ordered[keep]
    return pixels[chosen], rgb[chosen]


def color_metrics(rgb: np.ndarray) -> dict[str, object]:
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    return {
        "mean_rgb": np.mean(rgb, axis=0).tolist(),
        "median_rgb": np.median(rgb, axis=0).tolist(),
        "mean_luminance": float(np.mean(luminance)),
        "near_black_fraction": float(np.mean(luminance < 0.05)),
        "any_channel_clipped_fraction": float(np.mean(np.any(rgb >= 1.0 - 1e-12, axis=1))),
        "mean_channel_range": float(np.mean(np.ptp(rgb, axis=1))),
    }


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert D65 sRGB to CIELAB without an optional image package."""
    rgb = np.asarray(rgb, dtype=np.float64)
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float64,
    ).T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta ** 3,
        np.cbrt(xyz),
        xyz / (3.0 * delta * delta) + 4.0 / 29.0,
    )
    return np.column_stack(
        (
            116.0 * transformed[:, 1] - 16.0,
            500.0 * (transformed[:, 0] - transformed[:, 1]),
            200.0 * (transformed[:, 1] - transformed[:, 2]),
        )
    )


def photo_match_metrics(
    xyz: np.ndarray, rgb: np.ndarray, photo: np.ndarray, camera: dict
) -> dict[str, float | int | str]:
    pixels, visible_rgb = visible_projection(xyz, rgb, camera, photo.shape[:2])
    indices = np.rint(pixels).astype(np.int64)
    target_rgb = photo[indices[:, 1], indices[:, 0]]
    delta_e76 = np.linalg.norm(srgb_to_lab(visible_rgb) - srgb_to_lab(target_rgb), axis=1)
    srgb_l2 = np.linalg.norm(visible_rgb - target_rgb, axis=1)
    return {
        "visible_points": int(len(pixels)),
        "median_delta_e76": float(np.median(delta_e76)),
        "mean_delta_e76": float(np.mean(delta_e76)),
        "median_srgb_l2": float(np.median(srgb_l2)),
        "caution": "All projected points are area-weighted; shadows, occlusion, and residual registration error remain.",
    }


def style_3d_axis(axis, bounds: tuple[np.ndarray, np.ndarray]) -> None:
    lo, hi = bounds
    axis.set_xlim(lo[0], hi[0])
    axis.set_ylim(lo[1], hi[1])
    axis.set_zlim(lo[2], hi[2])
    axis.set_box_aspect(np.maximum(hi - lo, 1.0))
    axis.view_init(elev=20, azim=-50)
    axis.set_xlabel("X / mm")
    axis.set_ylabel("Y / mm")
    axis.set_zlabel("Z / mm")
    axis.set_facecolor("black")
    axis.grid(False)


def plot_3d_comparison(path: Path, schemes: dict[str, dict[str, np.ndarray]]) -> None:
    all_xyz = np.concatenate([schemes["baseline_current"][side]["xyz"] for side in ("left", "right")])
    lo, hi = np.percentile(all_xyz, [1.0, 99.0], axis=0)
    fig = plt.figure(figsize=(17, 5.8), dpi=180, facecolor="white")
    for index, (name, sides) in enumerate(schemes.items(), start=1):
        xyz = np.concatenate([sides[side]["xyz"] for side in ("left", "right")])
        rgb = np.concatenate([sides[side]["rgb"] for side in ("left", "right")])
        axis = fig.add_subplot(1, 3, index, projection="3d")
        axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb, s=3.2, depthshade=False)
        axis.set_title(LABELS[name])
        style_3d_axis(axis, (lo, hi))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_photo_comparison(
    path: Path,
    photo: np.ndarray,
    camera: dict,
    schemes: dict[str, dict[str, np.ndarray]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0), dpi=180, constrained_layout=True)
    for axis, (name, sides) in zip(axes, schemes.items()):
        xyz = np.concatenate([sides[side]["xyz"] for side in ("left", "right")])
        rgb = np.concatenate([sides[side]["rgb"] for side in ("left", "right")])
        pixels, visible_rgb = visible_projection(xyz, rgb, camera, photo.shape[:2])
        axis.imshow(np.clip(photo * 0.72, 0.0, 1.0))
        if len(pixels):
            axis.scatter(pixels[:, 0], pixels[:, 1], c=visible_rgb, s=13, alpha=0.9, linewidths=0)
        axis.set_xlim(0, photo.shape[1])
        axis.set_ylim(photo.shape[0], 0)
        axis.set_title(f"{LABELS[name]} ({len(pixels)} visible)")
        axis.axis("off")
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    right_document = load_json(REGISTRATION_DIR / "right_to_left_transform.json")
    right_transform = np.asarray(right_document["transform"], dtype=np.float64)
    camera = load_json(REGISTRATION_DIR / "manual_refined_camera_model.json")
    photo = np.asarray(Image.open(HERE / "camera_show.jpg").convert("RGB"), dtype=np.float64) / 255.0
    sources = {
        side: load_source_side(side, right_transform)
        for side in ("left", "right")
    }

    plotted: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    report_schemes: dict[str, object] = {}
    for name, white in WHITE_SCHEMES.items():
        scheme_dir = OUTPUT_DIR / name
        scheme_dir.mkdir(parents=True, exist_ok=True)
        plotted[name] = {}
        scheme_report = {"white_board_amplitudes": white.tolist(), "sides": {}}
        merged_xyz, merged_rgb, merged_rfl = [], [], []
        for side in ("left", "right"):
            recolored = recolor(sources[side], white)
            valid = recolored["valid"].astype(bool)
            write_npz(scheme_dir / f"{side}_pointcloud_all.npz", recolored)
            xyz = recolored["xyz_mm"][valid]
            rgb = recolored["rgb"][valid]
            rfl = recolored["reflectance"][valid]
            write_ply(scheme_dir / f"{side}_pointcloud.ply", xyz, rgb, rfl)
            plotted[name][side] = {"xyz": xyz, "rgb": rgb}
            merged_xyz.append(xyz)
            merged_rgb.append(rgb)
            merged_rfl.append(rfl)
            scheme_report["sides"][side] = {
                "valid_points": int(valid.sum()),
                **color_metrics(rgb),
            }
        xyz = np.concatenate(merged_xyz)
        rgb = np.concatenate(merged_rgb)
        rfl = np.concatenate(merged_rfl)
        write_ply(scheme_dir / "merged_pointcloud.ply", xyz, rgb, rfl)
        scheme_report["merged"] = {"valid_points": len(xyz), **color_metrics(rgb)}
        scheme_report["photo_projection_match"] = photo_match_metrics(
            xyz, rgb, photo, camera
        )
        report_schemes[name] = scheme_report

    plot_3d_comparison(OUTPUT_DIR / "three_white_schemes_3d.png", plotted)
    plot_photo_comparison(OUTPUT_DIR / "three_white_schemes_photo_overlay.png", photo, camera, plotted)
    source_paths = [
        INPUT_DIR / "left_pointcloud_all.npz",
        INPUT_DIR / "right_pointcloud_all.npz",
        REGISTRATION_DIR / "right_to_left_transform.json",
        REGISTRATION_DIR / "manual_registration_refined.json",
        REGISTRATION_DIR / "manual_refined_camera_model.json",
        HERE / "camera_show.jpg",
    ]
    report = {
        "purpose": "Non-destructive comparison only; rejected photo-derived white candidates are not written back.",
        "wavelength_order_nm": WAVELENGTHS_NM.tolist(),
        "source_amplitude_recovery": "raw_amplitude = source_reflectance * current_white",
        "geometry": {
            "left_frame": "unchanged",
            "right_frame": "accepted right_to_left_transform applied",
            "camera": "manual_refined_camera_model.json",
        },
        "schemes": report_schemes,
        "inputs_sha256": {str(path.resolve()): sha256(path) for path in source_paths},
        "outputs": [
            "three_white_schemes_3d.png",
            "three_white_schemes_photo_overlay.png",
            "baseline_current/merged_pointcloud.ply",
            "candidate_five_class/merged_pointcloud.ply",
            "candidate_white_wall/merged_pointcloud.ply",
        ],
    }
    (OUTPUT_DIR / "comparison_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        name: values["merged"] for name, values in report_schemes.items()
    }, ensure_ascii=False, indent=2))
    print(f"output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
