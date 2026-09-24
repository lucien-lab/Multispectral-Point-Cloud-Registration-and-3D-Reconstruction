"""Local-only Flask API for non-destructive manual photo registration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any
import weakref

from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from matplotlib import pyplot as plt
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge
import numpy as np

from color_calibration_core import load_camera_srgb
from joint_registration_core import CameraModel, apply_transform, project_camera, zbuffer_visible
from manual_registration_core import (
    ManualRegistrationConfig,
    compose_manual_camera,
    default_manual_config,
    manual_config_document,
    parse_manual_config,
)
from register_clouds_to_photo import (
    _lift_controls,
    _load_cloud_csv,
    _sample_photo,
    _sha256,
    _validate_output_root,
)


_OUTPUT_ROOT_NAME = "joint_registration_output"
_MANUAL_CONFIG_NAME = "manual_registration.json"
_EXPORT_NAMES = (
    "manual_photo_colored.ply",
    "manual_projection_overlay.png",
    "manual_camera_model.json",
    _MANUAL_CONFIG_NAME,
    "manual_export_report.json",
)


class InputChangedError(ValueError):
    """Raised when files used to construct the in-memory session have changed."""


@dataclass(frozen=True)
class ManualRegistrationPaths:
    """Fixed base artifacts and the one permitted manual-registration output root."""

    left_csv: Path
    right_csv: Path
    photo_path: Path
    camera_model_path: Path
    right_transform_path: Path
    controls_path: Path
    report_path: Path
    output_root: Path


def _require_file(path: Path, name: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise ValueError(f"missing input {name}: {Path(path)}")
    return resolved


def _input_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    try:
        return {str(path): _sha256(path) for path in paths}
    except OSError as error:
        raise InputChangedError("an input artifact is unavailable") from error


def _json_file(path: Path, name: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name}: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"invalid {name}: {path}")
    return document


def _camera_from_document(document: dict[str, Any]) -> CameraModel:
    required = (
        "rotation_vector",
        "translation_mm",
        "focal_px",
        "principal_point_px",
    )
    missing = [name for name in required if name not in document]
    if missing:
        raise ValueError(f"camera model missing {missing[0]}")
    try:
        camera = CameraModel(
            rotation_vector=np.asarray(document["rotation_vector"], dtype=np.float64),
            translation_mm=np.asarray(document["translation_mm"], dtype=np.float64),
            focal_px=float(document["focal_px"]),
            principal_point_px=np.asarray(document["principal_point_px"], dtype=np.float64),
            radial_k1=float(document.get("radial_k1", 0.0)),
        )
        # project_camera validates all shapes, finiteness, and focal length.
        project_camera(np.zeros((1, 3), dtype=np.float64), camera)
    except (TypeError, ValueError) as error:
        raise ValueError("camera model is invalid") from error
    return camera


def _camera_document(camera: CameraModel) -> dict[str, Any]:
    return {
        "rotation_vector": camera.rotation_vector.tolist(),
        "translation_mm": camera.translation_mm.tolist(),
        "focal_px": float(camera.focal_px),
        "principal_point_px": camera.principal_point_px.tolist(),
        "radial_k1": float(camera.radial_k1),
    }


def _controls_from_document(
    document: dict[str, Any], left, right, right_aligned: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    try:
        scan = np.asarray(document["scan_control_points"], dtype=np.float64)
        pixels = np.asarray(document["photo_control_points_px"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("control file must contain scan_control_points and photo_control_points_px") from error
    if (
        scan.ndim != 2
        or scan.shape[1] != 2
        or pixels.shape != scan.shape
        or len(scan) < 6
        or not np.all(np.isfinite(np.concatenate((scan.ravel(), pixels.ravel()))))
    ):
        raise ValueError("control points must be finite matching Nx2 arrays with N >= 6")
    try:
        points = _lift_controls(left, right, right_aligned, scan)
    except ValueError as error:
        raise ValueError(f"control points cannot be lifted: {error}") from error
    return points, pixels


def _absolute_output_root(path: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(path)))
    if root.name != _OUTPUT_ROOT_NAME:
        raise ValueError(f"output root must be named {_OUTPUT_ROOT_NAME}")
    current = Path(root.anchor)
    for component in root.parts[1:]:
        current /= component
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            break
        if stat.S_ISLNK(mode):
            raise ValueError(f"output root contains symlink component: {current}")
    return root


def _open_output_root(output_root: Path, left_csv: Path, right_csv: Path) -> int:
    root = _absolute_output_root(output_root)
    _validate_output_root(root, left_csv, right_csv)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise ValueError("output root must be an existing non-symlink directory") from error
    try:
        opened = os.fstat(descriptor)
        lexical = root.lstat()
        if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            lexical.st_dev,
            lexical.st_ino,
        ):
            raise ValueError("output root changed while it was being opened")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _regular_target_exists(root_fd: int, name: str) -> bool:
    try:
        info = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        raise ValueError(f"refusing symlink output target {name}")
    if stat.S_ISDIR(info.st_mode):
        raise ValueError(f"output target is a directory: {name}")
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"output target is not a regular file: {name}")
    return True


def _manual_target_exists(root_fd: int) -> bool:
    return _regular_target_exists(root_fd, _MANUAL_CONFIG_NAME)


def _read_manual_json(root_fd: int) -> dict[str, Any]:
    _manual_target_exists(root_fd)
    try:
        descriptor = os.open(_MANUAL_CONFIG_NAME, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except OSError as error:
        raise ValueError("unable to read manual registration configuration") from error
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid manual registration configuration") from error
    if not isinstance(document, dict):
        raise ValueError("invalid manual registration configuration")
    return document


def _atomic_bytes(root_fd: int, name: str, content: bytes) -> None:
    _regular_target_exists(root_fd, name)
    temporary = f".{name}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=root_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _regular_target_exists(root_fd, name)
        os.replace(
            temporary,
            name,
            src_dir_fd=root_fd,
            dst_dir_fd=root_fd,
        )
        os.fsync(root_fd)
    except OSError as error:
        try:
            os.unlink(temporary, dir_fd=root_fd)
        except FileNotFoundError:
            pass
        raise ValueError(f"unable to atomically save output {name}") from error


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_json(root_fd: int, name: str, document: dict[str, Any]) -> None:
    _atomic_bytes(root_fd, name, _json_bytes(document))


def _ply_bytes(
    xyz: np.ndarray,
    colors: np.ndarray,
    sides: np.ndarray,
    photo_valid: np.ndarray,
) -> bytes:
    rgb = np.clip(np.rint(colors * 255.0), 0, 255).astype(np.uint8)
    lines = ["ply", "format ascii 1.0", f"element vertex {len(xyz)}"]
    lines.extend(f"property float {name}" for name in ("x", "y", "z"))
    lines.extend(f"property uchar {name}" for name in ("red", "green", "blue", "side", "photo_color_valid"))
    lines.append("end_header")
    lines.extend(
        f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
        f"{int(color[0])} {int(color[1])} {int(color[2])} {int(side)} {int(valid)}"
        for point, color, side, valid in zip(xyz, rgb, sides, photo_valid)
    )
    return ("\n".join(lines) + "\n").encode("ascii")


def _overlay_png_bytes(
    photo: np.ndarray, projected: np.ndarray, depth: np.ndarray, sides: np.ndarray
) -> bytes:
    height, width = photo.shape[:2]
    inside = (
        (depth > 0.0)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < height)
    )
    figure, axis = plt.subplots(figsize=(13.333, 10), constrained_layout=True)
    try:
        axis.imshow(photo)
        axis.scatter(
            projected[inside, 0],
            projected[inside, 1],
            s=5,
            c=np.where(sides[inside] == 0, "#00d5ff", "#ffe100"),
            alpha=0.7,
        )
        axis.set_xlim(0, width)
        axis.set_ylim(height, 0)
        axis.set_axis_off()
        output = io.BytesIO()
        figure.savefig(output, format="png", dpi=180)
        return output.getvalue()
    finally:
        plt.close(figure)


def _parse_fixed_config(document: object, camera_name: str, transform_name: str) -> ManualRegistrationConfig:
    config = parse_manual_config(document)
    if document["base_camera_model"] != camera_name:
        raise ValueError("base_camera_model must name the fixed camera model")
    if document["base_right_transform"] != transform_name:
        raise ValueError("base_right_transform must name the fixed right transform")
    return config


def _json_config_request(camera_name: str, transform_name: str) -> ManualRegistrationConfig:
    if request.mimetype != "application/json":
        abort(415, description="Content-Type must be application/json")
    try:
        document = request.get_json(silent=False)
    except BadRequest as error:
        raise ValueError("request body must contain valid JSON") from error
    return _parse_fixed_config(document, camera_name, transform_name)


def _metric_document(
    points: np.ndarray,
    controls_xyz: np.ndarray,
    controls_pixels: np.ndarray,
    image_size: tuple[int, int],
    base_camera: CameraModel,
    config: ManualRegistrationConfig,
) -> dict[str, Any]:
    camera = compose_manual_camera(base_camera, config)
    projected, depth = project_camera(points, camera)
    width, height = image_size
    in_frame = (
        (depth > 0.0)
        & (projected[:, 0] >= 0.0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < height)
    )
    visible = zbuffer_visible(projected, depth, image_size, 40.0)
    control_projected, _ = project_camera(controls_xyz, camera)
    control_error = np.linalg.norm(control_projected - controls_pixels, axis=1)
    total = len(points)
    return {
        "point_count": int(total),
        "in_frame_count": int(np.count_nonzero(in_frame)),
        "visible_count": int(np.count_nonzero(visible)),
        "coverage_fraction": float(np.count_nonzero(visible) / total),
        "control_median_px": float(np.median(control_error)),
        "control_max_px": float(np.max(control_error)),
        "focal_px": float(camera.focal_px),
    }


def create_app(paths: ManualRegistrationPaths, diagnostic_mode: bool = False) -> Flask:
    """Load immutable base artifacts and return a local manual-registration app."""
    if not isinstance(paths, ManualRegistrationPaths):
        raise ValueError("paths must be ManualRegistrationPaths")
    left_path = _require_file(paths.left_csv, "left CSV")
    right_path = _require_file(paths.right_csv, "right CSV")
    photo_path = _require_file(paths.photo_path, "photo")
    camera_path = _require_file(paths.camera_model_path, "camera model")
    transform_path = _require_file(paths.right_transform_path, "right transform")
    controls_path = _require_file(paths.controls_path, "controls")
    report_path = _require_file(paths.report_path, "registration report")
    input_paths = (
        left_path,
        right_path,
        photo_path,
        camera_path,
        transform_path,
        controls_path,
        report_path,
    )
    input_hash_snapshot = _input_hashes(input_paths)
    try:
        output_root_fd = _open_output_root(paths.output_root, left_path, right_path)
        report = _json_file(report_path, "registration report")
        accepted = report.get("accepted") is True
        reason = str(report.get("reason", "base registration was rejected"))
        if not accepted and not diagnostic_mode:
            raise ValueError(reason)
        transform_document = _json_file(transform_path, "right transform")
        if transform_document.get("accepted") is not True:
            raise ValueError("right transform is not accepted")
        try:
            transform = np.asarray(transform_document["transform"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("right transform is invalid") from error

        left = _load_cloud_csv(left_path, "left")
        right = _load_cloud_csv(right_path, "right")
        try:
            right_aligned = apply_transform(right.xyz, transform)
        except ValueError as error:
            raise ValueError(f"right transform is invalid: {error}") from error
        base_camera = _camera_from_document(_json_file(camera_path, "camera model"))
        photo = load_camera_srgb(photo_path)
        if photo.ndim != 3 or photo.shape[2] != 3 or photo.shape[0] <= 0 or photo.shape[1] <= 0:
            raise ValueError("photo is invalid")
        image_size = (int(photo.shape[1]), int(photo.shape[0]))
        controls_xyz, controls_pixels = _controls_from_document(
            _json_file(controls_path, "controls"), left, right, right_aligned
        )
        points = np.vstack((left.xyz, right_aligned))
        camera_name, transform_name = camera_path.name, transform_path.name
        config_warning = None
        if _manual_target_exists(output_root_fd):
            try:
                config = _parse_fixed_config(
                    _read_manual_json(output_root_fd), camera_name, transform_name
                )
            except ValueError as error:
                config = default_manual_config()
                config_warning = str(error)
        else:
            config = default_manual_config()
        if _input_hashes(input_paths) != input_hash_snapshot:
            raise InputChangedError("input artifacts changed while creating the session")
    except Exception:
        if "output_root_fd" in locals():
            os.close(output_root_fd)
        raise

    frontend_root = Path(__file__).resolve().parent / "manual_registration_web"
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
    weakref.finalize(app, os.close, output_root_fd)

    def config_document(value: ManualRegistrationConfig) -> dict[str, Any]:
        return manual_config_document(value, camera_name, transform_name)

    def export_document(submitted: ManualRegistrationConfig) -> dict[str, Any]:
        document = config_document(submitted)
        if _input_hashes(input_paths) != input_hash_snapshot:
            raise InputChangedError("input artifacts changed since session startup")
        camera = compose_manual_camera(base_camera, submitted)
        projected, depth = project_camera(points, camera)
        colors, photo_valid = _sample_photo(photo, projected, depth)
        sides = np.concatenate(
            (
                np.zeros(len(left.xyz), dtype=np.uint8),
                np.ones(len(right_aligned), dtype=np.uint8),
            )
        )
        metrics = _metric_document(
            points, controls_xyz, controls_pixels, image_size, base_camera, submitted
        )
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("manual export metrics are non-finite")

        # All generated artifacts use the already-open trusted output directory.
        _atomic_bytes(
            output_root_fd,
            "manual_photo_colored.ply",
            _ply_bytes(points, colors, sides, photo_valid),
        )
        _atomic_bytes(
            output_root_fd,
            "manual_projection_overlay.png",
            _overlay_png_bytes(photo, projected, depth, sides),
        )
        _atomic_json(output_root_fd, "manual_camera_model.json", _camera_document(camera))
        config_bytes = _json_bytes(document)
        _atomic_bytes(output_root_fd, _MANUAL_CONFIG_NAME, config_bytes)

        hashes_after = _input_hashes(input_paths)
        if hashes_after != input_hash_snapshot:
            raise InputChangedError("input artifacts changed before export completion")
        checks = {
            "input_hashes_unchanged": input_hash_snapshot == hashes_after,
            "output_point_count_preserved": len(points) == len(left.xyz) + len(right_aligned),
            "metrics_are_finite": True,
            "photo_color_count_is_bounded": 0 <= int(np.count_nonzero(photo_valid)) <= len(points),
        }
        accepted = bool(all(checks.values()))
        if not accepted:
            raise ValueError("manual export did not satisfy acceptance checks")
        report = {
            "accepted": accepted,
            "checks": checks,
            "metrics": metrics,
            "input_hashes": input_hash_snapshot,
            "config_hash": hashlib.sha256(config_bytes).hexdigest(),
            "counts": {
                "left": len(left.xyz),
                "right": len(right_aligned),
                "merged": len(points),
                "photo_colored": int(np.count_nonzero(photo_valid)),
            },
            "files": list(_EXPORT_NAMES),
        }
        # The report is the completion marker and is intentionally written last.
        _atomic_json(output_root_fd, "manual_export_report.json", report)
        return report

    @app.get("/")
    def index():
        if not (frontend_root / "index.html").is_file():
            abort(404)
        return send_from_directory(frontend_root, "index.html")

    @app.get("/manual_registration_web/<path:filename>")
    def frontend_file(filename: str):
        if not frontend_root.is_dir():
            abort(404)
        return send_from_directory(frontend_root, filename)

    @app.get("/api/photo")
    def photo_endpoint():
        return send_file(photo_path, conditional=True)

    @app.get("/api/session")
    def session_endpoint():
        result = {
            "photo": {"width": image_size[0], "height": image_size[1]},
            "base_camera": _camera_document(base_camera),
            "config": config_document(config),
            "left_points": left.xyz.tolist(),
            "right_points": right_aligned.tolist(),
            "controls": {
                "points_xyz": controls_xyz.tolist(),
                "pixels_xy": controls_pixels.tolist(),
            },
            "base_status": {"accepted": accepted, "reason": reason},
        }
        if config_warning is not None:
            result["config_warning"] = config_warning
        return jsonify(result)

    @app.post("/api/save")
    def save_endpoint():
        nonlocal config, config_warning
        try:
            submitted = _json_config_request(camera_name, transform_name)
            document = config_document(submitted)
            _atomic_json(output_root_fd, _MANUAL_CONFIG_NAME, document)
        except ValueError as error:
            return jsonify({"error": str(error)}), 400
        config = submitted
        config_warning = None
        return jsonify({"config": document, "saved": True})

    @app.post("/api/metrics")
    def metrics_endpoint():
        try:
            submitted = _json_config_request(camera_name, transform_name)
            return jsonify(
                _metric_document(
                    points,
                    controls_xyz,
                    controls_pixels,
                    image_size,
                    base_camera,
                    submitted,
                )
            )
        except ValueError as error:
            return jsonify({"error": str(error)}), 400

    @app.post("/api/export")
    def export_endpoint():
        nonlocal config, config_warning
        try:
            submitted = _json_config_request(camera_name, transform_name)
            report = export_document(submitted)
        except InputChangedError as error:
            return jsonify({"error": str(error), "exported": False}), 409
        except ValueError as error:
            return jsonify({"error": str(error), "exported": False}), 400
        config = submitted
        config_warning = None
        return jsonify({"exported": True, "files": report["files"], "report": report})

    @app.errorhandler(RequestEntityTooLarge)
    def oversized_request(error):
        return jsonify({"error": "request body exceeds 64 KiB"}), 413

    return app
