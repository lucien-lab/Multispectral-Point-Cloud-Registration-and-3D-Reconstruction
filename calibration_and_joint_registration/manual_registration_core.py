"""Validated manual camera-registration deltas and composition helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import numbers

import numpy as np
from scipy.spatial.transform import Rotation

from joint_registration_core import CameraModel


_SCHEMA_VERSION = 1
_DOCUMENT_KEYS = frozenset(
    {
        "schema_version",
        "translation_delta_mm",
        "rotation_delta_deg",
        "manual_transform_4x4",
        "projection_zoom",
        "point_size",
        "point_opacity",
        "photo_brightness",
        "show_left",
        "show_right",
        "base_camera_model",
        "base_right_transform",
    }
)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _vector(value: object, name: str, limit: float) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=object)
    if array.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(
        _finite_number(component, f"{name}[{index}]")
        for index, component in enumerate(array)
    )
    if any(abs(component) > limit for component in result):
        raise ValueError(f"{name} values must be in [-{limit:g}, {limit:g}]")
    return result


def _finite_matrix(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    if array.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    return np.asarray(
        [
            [_finite_number(array[row, column], f"{name}[{row}][{column}]") for column in range(4)]
            for row in range(4)
        ],
        dtype=np.float64,
    )


def _validate_manual_matrix(value: object, expected: np.ndarray) -> None:
    matrix = _finite_matrix(value, "manual_transform_4x4")
    rotation = matrix[:3, :3]
    rigid = (
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=1e-9)
        and np.allclose(rotation.T @ rotation, np.eye(3), rtol=1e-9, atol=1e-9)
        and np.isclose(np.linalg.det(rotation), 1.0, rtol=1e-9, atol=1e-9)
    )
    if not rigid:
        raise ValueError("manual_transform_4x4 must be a rigid SE(3) transform")
    if not np.allclose(matrix, expected, rtol=1e-9, atol=1e-9):
        raise ValueError(
            "manual_transform_4x4 must be consistent with translation_delta_mm and rotation_delta_deg"
        )


@dataclass(frozen=True)
class ManualDisplayPreferences:
    """Non-geometric display controls for the manual-registration view."""

    show_left: bool = True
    show_right: bool = True
    point_opacity: float = 0.75
    point_size: float = 3.0
    photo_brightness: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.show_left, (bool, np.bool_)):
            raise ValueError("show_left must be a boolean")
        if not isinstance(self.show_right, (bool, np.bool_)):
            raise ValueError("show_right must be a boolean")
        opacity = _finite_number(self.point_opacity, "point_opacity")
        point_size = _finite_number(self.point_size, "point_size")
        photo_brightness = _finite_number(
            self.photo_brightness, "photo_brightness"
        )
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("point_opacity must be in [0, 1]")
        if point_size <= 0.0:
            raise ValueError("point_size must be positive")
        if photo_brightness <= 0.0:
            raise ValueError("photo_brightness must be positive")
        object.__setattr__(self, "show_left", bool(self.show_left))
        object.__setattr__(self, "show_right", bool(self.show_right))
        object.__setattr__(self, "point_opacity", opacity)
        object.__setattr__(self, "point_size", point_size)
        object.__setattr__(self, "photo_brightness", photo_brightness)


@dataclass(frozen=True)
class ManualRegistrationConfig:
    """Camera-frame delta, zoom, and independent display preferences."""

    schema_version: int = _SCHEMA_VERSION
    translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_euler_xyz_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)
    projection_zoom: float = 1.0
    display: ManualDisplayPreferences = field(default_factory=ManualDisplayPreferences)

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, (bool, np.bool_)) or not isinstance(
            self.schema_version, numbers.Integral
        ):
            raise ValueError("schema_version must be an integer")
        if int(self.schema_version) != _SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        zoom = _finite_number(self.projection_zoom, "projection_zoom")
        if not 0.5 <= zoom <= 2.0:
            raise ValueError("projection_zoom must be in [0.5, 2.0]")
        if not isinstance(self.display, ManualDisplayPreferences):
            raise ValueError("display must be ManualDisplayPreferences")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        object.__setattr__(
            self, "translation_mm", _vector(self.translation_mm, "translation_mm", 2000.0)
        )
        object.__setattr__(
            self,
            "rotation_euler_xyz_degrees",
            _vector(
                self.rotation_euler_xyz_degrees,
                "rotation_euler_xyz_degrees",
                30.0,
            ),
        )
        object.__setattr__(self, "projection_zoom", zoom)


def default_manual_config() -> ManualRegistrationConfig:
    """Return a neutral manual-registration configuration."""
    return ManualRegistrationConfig()


def parse_manual_config(document: dict) -> ManualRegistrationConfig:
    """Parse one exact-version manual-registration JSON document."""
    if not isinstance(document, dict):
        raise ValueError("manual registration document must be an object")
    unknown = set(document) - _DOCUMENT_KEYS
    if unknown:
        raise ValueError(f"manual registration document contains unknown keys: {sorted(unknown)}")
    missing = _DOCUMENT_KEYS - set(document)
    if missing:
        raise ValueError(f"manual registration document is missing keys: {sorted(missing)}")
    for name in ("base_camera_model", "base_right_transform"):
        if not isinstance(document[name], str) or not document[name]:
            raise ValueError(f"{name} must be a non-empty string")

    config = ManualRegistrationConfig(
        schema_version=document["schema_version"],
        translation_mm=document["translation_delta_mm"],
        rotation_euler_xyz_degrees=document["rotation_delta_deg"],
        projection_zoom=document["projection_zoom"],
        display=ManualDisplayPreferences(
            show_left=document["show_left"],
            show_right=document["show_right"],
            point_opacity=document["point_opacity"],
            point_size=document["point_size"],
            photo_brightness=document["photo_brightness"],
        ),
    )
    _validate_manual_matrix(
        document["manual_transform_4x4"], manual_delta_transform(config)
    )
    return config


def manual_delta_transform(config: ManualRegistrationConfig) -> np.ndarray:
    """Return the camera-frame homogeneous transform represented by ``config``."""
    if not isinstance(config, ManualRegistrationConfig):
        raise ValueError("config must be ManualRegistrationConfig")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler(
        "xyz", config.rotation_euler_xyz_degrees, degrees=True
    ).as_matrix()
    transform[:3, 3] = config.translation_mm
    return transform


def _validated_base_camera(base: CameraModel) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, float]:
    if not isinstance(base, CameraModel):
        raise ValueError("base must be CameraModel")
    rotation_vector = np.asarray(base.rotation_vector, dtype=np.float64)
    translation = np.asarray(base.translation_mm, dtype=np.float64)
    principal_point = np.asarray(base.principal_point_px, dtype=np.float64)
    if (
        rotation_vector.shape != (3,)
        or translation.shape != (3,)
        or principal_point.shape != (2,)
    ):
        raise ValueError("base camera vectors have invalid shape")
    focal = _finite_number(base.focal_px, "base.focal_px")
    radial_k1 = _finite_number(base.radial_k1, "base.radial_k1")
    if focal <= 0.0 or not np.all(
        np.isfinite(np.concatenate((rotation_vector, translation, principal_point)))
    ):
        raise ValueError("base camera parameters must be finite with positive focal_px")
    return rotation_vector.copy(), translation.copy(), focal, principal_point.copy(), radial_k1


def compose_manual_camera(
    base: CameraModel, config: ManualRegistrationConfig
) -> CameraModel:
    """Apply the camera-frame delta to an existing point-cloud camera model."""
    rotation_vector, translation, focal, principal_point, radial_k1 = _validated_base_camera(base)
    delta = manual_delta_transform(config)
    rotation = delta[:3, :3] @ Rotation.from_rotvec(rotation_vector).as_matrix()
    return CameraModel(
        rotation_vector=Rotation.from_matrix(rotation).as_rotvec(),
        translation_mm=delta[:3, :3] @ translation + delta[:3, 3],
        focal_px=focal * config.projection_zoom,
        principal_point_px=principal_point,
        radial_k1=radial_k1,
    )


def manual_config_document(
    config: ManualRegistrationConfig,
    base_camera_name: str,
    base_transform_name: str,
) -> dict:
    """Serialize a validated config with the named non-destructive base artifacts."""
    if not isinstance(config, ManualRegistrationConfig):
        raise ValueError("config must be ManualRegistrationConfig")
    for name, value in (
        ("base_camera_name", base_camera_name),
        ("base_transform_name", base_transform_name),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    return {
        "schema_version": config.schema_version,
        "translation_delta_mm": list(config.translation_mm),
        "rotation_delta_deg": list(config.rotation_euler_xyz_degrees),
        "manual_transform_4x4": manual_delta_transform(config).tolist(),
        "projection_zoom": config.projection_zoom,
        "point_size": config.display.point_size,
        "point_opacity": config.display.point_opacity,
        "photo_brightness": config.display.photo_brightness,
        "show_left": config.display.show_left,
        "show_right": config.display.show_right,
        "base_camera_model": base_camera_name,
        "base_right_transform": base_transform_name,
    }
