"""SE(3) 随机扰动、应用、求逆和误差计算。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Perturbation:
    matrix: np.ndarray
    euler_deg: np.ndarray
    translation_m: np.ndarray


def generate_perturbations(
    count: int,
    seed: int,
    max_rotation_deg: float,
    max_translation_m: float,
) -> list[Perturbation]:
    """按 XYZ 欧拉角和逐轴平移上限生成可复现的刚体扰动。"""
    if count <= 0:
        raise ValueError("count 必须为正数")
    rng = np.random.default_rng(seed)
    result: list[Perturbation] = []
    for _ in range(count):
        euler_deg = rng.uniform(-max_rotation_deg, max_rotation_deg, size=3)
        translation_m = rng.uniform(-max_translation_m, max_translation_m, size=3)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_euler("xyz", euler_deg, degrees=True).as_matrix()
        matrix[:3, 3] = translation_m
        result.append(Perturbation(matrix, euler_deg, translation_m))
    return result


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def transform_errors(estimated: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    """计算旋转测地角和欧氏平移误差。"""
    estimated = np.asarray(estimated, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    rotation_delta = estimated[:3, :3] @ ground_truth[:3, :3].T
    return {
        "rotation_error_deg": float(
            np.degrees(Rotation.from_matrix(rotation_delta).magnitude())
        ),
        "translation_error_m": float(
            np.linalg.norm(estimated[:3, 3] - ground_truth[:3, 3])
        ),
    }
