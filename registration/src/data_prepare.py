"""将已有精炼点云转换为实验使用的统一米制快照。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


WAVELENGTHS_NM = np.array([495, 696, 600, 803, 545, 642], dtype=np.int32)


def prepare_snapshot(input_path: Path, output_path: Path, side: str) -> dict[str, object]:
    """读取精炼 NPZ，验证字段并写出统一的实验快照。"""
    if side not in {"left", "right"}:
        raise ValueError("side 必须是 left 或 right")
    with np.load(input_path) as loaded:
        required = {"xyz_mm", "rgb", "reflectance", "wavelengths_nm"}
        missing = required.difference(loaded.files)
        if missing:
            raise ValueError(f"{input_path} 缺少字段: {sorted(missing)}")
        xyz_m = np.asarray(loaded["xyz_mm"], dtype=np.float64) / 1000.0
        rgb = np.clip(np.asarray(loaded["rgb"], dtype=np.float64), 0.0, 1.0)
        spectral6 = np.asarray(loaded["reflectance"], dtype=np.float64)
        wavelengths = np.asarray(loaded["wavelengths_nm"], dtype=np.int32)
        scan_index = (
            np.asarray(loaded["source_scan_index"], dtype=np.int32)
            if "source_scan_index" in loaded.files
            else np.arange(1, xyz_m.shape[0] + 1, dtype=np.int32)
        )

    if xyz_m.ndim != 2 or xyz_m.shape[1] != 3:
        raise ValueError("xyz_mm 必须是 N×3")
    if rgb.shape != xyz_m.shape:
        raise ValueError("rgb 必须与 xyz 具有相同的 N×3 形状")
    if spectral6.shape != (xyz_m.shape[0], 6):
        raise ValueError("reflectance 必须是 N×6")
    if not np.array_equal(wavelengths, WAVELENGTHS_NM):
        raise ValueError(f"波长顺序必须为 {WAVELENGTHS_NM.tolist()}")
    if not (np.isfinite(xyz_m).all() and np.isfinite(rgb).all() and np.isfinite(spectral6).all()):
        raise ValueError("实验输入包含 NaN 或 Inf")

    feature12 = np.column_stack((xyz_m, rgb, spectral6))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        xyz_m=xyz_m,
        rgb=rgb,
        spectral6=spectral6,
        wavelengths_nm=wavelengths,
        source_scan_index=scan_index,
        feature12=feature12,
        side=np.array(side),
        coordinate_unit=np.array("m"),
    )
    return {
        "side": side,
        "point_count": int(xyz_m.shape[0]),
        "coordinate_unit": "m",
        "wavelengths_nm": wavelengths.tolist(),
        "output": str(output_path),
    }


def load_prepared(path: Path) -> dict[str, np.ndarray]:
    """加载统一快照并复制数组，避免持有关闭后的 NPZ 句柄。"""
    with np.load(path) as loaded:
        return {key: np.asarray(loaded[key]).copy() for key in loaded.files}
