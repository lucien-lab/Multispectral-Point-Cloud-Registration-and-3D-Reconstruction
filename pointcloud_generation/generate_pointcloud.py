#!/usr/bin/env python3
"""将 20260328 ADC 数据转换为带光谱与 RGB 的左右点云。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d


WAVELENGTHS_NM = np.array([495, 696, 600, 803, 545, 642], dtype=np.int32)
TEMPLATE_POSITIONS = np.array([1, 147, 252, 408, 590, 795], dtype=np.int32)
# 旧白板参考幅值（保留备查）：
# [124.1064096774194, 120.2428838709678, 115.0127903225807,
#  166.3828838709677, 115.8136612903226, 226.6485516129032]
# 当前顺序对应 [495, 696, 600, 803, 545, 642] nm。
WHITE_BOARD_AMPLITUDES = np.array(
    [531.2975, 396.4109, 425.0606, 361.2250, 572.4702, 477.3115],
    dtype=np.float64,
)
# CIE 1931 标准色度观察者（2°），按 WAVELENGTHS_NM 顺序排列。
# 官方数据集 DOI: 10.25039/CIE.DS.xvudnb9b，原始来源 CIE 018:2019 Table 6。
CIE_1931_2DEG_XYZ = np.array(
    [
        [0.014700000000, 0.2586000000000, 0.353300000000],  # 495 nm
        [0.014790640000, 0.0053430590000, 0.000000000000],  # 696 nm
        [1.062200000000, 0.6310000000000, 0.000800000000],  # 600 nm
        [0.000008308358, 0.0000030003000, 0.000000000000],  # 803 nm
        [0.359700000000, 0.9803000000000, 0.013400000000],  # 545 nm
        [0.412098000000, 0.1596464000000, 0.000016200000],  # 642 nm
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class Config:
    sample_length: int = 13312
    sample_frequency_ghz: float = 10.4
    mainwave_threshold: int = 100
    max_range_m: float = 28.0
    refractive_index: float = 1.0
    echo_padding_samples: int = 800
    moving_average_window: int = 5
    detection_start: int = 300
    valid_peak_min: int = 100
    valid_peak_max: int = 1000
    segment_width: int = 60
    echo_pulse_scope: int = 30
    offset_scope: int = 90
    scanner_height_mm: float = 1900.0
    horizontal_scale: float = 1.25

    @property
    def valid_signal_length(self) -> int:
        max_tof_ns = self.refractive_index * self.max_range_m * 100.0 / 15.0
        return round(self.sample_frequency_ghz * max_tof_ns + self.echo_padding_samples)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser(
        description="复现 MATLAB 超光谱激光雷达流程，生成 left/right 彩色点云。"
    )
    parser.add_argument("--data-dir", type=Path, default=here, help="left.bin/right.bin 所在目录")
    parser.add_argument("--angle-dir", type=Path, default=root / "angle", help="角度文本目录")
    parser.add_argument("--output-dir", type=Path, default=here / "pointcloud_output", help="输出目录")
    parser.add_argument("--sides", nargs="+", choices=("left", "right"), default=("left", "right"))
    parser.add_argument("--batch-size", type=int, default=256, help="ADC 批处理帧数")
    parser.add_argument("--include-invalid-ply", action="store_true", help="PLY 也写入 MATLAB 判定为无效的帧")
    parser.add_argument("--no-preview", action="store_true", help="不生成 PNG 预览图")
    return parser.parse_args()


def read_angles(path: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    tokens = path.read_text(encoding="ascii").split()
    values = np.fromiter((int(token, 16) for token in tokens), dtype=np.uint16)
    groups = values.size // 6
    if groups < count:
        raise ValueError(f"{path} 只有 {groups} 组角度，少于 ADC 的 {count} 帧")
    values = values[: groups * 6].reshape(-1, 6)[:count]
    angle_x_raw = values[:, 3].astype(np.uint32) * 256 + values[:, 2]
    angle_y_raw = values[:, 5].astype(np.uint32) * 256 + values[:, 4]
    return angle_x_raw, angle_y_raw


def moving_average_5(signals: np.ndarray) -> np.ndarray:
    # MATLAB conv(x, ones(1,5)/5, 'same')：边界使用零填充。
    return convolve1d(
        signals.astype(np.float64),
        np.ones(5, dtype=np.float64) / 5.0,
        axis=1,
        mode="constant",
        cval=0.0,
    )


def sparse_xcorr_peak(signals: np.ndarray, detection_start: int) -> np.ndarray:
    """等价于 MATLAB xcorr(signal_det, sparse_template) 的非负延迟部分。"""
    work = signals.copy()
    work[:, :detection_start] = 0.0
    n_rows, n_cols = work.shape
    correlation = np.zeros((n_rows, n_cols), dtype=np.float64)
    for matlab_position in TEMPLATE_POSITIONS:
        position = int(matlab_position - 1)
        correlation[:, : n_cols - position] += work[:, position:]
    return np.argmax(correlation, axis=1).astype(np.int32) + 1


def process_adc(bin_path: Path, config: Config, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values_per_frame = config.sample_length
    file_values = bin_path.stat().st_size // np.dtype("<i2").itemsize
    if file_values % values_per_frame:
        raise ValueError(f"{bin_path} 大小不是 {values_per_frame} 个 int16 的整数倍")
    frame_count = file_values // values_per_frame
    raw = np.memmap(bin_path, dtype="<i2", mode="r", shape=(frame_count, values_per_frame))

    peak_indices = np.zeros(frame_count, dtype=np.int32)
    reflectance = np.zeros((frame_count, 6), dtype=np.float64)
    mainwave_ok = np.zeros(frame_count, dtype=bool)
    valid_len = config.valid_signal_length

    for first in range(0, frame_count, batch_size):
        last = min(first + batch_size, frame_count)
        block = np.asarray(raw[first:last])
        above = block > config.mainwave_threshold
        has_mainwave = np.any(above, axis=1)
        last_above = values_per_frame - 1 - np.argmax(above[:, ::-1], axis=1)

        # 对应 A2026_5_5.m：risingedge 附近 30 点内找峰，再按其原公式截窗。
        local_offsets = np.arange(-12, 18, dtype=np.int32)
        pulse_indices = last_above[:, None] + local_offsets[None, :]
        pulse_bounds_ok = (pulse_indices[:, 0] >= 0) & (pulse_indices[:, -1] < values_per_frame)
        safe_pulse_indices = np.clip(pulse_indices, 0, values_per_frame - 1)
        pulse = np.take_along_axis(block, safe_pulse_indices, axis=1)
        local_peak = np.argmax(pulse, axis=1)
        starts = last_above + local_peak - 49
        window_ok = (starts >= 0) & (starts + valid_len <= values_per_frame)
        batch_ok = has_mainwave & pulse_bounds_ok & window_ok
        mainwave_ok[first:last] = batch_ok

        safe_starts = np.clip(starts, 0, values_per_frame - valid_len)
        window_indices = safe_starts[:, None] + np.arange(valid_len, dtype=np.int32)[None, :]
        signals = np.take_along_axis(block, window_indices, axis=1)
        filtered = moving_average_5(signals)
        filtered[~batch_ok] = 0.0

        peaks = sparse_xcorr_peak(filtered, config.detection_start)
        peaks[~batch_ok] = 0
        peak_indices[first:last] = peaks

        peak_ok = batch_ok & (peaks >= config.valid_peak_min) & (peaks <= config.valid_peak_max)
        for row in np.flatnonzero(peak_ok):
            centers = TEMPLATE_POSITIONS + peaks[row] - 1  # MATLAB 1-based
            for channel, center in enumerate(centers):
                segment_start = int(center - config.segment_width // 2)
                segment_end = int(center + config.segment_width // 2)
                # MATLAB 区间包含两端，共 61 点。
                lo = max(segment_start - 1, 0)
                hi = min(segment_end, valid_len)
                segment = filtered[row, lo:hi]
                if segment.size == 0:
                    continue
                peak0 = lo + int(np.argmax(segment))
                peak_value = filtered[row, peak0]
                eps_start = max(peak0 - config.echo_pulse_scope // 2, 0)
                eps_end = min(peak0 + config.echo_pulse_scope // 2, valid_len - 1)
                offset_start = max(peak0 - config.offset_scope // 2, 0)
                offset_end = min(peak0 + config.offset_scope // 2, valid_len - 1)
                left_baseline = filtered[row, offset_start : eps_start + 1]
                right_baseline = filtered[row, eps_end : offset_end + 1]
                if left_baseline.size and right_baseline.size:
                    baseline = 0.5 * (left_baseline.mean() + right_baseline.mean())
                    reflectance[first + row, channel] = (
                        peak_value - baseline
                    ) / WHITE_BOARD_AMPLITUDES[channel]

        print(f"  {bin_path.name}: {last}/{frame_count} 帧", flush=True)

    del raw
    return peak_indices, reflectance, mainwave_ok


def coordinates(
    peak_indices: np.ndarray,
    angle_x_raw: np.ndarray,
    angle_y_raw: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ranges_m = 0.15 * peak_indices / config.sample_frequency_ghz / config.refractive_index
    azimuth = config.horizontal_scale * 2.0 * np.pi * (angle_x_raw - 19650.0) / 40000.0
    elevation = 2.0 * np.pi * (36500.0 - angle_y_raw) / 1200000.0
    horizontal_mm = 1000.0 * ranges_m * np.cos(elevation)
    xyz = np.column_stack(
        (
            horizontal_mm * np.cos(azimuth),
            -horizontal_mm * np.sin(azimuth),
            1000.0 * ranges_m * np.sin(elevation) + config.scanner_height_mm,
        )
    )
    return xyz, ranges_m, azimuth, elevation


def spectral_rgb(reflectance: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    xyz_weights = CIE_1931_2DEG_XYZ
    # 803 nm 为近红外，不参与可见色；其余通道按 MATLAB 的线性加权方式处理。
    visible = WAVELENGTHS_NM != 803
    normalization = np.sum(xyz_weights[visible, 1])
    cie_xyz = reflectance[:, visible] @ xyz_weights[visible] / normalization
    # 标准 CIE XYZ（D65）到线性 sRGB。
    xyz_to_linear_srgb = np.array(
        [[3.2404542, -1.5371385, -0.4985314],
         [-0.9692660, 1.8760108, 0.0415560],
         [0.0556434, -0.2040259, 1.0572252]],
        dtype=np.float64,
    )
    rgb_linear = cie_xyz @ xyz_to_linear_srgb.T
    rgb_linear = np.maximum(rgb_linear, 0.0)
    # IEC 61966-2-1 sRGB 光电转换函数。
    rgb = np.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
    )
    rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, cie_xyz, "CIE_1931_2deg_official_to_sRGB_D65"


def output_columns() -> list[str]:
    base = [
        "scan_index", "x_mm", "y_mm", "z_mm", "range_m",
        "angle_x_raw", "angle_y_raw", "azimuth_rad", "elevation_rad",
        "correlation_peak_index", "valid", "r", "g", "b", "red", "green", "blue",
    ]
    return base + [f"rfl_{wavelength}_nm" for wavelength in WAVELENGTHS_NM]


def assembled_rows(
    xyz: np.ndarray,
    ranges_m: np.ndarray,
    angle_x_raw: np.ndarray,
    angle_y_raw: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
    peak_indices: np.ndarray,
    valid: np.ndarray,
    rgb: np.ndarray,
    reflectance: np.ndarray,
) -> np.ndarray:
    rgb8 = np.rint(rgb * 255.0).astype(np.uint8)
    return np.column_stack(
        (
            np.arange(1, xyz.shape[0] + 1), xyz, ranges_m, angle_x_raw, angle_y_raw,
            azimuth, elevation, peak_indices, valid.astype(np.uint8), rgb, rgb8, reflectance,
        )
    )


def write_csv(path: Path, rows: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(output_columns())
        writer.writerows(rows)


def write_ply(path: Path, rows: np.ndarray, valid: np.ndarray, include_invalid: bool) -> int:
    selected = np.ones(valid.size, dtype=bool) if include_invalid else valid
    data = rows[selected]
    rfl_start = 17
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment Generated from MATLAB hyperspectral lidar pipeline\n")
        handle.write(f"element vertex {data.shape[0]}\n")
        for axis in ("x", "y", "z"):
            handle.write(f"property float {axis}\n")
        for color in ("red", "green", "blue"):
            handle.write(f"property uchar {color}\n")
        handle.write("property float range_m\nproperty int scan_index\n")
        for wavelength in WAVELENGTHS_NM:
            handle.write(f"property float rfl_{wavelength}_nm\n")
        handle.write("end_header\n")
        for row in data:
            fields = [
                f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}",
                str(int(row[14])), str(int(row[15])), str(int(row[16])),
                f"{row[4]:.9f}", str(int(row[0])),
            ]
            fields.extend(f"{value:.9g}" for value in row[rfl_start : rfl_start + 6])
            handle.write(" ".join(fields) + "\n")
    return data.shape[0]


def write_preview(path: Path, xyz: np.ndarray, rgb: np.ndarray, valid: np.ndarray, title: str) -> None:
    fig = plt.figure(figsize=(9, 7), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(xyz[valid, 0], xyz[valid, 1], xyz[valid, 2], c=rgb[valid], s=3, depthshade=False)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=20, azim=-50)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def process_side(side: str, args: argparse.Namespace, config: Config) -> dict[str, object]:
    bin_path = args.data_dir / f"{side}.bin"
    angle_name = "left.txt" if side == "left" else "rigt.txt"
    angle_path = args.angle_dir / angle_name
    if not bin_path.exists() or not angle_path.exists():
        raise FileNotFoundError(f"缺少 {bin_path} 或 {angle_path}")

    frame_count = bin_path.stat().st_size // (2 * config.sample_length)
    print(f"开始处理 {side}: {frame_count} 帧")
    angle_x_raw, angle_y_raw = read_angles(angle_path, frame_count)
    peak_indices, reflectance, mainwave_ok = process_adc(bin_path, config, args.batch_size)
    xyz, ranges_m, azimuth, elevation = coordinates(
        peak_indices, angle_x_raw, angle_y_raw, config
    )
    rgb, cie_xyz, color_source = spectral_rgb(reflectance)
    valid = (
        mainwave_ok
        & (peak_indices >= config.valid_peak_min)
        & (peak_indices <= config.valid_peak_max)
        & (ranges_m <= config.max_range_m)
        & np.all(np.isfinite(xyz), axis=1)
    )
    rows = assembled_rows(
        xyz, ranges_m, angle_x_raw, angle_y_raw, azimuth, elevation,
        peak_indices, valid, rgb, reflectance,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{side}_pointcloud_all.csv"
    ply_path = args.output_dir / f"{side}_pointcloud.ply"
    all_ply_path = args.output_dir / f"{side}_pointcloud_all.ply"
    npz_path = args.output_dir / f"{side}_pointcloud_all.npz"
    write_csv(csv_path, rows)
    ply_points = write_ply(ply_path, rows, valid, args.include_invalid_ply)
    all_ply_points = write_ply(all_ply_path, rows, valid, True)
    np.savez_compressed(
        npz_path,
        xyz_mm=xyz,
        rgb=rgb,
        rgb_uint8=np.rint(rgb * 255).astype(np.uint8),
        range_m=ranges_m,
        reflectance=reflectance,
        wavelengths_nm=WAVELENGTHS_NM,
        cie_xyz=cie_xyz,
        angle_x_raw=angle_x_raw,
        angle_y_raw=angle_y_raw,
        azimuth_rad=azimuth,
        elevation_rad=elevation,
        correlation_peak_index=peak_indices,
        valid=valid,
    )
    if not args.no_preview:
        write_preview(args.output_dir / f"{side}_pointcloud_preview.png", xyz, rgb, valid, f"{side} hyperspectral point cloud")

    result = {
        "side": side,
        "input_bin": str(bin_path.resolve()),
        "input_angle": str(angle_path.resolve()),
        "frames": int(frame_count),
        "valid_points": int(valid.sum()),
        "ply_points": int(ply_points),
        "all_ply_points": int(all_ply_points),
        "color_source": color_source,
        "range_m_valid_min": float(ranges_m[valid].min()) if np.any(valid) else None,
        "range_m_valid_max": float(ranges_m[valid].max()) if np.any(valid) else None,
        "outputs": {
            "csv": str(csv_path),
            "valid_ply": str(ply_path),
            "all_ply": str(all_ply_path),
            "npz": str(npz_path),
        },
    }
    print(f"完成 {side}: 有效点 {result['valid_points']}/{frame_count}")
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")
    config = Config()
    results = [process_side(side, args, config) for side in args.sides]
    metadata = {
        "algorithm": "Python port of A2026_5_5.m",
        "config": asdict(config),
        "wavelength_order_nm": WAVELENGTHS_NM.tolist(),
        "cmf_note": "Official CIE 1931 2-degree colour-matching functions from DOI 10.25039/CIE.DS.xvudnb9b (CIE 018:2019 Table 6). XYZ is converted to sRGB D65 with the standard matrix and sRGB transfer function. 803 nm is excluded from visible RGB.",
        "valid_rule": f"mainwave found and {config.valid_peak_min} <= correlation peak <= {config.valid_peak_max} and range <= {config.max_range_m} m",
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"全部完成，元数据: {metadata_path}")


if __name__ == "__main__":
    main()
