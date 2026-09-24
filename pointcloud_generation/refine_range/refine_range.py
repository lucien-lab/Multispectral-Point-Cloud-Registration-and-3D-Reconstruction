#!/usr/bin/env python3
"""筛选指定距离范围内的左右多光谱点云，并输出分析结果。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree


ROW_KEYS = (
    "xyz_mm",
    "rgb",
    "rgb_uint8",
    "range_m",
    "reflectance",
    "cie_xyz",
    "angle_x_raw",
    "angle_y_raw",
    "azimuth_rad",
    "elevation_rad",
    "correlation_peak_index",
    "valid",
)
DEFAULT_MIN_RANGE_M = 6.3
DEFAULT_MAX_RANGE_M = 10.0


def range_mask(
    ranges_m: np.ndarray,
    valid: np.ndarray,
    min_range_m: float,
    max_range_m: float,
) -> np.ndarray:
    """保留原本有效且处于闭区间 [min_range_m, max_range_m] 的点。"""
    ranges_m = np.asarray(ranges_m)
    valid = np.asarray(valid, dtype=bool)
    if ranges_m.shape != valid.shape:
        raise ValueError("range_m 与 valid 的形状不一致")
    if min_range_m > max_range_m:
        raise ValueError("最小距离不能大于最大距离")
    return (
        valid
        & np.isfinite(ranges_m)
        & (ranges_m >= min_range_m)
        & (ranges_m <= max_range_m)
    )


def filter_payload(
    payload: dict[str, np.ndarray],
    min_range_m: float,
    max_range_m: float,
) -> dict[str, np.ndarray]:
    """筛选逐点数组，并保留波长等全局信息。"""
    ranges_m = np.asarray(payload["range_m"])
    valid = np.asarray(payload["valid"], dtype=bool)
    keep = range_mask(ranges_m, valid, min_range_m, max_range_m)
    result: dict[str, np.ndarray] = {
        "source_scan_index": np.flatnonzero(keep).astype(np.int32) + 1,
        "wavelengths_nm": np.asarray(payload["wavelengths_nm"]).copy(),
    }
    for key in ROW_KEYS:
        if key in payload:
            result[key] = np.asarray(payload[key])[keep].copy()
    result["valid"] = np.ones(int(keep.sum()), dtype=bool)
    result["source_keep_mask"] = keep
    return result


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def save_npz(path: Path, payload: dict[str, np.ndarray], side_code: int) -> None:
    arrays = {
        key: value
        for key, value in payload.items()
        if key != "source_keep_mask"
    }
    arrays["side_code"] = np.full(
        payload["range_m"].shape[0],
        side_code,
        dtype=np.uint8,
    )
    np.savez_compressed(path, **arrays)


def csv_header(wavelengths_nm: np.ndarray) -> list[str]:
    return [
        "side",
        "source_scan_index",
        "x_mm",
        "y_mm",
        "z_mm",
        "range_m",
        "angle_x_raw",
        "angle_y_raw",
        "azimuth_rad",
        "elevation_rad",
        "correlation_peak_index",
        "r",
        "g",
        "b",
        "red",
        "green",
        "blue",
        *[f"rfl_{int(wavelength)}_nm" for wavelength in wavelengths_nm],
    ]


def iter_rows(side: str, payload: dict[str, np.ndarray]):
    count = payload["range_m"].shape[0]
    rgb = payload["rgb"]
    rgb8 = payload["rgb_uint8"]
    for index in range(count):
        yield [
            side,
            int(payload["source_scan_index"][index]),
            *payload["xyz_mm"][index].tolist(),
            float(payload["range_m"][index]),
            int(payload["angle_x_raw"][index]),
            int(payload["angle_y_raw"][index]),
            float(payload["azimuth_rad"][index]),
            float(payload["elevation_rad"][index]),
            int(payload["correlation_peak_index"][index]),
            *rgb[index].tolist(),
            *rgb8[index].astype(int).tolist(),
            *payload["reflectance"][index].tolist(),
        ]


def write_csv(
    path: Path,
    sides: list[tuple[str, dict[str, np.ndarray]]],
) -> None:
    wavelengths_nm = sides[0][1]["wavelengths_nm"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(csv_header(wavelengths_nm))
        for side, payload in sides:
            writer.writerows(iter_rows(side, payload))


def write_ply(
    path: Path,
    sides: list[tuple[str, dict[str, np.ndarray]]],
) -> int:
    wavelengths_nm = sides[0][1]["wavelengths_nm"]
    count = sum(payload["range_m"].shape[0] for _, payload in sides)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment Range-refined multispectral point cloud\n")
        handle.write(f"element vertex {count}\n")
        for axis in ("x", "y", "z"):
            handle.write(f"property float {axis}\n")
        for color in ("red", "green", "blue"):
            handle.write(f"property uchar {color}\n")
        handle.write("property float range_m\n")
        handle.write("property int source_scan_index\n")
        handle.write("property uchar side_code\n")
        for wavelength in wavelengths_nm:
            handle.write(f"property float rfl_{int(wavelength)}_nm\n")
        handle.write("end_header\n")

        for side, payload in sides:
            side_code = 0 if side == "left" else 1
            for index in range(payload["range_m"].shape[0]):
                xyz = payload["xyz_mm"][index]
                rgb8 = payload["rgb_uint8"][index]
                fields = [
                    *(f"{value:.6f}" for value in xyz),
                    *(str(int(value)) for value in rgb8),
                    f"{payload['range_m'][index]:.9f}",
                    str(int(payload["source_scan_index"][index])),
                    str(side_code),
                    *(f"{value:.9g}" for value in payload["reflectance"][index]),
                ]
                handle.write(" ".join(fields) + "\n")
    return count


def side_analysis(
    source: dict[str, np.ndarray],
    refined: dict[str, np.ndarray],
) -> dict[str, object]:
    source_valid = np.asarray(source["valid"], dtype=bool)
    xyz = refined["xyz_mm"]
    ranges = refined["range_m"]
    reflectance = refined["reflectance"]
    wavelengths = refined["wavelengths_nm"]
    spectral = {}
    for channel, wavelength in enumerate(wavelengths):
        values = reflectance[:, channel]
        spectral[str(int(wavelength))] = {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return {
        "source_total_frames": int(source["range_m"].shape[0]),
        "source_valid_points": int(source_valid.sum()),
        "retained_points": int(ranges.shape[0]),
        "removed_valid_points_outside_range": int(source_valid.sum() - ranges.shape[0]),
        "retained_percent_of_valid": float(100.0 * ranges.shape[0] / source_valid.sum()),
        "range_m": {
            "min": float(np.min(ranges)),
            "median": float(np.median(ranges)),
            "max": float(np.max(ranges)),
        },
        "bbox_mm": {
            "min": xyz.min(axis=0).tolist(),
            "max": xyz.max(axis=0).tolist(),
        },
        "spectral_reflectance": spectral,
    }


def exact_xyz_matches(left: np.ndarray, right: np.ndarray) -> int:
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
    left_view = np.ascontiguousarray(left, dtype=np.float64).view(dtype).ravel()
    right_view = np.ascontiguousarray(right, dtype=np.float64).view(dtype).ravel()
    return int(np.intersect1d(left_view, right_view).shape[0])


def overlap_analysis(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
) -> dict[str, object]:
    left_distances = cKDTree(right_xyz).query(left_xyz, k=1)[0]
    right_distances = cKDTree(left_xyz).query(right_xyz, k=1)[0]
    thresholds = (20, 30, 50)
    by_threshold = {}
    for threshold in thresholds:
        left_count = int(np.count_nonzero(left_distances <= threshold))
        right_count = int(np.count_nonzero(right_distances <= threshold))
        by_threshold[str(threshold)] = {
            "left_to_right_count": left_count,
            "left_to_right_percent": float(100.0 * left_count / left_xyz.shape[0]),
            "right_to_left_count": right_count,
            "right_to_left_percent": float(100.0 * right_count / right_xyz.shape[0]),
        }
    return {
        "exact_xyz_matches": exact_xyz_matches(left_xyz, right_xyz),
        "nearest_distance_mm": {
            "minimum": float(min(left_distances.min(), right_distances.min())),
            "left_to_right_median": float(np.median(left_distances)),
            "right_to_left_median": float(np.median(right_distances)),
        },
        "within_threshold_mm": by_threshold,
    }


def concatenate_payloads(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    merged = {
        key: np.concatenate((left[key], right[key]), axis=0)
        for key in ROW_KEYS
        if key in left and key in right
    }
    merged["source_scan_index"] = np.concatenate(
        (left["source_scan_index"], right["source_scan_index"])
    )
    merged["wavelengths_nm"] = left["wavelengths_nm"].copy()
    merged["side_code"] = np.concatenate(
        (
            np.zeros(left["range_m"].shape[0], dtype=np.uint8),
            np.ones(right["range_m"].shape[0], dtype=np.uint8),
        )
    )
    return merged


def write_preview(
    path: Path,
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    min_range_m: float,
    max_range_m: float,
) -> None:
    fig = plt.figure(figsize=(10, 8), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        left["xyz_mm"][:, 0],
        left["xyz_mm"][:, 1],
        left["xyz_mm"][:, 2],
        s=4,
        c="#2474B5",
        label=f"left ({left['xyz_mm'].shape[0]})",
        depthshade=False,
    )
    ax.scatter(
        right["xyz_mm"][:, 0],
        right["xyz_mm"][:, 1],
        right["xyz_mm"][:, 2],
        s=4,
        c="#E6812F",
        label=f"right ({right['xyz_mm'].shape[0]})",
        depthshade=False,
    )
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(
        f"Range-refined point clouds: {min_range_m:.1f}-{max_range_m:.1f} m"
    )
    ax.legend()
    ax.view_init(elev=20, azim=-55)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_spectral_plot(
    path: Path,
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
) -> None:
    wavelengths = left["wavelengths_nm"]
    order = np.argsort(wavelengths)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    ax.plot(
        wavelengths[order],
        np.median(left["reflectance"], axis=0)[order],
        "o-",
        label="left median",
    )
    ax.plot(
        wavelengths[order],
        np.median(right["reflectance"], axis=0)[order],
        "o-",
        label="right median",
    )
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Median reflectance")
    ax.set_title("Six-band reflectance after range filtering")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_readme(
    path: Path,
    analysis: dict[str, object],
) -> None:
    left = analysis["sides"]["left"]
    right = analysis["sides"]["right"]
    overlap = analysis["overlap"]
    min_range_m = analysis["filter"]["min_range_m_inclusive"]
    max_range_m = analysis["filter"]["max_range_m_inclusive"]
    text = f"""# 2026-07-26 距离筛选点云

筛选条件：保留原本有效且距离位于闭区间 `{min_range_m:.1f} m <= range <= {max_range_m:.1f} m` 的点。
原始重建结果未被修改。

## 点数

| 侧别 | 原始帧数 | 原始有效点 | 筛选后点数 | 保留比例 |
|---|---:|---:|---:|---:|
| left | {left['source_total_frames']} | {left['source_valid_points']} | {left['retained_points']} | {left['retained_percent_of_valid']:.2f}% |
| right | {right['source_total_frames']} | {right['source_valid_points']} | {right['retained_points']} | {right['retained_percent_of_valid']:.2f}% |

波长顺序：`{analysis['wavelength_order_nm']}` nm。

## 左右重叠

- XYZ 完全相同的点：{overlap['exact_xyz_matches']}
- 最近左右点距离：{overlap['nearest_distance_mm']['minimum']:.3f} mm
- 推荐同点阈值 30 mm 下：
  - left -> right：{overlap['within_threshold_mm']['30']['left_to_right_count']} 点
  - right -> left：{overlap['within_threshold_mm']['30']['right_to_left_count']} 点

## 文件

- `left_refined.*`、`right_refined.*`：分别保存左右点云。
- `merged_refined.*`：左右点云合并，`side_code=0` 表示 left，`side_code=1` 表示 right。
- NPZ：无损数组，适合 Python。
- CSV：逐点数值，适合 MATLAB、Excel 和人工检查。
- PLY：包含 XYZ、RGB、距离和六波段反射率，可直接可视化。
- `analysis.json`：完整数值统计。
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=here.parent / "2026_7_26")
    parser.add_argument("--output-dir", type=Path, default=here)
    parser.add_argument("--min-range", type=float, default=DEFAULT_MIN_RANGE_M)
    parser.add_argument("--max-range", type=float, default=DEFAULT_MAX_RANGE_M)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {}
    refined = {}
    for side in ("left", "right"):
        source_path = args.source_dir / f"{side}_pointcloud_all.npz"
        sources[side] = load_npz(source_path)
        refined[side] = filter_payload(
            sources[side],
            args.min_range,
            args.max_range,
        )
        save_npz(
            args.output_dir / f"{side}_refined.npz",
            refined[side],
            0 if side == "left" else 1,
        )
        write_csv(
            args.output_dir / f"{side}_refined.csv",
            [(side, refined[side])],
        )
        write_ply(
            args.output_dir / f"{side}_refined.ply",
            [(side, refined[side])],
        )

    merged = concatenate_payloads(refined["left"], refined["right"])
    np.savez_compressed(args.output_dir / "merged_refined.npz", **merged)
    write_csv(
        args.output_dir / "merged_refined.csv",
        [("left", refined["left"]), ("right", refined["right"])],
    )
    write_ply(
        args.output_dir / "merged_refined.ply",
        [("left", refined["left"]), ("right", refined["right"])],
    )

    analysis = {
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "filter": {
            "min_range_m_inclusive": args.min_range,
            "max_range_m_inclusive": args.max_range,
            "requires_source_valid": True,
        },
        "wavelength_order_nm": refined["left"]["wavelengths_nm"].astype(int).tolist(),
        "sides": {
            side: side_analysis(sources[side], refined[side])
            for side in ("left", "right")
        },
        "overlap": overlap_analysis(
            refined["left"]["xyz_mm"],
            refined["right"]["xyz_mm"],
        ),
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_readme(args.output_dir / "README.md", analysis)
    write_preview(
        args.output_dir / "left_right_refined_preview.png",
        refined["left"],
        refined["right"],
        args.min_range,
        args.max_range,
    )
    write_spectral_plot(
        args.output_dir / "six_band_median_comparison.png",
        refined["left"],
        refined["right"],
    )

    for side in ("left", "right"):
        stats = analysis["sides"][side]
        print(
            f"{side}: {stats['source_valid_points']} -> "
            f"{stats['retained_points']} points"
        )
    print(f"outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
