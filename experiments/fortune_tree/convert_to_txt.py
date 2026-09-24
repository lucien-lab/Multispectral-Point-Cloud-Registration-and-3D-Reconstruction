#!/usr/bin/env python3
"""通用工具：任意发财树扫描文件夹 → data_convert/ 下 RGB 点云 txt（存储时 ×4）。

输入一个扫描文件夹名（需含 el_re.txt / sbq/*.csv / gp/specData/spec_*.txt），
输出逗号分隔的 8 列点云：id,x,y,z,R,G,B,类别。

×4 说明：与 MATLAB 原版 scatter3(...,*4) 对齐，标定反射率在存储时即乘 4
（--multiply 可调），显示端（view_txt.py）不再放大。

用法示例：
    python convert_to_txt.py 20260201三维建模发财树60度
    python convert_to_txt.py 20260204发财树顶端
    python convert_to_txt.py --all            # 批量转换当前目录下所有扫描文件夹
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks


DEFAULT_BANDS = (957, 730, 552)  # R,G,B 对应的光谱行号（1-based）
SCRIPT_DIR = Path(__file__).resolve().parent
PAPERS_ROOT = SCRIPT_DIR.parent
# 默认标定文件：优先用扫描目录内的 dark.txt / white.txt，否则回退到 papers 根目录
DEFAULT_CALIBRATION_DIR = PAPERS_ROOT / "20260130分类场景八万点"
DEFAULT_WHITE_DIR = PAPERS_ROOT / "白板"


def load_numeric(path: Path) -> np.ndarray:
    try:
        return np.loadtxt(path, delimiter=",", ndmin=2)
    except (OSError, ValueError):
        try:
            return np.loadtxt(path, ndmin=2)
        except ValueError:
            # 部分 sbq CSV 文件末尾有两个科学计数法数值粘连无换行，
            # 从文本中安全地提取所有数值 token。
            text = path.read_text(encoding="utf-8")
            tokens = re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text)
            if not tokens:
                raise
            return np.asarray([float(token) for token in tokens], dtype=float)[:, None]


def calibration_paths(
    scan_dir: Path,
    dark_file: Path | None,
    white_file: Path | None,
) -> tuple[Path, Path]:
    """优先使用扫描目录内的标定文件，否则回退到 papers 根目录的参考标定。"""
    dark_path = dark_file or scan_dir / "dark.txt"
    white_path = white_file or scan_dir / "white.txt"
    if dark_file is None and not dark_path.is_file():
        dark_path = DEFAULT_CALIBRATION_DIR / "dark.txt"
    if white_file is None and not white_path.is_file():
        white_path = DEFAULT_WHITE_DIR / "white4-1.txt"
    if not dark_path.is_file():
        raise FileNotFoundError(f"找不到暗场标定文件：{dark_path}")
    if not white_path.is_file():
        raise FileNotFoundError(f"找不到白场标定文件：{white_path}")
    return dark_path, white_path


def calibration_values(
    dark_path: Path,
    white_path: Path,
    bands: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """rgb = (raw - dark) / (white - dark)，返回各波段的暗场值与分母。"""
    dark = load_numeric(dark_path)
    white = load_numeric(white_path)
    max_band = max(bands)
    if dark.shape[0] < max_band or white.shape[0] < max_band:
        raise ValueError(f"白场或暗场数据不足以提供波段 {bands}")
    if dark.shape[1] < 2 or white.shape[1] < 2:
        raise ValueError("白场和暗场数据至少需要两列")

    zero_based_bands = np.asarray(bands) - 1
    dark_values = dark[zero_based_bands, 1]
    denominator = white[zero_based_bands, 1] - dark_values
    if not np.isfinite(dark_values).all() or not np.isfinite(denominator).all():
        raise ValueError("白场或暗场在目标波段包含非有限值")
    if np.any(np.isclose(denominator, 0)):
        raise ValueError("white-dark 在目标波段存在零值，无法归一化")
    return dark_values, denominator


def point_coordinates(scan_dir: Path, angles: np.ndarray) -> dict[int, np.ndarray]:
    """从 sbq 测距数据 + el_re.txt 角度数据计算三维坐标（复现 MATLAB 原版逻辑）。"""
    coordinates: dict[int, np.ndarray] = {}
    sbq_dir = scan_dir / "sbq"
    if not sbq_dir.is_dir():
        raise FileNotFoundError(f"缺少 sbq 目录：{sbq_dir}")
    for path in sorted(sbq_dir.glob("*.csv"), key=lambda item: int(item.stem)):
        point_id = int(path.stem)
        if point_id < 1 or point_id > len(angles):
            continue
        raw = load_numeric(path).ravel()
        if raw.size != 1000 or not np.isfinite(raw).all():
            continue
        peaks, _ = find_peaks(raw[1:], distance=800)
        if peaks.size == 0:
            continue

        # 波峰位置 → 飞行时间 → 距离（采样率/声速折算为 400）
        distance = (peaks[0] + 1) / 400.0
        horizontal = np.deg2rad(angles[point_id - 1, 1])
        altitude = np.deg2rad(angles[point_id - 1, 2])
        x = distance * np.cos(altitude) * np.sin(horizontal)
        y = distance * np.cos(altitude) * np.cos(horizontal)
        z = distance * np.sin(altitude)

        # 绕 Y 轴旋转 180 度（与 MATLAB 原版对齐）
        coordinates[point_id] = np.array([-x, y, -z])
    return coordinates


def is_scan_directory(scan_dir: Path) -> bool:
    """判断一个目录是否是有效的扫描数据文件夹。"""
    return (
        scan_dir.is_dir()
        and (scan_dir / "el_re.txt").is_file()
        and (scan_dir / "sbq").is_dir()
        and (scan_dir / "gp" / "specData").is_dir()
    )


def process(
    scan_dir: Path,
    output: Path,
    bands: tuple[int, int, int] = DEFAULT_BANDS,
    dark_file: Path | None = None,
    white_file: Path | None = None,
    multiply: float = 4.0,
) -> Path:
    """扫描文件夹 → 点云 txt。RGB 在存储时乘以 multiply（默认 4），与 MATLAB 原版对齐。"""
    angles = load_numeric(scan_dir / "el_re.txt")
    if angles.shape[1] < 3:
        raise ValueError("el_re.txt 至少需要三列：编号、水平角、俯仰角")
    coordinates = point_coordinates(scan_dir, angles)
    if not coordinates:
        raise ValueError("未能从 sbq/*.csv 计算任何有效三维坐标")

    dark_path, white_path = calibration_paths(scan_dir, dark_file, white_file)
    dark_values, denominator = calibration_values(dark_path, white_path, bands)

    rows: list[np.ndarray] = []
    zero_based_bands = np.asarray(bands) - 1
    spec_dir = scan_dir / "gp" / "specData"
    for path in sorted(
        spec_dir.glob("spec_*.txt"), key=lambda item: int(item.stem.split("_")[1])
    ):
        point_id = int(path.stem.split("_")[1])
        xyz = coordinates.get(point_id)
        if xyz is None:
            continue
        spectrum = load_numeric(path)
        if spectrum.shape[0] < max(bands) or spectrum.shape[1] < 2:
            continue
        raw_rgb = spectrum[zero_based_bands, 1]
        rgba = (raw_rgb - dark_values) / denominator
        rgb = np.clip(rgba * multiply, 0.0, 1.0)  # 存储时 ×4（与 MATLAB scatter3(*4) 对齐）
        # id,x,y,z,R,G,B,类别（类别默认 0）
        rows.append(np.concatenate(([point_id], xyz, rgb, [0])))

    if not rows:
        raise ValueError("没有可合并的坐标和光谱数据")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        np.vstack(rows),
        delimiter=",",
        fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%d"],
    )
    print(f"[OK] {scan_dir.name}: {len(rows)} 个点 → {output}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="扫描文件夹 → data_convert/ 下的 RGB 反射率点云 txt"
    )
    parser.add_argument(
        "scan_dir",
        nargs="?",
        type=str,
        help="扫描文件夹名（如 20260201三维建模发财树60度），可带路径",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="批量转换本脚本所在目录下所有有效的扫描文件夹",
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="输出 txt 路径（默认 data_convert/<文件夹名>.txt）"
    )
    parser.add_argument(
        "--bands",
        nargs=3,
        type=int,
        default=DEFAULT_BANDS,
        metavar=("R", "G", "B"),
        help="R/G/B 三个波段的光谱行号（1-based），默认 957 730 552",
    )
    parser.add_argument(
        "--multiply",
        type=float,
        default=4.0,
        help="存储时 RGB 倍率（与 MATLAB 原版 ×4 对齐，默认 4.0；设为 1 则存原始反射率）",
    )
    parser.add_argument("--dark", type=Path, help="暗场文件（默认自动回退）")
    parser.add_argument("--white", type=Path, help="白场文件（默认自动回退）")
    args = parser.parse_args()

    if not args.all and not args.scan_dir:
        parser.error("请提供扫描文件夹名，或使用 --all 批量转换")

    try:
        if args.all:
            targets = sorted(
                p for p in SCRIPT_DIR.iterdir()
                if is_scan_directory(p) and p.name != "data_convert"
            )
            if not targets:
                print("未找到任何有效的扫描文件夹")
                return 1
            print(f"批量转换 {len(targets)} 个扫描文件夹……")
            for target in targets:
                process(
                    target,
                    args.output
                    or SCRIPT_DIR / "data_convert" / f"{target.name}.txt",
                    tuple(args.bands),
                    args.dark,
                    args.white,
                    args.multiply,
                )
        else:
            scan_dir = Path(args.scan_dir)
            if not scan_dir.is_absolute():
                # 允许输入纯文件夹名（自动在当前目录查找）
                candidate = SCRIPT_DIR / scan_dir
                if candidate.is_dir():
                    scan_dir = candidate
                else:
                    scan_dir = scan_dir.resolve()
            if not is_scan_directory(scan_dir):
                parser.error(
                    f"不是有效的扫描文件夹（缺少 el_re.txt / sbq / gp/specData）：{scan_dir}"
                )
            output = args.output or SCRIPT_DIR / "data_convert" / f"{scan_dir.name}.txt"
            process(scan_dir, output, tuple(args.bands), args.dark, args.white, args.multiply)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())