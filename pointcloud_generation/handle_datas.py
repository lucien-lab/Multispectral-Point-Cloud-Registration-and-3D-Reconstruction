#!/usr/bin/env python3
"""Convert per-point spectral files into a colourized point-cloud table.

This is the Python equivalent of ``handle_datas.m``.  Paths are discovered
relative to a dataset directory, so the script can be reused for other
datasets with the same file layout.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


def _load_numeric(path: Path) -> np.ndarray:
    """Load whitespace- or comma-separated numeric text as a 2-D array."""
    try:
        data = np.loadtxt(path, delimiter=None, ndmin=2)
    except ValueError:
        try:
            data = np.loadtxt(path, delimiter=",", ndmin=2)
        except (OSError, ValueError) as exc:
            raise ValueError(f"无法解析数据文件：{path}") from exc
    except OSError as exc:
        raise FileNotFoundError(f"找不到数据文件：{path}") from exc
    if data.size == 0:
        raise ValueError(f"数据文件为空：{path}")
    return np.asarray(data, dtype=float)


def _find_numbered_files(directory: Path, pattern: str) -> list[tuple[int, Path]]:
    regex = re.compile(pattern)
    found: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = regex.fullmatch(path.name)
        if match and path.is_file():
            found.append((int(match.group(1)), path))
    return sorted(found)


def process_dataset(
    dataset_dir: str | Path,
    *,
    output: str | Path | None = None,
    coordinate_file: str | Path | None = None,
    dark_file: str | Path | None = None,
    white_file: str | Path | None = None,
    spectral_dir: str | Path | None = None,
    bands: tuple[int, int, int] = (957, 730, 552),
    no_calibration: bool = False,
    class_label: int = 0,
) -> Path:
    """Process one dataset and return the generated output path.

    ``bands`` use MATLAB's 1-based row numbering.  Missing spectral files and
    point IDs absent from the coordinate file are skipped with a summary.
    """
    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"数据集目录不存在：{root}")

    coord_path = Path(coordinate_file) if coordinate_file else root / "dy.txt"
    dark_path = Path(dark_file) if dark_file else root / "dark.txt"
    white_path = Path(white_file) if white_file else root / "white.txt"
    spec_path = Path(spectral_dir) if spectral_dir else root / "gp" / "specData"
    # 默认写入运行目录下固定的 dataset 文件夹，文件名取数据集目录名。
    # 例如数据集为 /data/scene_a -> ./dataset/scene_a.txt
    out_path = Path(output) if output else Path.cwd() / "dataset" / f"{root.name}.txt"
    if not coord_path.is_absolute():
        coord_path = root / coord_path
    if not dark_path.is_absolute():
        dark_path = root / dark_path
    if not white_path.is_absolute():
        white_path = root / white_path
    if not spec_path.is_absolute():
        spec_path = root / spec_path
    if not out_path.is_absolute():
        out_path = root / out_path

    coordinates = _load_numeric(coord_path)
    if coordinates.shape[1] < 4:
        raise ValueError(f"坐标文件至少需要 4 列（编号、X、Y、Z）：{coord_path}")
    by_id = {int(row[0]): row[1:4] for row in coordinates if np.isfinite(row[0])}

    dark = white = None
    if not no_calibration:
        dark = _load_numeric(dark_path)
        white = _load_numeric(white_path)
        max_band = max(bands)
        if dark.shape[0] < max_band or white.shape[0] < max_band or dark.shape[1] < 2 or white.shape[1] < 2:
            raise ValueError(f"dark/white 文件无法提供波段 {bands} 的第二列数据")
        denominator = white[np.array(bands) - 1, 1] - dark[np.array(bands) - 1, 1]
        if np.any(np.isclose(denominator, 0)):
            raise ValueError("white-dark 在目标波段存在零值，无法归一化")
    elif not dark_path.exists() or not white_path.exists():
        print("提示：未启用 dark/white 校准，将直接输出原始 RGB 强度。", file=sys.stderr)

    files = _find_numbered_files(spec_path, r"spec_(\d+)\.[^.]+")
    if not files:
        raise FileNotFoundError(f"目录中没有 spec_N 文件：{spec_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skipped_spectra = skipped_coordinates = 0
    rows: list[tuple[float, ...]] = []
    zero_based_bands = np.array(bands) - 1
    for point_id, path in files:
        xyz = by_id.get(point_id)
        if xyz is None:
            skipped_coordinates += 1
            continue
        try:
            spectrum = _load_numeric(path)
            if spectrum.shape[0] < max(bands) or spectrum.shape[1] < 2:
                raise ValueError("光谱长度或列数不足")
            raw_rgb = spectrum[zero_based_bands, 1]
            if dark is not None and white is not None:
                rgb = (raw_rgb - dark[zero_based_bands, 1]) / (white[zero_based_bands, 1] - dark[zero_based_bands, 1])
            else:
                rgb = raw_rgb
        except (OSError, ValueError) as exc:
            skipped_spectra += 1
            print(f"警告：跳过 {path.name}（{exc}）", file=sys.stderr)
            continue
        rows.append((point_id, float(xyz[0]), float(xyz[1]), float(xyz[2]), *map(float, rgb), class_label))

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(f"{int(row[0])},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f},{row[6]:.6f},{int(row[7])}\n")
    print(f"已处理 {len(rows)} 个点，输出：{out_path}")
    if skipped_coordinates or skipped_spectra:
        print(f"跳过：{skipped_coordinates} 个无坐标编号，{skipped_spectra} 个无效光谱。")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="将光谱文件与点坐标合并为 RGB 点云数据")
    parser.add_argument("dataset", nargs="?", default=Path.cwd(), help="数据集根目录，默认使用当前执行目录")
    parser.add_argument("-o", "--output", help="输出文件路径；默认 ./dataset/<数据集名>.txt")
    parser.add_argument("--coordinates", help="坐标文件，默认 dy.txt")
    parser.add_argument("--dark", help="暗场文件，默认 dark.txt")
    parser.add_argument("--white", help="白场文件，默认 white.txt")
    parser.add_argument("--spectral-dir", help="光谱目录，默认 gp/specData")
    parser.add_argument("--bands", nargs=3, type=int, metavar=("R", "G", "B"), default=(957, 730, 552), help="RGB 波段行号（从 1 开始）")
    parser.add_argument("--no-calibration", action="store_true", help="不使用 dark/white 校准")
    parser.add_argument("--class-label", type=int, default=0, help="输出类别标签，默认 0")
    args = parser.parse_args()
    try:
        process_dataset(args.dataset, output=args.output, coordinate_file=args.coordinates, dark_file=args.dark, white_file=args.white, spectral_dir=args.spectral_dir, bands=tuple(args.bands), no_calibration=args.no_calibration, class_label=args.class_label)
    except (FileNotFoundError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
