#!/usr/bin/env python3
"""通用工具：RGB 点云 txt → Open3D 交互查看。

用法示例：
    python view_txt.py data_convert/20260201三维建模发财树60度.txt
    python view_txt.py data_convert/20260204发财树顶端.txt

×4 说明：数据文件已由 convert_to_txt.py 在存储时乘 4（与 MATLAB scatter3(*4) 对齐），
此处显示层默认不再放大；如需额外缩放可用 --multiply。

操作说明：
    左键拖动旋转；中键拖动/Ctrl+左键 平移；滚轮缩放；
    +/- 调整点大小；R 重置视图；Q/Esc 退出。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def _detect_id_column(data: np.ndarray) -> bool:
    """嗅探第 0 列是否为 id 列（data_convert 格式：id,x,y,z,R,G,B,c）。

    id 特征：非负、整数、递增；否则视为 cripped/合并点云格式
    （x,y,z,R,G,B,...，第 0 列为浮点坐标）。
    """
    if data.shape[0] < 2:
        return False
    c0 = data[:, 0]
    is_integer = bool(np.all(np.isfinite(c0)) and np.allclose(c0, np.round(c0), atol=1e-6))
    is_increasing = bool(np.all(np.diff(c0) >= 0))
    return is_integer and is_increasing and c0.min() >= 0


def load_point_cloud_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """从逗号分隔的点云 txt 加载 XYZ 与 RGB（兼容 0~1 反射率与 0~255 强度）。

    支持两种列布局：
      - data_convert 格式：id,x,y,z,R,G,B,类别（第 0 列为整数 id）
      - cripped/合并点云格式：x,y,z,R,G,B,强度?,类别（第 0 列为浮点坐标）
    """
    file_path = Path(path)
    try:
        is_empty = file_path.stat().st_size == 0
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"找不到或无法读取点云文件：{file_path}") from exc
    if is_empty:
        raise ValueError(f"点云文件为空：{file_path}")

    try:
        data = np.loadtxt(file_path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法解析点云文件：{file_path}") from exc
    if data.size == 0 or data.shape[0] == 0:
        raise ValueError(f"点云数据为空：{file_path}")
    if data.shape[1] < 7:
        raise ValueError(f"点云数据列数不足 7 列：{file_path}")

    has_id = _detect_id_column(data)
    if has_id:
        xyz = data[:, 1:4].astype(float)
        rgb = data[:, 4:7].astype(float)
    else:
        # cripped/合并点云格式：x,y,z,R,G,B,...
        xyz = data[:, 0:3].astype(float)
        rgb = data[:, 3:6].astype(float)

    rgb_max = np.nanmax(rgb)
    if rgb_max > 4096.0:
        rgb = rgb / 65535.0  # 16 位光谱强度
    elif rgb_max > 4.0:
        rgb = rgb / 256.0    # 0~255 原始 RGB
    return xyz, np.clip(rgb, 0.0, 1.0)


def show_point_cloud(xyz: np.ndarray, rgb: np.ndarray, title: str, multiply: float = 1.0) -> None:
    """打开 Open3D 交互窗口。数据已含 ×4（convert_to_txt.py 存储时乘），默认不再放大。"""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("未安装 Open3D，无法打开点云交互窗口。") from exc

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    display_rgb = np.clip(np.asarray(rgb, dtype=float) * multiply, 0.0, 1.0)
    point_cloud.colors = o3d.utility.Vector3dVector(display_rgb)

    initial_point_size = 5.0
    if not visualizer.create_window(
        window_name=f"{title} | 初始点大小：{initial_point_size:g}"
    ):
        raise RuntimeError("无法创建 Open3D 点云窗口。")

    try:
        if not visualizer.add_geometry(point_cloud):
            raise RuntimeError("无法添加点云到 Open3D 窗口。")
        visualizer.reset_view_point(True)

        render_option = visualizer.get_render_option()
        render_option.point_size = initial_point_size

        def change_point_size(delta: float) -> bool:
            render_option.point_size = max(1.0, render_option.point_size + delta)
            print(f"当前点大小：{render_option.point_size:g}")
            return False

        def reset_view(_: object) -> bool:
            visualizer.reset_view_point(True)
            return False

        visualizer.register_key_callback(ord("+"), lambda _: change_point_size(1.0))
        visualizer.register_key_callback(ord("="), lambda _: change_point_size(1.0))
        visualizer.register_key_callback(ord("-"), lambda _: change_point_size(-1.0))
        visualizer.register_key_callback(ord("_"), lambda _: change_point_size(-1.0))
        visualizer.register_key_callback(ord("R"), reset_view)
        print(
            "操作说明：左键拖动旋转；中键拖动或 Ctrl+左键拖动平移；滚轮缩放；"
            "+/- 调整点大小；R 重置视图；Q/Esc 退出。"
        )
        visualizer.run()
    finally:
        visualizer.destroy_window()


def main() -> int:
    parser = argparse.ArgumentParser(description="交互查看 RGB 点云 txt")
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=SCRIPT_DIR / "data_convert" / "20260201三维建模发财树60度.txt",
        help="点云 txt 文件（8 列：id,x,y,z,R,G,B,类别）",
    )
    parser.add_argument(
        "--multiply",
        type=float,
        default=1.0,
        help="显示时额外 RGB 倍率（数据已含 ×4，默认 1.0）",
    )
    args = parser.parse_args()

    try:
        xyz, rgb = load_point_cloud_data(args.input_file)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"载入 {len(xyz)} 个点：{args.input_file}")

    try:
        show_point_cloud(xyz, rgb, args.input_file.name, args.multiply)
    except RuntimeError as exc:
        print(f"显示错误：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())