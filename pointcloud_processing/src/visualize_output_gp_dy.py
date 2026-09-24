"""Load and interactively view point-cloud coordinates and colors.

显示时按 MATLAB 原版习惯将 RGB 放大 4 倍（仅显示，不改数据文件）。
"""

import argparse
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


def default_input_path() -> Path:
    """Return the fixed dataset directory's file for this project."""
    project_root = Path(__file__).resolve().parents[1]
    papers_root = project_root.parent
    return papers_root / "dataset" / f"{project_root.name}.txt"


def load_point_cloud_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load XYZ and RGB arrays from a comma-delimited point-cloud file."""
    file_path = Path(path)

    try:
        is_empty_file = file_path.stat().st_size == 0
    except FileNotFoundError as exc:
        raise ValueError(f"找不到点云数据文件：{file_path}") from exc
    except OSError as exc:
        raise ValueError(f"无法读取点云数据文件：{file_path}") from exc

    if is_empty_file:
        raise ValueError(f"点云数据为空：{file_path}")

    try:
        data = np.loadtxt(file_path, delimiter=",", ndmin=2)
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法解析点云数据文件：{file_path}") from exc

    if data.size == 0 or data.shape[0] == 0:
        raise ValueError(f"点云数据为空：{file_path}")
    if data.shape[1] < 7:
        raise ValueError(f"点云数据列数不足 7 列：{file_path}")

    rgb = data[:, 4:7].astype(float)
    # 兼容已归一化的 RGB（0~1）和 0~255/256 的原始 RGB。
    rgb_max = np.nanmax(rgb)
    if rgb_max > 4096.0:
        # 生成文件可能保存为 16 位光谱强度。
        rgb = rgb / 65535.0
    elif rgb_max > 4.0:
        rgb = rgb / 256.0
    return data[:, 1:4].astype(float), np.clip(rgb, 0.0, 1.0)


def show_point_cloud(xyz: np.ndarray, rgb: np.ndarray, title: str) -> None:
    """Show one Open3D point cloud in an interactive window."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("未安装 Open3D，无法打开点云交互窗口。") from exc

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(xyz)
    # 与 MATLAB 原版 handle_datas.m / handle_gp_datas.m 的 scatter3(...,*4) 对齐：
    # 仅显示时把 RGB 放大 4 倍（超 1 截断），数据文件里保持原始反射率不变。
    display_rgb = np.clip(np.asarray(rgb, dtype=float) * 4.0, 0.0, 1.0)
    point_cloud.colors = o3d.utility.Vector3dVector(display_rgb)

    initial_point_size = 5.0
    window_title = f"{title} | 初始点大小：{initial_point_size:g}"
    window_created = visualizer.create_window(window_name=window_title)
    if not window_created:
        raise RuntimeError("无法创建 Open3D 点云窗口。")

    try:
        if not visualizer.add_geometry(point_cloud):
            raise RuntimeError("无法添加点云到 Open3D 窗口。")
        visualizer.reset_view_point(True)

        render_option = visualizer.get_render_option()
        render_option.point_size = initial_point_size

        def change_point_size(delta: float) -> bool:
            updated_size = max(1.0, render_option.point_size + delta)
            render_option.point_size = updated_size
            print(f"当前点大小：{updated_size:g}")
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


def main(argv: Sequence[str] | None = None) -> int:
    """Load an input file and open its interactive point-cloud viewer."""
    parser = argparse.ArgumentParser(description="显示 output_gp_dy 点云数据")
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=default_input_path(),
        help="逗号分隔的点云数据文件（默认：dataset/<项目目录名>.txt）",
    )
    arguments = parser.parse_args(argv)

    try:
        xyz, rgb = load_point_cloud_data(arguments.input_file)
    except ValueError as exc:
        print(f"数据错误：{exc}", file=sys.stderr)
        return 1

    try:
        show_point_cloud(xyz, rgb, f"点云查看器：{arguments.input_file.name}")
    except RuntimeError as exc:
        print(f"显示错误：{exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
