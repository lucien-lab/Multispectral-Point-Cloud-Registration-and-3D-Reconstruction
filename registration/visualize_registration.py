#!/usr/bin/env python3
"""通过GUI选择并使用Open3D交互查看配准实验点云。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib
import numpy as np
import open3d as o3d


COLOR_MODES = ("original", "height", "distance", "x", "y", "z")
KEY_COLOR_MODES = {
    "0": "original",
    "1": "height",
    "2": "distance",
    "3": "x",
    "4": "y",
    "5": "z",
}
SUPPORTED_EXTENSIONS = (".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts")


@dataclass(frozen=True)
class CloudData:
    path: Path
    points: np.ndarray
    original_colors: np.ndarray
    had_file_colors: bool


def choose_point_cloud(initial_dir: Path) -> Path | None:
    """弹出系统文件选择框；取消时返回 None。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError(
            "当前Python环境没有Tk GUI支持，请改用 --file 指定点云路径"
        ) from exc

    initial_dir = initial_dir.expanduser().resolve()
    if not initial_dir.is_dir():
        initial_dir = Path.cwd()
    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.update()
        selected = filedialog.askopenfilename(
            title="选择要可视化的点云文件",
            initialdir=str(initial_dir),
            filetypes=(
                ("Point clouds", "*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts"),
                ("PLY", "*.ply"),
                ("PCD", "*.pcd"),
                ("All files", "*.*"),
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "无法创建GUI文件选择器，请确认当前处于macOS图形桌面，"
            "或改用 --file 指定点云路径"
        ) from exc
    finally:
        if root is not None:
            root.destroy()
    return Path(selected) if selected else None


def resolve_input_file(
    cli_file: Path | None,
    initial_dir: Path,
    chooser: Callable[[Path], Path | None] = choose_point_cloud,
) -> Path | None:
    """命令行路径优先；未提供时调用GUI选择器。"""
    selected = cli_file if cli_file is not None else chooser(initial_dir)
    if selected is None:
        return None
    path = Path(selected).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到点云文件: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"不支持的点云格式 {path.suffix!r}；支持 {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return path


def load_point_cloud(path: Path) -> CloudData:
    """用Open3D加载点云并保留文件内RGB；缺少RGB时使用灰色。"""
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f"点云为空或无法解析: {path}")
    points = np.asarray(cloud.points, dtype=np.float64).copy()
    if not np.isfinite(points).all():
        raise ValueError(f"点云坐标包含NaN或Inf: {path}")
    had_colors = cloud.has_colors() and len(cloud.colors) == len(cloud.points)
    colors = (
        np.clip(np.asarray(cloud.colors, dtype=np.float64).copy(), 0.0, 1.0)
        if had_colors
        else np.full((len(points), 3), 0.65, dtype=np.float64)
    )
    return CloudData(path.resolve(), points, colors, had_colors)


def normalized_scalar_colors(values: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    colors = np.full((values.size, 3), 0.5, dtype=np.float64)
    if not finite.any():
        return colors
    low, high = np.percentile(values[finite], [2.0, 98.0])
    if high <= low:
        high = low + 1.0
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    colors[finite] = matplotlib.colormaps[cmap_name](normalized[finite])[:, :3]
    return colors


def colors_for_mode(data: CloudData, mode: str) -> np.ndarray:
    if mode == "original":
        return data.original_colors.copy()
    if mode == "height":
        values = data.points[:, 2]
    elif mode == "distance":
        values = np.linalg.norm(data.points, axis=1)
    elif mode in {"x", "y", "z"}:
        values = data.points[:, {"x": 0, "y": 1, "z": 2}[mode]]
    else:
        raise ValueError(f"未知着色模式: {mode}")
    return normalized_scalar_colors(values)


def view_state_path(point_cloud_path: Path) -> Path:
    return point_cloud_path.parent / f".{point_cloud_path.name}.view.json"


def screenshot_path(point_cloud_path: Path, timestamp: datetime | None = None) -> Path:
    timestamp = datetime.now() if timestamp is None else timestamp
    return point_cloud_path.parent / (
        f"{point_cloud_path.stem}_viewer_{timestamp:%Y%m%d_%H%M%S_%f}.png"
    )


def coordinate_frame_spec(data: CloudData) -> tuple[np.ndarray, float]:
    diagonal = float(np.linalg.norm(np.ptp(data.points, axis=0)))
    return np.zeros(3, dtype=float), max(0.1, 0.08 * diagonal)


def print_stats(data: CloudData) -> None:
    minimum = data.points.min(axis=0)
    maximum = data.points.max(axis=0)
    distance = np.linalg.norm(data.points, axis=1)
    print(f"文件: {data.path}")
    print(f"点数: {len(data.points)}")
    print(f"文件颜色: {'有' if data.had_file_colors else '无（使用灰色）'}")
    for axis, low, high in zip("XYZ", minimum, maximum, strict=True):
        print(f"{axis}: {low:.6f} ~ {high:.6f} m")
    print(f"原点距离: {distance.min():.6f} ~ {distance.max():.6f} m")


class RegistrationViewer:
    def __init__(
        self,
        data: CloudData,
        point_size: float = 3.0,
        background: str = "black",
        show_axis: bool = True,
        reset_view: bool = False,
    ):
        self.data = data
        self.mode = "original"
        self.point_size = point_size
        self.background = background
        self.show_axis = show_axis
        self.reset_view = reset_view
        self.cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data.points))
        self.cloud.colors = o3d.utility.Vector3dVector(data.original_colors)
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()

    def refresh(self) -> None:
        self.cloud.colors = o3d.utility.Vector3dVector(colors_for_mode(self.data, self.mode))
        self.visualizer.update_geometry(self.cloud)
        render = self.visualizer.get_render_option()
        render.point_size = self.point_size
        render.background_color = (
            np.zeros(3, dtype=float)
            if self.background == "black"
            else np.ones(3, dtype=float)
        )
        print(
            f"着色={self.mode}；点大小={self.point_size:.1f}；背景={self.background}"
        )

    def set_mode(self, mode: str):
        def callback(_: o3d.visualization.Visualizer) -> bool:
            self.mode = mode
            self.refresh()
            return False

        return callback

    def change_point_size(self, amount: float):
        def callback(_: o3d.visualization.Visualizer) -> bool:
            self.point_size = float(np.clip(self.point_size + amount, 1.0, 20.0))
            self.refresh()
            return False

        return callback

    def toggle_background(self, _: o3d.visualization.Visualizer) -> bool:
        self.background = "white" if self.background == "black" else "black"
        self.refresh()
        return False

    def reset_camera(self, _: o3d.visualization.Visualizer) -> bool:
        self.visualizer.reset_view_point(True)
        print("已重置视角")
        return False

    def save_screenshot(self, _: o3d.visualization.Visualizer) -> bool:
        target = screenshot_path(self.data.path)
        try:
            self.visualizer.capture_screen_image(str(target), do_render=True)
            if target.is_file() and target.stat().st_size > 0:
                print(f"截图已保存: {target}")
            else:
                print(f"截图保存失败，未生成有效文件: {target}")
        except Exception as exc:
            print(f"截图保存失败: {exc}")
        return False

    def restore_view(self) -> None:
        path = view_state_path(self.data.path)
        if self.reset_view or not path.is_file():
            return
        try:
            parameters = o3d.io.read_pinhole_camera_parameters(str(path))
            restored = self.visualizer.get_view_control().convert_from_pinhole_camera_parameters(
                parameters,
                allow_arbitrary=True,
            )
            print(f"{'已恢复' if restored else '无法恢复'}视角: {path}")
        except Exception as exc:
            print(f"恢复视角失败，将使用自动视角: {exc}")

    def save_view(self) -> None:
        path = view_state_path(self.data.path)
        try:
            parameters = self.visualizer.get_view_control().convert_to_pinhole_camera_parameters()
            if o3d.io.write_pinhole_camera_parameters(str(path), parameters):
                print(f"已保存当前视角: {path}")
            else:
                print(f"当前视角保存失败: {path}")
        except Exception as exc:
            print(f"保存视角失败: {exc}")

    def run(self) -> None:
        if not self.visualizer.create_window(
            window_name=f"Registration Point Cloud - {self.data.path.name}",
            width=1280,
            height=800,
        ):
            raise RuntimeError("Open3D窗口创建失败，请在macOS图形桌面环境中运行")
        self.visualizer.add_geometry(self.cloud)
        if self.show_axis:
            origin, size = coordinate_frame_spec(self.data)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=size,
                origin=origin,
            )
            self.visualizer.add_geometry(axis)

        for key, mode in KEY_COLOR_MODES.items():
            self.visualizer.register_key_callback(ord(key), self.set_mode(mode))
        for key in ("=", "+"):
            self.visualizer.register_key_callback(ord(key), self.change_point_size(1.0))
        for key in ("-", "_"):
            self.visualizer.register_key_callback(ord(key), self.change_point_size(-1.0))
        self.visualizer.register_key_callback(ord("B"), self.toggle_background)
        self.visualizer.register_key_callback(ord("R"), self.reset_camera)
        self.visualizer.register_key_callback(ord("S"), self.save_screenshot)

        self.refresh()
        self.visualizer.reset_view_point(True)
        self.restore_view()
        print("快捷键: 0=原始RGB 1=高度 2=原点距离 3=X 4=Y 5=Z")
        print("快捷键: +/-=点大小 B=黑白背景 R=重置视角 S=截图 Q/Esc=退出")
        print("鼠标: 左键旋转，中键平移，滚轮缩放")
        self.visualizer.run()
        self.save_view()
        self.visualizer.destroy_window()


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="直接指定点云；省略时弹出GUI选择")
    parser.add_argument(
        "--initial-dir",
        type=Path,
        default=root / "results",
        help="GUI文件选择器的初始目录",
    )
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument("--no-axis", action="store_true")
    parser.add_argument("--reset-view", action="store_true")
    parser.add_argument("--info", action="store_true", help="仅输出统计，不打开Open3D窗口")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.point_size <= 0:
        raise ValueError("--point-size 必须大于0")
    path = resolve_input_file(args.file, args.initial_dir)
    if path is None:
        print("未选择文件，已退出。")
        return
    data = load_point_cloud(path)
    print_stats(data)
    if not args.info:
        RegistrationViewer(
            data,
            point_size=args.point_size,
            background=args.background,
            show_axis=not args.no_axis,
            reset_view=args.reset_view,
        ).run()


if __name__ == "__main__":
    main()
