#!/usr/bin/env python3
"""使用 Open3D 交互式显示生成的超光谱点云。"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import open3d as o3d


WAVELENGTHS = (495, 696, 600, 803, 545, 642)
COLOR_MODES = ("side", "rgb", "height", "distance", *(f"rfl-{w}" for w in WAVELENGTHS))
KEY_COLOR_MODES = {
    "0": "side",
    "1": "rgb",
    "2": "height",
    "3": "distance",
    "4": "rfl-495",
    "5": "rfl-696",
    "6": "rfl-600",
    "7": "rfl-803",
    "8": "rfl-545",
    "9": "rfl-642",
}


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="交互式查看 left/right 超光谱点云")
    parser.add_argument("--input-dir", type=Path, default=here / "pointcloud_output")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--color", choices=COLOR_MODES, default="side")
    parser.add_argument("--all", action="store_true", help="显示无效帧；默认仅显示有效点")
    parser.add_argument("--min-distance", type=float, default=3.0, help="最小显示距离，单位 m（默认 3.0）")
    parser.add_argument("--max-distance", type=float, help="最大显示距离，单位 m（默认不限制）")
    parser.add_argument("--brightness", type=float, default=1.0, help="RGB显示亮度倍数（默认 1.0）")
    parser.add_argument("--point-size", type=float, default=3.0)
    parser.add_argument("--background", choices=("black", "white"), default="black")
    parser.add_argument("--no-axis", action="store_true", help="不显示坐标轴")
    parser.add_argument("--reset-view", action="store_true", help="忽略上次保存的视角，从自动视角开始")
    parser.add_argument("--info", action="store_true", help="只打印数据统计，不打开窗口")
    return parser.parse_args()


def load_side(path: Path, side: str) -> dict[str, np.ndarray]:
    file_path = path / f"{side}_pointcloud_all.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"找不到 {file_path}，请先运行 generate_pointcloud.py")
    with np.load(file_path) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    result["side_name"] = np.full(result["xyz_mm"].shape[0], side)
    return result


def concatenate(data_sets: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = ("xyz_mm", "rgb", "rgb_uint8", "range_m", "reflectance", "valid", "side_name")
    return {key: np.concatenate([data[key] for data in data_sets], axis=0) for key in keys}


def normalized_scalar_colors(values: np.ndarray, cmap_name: str) -> np.ndarray:
    finite = np.isfinite(values)
    colors = np.full((values.size, 3), 0.35, dtype=np.float64)
    if not np.any(finite):
        return colors
    lo, hi = np.percentile(values[finite], [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    colors[finite] = matplotlib.colormaps[cmap_name](normalized[finite])[:, :3]
    return colors


def colors_for_mode(
    data: dict[str, np.ndarray], indices: np.ndarray, mode: str, brightness: float
) -> np.ndarray:
    if mode == "side":
        names = data["side_name"][indices]
        colors = np.full((indices.size, 3), 0.5, dtype=np.float64)
        colors[names == "left"] = [0.0, 1.0, 0.0]
        colors[names == "right"] = [1.0, 0.0, 0.0]
        return colors
    if mode == "rgb":
        return np.clip(data["rgb"][indices] * brightness, 0.0, 1.0)
    if mode == "height":
        return normalized_scalar_colors(data["xyz_mm"][indices, 2], "turbo")
    if mode == "distance":
        return normalized_scalar_colors(data["range_m"][indices], "viridis")
    wavelength = int(mode.split("-", 1)[1])
    channel = WAVELENGTHS.index(wavelength)
    return normalized_scalar_colors(data["reflectance"][indices, channel], "inferno")


def selection(
    data: dict[str, np.ndarray],
    include_invalid: bool,
    min_distance: float,
    max_distance: float | None,
) -> np.ndarray:
    selected = np.ones(data["valid"].size, dtype=bool)
    if not include_invalid:
        selected &= data["valid"]
    selected &= data["range_m"] >= min_distance
    if max_distance is not None:
        selected &= data["range_m"] <= max_distance
    return np.flatnonzero(selected)


def print_stats(
    data: dict[str, np.ndarray],
    side: str,
    include_invalid: bool,
    min_distance: float,
    max_distance: float | None,
) -> None:
    valid = data["valid"].astype(bool)
    indices = selection(data, include_invalid, min_distance, max_distance)
    xyz = data["xyz_mm"][indices]
    print(f"范围: {side}")
    print(f"总点数: {valid.size}")
    print(f"有效点: {valid.sum()}")
    upper = "不限制" if max_distance is None else f"{max_distance:.3f} m"
    print(f"距离筛选: {min_distance:.3f} m ~ {upper}")
    print(f"当前显示点: {indices.size}")
    if xyz.size:
        for column, name in enumerate(("X", "Y", "Z")):
            print(f"{name}: {xyz[:, column].min():.3f} ~ {xyz[:, column].max():.3f} mm")
        ranges = data["range_m"][indices]
        print(f"距离: {ranges.min():.3f} ~ {ranges.max():.3f} m")


class Viewer:
    def __init__(self, data: dict[str, np.ndarray], args: argparse.Namespace):
        self.data = data
        self.args = args
        self.mode = args.color
        self.include_invalid = args.all
        self.cloud = o3d.geometry.PointCloud()
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()

    @property
    def view_state_path(self) -> Path:
        return self.args.input_dir / f"viewer_view_{self.args.side}.json"

    def restore_view(self) -> bool:
        if self.args.reset_view or not self.view_state_path.exists():
            return False
        try:
            parameters = o3d.io.read_pinhole_camera_parameters(str(self.view_state_path))
            restored = self.visualizer.get_view_control().convert_from_pinhole_camera_parameters(
                parameters, allow_arbitrary=True
            )
        except Exception as exc:
            print(f"无法恢复上次视角，将使用自动视角: {exc}")
            return False
        if restored:
            print(f"已恢复上次视角: {self.view_state_path}")
        else:
            print("上次视角与当前窗口不兼容，将使用自动视角")
        return restored

    def save_view(self) -> None:
        try:
            parameters = self.visualizer.get_view_control().convert_to_pinhole_camera_parameters()
            if o3d.io.write_pinhole_camera_parameters(str(self.view_state_path), parameters):
                print(f"已保存当前视角: {self.view_state_path}")
            else:
                print(f"当前视角保存失败: {self.view_state_path}")
        except Exception as exc:
            print(f"当前视角保存失败: {exc}")

    def refresh(self, reset_view: bool = False) -> None:
        indices = selection(
            self.data, self.include_invalid, self.args.min_distance, self.args.max_distance
        )
        self.cloud.points = o3d.utility.Vector3dVector(self.data["xyz_mm"][indices])
        self.cloud.colors = o3d.utility.Vector3dVector(
            colors_for_mode(self.data, indices, self.mode, self.args.brightness)
        )
        self.visualizer.update_geometry(self.cloud)
        if reset_view:
            self.visualizer.reset_view_point(True)
        state = "全部点" if self.include_invalid else "仅有效点"
        print(f"着色: {self.mode}；{state}；显示 {indices.size} 点")

    def set_mode(self, mode: str):
        def callback(_: o3d.visualization.Visualizer) -> bool:
            self.mode = mode
            self.refresh()
            return False

        return callback

    def toggle_valid(self, _: o3d.visualization.Visualizer) -> bool:
        self.include_invalid = not self.include_invalid
        self.refresh(reset_view=True)
        return False

    def save_screenshot(self, _: o3d.visualization.Visualizer) -> bool:
        target = self.args.input_dir / f"viewer_{datetime.now():%Y%m%d_%H%M%S}.png"
        self.visualizer.capture_screen_image(str(target), do_render=True)
        print(f"截图已保存: {target}")
        return False

    def run(self) -> None:
        if not self.visualizer.create_window(
            window_name=f"Hyperspectral Point Cloud - {self.args.side}", width=1280, height=800
        ):
            raise RuntimeError("Open3D 窗口创建失败；请在图形桌面环境运行")
        self.visualizer.add_geometry(self.cloud)
        if not self.args.no_axis:
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1000.0, origin=(0, 0, 0))
            self.visualizer.add_geometry(axis)

        for key, mode in KEY_COLOR_MODES.items():
            self.visualizer.register_key_callback(ord(key), self.set_mode(mode))
        self.visualizer.register_key_callback(ord("V"), self.toggle_valid)
        self.visualizer.register_key_callback(ord("S"), self.save_screenshot)

        render = self.visualizer.get_render_option()
        render.point_size = self.args.point_size
        render.background_color = np.zeros(3) if self.args.background == "black" else np.ones(3)
        self.refresh(reset_view=True)
        self.restore_view()
        print("快捷键: 0=左右来源 1=RGB 2=高度 3=距离 4~9=六波长反射率 V=切换有效/全部 S=截图")
        print("鼠标: 左键旋转，中键平移，滚轮缩放；关闭窗口或按 Q/Esc 退出")
        self.visualizer.run()
        self.save_view()
        self.visualizer.destroy_window()


def main() -> None:
    args = parse_args()
    if args.point_size <= 0:
        raise ValueError("--point-size 必须大于 0")
    if args.brightness <= 0:
        raise ValueError("--brightness 必须大于 0")
    if args.min_distance < 0:
        raise ValueError("--min-distance 不能小于 0")
    if args.max_distance is not None and args.max_distance < args.min_distance:
        raise ValueError("--max-distance 不能小于 --min-distance")
    sides = ("left", "right") if args.side == "both" else (args.side,)
    data = concatenate([load_side(args.input_dir, side) for side in sides])
    print_stats(data, args.side, args.all, args.min_distance, args.max_distance)
    if not args.info:
        Viewer(data, args).run()


if __name__ == "__main__":
    main()
