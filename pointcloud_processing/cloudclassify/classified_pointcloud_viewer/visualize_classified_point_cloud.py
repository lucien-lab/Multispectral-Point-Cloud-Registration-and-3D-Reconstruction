from __future__ import annotations

import colorsys
import argparse
import json
import os
from pathlib import Path

import numpy as np


REQUIRED_COLUMNS = {"x", "y", "z", "r", "g", "b", "class_id"}
DEFAULT_STATE_PATH = Path("point_cloud_viewer_state.json")


def relative_path_key(path: Path) -> str:
    """将路径规范化为相对于当前启动目录的状态文件键。"""
    return Path(os.path.relpath(path, start=Path.cwd())).as_posix()


def read_state_document(state_path: Path) -> dict[str, object]:
    """读取状态文件，并将历史绝对路径键迁移为相对路径键。"""
    try:
        document = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, TypeError):
        return {}
    if not isinstance(document, dict):
        return {}

    files = document.get("files")
    if not isinstance(files, dict):
        return document

    normalized_files: dict[str, object] = {}
    for key, state in files.items():
        if not isinstance(key, str):
            continue
        normalized_key = relative_path_key(Path(key))
        if normalized_key not in normalized_files or key == normalized_key:
            normalized_files[normalized_key] = state

    if normalized_files != files:
        document["files"] = normalized_files
        try:
            state_path.write_text(
                json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass
    return document


def load_viewer_state(input_path: Path, state_path: Path) -> dict[str, object]:
    """读取指定输入文件的已保存显示状态；无效状态返回空字典。"""
    try:
        document = read_state_document(state_path)
        files = document.get("files", {})
        state = files.get(relative_path_key(input_path), {})
    except (AttributeError, OSError, TypeError):
        return {}

    if not isinstance(state, dict):
        return {}
    color_mode = state.get("color_mode")
    point_size = state.get("point_size")
    if color_mode not in {"original", "class"}:
        return {}
    if isinstance(point_size, bool) or not isinstance(point_size, int) or point_size < 1:
        return {}
    return state


def save_viewer_state(
    input_path: Path, state: dict[str, object], state_path: Path
) -> None:
    """保存指定输入文件的显示状态，保留其他输入文件的状态。"""
    document = read_state_document(state_path)

    files = document.get("files")
    if not isinstance(files, dict):
        files = {}
        document["files"] = files
    files[relative_path_key(input_path)] = state
    state_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_classified_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取带表头的分类点云文本，返回 XYZ、归一化 RGB 与类别编号。"""
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    available_columns = set(table.dtype.names or ())
    missing_columns = sorted(REQUIRED_COLUMNS - available_columns)
    if missing_columns:
        raise ValueError(f"缺少必需列: {', '.join(missing_columns)}")

    table = np.atleast_1d(table)
    xyz = np.column_stack((table["x"], table["y"], table["z"])).astype(float)
    rgb = np.column_stack((table["r"], table["g"], table["b"])).astype(float)
    rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    class_ids = table["class_id"].astype(np.int64)
    return xyz, rgb, class_ids


def class_colors(class_ids: np.ndarray) -> np.ndarray:
    """为每个类别编号生成稳定且容易区分的 RGB 颜色。"""
    unique_ids = np.unique(class_ids.astype(np.int64))
    palette = {
        int(class_id): colorsys.hsv_to_rgb(
            (int(class_id) * 0.61803398875) % 1.0,
            0.72,
            0.95,
        )
        for class_id in unique_ids
    }
    return np.asarray([palette[int(class_id)] for class_id in class_ids], dtype=float)


class PointCloudViewer:
    """维护点云的颜色模式与点大小，并运行 Open3D 交互窗口。"""

    def __init__(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        class_ids: np.ndarray,
        input_path: Path | None = None,
        state_path: Path = DEFAULT_STATE_PATH,
    ) -> None:
        self.xyz = xyz
        self.original_colors = rgb
        self.class_color_values = class_colors(class_ids)
        self.active_colors = self.original_colors
        self.color_mode = "original"
        self.point_size = 3
        self.input_path = Path(relative_path_key(input_path)) if input_path is not None else None
        self.state_path = state_path
        self._saved_state = (
            load_viewer_state(self.input_path, self.state_path)
            if self.input_path is not None
            else {}
        )
        if self._saved_state:
            self.set_color_mode(self._saved_state["color_mode"])
            self.point_size = self._saved_state["point_size"]
        self._point_cloud = None

    def change_point_size(self, delta: int) -> None:
        self.point_size = max(1, self.point_size + delta)

    def set_color_mode(self, mode: str) -> None:
        if mode == "original":
            self.active_colors = self.original_colors
        elif mode == "class":
            self.active_colors = self.class_color_values
        else:
            raise ValueError(f"未知颜色模式: {mode}")
        self.color_mode = mode

    def _update_colors(self, vis) -> bool:
        import open3d as o3d

        self._point_cloud.colors = o3d.utility.Vector3dVector(self.active_colors)
        vis.update_geometry(self._point_cloud)
        vis.update_renderer()
        return False

    def _show_original(self, vis) -> bool:
        self.set_color_mode("original")
        return self._update_colors(vis)

    def _show_class(self, vis) -> bool:
        self.set_color_mode("class")
        return self._update_colors(vis)

    def _change_size(self, vis, delta: int) -> bool:
        self.change_point_size(delta)
        vis.get_render_option().point_size = self.point_size
        vis.update_renderer()
        print(f"点大小：{self.point_size}")
        return False

    def _increase_size(self, vis) -> bool:
        return self._change_size(vis, 1)

    def _decrease_size(self, vis) -> bool:
        return self._change_size(vis, -1)

    @staticmethod
    def _reset_view(vis) -> bool:
        vis.reset_view_point(True)
        return False

    @staticmethod
    def _camera_to_state(vis) -> dict[str, object] | None:
        try:
            parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
            return {
                "intrinsic": {
                    "width": parameters.intrinsic.width,
                    "height": parameters.intrinsic.height,
                    "matrix": np.asarray(parameters.intrinsic.intrinsic_matrix).tolist(),
                },
                "extrinsic": np.asarray(parameters.extrinsic).tolist(),
            }
        except RuntimeError:
            return None

    @staticmethod
    def _restore_camera(vis, o3d, camera_state: object) -> None:
        if not isinstance(camera_state, dict):
            return
        try:
            intrinsic = camera_state["intrinsic"]
            width = int(intrinsic["width"])
            height = int(intrinsic["height"])
            matrix = np.asarray(intrinsic["matrix"], dtype=float).reshape(3, 3)
            extrinsic = np.asarray(camera_state["extrinsic"], dtype=float).reshape(4, 4)
            parameters = o3d.camera.PinholeCameraParameters()
            parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, matrix)
            parameters.extrinsic = extrinsic
            vis.get_view_control().convert_from_pinhole_camera_parameters(
                parameters, allow_arbitrary=True
            )
        except (KeyError, TypeError, ValueError, RuntimeError):
            return

    def _save_current_state(self, vis) -> None:
        if self.input_path is None:
            return
        state: dict[str, object] = {
            "color_mode": self.color_mode,
            "point_size": self.point_size,
        }
        camera_state = self._camera_to_state(vis)
        if camera_state is not None:
            state["camera"] = camera_state
        save_viewer_state(self.input_path, state, self.state_path)

    def run(self) -> None:
        import open3d as o3d

        self._point_cloud = o3d.geometry.PointCloud()
        self._point_cloud.points = o3d.utility.Vector3dVector(self.xyz)
        self._point_cloud.colors = o3d.utility.Vector3dVector(self.active_colors)
        vis = o3d.visualization.VisualizerWithKeyCallback()
        try:
            if not vis.create_window(window_name="分类点云查看器"):
                raise RuntimeError("无法创建 Open3D 可视化窗口")
            vis.add_geometry(self._point_cloud)
            vis.get_render_option().point_size = self.point_size
            self._restore_camera(vis, o3d, self._saved_state.get("camera"))
            vis.register_key_callback(ord("0"), self._show_original)
            vis.register_key_callback(ord("1"), self._show_class)
            for key in (ord("+"), ord("=")):
                vis.register_key_callback(key, self._increase_size)
            for key in (ord("-"), ord("_")):
                vis.register_key_callback(key, self._decrease_size)
            vis.register_key_callback(ord("R"), self._reset_view)
            print("按 0 显示原始 RGB；按 1 显示分类标签颜色。")
            print("按 +/= 增大点大小；按 -/_ 减小点大小；R 重置视角；Q 或 Esc 退出。")
            vis.run()
        finally:
            try:
                self._save_current_state(vis)
            finally:
                vis.destroy_window()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化带 RGB 和 class_id 的点云文本")
    default_path = Path("cropped_point_cloud_classified.txt")
    parser.add_argument("path", nargs="?", type=Path, default=default_path, help="输入点云 .txt 文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xyz, rgb, class_ids = load_classified_point_cloud(args.path)
    print(f"已加载 {args.path}，共 {len(xyz)} 个点。")
    PointCloudViewer(xyz, rgb, class_ids, input_path=args.path).run()


if __name__ == "__main__":
    main()
