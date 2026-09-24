# 分类点云查看器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 创建一个 Open3D 分类点云查看器，默认显示 RGB，并支持类别颜色和点大小的键盘切换。

**Architecture:** `visualize_classified_point_cloud.py` 将数据读取、颜色计算和窗口回调分开。文件加载函数返回 NumPy 数组；`PointCloudViewer` 保存当前颜色模式和点大小，并通过 Open3D `VisualizerWithKeyCallback` 更新渲染器。测试直接导入前两类纯函数，不启动 GUI。

**Tech Stack:** Python、NumPy、Open3D、unittest；运行和测试均使用 Conda 环境 `ms_pointcloud_midterm`。

## Global Constraints

- 输入为包含 `x,y,z,r,g,b,class_id` 表头的逗号分隔 `.txt` 文件。
- RGB 输入范围为 0--255，传给 Open3D 前必须裁剪、归一化到 0--1。
- `0` 显示原始 RGB，`1` 显示稳定的类别颜色。
- `+`/`=` 增大点大小，`-`/`_` 减小点大小，最小值为 1。
- 仅在创建 GUI 窗口时导入 Open3D；窗口销毁必须在 `finally` 中执行。
- 不修改输入的 `.txt` 或 `class_id_mapping.csv`。
- 状态保存至项目根目录 `point_cloud_viewer_state.json`，并以输入文件的解析后绝对路径隔离不同点云的状态。
- 每次窗口关闭时保存颜色模式、点大小和相机内外参；不存在、损坏或无效状态必须回退默认显示，不能阻止窗口打开。

---

### Task 1: 数据读取与颜色工具

**Files:**
- Create: `visualize_classified_point_cloud.py`
- Create: `tests/test_visualize_classified_point_cloud.py`

**Interfaces:**
- Produces: `load_classified_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]`，依次返回 `(xyz, rgb, class_ids)`。
- Produces: `class_colors(class_ids: np.ndarray) -> np.ndarray`，返回形状 `(N, 3)`、范围 `[0, 1]` 的确定性颜色。

- [ ] **Step 1: Write the failing tests**

```python
def test_load_classified_point_cloud_normalizes_rgb(tmp_path):
    data = tmp_path / "cloud.txt"
    data.write_text("point_id,x,y,z,r,g,b,class_id\\n1,1,2,3,255,128,0,7\\n")
    xyz, rgb, class_ids = viewer.load_classified_point_cloud(data)
    np.testing.assert_allclose(xyz, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(rgb, [[1.0, 128 / 255, 0.0]])
    np.testing.assert_array_equal(class_ids, [7])
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_visualize_classified_point_cloud.PointCloudDataTests.test_load_classified_point_cloud_normalizes_rgb -v`

Expected: FAIL because `visualize_classified_point_cloud` or `load_classified_point_cloud` does not exist.

- [ ] **Step 3: Implement the minimal loader and deterministic color function**

```python
def load_classified_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    required = {"x", "y", "z", "r", "g", "b", "class_id"}
    available = set(table.dtype.names or ())
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"缺少必需列: {', '.join(missing)}")
    table = np.atleast_1d(table)
    xyz = np.column_stack((table["x"], table["y"], table["z"])).astype(float)
    rgb = np.clip(np.column_stack((table["r"], table["g"], table["b"])), 0, 255) / 255.0
    return xyz, rgb.astype(float), table["class_id"].astype(np.int64)

def class_colors(class_ids: np.ndarray) -> np.ndarray:
    unique_ids = np.unique(class_ids.astype(np.int64))
    palette = {class_id: hsv_to_rgb((class_id * 0.61803398875) % 1.0, 0.72, 0.95)
               for class_id in unique_ids}
    return np.asarray([palette[int(class_id)] for class_id in class_ids])
```

- [ ] **Step 4: Extend tests for stable class colors and missing columns**

```python
def test_class_colors_is_stable_and_groups_equal_labels():
    colors = viewer.class_colors(np.array([5, 2, 5]))
    np.testing.assert_allclose(colors[0], colors[2])
    assert not np.allclose(colors[0], colors[1])

def test_loader_rejects_missing_required_column(tmp_path):
    data = tmp_path / "bad.txt"
    data.write_text("x,y,z,r,g,b\\n0,0,0,0,0,0\\n")
    with self.assertRaisesRegex(ValueError, "class_id"):
        viewer.load_classified_point_cloud(data)
```

- [ ] **Step 5: Run all unit tests and compile**

Run: `conda run -n ms_pointcloud_midterm python -m unittest discover -s tests -v && conda run -n ms_pointcloud_midterm python -m py_compile visualize_classified_point_cloud.py tests/test_visualize_classified_point_cloud.py`

Expected: all tests pass and compilation exits with status 0.

### Task 2: Open3D 交互窗口

**Files:**
- Modify: `visualize_classified_point_cloud.py`
- Modify: `tests/test_visualize_classified_point_cloud.py`

**Interfaces:**
- Consumes: `load_classified_point_cloud(path)` 和 `class_colors(class_ids)`。
- Produces: `PointCloudViewer(xyz, rgb, class_ids)`；`change_point_size(delta: int) -> None` 与 `set_color_mode(mode: str) -> None`。

- [ ] **Step 1: Write failing state tests without GUI**

```python
def test_viewer_changes_point_size_with_floor_one():
    viewer_state = viewer.PointCloudViewer(np.zeros((1, 3)), np.zeros((1, 3)), np.array([0]))
    viewer_state.change_point_size(-99)
    assert viewer_state.point_size == 1
    viewer_state.change_point_size(2)
    assert viewer_state.point_size == 3

def test_viewer_switches_between_original_and_class_colors():
    original = np.array([[0.1, 0.2, 0.3]])
    viewer_state = viewer.PointCloudViewer(np.zeros((1, 3)), original, np.array([9]))
    viewer_state.set_color_mode("class")
    assert not np.allclose(viewer_state.active_colors, original)
    viewer_state.set_color_mode("original")
    np.testing.assert_allclose(viewer_state.active_colors, original)
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_visualize_classified_point_cloud.PointCloudViewerTests -v`

Expected: FAIL because `PointCloudViewer` does not exist.

- [ ] **Step 3: Implement state class, callbacks, and command-line entrypoint**

```python
class PointCloudViewer:
    def change_point_size(self, delta: int) -> None:
        self.point_size = max(1, self.point_size + delta)

    def set_color_mode(self, mode: str) -> None:
        self.active_colors = self.class_color_values if mode == "class" else self.original_colors

    def run(self) -> None:
        import open3d as o3d
        self._point_cloud = o3d.geometry.PointCloud()
        self._point_cloud.points = o3d.utility.Vector3dVector(self.xyz)
        self._point_cloud.colors = o3d.utility.Vector3dVector(self.active_colors)
        vis = o3d.visualization.VisualizerWithKeyCallback()
        try:
            vis.create_window(window_name="分类点云查看器")
            vis.add_geometry(self._point_cloud)
            vis.register_key_callback(ord("0"), self._show_original)
            vis.register_key_callback(ord("1"), self._show_class)
            for key in (ord("+"), ord("=")):
                vis.register_key_callback(key, self._increase_size)
            for key in (ord("-"), ord("_")):
                vis.register_key_callback(key, self._decrease_size)
            vis.run()
        finally:
            vis.destroy_window()
```

- [ ] **Step 4: Run all tests and compile**

Run: `conda run -n ms_pointcloud_midterm python -m unittest discover -s tests -v && conda run -n ms_pointcloud_midterm python -m py_compile visualize_classified_point_cloud.py tests/test_visualize_classified_point_cloud.py`

Expected: all tests pass and compilation exits with status 0.

- [ ] **Step 5: Run GUI smoke test**

Run: `conda run -n ms_pointcloud_midterm python visualize_classified_point_cloud.py`

Expected: a window opens with original RGB. Manual verification is required for `0`, `1`, `+`, `=`, `-`, and `_`; verify middle mouse or Ctrl+left pans, `R` resets view, and `Q`/Esc closes.

### Task 3: 显示状态持久化与相机恢复

**Files:**
- Modify: `visualize_classified_point_cloud.py`
- Modify: `tests/test_visualize_classified_point_cloud.py`

**Interfaces:**
- Produces: `load_viewer_state(input_path: Path, state_path: Path) -> dict[str, object]`，返回指定输入文件的状态，异常或无效内容返回空字典。
- Produces: `save_viewer_state(input_path: Path, state: dict[str, object], state_path: Path) -> None`，保留状态文件中其他输入文件的记录。
- Produces: `PointCloudViewer(..., input_path: Path, state_path: Path)`，在 `run()` 中恢复并保存相机参数。

- [ ] **Step 1: Write failing state-file tests**

```python
def test_state_is_saved_and_loaded_per_resolved_input_path(self) -> None:
    state_path = self.temp_path("viewer-state.json")
    first = self.temp_path("first.txt")
    second = self.temp_path("second.txt")
    viewer.save_viewer_state(first, {"color_mode": "class", "point_size": 7}, state_path)
    viewer.save_viewer_state(second, {"color_mode": "original", "point_size": 2}, state_path)
    self.assertEqual(viewer.load_viewer_state(first, state_path)["point_size"], 7)
    self.assertEqual(viewer.load_viewer_state(second, state_path)["color_mode"], "original")

def test_invalid_state_file_falls_back_to_empty_state(self) -> None:
    state_path = self.temp_path("viewer-state.json")
    state_path.write_text("{invalid", encoding="utf-8")
    self.assertEqual(viewer.load_viewer_state(self.temp_path("cloud.txt"), state_path), {})

def test_invalid_state_fields_fall_back_to_empty_state(self) -> None:
    state_path = self.temp_path("viewer-state.json")
    cloud = self.temp_path("cloud.txt")
    state_path.write_text(json.dumps({"files": {str(cloud.resolve()):
        {"color_mode": "unknown", "point_size": 0}}}), encoding="utf-8")
    self.assertEqual(viewer.load_viewer_state(cloud, state_path), {})
```

- [ ] **Step 2: Run state-file tests and verify they fail**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_visualize_classified_point_cloud.ViewerStateTests -v`

Expected: FAIL because `save_viewer_state` and `load_viewer_state` do not exist.

- [ ] **Step 3: Implement JSON state read/write functions and state validation**

```python
def load_viewer_state(input_path: Path, state_path: Path) -> dict[str, object]:
    try:
        document = json.loads(state_path.read_text(encoding="utf-8"))
        state = document.get("files", {}).get(str(input_path.resolve()), {})
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(state, dict):
        return {}
    color_mode = state.get("color_mode")
    point_size = state.get("point_size")
    if color_mode not in {"original", "class"} or not isinstance(point_size, int) or point_size < 1:
        return {}
    return state

def save_viewer_state(input_path: Path, state: dict[str, object], state_path: Path) -> None:
    try:
        document = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        document = {}
    files = document.setdefault("files", {})
    files[str(input_path.resolve())] = state
    state_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 4: Run state-file tests and verify they pass**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_visualize_classified_point_cloud.ViewerStateTests -v`

Expected: PASS.

- [ ] **Step 5: Integrate display and camera restoration**

```python
state = load_viewer_state(self.input_path, self.state_path)
self.set_color_mode(state["color_mode"] if state else "original")
self.point_size = state["point_size"] if state else 3
camera_state = state.get("camera") if state else None
if isinstance(camera_state, dict):
    try:
        intrinsic = camera_state["intrinsic"]
        width, height = int(intrinsic["width"]), int(intrinsic["height"])
        matrix = np.asarray(intrinsic["matrix"], dtype=float).reshape(3, 3)
        extrinsic = np.asarray(camera_state["extrinsic"], dtype=float).reshape(4, 4)
        parameters = o3d.camera.PinholeCameraParameters()
        parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, matrix)
        parameters.extrinsic = extrinsic
        vis.get_view_control().convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
    except (KeyError, TypeError, ValueError, RuntimeError):
        pass

parameters = vis.get_view_control().convert_to_pinhole_camera_parameters()
camera_state = {
    "intrinsic": {
        "width": parameters.intrinsic.width,
        "height": parameters.intrinsic.height,
        "matrix": np.asarray(parameters.intrinsic.intrinsic_matrix).tolist(),
    },
    "extrinsic": np.asarray(parameters.extrinsic).tolist(),
}
save_viewer_state(self.input_path, {"color_mode": self.color_mode, "point_size": self.point_size,
                                    "camera": camera_state}, self.state_path)
```

- [ ] **Step 6: Run full automated verification**

Run: `conda run -n ms_pointcloud_midterm python -m unittest discover -s tests -v && conda run -n ms_pointcloud_midterm python -m py_compile visualize_classified_point_cloud.py tests/test_visualize_classified_point_cloud.py`

Expected: all tests pass and compilation exits with status 0.

- [ ] **Step 7: Run GUI smoke test**

Run: `conda run -n ms_pointcloud_midterm python visualize_classified_point_cloud.py`

Expected: change to class colors, adjust point size, rotate/zoom/pan, close, and reopen. The same color mode, point size, and camera view are restored. A malformed `point_cloud_viewer_state.json` must still allow the viewer to open with defaults.
