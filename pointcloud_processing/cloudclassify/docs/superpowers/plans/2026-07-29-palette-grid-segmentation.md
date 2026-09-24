# Palette Grid Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `palette.txt` 的 2143 个点写入一个新点云文件，并在第 9 列加入 0–24 标签，使 24 个前景标签对应 4×6 个等大正方形色块。

**Architecture:** 一个独立 Python 脚本读取 CloudCompare ASCII 点云格式，以 RANSAC 拟合 `y=ax+bz+c` 色卡前表面，并在 X-Z 投影中拟合 4×6 等边正方形网格。脚本按行优先生成 1–24 标签；网格外或非前表面点为背景 0。测试覆盖输入解析、规则编号、几何约束与完整输出。

**Tech Stack:** Python 3.11、NumPy、scikit-learn（仅用于稳健的颜色/一维中心聚类）、unittest；所有 Python 命令通过 `conda run -n ms_pointcloud_midterm` 执行。

## Global Constraints

- 不修改 `palette.txt`。
- 新点云为 `palette_labeled.txt`，保留原 8 列，在末尾追加 `label`。
- `label=0` 表示背景，`label=1..24` 表示前视图按从上到下、从左到右的 4×6 色块。
- 24 个色块在 X-Z 投影中必须为相同边长的正方形。
- 当前目录没有 `.git`，不执行提交操作。

---

### Task 1: 建立可测试的输入、网格和编号基础

**Files:**
- Create: `segment_palette_grid.py`
- Create: `tests/test_segment_palette_grid.py`

**Interfaces:**
- Produces: `load_palette(path: str) -> tuple[str, np.ndarray]`，返回首行列名和 `(N, 8)` 浮点数组。
- Produces: `grid_label(row: int, col: int) -> int`，返回 `row * 4 + col + 1`。
- Produces: `SquareGrid(x0: float, z_top: float, side: float, pitch_x: float, pitch_z: float)`；`cell_bounds(row, col) -> tuple[float, float, float, float]` 返回正方形的 X/Z 边界。

- [ ] **Step 1: 写入失败测试**

```python
def test_grid_labels_are_row_major_and_one_based():
    assert grid_label(0, 0) == 1
    assert grid_label(0, 3) == 4
    assert grid_label(5, 0) == 21
    assert grid_label(5, 3) == 24

def test_each_cell_has_equal_x_and_z_side_length():
    grid = SquareGrid(x0=-0.07, z_top=0.01, side=0.012,
                      pitch_x=0.015, pitch_z=0.015)
    xmin, xmax, zmin, zmax = grid.cell_bounds(2, 1)
    assert xmax - xmin == 0.012
    assert zmax - zmin == 0.012
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Expected: 导入 `segment_palette_grid` 失败，因为接口尚未实现。

- [ ] **Step 3: 实现最小输入与网格接口**

```python
def load_palette(path: str) -> tuple[str, np.ndarray]:
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        point_count = int(handle.readline().strip())
    data = np.loadtxt(path, comments="//", skiprows=2)
    if data.shape != (point_count, 8):
        raise ValueError("palette rows do not match declared point count")
    return header, data

def grid_label(row: int, col: int) -> int:
    return row * 4 + col + 1
```

实现 `SquareGrid.cell_bounds`，并对 `row in [0,5]`、`col in [0,3]` 以外的输入抛出 `ValueError`。

- [ ] **Step 4: 运行测试，确认通过**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Expected: 两个测试均通过。

### Task 2: 拟合前表面与 4×6 正方形网格

**Files:**
- Modify: `segment_palette_grid.py`
- Modify: `tests/test_segment_palette_grid.py`

**Interfaces:**
- Consumes: `(N, 8)` 点数据。
- Produces: `fit_palette_grid(points: np.ndarray) -> tuple[SquareGrid, FrontPlane]`，返回网格和前表面 `y=ax+bz+c` 及允许残差。
- Produces: `point_to_grid_cell(grid: SquareGrid, x: float, z: float) -> tuple[int, int] | None`，仅在点落入某一正方形内部时返回 `(row, col)`。

- [ ] **Step 1: 写入失败测试**

```python
def test_grid_cell_rejects_gaps_and_accepts_square_interior():
    grid = SquareGrid(x0=0.0, z_top=0.04, side=0.01,
                      pitch_x=0.012, pitch_z=0.012)
    assert point_to_grid_cell(grid, 0.005, 0.035) == (0, 0)
    assert point_to_grid_cell(grid, 0.0108, 0.035) is None
    assert point_to_grid_cell(grid, -0.001, 0.035) is None
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Expected: `point_to_grid_cell` 尚未定义而失败。

- [ ] **Step 3: 实现网格拟合与单元归属**

对亮度或饱和度较高的点形成候选集；用 RANSAC 拟合前表面平面。以该平面附近点的 X/Z 覆盖范围估计 4 个列中心和 6 个行中心，拟合统一 X/Z 网格节距；边长取两个节距中较小者的 80%，并强制同一 `side` 用于所有单元。实现 `point_to_grid_cell`，使网格间隙和方格外点返回 `None`。

- [ ] **Step 4: 运行测试，并对真实数据记录拟合参数**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Run: `conda run -n ms_pointcloud_midterm python segment_palette_grid.py --input palette.txt --dry-run`

Expected: 单元归属测试通过；dry-run 打印 `y_front`、正方形边长及 24 个单元边界，不创建输出文件。

### Task 3: 颜色校验、标签写入与端到端验证

**Files:**
- Modify: `segment_palette_grid.py`
- Modify: `tests/test_segment_palette_grid.py`
- Create: `palette_labeled.txt`
- Create: `palette_label_counts.csv`

**Interfaces:**
- Consumes: `fit_palette_grid`、输入点及每个单元内的稳健 RGB 中位数。
- Produces: `assign_labels(points: np.ndarray, grid: SquareGrid, y_front: float, y_tolerance: float) -> np.ndarray`，返回 `(N,)` 的整型标签。
- Produces: CLI `python segment_palette_grid.py --input palette.txt --output palette_labeled.txt --counts palette_label_counts.csv`。

- [ ] **Step 1: 写入失败测试**

```python
def test_assignment_uses_zero_for_background_and_1_to_24_for_cells():
    labels = assign_labels(points, grid, y_front=1.0, y_tolerance=0.001)
    assert labels.dtype.kind in "iu"
    assert set(labels).issubset(set(range(25)))
    assert labels[background_index] == 0
    assert labels[inside_first_cell_index] == 1

def test_writer_adds_one_label_column(tmp_path):
    write_labeled_cloud(tmp_path / "out.txt", header, points, labels)
    loaded = np.loadtxt(tmp_path / "out.txt", comments="//", skiprows=2)
    assert loaded.shape == (len(points), 9)
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Expected: 赋值和写入接口尚未实现而失败。

- [ ] **Step 3: 实现颜色校验、写入与统计**

用 RGB 候选点定位网格后，以前表面平面残差和正方形边界进行标注，保证色块内部不因阴影产生空洞；其他点标为背景。输出首行为原首行加 ` label`，第二行为 `2143`，后续每行写 8 个原始数值与整数标签。CSV 记录 `label,point_count` 共 25 行。

- [ ] **Step 4: 运行完整验证**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Run: `conda run -n ms_pointcloud_midterm python segment_palette_grid.py --input palette.txt --output palette_labeled.txt --counts palette_label_counts.csv`

Run: `conda run -n ms_pointcloud_midterm python -c "import numpy as np; d=np.loadtxt('palette_labeled.txt', comments='//', skiprows=2); labels=d[:, -1].astype(int); assert d.shape == (2143, 9); assert labels.min() == 0 and labels.max() == 24; assert len(np.unique(labels)) == 25; print(np.bincount(labels, minlength=25))"`

Expected: 所有单元测试通过；输出有 2143 行、9 列、标签只在 0–24 内，全部 25 个标签都出现。
