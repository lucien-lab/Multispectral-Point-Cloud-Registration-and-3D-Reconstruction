# Palette Seven-by-Seven Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 覆盖更新标签点云，使 24 个色块按 7×7 理想采样格保留原始点：每块最多 49 点，原始点不足时保留可用点。

**Architecture:** 在已拟合的 4×6 正方形网格中，为每格生成 49 个等距目标位置。将格内点投到 X-Z 平面，使用最小总代价的一对一匹配选择距离目标位置最近的点；仅第 20 块因原始点不足保留全部 44 点。未选择点为背景。

**Tech Stack:** Python 3.11、NumPy、SciPy `linear_sum_assignment`、unittest；Python 命令一律通过 `conda run -n ms_pointcloud_midterm` 执行。

## Global Constraints

- 不修改 `palette.txt`。
- 覆盖 `palette_labeled.txt` 与 `palette_label_counts.csv`。
- 仅从各自等边正方形内选择原始点，不跨格、不新增点。
- 每个前景标签最多 49 点；实际不足 49 的标签允许为 47 或 48 点，第 20 块为 44 点；其余点为 0。
- 输出保留 2143 行和原始 8 列，末列为整数标签。
- 当前目录没有 `.git`，不执行提交。

---

### Task 1: 实现 7×7 目标格与一对一选点

**Files:**
- Modify: `segment_palette_grid.py`
- Modify: `tests/test_segment_palette_grid.py`

**Interfaces:**
- Produces: `ideal_sample_positions(grid: SquareGrid, row: int, col: int, size: int = 7) -> np.ndarray`，返回 `(49, 2)` 的 X/Z 目标位置。
- Produces: `assign_regular_labels(points: np.ndarray, grid: SquareGrid, size: int = 7) -> np.ndarray`，返回 `(N,)` 整数标签。

- [ ] **Step 1: 写入失败测试**

```python
def test_ideal_sample_positions_form_a_seven_by_seven_square_grid():
    targets = ideal_sample_positions(grid, 0, 0)
    assert targets.shape == (49, 2)
    assert len(np.unique(targets[:, 0])) == 7
    assert len(np.unique(targets[:, 1])) == 7

def test_regular_assignment_caps_a_full_cell_at_49_points():
    labels = assign_regular_labels(points_from_one_8_by_8_cell, grid)
    assert np.count_nonzero(labels == 1) == 49
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Expected: 两个新接口尚未定义而失败。

- [ ] **Step 3: 实现最小规则采样逻辑**

为每个单元生成中心位于正方形内部的 7×7 等距位置。对格内 `N>=49` 点，以 49×N 的 X-Z 欧氏距离矩阵调用 `linear_sum_assignment` 并选出 49 个匹配点；对 `N<49` 点保留全部。所有格外点为 0。

- [ ] **Step 4: 运行测试，确认通过**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Expected: 全部测试通过。

### Task 2: 覆盖标签输出并进行几何与数量验证

**Files:**
- Modify: `segment_palette_grid.py`
- Modify: `palette_labeled.txt`
- Modify: `palette_label_counts.csv`

**Interfaces:**
- Consumes: `assign_regular_labels` 与既有 `write_labeled_cloud`、`write_label_counts`。
- Produces: CLI 参数 `--regular-size 7`，默认使用规则采样标签。

- [ ] **Step 1: 写入失败测试**

```python
def test_real_regular_labels_have_requested_counts():
    labels = assign_regular_labels(real_points, real_grid)
    counts = np.bincount(labels, minlength=25)
    assert np.all(counts[1:] <= 49)
    assert counts[20] == 44
    assert np.all((counts[1:] >= 47) | (counts[1:] == 44))
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Expected: 真实点云仍使用旧标签逻辑，计数断言失败。

- [ ] **Step 3: 切换 CLI 至规则标签并覆盖输出**

默认 CLI 调用 `assign_regular_labels`。运行：

```bash
conda run -n ms_pointcloud_midterm python segment_palette_grid.py \
  --input palette.txt --output palette_labeled.txt \
  --counts palette_label_counts.csv --regular-size 7
```

- [ ] **Step 4: 运行完整验证**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid -v`

Run: `conda run -n ms_pointcloud_midterm python -c "import numpy as np; d=np.loadtxt('palette_labeled.txt', comments='//', skiprows=2); c=np.bincount(d[:,-1].astype(int), minlength=25); assert d.shape==(2143,9); assert np.all(c[1:20]==49); assert c[20]==44; assert np.all(c[21:25]==49); print(c.tolist())"`

Expected: 测试通过；输出计数为背景 980，前景标签为 47–49 点，且第 20 块为 44 点。
