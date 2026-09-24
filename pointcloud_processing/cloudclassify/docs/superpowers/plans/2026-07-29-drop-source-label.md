# Drop Source Label Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从统一分类点云移除恒为 0 的 `source_label`，保留 7 个源字段与最后的 `class_id`。

**Architecture:** 修改现有写入函数，使其输出 8 列 CSV；重新生成分类点云。类别映射逻辑、ID 关联逻辑与映射 CSV 均不改变。

**Tech Stack:** Python 3.11、NumPy、unittest；所有 Python 命令通过 `conda run -n ms_pointcloud_midterm` 执行。

## Global Constraints

- 不修改父级 `cropped_point_cloud.txt`。
- `cropped_point_cloud_classified.txt` 仅含 `point_id,x,y,z,r,g,b,class_id` 八列。
- `class_id_mapping.csv` 保持 40 类内容不变。
- 当前目录没有 `.git`，不执行提交。

---

### Task 1: 调整输出列并重新生成

**Files:**
- Modify: `build_cropped_classification.py`
- Modify: `tests/test_build_cropped_classification.py`
- Modify: `cropped_point_cloud_classified.txt`

**Interfaces:**
- Produces: `write_classified_point_cloud(path, points, labels)`，写入 8 列 CSV。

- [ ] **Step 1: 写入失败测试**

```python
def test_writer_preserves_seven_source_columns_and_appends_class_id():
    write_classified_point_cloud(output, points, np.array([25]))
    assert header == "point_id,x,y,z,r,g,b,class_id"
    assert loaded.shape == (8,)
    assert int(loaded[-1]) == 25
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_build_cropped_classification -v`

Expected: 旧写入函数仍输出 `source_label`，列数与首行断言失败。

- [ ] **Step 3: 实现最小写入变更并重新生成输出**

写入 `point[0:7]` 与 `class_id`，不写 `point[7]`。运行：

```bash
conda run -n ms_pointcloud_midterm python build_cropped_classification.py \
  --source ../cropped_point_cloud.txt --class-dir . \
  --output cropped_point_cloud_classified.txt --mapping class_id_mapping.csv
```

- [ ] **Step 4: 完整验证**

Run: `conda run -n ms_pointcloud_midterm python -m unittest tests.test_segment_palette_grid tests.test_build_cropped_classification -v`

Run: `conda run -n ms_pointcloud_midterm python -c "import numpy as np; s=np.loadtxt('../cropped_point_cloud.txt', delimiter=','); d=np.loadtxt('cropped_point_cloud_classified.txt', delimiter=',', skiprows=1); assert d.shape==(36840,8); assert np.allclose(d[:,:7],s[:,:7]); assert d[:,-1].min()==0 and d[:,-1].max()==39"`

Expected: 全部测试通过；结果为 36,840×8，类别范围 0–39。
