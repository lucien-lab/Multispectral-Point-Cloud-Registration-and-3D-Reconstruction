# 分类点云

使用 Open3D 可视化带 RGB 和分类标签的点云文本文件。默认显示原始颜色，并可在窗口中切换为按类别着色。

## 分类方法

点云文本的每个点都带有一个 `class_id` 标签。`class_id_mapping.csv` 将该编号对应到具体类别名称，例如背景、瓶子、汽车或石块。查看器按 `1` 时，根据已有的 `class_id` 为同类点赋予相同颜色，用于观察分类结果；该操作只做显示，不会重新计算或修改原始分类标签。

## 文件说明

- `visualize_classified_point_cloud.py`：查看器主程序。
- `cropped_point_cloud_classified.txt`：默认输入点云，字段为 `point_id,x,y,z,r,g,b,class_id`。
- `class_id_mapping.csv`：类别编号与类别名称对照表。
- `point_cloud_viewer_state.json`：运行后自动生成的显示状态文件。


## 交互控制

| 按键或操作 | 功能 |
| --- | --- |
| `0` | 显示文件中的原始 RGB 颜色 |
| `1` | 按 `class_id` 显示分类标签颜色 |
| `+` / `=` | 增大点大小 |
| `-` / `_` | 减小点大小，最小为 1 |
| `R` | 重置视角 |
| `Q` / `Esc` | 退出窗口 |
| 中键或 `Ctrl` + 左键 | 平移视角 |

## 状态恢复与路径

关闭窗口时，查看器会把颜色模式、点大小和相机视角保存到 `point_cloud_viewer_state.json`。下次打开相同的相对路径时，自动恢复对应设置。

