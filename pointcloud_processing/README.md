# pointcloud_processing — 点云分类与多光谱 SLAM 数据处理

八万点场景点云的**分类**流程，以及多光谱 SLAM 的数据准备与可视化。

## 目录

```
pointcloud_processing/
├── src/
│   ├── core/                  # 分类核心：点云读取、光谱读取、特征提取、分类器、评价
│   ├── ui/                    # 可视化界面组件
│   ├── gui_app.py             # 分类交互界面入口
│   ├── main.py                # 主流程入口
│   ├── multi_spectral_slam.py # 多光谱点云 SLAM 流程
│   ├── generate_slam_data.py  # 生成 SLAM 输入数据
│   ├── export_for_cc.py       # 导出到 CloudCompare
│   └── visualize_output_gp_dy.py
├── cloudclassify/
│   ├── segment_palette_grid.py          # 7x7 调色板网格分割
│   ├── build_cropped_classification.py  # 裁剪区域分类标签构建
│   ├── tests/
│   ├── docs/                            # 设计文档（plans / specs）
│   └── classified_pointcloud_viewer/     # 分类结果查看器（含独立 README 与依赖）
```

## 运行

```bash
cd pointcloud_processing

# 分类主流程 / GUI
python src/main.py --help
python src/gui_app.py

# 调色板网格分割与裁剪分类
python cloudclassify/segment_palette_grid.py --help
python cloudclassify/build_cropped_classification.py --help

# 测试（需在对应模块目录下执行）
python -m pytest cloudclassify/classified_pointcloud_viewer/tests -q
cd cloudclassify && python -m pytest tests -q
```

> 部分测试依赖真实点云文本（`palette.txt`、`cropped_point_cloud_classified.txt` 等），这些数据不入库，缺失时相应用例会失败。
