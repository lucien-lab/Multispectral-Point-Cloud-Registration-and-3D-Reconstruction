# Multispectral Point Cloud Registration and 3D Reconstruction

多光谱（高光谱）激光雷达点云的**配准（registration）**与**三维重建 / 处理**代码库。

面向六波段反射率 + 几何的点云数据，包含：光谱-几何联合粗配准、几何 / RGB / 六波段三类 ICP 精配准与对比评测、颜色标定、点云生成与可视化、点云分类与 SLAM 数据处理、闭环地点识别描述子、以及 MATLAB 侧的数据处理与曲面显示工具。

> 本仓库只包含**代码**。采集数据、实验输出、论文 / 报告文档、专利材料均不纳入版本控制（见 `.gitignore`）。

---

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `registration/` | **核心**：六波段光谱 + 几何联合配准。粗配准（FPFH 33 维 + 光谱/RGB 联合评分）、`icp_geometry` / `icp_rgb` / `icp_spectral` / `icp_spectral_robust` 四种精配准、扰动与评测协议、三方法对比、权重消融、参数调优、结果报表 |
| `calibration_and_joint_registration/` | 白板颜色标定（`color_calibration_core`、`calibrate_white_from_photo`）、点云与照片配准、颜色-几何联合配准、手动配准 Web 小工具（HTML/JS + Flask 后端） |
| `pointcloud_generation/` | 由原始 bin + 角度文件生成点云、六波段反射率精化（`refine_range`）、几何 / 颜色 / 左右侧可视化、逐点光谱文件转彩色点云（`handle_datas`） |
| `reconstruction/` | 活动室场景配准前后重建对比、调色板点云 Poisson 重建、公开数据集（Open3D Demo ICP / Bunny）配准重建基线、ICP 学习脚本 |
| `pointcloud_processing/` | 八万点场景点云分类（`segment_palette_grid`、`build_cropped_classification`、特征提取 / 分类器 / 评价）、分类结果查看器、多光谱 SLAM 数据生成与 `multi_spectral_slam` |
| `slam_place_recognition/` | 多光谱 LiDAR SLAM 闭环地点识别：JSSS 谱描述子 + M2DP / Scan Context 基线、检索、位姿图、漂移与鲁棒性实验 |
| `experiments/fortune_tree/` | 发财树 / 转台多视角实验：配准、波段消融、可辨识性、重复性、六视角对比、指标分析等脚本 |
| `baselines/icp_registration/` | 早期 ICP 基线：纯几何 `icp_test` 与 RGB 联合 `icp_rgb`，含调色板重建与报告图脚本 |
| `matlab/` | MATLAB 工具：gp 数据读取与整理、点云曲面显示、CMF 生成、A2026_5_5 处理脚本 |

各子目录内的相对结构与原始工程保持一致（`src/` 包、`tests/` 测试、`configs/` 配置），便于直接运行与 pytest 采集。

---

## 环境依赖

```bash
conda env create -f registration/environment.yml   # 推荐：python 3.11 + numpy/scipy/matplotlib + open3d
# 或
pip install -r requirements.txt
```

核心依赖：`numpy`、`scipy`、`matplotlib`、`open3d`、`pandas`、`scikit-learn`、`Pillow`、`Flask`。

---

## 快速开始

```bash
# 1) 光谱/几何联合配准实验（读 registration/configs 配置）
python registration/run_experiment.py --help
python registration/run_spectral_recommended_experiment.py --help
python registration/tune_coarse_registration.py --help

# 2) 运行回归测试
pytest registration/tests -q
pytest pointcloud_generation -q
pytest calibration_and_joint_registration/tests -q

# 3) 由原始 bin 生成点云（自行准备自己的数据目录）
python pointcloud_generation/generate_pointcloud.py --help

# 4) 可视化
python pointcloud_generation/visualize_pointcloud.py --help
```

数据路径均通过命令行参数或配置项传入，仓库内**不含**任何数据文件，示例数据请按脚本 `--help` 中的目录约定自行准备。

---

## 数据与文档约定

- **不入库**：原始 `*.bin` / `*.ply` / `*.pcd` / `*.mat` / `*.csv` / `*.txt` 导出、结果目录、图片输出。
- **不入库**：论文与报告文档（`*.docx`、`*.pdf`、草稿、评审意见、报告目录）。
- **不纳入**：专利申请相关材料与脚本。
- 仅保留：算法与工具代码、单元测试、运行配置（`*.json` / `*.yml`）、与代码直接相关的说明文档（`*.md`）。

## 引用与许可

代码用于多光谱 LiDAR 点云配准与三维重建研究。论文发表后请引用对应工作；如需转载或商用请先联系作者。
