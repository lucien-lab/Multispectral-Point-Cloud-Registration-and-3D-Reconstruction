# 调色板点云 ICP 配准与三维重建实验指南

本目录用于完成两组高光谱激光雷达调色板扫描点云的刚体配准、点云融合、Poisson 曲面重建和定量评价。当前实验以 `up_t.txt` 为移动点云、`down.txt` 为固定点云，先通过点到点 ICP 求解刚体变换，再比较“未配准直接融合”和“ICP 后融合”两种输入的三维重建结果。

> 本实验的 RGB 三通道只是光谱属性的简化代理。`icp_test.py` 是仅使用 XYZ 的几何基线，`icp_rgb.py` 在局部几何候选中加入 RGB 联合约束。RGB 实验不能直接证明算法对完整高光谱数据具有鲁棒性。

## 1. 实验流程

```text
up.txt
  │  人工刚体变换 + RGB 高斯噪声
  ▼
up_t.txt ───────────────┐
  │                     │
  │ 点到点 ICP          │ 未配准直接融合
  ▼                     ▼
up_t_aligned_refined.txt + down.txt
  │                     │
  └── ICP 后融合 ───────┘
             │
             ▼
       Poisson 曲面重建
             │
             ▼
  PLY 网格、CSV 指标、PNG/PDF 图表
```

完整实验分为以下阶段：

1. 检查原始点云格式和空间尺度。
2. 对移动点云施加已知刚体扰动，并对 RGB 添加高斯噪声。
3. 使用 ICP 将移动点云对齐到固定点云。
4. 保存对齐点云、4×4 变换矩阵和融合点云。
5. 分别对未配准融合点云与 ICP 后融合点云执行 Poisson 重建。
6. 计算配准指标、网格指标和双向重建误差，并生成报告图片。

## 2. 目录中的主要文件

### 输入数据

| 文件 | 作用 |
|---|---|
| `up.txt` | 原始移动扫描点云 |
| `up_t.txt` | 加入刚体位姿扰动和 RGB 高斯噪声后的移动点云 |
| `down.txt` | 固定扫描点云，即 ICP 目标点云 |

### 核心代码

| 文件 | 作用 |
|---|---|
| `icp_test.py` | 点到点 ICP、刚体变换保存、点云融合和配准前后可视化 |
| `icp_rgb.py` | 几何初始化、RGB-几何联合对应、加权刚体求解和消融指标 |
| `generate_report_figures.py` | 复现论文中的 ICP 指标、收敛曲线和参数敏感性分析 |
| `reconstruct_palette.py` | Poisson 曲面重建、网格评价和重建图表生成 |
| `visualize_pointclouds.py` | 原始点云概览、投影和标签分布检查 |

### 关键结果

| 文件 | 内容 |
|---|---|
| `up_t_aligned_refined.txt` | ICP 后的移动点云，保留原始属性列 |
| `icp_transform_refined.txt` | 从 `up_t.txt` 到 `down.txt` 的 4×4 齐次变换矩阵 |
| `icp_merged_refined.csv` | ICP 后融合点云，末列为点云来源标记 |
| `report_metrics.csv` | ICP 配准前后定量指标 |
| `report_convergence.csv` | 每轮 ICP 的平均对应点距离 |
| `report_parameter_sweep.csv` | 不同对应点保留比例的敏感性结果 |
| `up_t_aligned_rgb.txt` | RGB-几何联合 ICP 后的移动点云 |
| `icp_transform_rgb.txt` | RGB-几何联合 ICP 的 4×4 齐次变换矩阵 |
| `icp_rgb_metrics.csv` | 纯几何与 RGB-几何 ICP 的消融对比指标 |
| `icp_rgb_convergence.csv` | RGB-几何联合 ICP 的逐轮收敛记录 |
| `icp_rgb_comparison.png` | 几何误差与 RGB 对应残差对比图 |
| `palette_reconstruction_before.ply` | 未配准直接融合后的 Poisson 网格 |
| `palette_reconstruction_after.ply` | ICP 后融合的 Poisson 网格 |
| `reconstruction_metrics.csv` | 两种重建条件的网格与误差指标 |
| `reconstruction_figure*.png/.pdf` | 三维重建对比、指标、误差 CDF 和空间误差图 |

## 3. 数据格式

输入 TXT 文件使用英文逗号分隔，无表头，每行 7 列：

```text
X,Y,Z,R,G,B,label
```

示例：

```text
-0.06303314,1.46366010,-0.02498288,10,9,164,0.000000
```

- `X,Y,Z`：三维坐标。当前实验按米理解，因此 0.005 对应 5 mm。
- `R,G,B`：0–255 范围内的颜色或光谱代理值。
- `label`：原始类别或波段相关标签，ICP 不使用该列。
- 文件不能包含表头、空字符串、NaN 或无穷值。

`icp_merged_refined.csv` 带有表头，其最后一列为 `source_flag`：移动点云为 1，固定点云为 0。

在使用新数据前，至少检查以下事项：

- 两组点云是否采用相同坐标单位和坐标系定义。
- RGB 是否确实位于第 4–6 列。
- 点云是否存在 NaN、重复点或极端离群点。
- 两组扫描是否具有足够的几何重叠区域。

## 4. Conda 环境

当前结果使用以下环境验证：

```text
环境名      ms_pointcloud_midterm
Python      3.11.15
NumPy       2.4.6
SciPy       1.17.1
Matplotlib  3.11.0
Open3D      0.19.0
```

若本机已经存在该环境，可直接检查依赖：

```bash
conda run -n ms_pointcloud_midterm python -c \
  "import numpy, scipy, matplotlib, open3d; print(numpy.__version__, scipy.__version__, matplotlib.__version__, open3d.__version__)"
```

若需要新建环境，可先执行：

```bash
conda create -n ms_pointcloud_midterm python=3.11 -y
conda activate ms_pointcloud_midterm
conda install -c conda-forge numpy scipy matplotlib pandas -y
python -m pip install open3d==0.19.0
```

在 macOS 或受限环境中，Matplotlib 可能提示缓存目录不可写。可在项目内建立缓存目录：

```bash
mkdir -p .cache/matplotlib .cache/fontconfig
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export XDG_CACHE_HOME="$PWD/.cache"
```

## 5. 快速复现本报告实验

以下命令均应在本目录执行：

```bash
cd /Users/lucien/workspace/graduate-study/papers/ICP_ws
```

### 5.1 检查输入点云

```bash
conda run -n ms_pointcloud_midterm python visualize_pointclouds.py
```

输出：

- `pointcloud_overview.png`
- `pointcloud_comparison.png`
- 终端中的点数、坐标范围和标签分布

### 5.2 执行论文参数下的精细 ICP

> 下列命令会覆盖同名的 `up_t_aligned_refined.txt`、`icp_transform_refined.txt`、`icp_merged_refined.csv` 和配准预览图。需要保留历史实验时，应先改用带实验编号的输出文件名。

```bash
conda run -n ms_pointcloud_midterm python icp_test.py \
  --source up_t.txt \
  --target down.txt \
  --max-iterations 500 \
  --tolerance 1e-12 \
  --distance-quantile 1.0 \
  --aligned-output up_t_aligned_refined.txt \
  --transform-output icp_transform_refined.txt \
  --output icp_merged_refined.csv \
  --plot-output icp_registration_refined.png
```

本组数据的参考结果为：

```text
迭代次数：107
配准后平均最近邻距离：约 0.002530 m
配准后最近邻距离中位数：约 0.002209 m
配准后双向 Chamfer：约 0.003656 m
```

注意：`icp_test.py` 自身的默认参数是 `80 / 1e-8 / 0.7`，适合快速测试或部分重叠点云，并不是本报告采用的精细参数。复现实验时必须显式传入上述参数。

### 5.3 执行 RGB-几何联合 ICP

下列命令使用纯几何 ICP 获得初始位姿，再在每个源点的 30 个局部几何候选中加入 RGB 约束：

```bash
MPLCONFIGDIR="$PWD/.cache/matplotlib" \
XDG_CACHE_HOME="$PWD/.cache" \
conda run -n ms_pointcloud_midterm python icp_rgb.py \
  --source up_t.txt \
  --target down.txt \
  --rgb-weight 1.0 \
  --rgb-scale 30 \
  --local-candidates 30 \
  --distance-quantile 0.9
```

该命令默认生成：

- `up_t_aligned_rgb.txt` 与 `icp_transform_rgb.txt`。
- `icp_merged_rgb.csv` 与 `icp_registration_rgb.png`。
- `icp_rgb_convergence.csv`、`icp_rgb_metrics.csv` 和 `icp_rgb_comparison.png`。

本组数据的参考结果为：联合对应点 RGB 平均残差约为 60.87，而纯几何对应的 RGB 平均残差约为 147.84；独立几何 Chamfer 约由 3.656 mm 变为 3.679 mm。RGB 一致性明显提高，但几何 Chamfer 约增加 0.6%，因此应将结果解释为颜色—几何折中，而不是所有指标均提高。

### 5.4 生成纯几何 ICP 报告指标与图表

```bash
conda run -n ms_pointcloud_midterm python generate_report_figures.py
```

该脚本固定使用：

- 最大迭代次数：500
- 收敛阈值：`1e-12`
- 最终对应点保留比例：1.0
- 敏感性分析比例：0.4、0.5、0.6、0.7、0.8、0.9、1.0

它会生成 4 张 PNG、4 张 PDF、3 个 CSV 和 `icp_experiment_results.md`。脚本会重新运行 ICP，但只在内存中使用对齐点云，不会重新保存 `up_t_aligned_refined.txt`。

### 5.5 执行 Poisson 三维重建与评价

确认 `up_t_aligned_refined.txt` 已由步骤 5.2 生成后执行：

```bash
conda run -n ms_pointcloud_midterm python reconstruct_palette.py
```

该脚本会同时重建两种条件：

- 未配准重建：`up_t.txt + down.txt`
- ICP 后重建：`up_t_aligned_refined.txt + down.txt`

核心重建参数：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `SEED` | 42 | 网格表面均匀采样随机种子 |
| `POISSON_DEPTH` | 8 | Poisson 八叉树深度 |
| `NORMAL_RADIUS` | 0.012 m | 法向估计搜索半径 |
| `NORMAL_MAX_NN` | 40 | 法向估计最大邻点数 |
| 法向一致化邻域 | 30 | 一致切平面法向的邻域大小 |
| `COVERAGE_THRESHOLD` | 0.005 m | 精确率、完整率和 F1 的距离阈值 |
| `SURFACE_SAMPLES` | 20000 | 网格到参考点云方向的采样数 |
| Poisson `scale` | 1.10 | 重建区域相对输入包围盒的尺度 |
| Poisson `n_threads` | 1 | 降低并行计算造成的不确定性 |

当前调色板点云的中位点间距约为 1.5–1.7 mm，因此 12 mm 法向半径能够覆盖多个邻点，5 mm 阈值约为点间距的 3 倍。更换数据尺度后，不能直接照搬这两个距离参数。

本报告对应的重建参考结果：

| 指标 | 未配准重建 | ICP 后重建 |
|---|---:|---:|
| 网格顶点数 | 7577 | 6688 |
| 网格三角面数 | 14929 | 12990 |
| 显著连通分量 | 2 | 1 |
| 对称 Chamfer | 84.985 mm | 1.979 mm |
| 5 mm 表面精确率 | 11.54% | 88.74% |
| 5 mm 完整率 | 91.84% | 91.41% |
| 5 mm F1 | 20.50% | 90.06% |

数值可能因平台、Open3D 版本和浮点计算产生极小差异，但不应改变主要趋势。

## 6. 人工扰动的复现方法

当前 `up_t.txt` 由 `up.txt` 施加以下扰动得到：

- 绕 Z 轴旋转 20°。
- 平移向量为 `(0.10, -0.05, 0.03)`。
- RGB 噪声服从 `N(0, 8²)`。
- 随机种子为 42。
- 加噪后 RGB 裁剪到 `[0, 255]`。

核心计算形式为：

```python
theta = np.deg2rad(20.0)
rotation = np.array([
    [np.cos(theta), -np.sin(theta), 0.0],
    [np.sin(theta),  np.cos(theta), 0.0],
    [0.0,            0.0,           1.0],
])
translation = np.array([0.10, -0.05, 0.03])

transformed_xyz = (rotation @ xyz.T).T + translation
rng = np.random.default_rng(42)
noisy_rgb = np.clip(np.rint(rgb + rng.normal(0.0, 8.0, rgb.shape)), 0, 255)
```

为保证对照实验公平，除 XYZ 和 RGB 外的其他属性列应原样保留。开展多次噪声实验时，应记录每次随机种子，并报告均值和标准差，而不是只挑选单次结果。

## 7. ICP 算法与参数解释

`icp_test.py` 实现的是点到点 ICP：

1. 使用 `scipy.spatial.cKDTree` 为当前移动点寻找固定点云中的最近邻。
2. 按 `distance_quantile` 保留距离较小的对应点。
3. 对保留的点对进行 SVD，求解最小二乘刚体旋转和平移。
4. 将增量变换累积到移动点云。
5. 当相邻两轮裁剪后平均对应距离变化小于 `tolerance` 时停止。

### `distance_quantile`

- `1.0`：保留全部最近邻。本组调色板数据的参数扫描中效果最好。
- `0.6–0.9`：适用于存在少量离群点或部分重叠的数据。
- `0.4–0.6`：裁剪较强，可能提升离群点鲁棒性，也可能丢失有效几何约束。

不要只根据单向平均最近邻距离选择该参数，应同时检查反向距离、双向 Chamfer、重叠区域和最终网格伪影。

### `max_iterations` 与 `tolerance`

- 增大最大迭代次数只提供更多优化机会，不保证跳出局部最优。
- `tolerance` 必须结合坐标单位理解。以米为单位时，`1e-12` 是非常严格的停止条件。
- 快速调试可使用 `80 / 1e-8`，最终实验再使用更严格参数。

### 初始位姿

ICP 是局部优化算法。若新数据存在大角度旋转、严重平移、对称结构或较低重叠率，建议先加入粗配准，例如：

- 人工选取对应点。
- FPFH 特征配合 RANSAC。
- 基于传感器外参或扫描位姿提供初值。

粗配准完成后，再用当前点到点 ICP 精化。

### RGB-几何联合方程

`icp_rgb.py` 不把 RGB 当作三维空间坐标，也不对 RGB 施加刚体旋转。对当前位姿下的源点 \(p_i\) 及其颜色 \(c_i\)，在目标点 \(q_j\) 及颜色 \(d_j\) 中建立对应关系：

\[
j_i^*=\arg\min_{j\in\mathcal N_K(p_i)}
\left[
\frac{\|Rp_i+t-q_j\|_2^2}{\sigma_g^2}
+\lambda_{rgb}\frac{\|c_i-d_j\|_2^2}{\sigma_c^2}
\right].
\]

其中：

- \(\mathcal N_K(p_i)\) 是当前空间位置附近的 \(K\) 个目标候选点，默认 \(K=30\)。
- \(\sigma_g\) 是几何尺度，由 8 mm 逐步退火到 3 mm。
- \(\sigma_c=30\) 是 RGB 向量的鲁棒尺度，不是单通道高斯噪声标准差。
- \(\lambda_{rgb}=1.0\) 控制颜色项相对权重。

确定对应关系后，颜色残差生成高斯权重，几何残差生成 Cauchy 鲁棒权重：

\[
w_i=
\exp\left(-\frac{\lambda_{rgb}}{2}\frac{\|c_i-d_{j_i^*}\|_2^2}{\sigma_c^2}\right)
\cdot
\frac{1}{1+\left(\frac{\|Rp_i+t-q_{j_i^*}\|_2}{2.5\,m_g}\right)^2},
\]

其中 \(m_g\) 为当前保留对应点几何距离的中位数。刚体增量通过加权 SVD 求解：

\[
(R^*,t^*)=\arg\min_{R\in SO(3),t}
\sum_i w_i\|Rp_i+t-q_{j_i^*}\|_2^2.
\]

RGB 项对固定点对而言不直接包含 \(R,t\)，但它通过对应点选择 \(j_i^*\) 和权重 \(w_i\) 进入迭代方程。若直接把 RGB 拼成 XYZRGB 六维坐标并求刚体变换，会错误地允许空间旋转混合颜色轴，因此不可采用。

### RGB-几何 ICP 的两阶段流程与参数

1. 使用 `icp_test.py` 中的纯几何 ICP 和全部对应点获得稳定初始位姿。
2. 对每个移动点查询局部 \(K\) 个几何候选。
3. 用归一化 XYZ+RGB 联合代价选择候选并裁剪联合残差最大的 10%。
4. 用颜色高斯权重和几何 Cauchy 权重执行加权 SVD。
5. 几何尺度退火完成且联合误差与位姿增量同时稳定后停止。

参数调节建议：

- `rgb-weight=0` 退化为局部几何对应，可作为代码消融检查。
- RGB 噪声增大时可适当增大 `rgb-scale`，但应通过多随机种子实验选择。
- `local-candidates` 太小会使 RGB 无法纠正几何错配，太大则可能跨色块误配。
- 当前默认值来自这一对调色板点云，不能未经验证直接用于其他场景。
- 选择参数时应同时报告几何 Chamfer 和 RGB 残差，不能只优化其中一项。

## 8. 三维重建指标解释

`reconstruction_metrics.csv` 中的主要字段如下：

| 字段 | 定义 |
|---|---|
| `mesh_vertices` | 重建网格顶点数 |
| `mesh_triangles` | 重建网格三角面数 |
| `significant_components` | 三角面数量不低于网格总三角面 1% 的连通分量数 |
| `input_to_mesh_mean` | 当前输入点云到自身重建网格的平均距离 |
| `input_to_mesh_rmse` | 当前输入点云到自身重建网格距离的均方根 |
| `reference_to_mesh_mean` | 参考融合点云到重建网格的平均距离 |
| `mesh_to_reference_mean` | 20000 个网格表面采样点到参考融合点云的平均最近邻距离 |
| `symmetric_chamfer` | 上述两个方向平均距离的算术平均，不是平方 Chamfer |
| `surface_precision_5mm` | 距参考融合点云不超过 5 mm 的网格表面采样点比例 |
| `reference_completeness_5mm` | 距重建网格不超过 5 mm 的参考点比例 |
| `f1_5mm` | 表面精确率与完整率的调和平均 |

仅使用 `input_to_mesh_rmse` 可能得出错误结论：未配准点云也可能被 Poisson 网格很好地局部拟合，但网格会在错位扫描之间生成额外伪表面。因此，至少应联合报告：

- 双向 Chamfer。
- 阈值精确率、完整率和 F1。
- 显著连通分量。
- 网格外观及空间误差图。

## 9. 评价边界与实验诚信

当前 `reconstruct_palette.py` 把 ICP 后融合点云作为两种网格共同的参考点云。这种设计可以评价未配准重建相对于最终融合结构产生了多少额外表面，但存在以下限制：

- 参考点云不是独立的高精度真值网格。
- 指标反映内部一致性，不能解释为绝对表面精度。
- 网格到点云方向通过 20000 个表面点近似，存在采样误差。
- 当前只使用一对扫描，不能给出跨场景泛化结论。
- 原报告的纯几何基线没有使用 RGB；新增联合 ICP 只使用三个颜色通道，仍未评价完整高光谱纹理的保持程度。

正式论文实验建议增加：

- 多个扫描距离、入射角、重叠率和噪声水平。
- 每组条件多个随机种子，并报告均值、标准差和置信区间。
- 独立结构光、标定板或高精度网格作为真值。
- 旋转误差、平移误差、法向一致性、孔洞率和表面粗糙度。
- 完整高光谱波段下的光谱角、波段 RMSE 或纹理保持指标。

## 10. 更换新数据时的操作清单

1. 将新点云转换为英文逗号分隔的数值文件。
2. 统一坐标单位，并记录传感器坐标系方向。
3. 使用 `visualize_pointclouds.py` 或等价脚本检查范围、颜色和标签。
4. 先使用宽松参数试运行 ICP，并观察配准图。
5. 扫描 `distance_quantile`、最大迭代次数和停止阈值。
6. 将最终对齐结果保存为新的文件名，避免覆盖旧实验。
7. 根据新数据的中位点间距调整法向半径和重建阈值。
8. 对 Poisson `depth` 做敏感性实验，检查孔洞、伪表面和计算量。
9. 使用独立真值时，修改 `reconstruct_palette.py` 中的 `reference`。
10. 保存命令、环境版本、随机种子、原始指标 CSV 和未裁剪图片。

`reconstruct_palette.py` 当前固定读取以下文件：

```python
up_t.txt
down.txt
up_t_aligned_refined.txt
```

若更换数据，可修改 `main()` 中的三个文件名；重建参数集中定义在脚本开头。为了批量实验，后续建议把这些常量改成命令行参数，并为每个实验建立独立输出目录。

## 11. 两篇报告的修改与同步指南

本实验结果分别用于以下两篇中期报告：

| 报告 | 当前工作区文件 | 6.4 节核心问题 |
|---|---|---|
| 多光谱信息与几何信息融合的点云配准研究 | `多光谱信息与几何信息融合的点云配准研究_中期报告_加入6.4实验.docx` | ICP 能否恢复两组扫描之间的刚体位姿并降低点云—点云误差 |
| 多光谱激光雷达点云三维重建研究 | `多光谱激光雷达点云三维重建_中期报告_6.4三维重建修订.docx` | ICP 后的融合能否减少 Poisson 网格伪影并提高三维重建质量 |

两篇报告使用同一组输入点云和同一次 ICP 结果，但研究问题、评价对象、指标和结论不能互相替代。

### 11.1 修改前的通用原则

1. 先完成实验并确认 CSV 与图片，再修改 Word，不能先在报告中填写预期数值。
2. 把 CSV 文件作为数值的唯一来源，避免从图片刻度或旧报告中手工估算。
3. 每次修改另存新版本，不覆盖原始报告。建议文件名包含日期或实验编号。
4. 表格中的数值、正文中的数值、图中标注和图题必须保持一致。
5. 所有距离必须标明单位。当前坐标单位按米解释，报告中可统一换算为毫米。
6. 修改实验参数后，旧图、旧表和旧结论必须同时更新，不能只替换某一项。
7. Word 中保持原有标题层级、中文字体、段落缩进、表格样式、图题格式和章节编号。
8. 插图时锁定纵横比，避免为了填满页面而拉伸点云或网格图。
9. 新报告生成后，应检查正文提取、图片清晰度、分页、图表编号和 DOCX 压缩完整性。

建议为每次正式实验建立独立目录：

```text
experiments/
└── 2026-07-14_palette_icp_poisson/
    ├── inputs/
    ├── registration/
    ├── reconstruction/
    ├── figures/
    ├── metrics/
    ├── commands.txt
    └── environment.txt
```

当前脚本会覆盖部分同名输出，因此批量实验前应改用实验编号文件名，或把每次运行放在独立副本目录中。

### 11.2 配准研究报告的修改指南

目标报告：

```text
多光谱信息与几何信息融合的点云配准研究_中期报告_加入6.4实验.docx
```

建议第 6.4 节标题保持为：

```text
6.4 高光谱激光雷达扫描调色板实验
```

该节的关键词应是“刚体配准、点到点 ICP、对应关系、收敛、参数敏感性”，三维重建只能作为后续用途简要提及，不能成为该报告的主要结论。

#### 数据来源

- 移动点云：`up_t.txt`，1523 点。
- 固定点云：`down.txt`，1493 点。
- 人工位姿扰动：绕 Z 轴 20°，平移 `(0.10, -0.05, 0.03)`。
- RGB 扰动：`N(0, 8²)`，随机种子 42，裁剪到 `[0,255]`。
- 精细 ICP 参数：最大 500 次迭代、收敛阈值 `1e-12`、对应点保留比例 1.0。

#### 表格数值来源

配准表格只能使用：

```text
report_metrics.csv
```

当前实验的参考值如下，数值单位为米：

| 配准指标 | 配准前 | 配准后 | 降幅 |
|---|---:|---:|---:|
| 平均最近邻距离 | 0.384196 | 0.002530 | 99.34% |
| 最近邻距离中位数 | 0.384185 | 0.002209 | 99.42% |
| 最近邻 RMSE | 0.384670 | 0.003102 | 99.19% |
| 反向平均最近邻距离 | 0.384948 | 0.004781 | 98.76% |
| 点云—点云双向 Chamfer | 0.384572 | 0.003656 | 99.05% |

若报告统一使用毫米，应将所有数值同时乘以 1000，并在表头或指标名称中明确写出 `/mm`。

#### 应使用的图片

| 顺序 | 文件 | 建议图题 |
|---|---|---|
| 图 1 | `report_figure1_registration.png` | ICP 配准前后的三维点云及 X-Z 投影对比 |
| 图 2 | `report_figure2_metrics.png` | ICP 配准前后距离误差指标对比 |
| 图 3 | `report_figure3_convergence.png` | ICP 平均对应点距离的迭代收敛曲线 |
| 图 4 | `report_figure4_parameter_sensitivity.png` | 对应点保留比例对配准误差的影响 |

正式插入 Word 时优先使用 PNG；PDF 版本适合后续 LaTeX 或矢量排版。图片更新后必须同步检查图题中的“配准前/配准后”“对数尺度”和对应点比例描述。

#### 推荐正文结构

1. **实验目的**：验证人工位姿偏差下的几何对齐能力。
2. **数据和扰动**：说明点数、旋转、平移、RGB 噪声和随机种子。
3. **ICP 方法**：说明 KD-tree 最近邻、SVD 刚体求解、累积变换和停止条件。
4. **定量结果**：引用 `report_metrics.csv`，同时报告单向和双向指标。
5. **收敛与参数分析**：引用 `report_convergence.csv` 和 `report_parameter_sweep.csv`。
6. **结论与限制**：旧版结果应说明 RGB 未参与对应搜索，属于几何配准基线；加入本次联合实验后，应把纯几何与几何+RGB 分列报告。

#### 配准报告中不能出现的过度结论

- 不应写“验证了完整高光谱特征能够提高配准精度”。当前代码没有使用完整光谱波段。
- 对旧版纯几何实验，不应写“证明算法对 RGB 噪声具有鲁棒性”，因为 RGB 没有参与距离计算。
- 对新增联合实验，也只能写“在当前 RGB 高斯扰动和单对点云下改善了所选对应点的颜色一致性”；不能推广到完整高光谱鲁棒性。
- 加入 RGB 实验时，应在方法部分写出第 7 节的联合对应方程与加权 SVD 方程，并引用 `icp_rgb_metrics.csv` 和 `icp_rgb_comparison.png`。
- 不应把配准后的最近邻误差称为三维重建误差。
- 不应根据单次扫描对泛化能力、稳定性或统计显著性下结论。
- 不应把人工施加的扰动参数直接当成 ICP 求得变换的误差；若要报告旋转和平移误差，应显式计算估计变换与真值逆变换之间的差异。

### 11.3 三维重建研究报告的修改指南

目标报告：

```text
多光谱激光雷达点云三维重建_中期报告_6.4三维重建修订.docx
```

建议第 6.4 节标题保持为：

```text
6.4 高光谱激光雷达扫描调色板三维重建实验
```

该节的关键词应是“三维重建、点云融合、Poisson 曲面、网格质量、表面误差和拓扑连通性”。ICP 在本报告中是重建前端，不应占据主要篇幅。

#### 数据与重建条件

- 未配准重建输入：`up_t.txt + down.txt`。
- ICP 后重建输入：`up_t_aligned_refined.txt + down.txt`。
- 两种输入均为 3016 点，并采用完全相同的法向估计、Poisson 和裁剪参数。
- 当前共同参考：ICP 后融合点云。
- 网格到点云方向：均匀采样 20000 个表面点近似计算。
- 5 mm 阈值用于表面精确率、完整率和 F1。

#### 表格数值来源

三维重建表格只能使用：

```text
reconstruction_metrics.csv
```

建议至少报告以下指标：

| 重建指标 | 未配准融合重建 | ICP 后融合重建 | 变化 |
|---|---:|---:|---:|
| 网格顶点数 | 7577 | 6688 | 减少 11.73% |
| 网格三角面数 | 14929 | 12990 | 减少 12.99% |
| 显著连通分量 | 2 | 1 | 由 2 减至 1 |
| 网格—参考点云对称 Chamfer | 84.985 mm | 1.979 mm | 下降 97.67% |
| 5 mm 表面精确率 | 11.54% | 88.74% | 提高 77.21 个百分点 |
| 5 mm 完整率 | 91.84% | 91.41% | 基本不变 |
| 5 mm F1 | 20.50% | 90.06% | 提高 69.56 个百分点 |

输入点云到自身重建网格的 RMSE 从约 2.985 mm 变为 3.020 mm，基本没有改善。报告中应保留这一事实，并解释单向输入拟合误差不能识别额外伪表面，因此需要双向 Chamfer、F1 和连通分量共同评价。

#### 应使用的图片

| 顺序 | 文件 | 建议图题 |
|---|---|---|
| 图 1 | `reconstruction_figure1_mesh_comparison.png` | 未配准融合与 ICP 后融合的 Poisson 三维重建表面对比 |
| 图 2 | `reconstruction_figure2_metrics.png` | 三维重建误差、5 mm 阈值指标与拓扑连通性对比 |
| 图 3 | `reconstruction_figure3_error_distribution.png` | 重建表面到最终融合点云距离的累计分布 |
| 图 4 | `reconstruction_figure4_error_map.png` | ICP 后融合重建表面的空间误差分布 |

#### 推荐正文结构

1. **重建目的**：比较配准前端对最终网格质量的影响。
2. **对照设计**：说明未配准融合和 ICP 后融合仅位姿条件不同，其余重建参数一致。
3. **重建方法**：说明法向估计、法向一致化、Poisson 深度、尺度和包围盒裁剪。
4. **评价方法**：定义参考点云、双向 Chamfer、5 mm 精确率/完整率/F1 和显著连通分量。
5. **结果分析**：结合外观、定量指标、误差 CDF 和空间热图解释伪表面与边界误差。
6. **局限性**：说明没有独立真值网格、只有一对扫描、RGB 只是光谱代理。

#### 三维重建报告中不能出现的过度结论

- 不应把 ICP 后融合点云称为独立真值或绝对真值。
- 不应把内部一致性指标解释为绝对重建精度。
- 不应把表面采样近似的 Chamfer 写成严格解析距离而不说明采样数。
- 不应声称已经评价完整高光谱纹理或光谱保持质量。
- 不应只展示 ICP 收敛曲线而缺少实际网格、误差图和拓扑指标。

### 11.4 两种 Chamfer 指标不得混用

两篇报告都出现 Chamfer，但评价对象不同：

| 所属报告 | 计算对象 | 当前数值 | 数据来源 |
|---|---|---:|---|
| 配准研究报告 | 移动点云与固定点云的双向最近邻平均 | 配准后约 0.003656 m | `report_metrics.csv` |
| 三维重建报告 | 网格表面与 ICP 后融合参考点云的双向平均 | ICP 后重建约 1.979 mm | `reconstruction_metrics.csv` |

两者不能直接比较，也不能在正文中使用相同的“Chamfer 下降”描述而省略评价对象。修改报告时必须同时写明：

- 比较的是点云—点云还是网格—点云。
- 单位是米还是毫米。
- 是否使用表面采样近似。
- 参考点云或真值网格是什么。

### 11.5 实验更新后同步两篇报告的标准流程

1. 保存本次实验使用的 `up_t.txt`、`down.txt` 和随机种子。
2. 运行第 5.2 节命令，生成新的对齐点云和变换矩阵。
3. 运行 `generate_report_figures.py`，检查配准 CSV 与 4 张配准图。
4. 运行 `reconstruct_palette.py`，检查重建 CSV、两份 PLY 和 4 张重建图。
5. 比较新旧指标，确认变化来自预期参数或数据，而不是单位、文件名或参考对象错误。
6. 先更新配准研究报告的表格、图片、结果段和局限性。
7. 再更新三维重建报告的表格、图片、结果段和局限性。
8. 检查两篇报告中的点数、扰动、ICP 参数和实际迭代次数是否一致。
9. 检查两篇报告的 Chamfer 定义和单位是否各自正确。
10. 另存新 DOCX，打开后检查图片比例、分页、图表编号和目录更新情况。

### 11.6 Word 修改后的检查清单

#### 内容一致性

- [ ] 报告标题与文件名对应研究主题。
- [ ] 6.4 节标题正确，未把配准实验和重建实验混为一节。
- [ ] 输入文件、点数、扰动和随机种子与代码一致。
- [ ] ICP 参数与实际命令一致。
- [ ] Poisson 参数与 `reconstruct_palette.py` 一致。
- [ ] 表格数值可在对应 CSV 中逐项找到。
- [ ] 正文中的百分比由同一组表格数值计算。
- [ ] Chamfer 的评价对象和单位明确。
- [ ] 图题与实际图片内容一致。
- [ ] 局限性没有被结果段中的强结论抵消。

#### 版式与文件完整性

- [ ] 图片纵横比正确，文字和色条清晰。
- [ ] 表格没有跨页断裂或超出页边距。
- [ ] 图号、表号和正文交叉引用连续。
- [ ] 标题样式与前后章节一致。
- [ ] 目录、页码和章节编号已更新。
- [ ] 原始报告仍保留，新文件采用新版本名。
- [ ] DOCX 能正常打开，嵌入图片没有丢失。

建议的版本命名方式：

```text
多光谱信息与几何信息融合的点云配准研究_中期报告_6.4实验编号_日期.docx
多光谱激光雷达点云三维重建_中期报告_6.4实验编号_日期.docx
```

## 12. 常见问题

### ICP 收敛但点云仍然错位

可能原因包括初值过差、几何对称、重叠率过低或最近邻落入错误区域。先检查配准前后图，再加入粗配准，不要单纯增加迭代次数。

### 配准指标很好，但重建网格存在大面积伪表面

单向最近邻指标可能忽略多余结构。检查反向距离、对称 Chamfer、表面精确率、F1 和连通分量，并观察误差热图。

### Poisson 网格过于平滑

可适当提高 `POISSON_DEPTH`，但同时会增加内存、计算量和噪声敏感性。还应检查法向估计半径是否过大。

### Poisson 网格破碎或法向混乱

检查点密度、法向半径、邻点数和法向一致化邻域。对于多视点数据，还应确保各扫描的法向朝向一致。

### 5 mm F1 几乎全为 0 或全为 1

阈值与数据单位或点间距不匹配。先计算中位最近邻间距，再选择具有物理意义的阈值。

### Matplotlib 报字体或缓存权限警告

按第 4 节把 `MPLCONFIGDIR` 和 `XDG_CACHE_HOME` 指向项目内的可写缓存目录。中文字体缺失时可安装中文字体，或修改脚本中的字体列表。

## 13. 建议的实验记录格式

每次实验建议至少记录：

```text
实验编号：
日期：
Git 提交或代码版本：
Conda 环境与依赖版本：
移动点云 / 固定点云：
坐标单位：
点数和中位点间距：
扰动旋转 / 平移：
噪声模型与随机种子：
ICP 参数：
Poisson 参数：
参考真值或参考点云：
配准指标：
重建指标：
输出目录：
异常现象与解释：
```

保持原始点云、配置、变换矩阵、CSV 指标和最终图片一一对应，是后续撰写论文和复核实验结论的基础。
