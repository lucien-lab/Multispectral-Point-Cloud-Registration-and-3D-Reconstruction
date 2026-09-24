# 左右多模态点云配准实验

本目录比较三种将扰动后的左侧点云恢复到右侧点云的方法：

1. Open3D point-to-plane 几何 ICP；
2. Open3D RGB Colored ICP；
3. 参考 Park、Zhou、Koltun（ICCV 2017）的几何 + 真实六波段联合 ICP。

同一个 trial 继续使用完全相同的扰动 source 和 target，但三种方法分别运行自己的粗配准，再把各自的 `T_coarse` 交给对应细配准：

1. geometry：33维标准化FPFH，仅用XYZ几何和重叠率评分；
2. RGB：33维FPFH + 3维真实RGB，用几何、RGB和重叠率联合评分；
3. spectral6：33维FPFH + 6维真实光谱，用几何、六波段和重叠率联合评分。

因此，geometry完整流程不读取RGB或光谱，RGB完整流程不读取六波段；六波段流程沿用原有FPFH+六波段联合粗配准思路。

实验使用 `left.bin` 和 `right.bin` 对应的 6.3–10 m 有效点云。坐标统一为米，六波段顺序固定为：

```text
[495, 696, 600, 803, 545, 642] nm
```

## 数据目录

- `data/raw/`：指向仓库内原始 bin 和角度文件的相对符号链接，代码只读访问，不修改原文件。
- `data/processed/*_refined_input.npz`：本实验采用的精炼输入快照。
- `data/processed/left_points.npz`、`right_points.npz`：运行时生成的统一米制数据。

每个统一快照包含：

```text
xyz_m           N×3 空间坐标
rgb             N×3 真实光谱转换的RGB
spectral6       N×6 实测六波段反射率
wavelengths_nm  6个波长
feature12       [x,y,z,r,g,b,s1,s2,s3,s4,s5,s6]
```

## 运行环境

项目默认使用已有 Conda 环境 `ms_pointcloud_midterm`：

```bash
conda run -n ms_pointcloud_midterm python -c \
  "import open3d, numpy, scipy; print(open3d.__version__, numpy.__version__, scipy.__version__)"
```

若需要重建环境，可以参考 `environment.yml`，但无需为已有环境重复安装。

## 先调优三种粗配准

正式比较前先运行：

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
conda run -n ms_pointcloud_midterm \
python '2026_07_26实验/tune_coarse_registration.py' \
  --config '2026_07_26实验/configs/default.json' \
  --method all
```

该命令分别搜索geometry 18组、RGB 27组、spectral6 27组配置，每组只使用trial 1–5。选择顺序固定为：

1. 成功数量最多；
2. `mean(rotation_error/2 + translation_error/0.10)`最小；
3. 平移误差中位数最小；
4. 平均运行时间最短；
5. `config_id`字典序最小。

选出的完整配置保存到：

```text
results/coarse_tuning/{geometry|rgb|spectral6}/selected_config.json
```

主实验会自动读取这些文件。trial 6–10不会参与选参，只用于独立评价。

## 完整运行

在仓库根目录执行：

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
conda run -n ms_pointcloud_midterm \
python '2026_07_26实验/run_experiment.py' \
  --config '2026_07_26实验/configs/default.json'
```

其他入口：

```bash
# 仅准备数据
OMP_NUM_THREADS=1 conda run -n ms_pointcloud_midterm \
python '2026_07_26实验/run_experiment.py' --prepare-only

# 仅运行RGB方法
OMP_NUM_THREADS=1 conda run -n ms_pointcloud_midterm \
python '2026_07_26实验/run_experiment.py' --method rgb

# 仅运行第3组扰动的六波段方法
OMP_NUM_THREADS=1 conda run -n ms_pointcloud_midterm \
python '2026_07_26实验/run_experiment.py' --method spectral6 --trial 3
```

每次不带 `--trial` 的运行会按随机种子42重新产生完全相同的10组扰动。
`OMP_NUM_THREADS=1` 用于避免 Open3D 并行 RANSAC 因线程调度产生跨进程差异；
需要可复现实验结果时请保留该设置。单独运行某个方法会重写总汇总表，使其只包含该次执行的方法；正式比较请使用 `--method all`。

## 导出三方法共享最佳 Trial 对比报告

当 geometry、RGB 和鲁棒六通道方法的 trial 6–10 原始结果均已存在时，可在仓库根目录执行：

```bash
conda run -n ms_pointcloud_midterm python \
  2026_07_26实验/build_three_method_best_trial_report.py
```

该命令从三个方法共同的留出集选择**同一个**最佳 trial：先比较三种方法平均的
`rotation_error_deg / 2 + translation_error_m / 0.10`，并依次用平均 100 mm
重叠率、平均 250 mm 重叠率和更小的 trial 编号打破平局。它将可独立移动的结果写入：

```text
results/three_method_best_trial_comparison/
```

该目录的 CSV、JSON 和 Markdown 只保留位姿误差、重叠率、距离统计、fitness、RMSE
及 `pose_score`；有意不导出 `success` 或 `success_rate` 字段。三种方法使用相同的
`T_perturb.json` 与 `T_gt.json`，每个方法目录均含以下 PLY（为实际文件副本，不是符号链接）：

- `initial_merged.ply`：扰动后的 source 为绿色、target 为红色；
- `registered_merged.ply`：配准后的 source 为绿色、target 为红色，便于诊断几何重叠；
- `registered_true_rgb.ply`：配准后的 source 与 target 都保留各自真实 RGB，便于观察真实颜色。

## 不重跑 ICP，补齐联合评价与初始 PLY

已有 `T_perturb.json`、`T_gt.json` 和 `T_est.json` 时，可以直接运行：

```bash
conda run -n ms_pointcloud_midterm \
python '2026_07_26实验/backfill_evaluation_outputs.py' \
  --config '2026_07_26实验/configs/default.json'
```

该命令不会进入粗配准或细配准，只会：

1. 根据已保存变换重新计算 rotation error、translation error；
2. 计算 100 mm 和 250 mm 双向最近邻重叠率；
3. 为每个方法、每个 trial 生成配准前左右点云合并 PLY；
4. 重建逐次结果、方法汇总和三种算法参数汇总。

命令可重复运行，不会修改 `T_perturb.json`、`T_gt.json`、`T_est.json`、
配准后点云或收敛记录。

## 实验设置

- geometry粗配准：33维标准化FPFH
- RGB粗配准：33维FPFH + 3维median/MAD标准化真实RGB
- 六波段粗配准：33维FPFH + 6维median/MAD标准化真实光谱
- 平衡搜索体素/对应距离：`(0.10,0.15)`、`(0.12,0.18)`、`(0.16,0.24) m`
- RANSAC：8个固定种子候选，`ransac_n=3`、每个最多 `30000` 次
- geometry搜索法向半径倍数：`2.0/3.0`，重叠权重：`0.10/0.20/0.30`
- RGB搜索特征权重：`0.5/1.0/2.0`，RGB数据占比：`0.20/0.40/0.60`
- 六波段搜索特征权重：`1.0/2.0/3.0`，光谱数据占比：`0.30/0.50/0.70`
- 安全约束：旋转不超过 `25°`，平移范数不超过 `1.25 m`
- 每轴旋转扰动：`[-10°, 10°]`
- 每轴平移扰动：`[-0.50 m, 0.50 m]`
- 体素金字塔：`0.20 / 0.10 / 0.05 m`
- 对应阈值：`0.80 / 0.40 / 0.20 m`
- 迭代上限：`80 / 50 / 30`
- 统一评价阈值：`0.20 m`
- 成功标准：旋转误差不超过 `2°` 且平移误差不超过 `0.10 m`

六波段方法用目标点局部切平面拟合每个波段的梯度。6个光谱残差和point-to-plane几何残差共同进入SE(3) Gauss–Newton正规方程。光谱按目标点云的median/MAD标准化。当前默认值是与RGB Colored ICP相同权重的消融配置：

```text
lambda_geo  = 0.968
lambda_spec = 0.032
```

原 `0.5/0.5` 六波段结果会由
`run_spectral_weight_ablation.py` 不可覆盖地保存到
`results/weight_ablation/spectral_050_050_baseline/`。这次消融只改变
六波段细配准权重；六波段粗配准的特征与候选评分权重保持原值。

三种粗配准都不依赖初始位姿，但在低重叠、重复平面或属性不稳定场景中仍可能找不到可靠候选。程序把单位矩阵作为基线；只有候选联合评分至少改善 `0.03` 时才接受，否则显式记录 `fallback_identity=true` 并继续细配准，避免灾难性初始化。

如需与原始“单位矩阵初始化”流程做消融对照，可复制配置文件并修改：

```json
"coarse_registration": {
  "enabled": false
}
```

禁用时程序将明确记录 `enabled=false`，并把单位矩阵保存为 `T_coarse`。

## 输出

```text
results/
├── coarse_registration/
│   ├── geometry/trial_001...trial_010/
│   ├── rgb/trial_001...trial_010/
│   └── spectral6/trial_001...trial_010/
├── coarse_tuning/
│   ├── geometry/
│   ├── rgb/
│   └── spectral6/
├── icp_geometry/trial_001...trial_010/
├── icp_rgb/trial_001...trial_010/
├── icp_spectral6/trial_001...trial_010/
├── weight_ablation/
│   ├── spectral_050_050_baseline/
│   └── spectral_0968_0032/
├── baseline_shared_coarse_results_summary.csv
├── results_summary.csv
├── method_summary.csv
├── method_summary.md
├── coarse_tuning_summary.csv
├── coarse_tuning_summary.md
├── heldout_method_summary.csv
├── heldout_method_summary.md
├── all_trials_method_summary.csv
├── all_trials_method_summary.md
├── method_parameters_summary.csv
├── method_parameters_summary.md
├── trial_001_method_comparison.png
├── registration_error_boxplots.png
└── coarse_and_fine_error_boxplots.png/.svg
```

每个粗配准 trial 保存：

- `T_coarse.json`：该方法最终选定的专属粗配准初值；
- `coarse_metrics.json`：粗配准旋转误差、平移误差、fitness、RMSE、运行时间及参数；
- `coarse_candidates.csv`、`coarse_candidates.json`：identity基线和全部RANSAC候选的变换、联合分数、接受状态及拒绝原因；
- `source_coarse_aligned.ply`：粗配准后的左点云；
- `source_target_coarse_merged.ply`：粗配准后的 source 绿色、target 红色，用于判断几何重叠；
- `source_target_coarse_rgb_merged.ply`：粗配准后的 source 与 target 均保留各自真实 RGB。

每个 trial 保存：

- `T_perturb.json`：施加到左点云的随机变换；
- `T_gt.json`：理论恢复真值，即 `inverse(T_perturb)`；
- `T_est.json`：算法估计变换；
- `metrics.json`：旋转误差、平移误差、100/250 mm双向重叠率、fitness、RMSE和成功标记；
- `convergence.csv`：逐层或逐迭代记录；
- `source_perturbed.ply`、`source_registered.ply`；
- `source_target_initial_merged.ply`：配准前的扰动source为绿色、target为红色，两个点云位于同一个PLY；
- `source_target_merged.ply`：source绿色、target红色，用于判断几何重叠；
- `source_target_rgb_merged.ply`：source与target均保留各自真实RGB，用于观察配准后的实际颜色；
- 第1组还包含 `before_after.png`，依次显示粗配准前、粗配准后和细配准后三个阶段。

`results_summary.csv` 的每个方法行同时包含：

```text
coarse_method
coarse_config_id
coarse_rotation_error_deg
coarse_translation_error_m
coarse_fitness
coarse_inlier_rmse_m
coarse_runtime_s
```

这些字段来自对应方法自己的粗配准，同一trial的三种方法通常不同。

联合评价中的：

- `rotation_error_deg` 和 `translation_error_m` 衡量估计变换相对真实
  `T_gt` 的位姿误差；
- `overlap_100mm_pct` 和 `overlap_250mm_pct` 是完整点云两个方向最近邻
  覆盖率的平均值，单位为百分比。

重叠率高只说明两个点集更靠近，不能单独证明恢复了正确物理位姿，因此应与
rotation error 和 translation error 联合判断。

`method_parameters_summary.*` 每种方法一行，集中记录专属粗配准配置、
三级体素/对应距离/迭代次数、RGB或六波段权重、六波长顺序和统一评价阈值。

`heldout_method_summary.*`只汇总trial 6–10；`all_trials_method_summary.*`汇总全部10组。`baseline_shared_coarse_results_summary.csv`保留旧分支共享六波段粗配准结果，便于消融对照。

完整环境、数据校验值、参数和结果记录在 `experiment_log.md`，运行过程记录在 `logs/`。

只重跑六波段权重消融并自动生成对比：

```bash
OMP_NUM_THREADS=1 conda run -n ms_pointcloud_midterm python \
  2026_07_26实验/run_spectral_weight_ablation.py \
  --config 2026_07_26实验/configs/default.json
```

验证现有消融结果而不重新计算：

```bash
conda run -n ms_pointcloud_midterm python \
  2026_07_26实验/run_spectral_weight_ablation.py \
  --config 2026_07_26实验/configs/default.json \
  --verify-only
```

## 逐层验证与回退

三种细配准方法都使用固定的 `0.10 m` 验证点云检查每个金字塔层级。
验证对应距离固定为 `0.30 m`，仅保留误差最小的70%对应点，并在分数中加入重叠率约束：

- 几何ICP使用几何距离和重叠率；
- RGB ICP使用几何距离、真实RGB差异和重叠率；
- 六波段ICP使用几何距离、MAD标准化六波段差异和重叠率。

候选层级只有在验证分数至少改善0.5%，并且覆盖率下降不超过2个百分点时才会接受；
否则恢复该层开始前的变换。该判断不使用 `T_gt`，真实变换只用于实验结束后的旋转和平移误差评价。

每个 `convergence.csv` 会记录：

```text
validation_score_before
validation_score_after
validation_coverage_before
validation_coverage_after
validation_relative_improvement
accepted
rollback_reason
```

六波段方法的 `record_type=iteration` 表示原有Gauss–Newton迭代，
`record_type=level_validation` 表示金字塔层级接受或回退结果。

## 交互查看PLY

推荐直接启动新的注册点云查看器。未传入文件路径时，会弹出系统GUI文件选择框，默认打开本实验的 `results/`：

```bash
MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
conda run -n ms_pointcloud_midterm \
python '2026_07_26实验/visualize_registration.py'
```

也可以跳过文件选择框，直接打开指定文件：

```bash
MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
conda run -n ms_pointcloud_midterm \
python '2026_07_26实验/visualize_registration.py' \
  --file '2026_07_26实验/results/icp_rgb/trial_002/source_target_rgb_merged.ply'
```

如果要区分左右点云、检查几何重叠，请把文件名改回
`source_target_merged.ply`，其中左侧 source 为绿色、右侧 target 为红色。

支持的点云格式为 PLY、PCD、XYZ、XYZN、XYZRGB 和 PTS。常用参数：

```text
--point-size 5       初始点大小
--background white  白色背景
--no-axis            隐藏坐标轴
--reset-view         不恢复该文件上次保存的视角
--info               只输出点数和坐标范围，不打开窗口
```

窗口快捷键：

```text
0  文件保存的原始RGB（真实RGB文件显示测量颜色，诊断文件显示左绿右红）
1  按高度着色
2  按到坐标原点的距离着色
3/4/5  分别按X/Y/Z着色
+/-  增大或减小点大小
B    切换黑色/白色背景
R    重置视角
S    保存截图到当前PLY所在目录
Q/Esc 退出
```

鼠标左键旋转、中键平移、滚轮缩放。关闭窗口时会为每个点云单独保存视角，例如 `.source_target_merged.ply.view.json`，下次打开同一文件时自动恢复。坐标轴位于真实坐标原点 `(0,0,0)`，与距离着色使用的原点一致。

## 测试

```bash
MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
conda run -n ms_pointcloud_midterm \
python -m unittest discover \
  -s '2026_07_26实验/tests' -p 'test_*.py' -v
```

测试覆盖数据转换、SE(3)、统一评价、33/36/39维描述子、模态隔离、多候选安全选择与可复现性、18/27/27组搜索空间、trial 1–5与6–10隔离、单方法失败隔离、三种ICP合成恢复、六波段切平面梯度和端到端文件输出。

## 方法说明

参考论文：

> Jaesik Park, Qian-Yi Zhou, Vladlen Koltun. Colored Point Cloud Registration Revisited. ICCV 2017.

Open3D的Colored ICP只支持RGB三通道，因此六波段方法不能直接调用 `registration_colored_icp`。本实验保留论文的切平面连续属性思想，将单通道光度残差扩展为6个实测光谱残差，并与point-to-plane残差联合优化。

该实验不会假定六波段方法一定优于其他方法。中等扰动、部分重叠和真实光谱噪声都可能导致某些trial失败，程序会保留失败状态而不会更换随机种子。

## 鲁棒六通道独立实验

推荐版六通道实验不会覆盖现有三算法结果，全部写入：

```text
results/spectral6_recommended_experiment/
```

完整自动协议：

```bash
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp \
conda run -n ms_pointcloud_midterm python \
  '2026_07_26实验/run_spectral_recommended_experiment.py' \
  --config '2026_07_26实验/configs/spectral6_recommended.json' \
  --stage all
```

第一轮使用 trial 1–5 拟合左右逐波段校正并选参，trial 6–10
与同期重新运行的 RGB 结果比较。只有联合门槛未通过时，程序才会自动：

1. 用 trial 1–10 重新拟合校正和二次选参；
2. 冻结参数；
3. 在全新扰动 trial 11–20 上同时运行六通道和 RGB 最终确认；
4. 如实报告是否超过 RGB，不继续循环调参。

也可以分阶段运行：

```bash
# 第一轮只拟合相对光谱校正
conda run -n ms_pointcloud_midterm python \
  '2026_07_26实验/run_spectral_recommended_experiment.py' \
  --stage calibrate

# 第一轮校正并调参，不运行held-out评价
conda run -n ms_pointcloud_midterm python \
  '2026_07_26实验/run_spectral_recommended_experiment.py' \
  --stage tune

# 第二轮校正并调参
conda run -n ms_pointcloud_midterm python \
  '2026_07_26实验/run_spectral_recommended_experiment.py' \
  --stage retune

# 使用已经冻结的第二轮参数评价trial 11–20
conda run -n ms_pointcloud_midterm python \
  '2026_07_26实验/run_spectral_recommended_experiment.py' \
  --stage confirm --variant robust_full
```

`--variant` 支持 `baseline`、`calibrated`、`aligned_validation`、
`robust_full` 和 `all`；`--trial N` 可只运行一组扰动。完整入口会保存：

- 四组消融结果、RGB同期对照和联合门槛判定；
- 校正参数、对应点CSV、拟合前后散点图和通道权重；
- 每个trial的变换、指标、收敛记录、诊断记录、PLY和示意图；
- 旋转/平移/重叠率/Chamfer箱线图；
- 输入SHA-256、20组SE(3)、软件环境和冻结参数位置；
- `logs/` 下带时间戳的完整命令行输出与错误日志。
