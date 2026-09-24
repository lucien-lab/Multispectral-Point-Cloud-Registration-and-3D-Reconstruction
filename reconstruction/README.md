# reconstruction — 三维重建

配准结果的下游重建实验：活动室场景配准前后重建对比、调色板点云 Poisson 曲面重建、公开数据集配准重建基线。

## 目录

| 路径 | 说明 |
| --- | --- |
| `activity_room_experiment.py` | 活动室场景：配准（配准前 / 配准后）→ 网格重建 → 误差指标与对比图 |
| `public_dataset_baseline/run_ms_registration_reconstruction.py` | 公开数据集上的多光谱配准 + 重建基线流程 |
| `public_dataset_baseline/download_datasets.py` | 公开数据集（Open3D Demo ICP、Stanford Bunny）下载/缓存 |
| `public_dataset_baseline/view_ply.py` | 重建结果查看 |
| `icp_learning/global_icp.py` | 全局 ICP（特征匹配 + RANSAC）实现 |
| `icp_learning/demo_vis.py` | 配准过程可视化演示 |
| `icp_learning/test_ds.py` | 数据集下载 / 读取自检 |

调色板的 ICP 配准 + Poisson 重建脚本见 `../baselines/icp_registration/reconstruct_palette.py`。

## 运行

```bash
python reconstruction/activity_room_experiment.py --help
python reconstruction/public_dataset_baseline/run_ms_registration_reconstruction.py --help
python reconstruction/icp_learning/global_icp.py --help
```

> 输入点云（`*.ply` / `*.txt` / `*.pcd`）与输出网格、指标 CSV、图像均不入库。
