# 发财树 0~300° 六视角配准实验（experiment_cripped）

日期：2026-09-05（重新实验，删除旧 registered/、registered_v2/、experiment_20260201 流程产物后重建）

## 原则

- **只利用点间相对几何**：FPFH 特征匹配 → RANSAC 估计刚体变换 → ICP 精化。
- 不使用各帧绝对坐标作为对齐依据（各帧为独立扫描仪坐标系，绝对位置相似是裁剪/归一化造成的假象，不代表朝向对齐）。
- 不使用名义角度（0/60/120/180/240/300°）作先验——实测帧间真实旋转角与命名不完全一致，特征配准可自动求出。

## 方法（每帧 → 0° 基准帧）

1. 体素下采样 3mm + 法线估计
2. FPFH 特征计算
3. RANSAC 全局配准（12 组参数组合取最优 fitness）
4. point-to-plane ICP 精化（4 组阈值取最优）

## 配准结果（RANSAC 解 + ICP 精化）

| 帧 | RANSAC 解 | ICP 后 fitness | rmse |
|---|---|---|---|
| 60° | rot 27° / shift 585mm | 0.982 | 6.9mm |
| 120° | rot 125° / shift 2217mm | 0.991 | 6.8mm |
| 180° | rot 179° / shift 2488mm | 0.984 | 7.1mm |
| 240° | rot 151° / shift 2417mm | 0.976 | 7.3mm |
| 300° | rot 15° / shift 302mm | 0.958 | 7.6mm |

> 大 shift 配合旋转实质是「绕树轴附近某点的旋转」的表达，变换后点云回到基准帧区域（已验证 bbox 重合），非错误解。

## 输出

- `combined.txt` / `combined.pcd`：六帧配准合并后的完整点云（8136 点，体素合并去重）
- `combined.png`：4 视角静态渲染
- `combined_rot.gif`：绕竖轴旋转动画
- `aligned_*.txt`：各帧配准到 0° 基准坐标系后的单帧点云

## 复查命令

conda activate ms_pointcloud_midterm
python view_txt.py experiment_cripped/combined.txt