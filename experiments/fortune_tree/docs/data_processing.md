# 20260201 三维建模发财树 60° 数据处理与可视化

## 1. 数据结构

数据集目录：`20260201三维建模发财树60度/`

```
20260201三维建模发财树60度/
├── el_re.txt                    # 角度数据（编号、水平角、俯仰角）
├── gp/
│   └── specData/                # 光谱数据（每个测点一个文件）
│       ├── spec_1.txt           # 格式：波长nm \t 强度（2048行）
│       ├── spec_2.txt
│       └── ...（共13688个文件）
└── sbq/                         # 原始测距数据（用于计算三维坐标）
    ├── 1.csv
    ├── 2.csv
    └── ...（共13688个文件）
```

**关键文件说明：**

| 文件 | 内容 | 用途 |
|---|---|---|
| `el_re.txt` | 每行：`编号,水平角,俯仰角` | 计算三维坐标的原始角度 |
| `gp/specData/spec_N.txt` | 每行：`波长nm \t 强度`，共2048行 | 选取RGB三个波段用于着色 |
| `sbq/*.csv` | 原始测距数据 | 从波峰计算每个点的距离 |

---

## 2. 处理流程概览

```
原始数据                        校准数据
┌─────────────┐                ┌─────────────┐
│  sbq/*.csv  │                │ dark.txt    │（20260130分类场景八万点/dark.txt）
│  el_re.txt  │                │ white4-1.txt│（白板/white4-1.txt）
└──────┬──────┘                └──────┬──────┘
       │                               │
       ▼                               ▼
  计算三维坐标                   暗场/白场标定
       │                               │
       └───────────┬───────────────────┘
                   ▼
          RGB点云数据文件
          (data_convert/20260201三维建模发财树60度.txt)
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
   交互查看器   离屏PNG渲染   数据导出
```

---

## 3. 处理代码详解

### 3.1 三维坐标计算

坐标从 `sbq/*.csv` 和 `el_re.txt` 计算，逻辑在 `convert_to_txt.py` 的 `point_coordinates()` 函数中：

```python
def point_coordinates(scan_dir: Path, angles: np.ndarray) -> dict[int, np.ndarray]:
    """从 sbq 测距数据和角度数据计算三维坐标"""
    coordinates: dict[int, np.ndarray] = {}
    for path in sorted((scan_dir / "sbq").glob("*.csv"), key=lambda item: int(item.stem)):
        point_id = int(path.stem)
        if point_id < 1 or point_id > len(angles):
            continue
        raw = load_numeric(path).ravel()
        if raw.size != 1000 or not np.isfinite(raw).all():
            continue
        
        # 从波峰位置计算距离
        peaks, _ = find_peaks(raw[1:], distance=800)
        if peaks.size == 0:
            continue
        distance = (peaks[0] + 1) / 400.0
        
        # 从角度计算XYZ坐标
        horizontal = np.deg2rad(angles[point_id - 1, 1])
        altitude = np.deg2rad(angles[point_id - 1, 2])
        x = distance * np.cos(altitude) * np.sin(horizontal)
        y = distance * np.cos(altitude) * np.cos(horizontal)
        z = distance * np.sin(altitude)
        
        # 绕Y轴旋转180度（MATLAB原版逻辑）
        coordinates[point_id] = np.array([-x, y, -z])
    return coordinates
```

**原理说明：**
- 每个点的 `sbq/*.csv` 包含1000个原始测距值
- 通过找波峰（`find_peaks`）确定飞行时间 → 距离 = 峰值位置 / 400（声速/采样率）
- 用球坐标系（距离 + 水平角 + 俯仰角）转换为XYZ笛卡尔坐标
- 绕Y轴旋转180度与MATLAB原版对齐

### 3.2 暗场/白场标定

标定公式：`rgb = (raw - dark) / (white - dark)`

```python
def calibration_values(dark_path, white_path, bands):
    dark = load_numeric(dark_path)
    white = load_numeric(white_path)
    max_band = max(bands)
    if dark.shape[0] < max_band or white.shape[0] < max_band:
        raise ValueError(f"白场或暗场数据不足以提供波段 {bands}")
    if dark.shape[1] < 2 or white.shape[1] < 2:
        raise ValueError("白场和暗场数据至少需要两列")
    
    zero_based_bands = np.asarray(bands) - 1
    dark_values = dark[zero_based_bands, 1]
    denominator = white[zero_based_bands, 1] - dark_values
    if np.any(np.isclose(denominator, 0)):
        raise ValueError("white-dark 在目标波段存在零值，无法归一化")
    return dark_values, denominator
```

**关键波段（MATLAB 1-based 行号）：**

| 通道 | 波段行号 | 波长 | 语义 | 白板值 | 暗场值 |
|---|---|---|---|---|---|
| R | 957 | 650.14 nm | 红光 | 38011 | 1770 |
| G | 730 | 560.31 nm | 绿光 | 20057 | 1847 |
| B | 552 | 490.38 nm | 蓝光 | 4164 | 1940 |

### 3.3 数据合并与输出

```python
def process(scan_dir, output, bands, dark_file=None, white_file=None):
    # 1. 读取角度数据
    angles = load_numeric(scan_dir / "el_re.txt")
    
    # 2. 计算三维坐标
    coordinates = point_coordinates(scan_dir, angles)
    
    # 3. 计算标定值
    dark_path, white_path = calibration_paths(scan_dir, dark_file, white_file)
    dark_values, denominator = calibration_values(dark_path, white_path, bands)
    
    # 4. 合并坐标和光谱
    rows = []
    zero_based_bands = np.asarray(bands) - 1
    for path in sorted((scan_dir / "gp" / "specData").glob("spec_*.txt"), ...):
        point_id = int(path.stem.split("_")[1])
        xyz = coordinates.get(point_id)
        if xyz is None:
            continue
        spectrum = load_numeric(path)
        raw_rgb = spectrum[zero_based_bands, 1]
        rgb = (raw_rgb - dark_values) / denominator
        rows.append(np.concatenate(([point_id], xyz, rgb, [0])))
    
    # 5. 写入文件（标定反射率在存储时 ×4，与 MATLAB scatter3(*4) 对齐）
    np.savetxt(output, np.vstack(rows), delimiter=",",
               fmt=["%d", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%.6f", "%d"])
```

**输出格式：** `id,x,y,z,R,G,B,类别`（每行一个点，8列逗号分隔）

---

## 4. 运行命令（2026-09 起改为通用脚本）

> ⚠️ 所有发财树数据与脚本已归入 `fortune_tree/` 文件夹：
> 数据（0°~300°、顶端）在 `fortune_tree/` 下，产出在 `fortune_tree/data_convert/` 下。
> 标定文件仍位于 papers 根目录（`20260130分类场景八万点/dark.txt`、`白板/white4-1.txt`），脚本内部已自动回退查找，无需手动指定。

三个通用脚本（均以"输入即转换"为目标，可在 fortune_tree 目录下任意位置运行）：

| 脚本 | 输入 | 输出 |
|---|---|---|
| `convert_to_txt.py` | 扫描**文件夹名** | `data_convert/<文件夹名>.txt` |
| `render_png.py` | 点云 **txt** | PNG 图片 |
| `view_txt.py` | 点云 **txt** | Open3D 交互窗口 |

### 4.1 生成数据文件（任意角度）

```bash
cd fortune_tree
conda activate ms_pointcloud_midterm

# 单个文件夹：输入名字即可，自动输出到 data_convert/
python convert_to_txt.py 20260201三维建模发财树60度

# 批量转换当前目录下所有有效的扫描文件夹
python convert_to_txt.py --all

# 自定义输出路径 / 自定义波段
python convert_to_txt.py 20260204发财树顶端 -o out/top.txt
python convert_to_txt.py 20260201三维建模发财树60度 --bands 957 730 552
```

**输出：** `fortune_tree/data_convert/<文件夹名>.txt`（8 列：id,x,y,z,R,G,B,类别）

### 4.2 交互式查看器（输入 txt）

```bash
cd fortune_tree
python view_txt.py data_convert/20260201三维建模发财树60度.txt
```

**操作说明：**
- 左键拖动 → 旋转
- 中键拖动 / Ctrl+左键 → 平移
- 滚轮 → 缩放
- `+` / `-` → 调整点大小
- `R` → 重置视图
- `Q` / `Esc` → 退出
- 交互查看器默认不再放大（数据已含 ×4，与 MATLAB 原版一致）

### 4.3 离屏 PNG 渲染（输入 txt）

```bash
cd fortune_tree
# 默认：数据已含 ×4 + p95 拉伸 + gamma 0.6，输出到与输入同名的 .png
python render_png.py data_convert/20260201三维建模发财树60度.txt

# 指定角度和增强参数
python render_png.py data_convert/20260204发财树顶端.txt \
  --azimuth 45 --elevation 20 \
  --multiply 4 --stretch 95 --gamma 0.6 \
  -o out/top_enhanced.png
```

**渲染参数说明：**

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--multiply` | 4.0 | RGB 整体倍率（与 MATLAB 原版一致） |
| `--stretch` | 100 | 按该百分位拉伸 RGB（100=不拉伸） |
| `--gamma` | 1.0 | 伽马增强（<1 提亮暗部） |
| `--azimuth` | 30 | 水平视角（度） |
| `--elevation` | 15 | 俯仰视角（度） |
| `--size` | 1280 | 输出像素尺寸 |

---

## 5. 颜色处理分析

### 5.1 为什么原始数据看起来全黑？

**根本原因：** 反射率值极低

| 通道 | 中位数 | p95 | 最大值 |
|---|---|---|---|
| R | 0.0057 | 0.048 | 0.858 |
| G | 0.0058 | 0.035 | 0.806 |
| B | 0.0013 | 0.027 | 0.930 |

**亮度分布（RGB 均值）：**
- 70.3% 的点亮度 < 0.01
- 97.5% 的点亮度 < 0.05
- 中位数仅 0.005

**原因分析：**
1. 发财树是深绿色植物，本身反射率低
2. 景物原始光谱强度（~1930-1941）仅略高于暗场（1770-1940），信号非常弱
3. B 通道（552nm）白板值仅 4164，是 R 通道的 1/9，光源蓝端本身就弱

### 5.2 ×4 的实现

**代码位置：** `convert_to_txt.py` 的 `process()` 函数——标定反射率在**存储时**即乘 4：

```python
raw_rgb = spectrum[zero_based_bands, 1]
rgba = (raw_rgb - dark_values) / denominator
rgb = np.clip(rgba * multiply, 0.0, 1.0)  # ×4 后超 1 截断，写入数据文件
```

即：`data_convert/*.txt` 里存的已经是 ×4 后的值（`--multiply` 可调，设 1 则存原始反射率）。

**效果对比：**

| 渲染方式 | 非背景均值 RGB | 视觉效果 |
|---|---|---|
| 原始（不乘） | [4, 4, 4] /255 | 全黑，什么都看不清 |
| **×4** | [16, 18, 14] /255 | 亮度线性 4 倍，仍偏黑 |
| p95 拉伸 + gamma 0.6 | [98, 126, 81] /255 | 颜色鲜明，可看出绿色植物 |

### 5.3 ×4 与 MATLAB 原版的一致性

**MATLAB 原版（`handle_datas.m` 第58行）：**
```matlab
scatter3(-data_color(:,2), data_color(:,3), data_color(:,4),
         10, data_color(:,5:7)*4, 'filled');  ← RGB×4 显示
```

**Python 版（`convert_to_txt.py`，存储时乘）：**
```python
rgb = np.clip((raw_rgb - dark_values) / denominator * 4.0, 0.0, 1.0)  # 写入数据文件
```

**关键点：**
- ×4 在**存储层**（`convert_to_txt.py` 写文件时），数据文件即显示值
- 查看器（`view_txt.py`）默认 `--multiply 1.0`，不再放大，避免双重 ×4

---

## 6. 生成的文件

```
data_convert/
├── 20260201三维建模发财树60度.txt     # 13659 个点
├── 20260201三维建模发财树120度.txt    # 13639 个点
├── 20260201三维建模发财树180度.txt    # 13646 个点
├── 20260201三维建模发财树240度.txt    # 13651 个点
├── 20260201三维建模发财树300度.txt    # 13643 个点
├── 20260201三维重建发财树0度.txt      # 13646 个点
├── 20260201三维重建发财树300度.txt    # 2446 个点
└── 20260204发财树顶端.txt            # 6556 个点

convert_to_txt.py   # 扫描文件夹 → txt 通用转换工具
render_png.py       # txt → PNG 离屏渲染工具（默认 ×4 + p95 拉伸 + gamma）
view_txt.py         # txt → Open3D 交互查看工具
```

数据文件均为 **×4 后的标定反射率**（rgb = (raw-dark)/(white-dark)×4，超 1 截断），存储与显示一致，下游工具不再缩放。

> ⚠️ 若文件由 `--multiply 1` 生成（原始反射率），查看器需加 `--multiply 4` 才能得到同等亮度。

---

## 7. 常见问题

### Q1: 为什么 B 通道有负值？

**答：** B 通道（552nm）有 5238 个点（38%）为负。原因是该波段信号（raw≈1941）与暗场（dark≈1940）在同一噪声水平，`(raw - dark)` 直接得负。语义上表示"该波段无有效信号"。

### Q2: `_x4.txt` 文件能用吗？

**答：** 不要再用。现在的数据文件本身已含 ×4（`convert_to_txt.py` 存储时乘），旧 `_x4.txt` 与现文件等价但格式不统一，且容易与旧查看器混淆，统一用 `data_convert/` 下新生成的文件。

### Q3: 如何修改显示效果？

**答：** 调整 `render_png.py` 的参数：
```bash
# 更亮（×8）
python render_png.py data_convert/20260201三维建模发财树60度.txt --multiply 8

# 拉伸 + 提亮
python render_png.py data_convert/20260201三维建模发财树60度.txt --stretch 95 --gamma 0.5

# 换个角度
python render_png.py data_convert/20260201三维建模发财树60度.txt --azimuth 60 --elevation 30
```

### Q4: handle_datas.py 能处理这个数据集吗？

**答：** 不能。`handle_datas.py` 需要 `dy.txt` 坐标文件（4列：编号,X,Y,Z），而本数据集没有这个文件。坐标需要从 `sbq/*.csv` + `el_re.txt` 计算，所以必须用 `convert_to_txt.py`。

### Q5: 数据文件里的 RGB 值是 ×4 的吗？

**答：** 是。`data_convert/` 下存的是**×4 后的标定反射率**（`rgb = (raw - dark) / (white - dark) × 4`，超 1 截断），写入时在 `convert_to_txt.py` 完成。如需原始反射率，用 `python convert_to_txt.py <文件夹> --multiply 1` 重新生成。

---

## 8. 依赖环境

```bash
# 创建 conda 环境
conda create -n ms_pointcloud_midterm python=3.11
conda activate ms_pointcloud_midterm

# 安装依赖
pip install numpy scipy open3d matplotlib scikit-learn
```

**测试环境：**
- numpy 2.4.6
- scipy 1.17.1
- open3d 0.19.0
- matplotlib 3.11.0
- scikit-learn 1.9.0

---

## 9. 快速开始（通用版）

```bash
# 0. 进入工作目录、激活环境
cd fortune_tree
conda activate ms_pointcloud_midterm

# 1. 生成数据（输入文件夹名，输出到 data_convert/）
python convert_to_txt.py 20260201三维建模发财树60度
#   或批量：python convert_to_txt.py --all

# 2. 打开交互查看器（输入 txt）
python view_txt.py data_convert/20260201三维建模发财树60度.txt

# 3. 生成 PNG（可选，输入 txt）
python render_png.py data_convert/20260201三维建模发财树60度.txt
```
