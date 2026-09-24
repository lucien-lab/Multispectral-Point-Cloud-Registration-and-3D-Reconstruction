# Python 点云生成

该脚本从 `left.bin`、`right.bin` 和对应角度文件直接生成左右点云，不修改原始数据。

## 运行

在项目根目录执行：

```bash
MPLCONFIGDIR=/tmp/matplotlib conda run -n ms_pointcloud_midterm \
  python 20260328/generate_pointcloud.py
```

只处理单侧：

```bash
conda run -n ms_pointcloud_midterm \
  python 20260328/generate_pointcloud.py --sides left
```

默认输出目录为 `20260328/pointcloud_output/`：

- `left_pointcloud.ply` / `right_pointcloud.ply`：有效点，含 XYZ、RGB、距离和 6 通道反射率。
- `*_pointcloud_all.ply`：包含全部 ADC 帧的 PLY，便于完整复现 MATLAB 的绘图结果。
- `*_pointcloud_all.csv`：全部 ADC 帧，含坐标、角度、相关峰、有效标记、浮点/8 位 RGB 和反射率。
- `*_pointcloud_all.npz`：适合 NumPy 后续处理的无损数组。
- `*_pointcloud_preview.png`：彩色点云预览。
- `metadata.json`：参数、波长顺序、点数和数值范围。

波长/采集通道顺序沿用 `A2026_5_5.m`：`495, 696, 600, 803, 545, 642 nm`。RGB 仅使用可见光通道，803 nm 仍保存在光谱字段中。

颜色使用 CIE 官方的 CIE 1931 标准色度观察者（2°）数据，数据集 DOI 为 `10.25039/CIE.DS.xvudnb9b`，原始来源为 CIE 018:2019 Table 6。CIE XYZ 使用标准 D65 矩阵和 sRGB 传递函数转换为显示 RGB。仓库原有的非标准 `cmf-461-700nm.mat` 不再参与计算。

## 交互式可视化

同时显示左右有效点：

```bash
conda run -n ms_pointcloud_midterm \
  python 20260328/visualize_pointcloud.py
```

默认按扫描来源着色：左侧为绿色，右侧为红色。查看 `2026_7_26`
数据中距离为 6.3～10 m 的左右点云：

```bash
conda run -n ms_pointcloud_midterm \
  python 20260328/visualize_pointcloud.py \
  --input-dir 20260328/2026_7_26 \
  --side both --color side \
  --min-distance 6.3 --max-distance 10
```

只显示左侧并按高度着色：

```bash
conda run -n ms_pointcloud_midterm \
  python 20260328/visualize_pointcloud.py --side left --color height
```

常用选项：

- `--side left|right|both`：选择数据。
- `--color side|rgb|height|distance|rfl-495|...`：初始着色方式，默认 `side`。
- `--all`：包含无效帧。
- `--point-size 5`：调整点大小。
- `--background white`：使用白色背景。
- `--info`：仅打印范围与点数，不打开窗口。

窗口快捷键：`0` 为左右来源（左绿右红），`1` 为标准 CIE RGB，`2` 为高度，`3` 为距离，`4`～`9` 为六波长反射率，`V` 切换有效点/全部点，`S` 保存截图。鼠标左键旋转、中键平移、滚轮缩放。
