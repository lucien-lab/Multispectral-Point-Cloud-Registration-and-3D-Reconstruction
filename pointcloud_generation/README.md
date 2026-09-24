# 20260328 点云可视化与颜色恢复说明

## 1. 原始数据

本目录的主要输入数据为：

```text
left.bin          左侧 ADC 原始数据，9000 帧
right.bin         右侧 ADC 原始数据，11000 帧
../angle/left.txt 左侧角度数据
../angle/rigt.txt 右侧角度数据（文件名 rigt 应为 right）
```

每帧包含 13312 个小端 `int16` ADC 采样点。程序从一帧中提取六个时分复用回波，当前使用的波长时序为：

```text
[495, 696, 600, 803, 545, 642] nm
```

## 2. 原来可以生成彩色点云的 Python 代码

完整生成程序是：

```text
generate_pointcloud.py
```

交互式显示程序是：

```text
visualize_pointcloud.py
```

原来生成可见彩色点云时使用的白板参考幅值为：

```python
WHITE_BOARD_AMPLITUDES = np.array(
    [124.1064096774194, 120.2428838709678, 115.0127903225807,
     166.3828838709677, 115.8136612903226, 226.6485516129032],
    dtype=np.float64,
)
```

这六个值依次对应：

| 通道 | 波长 | 原白板幅值 |
|---:|---:|---:|
| 1 | 495 nm | 124.1064 |
| 2 | 696 nm | 120.2429 |
| 3 | 600 nm | 115.0128 |
| 4 | 803 nm | 166.3829 |
| 5 | 545 nm | 115.8137 |
| 6 | 642 nm | 226.6486 |

原程序的反射率计算为：

```python
reflectance[channel] = (
    peak_value - baseline
) / WHITE_BOARD_AMPLITUDES[channel]
```

颜色转换位于 `generate_pointcloud.py` 的 `spectral_rgb()` 函数，主要步骤为：

```text
六波段反射率
→ 排除 803 nm 近红外通道
→ CIE 1931 XYZ
→ 线性 sRGB
→ sRGB Gamma
→ 裁剪到 [0,1]
```

生成左右彩色点云：

```bash
cd /Users/lucien/workspace/graduate-study/papers/3_28/20260328
python3 generate_pointcloud.py --sides left right
```

生成结果位于：

```text
pointcloud_output/left_pointcloud.ply
pointcloud_output/right_pointcloud.ply
pointcloud_output/left_pointcloud_preview.png
pointcloud_output/right_pointcloud_preview.png
pointcloud_output/left_pointcloud_all.npz
pointcloud_output/right_pointcloud_all.npz
```

交互式查看 RGB 点云：

```bash
python3 visualize_pointcloud.py --side both --color rgb
```

快捷键：

```text
1：RGB
2：按高度着色
3：按距离着色
4～9：分别查看六个波长的反射率伪彩色
V：切换有效点和全部点
S：保存截图
```

需要的 Python 包：

```text
numpy
scipy
matplotlib
open3d（仅交互查看需要）
```

## 3. 为什么更新白板幅值后看起来没有颜色

当前更新后的白板幅值为：

```python
WHITE_BOARD_AMPLITUDES = np.array(
    [531.2975, 396.4109, 425.0606,
     361.2250, 572.4702, 477.3115],
    dtype=np.float64,
)
```

反射率采用“目标幅值除以白板幅值”。新白板值比原值大约高 2～5 倍，因此反射率会明显降低：

| 波长 | 新反射率约为原来的 |
|---:|---:|
| 495 nm | 23.4% |
| 696 nm | 30.3% |
| 600 nm | 27.1% |
| 803 nm | 46.1% |
| 545 nm | 20.2% |
| 642 nm | 47.5% |

使用新参数重新计算后，本批数据的平均 RGB8 大约为：

```text
left： (39, 17, 3)
right：(40, 17, 3)
```

这已经非常接近黑色。在黑色背景的点云查看器中，会表现为“没有颜色”甚至“看不到点”。这不代表 PLY 没有 RGB 字段，而是 RGB 数值太低。

造成这种现象的可能原因包括：

1. 新白板幅值不是与 `left.bin/right.bin` 同期采集的；
2. 新白板采集使用了不同的 ADC 增益、激光功率或白板距离；
3. `A0Baiban.m` 的白板值属于另一批实验；
4. 当前颜色转换没有自动曝光或白点校正；
5. 最短波长只有 495 nm，蓝色信息本来就不足。

## 4. 处理方法一：恢复原来的可视化效果

如果目标是复现原来可以正常看见颜色的点云，应把 `generate_pointcloud.py` 中的有效配置恢复为原数组：

```python
WHITE_BOARD_AMPLITUDES = np.array(
    [124.1064096774194, 120.2428838709678, 115.0127903225807,
     166.3828838709677, 115.8136612903226, 226.6485516129032],
    dtype=np.float64,
)
```

然后重新运行：

```bash
python3 generate_pointcloud.py --sides left right
python3 visualize_pointcloud.py --side both --color rgb
```

这种方法能够恢复原来的显示效果，但旧白板值未必代表真实、可追溯的反射率标定。

## 5. 处理方法二：保留新白板标定，同时增强显示

如果确认新白板值与场景数据的采集条件一致，应保留新值，并在 RGB 计算后增加一个统一曝光系数。统一曝光只负责让点云可见，不应分别随意调整 R、G、B。

可在 `process_side()` 中的以下代码之后：

```python
rgb, cie_xyz, color_source = spectral_rgb(reflectance)
```

加入用于显示的自动曝光：

```python
luminance = (
    0.2126 * rgb[:, 0]
    + 0.7152 * rgb[:, 1]
    + 0.0722 * rgb[:, 2]
)

usable = mainwave_ok & np.isfinite(luminance) & (luminance > 0)
if np.any(usable):
    reference = np.percentile(luminance[usable], 75)
    exposure = 0.55 / max(reference, 1e-6)
    rgb = np.clip(rgb * exposure, 0.0, 1.0)
```

说明：

- `0.55` 是目标显示亮度，可以在 `0.35～0.65` 范围内调整；
- 该处理适合可视化，不应把增强后的 RGB 当作原始反射率；
- 六波段反射率仍应保存增强前的数值；
- 论文中应说明点云颜色经过统一曝光增强。

## 6. 推荐的科学标定方式

如果颜色用于实验结论，而不仅是展示，应采集：

```text
Dark/       激光关闭或遮挡接收端的暗场数据
White_Pre/  场景采集前的白板数据
White_Post/ 场景采集后的白板数据
ColorChart/ 标准色卡数据
```

反射率应计算为：

```text
(目标幅值 - 暗场幅值)
----------------------- × 白板标准反射率
(白板幅值 - 暗场幅值)
```

白板必须尽量与场景保持相同的：

- ADC 增益；
- 激光功率；
- 白板距离和朝向；
- 环境光；
- 光学与采样配置。

然后使用色卡拟合六波段到 XYZ 的颜色校正矩阵，而不是只依赖人工的 RGB 增益。

## 7. 当前可直接查看的结果

使用新白板幅值生成的预览：

```text
recalibrated_color/left_updated_color.png
recalibrated_color/right_updated_color.png
recalibrated_color/left_right_updated_color.png
```

无颜色的几何点云：

```text
geometry_only/left_geometry_only.ply
geometry_only/right_geometry_only.ply
geometry_only/left_right_geometry_comparison.png
```

注意：修改白板参数后，已有的 CSV、NPZ、PLY 和 PNG 不会自动更新，必须重新运行生成程序。
