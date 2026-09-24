# matlab — MATLAB 数据处理与曲面显示工具

| 文件 | 作用 |
| --- | --- |
| `handle_gp_datas.m` | gp（光谱采集）数据目录整理与拼接（Python 等价实现见 `../pointcloud_generation/handle_datas.py`） |
| `get_points_I_wanted.m` | 按条件挑选点 |
| `gs_surface_display.m` / `gs_surface_compatible.m` / `gs_advanced_surface_fixed.m` | 点云曲面显示（基础 / 兼容 / 进阶三版） |
| `show_surface.m` / `main.m` / `test.m` | 入口与测试脚本 |
| `A2026_5_5.m` | 2026-05-05 采集数据的处理脚本（波形峰值、反射率、点云与颜色输出） |
| `generate_cmf.m` | 生成 CIE 颜色匹配函数（CMF）表 |

数据（`*.mat` / `*.txt` / `*.bin`）不入库；脚本中的路径按各自数据目录约定配置后再运行。
