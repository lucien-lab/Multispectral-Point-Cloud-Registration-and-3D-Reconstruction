# calibration_and_joint_registration — 颜色标定与颜色/几何联合配准

多光谱点云的**白板辐射/颜色标定**与**光谱-几何联合配准**模块，以及配套的手动配准可视化工具。

## 代码说明

| 文件 | 作用 |
| --- | --- |
| `color_calibration_core.py` | 颜色标定核心：白板参考拟合、通道增益求解、可解释性检查 |
| `calibrate_white_from_photo.py` | 由白板照片标定点云颜色（CLI，含输出审计与写回保护） |
| `register_clouds_to_photo.py` | 点云与照片的外参配准 |
| `joint_registration_core.py` | 颜色 + 几何联合的配准目标与优化 |
| `manual_registration_core.py` | 手动配准（人工给点 / 半自动）核心逻辑 |
| `manual_registration_app.py` | Flask 后端，驱动 `manual_registration_web/` 前端 |
| `manual_registration_web/` | 浏览器端交互式配准界面（原生 JS/CSS，`registration_core.mjs` 为纯逻辑，可在 Node 下测试） |
| `tests/` | 单元测试与 CLI 端到端测试（含 `manual_registration_web.test.mjs`） |

## 运行

```bash
# 白板标定
python calibrate_white_from_photo.py --help

# 点云-照片配准
python register_clouds_to_photo.py --help

# 手动配准界面（前端 http://127.0.0.1:5000）
python manual_registration_app.py

# 测试（必须在模块目录下执行，脚本按相对路径解析数据/输出）
cd . && python -m pytest tests -q
node --test tests/manual_registration_web.test.mjs   # 可选：前端逻辑测试
```

> 数据（`pointcloud_output/` 等）与标定输出（`calibration_output/`）不入库，需按脚本 `--help` 约定的目录结构自行准备。
