# experiments/fortune_tree — 发财树多视角配准实验

发财树（含转台）多视角扫描数据的配准与波段信息量分析实验代码。

## 目录

```
experiments/fortune_tree/
├── convert_to_txt.py                 # 原始采集格式 -> 文本点云
├── register.py / register_v2.py       # 两版配准实现（v2 加入光谱联合约束）
├── inspect_views.py / view_txt.py     # 视角检查与点云查看
├── experiment_20260201.py            # 视角配准实验
├── experiment_cripped.py             # 裁剪（cropped）数据实验
├── experiment_rawdata.py             # 原始数据主实验流程
├── experiment_rawdata/               # 原始数据实验子脚本
│   ├── abc_three_way.py              # 几何 / RGB / 多光谱 三方对比
│   ├── six_views_compare.py          # 六视角对比
│   ├── rgb_vs_ms_vs_geo.py           # RGB vs 多光谱 vs 仅几何
│   ├── invariance_analysis.py        # 光谱不变性分析
│   ├── identifiability_weighted.py   # 波段可辨识性 / 加权
│   ├── repeatability.py / noise_scale.py
│   ├── heldout_band_check.py / nested_info_cv.py / nested_info_check.py
│   ├── pickrule_check.py / noise_scale.py / which_metric_best.py
│   ├── metrics_analysis.py / unified_ext_metrics.py / compare_metrics.py
│   ├── registered_3d.py              # 配准后三维结果
│   └── figure5_*.py / _tmp_fig.py / _gen_analysis.py / _append_sec11.py
├── experiment_cripped/               # 裁剪数据实验（含单文件脚本与原理说明）
└── docs/                             # 数据处理与 RGBD->多光谱配准说明
```

## 运行

```bash
cd experiments/fortune_tree
python register_v2.py --help
python experiment_rawdata.py --help
python experiment_rawdata/six_views_compare.py --help
```

> 采集数据（`*.txt` / `*.pcd`）、指标 JSON、图像与 GIF 均不入库，需要自行准备数据目录。
