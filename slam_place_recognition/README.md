# slam_place_recognition — 多光谱 LiDAR SLAM 闭环地点识别

基于谱描述子（JSSS）的多光谱 LiDAR SLAM **闭环检测 / 地点识别**流程：描述子提取、候选检索、评价与位姿图优化。

```
slam_place_recognition/
├── src/
│   ├── descriptors/          # jsss.py（本文方法）、m2dp.py / scan_context.py / baselines.py（对比方法）
│   ├── spectral_model.py     # 多光谱反射率建模 / 合成
│   ├── pipeline.py           # 端到端流水线：里程计 -> 候选 -> 描述子判别 -> PR 指标 -> 位姿图 -> ATE
│   ├── retrieval.py          # 检索与相似度
│   ├── pose_graph.py / drift.py
│   ├── augment.py / synth.py / prepare_data.py / kitti_fetch.py / fetch_kitti.py
│   ├── run_main.py / run_ablation_bands.py / run_ablation_components.py / run_robustness.py
│   ├── evaluation: self_test.py / diagnose_similarity.py / compare_with_paper.py / paper_audit.py
│   └── make_figures.py / make_paper_figures.py
└── requirements.txt
```

## 运行

```bash
cd slam_place_recognition
pip install -r requirements.txt

python src/self_test.py          # 自检
python src/run_main.py --help    # 主实验
python src/run_ablation_bands.py --help
python src/run_ablation_components.py --help
python src/run_robustness.py --help
```

> KITTI / SemanticKITTI 与合成数据集不随仓库分发（见 `src/README.md` 的数据准备说明）。
