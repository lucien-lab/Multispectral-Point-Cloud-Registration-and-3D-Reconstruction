# src/ — JSSS 多光谱 LiDAR SLAM 闭环检测（论文方法复现 + 审稿问题可执行化）

本目录按 `JSSS_LiDAR_SLAM_v3(修改).docx` 的论述实现了**完整可运行**的流水线，
并把 `修改意见.md` / `修改意见_多模型评审汇总.md` 中的每一条评审意见变成**可执行的检查或可切换的实验选项**。

```
里程计 → 闭环候选（tau_odom） → JSSS 描述子判别 → 精度/召回/PR → 位姿图 → ATE
```

---

## 1. 快速开始

### 1.1 真实 KITTI 序列 05（数据已就位）

```bash
conda activate ms_pointcloud_midterm
cd src
python prepare_data.py --check kitti        # 校验布局
python self_test.py                         # 不变量自检
python run_main.py --dataset kitti --frames 1400 --out ../results/kitti
python run_ablation_components.py --frames 1400 --protocol paper_loose --out ../results/kitti
python compare_with_paper.py  --dataset kitti --frames 1400 --out ../results/kitti
python diagnose_similarity.py --dataset kitti --frames 1400 --tau-odom 8
python paper_audit.py --demo-posegraph
```

一次完整 KITTI 运行约 3 分钟（1400 帧 × 127k 点）。

### 1.2 合成序列（无网络也能跑）

```bash
cd src
python prepare_data.py                      # 生成 dat/synthetic 元数据
python run_main.py --quick                  # 冒烟：~10 s
python run_main.py --frames 400 --points 20000
python run_ablation_bands.py    --frames 300 --points 12000
python run_robustness.py --sweep all
python make_figures.py --frames 300 --points 12000
```

`../run_all.sh` 按 8 步顺序跑完整套合成流程（`bash run_all.sh quick` 为冒烟模式）。

### 1.3 数据准备（下载）

```bash
python fetch_kitti.py --sequence 05 --frames 1400 --threads 64 --block-mb 2
python fetch_kitti.py --sequence 05 --labels-only
```

用 HTTP Range 从 84.8 GB 的 velodyne 归档里只取需要的成员；详见 `../dat/README.md`
（含两个已踩过的坑：位姿的相机坐标系、SemanticKITTI raw vs learning 标签 id）。

---

## 2. 模块结构与论文对应

| 文件 | 论文位置 | 内容 |
|---|---|---|
| `config.py` | 3.3 / 4.1 / 4.2 / 4.3 | 所有超参：8 波段、σ_s=0.02、τ_odom、τ_sim、40 个阈值、500/100 次迭代、相似度权重 |
| `spectral_model.py` | 4.1 / 6.4.1(a) | 材质→8 波段反射率先验（沥青 0.08–0.10、混凝土 0.20–0.35、植被红边跃升、1550 nm 水汽吸收…）、SemanticKITTI 标签映射、$s=g_i\mu_c+\delta_{c,i}+\varepsilon_i$ |
| `synth.py` | 4.1 | 程序化合成序列（两圈闭合路线、材质栅格、街区组成场、**同材质不同地点**控制量） |
| `data_io.py` | 4.1 | 真实 KITTI + SemanticKITTI 读取；接口与合成后端一致 |
| `drift.py` | 4.2 | 正弦叠加有界漂移，1.0x 时 RMS 自动标定为 7.257 m（与表 6 一致）；提供“把漂移烤进点云”的可选函数 |
| `augment.py` | — | 波段裁剪 / 光谱噪声 / XYZ 噪声+丢点 / 漂移注入点云的包装器 |
| `descriptors/jsss.py` | 3.3 | $h=[h_{quad}\|h_{hist}\|h_{stats}]$，$D=4B+N_{bins}+3B$；归一化逆欧氏距离；可切换权重 |
| `descriptors/scan_context.py` | 3.4 / 5.5 | Nr=15, Ns=40，列位移对齐余弦距离 |
| `descriptors/m2dp.py` | 3.4 / 5.5 | PCA 投影 + 多视角密度直方图；**主轴符号已确定化**（否则跨帧不可复现） |
| `descriptors/baselines.py` | 3.4 | 纯空间、径向直方图、全局（光谱范数）直方图、强度直方图、7 维强度统计量、波段求和 |
| `retrieval.py` | 3.1 / 4.3 | 候选对生成、**三套评测协议**、PR 曲线、F1-max、AUC、AP、accept-all 上限 |
| `pose_graph.py` | 4.3 / 4.4 | Huber 鲁棒松弛；**四种闭环约束测量模型**（copy / coincident / oracle / noisy） |
| `pipeline.py` | 5 | 特征只提取一次（与漂移无关）+ 相似度缓存 + ATE 评估 |
| `run_*.py` | 5.1–5.6 | 表 1–表 6、分量/波段消融、三类鲁棒性扫描 |
| `paper_audit.py` | — | 用论文自己报告的数字复算，输出 PASS/FAIL 清单；演示位姿图退化 |

---

## 3. 与论文的三处**有意偏离**（都是为了修掉审稿指出的问题）

1. **评测协议可选且必须声明**。论文混用了三套协议（表 1–4 无时域间隔、表 5–6 要求间隔 ≥10 帧），
   差 20 倍闭环数。代码提供 `paper_loose` / `paper_pr` / `fixed`，默认全部报告 **Recall**（论文表 1–4 完全没报召回）。
2. **阈值比较可切换**。论文表 1–4 对所有方法用同一个 τ_sim=0.65，而表 5 显示各方法最优阈值横跨 0.017–1.000。
   代码默认 `threshold_mode="per_method_opt"`，同时保留 `fixed_tau` 复现原协议。
3. **闭环约束的来源必须说明**。`pose_graph` 的四种模型让这一点不可回避：
   见下文第 5 节。

---

## 4. 可切换的核心参数

| 参数 | 位置 | 取值 | 作用 |
|---|---|---|---|
| `jsss.weight_mode` | config | `equal` / `paper662` | 论文 3.3（取平均）与 6.6.2（0.5/0.3/0.2）自相矛盾，两者都可跑 |
| `jsss.weights` | config | 任意三元组 | 手工定权；`run_ablation_components.py --search-weights` 可网格搜索 |
| `spectral.within_class_std` | config | 0 / 0.01–0.05 | 类内光谱变化 δ（默认 0 = 论文原模型） |
| `spectral.gain_std` | config | 0 / 0.02–0.1 | 距离/入射角/标定增益 g_i |
| `geom.sigma_xyz`, `geom.drop_rate` | config | 0–0.10 m / 0–30% | 真正的点云几何噪声与丢点 |
| `retrieval.protocol` | config | 三选一 | 评测协议 |
| `posegraph.huber_delta` | config | 1.0 m | Huber 核阈值 |
| `synth.composition_variation` | config | 0 → 1 | 0 = 全路线材质组成相同（只剩类间差异）；1 = 强局部组成差异 |

---

## 5. 代码得出的两个结论（评审用）

**(A) 位姿图松弛在论文描述的方案下是恒等变换。**
若闭环约束的相对位移直接取自**漂移后的里程计**（`copy`），则里程计轨迹是松弛迭代的精确不动点，
ATE 改进恒为 **0.00%**：

```
drift 1.0x  loops=182
  copy        ATE 7.257 -> 7.257 m   improvement +0.00%
  coincident  ATE 7.257 -> 8.271 m   improvement -13.98%
  oracle      ATE 7.257 -> 5.341 m   improvement +26.41%
  noisy       ATE 7.257 -> 5.354 m   improvement +26.23%
```

论文报告 +7.0% / +1.7%，却在 4.3/4.4 从未提及配准（6.4.2(5) 反而自认“未做闭环验证”）。
因此表 6 的改善必须另有来源，需要在论文中说明清楚。

**(B) 表 5 中 JSSS 在 1.0x 的 F1-max 恰好等于“全部接受”的上限。**
277 候选 / 201 正例 ⇒ P=0.726, R=1, F1=0.8410 = 论文报告的 0.841。
`run_main.py` 会直接打印 `accept_all_F1` 供对照。

---

## 6. 输出

`../results/` 下：

| 文件 | 内容 |
|---|---|
| `table1_search_radius.csv` | 表 1（含召回） |
| `table2_bands.csv` | 表 2（**每个波段数都在自身最优阈值下评估**） |
| `table3_components.csv` | 表 3 的**全组合** Q/H/S/QH/QS/HS/QHS |
| `table4_drift.csv` | 表 4（含召回） |
| `table5_pr.csv` | 表 5（含 AP、accept-all 上限） |
| `table6_trajectory.csv` | 表 6（RMSE 与平均误差两种口径并列） |
| `robustness_{drift,spectral,geometric,place}.csv` | 三类误差**分离**的鲁棒性曲线 + 同材质不同地点对照 |
| `fig1_pr_curves.png`, `fig3_component_ablation.png` | PR 曲线与分量消融 |
| `paper_audit.txt` | 论文自洽性 PASS/FAIL 清单 |

---

## 7. 说明

* 合成数据的绝对数值**不等于**论文数值（场景与 KITTI 05 不同）；代码的目标是复现**方法**、
  并让评审问题可复算。拿到真实 KITTI 数据后把 `--dataset kitti` 打开即可。
* `MatrixNormalizer` 只使用序列自身的统计量（逐维 z-score），不使用任何标签，因此对相似度阈值是公平的。
