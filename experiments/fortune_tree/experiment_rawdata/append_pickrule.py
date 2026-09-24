# -*- coding: utf-8 -*-
"""把候选选取规则检验（表 6）与修订后的结论追加进 rgb_vs_ms_vs_geo.md"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MD = HERE / "rgb_vs_ms_vs_geo.md"
J = json.loads((HERE / "rgb_vs_ms_vs_geo.json").read_text())
P = json.loads((HERE / "pickrule_check.json").read_text())
arms = {r["arm"]: r for r in J["arms"]}
A = P["arms"]["A 几何（无光谱）"]
BR1, BR2 = P["arms"]["B 几何+RGB"]["R1"], P["arms"]["B 几何+RGB"]["R2"]
CR1, CR2 = P["arms"]["C 几何+多光谱"]["R1"], P["arms"]["C 几何+多光谱"]["R2"]

head = MD.read_text().split("## 结论")[0].rstrip() + "\n"

L = []
L.append("## 表 6　候选选取规则的影响（同一批候选位姿、两种规则）\n")
L.append("R1（现有规则）：候选需满足 fit@20 mm ≥ 0.90，再取 SAM 最小；")
L.append("R2（几何优先）：取 fit@5 mm 最大的候选。两规则使用完全相同的候选位姿，仅选取方式不同。\n")
L.append("| 臂 / 选取规则 | fit@5mm↑ | fit@10mm↑ | fit@20mm↑ | rmse@20mm↓ | 互最近邻↑ | SAM(500–800)↓ | 估计旋转角(°) |")
L.append("|---|---|---|---|---|---|---|---|")
L.append(f"| A 几何（无光谱，无择优问题） | {A['fit@5mm']:.4f} | {A['fit@10mm']:.4f} | {A['fit@20mm']:.4f} | "
         f"{A['rmse@20mm']:.2f} | {A['fit_mutual@20mm']:.4f} | {A['sam_self']:.2f} | — |")
for nm, r, tag in (("B 几何+RGB", BR1, "R1 光谱优先（现有）"), ("B 几何+RGB", BR2, "R2 几何优先"),
                   ("C 几何+多光谱", CR1, "R1 光谱优先（现有）"), ("C 几何+多光谱", CR2, "R2 几何优先")):
    L.append(f"| {nm}（{tag}） | {r['fit@5mm']:.4f} | {r['fit@10mm']:.4f} | {r['fit@20mm']:.4f} | "
             f"{r['rmse@20mm']:.2f} | {r['fit_mutual@20mm']:.4f} | {r['sam_self']:.2f} | {r['rot']:.2f} |")

L.append("\n多光谱臂（C）逐帧的候选选择：\n")
L.append("| 视角 | R1 选中候选 | R1 旋转角(°) | R1 fit@5mm | R1 SAM(°) | R2 选中候选 | R2 旋转角(°) | R2 fit@5mm | R2 SAM(°) |")
L.append("|---|---|---|---|---|---|---|---|---|")
for r in P["arms"]["C 几何+多光谱"]["per_frame"]:
    L.append(f"| {r['deg']}° | {r['R1']['init']} | {r['R1']['rot']:.1f} | {r['R1']['fit@5mm']:.4f} | "
             f"{r['R1']['sam_self']:.2f} | {r['R2']['init']} | {r['R2']['rot']:.1f} | "
             f"{r['R2']['fit@5mm']:.4f} | {r['R2']['sam_self']:.2f} |")

L.append("\n## 结论\n")
L.append(f"1. **几何类指标不能区分三种方式。** 几何优先选取（R2）下，fit@5 mm 为 A {A['fit@5mm']:.4f}、"
         f"B {BR2['fit@5mm']:.4f}、C {CR2['fit@5mm']:.4f}，三者相差不超过 0.009；"
         f"fit@10 mm 与 rmse@20 mm 上 B（{BR2['fit@10mm']:.4f} / {BR2['rmse@20mm']:.2f} mm）与"
         f"C（{CR2['fit@10mm']:.4f} / {CR2['rmse@20mm']:.2f} mm）还略优于 A"
         f"（{A['fit@10mm']:.4f} / {A['rmse@20mm']:.2f} mm）。"
         f"即几何信息主导位姿收敛，光谱描述子不改变几何收敛水平，配准差异只体现在光谱类指标上。\n")
L.append(f"2. **统一口径的光谱一致性上，多光谱明显优于 RGB，而 RGB 劣于纯几何。** "
         f"R2 下 SAM(500–800) 为 A {A['sam_self']:.2f}°、C {CR2['sam_self']:.2f}°、B {BR2['sam_self']:.2f}°："
         f"多光谱臂与纯几何臂持平，RGB 臂比二者差 {BR2['sam_self']-A['sam_self']:.2f}°。"
         f"原因是 RGB 的 3 个通道不足以区分空间紧邻的同一表面点（表 2 判别比 2.92，"
         f"低于 500–800 nm 的 {J['discriminability'][2]['ratio']:.2f}），"
         f"据此构造的光谱项把位姿拉向光谱角更小但几何不更优的解。\n")
L.append(f"3. **多光谱的作用不是提高几何精度，而是支撑可辨识性判定与逐点加权。** "
         f"RGB 缺少 800 nm 波段，无法计算 NDVI（表 3，NDVI 的 Cohen d 为 4.01，"
         f"最佳的仅可见光代理指标 NGRDI 仅 3.05），而 NDVI 是本案可辨识性权重中植被同类性项 s 的判据，"
         f"故 RGB 方案无法实施逐点可辨识性加权，也无法给出位姿可辨识性判定。\n")
L.append(f"4. **现有择优规则存在「少转」退化风险（重要警告）。** 现有规则 R1 先以 fit@20 mm ≥ 0.90 过滤、"
         f"再取 SAM 最小，而恒等变换本身 fit@20 mm = {arms['F 不配准（恒等）']['fit@20mm']:.4f} ≥ 0.90、"
         f"SAM = {arms['F 不配准（恒等）']['sam500_800']:.2f}°（全表最低），"
         f"因此该规则在多光谱臂上 {sum(1 for r in P['arms']['C 几何+多光谱']['per_frame'] if r['R1']['init']=='单位阵')}/5 帧选中了"
         f"「几乎不旋转」的单位阵候选（"
         + "、".join(f"{r['deg']}° 帧旋转角仅 {r['R1']['rot']:.1f}°"
                     for r in P['arms']['C 几何+多光谱']['per_frame'] if r['R1']['init'] == '单位阵')
         + f"），由此得到的低 SAM 与高互最近邻是选取规则的产物，不能作为「多光谱更准」的证据。"
         f"改用几何优先规则 R2 后，多光谱臂的 SAM 优势消失（{CR2['sam_self']:.2f}° 对 "
         f"{A['sam_self']:.2f}°），同时几何指标回升到与纯几何持平。"
         f"该现象说明：**光谱一致性不能单独作为位姿选取或正确性判据**，"
         f"必须与位姿可辨识性判定（本案步骤 S10 的四层评价与可辨识性告警）配合使用。\n")
L.append(f"5. **波段范围与维度同等重要。** 可见光窄带（450–700 nm，634 维）与全波段（2048 维）的判别比"
         f"分别仅 {J['discriminability'][1]['ratio']:.2f} 与 {J['discriminability'][4]['ratio']:.2f}，"
         f"均低于 500–800 nm 的 {J['discriminability'][2]['ratio']:.2f}，说明并非维度越多越好，"
         f"波段选择本身是配准性能的关键环节（对应说明书的预设波段区间）。\n")
L.append(f"6. **代价**：几何+多光谱单帧平均耗时 {arms['C 几何+多光谱（500–800 nm）']['runtime_s']/5:.1f} s，"
         f"约为纯几何（{arms['A 几何（无光谱）']['runtime_s']/5:.1f} s）的 "
         f"{arms['C 几何+多光谱（500–800 nm）']['runtime_s']/arms['A 几何（无光谱）']['runtime_s']:.1f} 倍，"
         f"主要来自光谱角计算与 PCA。\n")
L.append("\n> 本文件由 `rgb_vs_ms_vs_geo.py`（配准实验）、`pickrule_check.py`（择优规则检验）与")
L.append("> `summarize_rgb_vs_ms.py` + `append_pickrule.py`（汇总）自动生成；")
L.append("> 原始数值见 `rgb_vs_ms_vs_geo.json` 与 `pickrule_check.json`。\n")

MD.write_text(head + "\n".join(L) + "\n")
print("\n".join(L))
print("\n[更新] %s" % MD.name)
