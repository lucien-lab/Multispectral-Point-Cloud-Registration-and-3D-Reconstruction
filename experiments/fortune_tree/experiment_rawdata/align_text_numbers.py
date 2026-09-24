#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为「实施例二」文字对齐（图 4 / 表 1 统一协议）计算增量、配对比较与相关系数。

输入：unified_ext_metrics.json（三口径四阈值 + 逐帧）、abc_three_way.json（逐帧 rmse/rot/shift/SAM）
输出：纯打印（供人工撰写专利文字），另存 align_text_numbers.json
"""
import json
import numpy as np

E = json.load(open("unified_ext_metrics.json"))
J = json.load(open("abc_three_way.json"))
FR = [60, 120, 180, 240, 300]
ID, A, B, C = "identity", "A 几何", "B 几何+RGB", "C 几何+多光谱(本发明)"
SHORT = {ID: "恒等变换（不配准）", A: "几何配准", B: "几何与可见光融合配准", C: "几何与多光谱融合配准"}
OUT = {}


def g(arm, mask, key):
    return E["results"][f"{arm}|{mask}"]["mean"][key]


def sd(arm, mask, key):
    return E["results"][f"{arm}|{mask}"]["std"][key]


print("=" * 100)
print("一、三口径 × 四阈值（可用于替换原文 [127] 与 [130]）")
for mask, label in (("all", "全部点集"), ("plant", "可辨识点集（NDVI≥0.5）"), ("far", "远离参考旋转轴点集（径向距离大于 20 mm）")):
    print("\n【%s】5 mm / 10 mm / 20 mm / 30 mm" % label)
    for arm in (ID, A, B, C):
        print("   %-14s %.4f / %.4f / %.4f / %.4f   互近邻@20mm %.4f   光谱角 %.2f°"
              % (SHORT[arm], g(arm, mask, "fit5"), g(arm, mask, "fit10"),
                 g(arm, mask, "fit20"), g(arm, mask, "fit30"),
                 g(arm, mask, "mutual"), g(arm, mask, "sam")))

print("\n" + "=" * 100)
print("二、相对恒等变换（不配准）的增量（可用于替换 [132] 与 [133]）")
for mask, label in (("all", "全部点集"), ("plant", "可辨识点集")):
    print("\n【%s】" % label)
    for arm in (A, B, C):
        d = {k: g(arm, mask, k) - g(ID, mask, k) for k in ("fit5", "fit10", "fit20", "fit30", "mutual", "sam")}
        print("   %-14s Δ5 %+.4f  Δ10 %+.4f  Δ20 %+.4f  Δ30 %+.4f  Δ互近邻 %+.4f  Δ光谱角 %+.2f°"
              % (SHORT[arm], d["fit5"], d["fit10"], d["fit20"], d["fit30"], d["mutual"], d["sam"]))
        OUT[f"incr|{mask}|{arm}"] = {k: round(v, 4) for k, v in d.items()}

print("\n" + "=" * 100)
print("三、逐视角配对比较（C 相对 A，5 个视角；效应量=平均差/标准差）")
pf = E["per_frame"]
RND = {"identity": [None], A: [0, 1, 2], B: [0, 1, 2], C: [0, 1, 2]}


def pfv(arm, mask, frame, key):
    v = [pf[f"{arm}|{mask}|{r}|{frame}"][key] for r in RND[arm]]
    return float(np.nanmean(v))
rows = {}
for key in ("fit5", "fit10", "mutual", "sam"):
    d = np.array([pfv(C,"all",f,key) - pfv(A,"all",f,key) for f in FR])
    rows["all|" + key] = (float(d.mean()), float(d.std(ddof=1)), int((d > 0).sum()), d.round(4).tolist())
    print("   全部点集 %-8s 平均差 %+8.4f  标准差 %.4f  效应量 %+6.2f  更优视角数 %d/5  %s"
          % (key, d.mean(), d.std(ddof=1), d.mean() / d.std(ddof=1), (d > 0).sum(), d.round(3).tolist()))
for key in ("fit5", "fit10", "mutual", "sam"):
    d = np.array([pfv(C,"plant",f,key) - pfv(A,"plant",f,key) for f in FR])
    rows["plant|" + key] = (float(d.mean()), float(d.std(ddof=1)), int((d > 0).sum()), d.round(4).tolist())
    print("   可辨识点集 %-8s 平均差 %+8.4f  标准差 %.4f  效应量 %+6.2f  更优视角数 %d/5  %s"
          % (key, d.mean(), d.std(ddof=1), d.mean() / d.std(ddof=1), (d > 0).sum(), d.round(3).tolist()))
OUT["paired_C_vs_A"] = {k: [round(v[0], 4), round(v[1], 4), v[2], v[3]] for k, v in rows.items()}

print("\n" + "=" * 100)
print("四、指标相关性（10 个配准结果 = 5 视角 × 2 方法，方法取 A 与 C）")
tr = []
for arm in (A, C):
    for d in FR:
        r = J[arm]["per_frame"][str(d)]
        tr.append({"arm": arm, "d": d, "fit5": r["fit@5mm"], "rmse20": r["rmse@20mm"],
                   "rot": r["rot"], "shift": r["shift_mm"], "sam": r["SAM500_800"],
                   "plant_fit5": r["plant_fit@5mm"], "plant_rmse20": r["plant_rmse@20mm"],
                   "plant_sam": r["plant_SAM500_800"]})
X = lambda k: np.array([t[k] for t in tr])  # noqa: E731
for nm, a, b in (("几何一致性(fit5,全部点) ↔ 20 mm 距离均方根误差", "fit5", "rmse20"),
                 ("光谱角 ↔ 旋转误差", "sam", "rot"),
                 ("光谱角 ↔ 平移量", "sam", "shift"),
                 ("几何一致性 ↔ 光谱角", "fit5", "sam"),
                 ("可辨识点集几何一致性 ↔ 可辨识点集光谱角", "plant_fit5", "plant_sam")):
    c = float(np.corrcoef(X(a), X(b))[0, 1])
    print("   %-46s r = %+.3f" % (nm, c))
    OUT["corr|" + nm] = round(c, 3)

print("\n" + "=" * 100)
print("五、跨视角融合一致性（可用于替换 [136]）")
for arm in (ID, A, B, C):
    f = E["fusion"][arm]
    print("   %-14s 全部点 中位数 %.2f / 90 分位 %.2f ；可辨识点集 %.2f / %.2f ；远离轴 %.2f / %.2f"
          % (SHORT[arm], f["all"]["median_mm"], f["all"]["p90_mm"],
             f["plant"]["median_mm"], f["plant"]["p90_mm"],
             f["far"]["median_mm"], f["far"]["p90_mm"]))
OUT["fusion"] = E["fusion"]

json.dump(OUT, open("align_text_numbers.json", "w"), indent=1, ensure_ascii=False)
print("\n[保存] align_text_numbers.json")
