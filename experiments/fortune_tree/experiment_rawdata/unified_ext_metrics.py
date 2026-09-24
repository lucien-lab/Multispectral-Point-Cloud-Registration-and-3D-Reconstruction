#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一协议（abc_three_way.json / 表 1 同源，3 轮重复）下的扩展指标复算。

目的：专利「实施例二」原文字（0.5133 / 0.5016 / 0.5112 / 0.4627 等）取自早期四臂协议
（恒等变换、几何配准（原始点云）、几何配准（语义分割点云）、几何光谱联合配准），
其中「几何光谱联合配准」在可辨识点集口径最低；而表 1 与改版图 4 的统一协议结论为
「几何与多光谱融合配准最高」。本脚本按与 invariance_analysis.py 完全相同的指标定义，
为统一协议的三种方式（A 几何 / B 几何+可见光 / C 几何+多光谱）重算同一批指标，
使实施例二的文字可与图 4、表 1 对齐。

指标定义（照抄 experiment_rawdata/invariance_analysis.py 的 evaluate）：
    fit@τ = 变换后源点到基准视角点云的最近邻距离 < τ 的点占比
    互为最近邻一致性 = 命中 20 mm 的源点中同时是该目标点最近邻的比例（以该口径点数为分母）
    光谱角 = 命中 20 mm 的对应点对在 500–800 nm 归一化反射率上的夹角均值（度）
掩码三口径：all / plant(NDVI≥0.5) / far(r>20mm)

轮次口径（与 abc_three_way.py / 表 1 一致）：
    先对 5 个视角取平均，再对 3 轮取均值±标准差；identity 为确定性结果无轮次。

产出：unified_ext_metrics.json
"""
import json
import numpy as np
from scipy.spatial import cKDTree

FR = [60, 120, 180, 240, 300]
ROUNDS = 3
AXIS = np.array([0.0009, 1.2505])
NDVI_TH, R_TH = 0.5, 0.020
ARMS = ["A 几何", "B 几何+RGB", "C 几何+多光谱(本发明)"]
ARMLIST = ["identity"] + ARMS

wl = np.load("cache/frame_000.npz")["wavelength"]
IDX = lambda nm: int(np.argmin(np.abs(wl - nm)))          # noqa: E731
I650, I800, B0, B1 = IDX(650), IDX(800), IDX(500), IDX(800)


def load(d):
    z = np.load(f"cache/frame_{d:03d}.npz")
    R = z["refl"].astype(float)
    r6, r8 = np.clip(R[:, I650], 0, None), np.clip(R[:, I800], 0, None)
    ndvi = (r8 - r6) / np.maximum(r8 + r6, 1e-9)
    s = np.clip(R[:, B0:B1 + 1], 0, None)
    s = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
    xyz = z["xyz"].astype(float)
    return {"xyz": xyz, "s": s, "ndvi": ndvi,
            "r": np.hypot(xyz[:, 0] - AXIS[0], xyz[:, 1] - AXIS[1])}


D = {d: load(d) for d in [0] + FR}
MASKS = {"all": lambda k: np.ones(len(D[k]["xyz"]), bool),
         "plant": lambda k: D[k]["ndvi"] >= NDVI_TH,
         "far": lambda k: D[k]["r"] > R_TH}


def sam_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def evaluate(sk, s0, T, mk, m0, taus=(0.005, 0.010, 0.020, 0.030)):
    P = (T[:3, :3] @ sk["xyz"][mk].T).T + T[:3, 3]
    S, Q = sk["s"][mk], s0["s"][m0]
    kd = cKDTree(s0["xyz"][m0])
    out = {}
    for t in taus:
        d, _ = kd.query(P, distance_upper_bound=t)
        out[f"fit{int(t*1000)}"] = float(np.isfinite(d).mean())
    d, j = kd.query(P, distance_upper_bound=0.020)
    ok = np.isfinite(d)
    out["rmse20"] = float(np.sqrt((d[ok] ** 2).mean()) * 1000) if ok.any() else float("nan")
    if ok.any():
        idxs = np.arange(len(P))[ok]
        _, i2 = cKDTree(P).query(s0["xyz"][m0][j[ok]], distance_upper_bound=0.020)
        m = np.zeros(len(P), bool)
        m[idxs[i2 == idxs]] = True
        out["mutual"] = float(m.mean())
        out["sam"] = float(sam_deg(S[ok], Q[j[ok]]).mean())
    else:
        out["mutual"], out["sam"] = 0.0, float("nan")
    out["n"] = int(len(P))
    return out


POSES = np.load("abc_poses.npz")


def T_of(arm, d, rnd):
    if arm == "identity" or d == 0:
        return np.eye(4)
    return POSES[f"{arm}|{d}"] if rnd is None else POSES[f"{arm}|{d}|r{rnd}"]


def rounds_of(arm):
    return [None] if arm == "identity" else list(range(ROUNDS))


# ---------------- 1) 口径 × 方式 指标（含轮次聚合） ----------------
RES, RES_PF = {}, {}
for arm in ARMLIST:
    for mask in MASKS:
        per_round = []
        for rnd in rounds_of(arm):
            vals = {}
            for d in FR:
                r = evaluate(D[d], D[0], T_of(arm, d, rnd), MASKS[mask](d), MASKS[mask](0))
                RES_PF[f"{arm}|{mask}|{rnd}|{d}"] = r
                for k, v in r.items():
                    vals.setdefault(k, []).append(v)
            per_round.append({k: float(np.nanmean(v)) for k, v in vals.items()})
        keys = per_round[0].keys()
        RES[f"{arm}|{mask}"] = {
            "mean": {k: float(np.nanmean([p[k] for p in per_round])) for k in keys},
            "std": {k: float(np.nanstd([p[k] for p in per_round])) for k in keys},
            "per_round": [{k: round(p[k], 4) for k in keys} for p in per_round]}

print("=== 统一协议扩展指标（先 5 帧平均，再 3 轮均值±标准差；identity 无轮次）===")
print("%-24s | %-15s %-15s %-15s %-15s | %-15s %-13s" %
      ("方式 / 口径", "fit@5mm", "fit@10mm", "fit@20mm", "fit@30mm", "互为最近邻@20", "光谱角(°)"))
for arm in ARMLIST:
    for mask in ["all", "plant", "far"]:
        m, s = RES[f"{arm}|{mask}"]["mean"], RES[f"{arm}|{mask}"]["std"]
        f = lambda k: ("%7.4f±%.4f" % (m[k], s[k])) if s[k] > 0 else ("%7.4f       " % m[k])  # noqa: E731
        print("%-24s | %-15s %-15s %-15s %-15s | %-15s %-13.2f"
              % (f"{arm} / {mask}", f("fit5"), f("fit10"), f("fit20"), f("fit30"),
                 f("mutual"), m["sam"]))

# ---------------- 2) 跨视角融合一致性 ----------------
print("\n=== 跨视角融合一致性（每点在其所属视角之外的其他视角中查最近点）===")
FUS = {}
for arm in ARMLIST:
    acc = {}
    for rnd in rounds_of(arm):
        allp = []
        for d in FR:
            T = T_of(arm, d, rnd)
            allp.append((T[:3, :3] @ D[d]["xyz"].T).T + T[:3, 3])
        for mask in MASKS:
            dists = []
            for i in range(len(FR)):
                mk = MASKS[mask](FR[i])
                P = allp[i][mk]
                kj = [j for j in range(len(FR)) if j != i]
                others = np.vstack([allp[j] for j in kj])
                mm = np.concatenate([MASKS[mask](FR[j]) for j in kj])
                kd = cKDTree(others[mm])
                dd, _ = kd.query(P)
                dists.append(dd)
            dd = np.concatenate(dists) * 1000.0
            acc.setdefault(mask, []).append((float(np.median(dd)), float(np.percentile(dd, 90))))
    FUS[arm] = {mask: {"median_mm": float(np.mean([v[0] for v in acc[mask]])),
                       "p90_mm": float(np.mean([v[1] for v in acc[mask]])),
                       "median_sd": float(np.std([v[0] for v in acc[mask]])),
                       "p90_sd": float(np.std([v[1] for v in acc[mask]]))} for mask in MASKS}
    for mask in ["all", "plant", "far"]:
        v = FUS[arm][mask]
        print("%-24s | 中位数 %6.2f mm  90 分位 %6.2f mm" % (f"{arm} / {mask}", v["median_mm"], v["p90_mm"]))

json.dump({"axis": AXIS.tolist(), "ndvi_th": NDVI_TH, "r_th": R_TH, "rounds": ROUNDS,
           "results": RES, "fusion": FUS, "per_frame": RES_PF,
           "note": "统一协议 = abc_three_way.json 同源位姿（含 3 轮）；指标定义同 invariance_analysis.py"},
          open("unified_ext_metrics.json", "w"), indent=1)
print("\n[保存] unified_ext_metrics.json")
