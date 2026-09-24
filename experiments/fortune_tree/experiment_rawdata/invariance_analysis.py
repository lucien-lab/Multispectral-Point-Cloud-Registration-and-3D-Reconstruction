#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旋转不变结构分析：identity 的高分是否来自"旋转不变的近轴结构"（花盆/树干）？
用户质疑："identity 指标高，是不是因为评价是自己和自己？"

结论链：
  1) identity 评价的是帧 k vs 帧 0（跨帧、不同点集），不是自己配自己
     —— 自配上限为 fit=1.0/rmse=0/SAM=0，而 identity 实际 0.90/7.95mm/8.01°。
  2) 但存在旋转不变结构：0°↔60° 有 1081 点坐标完全重合（≤1µm），
     NDVI 中位 +0.17（非植物）、离轴 16.5mm（近轴圆柱 = 花盆/树干）、
     点对 SAM 仅 3.05°（同材质）。120–300° 与 0° 无精确重合（网格不同）。
  3) 决定性检验：剔除 (a) 非植物点 (b) 近轴点 后重算四方法指标，
     看 identity 的优势是否还在。
"""
import json
import numpy as np
from scipy.spatial import cKDTree

CR = "../cripped"
FILES = {0: "20260904三维重建发财树0度.txt",
         60: "20260904三维建模发财树60度.txt",
         120: "20260904三维建模发财树120度.txt",
         180: "20260904三维建模发财树180度.txt",
         240: "20260904三维建模发财树240度.txt",
         300: "20260904三维建模发财树300度.txt"}
FR = [60, 120, 180, 240, 300]
AXIS = np.array([0.0009, 1.2505])   # 转台轴（水平坐标）
NDVI_TH = 0.5                        # 植物/非植物阈值
R_TH = 0.020                         # 近轴阈值（m）

wl = np.load("cache/frame_000.npz")["wavelength"]
def idx(nm): return int(np.argmin(np.abs(wl - nm)))
I650, I800 = idx(650), idx(800)
B0, B1 = idx(500), idx(800)

D = {}
for d, f in FILES.items():
    a = np.loadtxt(f"{CR}/{f}", delimiter=",")
    z = np.load(f"cache/frame_{d:03d}.npz")
    R = z["refl"].astype(float)
    r650, r800 = np.clip(R[:, I650], 0, None), np.clip(R[:, I800], 0, None)
    ndvi = (r800 - r650) / np.maximum(r800 + r650, 1e-9)
    s = np.clip(R[:, B0:B1 + 1], 0, None)
    s = s / np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
    D[d] = {"xyz": a[:, :3], "s": s, "ndvi": ndvi,
            "r": np.hypot(a[:, 0] - AXIS[0], a[:, 1] - AXIS[1])}

Tc = json.load(open("compare/transforms_cripped.json"))
Z = np.load("cache/registration.npz")
T = {"identity": {d: np.eye(4) for d in FR},
     "M_cripped": {d: np.array(Tc[str(d)]) for d in FR},
     "M_ref": {d: Z[f"T_ref_{d}"] for d in FR},
     "M_joint": {d: Z[f"T_joint_{d}"] for d in FR}}


def sam_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def evaluate(sk, s0, T, mask_k, mask_0, tau=0.02):
    """在给定掩码（源/目标各自过滤）下计算 fit@5mm / 互近邻@20mm / SAM。"""
    P = (T[:3, :3] @ sk["xyz"][mask_k].T).T + T[:3, 3]
    S, Q = sk["s"][mask_k], s0["s"][mask_0]
    kd = cKDTree(s0["xyz"][mask_0])
    d, j = kd.query(P, distance_upper_bound=tau)
    ok = np.isfinite(d)
    d5, _ = kd.query(P, distance_upper_bound=0.005)
    fit5 = float(np.isfinite(d5).mean())
    if ok.any():
        idxs = np.arange(len(P))[ok]
        _, i2 = cKDTree(P).query(s0["xyz"][mask_0][j[ok]], distance_upper_bound=tau)
        m = np.zeros(len(P), bool)
        m[idxs[i2 == idxs]] = True
        mut = float(m.mean())
        sam = float(sam_deg(S[ok], Q[j[ok]]).mean())
    else:
        mut, sam = 0.0, float("nan")
    return {"fit5": fit5, "mutual": mut, "sam": sam}


MASKS = {"all": lambda k: np.ones(len(D[k]["xyz"]), bool),
         "plant(NDVI≥0.5)": lambda k: D[k]["ndvi"] >= NDVI_TH,
         "far(r>20mm)": lambda k: D[k]["r"] > R_TH}

print("各帧点数与掩码规模：")
for k in [0] + FR:
    print(f"  {k:>3}°: n={len(D[k]['xyz'])}  plant={MASKS['plant(NDVI≥0.5)'](k).sum():4d}"
          f" ({100*MASKS['plant(NDVI≥0.5)'](k).mean():.0f}%)"
          f"  far={MASKS['far(r>20mm)'](k).sum():4d}"
          f" ({100*MASKS['far(r>20mm)'](k).mean():.0f}%)")

res = {}
for mname in MASKS:
    print(f"\n=== 掩码 {mname} ===")
    print(f"{'方法':>10} | {'fit@5mm':>8} {'互近邻':>7} {'SAM(°)':>7}   （5 帧平均）")
    row = {}
    for meth in T:
        vals = {"fit5": [], "mutual": [], "sam": []}
        for d in FR:
            mk = MASKS[mname](d)
            m0 = MASKS[mname](0)
            r = evaluate(D[d], D[0], T[meth][d], mk, m0)
            for k in vals:
                vals[k].append(r[k])
        row[meth] = {k: float(np.nanmean(v)) for k, v in vals.items()}
        print(f"{meth:>10} | {row[meth]['fit5']:>8.4f} {row[meth]['mutual']:>7.4f} {row[meth]['sam']:>7.2f}")
    res[mname] = row

out = {"axis": AXIS.tolist(), "ndvi_th": NDVI_TH, "r_th": R_TH, "results": res}

# ---------- 成对 1-DOF 扫描 + 传递性检验（植物点上） ----------
# 若 fit5 峰对应真实刚体旋转，则 φ*(A→C) ≈ φ*(A→B)+φ*(B→C) (mod 360)
phis = np.arange(-180, 180, 2.0)
Rz = [np.array([[np.cos(np.radians(p)), -np.sin(np.radians(p)), 0],
                [np.sin(np.radians(p)), np.cos(np.radians(p)), 0], [0, 0, 1]]) for p in phis]
piv = np.array([AXIS[0], AXIS[1], 0.0])
# 注意：成对扫描必须在植物点上进行（与上面掩码一致），否则被旋转不变的近轴结构主导
PMASK = {d: D[d]["ndvi"] >= NDVI_TH for d in D}
KD = {d: cKDTree(D[d]["xyz"][PMASK[d]]) for d in D}


def fit5_curve(src, dst):
    out = np.zeros(len(phis))
    X = D[src]["xyz"][PMASK[src]]
    for i, R in enumerate(Rz):
        P = (R @ (X - piv).T).T + piv
        d5, _ = KD[dst].query(P, distance_upper_bound=0.005)
        out[i] = np.isfinite(d5).mean()
    return out


print("\n【成对 fit@5mm 最优角矩阵】(植物点 NDVI≥%.1f, 行=源帧, 列=基准帧)" % NDVI_TH)
print("      " + "".join(f"{d:>7}" for d in D))
PM = {}
for s in D:
    row = []
    for t in D:
        if s == t:
            row.append("   —  "); PM[(s, t)] = 0.0; continue
        c = fit5_curve(s, t); i = int(c.argmax()); PM[(s, t)] = float(phis[i])
        row.append(f"{phis[i]:>7.0f}")
    print(f"{s:>4}: " + "".join(row))
errs = []
keys = list(D)
for a in keys:
    for b in keys:
        for c in keys:
            if len({a, b, c}) < 3:
                continue
            e = abs((PM[(a, b)] + PM[(b, c)] - PM[(a, c)] + 180) % 360 - 180)
            errs.append({"err": e, "chain": f"{a}->{b}->{c}"})
errs.sort(key=lambda x: x["err"])
trans = {"median_err": float(np.median([e["err"] for e in errs])),
         "mean_err": float(np.mean([e["err"] for e in errs])),
         "p90_err": float(np.percentile([e["err"] for e in errs], 90)),
         "best5": errs[:5], "worst5": errs[-5:]}
print(f"\n【传递性】中位误差 {trans['median_err']:.1f}°  平均 {trans['mean_err']:.1f}°  90分位 {trans['p90_err']:.1f}°")
print("  最自洽 5 个:", [(e["chain"], round(e["err"], 1)) for e in trans["best5"]])
print("  最不自洽 5 个:", [(e["chain"], round(e["err"], 1)) for e in trans["worst5"]])
out["pairwise_phi"] = {f"{s}_{t}": v for (s, t), v in PM.items()}
out["transitivity"] = trans

with open("invariance.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("\n[保存] invariance.json")
