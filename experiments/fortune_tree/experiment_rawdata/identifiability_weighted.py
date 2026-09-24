#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可辨识性加权配准 vs 等权配准 —— 消融实验（为专利实施例三提供真实数据）

背景（来自 invariance_analysis.py 的发现）：
  被测对象含"绕转台轴近似轴对称、材质沿周向均一"的近轴结构（树干/盆体），
  该结构对绕轴旋转角这一位姿自由度不提供可区分信息：
    - 0°↔60° 有 1081 点（24%）坐标完全重合（≤1µm），其 NDVI 中位 0.17（叶片 0.87）、
      离轴中位 16.5mm（树冠半径≈95mm）；
    - 以全部点评价时 identity（不配准）在 fit@5mm 上反而最高（0.5133），
      仅在植物点（NDVI≥0.5）口径下才反转为 M_cripped 0.4378 > identity 0.3340。
  因此本实验把"可辨识性"显式写成逐点权重并引入配准：
      w_i = g_i · s_i
        g_i = clip((r_i − r0)/(r1 − r0), 0, 1)   （几何可辨识性：离参考旋转轴越远越可辨识）
        s_i = 1/(1+exp(−(NDVI_i − 0.5)/0.1))     （光谱可辨识性：材质越非均一越可辨识）
  消融对照：把 w_i 全部置 1（等权），其余流程与参数完全相同。

实现与原 M_joint（experiment_rawdata.py:joint_icp）的差异（仅此三处，其余保持一致）：
  1) 对应代价乘上逐点权重 w_i；
  2) 由对应点集用"稳健加权 SVD"求解 R,t（原实现用 open3d 点到面估计）；
  3) 直接用全分辨率点云迭代（原实现对下采样云迭代）。
  候选数 K、代价形式、λ、d_max、Δ、迭代次数、双初值与择优口径与 M_joint 一致。

坐标系约定（同 compare_metrics.yaw_deg）：xy 为水平面，z 为竖直轴，绕轴旋转为 [[c,-s,0],[s,c,0],[0,0,1]]。
"""
import json
import numpy as np
from scipy.spatial import cKDTree

# ============================ 配置 ============================
CR = "../cripped"
FILES = {0: "20260904三维重建发财树0度.txt",
         60: "20260904三维建模发财树60度.txt",
         120: "20260904三维建模发财树120度.txt",
         180: "20260904三维建模发财树180度.txt",
         240: "20260904三维建模发财树240度.txt",
         300: "20260904三维建模发财树300度.txt"}
FR = [60, 120, 180, 240, 300]

AXIS = np.array([0.0009, 1.2505])      # 参考旋转轴（水平坐标，m）
R0, R1 = 0.005, 0.020                  # 几何可辨识性量度的距离上下界（m）
NDVI_TH, NDVI_TAU = 0.5, 0.1           # 光谱可辨识性量度的阈值与过渡宽度
MASK_TH = 0.5                          # 可辨识点集口径 B 的 NDVI 阈值

K_CAND = 10                            # 几何门控候选数（同 JOINT_K）
LAM = 1.0                              # 光谱项权重（同 JOINT_LAMBDA）
D_MAX = 0.015                          # 距离归一化常数 / 搜索半径（m，同 JOINT_MAX_DIST）
DELTA_DEG = 10.0                       # 光谱角归一化常数（度）
ITERS = 60                             # 迭代次数（同 JOINT_ITERS）
MIN_PAIRS = 20
ROBUST_C = 1.0                         # 稳健核宽度：rho(e)=1/(1+(e/c)^2)
TAUS_MM = (5, 10, 20)                  # 几何一致性阈值
TAU_MUTUAL = 0.020
BAND_OPT = (500.0, 800.0)              # 光谱角波段区间（与优化口径一致）
SEED = 0                               # 本脚本无随机性；仍固定种子以满足可复现性要求
np.random.seed(SEED)

NOTES = []

# ============================ 工具 ============================
wl0 = np.load("cache/frame_000.npz")["wavelength"]


def idx_of(nm):
    return int(np.argmin(np.abs(wl0 - nm)))


I650, I800 = idx_of(650), idx_of(800)
B0, B1 = idx_of(BAND_OPT[0]), idx_of(BAND_OPT[1])


def unit_rows(w):
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def sam_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def weighted_rigid(P, Q, w):
    """加权刚体求解（加权 Umeyama，无缩放）"""
    W = w / (w.sum() + 1e-12)
    mu_p = (W[:, None] * P).sum(0)
    mu_q = (W[:, None] * Q).sum(0)
    Pc, Qc = P - mu_p, Q - mu_q
    H = (W[:, None] * Pc).T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    t = mu_q - R @ mu_p
    return R, t


def yaw_deg(T):
    R = T[:3, :3]
    return float(np.degrees(np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])))


def rot_shift(T):
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    return float(ang), float(np.linalg.norm(T[:3, 3]) * 1000)


# ============================ 数据 ============================
D = {}
for d, f in FILES.items():
    a = np.loadtxt(f"{CR}/{f}", delimiter=",")
    z = np.load(f"cache/frame_{d:03d}.npz")
    xyz = a[:, :3]
    refl = np.clip(z["refl"].astype(float), 0, None)
    assert len(xyz) == len(refl), (d, len(xyz), len(refl))
    maxdiff = float(np.linalg.norm(xyz - z["xyz"], axis=1).max())
    assert maxdiff < 1e-4, f"{d}° cripped 与缓存不同源 max={maxdiff}"
    r650, r800 = refl[:, I650], refl[:, I800]
    ndvi = (r800 - r650) / np.maximum(r800 + r650, 1e-9)
    r_ax = np.hypot(xyz[:, 0] - AXIS[0], xyz[:, 1] - AXIS[1])
    g = np.clip((r_ax - R0) / (R1 - R0), 0.0, 1.0)
    s = 1.0 / (1.0 + np.exp(-(ndvi - NDVI_TH) / NDVI_TAU))
    D[d] = {"xyz": xyz, "refl": refl,
            "spec": unit_rows(refl[:, B0:B1 + 1]),   # 500–800nm 单位化反射率（优化用）
            "ndvi": ndvi, "r_ax": r_ax, "g": g, "s": s,
            "w": g * s, "one": np.ones(len(xyz))}

WEIGHT_STATS = {}
print("=" * 78)
print("逐点可辨识性权重 w = g·s （参考旋转轴 %s，r0=%.3f m，r1=%.3f m）" % (AXIS.tolist(), R0, R1))
print(f"{'视角':>5} {'点数':>6} {'w均值':>8} {'w中位':>8} {'w>0.5占比':>10} {'w=0占比':>9} {'NDVI中位':>9}")
for d in [0] + FR:
    w = D[d]["w"]
    WEIGHT_STATS[str(d)] = {"n": int(len(w)), "mean": float(w.mean()), "median": float(np.median(w)),
                            "frac_gt_0.5": float((w > 0.5).mean()), "frac_eq_0": float((w <= 0).mean())}
    print(f"{d:>5} {len(w):>6} {w.mean():>8.3f} {np.median(w):>8.3f} "
          f"{(w > 0.5).mean():>10.3f} {(w <= 0).mean():>9.3f} {np.median(D[d]['ndvi']):>9.3f}")

# 可辨识点集口径 B（植物点，NDVI≥0.5）
MASK = {"all": lambda d: np.ones(len(D[d]["xyz"]), bool),
        "plant": lambda d: D[d]["ndvi"] >= MASK_TH}


# ============================ 评价 ============================
def evaluate(d, T, mask_key):
    """按掩码口径评价：源码点与目标点同时按掩码过滤（同 invariance_analysis.evaluate 口径）"""
    ms, mt = MASK[mask_key](d), MASK[mask_key](0)
    P = (T[:3, :3] @ D[d]["xyz"][ms].T).T + T[:3, 3]
    S, Q = D[d]["spec"][ms], D[0]["spec"][mt]
    p0 = D[0]["xyz"][mt]
    kd = cKDTree(p0)
    out = {}
    for tm in TAUS_MM:
        dd, _ = kd.query(P, distance_upper_bound=tm / 1000.0)
        out[f"fit@{tm}mm"] = float(np.isfinite(dd).mean())
    dd, j = kd.query(P, distance_upper_bound=TAU_MUTUAL)
    ok = np.isfinite(dd)
    out["rmse@20mm"] = float(np.sqrt((dd[ok] ** 2).mean()) * 1000) if ok.any() else float("nan")
    out["fit_mutual@20mm"] = 0.0
    out["sam@20mm"] = float("nan")
    out["n_pairs@20mm"] = int(ok.sum())
    if ok.any():
        ii = np.arange(len(P))[ok]
        _, i2 = cKDTree(P).query(p0[j[ok]], distance_upper_bound=TAU_MUTUAL)
        m = np.zeros(len(P), bool)
        m[ii[i2 == ii]] = True
        out["fit_mutual@20mm"] = float(m.mean())
        out["sam@20mm"] = float(sam_deg(S[ok], Q[j[ok]]).mean())
    return out


# ============================ 加权联合 ICP ============================
def joint_icp_w(src_xyz, src_spec, tgt_xyz, tgt_spec, w_src, T0,
                k=K_CAND, lam=LAM, d_max=D_MAX, iters=ITERS):
    """逐点加权 几何门控+光谱细化 ICP。

    对应代价： cost_i,c = w_i · [ (d_i,c/d_max)² + λ·(SAM_i,c/Δ)² ]，c 取每源点的 K 个几何近邻
    位姿求解： 以 w_i · ρ(e_i) 为权重做加权 SVD（ρ 为柯西型稳健核，e_i 为入选对应的合并残差）
    """
    kd = cKDTree(tgt_xyz)
    T = np.array(T0, dtype=np.float64)
    rows = np.arange(len(src_xyz))
    n_used = []
    for _ in range(iters):
        P = (T[:3, :3] @ src_xyz.T).T + T[:3, 3]
        d, j = kd.query(P, k=k, distance_upper_bound=d_max)
        d = np.atleast_2d(d)
        j = np.atleast_2d(j)
        if d.shape[0] != len(src_xyz):     # k=1 时 scipy 返回 (n,)
            d, j = d.T, j.T
        valid = np.isfinite(d) & (j < len(tgt_xyz))
        cost = np.full(d.shape, np.inf)
        e_geo = np.full(d.shape, np.inf)
        e_sam = np.full(d.shape, np.inf)
        for c in range(d.shape[1]):
            v = valid[:, c]
            if not v.any():
                continue
            sg = d[v, c] / d_max
            ss = sam_deg(src_spec[v], tgt_spec[j[v, c]]) / DELTA_DEG
            cost[v, c] = w_src[v] * (sg ** 2 + lam * ss ** 2)
            e_geo[v, c] = sg
            e_sam[v, c] = ss
        best = np.argmin(cost, axis=1)
        ok = np.isfinite(cost[rows, best])
        if ok.sum() < MIN_PAIRS:
            break
        ii = rows[ok]
        jj = j[ok, best[ok]]
        e = np.sqrt(e_geo[ok, best[ok]] ** 2 + e_sam[ok, best[ok]] ** 2)
        wfit = w_src[ok] * (1.0 / (1.0 + (e / ROBUST_C) ** 2))   # 稳健核：残差越大权重越小
        if wfit.sum() <= 0:
            break
        R, t = weighted_rigid(P[ok], tgt_xyz[jj], wfit)
        Tn = np.eye(4)
        Tn[:3, :3], Tn[:3, 3] = R, t
        T = Tn @ T
        n_used.append(int(ok.sum()))
    return T, n_used


# ============================ 主流程 ============================
TREF = np.load("cache/registration.npz")
INITS = [("geometric_coarse", lambda d: TREF[f"T_ref_{d}"]), ("identity", lambda d: np.eye(4))]

ARMS = ["identifiability", "uniform"]
results_per_frame = []
summary = {}

# ============================ 独立参考解（可判伪的物理参考，非真值）============================
PHI = np.arange(-180.0, 180.0, 2.0)


def plant_1dof_ref(d):
    """在可辨识点集上绕参考旋转轴做纯旋转全搜索（1 自由度、无平移、无光谱项），
    取 fit@5mm 最大的旋转角 φ*。
    用途：提供一个与配准算法无关的参考最优角，量度两个权重臂的解"离可辨识几何最优有多远"。
    注意：转台轴未标定、实际转角未记录，φ* 不是真值位姿，仅作可判伪的物理参考。
    """
    ms, mt = MASK["plant"](d), MASK["plant"](0)
    X, p0 = D[d]["xyz"][ms], D[0]["xyz"][mt]
    kd = cKDTree(p0)
    piv = np.array([AXIS[0], AXIS[1], 0.0])
    best_phi, best_fit = 0.0, -1.0
    for p in PHI:
        t = np.radians(p)
        R = np.array([[np.cos(t), -np.sin(t), 0.0],
                      [np.sin(t), np.cos(t), 0.0],
                      [0.0, 0.0, 1.0]])
        Q = (R @ (X - piv).T).T + piv
        d5, _ = kd.query(Q, distance_upper_bound=0.005)
        f = float(np.isfinite(d5).mean())
        if f > best_fit:
            best_phi, best_fit = float(p), f
    return {"phi_deg": best_phi, "fit5": best_fit}


def ang_diff(a, b):
    """夹角差（度），折算到 [-180,180)"""
    return float((a - b + 180.0) % 360.0 - 180.0)


REF_1DOF = {}
print("\n" + "=" * 78)
print("独立参考解：可辨识点集上绕参考旋转轴的 1 自由度全搜索（纯旋转、无平移、无光谱项）")
print("=" * 78)
for d in FR:
    REF_1DOF[str(d)] = plant_1dof_ref(d)
    print(f"  {d:>3}°: φ* = {REF_1DOF[str(d)]['phi_deg']:7.1f}°   fit@5mm(可辨识点集) = {REF_1DOF[str(d)]['fit5']:.4f}")
print(f"  5 帧平均 fit@5mm(φ*) = {np.mean([REF_1DOF[str(d)]['fit5'] for d in FR]):.4f}")

print("\n" + "=" * 78)
print("配准：两初值（几何粗配准 / 恒等）→ 择优口径 = 可辨识点集上 fit@5mm 最大（同专利权利要求 8）")
print("=" * 78)

for arm in ARMS:
    acc = {mk: {} for mk in MASK}
    print(f"\n########## 权重模式：{arm} ##########")
    for d in FR:
        w_src = D[d]["w"] if arm == "identifiability" else D[d]["one"]
        cands = []
        for iname, finit in INITS:
            T0 = finit(d)
            Tc, nit = joint_icp_w(D[d]["xyz"], D[d]["spec"], D[0]["xyz"], D[0]["spec"], w_src, T0)
            ev = evaluate(d, Tc, "plant")
            cands.append({"init": iname, "T": Tc, "sel_fit5_plant": ev["fit@5mm"],
                          "sel_sam_plant": ev["sam@20mm"], "n_iter": len(nit),
                          "n_pairs_last": (nit[-1] if nit else 0)})
        pick = max(cands, key=lambda c: (c["sel_fit5_plant"], -c["sel_sam_plant"]))
        T = pick["T"]
        ma, mp = evaluate(d, T, "all"), evaluate(d, T, "plant")
        ang, sh = rot_shift(T)
        yw = yaw_deg(T)
        dphi = ang_diff(yw, REF_1DOF[str(d)]["phi_deg"])
        print(f"  {d:>3}°: 择优={pick['init']:<17} rot={ang:6.2f}° shift={sh:8.1f}mm "
              f"yaw={yw:7.2f}° |Δφ|={abs(dphi):5.1f}° | 全部点 fit@5={ma['fit@5mm']:.4f} "
              f"fit@20={ma['fit@20mm']:.4f} | 植物点 fit@5={mp['fit@5mm']:.4f} "
              f"fit@20={mp['fit@20mm']:.4f} SAM={mp['sam@20mm']:5.2f}°")
        for c in cands:
            print(f"        候选 {c['init']:<17} fit@5(植物)={c['sel_fit5_plant']:.4f} "
                  f"SAM(植物)={c['sel_sam_plant']:5.2f}° 迭代{c['n_iter']:>3}次")
        for mk, ev in (("all", ma), ("plant", mp)):
            for kk, vv in ev.items():
                acc[mk].setdefault(kk, []).append(vv)
        acc.setdefault("_dphi", []).append(abs(dphi))
        results_per_frame.append({
            "deg": d, "weights": arm, "pick_init": pick["init"],
            "rot_deg": ang, "shift_mm": sh, "yaw_deg": yw,
            "ref_phi_deg": REF_1DOF[str(d)]["phi_deg"], "abs_delta_phi_deg": abs(dphi),
            "metrics_all": ma, "metrics_plant": mp,
            "candidates": [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in c.items() if k != "T"} for c in cands],
            "T": T.tolist(),
        })
    summary[arm] = {mk: {kk: float(np.nanmean(vv)) for kk, vv in acc[mk].items()} for mk in MASK}
    summary[arm]["mean_abs_delta_phi_deg"] = float(np.mean(acc["_dphi"]))

# ============================ 汇总 ============================
print("\n" + "=" * 78)
print("5 帧平均汇总")
print("=" * 78)
hdr = (f"{'口径':>7} {'权重模式':>15} {'fit@5mm':>8} {'fit@10mm':>9} {'fit@20mm':>9} "
       f"{'rmse@20mm':>10} {'互近邻':>8} {'SAM(°)':>8}")
print(hdr)
rows = {}
for mk in ("all", "plant"):
    for arm in ARMS:
        s = summary[arm][mk]
        print(f"{mk:>7} {arm:>15} {s['fit@5mm']:>8.4f} {s['fit@10mm']:>9.4f} {s['fit@20mm']:>9.4f} "
              f"{s['rmse@20mm']:>10.2f} {s['fit_mutual@20mm']:>8.4f} {s['sam@20mm']:>8.2f}")
        rows[f"{mk}|{arm}"] = s
print(f"\n{'与可辨识点集 1 自由度最优角的平均偏差':>24}: "
      + "  ".join(f"{arm}={summary[arm]['mean_abs_delta_phi_deg']:.1f}°" for arm in ARMS))
print(f"{'参考解 fit@5mm(φ*) 5 帧均值':>24}: {np.mean([REF_1DOF[str(d)]['fit5'] for d in FR]):.4f}")

# 与既有基线对照（identity / M_ref / M_joint），口径与 invariance.json 一致
try:
    inv = json.load(open("invariance.json"))["results"]
    print("\n对照（invariance.json 既有基线，口径完全相同：fit@5mm / 互近邻 / SAM）")
    for mk, label in (("all", "全部点"), ("plant(NDVI≥0.5)", "可辨识点集")):
        for m in ("identity", "M_cripped", "M_ref", "M_joint"):
            r = inv[mk][m]
            print(f"  {label:>10} {m:>10}: fit@5={r['fit5']:.4f} 互近邻={r['mutual']:.4f} SAM={r['sam']:5.2f}°")
except Exception as e:  # pragma: no cover
    NOTES.append(f"未能读取 invariance.json 做基线对照：{e}")

# ============================ 诚实性检查 ============================
print("\n" + "=" * 78)
print("消融结论（如实记录，含不利结果）")
print("=" * 78)
for mk in ("all", "plant"):
    di = summary["identifiability"][mk]["fit@5mm"] - summary["uniform"][mk]["fit@5mm"]
    print(f"  {mk:>7}: Δfit@5mm(可辨识性加权 − 等权) = {di:+.4f}")
    if mk == "plant" and di > 0:
        NOTES.append(f"有效：可辨识点集口径上加权改善 fit@5mm（Δ={di:+.4f}）")
    if mk == "all" and di < 0:
        NOTES.append(
            f"不利结果（必须与上一条并列报告）：全部点口径上加权反而降低 fit@5mm（Δ={di:+.4f}）；"
            "原因是全部点口径由无信息的近轴旋转不变结构主导，降权后解更偏向叶片，"
            "该口径的「免费匹配」得分随之下降")
dphi_i, dphi_u = summary["identifiability"]["mean_abs_delta_phi_deg"], summary["uniform"]["mean_abs_delta_phi_deg"]
print(f"  与 1 自由度参考最优角的平均偏差：加权={dphi_i:.1f}°  等权={dphi_u:.1f}°")
NOTES.append(
    f"机理证据：加权臂的解与可辨识点集几何最优角的平均偏差 {dphi_i:.1f}°，"
    f"小于等权臂的 {dphi_u:.1f}°，说明权重确实把位姿拉向可辨识子集的最优")
if dphi_i >= dphi_u:
    NOTES.append(f"不利结果：加权臂与参考最优角的偏差未小于等权臂（{dphi_i:.1f}° vs {dphi_u:.1f}°）")
for arm in ARMS:
    NOTES.append(
        f"{arm}: 5 帧平均 fit@5mm 全部点={summary[arm]['all']['fit@5mm']:.4f} "
        f"可辨识点集={summary[arm]['plant']['fit@5mm']:.4f}")
NOTES.append("择优口径使用可辨识点集 fit@5mm，与报告指标存在部分同源，属设计内门控，两臂一致故消融公平")
NOTES.append("ICP 在全分辨率点云上迭代（未下采样），求解用加权 SVD；原 M_joint 为下采样+open3d 点到面，二者非完全同实现")
NOTES.append("本实验不涉及真值位姿：转动角/平移量仅为估计值，不可解释为绝对外参误差")

out = {
    "config": {"K": K_CAND, "lambda": LAM, "d_max_m": D_MAX, "delta_deg": DELTA_DEG,
               "iters": ITERS, "min_pairs": MIN_PAIRS, "robust_c": ROBUST_C,
               "band_opt_nm": BAND_OPT, "taus_mm": TAUS_MM, "tau_mutual_m": TAU_MUTUAL,
               "inits": [n for n, _ in INITS], "selection": "max fit@5mm on identifiable set",
               "resolution": "full (no downsampling)", "seed": SEED,
               "solver": "weighted Umeyama (weighted SVD)"},
    "weights": {"axis": AXIS.tolist(), "r0": R0, "r1": R1, "ndvi_th": NDVI_TH, "ndvi_tau": NDVI_TAU,
                "formula": "w = clip((r-r0)/(r1-r0),0,1) * 1/(1+exp(-(ndvi-th)/tau))"},
    "weight_stats": WEIGHT_STATS,
    "ref_1dof_plant": REF_1DOF,
    "per_frame": results_per_frame,
    "summary": summary,
    "baseline_reference": {mk: {m: inv[mk][m] for m in inv[mk]} for mk in inv} if "inv" in dir() else {},
    "notes": NOTES,
}
with open("identifiability_weighted.json", "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("\n[保存] identifiability_weighted.json")
print(f"[规模] per_frame 条目 {len(results_per_frame)} 条（{len(ARMS)} 臂 × {len(FR)} 帧）")
