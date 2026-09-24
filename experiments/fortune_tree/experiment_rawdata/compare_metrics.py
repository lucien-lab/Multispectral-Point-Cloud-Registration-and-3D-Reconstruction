#!/usr/bin/env python3
"""三方法配准指标统一对比：M_cripped（experiment_cripped）vs M_ref vs M_joint

问题：三者口径不可比 —— experiment_cripped 在 3mm 下采样云上算 fitness@20mm，
      experiment_rawdata 在全分辨率上算。不能直接拿两边的数字对比。

做法：
  1. 重跑 experiment_cripped 的流程取得其变换矩阵（其代码未保存变换）
  2. 三者的变换统一施加到**同一份全分辨率 cripped 点云**上
  3. 用**同一套指标口径**重算 A 几何 / B 光谱 / C 物理先验 / D 代价 四类指标
  4. 排名 + 综合评价

输出：compare/transforms_cripped.json、compare/comparison.json、compare/*.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiment_cripped"))

import experiment_cripped as EC  # noqa: E402  复用其 RANSAC + ICP

CRIPPED = ROOT / "cripped"
CACHE = HERE / "cache"
CMP = HERE / "compare"
FRAMES = [
    ("20260904三维重建发财树0度.txt", 0),
    ("20260904三维建模发财树60度.txt", 60),
    ("20260904三维建模发财树120度.txt", 120),
    ("20260904三维建模发财树180度.txt", 180),
    ("20260904三维建模发财树240度.txt", 240),
    ("20260904三维建模发财树300度.txt", 300),
]
THRESHOLDS = (0.005, 0.010, 0.020, 0.030)
BAND_OPT, BAND_EVAL = (500.0, 800.0), (450.0, 900.0)
METHODS = ("M_cripped", "M_ref", "M_joint")


# ----------------------------- 基础工具 -----------------------------
def unit(w):
    w = np.asarray(w, float)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def sam_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def load_source():
    """全分辨率 cripped 点云 + 从 rawdata 缓存挂接 2048 波段光谱"""
    src = {}
    wl = None
    for fname, deg in FRAMES:
        a = np.loadtxt(CRIPPED / fname, delimiter=",")
        z = np.load(CACHE / f"frame_{deg:03d}.npz")
        xyz, refl = a[:, :3], np.clip(z["refl"].astype(float), 0, None)
        if wl is None:
            wl = z["wavelength"]
            mo = (wl >= BAND_OPT[0]) & (wl <= BAND_OPT[1])
            me = (wl >= BAND_EVAL[0]) & (wl <= BAND_EVAL[1])
        # 校验 1:1（同样点集，顺序一致）
        assert len(xyz) == len(refl), (fname, len(xyz), len(refl))
        d = np.linalg.norm(xyz - z["xyz"], axis=1).max()
        assert d < 1e-4, f"{fname} 与缓存不同源 max={d}"  # 仅 6 位小数舍入差
        src[deg] = {"xyz": xyz, "so": unit(refl[:, mo]), "se": unit(refl[:, me])}
    return src, wl


# ----------------------------- M_cripped -----------------------------
def get_cripped_transforms() -> dict:
    CMP.mkdir(exist_ok=True)
    cache_f = CMP / "transforms_cripped.json"
    if cache_f.exists():
        return {int(k): np.array(v) for k, v in json.loads(cache_f.read_text()).items()}
    o3d = EC.o3d
    o3d.utility.random.seed(0)
    base = EC.load_pcd(CRIPPED / FRAMES[0][0], 0.003)
    f_base = EC.fpfh(base)
    out, times = {}, {}
    for fname, deg in FRAMES[1:]:
        t0 = time.time()
        src = EC.load_pcd(CRIPPED / fname, 0.003)
        f_src = EC.fpfh(src)
        T0, _ = EC.global_reg(src, base, f_src, f_base, 0.003)
        T1 = EC.refine_icp(src, base, T0).transformation
        out[deg] = T1.tolist()
        times[deg] = time.time() - t0
        print(f"    M_cripped {deg:>3}° rot={np.degrees(np.arccos(np.clip((np.trace(T1[:3,:3])-1)/2,-1,1))):7.2f}°"
              f"  {times[deg]:.1f}s")
    cache_f.write_text(json.dumps(out, indent=2))
    (CMP / "cripped_runtime.json").write_text(json.dumps(times, indent=2))
    return {int(k): np.array(v) for k, v in out.items()}


# ----------------------------- 统一指标 -----------------------------
def evaluate(src_xyz, src_so, src_se, tgt_xyz, tgt_so, tgt_se, T, tau, mutual=False):
    p = (T[:3, :3] @ src_xyz.T).T + T[:3, 3]
    d, j = cKDTree(tgt_xyz).query(p, distance_upper_bound=tau)
    ok = np.isfinite(d)
    if mutual and ok.sum():
        idx = np.arange(len(p))[ok]
        _, i2 = cKDTree(p).query(tgt_xyz[j[ok]], distance_upper_bound=tau)
        keep = i2 == idx
        m = np.zeros(len(p), bool)
        m[idx[keep]] = True
        ok = m
    n = int(ok.sum())
    if n == 0:
        return {"n_pairs": 0, "fitness": 0.0, "rmse_mm": np.nan,
                "sam_opt_deg": np.nan, "sam_eval_deg": np.nan}
    jj = j[ok]
    return {"n_pairs": n,
            "fitness": float(ok.mean()),
            "rmse_mm": float(np.sqrt((d[ok] ** 2).mean()) * 1000),
            "sam_opt_deg": float(sam_deg(src_so[ok], tgt_so[jj]).mean()),
            "sam_eval_deg": float(sam_deg(src_se[ok], tgt_se[jj]).mean())}


def rot_shift(T):
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    return float(ang), float(np.linalg.norm(T[:3, 3]) * 1000)


def yaw_deg(T):
    """绕竖直轴 z 的偏航角分量（度）。

    坐标约定（见 convert_to_txt.point_coordinates）：
      y = d·cos(alt)·cos(hor)  → 扫描器视线方向
      z = d·sin(alt)           → **竖直向上**
    故水平面为 (x, y)，绕竖直轴 z 的纯旋转为 [[c,-s,0],[s,c,0],[0,0,1]]。
    对 2×2 水平块做投影，非纯偏航时也稳健。
    """
    R = T[:3, :3]
    return float(np.degrees(np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])))


def rot_axis(T):
    """旋转轴（单位向量，朝 +z 取向）与旋转角；角度过小时轴无定义"""
    R = T[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if ang < 1.0:
        return None, float(ang)
    w, v = np.linalg.eig(R)
    ax = np.real(v[:, int(np.argmin(np.abs(w - 1)))])
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    if ax[2] < 0:
        ax = -ax
    return ax, float(ang)


def axis_vs_vertical(T):
    """旋转轴与竖直 z 轴的夹角（°，0=完美偏航；旋转<1° 时返回 nan）"""
    ax, _ = rot_axis(T)
    return float("nan") if ax is None else float(np.degrees(np.arccos(np.clip(abs(ax[2]), 0, 1))))


def axis_xy(T):
    """转轴在水平 (x,y) 平面上的位置（m）；退化则 nan"""
    R, t = T[:3, :3], T[:3, 3]
    A = np.array([[1 - R[0, 0], -R[0, 1]], [-R[1, 0], 1 - R[1, 1]]])
    if abs(np.linalg.det(A)) < 1e-9:
        return np.array([np.nan, np.nan])
    return np.linalg.solve(A, np.array([t[0], t[1]]))


def merged_count(xyz_list, voxel=0.003):
    """合并后去重点数（覆盖/冗余指标的替代）"""
    pts = np.vstack(xyz_list)
    keys = np.floor(pts / voxel).astype(np.int64)
    return int(len(np.unique(keys, axis=0)))


# ----------------------------- 主流程 -----------------------------
def main() -> None:
    CMP.mkdir(exist_ok=True)
    print("=" * 100)
    print("三方法配准指标统一对比：M_cripped  vs  M_ref  vs  M_joint")
    print("=" * 100)
    src, wl = load_source()
    print(f"[源数据] 6 帧全分辨率 cripped 点云，点数 "
          f"{[len(src[d]['xyz']) for _, d in FRAMES]}；光谱 {len(wl)} 波段 "
          f"({wl[0]:.1f}–{wl[-1]:.1f} nm)")

    reg = np.load(CACHE / "registration.npz")
    print("\n[1/4] 取得三方法变换矩阵")
    print("  M_cripped: 重跑 experiment_cripped（FPFH+RANSAC 12 组参数 + point-to-plane ICP 4 阈值）")
    T_crip = get_cripped_transforms()
    FR = (60, 120, 180, 240, 300)
    T = {"M_cripped": T_crip,
         "M_ref": {d: reg[f"T_ref_{d}"] for d in FR},
         "M_joint": {d: reg[f"T_joint_{d}"] for d in FR},
         "identity": {d: np.eye(4) for d in FR}}   # 不配准基线
    EVAL = ("identity", "M_cripped", "M_ref", "M_joint")

    B, base_deg = src[0], 0
    results = {"source_points": [len(src[d]["xyz"]) for _, d in FRAMES]}

    # ---------- A. 几何指标 ----------
    print("\n[2/4] A. 几何指标（全分辨率、同一口径；identity=不配准基线）")
    geo = {m: {d: {} for d in FR} for m in EVAL}
    for m in EVAL:
        for deg in FR:
            s, Tt = src[deg], T[m][deg]
            for tau in THRESHOLDS:
                geo[m][deg][f"fit@{int(tau*1000)}mm"] = evaluate(
                    s["xyz"], s["so"], s["se"], B["xyz"], B["so"], B["se"], Tt, tau)["fitness"]
            for tau in (0.010, 0.020):
                geo[m][deg][f"rmse@{int(tau*1000)}mm"] = evaluate(
                    s["xyz"], s["so"], s["se"], B["xyz"], B["so"], B["se"], Tt, tau)["rmse_mm"]
            mu = evaluate(s["xyz"], s["so"], s["se"], B["xyz"], B["so"], B["se"], Tt, 0.020, mutual=True)
            geo[m][deg]["fit_mutual@20mm"] = mu["fitness"]
            geo[m][deg]["rmse_mutual@20mm"] = mu["rmse_mm"]
            geo[m][deg]["n_pairs_mutual"] = mu["n_pairs"]

    keys = ["fit@5mm", "fit@10mm", "fit@20mm", "fit@30mm",
            "rmse@10mm", "rmse@20mm", "fit_mutual@20mm"]
    print(f"{'方法':>11} " + " ".join(f"{k:>15}" for k in keys))
    for m in EVAL:
        vals = [np.mean([geo[m][d][k] for d in FR]) for k in keys]
        tag = "  ← 基线" if m == "identity" else ""
        print(f"{m:>11} " + " ".join(f"{v:>15.4f}" for v in vals) + tag)
    print("  ※ fit@5mm 是本数据唯一未被饱和的几何指标；三个方法在 5mm 上均与 identity 同量级(≈0.46–0.51)")
    print("  ※ fit@20mm / rmse@20mm 已饱和（对象仅 ≈200mm，阈值占其 10%）——赢它属'优化了该指标'，有循环性")
    results["vs_identity"] = {m: {k: float(np.mean([geo[m][d][k] for d in FR])
                                        - np.mean([geo["identity"][d][k] for d in FR]))
                                 for k in keys} for m in METHODS}
    for m in METHODS:
        d5 = results["vs_identity"][m]["fit@5mm"]
        print(f"  → {m:>11} 相对 identity 的 Δfit@5mm = {d5:+.4f}  "
              f"({'优于' if d5 > 0 else '**劣于**'}不配准)")

    # ---------- B. 光谱指标 ----------
    print("\n[3/4] B/C/D. 光谱 / 物理先验 / 代价")
    spec = {m: {d: {} for d in FR} for m in EVAL}
    for m in EVAL:
        for deg in FR:
            s, Tt = src[deg], T[m][deg]
            r20 = evaluate(s["xyz"], s["so"], s["se"], B["xyz"], B["so"], B["se"], Tt, 0.020)
            r10 = evaluate(s["xyz"], s["so"], s["se"], B["xyz"], B["so"], B["se"], Tt, 0.010)
            rm = evaluate(s["xyz"], s["so"], s["se"], B["xyz"], B["so"], B["se"], Tt, 0.020, mutual=True)
            spec[m][deg] = {"sam_opt@20mm": r20["sam_opt_deg"], "sam_eval@20mm": r20["sam_eval_deg"],
                            "sam_opt@10mm": r10["sam_opt_deg"], "sam_opt_mutual@20mm": rm["sam_opt_deg"]}

    skeys = ["sam_opt@20mm", "sam_opt@10mm", "sam_opt_mutual@20mm", "sam_eval@20mm"]
    print(f"{'方法':>11} " + " ".join(f"{k:>22}" for k in skeys))
    for m in EVAL:
        vals = [np.mean([spec[m][d][k] for d in FR]) for k in skeys]
        tag = "  ← 基线" if m == "identity" else ""
        print(f"{m:>11} " + " ".join(f"{v:>22.2f}" for v in vals) + tag)
    results["vs_identity_spec"] = {m: {k: float(np.mean([spec[m][d][k] for d in FR])
                                                 - np.mean([spec["identity"][d][k] for d in FR]))
                                        for k in skeys} for m in METHODS}

    # ---------- C. 物理先验 ----------
    print("\n  C. 物理先验（转台假设：帧间应为绕竖直轴的纯偏航，步长≈±60°，无需真值）")
    phys = {m: {} for m in EVAL}
    for m in EVAL:
        yaws = [0.0] + [yaw_deg(T[m][d]) for d in FR]
        axs = []
        for d in FR:
            a = axis_xy(T[m][d])
            if np.all(np.isfinite(a)):
                axs.append(a)
        axs = np.array(axs) if axs else np.zeros((0, 2))
        steps = np.diff(yaws)
        axis_viable = [axis_vs_vertical(T[m][d]) for d in FR]
        finite = [a for a in axis_viable if np.isfinite(a)]
        phys[m] = {
            "yaw_deg": [float(y) for y in yaws],
            "yaw_steps_deg": [float(s) for s in steps],
            "yaw_step_mean_deg": float(np.mean(steps)),
            "yaw_step_std_deg": float(np.std(steps)),
            "yaw_step_err_vs_60_deg": float(np.mean(np.abs(np.abs(steps) - 60.0))),
            "yaw_step_max_err_deg": float(np.max(np.abs(np.abs(steps) - 60.0))),
            "axis_vs_vertical_deg": axis_viable,
            "axis_vs_vertical_mean_deg": float(np.mean(finite)) if finite else None,
            "n_axis_defined": len(finite),
            "axis_xy_mean": [float(axs[:, 0].mean()), float(axs[:, 1].mean())] if len(axs) else None,
            "axis_xy_std_mm": float(np.linalg.norm(axs - axs.mean(0), axis=1).mean() * 1000) if len(axs) else None,
            "rot_shift": {str(d): rot_shift(T[m][d]) for d in FR},
        }
    print(f"{'方法':>11} {'yaw序列(°)':>48} {'|Δ|-60均值':>10} {'步长std':>8} {'转轴⊥竖直角':>12} {'平均旋转':>9}")
    for m in EVAL:
        p = phys[m]
        seq = " ".join(f"{y:7.1f}" for y in p["yaw_deg"])
        av = p["axis_vs_vertical_mean_deg"]
        avs = f"{av:.1f}°({p['n_axis_defined']}/5)" if av is not None else "n/a(无旋转)"
        rr = np.mean([p["rot_shift"][str(d)][0] for d in FR])
        print(f"{m:>11} {seq:>48} {p['yaw_step_err_vs_60_deg']:>10.1f} {p['yaw_step_std_deg']:>8.1f} "
              f"{avs:>12} {rr:>8.1f}°")
    print("  ※ 坐标约定：z 为竖直轴（z=d·sin(alt)），水平面为 (x,y)；理想转台：绕 z 纯偏航、步长 ±60°、转轴⊥竖直角 0°。")

    # ---------- D. 代价 ----------
    cost = {}
    for m in METHODS:
        xyzs = [B["xyz"]] + [(T[m][d][:3, :3] @ src[d]["xyz"].T).T + T[m][d][:3, 3] for d in FR]
        cost[m] = {"merged_voxel3mm": merged_count(xyzs),
                   "avg_rot_deg": float(np.mean([rot_shift(T[m][d])[0] for d in FR])),
                   "avg_shift_mm": float(np.mean([rot_shift(T[m][d])[1] for d in FR]))}
    cost["M_cripped"]["ransac_param_combos"] = 12
    cost["M_ref"]["ransac_param_combos"] = 1
    cost["M_joint"]["ransac_param_combos"] = 3
    print("\n  D. 代价")
    for m in METHODS:
        print(f"{m:>11}  合并去重点数(3mm)={cost[m]['merged_voxel3mm']:>6}  "
              f"平均旋转={cost[m]['avg_rot_deg']:>7.1f}°  平均平移={cost[m]['avg_shift_mm']:>8.1f}mm")

    # ---------- 排名 ----------
    print("\n[4/4] 综合排名（同口径归一化后）")
    ranking = {}
    for k, lower_better in (("fit@5mm", False), ("fit@20mm", False), ("fit_mutual@20mm", False),
                            ("rmse@20mm", True), ("sam_opt@20mm", True),
                            ("sam_opt_mutual@20mm", True), ("sam_eval@20mm", True)):
        vals = {}
        for m in METHODS:
            vals[m] = float(np.mean([geo[m][d][k] for d in FR])) if k in geo[m][60] \
                else float(np.mean([spec[m][d][k] for d in FR]))
        order = sorted(vals, key=lambda m: vals[m], reverse=not lower_better)
        ranking[k] = {"values": vals, "rank": order}
        print(f"  {k:>22}: " + "  ".join(f"{m}={vals[m]:.4f}(#{i+1})" for i, m in enumerate(order)))
    for k, lower_better in (("yaw_step_std_deg", True), ("yaw_step_err_vs_60_deg", True)):
        vals = {m: phys[m][k] for m in METHODS}
        order = sorted(vals, key=lambda m: vals[m], reverse=not lower_better)
        ranking[k] = {"values": vals, "rank": order}
        print(f"  {k:>22}: " + "  ".join(f"{m}={vals[m]:.2f}(#{i+1})" for i, m in enumerate(order)))

    # 综合得分（平均名次）
    score = {m: [] for m in METHODS}
    for k, r in ranking.items():
        for i, m in enumerate(r["rank"]):
            score[m].append(i + 1)
    print("\n  === 综合（各指标名次平均，越小越好）===")
    final = sorted(METHODS, key=lambda m: np.mean(score[m]))
    for m in final:
        print(f"    {m:>11}: 平均名次 {np.mean(score[m]):.2f}  名次分布 {score[m]}")
    print(f"\n  >>> 综合最优：{final[0]}")

    results.update({"geometry": geo, "spectral": spec, "physics": phys, "cost": cost,
                    "ranking": ranking,
                    "final_score": {m: {"avg_rank": float(np.mean(score[m])), "ranks": score[m]}
                                    for m in METHODS}, "winner": final[0]})
    (CMP / "comparison.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[保存] compare/comparison.json")
    plot(results)


def plot(R) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods, cols = ["M_cripped", "M_ref", "M_joint"], ["#4C72B0", "#C44E52", "#55A868"]
    x = np.arange(len(methods))
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))

    specs = [
        ("fit@5mm", "① fitness@5mm ↑（唯一未饱和的几何指标）", "geometry", False),
        ("fit@20mm", "② fitness@20mm ↑（已饱和，有循环性）", "geometry", False),
        ("rmse@20mm", "③ rmse@20mm (mm) ↓", "geometry", True),
        ("sam_opt@20mm", "④ SAM(500-800)° ↓", "spectral", True),
        ("sam_opt_mutual@20mm", "⑤ SAM 互最近邻° ↓", "spectral", True),
    ]
    for a, (k, title, grp, low) in zip(ax.flat, specs):
        v = [np.mean([R[grp][m][d][k] for d in (60, 120, 180, 240, 300)]) for m in methods]
        vi = np.mean([R[grp]["identity"][d][k] for d in (60, 120, 180, 240, 300)])
        order = np.argsort(v) if low else np.argsort(v)[::-1]
        bars = a.bar(x, v, color=[cols[i] for i in range(3)])
        for i in order:
            bars[i].set_edgecolor("gold"); bars[i].set_linewidth(4)
        a.axhline(vi, color="gray", ls="--", lw=1.5, label="identity(不配准)")
        a.set_xticks(x); a.set_xticklabels(methods, rotation=10)
        a.set_title(title); a.grid(alpha=0.3, axis="y"); a.legend(fontsize=8)
        for i, b in enumerate(bars):
            a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v[i]:.4f}" if v[i] < 2 else f"{v[i]:.1f}",
                   ha="center", va="bottom", fontsize=8)

    a = ax.flat[5]
    for m, c in zip(methods, cols):
        a.plot([0, 60, 120, 180, 240, 300], R["physics"][m]["yaw_deg"], "o-", color=c, label=m)
    a.set_xlabel("名义角度 (°)"); a.set_ylabel("估计 yaw (°)")
    a.set_title("⑥ 转台物理先验：yaw 序列应≈名义角")
    a.legend(); a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CMP / "comparison.png", dpi=120)
    print("[保存] compare/comparison.png")


if __name__ == "__main__":
    main()
