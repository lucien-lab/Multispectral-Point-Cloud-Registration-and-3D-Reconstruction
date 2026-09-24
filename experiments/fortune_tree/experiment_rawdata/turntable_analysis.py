#!/usr/bin/env python3
"""转台物理约束配准 —— 为三方法提供"模型基准真值"，裁决谁对

原理：若扫描器固定、植株在转台上绕竖直轴旋转，则所有帧共享同一坐标系的
      同一根竖直转轴，帧 k → 帧 0 的对齐只剩 **1 个自由度**（偏航角 φ_k）：

          T(φ) = 绕竖直线 {(ax, ay)} 转 φ

      于是可以直接对 φ 做 1D 全搜索，得到"物理最优角"（不受 ICP 局部极小影响），
      作为基准真值来评判 M_cripped / M_ref / M_joint 估计的角度是否正确。

同时检验：
  ① 三方法各自估出的转轴 (ax, ay) 是否一致（转台轴上各帧必须一致）
  ② 三方法的 φ_k 是否落在物理最优角附近

输出：turntable.json、visualization/turntable_sweep.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "cache"
CRIPPED = ROOT / "cripped"
FRAMES = [(0, "20260904三维重建发财树0度.txt"), (60, "20260904三维建模发财树60度.txt"),
          (120, "20260904三维建模发财树120度.txt"), (180, "20260904三维建模发财树180度.txt"),
          (240, "20260904三维建模发财树240度.txt"), (300, "20260904三维建模发财树300度.txt")]
DEGS = [60, 120, 180, 240, 300]
BAND_OPT = (500.0, 800.0)


def unit(w):
    w = np.asarray(w, float)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def sam_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def rot_about_vertical(phi_deg, ax, ay):
    """绕竖直线 {(ax, ay)} 旋转 phi（度）的 4×4 变换"""
    t = np.radians(phi_deg)
    c, s = np.cos(t), np.sin(t)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    a = np.array([ax, ay, 0.0])
    T[:3, 3] = a - R @ a
    return T


def eval_pair(src_xyz, src_s, tgt_xyz, tgt_s, T, tau):
    p = (T[:3, :3] @ src_xyz.T).T + T[:3, 3]
    kd = cKDTree(tgt_xyz)
    d, j = kd.query(p, distance_upper_bound=tau)
    ok = np.isfinite(d)
    n = int(ok.sum())
    fit = ok.mean()
    # 真 fitness@5mm：独立以 5mm 为上限查询（不能写成 (d<5mm).mean()，那是 P(d<5|d<tau) 条件比例）
    d5, _ = kd.query(p, distance_upper_bound=0.005)
    fit5 = float(np.isfinite(d5).mean())
    # 互为最近邻
    if n:
        idx = np.arange(len(p))[ok]
        _, i2 = cKDTree(p).query(tgt_xyz[j[ok]], distance_upper_bound=tau)
        keep = i2 == idx
        m = np.zeros(len(p), bool)
        m[idx[keep]] = True
        mut = m.mean()
        s = float(sam_deg(src_s[ok], tgt_s[j[ok]]).mean())
    else:
        mut, s = 0.0, float("nan")
    rmse = float(np.sqrt((d[ok] ** 2).mean()) * 1000) if n else float("nan")
    return {"fit5": fit5, "fit_mutual": float(mut), "sam": s, "rmse20": rmse, "n": n}


def axis_xy(T):
    R, t = T[:3, :3], T[:3, 3]
    A = np.array([[1 - R[0, 0], -R[0, 1]], [-R[1, 0], 1 - R[1, 1]]])
    if abs(np.linalg.det(A)) < 1e-9:
        return np.array([np.nan, np.nan])
    return np.linalg.solve(A, np.array([t[0], t[1]]))


def yaw_deg(T):
    R = T[:3, :3]
    return float(np.degrees(np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])))


def main() -> None:
    reg = np.load(CACHE / "registration.npz")
    wl = np.load(CACHE / "frame_000.npz")["wavelength"]
    mo = (wl >= BAND_OPT[0]) & (wl <= BAND_OPT[1])
    S = {}
    for deg, fname in FRAMES:
        a = np.loadtxt(CRIPPED / fname, delimiter=",")
        z = np.load(CACHE / f"frame_{deg:03d}.npz")
        S[deg] = {"xyz": a[:, :3], "s": unit(np.clip(z["refl"].astype(float), 0, None)[:, mo])}
    B = S[0]

    cripped_T = {int(k): np.array(v) for k, v in
                 json.loads((HERE / "compare" / "transforms_cripped.json").read_text()).items()}
    T = {"M_cripped": cripped_T,
         "M_ref": {d: reg[f"T_ref_{d}"] for d in DEGS},
         "M_joint": {d: reg[f"T_joint_{d}"] for d in DEGS},
         "identity": {d: np.eye(4) for d in DEGS}}

    # ---------- ① 各方法估出的转轴是否一致（转台轴在扫描器坐标系中固定）----------
    print("=" * 100)
    print("① 转轴一致性：转台轴上各帧的 (ax, ay) 必须相同（同轴性检验，无需真值）")
    print("=" * 100)
    axes = {}
    for m in ("M_cripped", "M_ref", "M_joint"):
        ax = np.array([axis_xy(T[m][d]) for d in DEGS])
        good = ax[np.all(np.isfinite(ax), axis=1)]
        axes[m] = {"per_frame": ax.tolist(),
                   "median": np.median(good, axis=0).tolist() if len(good) else None,
                   "std_mm": float(np.linalg.norm(good - np.median(good, 0), axis=1).mean() * 1000)
                   if len(good) else None,
                   "span_mm": float(np.linalg.norm(good.max(0) - good.min(0)) * 1000) if len(good) else None}
        a = axes[m]
        print(f"{m:>10}: 轴心(x,y)=({a['median'][0]:+.4f}, {a['median'][1]:+.4f}) m  "
              f"各帧偏离中位 {a['std_mm']:>8.1f} mm   极差 {a['span_mm']:>8.1f} mm")
    print("  ※ 同轴性越好 = 越符合「固定转轴」的物理约束；identity 无旋转，轴不定义。")

    # 以 M_ref 与 M_cripped 的一致轴心作为转轴估计
    cands = [np.array(axes[m]["median"]) for m in ("M_cripped", "M_ref")]
    ax0, ay0 = np.mean(cands, axis=0)
    print(f"\n  取 M_cripped/M_ref 轴心中点作为转轴：({ax0:+.4f}, {ay0:+.4f}) m")
    print(f"  注意：轴心到植株质心的水平距离 = "
          f"{np.linalg.norm(np.array([ax0, ay0]) - B['xyz'][:, :2].mean(0)) * 1000:.0f} mm")

    # ---------- ② 1D 全搜索物理最优偏航角 ----------
    print("\n" + "=" * 100)
    print("② 1D 全搜索：绕该竖轴转 φ，找物理最优角（转台只有 1 个自由度）")
    print("=" * 100)
    phis = np.arange(-180, 180, 1.0)
    sweep, best = {}, {}
    for deg in DEGS:
        rows = [eval_pair(S[deg]["xyz"], S[deg]["s"], B["xyz"], B["s"],
                          rot_about_vertical(p, ax0, ay0), 0.02) for p in phis]
        sweep[deg] = rows
        best[deg] = {
            "fit5": int(np.argmax([r["fit5"] for r in rows])),
            "fit_mutual": int(np.argmax([r["fit_mutual"] for r in rows])),
            "sam": int(np.nanargmin([r["sam"] for r in rows]))}
    print(f"{'帧':>5} {'标称':>6} | {'最优φ(fit5)':>12} {'最优φ(互近邻)':>13} {'最优φ(SAM)':>12} | "
          f"{'M_cripped':>10} {'M_ref':>8} {'M_joint':>8} {'identity':>9}")
    print("-" * 100)
    tbl = {}
    for deg in DEGS:
        row = {k: float(phis[v]) for k, v in best[deg].items()}
        got = {m: yaw_deg(T[m][deg]) for m in ("M_cripped", "M_ref", "M_joint")}
        tbl[deg] = {"best_phi": row, "methods": got}
        print(f"{deg:>5} {deg:>6} | {row['fit5']:>12.1f} {row['fit_mutual']:>13.1f} {row['sam']:>12.1f} | "
              f"{got['M_cripped']:>10.1f} {got['M_ref']:>8.1f} {got['M_joint']:>8.1f} {0.0:>9.1f}")
    print("\n  ※ 若 ICP 无偏，M_ref/M_cripped 的 φ 应落在「最优φ(互近邻)」附近。")

    # ---------- ③ 与物理最优角的偏差 ----------
    print("\n" + "=" * 100)
    print("③ 与物理最优角的偏差（度，环绕折叠）")
    print("=" * 100)
    err = {}
    for ref, name in (("fit_mutual", "互最近邻口径"), ("fit5", "fit@5mm 口径")):
        err[ref] = {}
        for m in ("M_cripped", "M_ref", "M_joint", "identity"):
            ds = []
            for deg in DEGS:
                a, b = tbl[deg]["methods"].get(m, 0.0), tbl[deg]["best_phi"][ref]
                ds.append(abs((a - b + 180) % 360 - 180))
            err[ref][m] = {"mean_deg": float(np.mean(ds)), "values": [float(d) for d in ds]}
        print(f"  参考 = 最优φ({name}):")
        for m in ("M_cripped", "M_ref", "M_joint", "identity"):
            e = err[ref][m]
            print(f"    {m:>10} 平均偏差 {e['mean_deg']:>6.1f}°  " +
                  " ".join(f"{d:5.1f}" for d in e["values"]))

    # ---------- ④ 各方法在「物理最优角」处的可达指标（上界）----------
    print("\n" + "=" * 100)
    print("④ 若强行用物理最优角（1 自由度）能达到的指标 —— 这是本数据的上界")
    print("=" * 100)
    upper = {}
    for deg in DEGS:
        r = sweep[deg][best[deg]["fit_mutual"]]
        upper[deg] = {k: r[k] for k in ("fit5", "fit_mutual", "sam", "rmse20")}
    print(f"{'指标':>14} {'物理最优(1自由度)':>18} {'M_cripped':>12} {'M_ref':>10} {'M_joint':>10} {'identity':>10}")
    for k, lo in (("fit5", False), ("fit_mutual", False), ("sam", True), ("rmse20", True)):
        u = np.mean([upper[d][k] for d in DEGS])
        vals = {}
        for m in ("M_cripped", "M_ref", "M_joint", "identity"):
            vs = []
            for deg in DEGS:
                vs.append(eval_pair(S[deg]["xyz"], S[deg]["s"], B["xyz"], B["s"], T[m][deg], 0.02)[k])
            vals[m] = float(np.mean(vs))
        print(f"{k:>14} {u:>18.4f} " + " ".join(f"{vals[m]:>10.4f}" for m in
                                                 ("M_cripped", "M_ref", "M_joint", "identity")))
        upper[f"mean_{k}"] = u
        for m in vals:
            upper[f"{m}_{k}"] = vals[m]

    out = {"axis": axes, "axis_used": [float(ax0), float(ay0)], "per_frame": tbl,
           "errors": err, "upper_bound": upper,
           "sweep": {str(d): {"phi": phis.tolist(),
                              "fit5": [r["fit5"] for r in sweep[d]],
                              "fit_mutual": [r["fit_mutual"] for r in sweep[d]],
                              "sam": [r["sam"] for r in sweep[d]]} for d in DEGS}}
    (HERE / "turntable.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("\n[保存] turntable.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, deg in enumerate(DEGS):
        axs[0].plot(phis, [r["fit5"] for r in sweep[deg]], color=cmap(i), label=f"{deg}°")
        axs[1].plot(phis, [r["fit_mutual"] for r in sweep[deg]], color=cmap(i), label=f"{deg}°")
        axs[2].plot(phis, [r["sam"] for r in sweep[deg]], color=cmap(i), label=f"{deg}°")
        for a, k in zip(axs, ("fit5", "fit_mutual", "sam")):
            a.axvline(phis[best[deg][k]], color=cmap(i), ls=":", lw=0.8, alpha=0.7)
    for a, t in zip(axs, ("fitness@5mm ↑", "互最近邻 fitness ↑", "SAM(500-800)° ↓")):
        a.set_xlabel("绕竖轴的偏航角 φ (°)"); a.set_title(t); a.grid(alpha=0.3); a.legend(fontsize=8)
    fig.suptitle("转台物理约束（1 自由度）下的偏航角全搜索：虚线=各帧最优角")
    fig.tight_layout()
    fig.savefig(HERE / "visualization" / "turntable_sweep.png", dpi=120)
    print("[保存] visualization/turntable_sweep.png")


if __name__ == "__main__":
    main()
