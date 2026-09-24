# -*- coding: utf-8 -*-
"""修正版择优规则检验 + 全域指标扫描：究竟哪些指标体现「多光谱+几何」最好

修正项：上一版表 6 中 B 臂的 SAM 列用的是 RGB 自身波段，而 A/C 用的是 500–800 nm，
       跨臂比较不成立。本脚本对每条臂选定的位姿，统一在 500–800 nm 与 450–900 nm 上计算 SAM。

评价口径：全部点 / 植物点(NDVI≥0.5) / 远离参考旋转轴(r_ax>20 mm)；
指标：fit@5/10/20 mm、rmse@20 mm、互为最近邻@20 mm、SAM(500–800)、SAM(450–900)。
择优规则：R1 = fit@20mm≥0.90 后取 SAM(自身波段) 最小（现有）；R2 = 取 fit@5mm 最大（几何优先）。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "cache"
_s = importlib.util.spec_from_file_location("expraw", ROOT / "experiment_rawdata.py")
E = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E)
E.CACHE = CACHE

WL = E.load_frame(0)["wavelength"]
I_RGB = [b - 1 for b in E.RGB_BANDS_1BASED]
B_MS = np.where((WL >= 500) & (WL <= 800))[0]
B_EV = np.where((WL >= 450) & (WL <= 900))[0]
I650 = int(np.argmin(np.abs(WL - 650.0)))
I800 = int(np.argmin(np.abs(WL - 800.0)))
AXIS = np.array([0.0009, 1.2505])       # 参考旋转轴水平坐标（m）
FRAMES = [60, 120, 180, 240, 300]
TAU = 0.020


def unit_rows(w):
    w = np.asarray(w, float)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def clip(f):
    return np.clip(f["refl"].astype(np.float64), 0.0, None)


def desc_of(frames, idx, dims=E.PCA_DIMS):
    X = np.vstack([unit_rows(clip(f)[:, idx])[:: max(1, len(f["xyz"]) // 5000)]
                   for f in frames.values()])
    d = int(min(dims, len(idx)))
    mu = X.mean(0)
    ev, evec = np.linalg.eigh(((X - mu).T @ (X - mu)) / max(len(X) - 1, 1))
    o = np.argsort(ev)[::-1][:d]
    return {g: {"spec": unit_rows(clip(f)[:, idx]),
                "pca": unit_rows((unit_rows(clip(f)[:, idx]) - mu) @ evec[:, o])}
            for g, f in frames.items()}


def masks(f):
    r = clip(f)
    ndvi = (r[:, I800] - r[:, I650]) / np.maximum(r[:, I800] + r[:, I650], 1e-9)
    rax = np.hypot(f["xyz"][:, 0] - AXIS[0], f["xyz"][:, 1] - AXIS[1])
    return {"全部点": np.ones(len(f["xyz"]), bool),
            "植物点(NDVI≥0.5)": ndvi >= 0.5,
            "远离轴(r>20mm)": rax > 0.020}


def eval_mask(xyz_s, xyz_t, T, ms, mt, spec_ms, spec_mt, b_ev):
    P = (T[:3, :3] @ xyz_s[ms].T).T + T[:3, 3]
    q = xyz_t[mt]
    kd = cKDTree(q)
    o = {}
    for t in (5, 10, 20):
        d, _ = kd.query(P, distance_upper_bound=t / 1000.0)
        o[f"fit@{t}mm"] = float(np.isfinite(d).mean())
    d, j = kd.query(P, distance_upper_bound=TAU)
    ok = np.isfinite(d)
    o["rmse@20mm"] = float(np.sqrt((d[ok] ** 2).mean()) * 1000) if ok.any() else float("nan")
    ii = np.arange(len(P))[ok]
    _, i2 = cKDTree(P).query(q[j[ok]], distance_upper_bound=TAU)
    m = np.zeros(len(P), bool)
    m[ii[i2 == ii]] = True
    o["fit_mutual@20mm"] = float(m.mean())
    for tag, bb in (("sam500_800", B_MS), ("sam450_900", b_ev)):
        a1 = unit_rows(spec_ms[ms][:, bb])[ok]
        a2 = unit_rows(spec_mt[mt][:, bb])[j[ok]]
        o[tag] = float(np.degrees(np.arccos(np.clip((a1 * a2).sum(1), -1, 1))).mean())
    return o


def main():
    frames = {g: E.load_frame(g) for g in [0] + FRAMES}
    fb = frames[0]
    MK = {g: masks(frames[g]) for g in [0] + FRAMES}
    b_dn, b_map = E.downsample_with_map(fb["xyz"], normals=True)
    b_fpfh = E.fpfh(b_dn)
    b_np, b_tn = np.asarray(b_dn.points), np.asarray(b_dn.normals)
    g_t = E.unit_cols(np.asarray(b_fpfh.data)) / np.sqrt(np.asarray(b_fpfh.data).shape[0])

    poses = {"A 几何": {}, "B 几何+RGB": {}, "C 几何+多光谱": {}}
    for g in FRAMES:
        f = frames[g]
        dn, mp = E.downsample_with_map(f["xyz"], normals=True)
        T0, _ = E.ransac_feature(dn, b_dn, E.fpfh(dn), b_fpfh)
        poses["A 几何"][g] = {"R1": E.refine_icp(dn, b_dn, T0).transformation, "R2": None}
        poses["A 几何"][g]["R2"] = poses["A 几何"][g]["R1"]
    for arm, idx in (("B 几何+RGB", I_RGB), ("C 几何+多光谱", B_MS)):
        desc = desc_of({g: frames[g] for g in [0] + FRAMES}, idx)
        for g in FRAMES:
            f = frames[g]
            dn, mp = E.downsample_with_map(f["xyz"], normals=True)
            fp = E.fpfh(dn)
            g_s = E.unit_cols(np.asarray(fp.data)) / np.sqrt(np.asarray(fp.data).shape[0])
            s_s = desc[g]["pca"][mp].T / np.sqrt(desc[g]["pca"].shape[1])
            s_t = desc[0]["pca"][b_map].T / np.sqrt(desc[0]["pca"].shape[1])
            Tr, _ = E.ransac_feature(dn, b_dn, E.make_feature(np.vstack([g_s, s_s])),
                                     E.make_feature(np.vstack([g_t, s_t])))
            cands = []
            for cn, Ti in (("融合 RANSAC", Tr), ("单位阵", np.eye(4))):
                Tj = E.joint_icp(np.asarray(dn.points), desc[g]["spec"][mp],
                                 b_np, b_tn, desc[0]["spec"][b_map], Ti)
                mm = eval_mask(f["xyz"], fb["xyz"], Tj, MK[g]["全部点"], MK[0]["全部点"],
                               unit_rows(clip(f)), unit_rows(clip(fb)), B_EV)
                ang = float(np.degrees(np.arccos(np.clip((np.trace(Tj[:3, :3]) - 1) / 2, -1, 1))))
                cands.append({"init": cn, "T": Tj, "rot": ang, **mm})
            okl = [c for c in cands if c["fit@20mm"] >= E.FITNESS_GATE]
            r1 = min(okl, key=lambda c: c["sam500_800"]) if okl else max(cands, key=lambda c: c["fit@20mm"])
            r2 = max(cands, key=lambda c: c["fit@5mm"])
            poses[arm][g] = {"R1": r1["T"], "R2": r2["T"]}

    # ---- 全域评价 ----
    out = {"cells": []}
    for arm, idx in (("A 几何", None), ("B 几何+RGB", I_RGB), ("C 几何+多光谱", B_MS)):
        # 评价统一用全波段反射率（仅按统一波段集取列），优化用各自描述子，二者分离
        spec = lambda g: unit_rows(clip(frames[g]))
        for rule in ("R1", "R2"):
            for mk in ("全部点", "植物点(NDVI≥0.5)", "远离轴(r>20mm)"):
                acc = []
                for g in FRAMES:
                    m = eval_mask(frames[g]["xyz"], fb["xyz"], poses[arm][g][rule],
                                  MK[g][mk], MK[0][mk], spec(g), spec(0), B_EV)
                    acc.append(m)
                agg = {k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]}
                out["cells"].append({"arm": arm, "rule": rule, "mask": mk, **agg})

    json.dump(out, open(HERE / "which_metric_best.json", "w"), indent=2, ensure_ascii=False,
              default=float)

    C = out["cells"]
    def get(arm, rule, mk, k):
        return [c[k] for c in C if c["arm"] == arm and c["rule"] == rule and c["mask"] == mk][0]
    print("=" * 122)
    print("统一口径指标扫描（5 帧平均）——哪一个指标体现「几何+多光谱」最好？")
    print("=" * 122)
    for rule in ("R1", "R2"):
        for mk in ("全部点", "植物点(NDVI≥0.5)", "远离轴(r>20mm)"):
            print("\n【规则 %s ／ 点集口径：%s】" % (rule, mk))
            print("%-16s %8s %8s %8s %9s %9s %10s %10s" % (
                "臂", "fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm", "互最近邻",
                "SAM(500–800)", "SAM(450–900)"))
            for arm in ("A 几何", "B 几何+RGB", "C 几何+多光谱"):
                print("%-16s %8.4f %8.4f %8.4f %9.2f %9.4f %10.2f %10.2f" % (
                    arm, get(arm, rule, mk, "fit@5mm"), get(arm, rule, mk, "fit@10mm"),
                    get(arm, rule, mk, "fit@20mm"), get(arm, rule, mk, "rmse@20mm"),
                    get(arm, rule, mk, "fit_mutual@20mm"), get(arm, rule, mk, "sam500_800"),
                    get(arm, rule, mk, "sam450_900")))

    print("\n" + "=" * 122)
    print("汇总：C（几何+多光谱）在每条指标上是否为三臂最优")
    print("=" * 122)
    for mk in ("全部点", "植物点(NDVI≥0.5)", "远离轴(r>20mm)"):
        for k, hi in (("fit@5mm", True), ("fit@10mm", True), ("fit@20mm", True),
                      ("rmse@20mm", False), ("fit_mutual@20mm", True),
                      ("sam500_800", False), ("sam450_900", False)):
            win = []
            for rule in ("R1", "R2"):
                v = {a: get(a, rule, mk, k) for a in ("A 几何", "B 几何+RGB", "C 几何+多光谱")}
                best = max(v, key=v.get) if hi else min(v, key=v.get)
                win.append("C" if best == "C 几何+多光谱" else best[0])
            print("  %-16s %-18s R1最优=%-2s R2最优=%-2s %s" % (
                mk, k, win[0], win[1], "← C 两规则均最优" if win == ["C", "C"] else ""))
    print("\n[保存] which_metric_best.json")


if __name__ == "__main__":
    main()
