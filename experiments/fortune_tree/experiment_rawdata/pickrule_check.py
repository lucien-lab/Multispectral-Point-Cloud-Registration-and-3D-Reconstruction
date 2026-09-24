# -*- coding: utf-8 -*-
"""候选选取规则的影响（几何优先 vs 光谱优先）——检查光谱臂的低 SAM 是否来自「少转」

现有管线的择优规则 R1：候选需满足 fit@20mm ≥ 0.90，再取 SAM 最小。
由于恒等变换本身 fit@20mm=0.9029、SAM=8.01°（全表最低），R1 可能选中「几乎不旋转」的解。
本脚本对同一批候选位姿再套一条几何优先规则 R2（取 fit@5mm 最大），比较两种规则的输出。

仅重跑 A/B/C 三臂（5 帧），候选级全分辨率评价。
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
FRAMES = [60, 120, 180, 240, 300]
TAU = 0.020


def unit_rows(w):
    w = np.asarray(w, float)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def clip_refl(f):
    return np.clip(f["refl"].astype(np.float64), 0.0, None)


def desc_of(frames, idx, dims=E.PCA_DIMS):
    X = np.vstack([unit_rows(clip_refl(f)[:, idx])[:: max(1, len(f["xyz"]) // 5000)]
                   for f in frames.values()])
    d = int(min(dims, len(idx)))
    mu = X.mean(0)
    Xc = X - mu
    ev, evec = np.linalg.eigh((Xc.T @ Xc) / max(len(Xc) - 1, 1))
    o = np.argsort(ev)[::-1][:d]
    comps = evec[:, o]
    return {g: {"spec": unit_rows(clip_refl(f)[:, idx]),
                "pca": unit_rows((unit_rows(clip_refl(f)[:, idx]) - mu) @ comps)}
            for g, f in frames.items()}


def full_metrics(xyz_s, xyz_t, spec_s, spec_t, T):
    P = (T[:3, :3] @ xyz_s.T).T + T[:3, 3]
    kd = cKDTree(xyz_t)
    o = {}
    for t in (5, 10, 20):
        dd, _ = kd.query(P, distance_upper_bound=t / 1000.0)
        o[f"fit@{t}mm"] = float(np.isfinite(dd).mean())
    dd, jj = kd.query(P, distance_upper_bound=TAU)
    ok = np.isfinite(dd)
    o["rmse@20mm"] = float(np.sqrt((dd[ok] ** 2).mean()) * 1000)
    ii = np.arange(len(P))[ok]
    _, i2 = cKDTree(P).query(xyz_t[jj[ok]], distance_upper_bound=TAU)
    m = np.zeros(len(P), bool)
    m[ii[i2 == ii]] = True
    o["fit_mutual@20mm"] = float(m.mean())
    a = spec_s[ok] / (np.linalg.norm(spec_s[ok], axis=1, keepdims=True) + 1e-12)
    b = spec_t[jj[ok]] / (np.linalg.norm(spec_t[jj[ok]], axis=1, keepdims=True) + 1e-12)
    o["sam_self"] = float(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).mean())
    return o


def main():
    frames = {g: E.load_frame(g) for g in [0] + FRAMES}
    fb = frames[0]
    b_dn, b_map = E.downsample_with_map(fb["xyz"], normals=True)
    b_fpfh = E.fpfh(b_dn)
    b_np, b_tn = np.asarray(b_dn.points), np.asarray(b_dn.normals)
    g_t = E.unit_cols(np.asarray(b_fpfh.data)) / np.sqrt(np.asarray(b_fpfh.data).shape[0])
    out = {"arms": {}, "candidates": []}

    # ---- A 几何（无候选择优问题）----
    acc = []
    for g in FRAMES:
        f = frames[g]
        dn, mp = E.downsample_with_map(f["xyz"], normals=True)
        T0, _ = E.ransac_feature(dn, b_dn, E.fpfh(dn), b_fpfh)
        T = E.refine_icp(dn, b_dn, T0).transformation
        acc.append(full_metrics(f["xyz"], fb["xyz"], unit_rows(clip_refl(f))[:, B_MS],
                                unit_rows(clip_refl(fb))[:, B_MS], T))
    out["arms"]["A 几何（无光谱）"] = {k: float(np.nanmean([a[k] for a in acc]))
                                   for k in acc[0]}

    for arm, idx in (("B 几何+RGB", I_RGB), ("C 几何+多光谱", B_MS)):
        desc = desc_of({g: frames[g] for g in [0] + FRAMES}, idx)
        rows = []
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
                mm = full_metrics(f["xyz"], fb["xyz"], desc[g]["spec"], desc[0]["spec"], Tj)
                ang = float(np.degrees(np.arccos(np.clip((np.trace(Tj[:3, :3]) - 1) / 2, -1, 1))))
                cands.append({"init": cn, "rot": ang, **mm})
                out["candidates"].append({"arm": arm, "deg": g, "init": cn, "rot": ang, **mm})
            # R1：fit@20mm≥0.90 后取 SAM 最小（现有规则）
            okl = [c for c in cands if c["fit@20mm"] >= E.FITNESS_GATE]
            r1 = min(okl, key=lambda c: c["sam_self"]) if okl else max(cands, key=lambda c: c["fit@20mm"])
            # R2：几何优先，取 fit@5mm 最大
            r2 = max(cands, key=lambda c: c["fit@5mm"])
            rows.append({"deg": g, "R1": r1, "R2": r2})
        agg = {}
        for rule in ("R1", "R2"):
            agg[rule] = {k: float(np.nanmean([r[rule][k] for r in rows]))
                         for k in ("fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm",
                                   "fit_mutual@20mm", "sam_self", "rot")}
        agg["per_frame"] = [{k: v for k, v in r.items()} for r in rows]
        out["arms"][arm] = agg

    json.dump(out, open(HERE / "pickrule_check.json", "w"), indent=2, ensure_ascii=False)

    print("=" * 108)
    print("表 6  候选选取规则的影响（5 帧平均；R1 = 现有规则「fit@20mm≥0.90 后取 SAM 最小」）")
    print("      R2 = 几何优先「取 fit@5mm 最大」")
    print("=" * 108)
    print("%-34s %8s %8s %8s %9s %9s %8s %7s" % (
        "臂 / 选取规则", "fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm", "互最近邻", "SAM(°)", "旋转(°)"))
    a = out["arms"]["A 几何（无光谱）"]
    print("%-34s %8.4f %8.4f %8.4f %9.2f %9.4f %8.2f %7s" % (
        "A 几何（无光谱）", a["fit@5mm"], a["fit@10mm"], a["fit@20mm"], a["rmse@20mm"],
        a["fit_mutual@20mm"], a["sam_self"], "—"))
    for arm in ("B 几何+RGB", "C 几何+多光谱"):
        for rule, tag in (("R1", "（R1 光谱优先·现有）"), ("R2", "（R2 几何优先）")):
            r = out["arms"][arm][rule]
            print("%-34s %8.4f %8.4f %8.4f %9.2f %9.4f %8.2f %7.2f" % (
                arm + tag, r["fit@5mm"], r["fit@10mm"], r["fit@20mm"], r["rmse@20mm"],
                r["fit_mutual@20mm"], r["sam_self"], r["rot"]))
    print("\n逐帧（C 臂）：规则选择与旋转角")
    for r in out["arms"]["C 几何+多光谱"]["per_frame"]:
        print("  %3d°: R1→%s(rot=%.1f°) fit5=%.4f SAM=%.2f | R2→%s(rot=%.1f°) fit5=%.4f SAM=%.2f"
              % (r["deg"], r["R1"]["init"], r["R1"]["rot"], r["R1"]["fit@5mm"], r["R1"]["sam_self"],
                 r["R2"]["init"], r["R2"]["rot"], r["R2"]["fit@5mm"], r["R2"]["sam_self"]))
    print("\n[保存] pickrule_check.json")


if __name__ == "__main__":
    main()
