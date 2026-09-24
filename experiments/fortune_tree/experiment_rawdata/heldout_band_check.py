# -*- coding: utf-8 -*-
"""留出波段（held-out bands）交叉验证：消除「光谱角当损失函数」的循环论证

问题：C 臂的目标函数含 λ·(SAM/Δ)²，因此用 SAM 评价 C 是自证（循环）。
办法：让**评价用的波段不出现在任何一条臂的优化波段中**。
  A 几何   ：不使用任何波段
  B 几何+RGB：优化用 RGB（490.4/560.3/650.1 nm）
  C 几何+多光谱：优化用 500–800 nm
  ⇒ 评价波段取 **450–495 nm 与 805–900 nm**（与三者优化波段均不相交）
另外给出口径对照：各臂在「自己拥有的优化波段」上的自评 SAM（不可横向比较）。

择优规则 R1（光谱优先）与 R2（几何优先）都报，便于对照循环性来源。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_s = importlib.util.spec_from_file_location("expraw", ROOT / "experiment_rawdata.py")
E = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E)
E.CACHE = HERE / "cache"

WL = E.load_frame(0)["wavelength"]
I_RGB = [b - 1 for b in E.RGB_BANDS_1BASED]
B_OPT_MS = np.where((WL >= 500) & (WL <= 800))[0]      # C 的优化波段
B_HOLD1 = np.where((WL >= 450) & (WL <= 495))[0]       # 留出口径 A（蓝紫）
B_HOLD2 = np.where((WL >= 805) & (WL <= 900))[0]       # 留出口径 B（近红外）
B_HOLD = np.concatenate([B_HOLD1, B_HOLD2])
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


def metrics(xyz_s, xyz_t, T, spec_full_s, spec_full_t, bands):
    P = (T[:3, :3] @ xyz_s.T).T + T[:3, 3]
    kd = cKDTree(xyz_t)
    o = {}
    for t in (5, 10, 20):
        d, _ = kd.query(P, distance_upper_bound=t / 1000.0)
        o[f"fit@{t}mm"] = float(np.isfinite(d).mean())
    d, j = kd.query(P, distance_upper_bound=TAU)
    ok = np.isfinite(d)
    o["rmse@20mm"] = float(np.sqrt((d[ok] ** 2).mean()) * 1000)
    for tag, bb in bands.items():
        if len(bb) == 0:
            o[tag] = float("nan")
            continue
        a = unit_rows(spec_full_s[:, bb])[ok]
        b = unit_rows(spec_full_t[:, bb])[j[ok]]
        o[tag] = float(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).mean())
    return o


def main():
    frames = {g: E.load_frame(g) for g in [0] + FRAMES}
    fb = frames[0]
    full = {g: unit_rows(clip(frames[g])) for g in [0] + FRAMES}   # 全波段，仅用于评价
    b_dn, b_map = E.downsample_with_map(fb["xyz"], normals=True)
    b_fpfh = E.fpfh(b_dn)
    b_np, b_tn = np.asarray(b_dn.points), np.asarray(b_dn.normals)
    g_t = E.unit_cols(np.asarray(b_fpfh.data)) / np.sqrt(np.asarray(b_fpfh.data).shape[0])

    BANDS = {"SAM_opt(500-800)": B_OPT_MS, "SAM_hold(all)": B_HOLD,
             "SAM_hold(450-495)": B_HOLD1, "SAM_hold(805-900)": B_HOLD2}
    arms = [("A 几何", None), ("B 几何+RGB", I_RGB), ("C 几何+多光谱", B_OPT_MS)]
    out = {}
    for arm, idx in arms:
        d = desc_of({g: frames[g] for g in [0] + FRAMES}, idx) if idx is not None else None
        acc = {"R1": [], "R2": []}
        for g in FRAMES:
            f = frames[g]
            dn, mp = E.downsample_with_map(f["xyz"], normals=True)
            fp = E.fpfh(dn)
            if idx is None:
                T0, _ = E.ransac_feature(dn, b_dn, fp, b_fpfh)
                T = E.refine_icp(dn, b_dn, T0).transformation
                m = metrics(f["xyz"], fb["xyz"], T, full[g], full[0], BANDS)
                m["SAM_self"] = float("nan")
                m["rot"] = float(np.degrees(np.arccos(np.clip(
                    (np.trace(T[:3, :3]) - 1) / 2, -1, 1))))
                acc["R1"].append(m); acc["R2"].append(m)
                continue
            gs = E.unit_cols(np.asarray(fp.data)) / np.sqrt(np.asarray(fp.data).shape[0])
            ss = d[g]["pca"][mp].T / np.sqrt(d[g]["pca"].shape[1])
            st = d[0]["pca"][b_map].T / np.sqrt(d[0]["pca"].shape[1])
            Tr, _ = E.ransac_feature(dn, b_dn, E.make_feature(np.vstack([gs, ss])),
                                     E.make_feature(np.vstack([g_t, st])))
            cands = []
            for cn, Ti in (("融合 RANSAC", Tr), ("单位阵", np.eye(4))):
                Tj = E.joint_icp(np.asarray(dn.points), d[g]["spec"][mp], b_np, b_tn,
                                 d[0]["spec"][b_map], Ti)
                m = metrics(f["xyz"], fb["xyz"], Tj, full[g], full[0], BANDS)
                m["init"] = cn
                rot = float(np.degrees(np.arccos(np.clip((np.trace(Tj[:3, :3]) - 1) / 2, -1, 1))))
                m["rot"] = rot
                # 该臂"自评"：在它自己的优化波段上算 SAM（仅作对照）
                if idx is not None:
                    aa = d[g]["spec"][d[g]["spec"].shape[1] and 0]  # 占位，下面重算
                    P = (Tj[:3, :3] @ f["xyz"].T).T + Tj[:3, 3]
                    dd, jj = cKDTree(fb["xyz"]).query(P, distance_upper_bound=TAU)
                    ok = np.isfinite(dd)
                    A = d[g]["spec"][ok]
                    Bm = d[0]["spec"][jj[ok]]
                    m["SAM_self"] = float(np.degrees(np.arccos(np.clip(
                        (unit_rows(A) * unit_rows(Bm)).sum(1), -1, 1))).mean())
                else:
                    m["SAM_self"] = float("nan")
                cands.append(m)
            okl = [c for c in cands if c["fit@20mm"] >= E.FITNESS_GATE]
            acc["R1"].append(min(okl, key=lambda c: c["SAM_self"]) if okl
                             else max(cands, key=lambda c: c["fit@20mm"]))
            acc["R2"].append(max(cands, key=lambda c: c["fit@5mm"]))
        def safe(vals):
            a = np.array(vals, float)
            return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")
        out[arm] = {r: {k: safe([m[k] for m in v])
                        for k in v[0] if isinstance(v[0][k], (int, float))}
                    for r, v in acc.items()}
        out[arm]["per_frame"] = {r: [{k: float(m[k]) for k in
                                      ("fit@5mm", "SAM_self", "SAM_hold(all)", "rot")}
                                     for m in v] for r, v in acc.items()}

    json.dump(out, open(HERE / "heldout_band_check.json", "w"), indent=2, ensure_ascii=False)
    print("=" * 112)
    print("留出波段检验（评价波段 450–495 ∪ 805–900 nm，与 A/B/C 的优化波段均不相交）")
    print(f"  优化波段：A 无 | B RGB 3 通道 | C 500–800 nm（{len(B_OPT_MS)} 维）")
    print(f"  留出波段：450–495 nm {len(B_HOLD1)} 维 + 805–900 nm {len(B_HOLD2)} 维 = {len(B_HOLD)} 维")
    print("=" * 112)
    for rule in ("R1", "R2"):
        print("\n【规则 %s】" % ("R1 光谱优先（SAM 参与选解 → 对 C 自证）" if rule == "R1"
                                else "R2 几何优先（SAM 不参与选解 → 公平）"))
        print("%-16s %8s %9s %11s %12s %13s %8s" % (
            "臂", "fit@5mm", "SAM_self", "SAM_hold(all)", "hold 450-495", "hold 805-900", "旋转(°)"))
        for arm, _ in arms:
            r = out[arm][rule]
            print("%-16s %8.4f %9.2f %11.2f %12.2f %13.2f %8.1f" % (
                arm, r["fit@5mm"], r["SAM_self"], r["SAM_hold(all)"],
                r["SAM_hold(450-495)"], r["SAM_hold(805-900)"], r["rot"]))
    print("\n逐帧（C 臂）：R2 规则下优化波段自评 vs 留出波段")
    for g, m in zip(FRAMES, out["C 几何+多光谱"]["per_frame"]["R2"]):
        print("  %3d°: fit5=%.4f SAM_self=%.2f SAM_hold=%.2f rot=%.1f°"
              % (g, m["fit@5mm"], m["SAM_self"], m["SAM_hold(all)"], m["rot"]))
    print("\n[保存] heldout_band_check.json")


if __name__ == "__main__":
    main()
