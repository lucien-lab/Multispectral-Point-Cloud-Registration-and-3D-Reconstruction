# -*- coding: utf-8 -*-
"""重复性检验：RANSAC 随机采样导致几何指标的运行间波动有多大

对 A（几何）臂重复 3 次、B（几何+RGB）臂重复 2 次，每次独立重跑 RANSAC + 细化，
统计 fit@5/10/20 mm 与 rmse@20 mm 的 5 帧平均在各次之间的波动范围。
用途：判断三臂之间 ≤0.01 的几何指标差异是否在噪声量级内。
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
B_MS = np.where((WL >= 500) & (WL <= 800))[0]
FRAMES = [60, 120, 180, 240, 300]
TAU = 0.020


def unit_rows(w):
    w = np.asarray(w, float)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def clip(f):
    return np.clip(f["refl"].astype(np.float64), 0.0, None)


def metrics(xyz_s, xyz_t, T):
    P = (T[:3, :3] @ xyz_s.T).T + T[:3, 3]
    kd = cKDTree(xyz_t)
    o = {}
    for t in (5, 10, 20):
        d, _ = kd.query(P, distance_upper_bound=t / 1000.0)
        o[f"fit@{t}mm"] = float(np.isfinite(d).mean())
    d, _ = kd.query(P, distance_upper_bound=TAU)
    ok = np.isfinite(d)
    o["rmse@20mm"] = float(np.sqrt((d[ok] ** 2).mean()) * 1000)
    return o


def main():
    frames = {g: E.load_frame(g) for g in [0] + FRAMES}
    fb = frames[0]
    b_dn, b_map = E.downsample_with_map(fb["xyz"], normals=True)
    b_fpfh = E.fpfh(b_dn)
    b_np, b_tn = np.asarray(b_dn.points), np.asarray(b_dn.normals)
    g_t = E.unit_cols(np.asarray(b_fpfh.data)) / np.sqrt(np.asarray(b_fpfh.data).shape[0])

    def desc(idx):
        X = np.vstack([unit_rows(clip(f)[:, idx])[:: max(1, len(f["xyz"]) // 5000)]
                       for f in frames.values()])
        mu = X.mean(0)
        ev, evec = np.linalg.eigh(((X - mu).T @ (X - mu)) / max(len(X) - 1, 1))
        o = np.argsort(ev)[::-1][:E.PCA_DIMS]
        return {g: {"spec": unit_rows(clip(f)[:, idx]),
                    "pca": unit_rows((unit_rows(clip(f)[:, idx]) - mu) @ evec[:, o])}
                for g, f in frames.items()}

    res = {}
    for arm, idx, reps in (("A 几何", None, 3), ("B 几何+RGB", I_RGB, 2)):
        d = desc(idx) if idx is not None else None
        reps_out = []
        for r in range(reps):
            acc = []
            for g in FRAMES:
                f = frames[g]
                dn, mp = E.downsample_with_map(f["xyz"], normals=True)
                fp = E.fpfh(dn)
                if idx is None:
                    T0, _ = E.ransac_feature(dn, b_dn, fp, b_fpfh)
                    T = E.refine_icp(dn, b_dn, T0).transformation
                else:
                    gs = E.unit_cols(np.asarray(fp.data)) / np.sqrt(np.asarray(fp.data).shape[0])
                    ss = d[g]["pca"][mp].T / np.sqrt(d[g]["pca"].shape[1])
                    st = d[0]["pca"][b_map].T / np.sqrt(d[0]["pca"].shape[1])
                    Tr, _ = E.ransac_feature(dn, b_dn, E.make_feature(np.vstack([gs, ss])),
                                             E.make_feature(np.vstack([g_t, st])))
                    # 固定用 R2 规则（fit@5 最大）以便与其他表一致
                    c1 = E.joint_icp(np.asarray(dn.points), d[g]["spec"][mp], b_np, b_tn,
                                     d[0]["spec"][b_map], Tr)
                    c2 = E.joint_icp(np.asarray(dn.points), d[g]["spec"][mp], b_np, b_tn,
                                     d[0]["spec"][b_map], np.eye(4))
                    m1 = metrics(f["xyz"], fb["xyz"], c1)
                    m2 = metrics(f["xyz"], fb["xyz"], c2)
                    T = c1 if m1["fit@5mm"] >= m2["fit@5mm"] else c2
                acc.append(metrics(f["xyz"], fb["xyz"], T))
            reps_out.append({k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]})
            print("  %-14s 第 %d 次: fit5=%.4f fit10=%.4f fit20=%.4f rmse20=%.2f"
                  % (arm, r + 1, acc[0] and reps_out[-1]["fit@5mm"], reps_out[-1]["fit@10mm"],
                     reps_out[-1]["fit@20mm"], reps_out[-1]["rmse@20mm"]))
        res[arm] = {"reps": reps_out,
                    "spread": {k: float(max(x[k] for x in reps_out) - min(x[k] for x in reps_out))
                               for k in reps_out[0]}}
    json.dump(res, open(HERE / "repeatability.json", "w"), indent=2, ensure_ascii=False)
    print("\n运行间波动（同臂重复，仅 RANSAC 随机采样不同）：")
    for arm, v in res.items():
        s = v["spread"]
        print("  %-14s fit@5mm %.4f | fit@10mm %.4f | fit@20mm %.4f | rmse@20mm %.2f mm"
              % (arm, s["fit@5mm"], s["fit@10mm"], s["fit@20mm"], s["rmse@20mm"]))
    print("\n[保存] repeatability.json")


if __name__ == "__main__":
    main()
