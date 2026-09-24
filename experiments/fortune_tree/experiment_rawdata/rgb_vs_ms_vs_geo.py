# -*- coding: utf-8 -*-
"""几何配准 / 几何+RGB 配准 / 几何+多光谱配准 三方式对比实验

三条主臂（其余代码路径完全一致，只替换光谱描述子）：
  A 几何        ：无光谱项，FPFH-RANSAC → 点到面 ICP
  B 几何+RGB    ：光谱描述子 = 3 个 RGB 通道（650.1 / 560.3 / 490.4 nm，PCA→3 维）
  C 几何+多光谱 ：光谱描述子 = 500–800 nm 全部 755 维（PCA→12 维）—— 本专利方法
对照臂：
  D 几何+可见光窄带（450–700 nm，634 维）：分离「通道数不足」与「波段范围不同」两个因素
  E 几何+全波段（276.8–1091.7 nm，2048 维）
  F 不配准（恒等变换）

统一协议（同 experiment_rawdata.py）：
  体素 3 mm + FPFH(2 cm) → 描述子 RANSAC（多阈值扫描）→ 位姿细化
  光谱臂：融合代价 cost=(d/d_max)²+λ(SAM/Δ)²，λ=1、K=10、d_max=0.015 m、Δ=10°、60 次迭代；
          多初值 {融合 RANSAC, 单位阵} 择优（先满足 fit@20mm≥0.90，再取 SAM 最小）
  全分辨率统一评价：fit@5/10/20 mm、rmse@20 mm、互为最近邻、光谱角 SAM
"""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE = HERE / "cache"

# 复用主实验脚本的函数（main() 有 __main__ 保护，导入不执行流程）
_s = importlib.util.spec_from_file_location("expraw", ROOT / "experiment_rawdata.py")
E = importlib.util.module_from_spec(_s)
_s.loader.exec_module(E)
E.CACHE = CACHE                       # 修正缓存路径（目录内副本的相对路径已过时）

FRAMES = [60, 120, 180, 240, 300]
TAUS_MM = (5, 10, 20)
TAU_MUTUAL = 0.020                    # m
PCA_DIMS = E.PCA_DIMS                 # 12
WL = E.load_frame(0)["wavelength"]
I_RGB = [b - 1 for b in E.RGB_BANDS_1BASED]        # 957/730/552（1-based）


def bmask(lo: float, hi: float) -> np.ndarray:
    return np.where((WL >= lo) & (WL <= hi))[0]


ARMS = [
    ("A 几何（无光谱）", "geo", None),
    ("B 几何+RGB（3 通道）", "spec", I_RGB),
    ("C 几何+多光谱（500–800 nm）", "spec", bmask(500, 800)),
    ("D 几何+可见光窄带（450–700 nm）", "spec", bmask(450, 700)),
    ("E 几何+全波段（2048）", "spec", np.arange(len(WL))),
]
BAND_NAME = {None: "无", "rgb": "RGB 3 通道"}


def unit_rows(w):
    w = np.asarray(w, float)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def clip_refl(f):
    return np.clip(f["refl"].astype(np.float64), 0.0, None)


def build_desc(frames, idx, dims=PCA_DIMS):
    """负值截断 → L2 归一化 → PCA → L2 归一化。返回每帧 {'spec':(n,d0),'pca':(n,d)}"""
    X = np.vstack([unit_rows(clip_refl(f)[:, idx])[:: max(1, len(f["xyz"]) // 5000)]
                   for f in frames.values()])
    d = int(min(dims, len(idx)))
    mu = X.mean(0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    ev, evec = np.linalg.eigh(cov)
    order = np.argsort(ev)[::-1][:d]
    comps, ratio = evec[:, order], ev[order] / ev.sum()
    out = {deg: {"spec": unit_rows(clip_refl(f)[:, idx]),
                 "pca": unit_rows((unit_rows(clip_refl(f)[:, idx]) - mu) @ comps)}
           for deg, f in frames.items()}
    return out, float(ratio.sum()), d


def evaluate(P, tgt_xyz, tgt_spec, src_spec, ok=None, j=None):
    """统一评价：fit@τ、rmse@20、互为最近邻、SAM（该臂所用波段）"""
    kd = cKDTree(tgt_xyz)
    out = {}
    for t in TAUS_MM:
        dd, _ = kd.query(P, distance_upper_bound=t / 1000.0)
        out[f"fit@{t}mm"] = float(np.isfinite(dd).mean())
    dd, jj = kd.query(P, distance_upper_bound=TAU_MUTUAL)
    okk = np.isfinite(dd)
    out["rmse@20mm"] = float(np.sqrt((dd[okk] ** 2).mean()) * 1000) if okk.any() else float("nan")
    out["fit_mutual@20mm"] = 0.0
    out["sam_self"] = float("nan")
    if okk.any():
        ii = np.arange(len(P))[okk]
        _, i2 = cKDTree(P).query(tgt_xyz[jj[okk]], distance_upper_bound=TAU_MUTUAL)
        m = np.zeros(len(P), bool)
        m[ii[i2 == ii]] = True
        out["fit_mutual@20mm"] = float(m.mean())
        a = src_spec[okk] / (np.linalg.norm(src_spec[okk], axis=1, keepdims=True) + 1e-12)
        b = tgt_spec[jj[okk]] / (np.linalg.norm(tgt_spec[jj[okk]], axis=1, keepdims=True) + 1e-12)
        out["sam_self"] = float(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).mean())
    return out, okk, jj


def rot_shift(T):
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    return float(ang), float(np.linalg.norm(T[:3, 3]) * 1000)


def main():
    frames = {deg: E.load_frame(deg) for deg in [0] + FRAMES}
    fb = frames[0]
    b_dn, b_map = E.downsample_with_map(fb["xyz"], normals=True)
    b_fpfh = E.fpfh(b_dn)
    b_np, b_tn = np.asarray(b_dn.points), np.asarray(b_dn.normals)
    g_t = E.unit_cols(np.asarray(b_fpfh.data)) / np.sqrt(np.asarray(b_fpfh.data).shape[0])

    rows, per_frame, diag = [], [], {}
    for name, kind, idx in ARMS:
        t0 = time.time()
        desc = None
        if kind == "spec":
            desc, evr, d = build_desc({k: frames[k] for k in [0] + FRAMES}, idx)
            diag[name] = {"n_bands": int(len(idx)), "pca_dims": d, "pca_var": evr,
                          "wl_range": [float(WL[idx].min()), float(WL[idx].max())]}
        else:
            d = 0
            diag[name] = {"n_bands": 0, "pca_dims": 0, "pca_var": 0.0, "wl_range": None}

        acc = []
        for deg in FRAMES:
            f = frames[deg]
            dn, mp = E.downsample_with_map(f["xyz"], normals=True)
            fp = E.fpfh(dn)
            T0, f0 = E.ransac_feature(dn, b_dn, fp, b_fpfh)
            if kind == "geo":
                T = E.refine_icp(dn, b_dn, T0).transformation
                pick, n_cand = "点到面 ICP", 1
            else:
                g_s = E.unit_cols(np.asarray(fp.data)) / np.sqrt(np.asarray(fp.data).shape[0])
                s_s = desc[deg]["pca"][mp].T / np.sqrt(desc[deg]["pca"].shape[1])
                s_t = desc[0]["pca"][b_map].T / np.sqrt(desc[0]["pca"].shape[1])
                fs = E.make_feature(np.vstack([g_s, s_s]))
                ft = E.make_feature(np.vstack([g_t, s_t]))
                Tr, _ = E.ransac_feature(dn, b_dn, fs, ft)
                cands = []
                for cn, Ti in (("融合 RANSAC", Tr), ("单位阵", np.eye(4))):
                    Tj = E.joint_icp(np.asarray(dn.points), desc[deg]["spec"][mp],
                                     b_np, b_tn, desc[0]["spec"][b_map], Ti)
                    mm = E.transform_metrics(np.asarray(dn.points), desc[deg]["spec"][mp],
                                             b_np, desc[0]["spec"][b_map], Tj)
                    cands.append((cn, Tj, mm["fitness"], mm["sam_deg"]))
                okl = [c for c in cands if c[2] >= E.FITNESS_GATE]
                p = min(okl, key=lambda c: c[3]) if okl else max(cands, key=lambda c: c[2])
                T, pick, n_cand = p[1], p[0], len(cands)

            P = (T[:3, :3] @ f["xyz"].T).T + T[:3, 3]
            spec_s = desc[deg]["spec"] if desc else unit_rows(clip_refl(f))
            spec_t = desc[0]["spec"] if desc else unit_rows(clip_refl(fb))
            m, _, _ = evaluate(P, fb["xyz"], spec_t, spec_s)
            ang, sh = rot_shift(T)
            # 统一评估波段（500–800 / 450–900 nm），用于横向比较位姿质量
            dd, jj = cKDTree(fb["xyz"]).query(P, distance_upper_bound=TAU_MUTUAL)
            okk = np.isfinite(dd)
            if desc is None:
                m["sam_self"] = float("nan")      # 几何臂无光谱项，只在统一评估波段上报
            for tag, (lo, hi) in (("sam500_800", (500, 800)), ("sam450_900", (450, 900))):
                mi = bmask(lo, hi)
                if okk.any():
                    aa = unit_rows(clip_refl(f)[:, mi])[okk]
                    bb = unit_rows(clip_refl(fb)[:, mi])[jj[okk]]
                    m[tag] = float(np.degrees(np.arccos(np.clip(
                        (aa * bb).sum(1), -1, 1))).mean())
                else:
                    m[tag] = float("nan")
            m.update(deg=deg, rot_deg=ang, shift_mm=sh, pick=pick, n_cand=n_cand)
            acc.append(m)
            per_frame.append({"arm": name, **m})
            print("  %-30s %3d°  fit5=%.4f fit20=%.4f rmse20=%.2f SAM=%.2f° rot=%.2f°"
                  % (name, deg, m["fit@5mm"], m["fit@20mm"], m["rmse@20mm"],
                     m["sam_self"], ang))

        agg = {"arm": name, "n_bands": diag[name]["n_bands"], "pca_dims": d,
               "wl_range": diag[name]["wl_range"], "runtime_s": time.time() - t0,
               "pca_var": diag[name]["pca_var"]}
        for k in ("fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm", "fit_mutual@20mm",
                  "sam_self", "sam500_800", "sam450_900", "rot_deg", "shift_mm"):
            agg[k] = float(np.nanmean([a[k] for a in acc]))
        rows.append(agg)

    # 恒等对照
    acc = []
    for deg in FRAMES:
        f = frames[deg]
        m, _, _ = evaluate(f["xyz"], fb["xyz"], unit_rows(clip_refl(fb)), unit_rows(clip_refl(f)))
        for tag, (lo, hi) in (("sam500_800", (500, 800)), ("sam450_900", (450, 900))):
            mi = bmask(lo, hi)
            dd, jj = cKDTree(fb["xyz"]).query(f["xyz"], distance_upper_bound=TAU_MUTUAL)
            ok = np.isfinite(dd)
            aa = unit_rows(clip_refl(f)[:, mi])[ok]
            bb = unit_rows(clip_refl(fb)[:, mi])[jj[ok]]
            m[tag] = float(np.degrees(np.arccos(np.clip((aa * bb).sum(1), -1, 1))).mean())
        m.update(deg=deg, rot_deg=0.0, shift_mm=0.0, pick="—", n_cand=0)
        acc.append(m)
    agg = {"arm": "F 不配准（恒等）", "n_bands": 0, "pca_dims": 0, "wl_range": None,
           "runtime_s": 0.0, "pca_var": 0.0}
    for k in ("fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm", "fit_mutual@20mm",
              "sam_self", "sam500_800", "sam450_900"):
        agg[k] = float(np.nanmean([a[k] for a in acc]))
    rows.append(agg)

    # 配准前光谱可判别性（0° 帧：帧内 3 mm 邻点 vs 帧内随机对）
    d2, j2 = cKDTree(fb["xyz"]).query(fb["xyz"], k=2)
    nb = d2[:, 1] < 0.003
    rng = np.random.default_rng(0)
    sets = [("RGB 3 通道", I_RGB), ("可见光窄带 450–700", bmask(450, 700)),
            ("500–800 nm", bmask(500, 800)), ("450–900 nm", bmask(450, 900)),
            ("全波段 276.8–1091.7", np.arange(len(WL)))]
    disc = []
    for nm, idx in sets:
        u = unit_rows(clip_refl(fb)[:, idx])
        sn = np.degrees(np.arccos(np.clip((u[nb] * u[j2[nb, 1]]).sum(1), -1, 1))).mean()
        i1, i2 = rng.integers(0, len(u), 3000), rng.integers(0, len(u), 3000)
        sr = np.degrees(np.arccos(np.clip((u[i1] * u[i2]).sum(1), -1, 1))).mean()
        disc.append({"desc": nm, "n_bands": int(len(idx)), "sam_nn_deg": float(sn),
                     "sam_rand_deg": float(sr), "ratio": float(sr / max(sn, 1e-9))})

    # RGB 能否替代近红外构造植被指数（植物点 = NDVI≥0.5 为参照）
    i650 = int(np.argmin(np.abs(WL - 650.0)))
    i800 = int(np.argmin(np.abs(WL - 800.0)))
    iR, iG, iB = I_RGB[0], I_RGB[1], I_RGB[2]
    r = clip_refl(fb)
    R, G, B = r[:, iR], r[:, iG], r[:, iB]
    r650, r800 = r[:, i650], r[:, i800]
    ndvi = (r800 - r650) / np.maximum(r800 + r650, 1e-9)
    plant = ndvi >= 0.5
    idxs = {
        "NDVI=(R800−R650)/(R800+R650)  需 800 nm": ndvi,
        "NGRDI=(G−R)/(G+R)  仅可见光": (G - R) / np.maximum(G + R, 1e-9),
        "GCC=G/(R+G+B)  仅可见光": G / np.maximum(R + G + B, 1e-9),
        "ExG=2G−R−B  仅可见光": 2 * G - R - B,
        "VARI=(G−R)/(G+R−B)  仅可见光": (G - R) / np.maximum(G + R - B, 1e-9),
    }
    veg = []
    for nm, v in idxs.items():
        a, b = v[plant], v[~plant]
        sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / max(len(a) + len(b) - 2, 1))
        veg.append({"index": nm, "mean_plant": float(a.mean()), "mean_other": float(b.mean()),
                    "cohen_d": float(abs(a.mean() - b.mean()) / max(sd, 1e-12)),
                    "plant_frac": float(plant.mean())})

    json.dump({"arms": rows, "per_frame": per_frame, "descriptor": diag,
               "discriminability": disc, "vegetation_index": veg,
               "protocol": {"voxel_m": E.VOXEL, "fpfh_radius_m": E.FPFH_RADIUS,
                            "joint": {"lambda": E.JOINT_LAMBDA, "K": E.JOINT_K,
                                      "d_max_m": E.JOINT_MAX_DIST, "delta_deg": 10.0,
                                      "iters": E.JOINT_ITERS, "fitness_gate": E.FITNESS_GATE},
                            "taus_mm": list(TAUS_MM), "tau_mutual_m": TAU_MUTUAL,
                            "pca_dims": PCA_DIMS, "rgb_bands_nm":
                                [float(WL[i]) for i in I_RGB]}},
              open(HERE / "rgb_vs_ms_vs_geo.json", "w"), indent=2, ensure_ascii=False)

    # ---------------- 打印表 1：三方式对比 ----------------
    print("\n" + "=" * 118)
    print("表 1  三种配准方式对比（5 帧平均；全部点口径；fit 越大越好，rmse/SAM 越小越好）")
    print("=" * 118)
    hdr = ("方式", "光谱维度", "fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm",
           "互最近邻", "SAM自身", "SAM(500–800)", "SAM(450–900)", "旋转(°)", "耗时(s)")
    print("%-30s %8s %8s %8s %8s %9s %9s %8s %11s %11s %8s %7s" % hdr)
    print("-" * 118)
    for r in rows:
        print("%-30s %8s %8.4f %8.4f %8.4f %9.2f %9.4f %8.2f %11.2f %11.2f %8.2f %7.1f" % (
            r["arm"], ("—" if r["n_bands"] == 0 else "%d维" % r["n_bands"]),
            r["fit@5mm"], r["fit@10mm"], r["fit@20mm"], r["rmse@20mm"],
            r["fit_mutual@20mm"], r.get("sam_self", float("nan")),
            r["sam500_800"], r["sam450_900"], r.get("rot_deg", float("nan")),
            r.get("runtime_s", 0.0)))

    print("\n" + "=" * 96)
    print("表 2  配准前光谱可判别性（0° 帧：帧内 3 mm 邻点对 vs 帧内随机对，光谱角均值）")
    print("=" * 96)
    print("%-24s %8s %12s %12s %8s" % ("描述子", "维度", "邻点SAM(°)", "随机对SAM(°)", "判别比"))
    for x in disc:
        print("%-24s %8d %12.2f %12.2f %8.2f" % (x["desc"], x["n_bands"],
              x["sam_nn_deg"], x["sam_rand_deg"], x["ratio"]))

    print("\n" + "=" * 96)
    print("表 3  可见光指数能否替代近红外植被指数（0° 帧，以 NDVI≥0.5 为植物点参照）")
    print("=" * 96)
    print("%-40s %12s %12s %9s" % ("指数", "植物点均值", "其余点均值", "Cohen d"))
    for x in veg:
        print("%-40s %12.4f %12.4f %9.2f" % (x["index"], x["mean_plant"],
              x["mean_other"], x["cohen_d"]))
    print("\n[保存] rgb_vs_ms_vs_geo.json")


if __name__ == "__main__":
    main()
