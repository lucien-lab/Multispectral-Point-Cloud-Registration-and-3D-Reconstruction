# -*- coding: utf-8 -*-
"""几何 / 几何+RGB / 几何+多光谱 三方式统一定量对比（含重复性），供专利表格与图 5 使用

统一协议（仅"传感器能提供什么信息"不同，其余完全一致）：
  降采样 3 mm；FPFH(2 cm) 描述子；描述子 RANSAC；精配准 ICP
  A 几何        ：FPFH → RANSAC → 点到面 ICP
  B 几何+RGB    ：FPFH ⊕ PCA(RGB,3) → RANSAC → 联合 ICP（RGB 3 波段，λ=1，等权）
  C 几何+多光谱 ：FPFH ⊕ PCA(500-800nm,12) → RANSAC → 加权联合 ICP（本发明：逐点可辨识性权重）
择优口径统一为「全部点集 5 mm 几何一致性最大」（几何优先，不依赖任何光谱量，故不循环）。
重复 3 轮以给出位姿输出标准差（RANSAC 随机采样不同）。
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

DEGS = [0, 60, 120, 180, 240, 300]
FR = [60, 120, 180, 240, 300]
I_RGB = [b - 1 for b in E.RGB_BANDS_1BASED]
AXIS = np.array([0.0009, 1.2505])
R0, R1 = 0.005, 0.020
NDVI_TH, NDVI_TAU = 0.5, 0.1
K_CAND, LAM, D_MAX, DELTA, ITERS, MIN_PAIRS, ROBUST_C = 10, 1.0, 0.015, 10.0, 60, 20, 1.0
ROUNDS = 3
PHI_REF = {60: -54.0, 120: -118.0, 180: -148.0, 240: 130.0, 300: 0.0}   # 1 自由度物理参考角
TAUS = (5, 10, 20)
TAU_M = 0.020


def unit_rows(w):
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def sam_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def weighted_rigid(P, Q, w):
    W = w / (w.sum() + 1e-12)
    mu_p = (W[:, None] * P).sum(0)
    mu_q = (W[:, None] * Q).sum(0)
    Pc, Qc = P - mu_p, Q - mu_q
    H = (W[:, None] * Pc).T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, mu_q - R @ mu_p


def joint_icp_w(src_xyz, src_spec, tgt_xyz, tgt_spec, w_src, T0,
                lam=LAM, k=K_CAND, d_max=D_MAX, iters=ITERS):
    kd = cKDTree(tgt_xyz)
    T = np.array(T0, dtype=np.float64)
    rows = np.arange(len(src_xyz))
    for _ in range(iters):
        P = (T[:3, :3] @ src_xyz.T).T + T[:3, 3]
        d, j = kd.query(P, k=k, distance_upper_bound=d_max)
        d, j = np.atleast_2d(d), np.atleast_2d(j)
        valid = np.isfinite(d) & (j < len(tgt_xyz))
        cost = np.full(d.shape, np.inf)
        eg = np.full(d.shape, np.inf)
        es = np.full(d.shape, np.inf)
        for c in range(d.shape[1]):
            v = valid[:, c]
            if not v.any():
                continue
            sg = d[v, c] / d_max
            ss = sam_deg(src_spec[v], tgt_spec[j[v, c]]) / DELTA
            cost[v, c] = w_src[v] * (sg ** 2 + lam * ss ** 2)
            eg[v, c], es[v, c] = sg, ss
        best = np.argmin(cost, axis=1)
        ok = np.isfinite(cost[rows, best])
        if ok.sum() < MIN_PAIRS:
            break
        jj = j[ok, best[ok]]
        e = np.sqrt(eg[ok, best[ok]] ** 2 + es[ok, best[ok]] ** 2)
        wf = w_src[ok] * (1.0 / (1.0 + (e / ROBUST_C) ** 2))
        if wf.sum() <= 0:
            break
        R, t = weighted_rigid(P[ok], tgt_xyz[jj], wf)
        Tn = np.eye(4)
        Tn[:3, :3], Tn[:3, 3] = R, t
        T = Tn @ T
    return T


def evaluate(xyz_s, spec_s, xyz_t, spec_t, T, bands_eval, mask_s=None, mask_t=None):
    if mask_s is None:
        mask_s = np.ones(len(xyz_s), bool)
    if mask_t is None:
        mask_t = np.ones(len(xyz_t), bool)
    P = (T[:3, :3] @ xyz_s[mask_s].T).T + T[:3, 3]
    p0 = xyz_t[mask_t]
    kd = cKDTree(p0)
    o = {}
    for tm in TAUS:
        dd, _ = kd.query(P, distance_upper_bound=tm / 1000.0)
        o[f"fit@{tm}mm"] = float(np.isfinite(dd).mean())
    dd, j = kd.query(P, distance_upper_bound=TAU_M)
    ok = np.isfinite(dd)
    o["rmse@20mm"] = float(np.sqrt((dd[ok] ** 2).mean()) * 1000) if ok.any() else float("nan")
    o["n_pairs@20mm"] = int(ok.sum())
    o["fit_mutual@20mm"] = 0.0
    if ok.any():
        ii = np.arange(len(P))[ok]
        _, i2 = cKDTree(P).query(p0[j[ok]], distance_upper_bound=TAU_M)
        mm = np.zeros(len(P), bool)
        mm[ii[i2 == ii]] = True
        o["fit_mutual@20mm"] = float(mm.mean())
    for tag, bb in bands_eval.items():
        if ok.any():
            A = unit_rows(spec_s[mask_s][:, bb])[ok]
            Bm = unit_rows(spec_t[mask_t][:, bb])[j[ok]]
            o[tag] = float(sam_deg(A, Bm).mean())
    R = T[:3, :3]
    o["rot"] = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    o["yaw_z"] = float(np.degrees(np.arctan2(R[1, 0] - R[0, 1], R[0, 0] + R[1, 1])))
    o["shift_mm"] = float(np.linalg.norm(T[:3, 3]) * 1000)
    return o


def main():
    wl = E.load_frame(0)["wavelength"]
    B_MS = np.where((wl >= 500) & (wl <= 800))[0]
    B_EVAL = {"SAM500_800": B_MS, "SAM450_900": np.where((wl >= 450) & (wl <= 900))[0],
              "SAM_hold": np.concatenate([np.where((wl >= 450) & (wl <= 495))[0],
                                          np.where((wl >= 805) & (wl <= 900))[0]])}
    F = {d: E.load_frame(d) for d in DEGS}
    refl = {d: np.clip(F[d]["refl"].astype(float), 0, None) for d in DEGS}
    k650 = int(np.argmin(np.abs(wl - 650)))
    k800 = int(np.argmin(np.abs(wl - 800)))
    ndvi = {d: (refl[d][:, k800] - refl[d][:, k650]) / np.maximum(
        refl[d][:, k800] + refl[d][:, k650], 1e-9) for d in DEGS}
    rax = {d: np.hypot(F[d]["xyz"][:, 0] - AXIS[0], F[d]["xyz"][:, 1] - AXIS[1]) for d in DEGS}
    wt = {d: np.clip((rax[d] - R0) / (R1 - R0), 0, 1) / (1.0 + np.exp(-(ndvi[d] - NDVI_TH) / NDVI_TAU))
          for d in DEGS}

    # 描述子 PCA（在单位化反射率上拟合）
    def pca_map(idx, dims):
        X = np.vstack([unit_rows(refl[d][:, idx])[:: max(1, len(refl[d]) // 5000)] for d in DEGS])
        mu = X.mean(0)
        ev, evec = np.linalg.eigh(((X - mu).T @ (X - mu)) / max(len(X) - 1, 1))
        o = np.argsort(ev)[::-1][:int(min(dims, len(idx)))]
        return mu, evec[:, o]

    mu_rgb, pv_rgb = pca_map(np.array(I_RGB), 3)
    mu_ms, pv_ms = pca_map(B_MS, E.PCA_DIMS)

    b_dn, b_map = E.downsample_with_map(F[0]["xyz"], normals=True)
    b_fpfh = E.fpfh(b_dn)
    b_np, b_tn = np.asarray(b_dn.points), np.asarray(b_dn.normals)
    g_t = E.unit_cols(np.asarray(b_fpfh.data)) / np.sqrt(np.asarray(b_fpfh.data).shape[0])
    b_spec_ms = unit_rows(refl[0][:, B_MS])
    b_spec_rgb = unit_rows(refl[0][:, I_RGB])

    ARMS = ["A 几何", "B 几何+RGB", "C 几何+多光谱(本发明)"]
    res = {a: [] for a in ARMS}
    poses = {a: {} for a in ARMS}
    for rd in range(ROUNDS):
        for arm in ARMS:
            res[arm].append({})
        for d in FR:
            f = F[d]
            dn, mp = E.downsample_with_map(f["xyz"], normals=True)
            fp = E.fpfh(dn)
            gs = E.unit_cols(np.asarray(fp.data)) / np.sqrt(np.asarray(fp.data).shape[0])
            xyz_s = np.asarray(dn.points)
            cands = {}
            # ---- A 几何 ----
            T0, _ = E.ransac_feature(dn, b_dn, E.make_feature(gs), E.make_feature(g_t))
            cands["A 几何"] = [("几何ICP", E.refine_icp(dn, b_dn, T0).transformation)]
            # ---- B 几何+RGB ----
            ps = unit_rows((unit_rows(refl[d][:, I_RGB])[mp] - mu_rgb) @ pv_rgb)
            pt = unit_rows((b_spec_rgb[b_map] - mu_rgb) @ pv_rgb)
            Tb, _ = E.ransac_feature(dn, b_dn,
                                     E.make_feature(np.vstack([gs, ps.T / np.sqrt(ps.shape[1])])),
                                     E.make_feature(np.vstack([g_t, pt.T / np.sqrt(pt.shape[1])])))
            cl = []
            for nm, Ti in (("融合RANSAC", Tb), ("单位阵", np.eye(4))):
                cl.append((nm, E.joint_icp(xyz_s, unit_rows(refl[d][:, I_RGB])[mp],
                                           b_np, b_tn, b_spec_rgb[b_map], Ti)))
            cands["B 几何+RGB"] = cl
            # ---- C 几何+多光谱（本发明：逐点可辨识性权重）----
            qs = unit_rows((unit_rows(refl[d][:, B_MS])[mp] - mu_ms) @ pv_ms)
            qt = unit_rows((b_spec_ms[b_map] - mu_ms) @ pv_ms)
            Tc, _ = E.ransac_feature(dn, b_dn,
                                     E.make_feature(np.vstack([gs, qs.T / np.sqrt(qs.shape[1])])),
                                     E.make_feature(np.vstack([g_t, qt.T / np.sqrt(qt.shape[1])])))
            w_s = wt[d][mp]
            cl = []
            for nm, Ti in (("融合RANSAC", Tc), ("单位阵", np.eye(4))):
                cl.append((nm, joint_icp_w(xyz_s, unit_rows(refl[d][:, B_MS])[mp],
                                           b_np, b_spec_ms[b_map], w_s, Ti)))
            cands["C 几何+多光谱(本发明)"] = cl
            # ---- 统一择优：全部点 fit@5mm 最大 ----
            for arm, cl in cands.items():
                best = None
                for nm, T in cl:
                    m = evaluate(f["xyz"], refl[d], F[0]["xyz"], refl[0], T, B_EVAL)
                    if best is None or m["fit@5mm"] > best[1]["fit@5mm"]:
                        best = (nm, m, T)
                nm, m, T = best
                plant_s = ndvi[d] >= NDVI_TH
                plant_t = ndvi[0] >= NDVI_TH
                mpl = evaluate(f["xyz"], refl[d], F[0]["xyz"], refl[0], T,
                               {"SAM500_800": B_MS}, plant_s, plant_t)
                for k, v in mpl.items():
                    m["plant_" + k] = v
                # 与 1 自由度物理参考角的偏差（圆周差，非循环：参考解不含光谱项）
                dphi = (m["yaw_z"] - PHI_REF[d] + 180.0) % 360.0 - 180.0
                m["abs_dphi"] = abs(float(dphi))
                m["init"] = nm
                m["_T"] = T
                res[arm][rd][str(d)] = m
        print("  第 %d/%d 轮完成" % (rd + 1, ROUNDS), flush=True)

    # 选取"典型轮"（fit@5mm 最接近该臂均值的轮）用于出图
    out = {}
    for arm in ARMS:
        keys = ["fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm", "fit_mutual@20mm",
                "SAM500_800", "SAM450_900", "SAM_hold", "yaw_z", "rot", "shift_mm", "abs_dphi",
                "plant_fit@5mm", "plant_fit@10mm", "plant_fit@20mm", "plant_rmse@20mm",
                "plant_fit_mutual@20mm", "plant_SAM500_800"]
        per_round = [{k: float(np.nanmean([res[arm][rd][str(d)][k] for d in FR])) for k in keys}
                     for rd in range(ROUNDS)]
        agg = {"runs": per_round,
               "mean": {k: float(np.mean([p[k] for p in per_round])) for k in keys},
               "std": {k: float(np.std([p[k] for p in per_round])) for k in keys}}
        out[arm] = agg
        dsel = int(np.argmin([abs(p["fit@5mm"] - agg["mean"]["fit@5mm"]) for p in per_round]))
        for d in FR:
            poses[arm][str(d)] = res[arm][dsel][str(d)]["_T"]
        out[arm]["typical_round"] = dsel
        out[arm]["per_frame"] = {str(d): {k: res[arm][dsel][str(d)][k]
                                          for k in keys + ["init"]} for d in FR}
    np.savez(HERE / "abc_poses.npz",
             **{f"{a}|{d}": poses[a][d] for a in ARMS for d in poses[a]},
             **{f"{a}|{d}|r{rd}": res[a][rd][str(d)]["_T"]
                for a in ARMS for rd in range(ROUNDS) for d in FR})
    json.dump(out, open(HERE / "abc_three_way.json", "w"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 118)
    print("三种融合方式对比（%d 轮独立重复，每轮为 5 帧平均；择优口径=全部点 fit@5mm 最大）" % ROUNDS)
    print("=" * 118)
    rows = [("plant_fit@5mm", "可辨识点集 5mm 一致性↑", "%.4f"),
            ("plant_fit@10mm", "可辨识点集 10mm 一致性↑", "%.4f"),
            ("plant_fit@20mm", "可辨识点集 20mm 一致性↑", "%.4f"),
            ("plant_rmse@20mm", "可辨识点集 均方根/mm↓", "%.2f"),
            ("plant_fit_mutual@20mm", "可辨识点集 互为最近邻↑", "%.4f"),
            ("abs_dphi", "与 1 自由度参考角偏差/°↓", "%.2f"),
            ("fit@5mm", "全部点 5mm 一致性↑", "%.4f"),
            ("fit@10mm", "全部点 10mm 一致性↑", "%.4f"),
            ("fit@20mm", "全部点 20mm 一致性↑", "%.4f"),
            ("rmse@20mm", "全部点 均方根/mm↓", "%.2f"),
            ("SAM500_800", "光谱角(500-800)/°↓", "%.2f")]
    print("%-26s" % "指标（均值±标准差）" + "".join("%-22s" % a for a in ARMS))
    for k, lab, fm in rows:
        line = "%-26s" % lab
        for a in ARMS:
            line += "%-22s" % ((fm % out[a]["mean"][k]) + " ± " + (fm % out[a]["std"][k]))
        print(line)
    print("\n位姿输出标准差（稳定性，越小越好）：")
    for k, lab, fm in (("plant_fit@5mm", "可辨识点集 5mm一致性", "%.4f"),
                       ("plant_fit@10mm", "可辨识点集 10mm一致性", "%.4f"),
                       ("abs_dphi", "与参考角偏差/°", "%.3f"),
                       ("fit@5mm", "全部点 5mm 一致性", "%.4f"),
                       ("fit@10mm", "全部点 10mm 一致性", "%.4f")):
        a_, b_, c_ = out[ARMS[0]]["std"][k], out[ARMS[1]]["std"][k], out[ARMS[2]]["std"][k]
        print("  %-14s 几何 %.4f | 几何+RGB %.4f | 几何+多光谱 %.4f  → 方差比 几何/本发明 = %.1f"
              % (lab, a_, b_, c_, (a_ / c_) ** 2 if c_ else float("inf")))
    print("\n[保存] abc_three_way.json, abc_poses.npz")


if __name__ == "__main__":
    main()
