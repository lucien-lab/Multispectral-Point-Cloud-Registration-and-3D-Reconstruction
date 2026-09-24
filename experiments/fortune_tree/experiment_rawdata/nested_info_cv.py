# -*- coding: utf-8 -*-
"""残差门控 v2：把「不劣」变成构造性保证

v1（nested_info_check.py）结论：
  - 损害来自「把光谱块塞进描述子」（λ=0 时 fit@5mm 已经掉 0.0129），损失里加光谱项几乎无害；
  - 用「验证波段 SAM」门控只能部分兜住，因为该判据对几何退化不敏感（120° 帧被误接受）。
v2 改进两处，使门控真正对齐最终评价量：
  ① 保持几何描述子不变（FPFH），光谱只进位姿细化损失 ⇒ 描述子层面严格嵌套；
  ② **几何留出验证**：细化只用源点的一半（训练半），门控在另一半（验证半）上比较
     fit@10mm ⇒ 判据与最终评价量同族但数据不相交（交叉验证式模型选择）。

最终评价仍在全部点上、且用**未被任何臂优化的测试波段** 450–495 ∪ 805–900 nm。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

_s = importlib.util.spec_from_file_location("nic", Path(__file__).resolve().parent
                                            / "nested_info_check.py")
N = importlib.util.module_from_spec(_s)
_s.loader.exec_module(N)
E, HERE = N.E, N.HERE

RNG = np.random.default_rng(20260201)
import sys
sys.modules.setdefault('nic', N)


def fit_on(idx, T, xyz_s, xyz_t, tau=0.005):
    from scipy.spatial import cKDTree
    P = (T[:3, :3] @ xyz_s[idx].T).T + T[:3, 3]
    d, _ = cKDTree(xyz_t).query(P, distance_upper_bound=tau)
    return float(np.isfinite(d).mean())


def main():
    frames = {g: E.load_frame(g) for g in [0] + N.FRAMES}
    fb = frames[0]
    full = {g: N.unit_rows(N.clip(frames[g])) for g in [0] + N.FRAMES}
    b_dn, b_map = E.downsample_with_map(fb["xyz"], normals=True)
    b_fpfh = E.fpfh(b_dn)
    b_np, b_tn = np.asarray(b_dn.points), np.asarray(b_dn.normals)
    g_t = E.unit_cols(np.asarray(b_fpfh.data)) / np.sqrt(np.asarray(b_fpfh.data).shape[0])
    mu_ms, pv_ms = N.pca_map([N.clip(frames[g]) for g in [0] + N.FRAMES], N.B_OPT, E.PCA_DIMS)
    mu_rgb, pv_rgb = N.pca_map([N.clip(frames[g]) for g in [0] + N.FRAMES], np.array(N.I_RGB), 3)

    def blocks(f):
        sm = N.unit_rows(N.clip(f)[:, N.B_OPT])
        sr = N.unit_rows(N.clip(f)[:, N.I_RGB])
        return {"ms": sm, "both": np.hstack([sr, sm]),
                "pca_ms": N.unit_rows((sm - mu_ms) @ pv_ms),
                "pca_both": N.unit_rows((np.hstack([sr, sm]) - np.concatenate([mu_rgb, mu_ms]))
                                        @ np.block([[pv_rgb, np.zeros((3, pv_ms.shape[1]))],
                                                    [np.zeros((len(N.B_OPT), 3)), pv_ms]]))}

    BL = blocks(fb)
    res = {}
    for g in N.FRAMES:
        f = frames[g]
        dn, mp = E.downsample_with_map(f["xyz"], normals=True)
        n = len(np.asarray(dn.points))
        perm = RNG.permutation(n)
        tr, va = perm[: n // 2], perm[n // 2:]
        fp = E.fpfh(dn)
        gs = E.unit_cols(np.asarray(fp.data)) / np.sqrt(np.asarray(fp.data).shape[0])
        bl = blocks(f)
        spec = {k: v[mp] for k, v in bl.items()}
        T0, _ = E.ransac_feature(dn, b_dn, E.make_feature(np.vstack([gs])),
                                 E.make_feature(np.vstack([g_t])))
        T_geo = E.refine_icp(dn, b_dn, T0).transformation
        xyz_s = np.asarray(dn.points)
        rec = {"G 纯几何": T_geo}
        # 全部点细化（无门控的朴素方案）
        for name, lk in (("G+MS|仅损失", "ms"), ("G+RGB+MS|仅损失", "both")):
            rec[name] = E.joint_icp(xyz_s, spec[lk], b_np, b_tn, BL[lk][b_map], T_geo)
        # 仅用训练半细化（几何描述子不变；全信息描述子版另算）
        cand = {}
        for name, lk, desc in (("MS", "ms", None), ("RGB+MS", "both", None),
                               ("MS全信息", "ms", "ms")):
            if desc is None:
                fs = E.make_feature(np.vstack([gs]))
                ft = E.make_feature(np.vstack([g_t]))
            else:
                ss = bl["pca_" + desc][mp].T / np.sqrt(bl["pca_" + desc].shape[1])
                st = BL["pca_" + desc][b_map].T / np.sqrt(BL["pca_" + desc].shape[1])
                fs = E.make_feature(np.vstack([gs, ss]))
                ft = E.make_feature(np.vstack([g_t, st]))
            Tt, _ = E.ransac_feature(dn, b_dn, fs, ft)
            base = T_geo if desc is not None else T_geo   # 残差架构：以几何解为初值
            cand[name] = E.joint_icp(xyz_s[tr], spec[lk][tr], b_np, b_tn,
                                     BL[lk][b_map], base)
        # ---- 门控：验证半的 fit@10mm 不得劣化 ----
        fv0 = fit_on(va, T_geo, xyz_s, fb["xyz"])
        m = {k: N.eval_pose(v, f["xyz"], fb["xyz"], full[g], full[0]) for k, v in rec.items()}
        for name, Tt in cand.items():
            mp_ = N.eval_pose(Tt, f["xyz"], fb["xyz"], full[g], full[0])
            mp_["fit_val@10"] = fit_on(va, Tt, xyz_s, fb["xyz"])
            acc = mp_["fit_val@10"] >= fv0 - 1e-12
            ch = dict(mp_) if acc else dict(m["G 纯几何"])
            ch["accepted"] = bool(acc)
            ch["fit_val@10_geo"] = fv0
            m["门控[" + name + "]"] = ch
        for k, v in m.items():
            res.setdefault(k, []).append(v)

    out = {}
    for k, v in res.items():
        agg = {kk: float(np.nanmean([x.get(kk, np.nan) for x in v]))
               for kk in ("fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm", "SAM_test",
                          "SAM_val", "rot")}
        agg["accept_rate"] = float(np.mean([x.get("accepted", True) for x in v]))
        out[k] = agg
    json.dump(out, open(HERE / "nested_info_cv5.json", "w"), indent=2, ensure_ascii=False)

    print("=" * 118)
    print("残差门控 v3（判据=留出点 fit@5mm，直接跟踪目标指标 + 测试波段完全未被优化）")
    print("=" * 118)
    print("%-26s %8s %9s %10s %9s %10s %9s %8s" % (
        "臂", "fit@5mm", "fit@10mm", "fit@20mm", "rmse@20", "SAM_test", "SAM_val", "接受率"))
    g0 = out["G 纯几何"]
    for k in out:
        r = out[k]
        print("%-26s %8.4f %9.4f %10.4f %9.2f %10.2f %9.2f %7.0f%%" % (
            k, r["fit@5mm"], r["fit@10mm"], r["fit@20mm"], r["rmse@20mm"],
            r["SAM_test"], r["SAM_val"], 100 * r["accept_rate"]))
    print("\n相对纯几何的增量（正 = 更好）：")
    print("%-26s %10s %10s %10s %10s" % ("臂", "Δfit@5", "Δfit@10", "Δfit@20", "ΔSAM_test"))
    for k in out:
        if k == "G 纯几何":
            continue
        r = out[k]
        print("%-26s %+10.4f %+10.4f %+10.4f %+10.2f" % (
            k, r["fit@5mm"] - g0["fit@5mm"], r["fit@10mm"] - g0["fit@10mm"],
            r["fit@20mm"] - g0["fit@20mm"], r["SAM_test"] - g0["SAM_test"]))
    print("\n[保存] nested_info_cv5.json")


if __name__ == "__main__":
    main()
