# -*- coding: utf-8 -*-
"""嵌套信息 + 残差门控实验：为什么「信息更多」反而可能更差，以及怎么保证「不劣于」

理论要点（对应你提到的残差网络）
------------------------------------------------------------------
ResNet 的保证不是「层数更多一定更好」，而是“y = x + f(x) 且 f 可以退化为 0”
（零初始化 / 权重衰减）⇒ 深层网络的解空间**包含**浅层网络的解 ⇒ 训练误差不劣。
把这一条搬到配准：
  - 「浅层」= 纯几何位姿 T_G；「残差支路」= 光谱细化 ΔT；「恒等短路」= 允许 ΔT = 0。
  - 只有当 (i) 新增信息构成的位姿族**包含**原族，(ii) 有一个**独立于测试指标的验证判据**
    来在接受/拒绝之间选择，才谈得上「不劣」。
  - 朴素拼接（把光谱块与 FPFH 等权拼在一起、损失里固定 λ=1）破坏了 (i)：
    描述子变了 ⇒ RANSAC 采样的对应集合变了 ⇒ 优化地形变了 ⇒ 没有「关掉光谱」的短路 ⇒ 无保证。

本实验用三个**互不相交**的波段把角色分开，杜绝循环：
  优化波段 B_opt = 500-650 nm   （进描述子 PCA 与 joint_icp 损失）
  验证波段 B_val = 650-800 nm   （**只**用于门控接受/拒绝）
  测试波段 B_test= 450-495 ∪ 805-900 nm（任何方法、任何门控都不用它）
  几何指标（fit@5/10/20、rmse@20）与参考轴偏航角 yaw_z 也不参与门控 ⇒ 均为独立评价量。

臂（信息量单调递增）：
  G            纯几何（FPFH + 点到面 ICP）
  G+RGB        FPFH ⊕ PCA(RGB,3)   + joint_icp(RGB)
  G+MS         FPFH ⊕ PCA(opt,12)  + joint_icp(opt)
  G+RGB+MS     FPFH ⊕ PCA(RGB+opt) + joint_icp(RGB+opt)
  G+MS|gate    门控：仅当 B_val 的谱角不劣于 T_G 时才接受 T_MS（残差/短路）
  G+RGB+MS|gate 同上
诊断臂（定位「更差」来自描述子还是损失）：
  G+MS|desc-only  描述子含光谱、损失 λ=0（等价几何 ICP）
  G+MS|loss-only  描述子纯 FPFH、损失含光谱
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
B_OPT = np.where((WL >= 500) & (WL <= 650))[0]
B_VAL = np.where((WL > 650) & (WL <= 800))[0]
B_T1 = np.where((WL >= 450) & (WL <= 495))[0]
B_T2 = np.where((WL >= 805) & (WL <= 900))[0]
B_TEST = np.concatenate([B_T1, B_T2])
FRAMES = [60, 120, 180, 240, 300]
TAU = 0.020
AXIS_XY = (0.0009, 1.2505)


def unit_rows(w):
    w = np.asarray(w, float)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def clip(f):
    return np.clip(f["refl"].astype(np.float64), 0.0, None)


def pca_map(refls, idx, dims):
    X = np.vstack([unit_rows(r[:, idx])[:: max(1, len(r) // 5000)] for r in refls])
    d = int(min(dims, len(idx)))
    mu = X.mean(0)
    ev, evec = np.linalg.eigh(((X - mu).T @ (X - mu)) / max(len(X) - 1, 1))
    o = np.argsort(ev)[::-1][:d]
    return mu, evec[:, o]


def eval_pose(T, xyz_s, xyz_t, full_s, full_t):
    P = (T[:3, :3] @ xyz_s.T).T + T[:3, 3]
    kd = cKDTree(xyz_t)
    o = {}
    for t in (5, 10, 20):
        d, _ = kd.query(P, distance_upper_bound=t / 1000.0)
        o[f"fit@{t}mm"] = float(np.isfinite(d).mean())
    d, j = kd.query(P, distance_upper_bound=TAU)
    ok = np.isfinite(d)
    o["rmse@20mm"] = float(np.sqrt((d[ok] ** 2).mean()) * 1000)
    for tag, bb in (("SAM_test", B_TEST), ("SAM_test_bl", B_T1), ("SAM_test_nir", B_T2),
                    ("SAM_val", B_VAL), ("SAM_opt", B_OPT), ("SAM_RGB", np.array(I_RGB))):
        A = unit_rows(full_s[:, bb])[ok]
        Bm = unit_rows(full_t[:, bb])[j[ok]]
        o[tag] = float(np.degrees(np.arccos(np.clip((A * Bm).sum(1), -1, 1))).mean())
    R = T[:3, :3]
    o["rot"] = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    o["yaw_z"] = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    o["shift_mm"] = float(np.linalg.norm(T[:3, 3]) * 1000)
    y, x = AXIS_XY
    o["dist_axis_mm"] = float(np.hypot(T[0, 3] - x, T[1, 3] - y) * 1000)
    return o


def main():
    frames = {g: E.load_frame(g) for g in [0] + FRAMES}
    fb = frames[0]
    full = {g: unit_rows(clip(frames[g])) for g in [0] + FRAMES}
    b_dn, b_map = E.downsample_with_map(fb["xyz"], normals=True)
    b_fpfh = E.fpfh(b_dn)
    b_np, b_tn = np.asarray(b_dn.points), np.asarray(b_dn.normals)
    g_t = E.unit_cols(np.asarray(b_fpfh.data)) / np.sqrt(np.asarray(b_fpfh.data).shape[0])

    mu_ms, pv_ms = pca_map([clip(frames[g]) for g in [0] + FRAMES], B_OPT, E.PCA_DIMS)
    mu_rgb, pv_rgb = pca_map([clip(frames[g]) for g in [0] + FRAMES], np.array(I_RGB), 3)
    both = np.concatenate([np.array(I_RGB), B_OPT])

    def blocks(f):
        spec_ms = unit_rows(clip(f)[:, B_OPT])
        spec_rgb = unit_rows(clip(f)[:, I_RGB])
        return {"ms": spec_ms, "rgb": spec_rgb, "both": np.hstack([spec_rgb, spec_ms]),
                "pca_ms": unit_rows((spec_ms - mu_ms) @ pv_ms),
                "pca_rgb": unit_rows((spec_rgb - mu_rgb) @ pv_rgb),
                "pca_both": unit_rows((np.hstack([spec_rgb, spec_ms])
                                      - np.concatenate([mu_rgb, mu_ms]))
                                     @ np.block([[pv_rgb, np.zeros((3, pv_ms.shape[1]))],
                                                 [np.zeros((len(B_OPT), 3)), pv_ms]]))}

    BL = blocks(fb)
    arms = {
        "G 纯几何":            dict(desc="fp", loss=None),
        "G+RGB":               dict(desc="rgb", loss="rgb"),
        "G+MS":                dict(desc="ms", loss="ms"),
        "G+RGB+MS":            dict(desc="both", loss="both"),
        "G+MS|仅描述子(λ=0)":    dict(desc="ms", loss="ms0"),
        "G+MS|仅损失":          dict(desc="fp", loss="ms"),
    }
    res = {k: [] for k in list(arms) + ["G+MS|门控", "G+RGB+MS|门控"]}
    gate_src = {}
    for g in FRAMES:
        f = frames[g]
        dn, mp = E.downsample_with_map(f["xyz"], normals=True)
        fp = E.fpfh(dn)
        gs = E.unit_cols(np.asarray(fp.data)) / np.sqrt(np.asarray(fp.data).shape[0])
        bl = blocks(f)
        T = {}
        for name, cfg in arms.items():
            dkind, lkind = cfg["desc"], cfg["loss"]
            if dkind == "fp":
                fs = E.make_feature(np.vstack([gs + 0.0]))
                ft = E.make_feature(np.vstack([g_t]))
            else:
                ss = bl["pca_" + dkind][mp].T / np.sqrt(bl["pca_" + dkind].shape[1])
                st = BL["pca_" + dkind][b_map].T / np.sqrt(BL["pca_" + dkind].shape[1])
                fs = E.make_feature(np.vstack([gs, ss]))
                ft = E.make_feature(np.vstack([g_t, st]))
            T0, _ = E.ransac_feature(dn, b_dn, fs, ft)
            if lkind is None:
                T[name] = E.refine_icp(dn, b_dn, T0).transformation
            else:
                lk = "ms" if lkind == "ms0" else lkind
                lam = 0.0 if lkind == "ms0" else E.JOINT_LAMBDA
                T[name] = E.joint_icp(np.asarray(dn.points), bl[lk][mp], b_np, b_tn,
                                      BL[lk][b_map], T0, lam=lam)
        m = {k: eval_pose(v, f["xyz"], fb["xyz"], full[g], full[0]) for k, v in T.items()}
        # ---- 门控（残差/短路）：验证波段不劣才接受光谱细化 ----
        for src, tgt in (("G+MS", "G+MS|门控"), ("G+RGB+MS", "G+RGB+MS|门控")):
            acc = m[src]["SAM_val"] <= m["G 纯几何"]["SAM_val"] + 1e-9
            chosen = dict(m[src]) if acc else dict(m["G 纯几何"])
            chosen["accepted"] = bool(acc)
            chosen["d_rot_vs_geo"] = m[src]["rot"] - m["G 纯几何"]["rot"]
            m[tgt] = chosen
        for k in m:
            res[k].append(m[k])
    out = {}
    for k, v in res.items():
        agg = {}
        for key in v[0]:
            if key == "accepted":
                agg["accept_rate"] = float(np.mean([x["accepted"] for x in v]))
            else:
                agg[key] = float(np.nanmean([x[key] for x in v]))
        agg["per_frame"] = [{kk: x.get(kk) for kk in ("fit@5mm", "SAM_test", "SAM_val",
                                                      "rot", "yaw_z", "accepted")} for x in v]
        out[k] = agg
    json.dump(out, open(HERE / "nested_info_check.json", "w"), indent=2, ensure_ascii=False)

    print("=" * 122)
    print("嵌套信息实验  优化波段 500–650nm(%d维) | 验证门控 650–800nm(%d维) | 测试 450–495∪805–900nm(%d维)"
          % (len(B_OPT), len(B_VAL), len(B_TEST)))
    print("=" * 122)
    print("%-22s %8s %9s %10s %10s %11s %9s %9s %8s" % (
        "臂", "fit@5mm", "fit@10mm", "fit@20mm", "rmse@20", "SAM_test", "SAM_val", "yaw_z", "旋转°"))
    for k in list(arms) + ["G+MS|门控", "G+RGB+MS|门控"]:
        r = out[k]
        print("%-22s %8.4f %9.4f %10.4f %10.2f %11.2f %9.2f %9.1f %8.1f" % (
            k, r["fit@5mm"], r["fit@10mm"], r["fit@20mm"], r["rmse@20mm"],
            r["SAM_test"], r["SAM_val"], r["yaw_z"], r["rot"]))
    print("\n门控接受率：G+MS|门控 %.0f%%  G+RGB+MS|门控 %.0f%%" % (
        100 * out["G+MS|门控"]["accept_rate"], 100 * out["G+RGB+MS|门控"]["accept_rate"]))
    print("\n相对纯几何的增量（测试口径，正=更好）：")
    print("%-22s %10s %10s %10s %10s" % ("臂", "Δfit@5mm", "Δfit@10mm", "Δfit@20mm", "ΔSAM_test"))
    g0 = out["G 纯几何"]
    for k in ["G+RGB", "G+MS", "G+RGB+MS", "G+MS|仅描述子(λ=0)", "G+MS|仅损失",
              "G+MS|门控", "G+RGB+MS|门控"]:
        r = out[k]
        print("%-22s %+10.4f %+10.4f %+10.4f %+10.2f" % (
            k, r["fit@5mm"] - g0["fit@5mm"], r["fit@10mm"] - g0["fit@10mm"],
            r["fit@20mm"] - g0["fit@20mm"], r["SAM_test"] - g0["SAM_test"]))
    print("\n[保存] nested_info_check.json")


if __name__ == "__main__":
    main()
