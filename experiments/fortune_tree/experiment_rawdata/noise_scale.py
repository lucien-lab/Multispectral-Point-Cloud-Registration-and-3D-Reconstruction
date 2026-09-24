# -*- coding: utf-8 -*-
"""重复性噪声标定：同一臂重复运行（仅 RANSAC 随机采样不同）→ 得到可比较的噪声带

用途：判定「多光谱臂 vs 纯几何臂」的指标差异是否超过同臂运行间波动。
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
REPEATS = 3


def main():
    frames = {g: E.load_frame(g) for g in [0] + N.FRAMES}
    fb = frames[0]
    full = {g: N.unit_rows(N.clip(frames[g])) for g in [0] + N.FRAMES}
    b_dn, b_map = E.downsample_with_map(fb["xyz"], normals=True)
    b_fpfh = E.fpfh(b_dn)
    b_np, b_tn = np.asarray(b_dn.points), np.asarray(b_dn.normals)
    g_t = E.unit_cols(np.asarray(b_fpfh.data)) / np.sqrt(np.asarray(b_fpfh.data).shape[0])
    mu_ms, pv_ms = N.pca_map([N.clip(frames[g]) for g in [0] + N.FRAMES], N.B_OPT, E.PCA_DIMS)
    BL_ms = N.unit_rows(N.clip(fb)[:, N.B_OPT])
    pca_base = N.unit_rows((BL_ms - mu_ms) @ pv_ms)

    rec = {"G 纯几何": [], "G+MS(描述子+损失)": [], "G+MS(仅损失)": []}
    for rep in range(REPEATS):
        for name in rec:
            rec[name].append([])
        for g in N.FRAMES:
            f = frames[g]
            dn, mp = E.downsample_with_map(f["xyz"], normals=True)
            fp = E.fpfh(dn)
            gs = E.unit_cols(np.asarray(fp.data)) / np.sqrt(np.asarray(fp.data).shape[0])
            sp_ms = N.unit_rows(N.clip(f)[:, N.B_OPT])
            pca_s = N.unit_rows((sp_ms[mp] - mu_ms) @ pv_ms)
            xyz_s = np.asarray(dn.points)
            T0, _ = E.ransac_feature(dn, b_dn, E.make_feature(np.vstack([gs])),
                                     E.make_feature(np.vstack([g_t])))
            Tg = E.refine_icp(dn, b_dn, T0).transformation
            ts = pca_s.T / np.sqrt(pca_s.shape[1])
            tt = pca_base[b_map].T / np.sqrt(pca_base.shape[1])
            T1, _ = E.ransac_feature(dn, b_dn, E.make_feature(np.vstack([gs, ts])),
                                     E.make_feature(np.vstack([g_t, tt])))
            Tm = E.joint_icp(xyz_s, sp_ms[mp], b_np, b_tn, BL_ms[b_map], T1)
            Tl = E.joint_icp(xyz_s, sp_ms[mp], b_np, b_tn, BL_ms[b_map], Tg)
            for name, T in (("G 纯几何", Tg), ("G+MS(描述子+损失)", Tm), ("G+MS(仅损失)", Tl)):
                rec[name][-1].append(N.eval_pose(T, f["xyz"], fb["xyz"], full[g], full[0]))
        print("  第 %d 轮完成" % (rep + 1), flush=True)

    out = {}
    for name, runs in rec.items():
        keys = ("fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm", "SAM_test", "yaw_z")
        avg = [{k: float(np.nanmean([m[k] for m in run])) for k in keys} for run in runs]
        out[name] = {"runs": avg,
                     "mean": {k: float(np.mean([a[k] for a in avg])) for k in keys},
                     "std": {k: float(np.std([a[k] for a in avg])) for k in keys},
                     "range": {k: float(max(a[k] for a in avg) - min(a[k] for a in avg))
                               for k in keys}}
    json.dump(out, open(HERE / "noise_scale.json", "w"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 104)
    print("重复性噪声标定（%d 次独立运行，每次为 5 帧平均）" % REPEATS)
    print("=" * 104)
    for k in ("fit@5mm", "fit@10mm", "fit@20mm", "rmse@20mm", "SAM_test", "yaw_z"):
        print("\n%-10s  每轮值 / 均值 / 标准差 / 极差" % k)
        for name in rec:
            r = out[name]
            vals = " ".join("%.4f" % a[k] for a in r["runs"])
            print("  %-20s %s | %.4f | %.4f | %.4f" % (name, vals, r["mean"][k],
                                                       r["std"][k], r["range"][k]))
    # 臂间差异 vs 噪声
    print("\n臂间差异 vs 同臂噪声（fit@5mm / fit@10mm）:")
    a, m, l = (out["G 纯几何"], out["G+MS(描述子+损失)"], out["G+MS(仅损失)"])
    for tag, x, y in (("G+MS(描述子+损失) - G", m, a), ("G+MS(仅损失) - G", l, a),
                      ("G+MS(描述子+损失) - G+MS(仅损失)", m, l)):
        for k in ("fit@5mm", "fit@10mm"):
            d = x["mean"][k] - y["mean"][k]
            sd = float(np.hypot(x["std"][k], y["std"][k]))
            print("  %-34s %-9s Δ=%+.4f  合成标准差=%.4f  |Δ|/sd=%.2f" % (
                tag, k, d, sd, abs(d) / sd if sd else float("nan")))
    print("\n[保存] noise_scale.json")


if __name__ == "__main__":
    main()
