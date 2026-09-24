#!/usr/bin/env python3
"""experiment_rawdata 诊断脚本 —— 三项关键诊断（结果写入 diagnostics.json）

① 光谱可判别性：帧内邻点 SAM vs 帧内随机对 SAM（按波段）
② 几何旋转判别力：绕树竖轴扫描 θ∈[0,360)，fitness / SAM 曲线
③ 帧间重叠：帧内点距 vs 帧间恒等近邻距离
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
FRAMES = [0, 60, 120, 180, 240, 300]


def load(deg: int) -> dict:
    z = np.load(CACHE / f"frame_{deg:03d}.npz")
    return {k: z[k] for k in z.files}


def unit(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, float)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def sam(a, b) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


res: dict = {}
f0 = load(0)
wl = f0["wavelength"]
rng = np.random.default_rng(0)

# ---------- ① 光谱可判别性 ----------
bands = {"all_276-1092": (276.0, 1092.0), "450-900": (450.0, 900.0),
         "500-800": (500.0, 800.0), "600-780": (600.0, 780.0)}
d, j = cKDTree(f0["xyz"]).query(f0["xyz"], k=2)
nb = d[:, 1] < 0.003
r0 = np.clip(f0["refl"].astype(float), 0, None)
disc = {}
for name, (lo, hi) in bands.items():
    m = (wl >= lo) & (wl <= hi)
    u = unit(r0[:, m])
    i1, i2 = rng.integers(0, len(u), 2000), rng.integers(0, len(u), 2000)
    disc[name] = {"dims": int(m.sum()),
                  "neighbor_sam_deg": float(sam(u[nb], u[j[nb, 1]]).mean()),
                  "random_sam_deg": float(sam(u[i1], u[i2]).mean())}
res["discriminability"] = disc

# ---------- ② 几何旋转判别力 ----------
m = (wl >= 500) & (wl <= 800)
b_spec = unit(r0[:, m])
b_xyz = f0["xyz"]
kd = cKDTree(b_xyz)
cx, cy = b_xyz[:, 0].mean(), b_xyz[:, 1].mean()
sweep = {}
for deg in FRAMES[1:]:
    f = load(deg)
    s_spec = unit(np.clip(f["refl"].astype(float), 0, None)[:, m])
    curve = []
    for th in np.arange(0, 360, 2.0):
        t = np.radians(th)
        c, s = np.cos(t), np.sin(t)
        T = np.eye(4)
        T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
        T[0, 3], T[1, 3] = cx - c * cx + s * cy, cy - s * cx - c * cy
        p = (T[:3, :3] @ f["xyz"].T).T + T[:3, 3]
        dd, jj = kd.query(p, distance_upper_bound=0.02)
        ok = np.isfinite(dd)
        if ok.sum() < 10:
            curve.append((float(th), 0.0, None))
            continue
        curve.append((float(th), float(ok.mean()),
                      float(sam(s_spec[ok], b_spec[jj[ok]]).mean())))
    fits = np.array([c[1] for c in curve])
    sams = np.array([c[2] if c[2] is not None else np.nan for c in curve])
    sweep[str(deg)] = {
        "theta_geo_best": float(curve[int(np.argmax(fits))][0]),
        "theta_sam_best": float(curve[int(np.nanargmin(sams))][0]),
        "fitness_min": float(fits.min()), "fitness_max": float(fits.max()),
        "sam_min_deg": float(np.nanmin(sams)), "sam_max_deg": float(np.nanmax(sams)),
        "curve": [(c[0], round(c[1], 4), None if c[2] is None else round(c[2], 3))
                  for c in curve[::5]],
    }
res["rotation_sweep"] = sweep

# ---------- ③ 帧间重叠 ----------
d0, _ = cKDTree(b_xyz).query(b_xyz, k=2)
overlap = {"within_frame_nn_mm": {"median": float(np.median(d0[:, 1]) * 1000),
                                  "p90": float(np.percentile(d0[:, 1], 90) * 1000)}}
for deg in FRAMES[1:]:
    dd, _ = cKDTree(b_xyz).query(load(deg)["xyz"], distance_upper_bound=0.05)
    ok = np.isfinite(dd)
    overlap[f"identity_{deg}"] = {
        "nn_median_mm": float(np.median(dd[ok]) * 1000),
        "frac_lt_3mm": float(np.mean(dd < 0.003)), "frac_lt_10mm": float(np.mean(dd < 0.01))}
res["overlap"] = overlap

(OUT / "diagnostics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
print(json.dumps({k: v for k, v in res.items() if k != "rotation_sweep"},
                 indent=2, ensure_ascii=False))
print("\n旋转扫描摘要:")
for deg, s in sweep.items():
    print(f"  {deg:>3}°: 几何最优θ={s['theta_geo_best']:.0f}°  SAM最优θ={s['theta_sam_best']:.0f}°  "
          f"fitness∈[{s['fitness_min']:.2f},{s['fitness_max']:.2f}]  SAM∈[{s['sam_min_deg']:.1f},{s['sam_max_deg']:.1f}]°")
print(f"\n[保存] {OUT/'diagnostics.json'}")
