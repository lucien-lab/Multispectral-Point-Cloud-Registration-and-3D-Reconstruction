#!/usr/bin/env python3
"""experiment_rawdata —— 从最原始 x,y,z 与初始维度(2048波段)光谱出发的多光谱联合配准

数据链路（全部从最原始数据出发）：
  sbq/*.csv(1000点波形) + el_re.txt(角度) → 原始 x,y,z
  cripped/*.txt(用户分割好的绿植点,~4000点/帧) → 反查原始点 ID（精确子集）
  gp/specData/spec_*.txt(2048波段) + dark/white → 标定反射率
  → 「几何 + 光谱」联合配准 → 由原始光谱转 RGB 可视化

与 experiment_cripped（仅几何/RGB）用三指标对比：
  ① fitness（20mm 重叠率）  ② inlier_rmse（mm）  ③ 光谱一致性（重叠点对平均光谱角 °）

关键设计（由数据诊断得出）：
  * 全 2048 波段的光谱角被低信号波段噪声淹没（帧内邻点 SAM 29° ≈ 帧内随机 32°），
    必须限定到信息波段；500–800nm 下帧内邻点 4.6° vs 随机 18.8°，判别力强。
  * 几何在绕轴 0~360° 全域 fitness 平坦（0.78~0.99），无旋转判别力；
    故联合配准中让光谱项真正参与对应选择（几何门控 + 光谱细化）。
  * 自证规避：配准用 500–800nm，评估另报 450–900nm（更宽、含额外信息）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from convert_to_txt import calibration_paths, load_numeric, point_coordinates  # noqa: E402

OUT = SCRIPT_DIR / "experiment_rawdata"
CACHE = OUT / "cache"
CRIPPED = SCRIPT_DIR / "cripped"

RGB_BANDS_1BASED = (957, 730, 552)     # R,G,B（MATLAB 1-based，同 convert_to_txt）

FRAMES: list[tuple[int, str, str]] = [
    (0,   "20260201三维重建发财树0度",  "20260904三维重建发财树0度.txt"),
    (60,  "20260201三维建模发财树60度",  "20260904三维建模发财树60度.txt"),
    (120, "20260201三维建模发财树120度", "20260904三维建模发财树120度.txt"),
    (180, "20260201三维建模发财树180度", "20260904三维建模发财树180度.txt"),
    (240, "20260201三维建模发财树240度", "20260904三维建模发财树240度.txt"),
    (300, "20260201三维建模发财树300度", "20260904三维建模发财树300度.txt"),
]

# ---- 几何配准参数（与 experiment_cripped 完全一致，保证可比）----
VOXEL = 0.003
FPFH_RADIUS = 0.02
RANSAC_MAX_CORR = 0.02
ICP_THRESHOLDS = (0.004, 0.006, 0.008, 0.012)
FITNESS_THRESHOLD = 0.02

# ---- 光谱参数 ----
BAND_OPT = (500.0, 800.0)      # 配准/描述子用波段（信息量最集中）
BAND_EVAL = (450.0, 900.0)     # 指标评估用波段（更宽，避免自证）
PCA_DIMS = 12
W_SPECTRAL = 1.0               # 融合特征中光谱块权重
JOINT_LAMBDA = 1.0             # 联合 ICP 光谱项权重
JOINT_K = 10                   # 几何门控候选数
JOINT_MAX_DIST = 0.015
JOINT_ITERS = 60
FITNESS_GATE = 0.90            # M_joint 候选筛选门槛


# ============================ 工具 ============================

def band_mask(wl: np.ndarray, rng: tuple[float, float]) -> np.ndarray:
    return (wl >= rng[0]) & (wl <= rng[1])


def idx_of(wl: np.ndarray, nm: float) -> int:
    return int(np.argmin(np.abs(wl - nm)))


def unit_rows(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def unit_cols(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    return w / (np.linalg.norm(w, axis=0, keepdims=True) + 1e-12)


def spectral_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def make_feature(data: np.ndarray) -> o3d.pipelines.registration.Feature:
    f = o3d.pipelines.registration.Feature()
    f.data = np.ascontiguousarray(data, dtype=np.float64)
    return f


def downsample_with_map(points: np.ndarray, voxel: float = VOXEL,
                        normals: bool = False):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, float))
    dn, _, trace = pcd.voxel_down_sample_and_trace(
        voxel, pcd.get_min_bound(), pcd.get_max_bound(), False)
    first = np.array([min(list(t)) for t in trace], dtype=int)
    if normals:
        dn.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    return dn, first


def fpfh(pcd) -> o3d.pipelines.registration.Feature:
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=FPFH_RADIUS, max_nn=100))


# ============================ Phase A ============================

def build_cache(deg, folder, crip_name, dark, white, wl, force=False) -> Path:
    dest = CACHE / f"frame_{deg:03d}.npz"
    if dest.exists() and not force:
        print(f"  [缓存] frame_{deg:03d}.npz 已存在，跳过")
        return dest
    scan = SCRIPT_DIR / folder
    angles = load_numeric(scan / "el_re.txt")
    coords = point_coordinates(scan, angles)
    ids = np.array(sorted(coords))
    xyz_all = np.array([coords[i] for i in ids])

    crip = np.loadtxt(CRIPPED / crip_name, delimiter=",")[:, :3]
    dist, idx = cKDTree(xyz_all).query(crip)
    if dist.max() > 1e-4:
        raise RuntimeError(f"{crip_name} 未精确匹配原始点（最大 {dist.max():.6f} m）")
    keep_ids, xyz = ids[idx], xyz_all[idx]

    spec_dir = scan / "gp" / "specData"
    spec = np.empty((len(keep_ids), len(wl)), dtype=np.float32)
    for k, pid in enumerate(keep_ids):
        spec[k] = np.loadtxt(spec_dir / f"spec_{int(pid)}.txt", usecols=(1,))
    refl = ((spec.astype(np.float64) - dark) / (white - dark)).astype(np.float32)

    np.savez_compressed(dest, deg=deg, ids=keep_ids, xyz=xyz,
                        spec=spec, refl=refl, wavelength=wl)
    print(f"  [缓存] {folder}: 原始 {len(ids)} → 保留 {len(keep_ids)} 点，光谱 {spec.shape}")
    return dest


def phase_cache(force=False) -> None:
    print("=" * 74)
    print("Phase A  原始入库：sbq+el_re → 原始 xyz ; cripped → 保留点 ; specData → 2048 波段")
    print("=" * 74)
    CACHE.mkdir(parents=True, exist_ok=True)
    dark_path, white_path = calibration_paths(SCRIPT_DIR / FRAMES[0][1], None, None)
    dark, white = np.loadtxt(dark_path)[:, 1], np.loadtxt(white_path)[:, 1]
    wl = np.loadtxt(dark_path)[:, 0]
    print(f"标定 {dark_path.name}/{white_path.name}  {len(wl)} 波段  {wl[0]:.1f}~{wl[-1]:.1f} nm")
    for deg, folder, crip_name in FRAMES:
        build_cache(deg, folder, crip_name, dark, white, wl, force)


def load_frame(deg: int) -> dict:
    z = np.load(CACHE / f"frame_{deg:03d}.npz")
    return {k: z[k] for k in z.files}


# ============================ Phase B ============================

def add_descriptors(frames: dict[int, dict]) -> dict:
    print("=" * 74)
    print("Phase B  光谱描述子（负值截断 / 波段选择 / L2 归一化 / PCA / NDVI / 红边）")
    print("=" * 74)
    wl = frames[0]["wavelength"]
    m_opt, m_eval = band_mask(wl, BAND_OPT), band_mask(wl, BAND_EVAL)
    print(f"配准波段 {BAND_OPT} → {m_opt.sum()} 维 ; 评估波段 {BAND_EVAL} → {m_eval.sum()} 维")
    i650, i700, i760, i800 = (idx_of(wl, v) for v in (650.14, 700, 760, 800))

    for f in frames.values():
        r = np.clip(f["refl"].astype(np.float64), 0.0, None)
        f["refl_clip"] = r
        f["spec_opt"] = unit_rows(r[:, m_opt])
        f["spec_eval"] = unit_rows(r[:, m_eval])
        r650, r800 = r[:, i650], r[:, i800]
        with np.errstate(invalid="ignore", divide="ignore"):
            f["ndvi"] = np.where(np.abs(r800 + r650) > 1e-9,
                                 (r800 - r650) / (r800 + r650), np.nan)
        f["rededge"] = r[:, i760] - r[:, i700]

    # 可判别性诊断（帧内邻点 vs 帧内随机）
    d, j = cKDTree(frames[0]["xyz"]).query(frames[0]["xyz"], k=2)
    nb = d[:, 1] < 0.003
    rng = np.random.default_rng(0)
    print("  可判别性诊断（0° 帧，帧内邻点<3mm vs 帧内随机对）:")
    for name, key in (("全波段", None), (f"{BAND_OPT}", "spec_opt"), (f"{BAND_EVAL}", "spec_eval")):
        if key is None:
            u = unit_rows(np.clip(frames[0]["refl"].astype(float), 0, None))
        else:
            u = frames[0][key]
        s_nb = spectral_angle_deg(u[nb], u[j[nb, 1]]).mean()
        i1, i2 = rng.integers(0, len(u), 2000), rng.integers(0, len(u), 2000)
        s_rnd = spectral_angle_deg(u[i1], u[i2]).mean()
        print(f"    {name:>16}: 邻点 SAM={s_nb:5.2f}°  随机对 SAM={s_rnd:5.2f}°")

    # PCA（配准波段，用于融合特征）
    X = np.vstack([f["spec_opt"][:: max(1, len(f["spec_opt"]) // 5000)] for f in frames.values()])
    mu = X.mean(axis=0)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1][:PCA_DIMS]
    comps, ratio = evecs[:, order], evals[order] / evals.sum()
    print(f"  PCA(配准波段): 拟合 {len(Xc)} 点 → {PCA_DIMS} 维，累计解释率 {ratio.sum()*100:.2f}%")
    for f in frames.values():
        f["spec_pca"] = unit_rows((f["spec_opt"] - mu) @ comps)

    frames["_meta"] = {"comps": comps, "mu": mu, "ratio": ratio,
                       "m_opt": m_opt, "m_eval": m_eval}
    for deg, f in frames.items():
        if deg == "_meta":
            continue
        print(f"    {deg:>3}°: {len(f['xyz']):>5} 点  NDVI 中位={np.nanmedian(f['ndvi']):.3f}  "
              f"红边中位={np.median(f['rededge']):.4f}")
    return frames


# ============================ Phase C ============================

def ransac_feature(src, tgt, f_src, f_tgt):
    best_T, best_f = np.eye(4), -1.0
    for max_d in (0.01, 0.02, 0.03):
        for ransac_n in (3, 4):
            res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                src, tgt, f_src, f_tgt, mutual_filter=True,
                max_correspondence_distance=max_d,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=ransac_n,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_d)],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 500))
            fit = o3d.pipelines.registration.evaluate_registration(
                src, tgt, RANSAC_MAX_CORR, res.transformation).fitness
            if fit > best_f:
                best_f, best_T = fit, res.transformation
    return best_T, best_f


def refine_icp(src, tgt, T0):
    best = None
    for max_d in ICP_THRESHOLDS:
        r = o3d.pipelines.registration.registration_icp(
            src, tgt, max_d, T0,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        if best is None or r.fitness > best.fitness:
            best = r
    return best


def joint_icp(sp, ss, tp, tn, ts, T0, lam=JOINT_LAMBDA, K=JOINT_K,
              max_dist=JOINT_MAX_DIST, iters=JOINT_ITERS, min_pairs=20) -> np.ndarray:
    """几何门控 + 光谱细化 ICP。

    对应代价: cost = (d/max_dist)² + λ·(SAM/10)²
    （几何距离门控 + 光谱角细化，取每源点 K 个几何近邻中代价最小者）
    """
    kd = cKDTree(tp)
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(tp)
    tgt.normals = o3d.utility.Vector3dVector(tn)
    work = o3d.geometry.PointCloud()
    work.points = o3d.utility.Vector3dVector(sp)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    T = np.array(T0, dtype=np.float64)
    for _ in range(iters):
        p = (T[:3, :3] @ sp.T).T + T[:3, 3]
        d, j = kd.query(p, k=K, distance_upper_bound=max_dist)
        d, j = np.atleast_2d(d), np.atleast_2d(j)
        valid = np.isfinite(d) & (j < len(tp))
        cost = np.full(d.shape, np.inf)
        for k in range(d.shape[1]):
            v = valid[:, k]
            if not v.any():
                continue
            sam = spectral_angle_deg(ss[v], ts[j[v, k]])
            cost[v, k] = (d[v, k] / max_dist) ** 2 + lam * (sam / 10.0) ** 2
        best = np.argmin(cost, axis=1)
        rows = np.arange(len(sp))
        ok = np.isfinite(cost[rows, best])
        if ok.sum() < min_pairs:
            break
        work.points = o3d.utility.Vector3dVector(p)
        corres = o3d.utility.Vector2iVector(
            np.stack([rows[ok], j[ok, best[ok]]], axis=1).astype(np.int32))
        T = est.compute_transformation(work, tgt, corres) @ T
    return T


def transform_metrics(pos_src, spec_src, pos_tgt, spec_tgt, T,
                      max_dist: float = FITNESS_THRESHOLD) -> dict:
    p = (T[:3, :3] @ pos_src.T).T + T[:3, 3]
    d, j = cKDTree(pos_tgt).query(p, distance_upper_bound=max_dist)
    ok = np.isfinite(d)
    out = {"fitness": float(ok.mean()), "n_pairs": int(ok.sum())}
    out["rmse_mm"] = float(np.sqrt((d[ok] ** 2).mean()) * 1000) if ok.sum() else float("nan")
    if ok.sum():
        sam = spectral_angle_deg(spec_src[ok], spec_tgt[j[ok]])
        out["sam_deg"] = float(sam.mean())
    else:
        out["sam_deg"] = float("nan")
    return out


def t_info(T) -> tuple[float, float]:
    R = np.asarray(T)[:3, :3]
    return (float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))),
            float(np.linalg.norm(np.asarray(T)[:3, 3]) * 1000))


def phase_register() -> None:
    print("=" * 74)
    print("Phase C  配准（每帧直接配到 0° 基准）")
    print("   M_ref  : 几何 FPFH + RANSAC + 点到面 ICP（复现 experiment_cripped）")
    print("   M_joint: 几何门控 + 光谱细化（融合特征 RANSAC → 联合 ICP，多初值择优）")
    print("=" * 74)
    frames = add_descriptors({deg: load_frame(deg) for deg, _, _ in FRAMES})
    meta = frames.pop("_meta")

    fb = frames[0]
    base_dn, map_b = downsample_with_map(fb["xyz"], normals=True)
    base_fpfh = fpfh(base_dn)
    base_np = np.asarray(base_dn.points)
    base_tn = np.asarray(base_dn.normals)

    results = {}
    for deg, folder, crip_name in FRAMES:
        if deg == 0:
            continue
        f = frames[deg]
        down, map_i = downsample_with_map(f["xyz"], normals=True)
        fp_src = fpfh(down)

        # ---------- M_ref：纯几何 ----------
        T0, f0 = ransac_feature(down, base_dn, fp_src, base_fpfh)
        reg = refine_icp(down, base_dn, T0)
        T_ref = reg.transformation

        # ---------- M_joint：几何 + 光谱 ----------
        g_s = unit_cols(np.asarray(fp_src.data)) / np.sqrt(np.asarray(fp_src.data).shape[0])
        s_s = f["spec_pca"][map_i].T / np.sqrt(PCA_DIMS)
        g_t = unit_cols(np.asarray(base_fpfh.data)) / np.sqrt(np.asarray(base_fpfh.data).shape[0])
        s_t = fb["spec_pca"][map_b].T / np.sqrt(PCA_DIMS)
        fused_src = make_feature(np.vstack([W_SPECTRAL * g_s, W_SPECTRAL * s_s]))
        fused_tgt = make_feature(np.vstack([W_SPECTRAL * g_t, W_SPECTRAL * s_t]))
        T_rans, f_rans = ransac_feature(down, base_dn, fused_src, fused_tgt)

        cands = []
        for init_name, T_init in (("fused_ransac", T_rans), ("identity", np.eye(4))):
            Tj = joint_icp(np.asarray(down.points), f["spec_opt"][map_i],
                           base_np, base_tn, fb["spec_opt"][map_b], T_init)
            mm = transform_metrics(np.asarray(down.points), f["spec_opt"][map_i],
                                   base_np, fb["spec_opt"][map_b], Tj)
            cands.append((init_name, Tj, mm["fitness"], mm["sam_deg"]))
        # 择优规则：几何合格(fitness≥FITNESS_GATE)前提下取光谱最一致
        ok_c = [c for c in cands if c[2] >= FITNESS_GATE]
        pick = min(ok_c, key=lambda c: c[3]) if ok_c else max(cands, key=lambda c: c[2])
        T_joint = pick[1]

        # ---------- 全分辨率三指标 ----------
        m_ref = transform_metrics(f["xyz"], f["spec_opt"], fb["xyz"], fb["spec_opt"], T_ref)
        m_jnt = transform_metrics(f["xyz"], f["spec_opt"], fb["xyz"], fb["spec_opt"], T_joint)
        e_ref = transform_metrics(f["xyz"], f["spec_eval"], fb["xyz"], fb["spec_eval"], T_ref)
        e_jnt = transform_metrics(f["xyz"], f["spec_eval"], fb["xyz"], fb["spec_eval"], T_joint)
        r_ang, r_sh = t_info(T_ref)
        j_ang, j_sh = t_info(T_joint)

        print(f"\n  {deg:>3}° （{len(f['xyz'])} 点 → 下采样 {len(down.points)}）")
        print(f"     M_ref  : 几何 RANSAC fit={f0:.3f} → ICP rot={r_ang:.2f}° shift={r_sh:.1f}mm")
        print(f"              fitness={m_ref['fitness']:.4f} rmse={m_ref['rmse_mm']:.2f}mm "
              f"SAM(500-800)={m_ref['sam_deg']:.2f}° SAM(450-900)={e_ref['sam_deg']:.2f}°")
        print(f"     M_joint: 融合 RANSAC fit={f_rans:.3f} → 联合ICP rot={j_ang:.2f}° shift={j_sh:.1f}mm "
              f"(择优={pick[0]})")
        for n, _, cf, cs in cands:
            print(f"              候选 {n:>13}: fitness={cf:.3f} SAM={cs:.2f}°")
        print(f"              fitness={m_jnt['fitness']:.4f} rmse={m_jnt['rmse_mm']:.2f}mm "
              f"SAM(500-800)={m_jnt['sam_deg']:.2f}° SAM(450-900)={e_jnt['sam_deg']:.2f}°")

        results[deg] = {
            "num_points": int(len(f["xyz"])),
            "T_ref": np.asarray(T_ref).tolist(), "T_joint": np.asarray(T_joint).tolist(),
            "ref": {**m_ref, "sam_eval_deg": e_ref["sam_deg"], "rot_deg": r_ang, "shift_mm": r_sh},
            "joint": {**m_jnt, "sam_eval_deg": e_jnt["sam_deg"], "rot_deg": j_ang, "shift_mm": j_sh},
            "joint_candidates": [{"init": n, "fitness": cf, "sam_deg": cs} for n, _, cf, cs in cands],
            "joint_pick": pick[0], "ransac_fit_ref": float(f0), "ransac_fit_joint": float(f_rans),
        }

    np.savez_compressed(CACHE / "registration.npz",
                        **{f"T_ref_{d}": np.array(r["T_ref"]) for d, r in results.items()},
                        **{f"T_joint_{d}": np.array(r["T_joint"]) for d, r in results.items()})
    (OUT / "registration").mkdir(exist_ok=True)
    (OUT / "registration" / "transforms.json").write_text(json.dumps(
        {"band_opt_nm": BAND_OPT, "band_eval_nm": BAND_EVAL,
         "pca_variance_ratio": meta["ratio"].tolist(), "frames": results},
        indent=2, ensure_ascii=False))
    print("\n[保存] registration/transforms.json")


# ============================ Phase D ============================

def phase_metrics() -> None:
    print("=" * 74)
    print("Phase D  三指标对比（① fitness  ② inlier_rmse  ③ 光谱一致性 SAM）")
    print("=" * 74)
    data = json.loads((OUT / "registration" / "transforms.json").read_text())
    rows = []
    for deg, r in data["frames"].items():
        for tag, name in (("ref", "M_ref"), ("joint", "M_joint")):
            m = r[tag]
            rows.append({"deg": int(deg), "method": name, "fitness": m["fitness"],
                         "rmse_mm": m["rmse_mm"], "sam_deg": m["sam_deg"],
                         "sam_eval_deg": m["sam_eval_deg"], "rot_deg": m["rot_deg"],
                         "shift_mm": m["shift_mm"], "n_pairs": m["n_pairs"]})

    print(f"\n{'帧':>5} {'方法':>8} {'fitness↑':>9} {'rmse(mm)↓':>10} {'SAM(°)↓':>9} "
          f"{'SAM宽(°)↓':>10} {'旋转(°)':>9} {'平移(mm)':>10}")
    for r in sorted(rows, key=lambda x: (x["deg"], x["method"])):
        print(f"{r['deg']:>5} {r['method']:>8} {r['fitness']:>9.4f} {r['rmse_mm']:>10.2f} "
              f"{r['sam_deg']:>9.2f} {r['sam_eval_deg']:>10.2f} {r['rot_deg']:>9.2f} {r['shift_mm']:>10.1f}")

    def agg(method, key):
        v = [r[key] for r in rows if r["method"] == method and np.isfinite(r[key])]
        return float(np.mean(v)) if v else float("nan")

    summary = {}
    print("\n" + "=" * 78)
    print("三指标均值对比（5 帧平均）")
    print("=" * 78)
    print(f"{'指标':>16} {'M_ref (几何/RGB)':>18} {'M_joint (原始xyz+光谱)':>24} {'变化':>12}")
    for key, label, good in (("fitness", "① fitness ↑", True),
                             ("rmse_mm", "② rmse(mm) ↓", False),
                             ("sam_deg", "③ SAM(500-800nm)° ↓", False),
                             ("sam_eval_deg", "③b SAM(450-900nm)° ↓", False)):
        a, b = agg("M_ref", key), agg("M_joint", key)
        summary[key] = {"M_ref": a, "M_joint": b}
        print(f"{label:>18} {a:>18.4f} {b:>24.4f} {b - a:>+12.4f}")
    (OUT / "metrics.json").write_text(json.dumps(
        {"per_frame": rows, "summary": summary, "band_opt_nm": data["band_opt_nm"],
         "band_eval_nm": data["band_eval_nm"]}, indent=2, ensure_ascii=False))
    print("\n[保存] metrics.json")


# ============================ Phase E ============================

# ============================ Phase E ============================

def render_views(pts: np.ndarray, cols: np.ndarray, path: Path,
                 views: list[tuple[int, int]], title: str, point_size: float = 1.2) -> None:
    """暗背景多视角渲染（颜色直接用传入值，不再二次放大）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(views)
    fig = plt.figure(figsize=(5 * n, 6), facecolor="black")
    fig.suptitle(title, color="white", fontsize=11)
    for i, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, n, i + 1, projection="3d", facecolor="black")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, c=cols, marker=".",
                   linewidths=0)
        ax.view_init(el, az)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)
    print(f"  [渲染] {path}")


def render_gif(pts: np.ndarray, cols: np.ndarray, path: Path, n_frames: int = 36) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig = plt.figure(figsize=(5, 5), facecolor="black")
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1.2, c=cols, marker=".", linewidths=0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()

    def upd(f):
        ax.view_init(elev=20, azim=f * 360 / n_frames)
        return (ax,)

    FuncAnimation(fig, upd, frames=n_frames, interval=90).save(path, writer="pillow", dpi=90)
    plt.close(fig)
    print(f"  [渲染] {path}")


def phase_visual() -> None:
    print("=" * 74)
    print("Phase E  配准后由原始光谱转 RGB 可视化（957/730/552 ×4）+ NDVI 伪彩")
    print("=" * 74)

    reg = np.load(CACHE / "registration.npz")
    dpath, wpath = calibration_paths(SCRIPT_DIR / FRAMES[0][1], None, None)
    dark, white = np.loadtxt(dpath)[:, 1], np.loadtxt(wpath)[:, 1]
    vis, regout = OUT / "visualization", OUT / "registration"
    vis.mkdir(exist_ok=True)
    wl = load_frame(FRAMES[0][0])["wavelength"]
    i650, i800 = idx_of(wl, 650.14), idx_of(wl, 800.0)

    pts_all, rgb_all, nd_all = [], [], []
    for deg, _, _ in FRAMES:
        f = load_frame(deg)
        T = np.eye(4) if deg == 0 else reg[f"T_joint_{deg}"]
        xyz = (T[:3, :3] @ f["xyz"].T).T + T[:3, 3]
        idx = [b - 1 for b in RGB_BANDS_1BASED]
        # 注意：f["refl"] 已是标定反射率 (raw-dark)/(white-dark)，此处只需 ×4（与 convert_to_txt 一致）
        rgb = np.clip(f["refl"][:, idx].astype(float) * 4.0, 0, 1)
        np.savetxt(regout / f"aligned_joint_{deg:03d}.txt",
                   np.hstack([xyz, rgb, np.zeros((len(xyz), 2))]), delimiter=",", fmt="%.6f")
        r = np.clip(f["refl"].astype(float), 0, None)
        with np.errstate(invalid="ignore", divide="ignore"):
            nd = np.where(np.abs(r[:, i800] + r[:, i650]) > 1e-9,
                          (r[:, i800] - r[:, i650]) / (r[:, i800] + r[:, i650]), np.nan)
        pts_all.append(xyz); rgb_all.append(rgb); nd_all.append(nd)
    pts, rgb, nd = np.vstack(pts_all), np.vstack(rgb_all), np.concatenate(nd_all)
    uniq = np.unique(np.round(pts, 6), axis=0, return_index=True)[1]
    pts, rgb, nd = pts[uniq], rgb[uniq], nd[uniq]
    print(f"合并点云 {len(pts)} 点  bbox x[{pts[:,0].min():+.3f},{pts[:,0].max():+.3f}] "
          f"y[{pts[:,1].min():+.3f},{pts[:,1].max():+.3f}] z[{pts[:,2].min():+.3f},{pts[:,2].max():+.3f}]")
    print(f"RGB(×4) 均值={rgb.mean(axis=0).round(4)}  亮度均值={rgb.mean():.4f}")

    # 显示增强：按亮度 p95 拉伸（保持色调）+ gamma 0.8（温和提亮暗部）
    scale = float(np.percentile(rgb.max(axis=1), 95)) or 1.0
    rgb_vis = np.clip(rgb / max(scale, 1e-6), 0, 1) ** 0.8
    print(f"显示增强: p95 拉伸因子={scale:.4f} + gamma 0.8 → 亮度均值={rgb_vis.mean():.4f} "
          f"通道均值={rgb_vis.mean(axis=0).round(3)}")

    np.savetxt(OUT / "combined.txt", np.hstack([pts, rgb, np.zeros((len(pts), 2))]),
               delimiter=",", fmt="%.6f")
    np.savetxt(OUT / "combined_enhanced.txt",
               np.hstack([pts, rgb_vis, np.zeros((len(pts), 2))]), delimiter=",", fmt="%.6f")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb_vis, 0, 1))
    o3d.io.write_point_cloud(str(OUT / "combined.pcd"), pcd)
    render_views(pts, np.clip(rgb_vis, 0, 1), OUT / "combined.png",
                 [(22, -60), (22, 30), (70, 30), (0, 0)],
                 "experiment_rawdata：原始 x,y,z + 全光谱联合配准（RGB 由原始光谱转出，p95+gamma 增强）")
    render_gif(pts, np.clip(rgb_vis, 0, 1), OUT / "combined_rot.gif")

    # 颜色增强对比图（拉伸前 / 拉伸后）
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(pts)
    pcd_raw.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1))
    render_views(pts, np.clip(rgb, 0, 1), vis / "color_before_enhance.png",
                 [(22, -60), (22, 30), (70, 30)], "增强前：RGB = 标定反射率 ×4（偏暗）")
    render_views(pts, np.clip(rgb_vis, 0, 1), vis / "color_after_enhance.png",
                 [(22, -60), (22, 30), (70, 30)], "增强后：p95 拉伸 + gamma 0.8（颜色鲜明）")

    ok = np.isfinite(nd)
    v = np.clip((nd[ok] - 0.2) / 0.7, 0, 1)
    cmap = np.stack([1 - v, 0.15 + 0.85 * v, 0.2 * (1 - v)], axis=1)
    pn = o3d.geometry.PointCloud()
    pn.points = o3d.utility.Vector3dVector(pts[ok])
    pn.colors = o3d.utility.Vector3dVector(np.clip(cmap, 0, 1))
    o3d.io.write_point_cloud(str(vis / "combined_ndvi.pcd"), pn)
    render_views(pts[ok], np.clip(cmap, 0, 1), vis / "combined_ndvi.png",
                 [(22, -60), (22, 30), (70, 30)], "NDVI 伪彩（红=低 / 绿=高）")
    print("[保存] combined.txt/.pcd/.png/.gif + combined_enhanced.txt + visualization/*")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["cache", "register", "metrics", "visual", "all"])
    ap.add_argument("--force-cache", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.phase in ("cache", "all"):
        phase_cache(a.force_cache)
    if a.phase in ("register", "all"):
        phase_register()
    if a.phase in ("metrics", "all"):
        phase_metrics()
    if a.phase in ("visual", "all"):
        phase_visual()


if __name__ == "__main__":
    main()