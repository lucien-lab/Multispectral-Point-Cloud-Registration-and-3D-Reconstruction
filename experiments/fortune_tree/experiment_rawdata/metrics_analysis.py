#!/usr/bin/env python3
"""experiment_rawdata 指标分析 —— 三指标（fitness / rmse / 光谱一致性 SAM）的可靠性分析

回答四个问题：
  ① 参考基线：恒等（不配准）、随机刚体变换、名义角度旋转 —— 指标落在哪个区间？
     20mm 阈值下 fitness 是否已饱和（天花板效应）？
  ② 阈值敏感性：fitness / rmse 在 5/10/20/30 mm 下如何变化？哪个阈值有判别力？
  ③ 配对检验：M_joint 相对 M_ref 的逐帧差值是否显著（n=5）？效应量多大？
  ④ 指标间关系：fitness / rmse / SAM / 重叠点数 / 旋转量 的相关性；是否存在此消彼长？

输出：metrics_analysis.json、visualization/metrics_analysis.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy import stats

OUT = Path(__file__).resolve().parent
CACHE = OUT / "cache"
FRAMES = [0, 60, 120, 180, 240, 300]
THRESHOLDS = (0.005, 0.010, 0.020, 0.030)
BAND_OPT, BAND_EVAL = (500.0, 800.0), (450.0, 900.0)


def load(deg: int) -> dict:
    z = np.load(CACHE / f"frame_{deg:03d}.npz")
    return {k: z[k] for k in z.files}


def unit(w):
    w = np.asarray(w, float)
    return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-12)


def sam(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=1), -1, 1)))


def evaluate(src_xyz, src_spec, tgt_xyz, tgt_spec, T, tau, mutual=False):
    """返回 fitness / rmse / SAM（可要求互为最近邻）"""
    p = (T[:3, :3] @ src_xyz.T).T + T[:3, 3]
    kd_t = cKDTree(tgt_xyz)
    d, j = kd_t.query(p, distance_upper_bound=tau)
    ok = np.isfinite(d)
    if mutual and ok.sum():
        kd_s = cKDTree(p)
        _, i2 = kd_s.query(tgt_xyz[j[ok]], distance_upper_bound=tau)
        idx = np.arange(len(p))[ok]
        keep = (i2 == idx)
        # 重建索引
        new_ok = np.zeros(len(p), bool)
        new_ok[idx[keep]] = True
        ok = new_ok
        d = np.where(ok, d, np.nan)
    n = int(ok.sum())
    if n == 0:
        return {"fitness": 0.0, "rmse_mm": float("nan"), "sam_deg": float("nan"), "n_pairs": 0}
    return {"fitness": float(ok.mean()),
            "rmse_mm": float(np.sqrt((d[ok] ** 2).mean()) * 1000),
            "sam_deg": float(sam(src_spec[ok], tgt_spec[j[ok]]).mean()),
            "n_pairs": n}


def rand_transform(rng):
    from scipy.spatial.transform import Rotation
    R = Rotation.random(random_state=rng).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.uniform(-0.15, 0.15, 3)
    return T


def nominal_transform(base_xyz, angle_deg):
    """绕树竖轴（过 0° 帧质心）旋转 angle_deg"""
    cx, cy = base_xyz[:, 0].mean(), base_xyz[:, 1].mean()
    t = np.radians(angle_deg)
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    T[0, 3], T[1, 3] = cx - c * cx + s * cy, cy - s * cx - c * cy
    return T


def main() -> None:
    reg = np.load(CACHE / "registration.npz")
    data = json.loads((OUT / "registration" / "transforms.json").read_text())
    wl = load(0)["wavelength"]
    mo, me = (wl >= BAND_OPT[0]) & (wl <= BAND_OPT[1]), (wl >= BAND_EVAL[0]) & (wl <= BAND_EVAL[1])

    F = {}
    for deg in FRAMES:
        f = load(deg)
        r = np.clip(f["refl"].astype(float), 0, None)
        F[deg] = {"xyz": f["xyz"], "so": unit(r[:, mo]), "se": unit(r[:, me])}
    B = F[0]
    rng = np.random.default_rng(42)
    results = {"config": {"band_opt_nm": BAND_OPT, "band_eval_nm": BAND_EVAL,
                          "thresholds_mm": [t * 1000 for t in THRESHOLDS],
                          "n_random": 20}}

    # ================= ① 参考基线 =================
    print("=" * 96)
    print("① 参考基线（5 帧平均，阈值 20mm）：恒等 / 随机刚体变换 / 名义角 / M_ref / M_joint")
    print("=" * 96)
    rand_rows = {deg: [] for deg in FRAMES[1:]}
    for _ in range(20):
        for deg in FRAMES[1:]:
            rand_rows[deg].append(evaluate(F[deg]["xyz"], F[deg]["so"], B["xyz"], B["so"],
                                           rand_transform(rng), 0.02))
    base_rows = []
    for deg in FRAMES[1:]:
        row = {"deg": deg}
        for name, T in (("identity", np.eye(4)),
                        ("nominal", nominal_transform(B["xyz"], deg)),
                        ("M_ref", reg[f"T_ref_{deg}"]),
                        ("M_joint", reg[f"T_joint_{deg}"])):
            row[name] = evaluate(F[deg]["xyz"], F[deg]["so"], B["xyz"], B["so"], T, 0.02)
        rr = rand_rows[deg]
        row["random"] = {k: float(np.mean([x[k] for x in rr])) for k in ("fitness", "rmse_mm", "sam_deg")}
        base_rows.append(row)

    hdr = f"{'配置':>10} {'fitness ↑':>10} {'rmse(mm) ↓':>11} {'SAM(°) ↓':>10}"
    print(hdr); print("-" * len(hdr))
    for name in ("random", "identity", "nominal", "M_ref", "M_joint"):
        fv = np.mean([r[name]["fitness"] for r in base_rows])
        rv = np.mean([r[name]["rmse_mm"] for r in base_rows])
        sv = np.mean([r[name]["sam_deg"] for r in base_rows])
        print(f"{name:>10} {fv:>10.4f} {rv:>11.2f} {sv:>10.2f}")
    results["baselines"] = {name: {
        "fitness": float(np.mean([r[name]["fitness"] for r in base_rows])),
        "rmse_mm": float(np.mean([r[name]["rmse_mm"] for r in base_rows])),
        "sam_deg": float(np.mean([r[name]["sam_deg"] for r in base_rows]))}
        for name in ("random", "identity", "nominal", "M_ref", "M_joint")}

    # 光谱指标的参考区间
    d0, j0 = cKDTree(B["xyz"]).query(B["xyz"], k=2)
    nb = d0[:, 1] < 0.003
    i1, i2 = rng.integers(0, len(B["so"]), 3000), rng.integers(0, len(B["so"]), 3000)
    sam_low, sam_high = float(sam(B["so"][nb], B["so"][j0[nb, 1]]).mean()), float(sam(B["so"][i1], B["so"][i2]).mean())
    results["sam_reference"] = {"same_surface_neighbor_deg": sam_low, "random_pair_deg": sam_high}
    print(f"\n光谱指标可达区间（0° 帧内）: 同表面邻点 {sam_low:.2f}°  ←→  随机点对 {sam_high:.2f}°")

    # ================= ② 阈值敏感性 =================
    print("\n" + "=" * 96)
    print("② 阈值敏感性：fitness / rmse 随距离阈值变化（5 帧平均）")
    print("=" * 96)
    sens = {}
    for name in ("identity", "M_ref", "M_joint"):
        sens[name] = {}
        for tau in THRESHOLDS:
            vals = []
            for deg in FRAMES[1:]:
                T = (np.eye(4) if name == "identity"
                     else reg[f"T_{'ref' if name == 'M_ref' else 'joint'}_{deg}"])
                vals.append(evaluate(F[deg]["xyz"], F[deg]["so"], B["xyz"], B["so"], T, tau))
            sens[name][f"{tau*1000:.0f}mm"] = {
                "fitness": float(np.mean([v["fitness"] for v in vals])),
                "rmse_mm": float(np.mean([v["rmse_mm"] for v in vals]))}
    print(f"{'配置':>10} " + " ".join(f"{t*1000:>7.0f}mm" for t in THRESHOLDS))
    for name in ("identity", "M_ref", "M_joint"):
        print(f"{name:>10} " + " ".join(f"{sens[name][f'{t*1000:.0f}mm']['fitness']:>9.4f}" for t in THRESHOLDS))
    print("   rmse(mm):")
    for name in ("identity", "M_ref", "M_joint"):
        print(f"{name:>10} " + " ".join(f"{sens[name][f'{t*1000:.0f}mm']['rmse_mm']:>9.2f}" for t in THRESHOLDS))
    results["threshold_sensitivity"] = sens

    # ================= ③ 配对检验 =================
    print("\n" + "=" * 96)
    print("③ 配对检验：M_joint − M_ref 逐帧差值（n=5）")
    print("=" * 96)
    d = {k: [] for k in ("fitness", "rmse_mm", "sam_deg", "sam_eval_deg")}
    for deg, r in data["frames"].items():
        for k in ("fitness", "rmse_mm", "sam_deg"):
            d[k].append(r["joint"][k] - r["ref"][k])
        d["sam_eval_deg"].append(r["joint"]["sam_eval_deg"] - r["ref"]["sam_eval_deg"])
    pretty = {"fitness": "① fitness", "rmse_mm": "② rmse(mm)", "sam_deg": "③ SAM(500-800)°",
              "sam_eval_deg": "③b SAM(450-900)°"}
    print(f"{'指标':>20} {'均值Δ':>10} {'标准差':>10} {'效应量':>8} {'改善帧数':>9} {'符号检验p':>10}")
    paired = {}
    for k, v in d.items():
        v = np.array(v)
        better = np.sum(v < 0) if k != "fitness" else np.sum(v > 0)
        p = stats.binomtest(int(better), len(v), 0.5, alternative="greater").pvalue
        eff = v.mean() / (v.std(ddof=1) + 1e-12)
        paired[k] = {"mean_delta": float(v.mean()), "std": float(v.std(ddof=1)),
                     "effect_size": float(eff), "n_better": int(better), "n": len(v),
                     "sign_test_p": float(p)}
        print(f"{pretty[k]:>20} {v.mean():>+10.4f} {v.std(ddof=1):>10.4f} {eff:>8.2f} "
              f"{int(better)}/{len(v):<7} {p:>10.4f}")
    results["paired_test"] = paired

    # ================= ④ 指标间关系 =================
    print("\n" + "=" * 96)
    print("④ 指标间关系（10 个解 = 5 帧 × 2 方法）")
    print("=" * 96)
    keys = ["fitness", "rmse_mm", "sam_deg", "rot_deg", "shift_mm", "n_pairs"]
    M = []
    for deg, r in data["frames"].items():
        for tag in ("ref", "joint"):
            M.append([r[tag][k] for k in keys])
    M = np.array(M, float)
    cm = np.corrcoef(M.T)
    print(f"{'':>10} " + " ".join(f"{k:>10}" for k in keys))
    for i, k in enumerate(keys):
        print(f"{k:>10} " + " ".join(f"{cm[i, j]:>10.2f}" for j in range(len(keys))))
    print("\n解读：fitness 与 rmse 强负相关（同一几何量的两种表达）；")
    print("      SAM 与 fitness/rmse 的相关性 → 反映光谱指标是否与几何指标同步（互补 = 低相关）")
    results["correlation"] = {k: {k2: float(cm[i, j]) for j, k2 in enumerate(keys)}
                              for i, k in enumerate(keys)}

    # 额外：互最近邻口径下的稳健性
    print("\n" + "=" * 96)
    print("⑤ 稳健性：改用「互为最近邻」口径重算（阈值 20mm）")
    print("=" * 96)
    rob = {}
    for name in ("M_ref", "M_joint"):
        vals = []
        for deg in FRAMES[1:]:
            T = reg[f"T_{'ref' if name == 'M_ref' else 'joint'}_{deg}"]
            vals.append(evaluate(F[deg]["xyz"], F[deg]["so"], B["xyz"], B["so"], T, 0.02, mutual=True))
        rob[name] = {"fitness": float(np.mean([v["fitness"] for v in vals])),
                     "rmse_mm": float(np.mean([v["rmse_mm"] for v in vals])),
                     "sam_deg": float(np.mean([v["sam_deg"] for v in vals]))}
        print(f"{name:>10} fitness={rob[name]['fitness']:.4f} rmse={rob[name]['rmse_mm']:.2f}mm "
              f"SAM={rob[name]['sam_deg']:.2f}°")
    results["mutual_nn"] = rob

    (OUT / "metrics_analysis.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[保存] metrics_analysis.json")

    # ================= 图 =================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    names = ["random", "identity", "nominal", "M_ref", "M_joint"]
    cols = ["#999999", "#4C72B0", "#DD8452", "#C44E52", "#55A868"]

    ax = axes[0]
    x = np.arange(len(names))
    w = 0.27
    ax.bar(x - w, [results["baselines"][n]["fitness"] for n in names], w, label="fitness@20mm ↑")
    ax.bar(x, [results["baselines"][n]["sam_deg"] / 20 for n in names], w,
           label="SAM/20 (°) ↓")
    ax.bar(x + w, [results["baselines"][n]["rmse_mm"] / 20 for n in names], w,
           label="rmse/20 (mm) ↓")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_title("① 参考基线（5 帧平均）")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    for n, c in (("identity", cols[1]), ("M_ref", cols[3]), ("M_joint", cols[4])):
        ax.plot([t * 1000 for t in THRESHOLDS],
                [sens[n][f"{t*1000:.0f}mm"]["fitness"] for t in THRESHOLDS],
                "o-", color=c, label=n)
    ax.set_xlabel("距离阈值 (mm)"); ax.set_ylabel("fitness")
    ax.set_title("② 阈值敏感性：20mm 处已饱和")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    rot = [r[tag]["rot_deg"] for r in data["frames"].values() for tag in ("ref", "joint")]
    samv = [r[tag]["sam_deg"] for r in data["frames"].values() for tag in ("ref", "joint")]
    ax.scatter(rot, samv, c="#C44E52")
    ax.axhline(sam_low, color="g", ls="--", label=f"同表面邻点 {sam_low:.1f}°")
    ax.axhline(sam_high, color="r", ls="--", label=f"随机点对 {sam_high:.1f}°")
    ax.axhline(results["baselines"]["identity"]["sam_deg"], color="b", ls=":",
               label=f"identity {results['baselines']['identity']['sam_deg']:.1f}°")
    ax.set_xlabel("配准施加的旋转量 (°)"); ax.set_ylabel("SAM (°)")
    ax.set_title(f"③ 光谱一致性与变换幅度 (r={cm[keys.index('sam_deg'), keys.index('rot_deg')]:.2f})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "visualization" / "metrics_analysis.png", dpi=120)
    print(f"[保存] visualization/metrics_analysis.png")


if __name__ == "__main__":
    main()
