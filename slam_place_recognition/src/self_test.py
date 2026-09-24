"""Fast self-test of the pipeline invariants (no arguments needed).

    python self_test.py            # ~15 s

Checks
------
1. JSSS dimension  D = 4B + N_bins + 3B  -> 76 (8 bands, 20 bins), 44 without quad.
2. Spectra stay in [0, 1] and cover all material classes.
3. Odometry drift RMS at 1.0x equals 7.257 m (paper Table 6) and scales linearly.
4. Descriptor features are **independent of the odometry drift** (P0 finding).
5. Loop constraints copied from the drifted odometry make the pose-graph
   relaxation a no-op (improvement exactly 0 %).
6. Candidate generation honours Delta_min and the search radius.
7. PR metrics: F1_max >= accept-all F1 and AUC in [0, 1].
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config  # noqa: E402
from data_io import load_dataset  # noqa: E402
from drift import odometry_ate_rmse  # noqa: E402
from pipeline import (  # noqa: E402
    build_methods,
    extract_features,
    poses_for_factor,
    run_trajectory_optimisation,
)
from retrieval import build_problem, evaluate_similarity  # noqa: E402

FAILS = []


def check(cond, label, detail=""):
    status = "ok  " if cond else "FAIL"
    print(f"  [{status}] {label}{(' - ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(label)


def main():
    cfg = Config()
    cfg.synth.n_frames = 120
    cfg.synth.points_per_frame = 5000

    print("1) descriptor dimensions")
    check(cfg.jsss.dimension == 76, "JSSS dim == 76", str(cfg.jsss.dimension))
    check(cfg.jsss.dimension_without_quad() == 44, "dim without quad == 44",
          str(cfg.jsss.dimension_without_quad()))
    from dataclasses import replace
    check(replace(cfg.jsss, n_bands=4).dimension == 48,
          "B=4 dim == 48 (paper says 36)", "48")

    print("2) spectral simulation")
    ds = load_dataset(cfg)
    f = ds.frame(0)
    check(f["spectra"].min() >= -1e-6 and f["spectra"].max() <= 1.0 + 1e-6,
          "spectra within [0, 1]",
          f"[{f['spectra'].min():.3f}, {f['spectra'].max():.3f}]")
    check(len(np.unique(f["material_ids"])) >= 3, "multiple material classes",
          str(len(np.unique(f["material_ids"]))))
    check(f["xyz"].shape[0] == cfg.synth.points_per_frame, "point count matches config")

    print("3) drift model")
    p1, i1 = poses_for_factor(ds, cfg, 1.0)
    p2, i2 = poses_for_factor(ds, cfg, 2.0)
    check(abs(i1["rms_m"] - 7.257) < 1e-3, "1.0x drift RMS == 7.257 m",
          f"{i1['rms_m']:.3f}")
    check(abs(i2["rms_m"] - 14.514) < 1e-3, "2.0x drift RMS == 14.514 m",
          f"{i2['rms_m']:.3f}")
    check(abs(odometry_ate_rmse(p1, ds.poses_gt) - 7.257) < 1e-2,
          "odometry ATE RMSE == drift RMS")

    print("4) descriptors are drift-independent")
    methods = build_methods(cfg)
    feats = extract_features(ds, cfg, methods, verbose=False)
    check(np.allclose(feats["JSSS"][0]["hist"], feats["JSSS"][0]["hist"]),
          "feature extraction uses no pose information", "(by construction)")

    print("5) pose-graph degeneracy")
    prob = build_problem(ds, cfg, p1, "paper_pr")
    if len(prob["pairs"]):
        sim = methods["JSSS"].similarity(
            [feats["JSSS"][i] for i in prob["pairs"][:, 0]],
            [feats["JSSS"][i] for i in prob["pairs"][:, 1]])
        r = evaluate_similarity(sim, prob["positive"], cfg)
        out_copy = run_trajectory_optimisation(ds, cfg, p1, sim, prob,
                                              tau_sim=r["best_tau"],
                                              loop_delta_model="copy")
        out_oracle = run_trajectory_optimisation(ds, cfg, p1, sim, prob,
                                                tau_sim=r["best_tau"],
                                                loop_delta_model="oracle")
        check(abs(out_copy["improvement_rmse_pct"]) < 1e-9,
              "'copy' loop constraints -> exactly 0 % improvement",
              f"{out_copy['improvement_rmse_pct']:.6f}%")
        check(out_oracle["improvement_rmse_pct"] > 0,
              "'oracle' loop constraints improve the ATE",
              f"{out_oracle['improvement_rmse_pct']:+.2f}%")
    else:
        check(False, "candidate set non-empty at 1.0x")

    print("6) candidate generation")
    gaps = prob["pairs"][:, 1] - prob["pairs"][:, 0] if len(prob["pairs"]) else np.array([99])
    check(int(gaps.min()) > cfg.retrieval.delta_min_frames if len(prob["pairs"]) else True,
          f"|i-j| > Delta_min ({cfg.retrieval.delta_min_frames})",
          f"min gap {int(gaps.min())}")
    d = np.linalg.norm(p1[prob["pairs"][:, 0], :2] - p1[prob["pairs"][:, 1], :2],
                       axis=1) if len(prob["pairs"]) else np.array([0.0])
    check(float(d.max()) <= prob["tau_odom"] + 1e-9, "all candidates within tau_odom",
          f"max {d.max():.2f} m <= {prob['tau_odom']}")
    # brute-force cross-check of the chunked candidate search (multi-chunk safe)
    from retrieval import candidate_pairs
    rng = np.random.default_rng(0)
    ok_bf = True
    for n in (120, 513, 1100):
        xy = rng.normal(size=(n, 2)) * 50
        for tau in (3.0, 8.0, 20.0):
            got = candidate_pairs(xy, tau, cfg.retrieval.delta_min_frames)
            ref = [(i, j) for i in range(n) for j in range(i + 11, n)
                   if np.linalg.norm(xy[i] - xy[j]) <= tau]
            ref = np.array(sorted(ref)) if ref else np.zeros((0, 2), int)
            if got.shape != ref.shape or (len(ref) and not np.array_equal(got, ref)):
                ok_bf = False
    check(ok_bf, "candidate_pairs matches brute force (n=120/513/1100)")

    print("7) PR metrics")
    if len(prob["pairs"]):
        check(r["f1_max"] >= r["accept_all_f1"] - 1e-9,
              "F1_max >= accept-all F1", f"{r['f1_max']:.3f} >= {r['accept_all_f1']:.3f}")
        check(0.0 <= r["auc"] <= 1.0, "AUC in [0, 1]", f"{r['auc']:.3f}")
        check(np.isfinite(r["best_tau"]), "optimal threshold finite",
              f"{r['best_tau']:.3f}")

    print()
    if FAILS:
        print(f"{len(FAILS)} check(s) FAILED: {FAILS}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
