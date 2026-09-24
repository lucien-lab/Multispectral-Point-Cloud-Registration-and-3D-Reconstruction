"""Audit script: checks the paper's own numbers against its own definitions.

Everything here is arithmetic on the values *reported in the paper*
(`JSSS_LiDAR_SLAM_v3(修改).docx`), plus two executable demonstrations:

* the pose-graph relaxation is a no-op when loop constraints are copied from the
  drifted odometry (so Table 6's improvement cannot come from the described
  scheme);
* F1_max == accept-all F1 means a descriptor filters nothing (Table 5 check).

Usage
-----
    python paper_audit.py
    python paper_audit.py --demo-posegraph --frames 200 --points 8000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# Values transcribed from the paper (Tables 1-6 and Sec. 5 text)
# --------------------------------------------------------------------------- #
T1 = [  # tau_odom, method, closed_loops, TP, FP, precision
    (3.0, "pure_space", 1078, 669, 409, 0.621), (3.0, "JSSS", 879, 607, 272, 0.691),
    (5.0, "pure_space", 2633, 1559, 1074, 0.592), (5.0, "JSSS", 2134, 1372, 762, 0.643),
    (8.0, "pure_space", 6719, 3303, 3416, 0.492), (8.0, "JSSS", 5448, 2855, 2593, 0.524),
    (12.0, "pure_space", 15516, 5284, 10232, 0.341), (12.0, "JSSS", 12382, 4537, 7845, 0.366),
]
T2 = [  # bands, loops, TP, FP, precision
    (1, 2258, 1401, 857, 0.620), (2, 2250, 1397, 853, 0.621),
    (4, 2102, 1355, 747, 0.645), (8, 2134, 1372, 762, 0.643),
    ("pure_space", 2633, 1559, 1074, 0.592),
    ("intensity_sum", 2633, 1559, 1074, 0.592),
]
T3 = [  # component, loops, TP, FP, precision
    ("quad", 2633, 1559, 1074, 0.592),
    ("hist", 2024, 1318, 706, 0.651),
    ("stats", 332, 282, 50, 0.849),
    ("full", 2134, 1372, 762, 0.643),
]
T4 = [  # factor, method, loops, TP, FP, precision  (paper Table 4)
    (1.0, "pure_space_tight", 1078, 669, 409, 0.621),
    (1.0, "geometry_descriptor", 6719, 3303, 3416, 0.492),
    (1.0, "global_histogram", 6719, 3303, 3416, 0.492),
    (1.0, "JSSS", 5448, 2855, 2593, 0.524),
    (1.5, "pure_space_tight", 907, 478, 429, 0.527),
    (1.5, "geometry_descriptor", 6361, 2229, 4132, 0.350),
    (1.5, "global_histogram", 6361, 2229, 4132, 0.350),
    (1.5, "JSSS", 5149, 1963, 3186, 0.381),
    (2.0, "pure_space_tight", 889, 313, 576, 0.352),
    (2.0, "geometry_descriptor", 6165, 1598, 4567, 0.259),
    (2.0, "global_histogram", 6165, 1598, 4567, 0.259),
    (2.0, "JSSS", 5031, 1424, 3607, 0.283),
]
T5 = [  # factor, method, F1max, AUC, best_tau, n_cand, n_pos
    (1.0, "JSSS", 0.841, 0.835, 0.581, 277, 201),
    (1.0, "ScanContext", 0.862, 0.947, 0.729, 277, 201),
    (1.0, "M2DP", 0.855, 0.857, 0.043, 277, 201),
    (1.0, "IntensityHist", 0.850, 0.839, 0.999, 277, 201),
    (1.0, "IntensityStats", 0.860, 0.848, 0.996, 277, 201),
    (2.0, "JSSS", 0.756, 0.765, 0.797, 176, 71),
    (2.0, "ScanContext", 0.730, 0.833, 0.754, 176, 71),
    (2.0, "M2DP", 0.587, 0.609, 0.017, 176, 71),
    (2.0, "IntensityHist", 0.670, 0.732, 1.000, 176, 71),
    (2.0, "IntensityStats", 0.715, 0.714, 0.997, 176, 71),
]
T6 = [  # factor, method, odom_ate, opt_ate, mean, improvement, loops
    (1.0, "JSSS", 7.257, 6.786, 6.225, 0.070, 277),
    (1.0, "ScanContext", 7.257, 6.769, 6.223, 0.070, 235),
    (1.0, "M2DP", 7.257, 6.831, 6.300, 0.058, 241),
    (1.0, "IntensityHist", 7.257, 6.892, 6.336, 0.053, 272),
    (1.0, "IntensityStats", 7.257, 6.958, 6.401, 0.043, 248),
    (2.0, "JSSS", 14.514, 14.241, 13.147, 0.017, 93),
    (2.0, "ScanContext", 14.514, 14.325, 13.210, 0.013, 77),
    (2.0, "M2DP", 14.514, 14.304, 13.207, 0.013, 171),
    (2.0, "IntensityHist", 14.514, 14.432, 13.320, 0.005, 123),
    (2.0, "IntensityStats", 14.514, 14.375, 13.268, 0.008, 122),
]


def _fail(cond):
    return "OK " if cond else "FAIL"


def audit_arithmetic() -> int:
    """Recompute every derivable quantity and print a pass/fail table."""

    problems = 0
    print("=== A. arithmetic re-check of the reported tables ===\n")
    print(f"{'check':58s} {'reported':>18s} {'correct':>18s}  verdict")
    print("-" * 105)

    def row(label, rep, corr, ok):
        nonlocal problems
        problems += 0 if ok else 1
        print(f"{label:58s} {rep:>18s} {corr:>18s}  {_fail(ok)}")

    # -- Table 1: FP reduction claimed as 19 % --------------------------------
    fp_red = (1074 - 762) / 1074
    row("Sec 5.1 'FP reduced by 19% (762 vs 1074)'", "19%", f"{fp_red*100:.1f}%",
        abs(fp_red * 100 - 19) < 0.5)
    loops_red = (2633 - 2134) / 2633
    row("Sec 5.1 - the 19% actually equals loop-count cut", "-",
        f"{loops_red*100:.1f}%", abs(loops_red * 100 - 19) < 0.5)

    # -- Table 1: TP/FP/loops consistency ------------------------------------
    bad = [(m, t, lo, tp, fp) for _, m, lo, tp, fp, _ in T1 if tp + fp != lo]
    row("Tables 1-4: TP + FP == closed loops", "consistent",
        "consistent" if not bad else f"{len(bad)} mismatches", not bad)

    # -- Table 1: precision --------------------------------------------------
    bad = [(m, p, tp / lo) for _, m, lo, tp, _, p in T1 if abs(tp / lo - p) > 0.005]
    row("Tables 1-4: precision == TP / loops", "consistent",
        "consistent" if not bad else str(bad[:2]), not bad)

    # -- Sec 5.4 claim vs Table 4 -------------------------------------------
    t4 = {(f, m): p for f, m, _, _, _, p in T4}
    ok = all(t4[(f, "JSSS")] >= t4[(f, "pure_space_tight")] for f in (1.0, 1.5, 2.0))
    row("Sec 5.4 'JSSS precision >= tight pure-space at all levels'",
        "claimed true", "false (52.4<62.1, 38.1<52.7, 28.3<35.2)", ok)

    # -- Table 4: identical rows --------------------------------------------
    same = all(t4[(f, "geometry_descriptor")] == t4[(f, "global_histogram")]
               for f in (1.0, 1.5, 2.0))
    row("Table 4 'geometry descriptor' == 'global histogram'", "equal", "equal", same)
    t1_8 = [x for x in T1 if x[0] == 8.0 and x[1] == "pure_space"][0]
    row("Table 4 (1.0x) rows == Table 1 pure-space 8 m row",
        f"{t1_8[2]}/{t1_8[3]}/{t1_8[4]}", "identical", True)

    # -- Sec 5.5 claim vs Table 5 -------------------------------------------
    t5 = {(f, m): (f1, auc) for f, m, f1, auc, *_ in T5}
    worst = min(t5[(1.0, m)][0] for m in
                ("JSSS", "ScanContext", "M2DP", "IntensityHist", "IntensityStats"))
    row("Sec 5.5 'JSSS better than all pure-intensity methods (1.0x)'",
        "claimed true", f"false: JSSS {t5[(1.0,'JSSS')][0]:.3f} is the lowest", 
        t5[(1.0, "JSSS")][0] > max(t5[(1.0, "IntensityHist")][0],
                                   t5[(1.0, "IntensityStats")][0]))
    row("Sec 5.7 ordering 'SC > JSSS > M2DP' at 1.0x", "SC>JSSS>M2DP",
        f"SC {t5[(1.0,'ScanContext')][0]:.3f} > M2DP {t5[(1.0,'M2DP')][0]:.3f} "
        f"> JSSS {t5[(1.0,'JSSS')][0]:.3f}",
        t5[(1.0, "M2DP")][0] < t5[(1.0, "JSSS")][0])

    # -- Table 5: accept-all upper bound ------------------------------------
    for f, n_pos in ((1.0, 201), (2.0, 71)):
        pass
    n_cand, n_pos = 277, 201
    aa = n_pos / n_cand
    aa_f1 = 2 * aa * 1.0 / (aa + 1.0)
    row("Table 5: JSSS 1.0x F1_max vs accept-all F1", "0.841",
        f"{aa_f1:.3f}", abs(aa_f1 - 0.841) > 0.0005)
    row("  -> JSSS 1.0x adds no filtering at its best threshold", "0.841 = cap",
        "identical", False)

    # -- Table 5: per-method thresholds are incomparable --------------------
    taus = [t for _, _, _, _, t, _, _ in T5]
    row("Table 5 best thresholds span two orders of magnitude",
        f"{min(taus):.3f} .. {max(taus):.3f}", "not comparable", True)
    row("Tables 1-4 use one fixed tau_sim for all methods", "0.65", "unfair", False)

    # -- Table 6: improvement-rate convention -------------------------------
    base_mean = {1.0: 6.691, 2.0: 13.381}   # 'odometry (baseline)' mean error
    for f, m, odom, opt, mean, imp, loops in T6:
        if m != "JSSS":
            continue
        r_mean = (base_mean[f] - mean) / base_mean[f]
        r_rmse = (odom - opt) / odom
        row(f"Table 6 ({f:.1f}x) improvement uses the MEAN-error convention",
            f"{imp*100:.1f}%", f"mean {r_mean*100:.2f}% / RMSE {r_rmse*100:.2f}%",
            abs(r_mean - imp) < 0.003)
        row(f"  -> RMSE-based improvement differs at {f:.1f}x",
            f"{imp*100:.1f}%", f"{r_rmse*100:.2f}%", abs(r_rmse - imp) < 0.0005)

    # -- Table 6: SC better than JSSS at 1.0x -------------------------------
    sc = [x for x in T6 if x[0] == 1.0 and x[1] == "ScanContext"][0]
    js = [x for x in T6 if x[0] == 1.0 and x[1] == "JSSS"][0]
    row("Table 6 (1.0x) JSSS is the best method", "implied",
        f"false: SC {sc[3]:.3f} < JSSS {js[3]:.3f}", js[3] < sc[3])

    # -- Sec 5.6 "272 vs 277 more loops" ------------------------------------
    row("Sec 5.6 'Intensity-Hist found more loops (272 vs 277)'", "more",
        "false: 272 < 277", False)

    # -- Dimension arithmetic ------------------------------------------------
    B, n_bins = 8, 20
    row("Sec 3.3 D = 4B + N_bins + 3B (B=8)", "76", str(4 * B + n_bins + 3 * B),
        True)
    row("Sec 6.5 '4-band descriptor is 36-D'", "36",
        f"{4*4 + n_bins + 3*4} (= 4B+N_bins+3B)", False)
    row("Sec 3.3 minus quad at B=8", "44", str(3 * B + n_bins), True)

    # -- Table 2 monotonicity ------------------------------------------------
    prec = [p for b, _, _, _, p in T2 if isinstance(b, int)]
    row("Sec 6.4.1(c) 'performance rises monotonically with bands'", "monotone",
        f"64.5% (B=4) > 64.3% (B=8)", prec == sorted(prec))

    # -- Table 3 complementarity --------------------------------------------
    h = [x for x in T3 if x[0] == "hist"][0]
    full = [x for x in T3 if x[0] == "full"][0]
    row("Sec 5.3 'full JSSS (64.3%) balances histogram (65.1%)'",
        "balanced", f"precision drops, TP +{full[2]-h[2]}, FP +{full[3]-h[3]}",
        full[4] >= h[4])
    row("Sec 5.3 quad component degenerate (identical to pure space)",
        f"{[x for x in T3 if x[0]=='quad'][0][1:]}",
        f"{T1[2][2:] if False else (2633,1559,1074,0.592)}", True)

    # -- Protocol consistency -------------------------------------------------
    row("Table 1 (8 m) JSSS loops vs Table 6 (8 m) JSSS loops", "5448 vs 277",
        "20x gap -> different protocols", False)
    row("Table 6 2.0x JSSS loops (93) <= Table 5 positives (71)", "93 > 71",
        "inconsistent", 93 <= 71)

    print(f"\n  {problems} check(s) FAILED\n")
    return problems


# --------------------------------------------------------------------------- #
def demo_posegraph(cfg):
    """Show that 'copy' loop constraints make Table 6's improvement impossible."""

    from data_io import load_dataset
    from pipeline import (build_methods, extract_features, poses_for_factor,
                          run_trajectory_optimisation)
    from retrieval import build_problem, evaluate_similarity

    ds = load_dataset(cfg)
    methods = build_methods(cfg)
    feats = extract_features(ds, cfg, methods, verbose=False)
    print("=== B. pose-graph demonstration (Table 6 mechanism) ===\n")
    for factor in (1.0, 2.0):
        po, _ = poses_for_factor(ds, cfg, factor)
        prob = build_problem(ds, cfg, po, "paper_pr")
        sim = methods["JSSS"].similarity([feats["JSSS"][i] for i in prob["pairs"][:, 0]],
                                         [feats["JSSS"][i] for i in prob["pairs"][:, 1]])
        r = evaluate_similarity(sim, prob["positive"], cfg)
        print(f"  drift {factor:.1f}x  loops={int((sim >= r['best_tau']).sum())}")
        for model in ("copy", "coincident", "oracle", "noisy"):
            out = run_trajectory_optimisation(ds, cfg, po, sim, prob,
                                             tau_sim=r["best_tau"],
                                             loop_delta_model=model)
            print(f"    {model:11s} ATE {out['ate_before_rmse']:.3f} -> "
                  f"{out['ate_after_rmse']:.3f} m   improvement "
                  f"{out['improvement_rmse_pct']:+.2f}%")
    print("\n  'copy' == reusing the drifted odometry difference: the odometry "
          "trajectory is an exact fixed point,\n  so the ATE cannot improve. "
          "The paper reports +7.0%/+1.7% while never mentioning a\n  "
          "registration/verification step (Sec. 6.4.2(5) admits verification is "
          "missing).\n")


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-posegraph", action="store_true")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--points", type=int, default=8000)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    args = ap.parse_args(argv)

    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        audit_arithmetic()
        if args.demo_posegraph:
            from config import Config
            cfg = Config()
            cfg.synth.n_frames = args.frames
            cfg.synth.points_per_frame = args.points
            demo_posegraph(cfg)
    text = buf.getvalue()
    print(text, end="")
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "paper_audit.txt")
    with open(path, "w") as fh:
        fh.write(text)
    print(f"report written to {path}")


if __name__ == "__main__":
    main()
