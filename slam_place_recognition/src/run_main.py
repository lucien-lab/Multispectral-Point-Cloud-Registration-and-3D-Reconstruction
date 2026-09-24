"""Reproduce the six result tables of the paper (Sec. 5) and export CSVs.

Usage
-----
    python run_main.py --frames 400 --points 20000 --drift 1.0 1.5 2.0
    python run_main.py --protocol fixed            # recommended protocol
    python run_main.py --quick                     # tiny smoke run

Tables produced (paper numbering)
---------------------------------
T1  precision vs odometry search radius (PureSpace vs JSSS)
T2  band-count ablation (see also run_ablation_bands.py)
T3  component ablation (see also run_ablation_components.py)
T4  robustness across drift factors
T5  PR metrics (F1-max, AUC, best threshold) at 1.0x and 2.0x
T6  trajectory error after pose-graph optimisation

Every table additionally reports **recall**, which the paper's Tables 1-4 omit.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, RESULTS_DIR  # noqa: E402
from pipeline import (  # noqa: E402
    build_methods,
    poses_for_factor,
    prepare,
    run_trajectory_optimisation,
)
from retrieval import build_problem  # noqa: E402

DRIFT_FACTORS = (1.0, 1.5, 2.0)


# --------------------------------------------------------------------------- #
def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        if not np.isfinite(v):
            return "-"
        return f"{v:.{nd}f}"
    return str(v)


def table1(env, factors=(1.0,)):
    cfg: Config = env["cfg"]
    ds, feats, ex = env["ds"], env["features"], env["experiment"]
    rows = []
    for factor in factors:
        po, info = poses_for_factor(ds, cfg, factor)
        for tau in (3.0, 5.0, 8.0, 12.0):
            prob = build_problem(ds, cfg, po, "paper_loose", tau_odom_override=tau)
            for name in ("PureSpace_tight", "JSSS"):
                r = ex.evaluate(ds, feats, po, protocol="paper_loose",
                                names=[name], fixed_tau=cfg.retrieval.tau_sim,
                                problem=prob)[name]
                rows.append({
                    "drift": factor, "tau_odom_m": tau, "method": name,
                    "closed_loops": r["closed_loops"], "TP": r["tp"], "FP": r["fp"],
                    "precision": r["precision"], "recall": r["recall"],
                    "f1": r["f1"], "n_cand": r["n_cand"], "n_pos": r["n_pos"]})
    df = pd.DataFrame(rows)
    print("\n=== Table 1: precision vs odometry search radius "
          f"(protocol=paper_loose, tau_sim={cfg.retrieval.tau_sim}) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


def table4(env, factors=DRIFT_FACTORS):
    cfg: Config = env["cfg"]
    ds, feats, ex = env["ds"], env["features"], env["experiment"]
    names = ["PureSpace_tight", "RadialHist", "GlobalHist_norm",
             "IntensityHist", "IntensityStats", "M2DP", "ScanContext", "JSSS"]
    rows = []
    for factor in factors:
        po, info = poses_for_factor(ds, cfg, factor)
        prob = build_problem(ds, cfg, po, "paper_loose", tau_odom_override=8.0)
        res = ex.evaluate(ds, feats, po, protocol="paper_loose", names=names,
                          fixed_tau=cfg.retrieval.tau_sim, problem=prob)
        for name, r in res.items():
            rows.append({"drift": factor, "odom_rms_m": info["rms_m"], "method": name,
                         "closed_loops": r["closed_loops"], "TP": r["tp"],
                         "FP": r["fp"], "precision": r["precision"],
                         "recall": r["recall"], "f1": r["f1"]})
    df = pd.DataFrame(rows)
    print("\n=== Table 4: robustness across odometry-drift factors "
          "(tau_odom=8 m, tau_sim=0.65) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return df


def table5(env, factors=DRIFT_FACTORS):
    """PR metrics under the paper's PR protocol (tau_odom = 8 m, gap >= 10)."""

    cfg: Config = env["cfg"]
    ds, feats, ex = env["ds"], env["features"], env["experiment"]
    names = ["JSSS", "ScanContext", "M2DP", "IntensityHist", "IntensityStats",
             "GlobalHist_norm", "RadialHist", "PureSpace_tight"]
    rows = []
    for factor in factors:
        po, _ = poses_for_factor(ds, cfg, factor)
        prob = build_problem(ds, cfg, po, "paper_pr")
        res = ex.evaluate(ds, feats, po, protocol="paper_pr", names=names,
                          problem=prob)
        for name, r in res.items():
            rows.append({"drift": factor, "method": name, "n_cand": r["n_cand"],
                         "n_pos": r["n_pos"], "F1_max": r["f1_max"],
                         "AUC": r["auc"], "AP": r["ap"], "best_tau": r["best_tau"],
                         "P@best": r["precision"], "R@best": r["recall"],
                         "accept_all_F1": r["accept_all_f1"]})
    df = pd.DataFrame(rows)
    print("\n=== Table 5: PR-curve metrics (protocol=paper_pr) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("  note: 'accept_all_F1' is the F1 obtained by accepting every "
          "candidate; if F1_max == accept_all_F1 the descriptor filters nothing.")
    return df


def table6(env, factors=DRIFT_FACTORS, iterations=None,
           loop_delta_models=("copy", "coincident", "oracle", "noisy")):
    """Trajectory error after pose-graph optimisation.

    Reported for **every** loop-constraint measurement model so that the
    degeneracy of the paper's (undocumented) scheme is visible in the output.
    """

    cfg: Config = env["cfg"]
    ds, feats, ex = env["ds"], env["features"], env["experiment"]
    names = ["JSSS", "ScanContext", "M2DP", "IntensityHist", "IntensityStats"]
    rows = []
    for factor in factors:
        po, _ = poses_for_factor(ds, cfg, factor)
        prob = build_problem(ds, cfg, po, "paper_pr")
        res = ex.evaluate(ds, feats, po, protocol="paper_pr", names=names,
                          problem=prob)
        for name in names:
            r = res[name]
            sim = ex.caches[name].get(prob["pairs"])
            for model in loop_delta_models:
                out = run_trajectory_optimisation(ds, cfg, po, sim, prob,
                                                  tau_sim=r["best_tau"],
                                                  iterations=iterations,
                                                  loop_delta_model=model)
                rows.append({"drift": factor, "method": name,
                             "loop_delta": model,
                             "odom_ATE_RMSE": out["ate_before_rmse"],
                             "opt_ATE_RMSE": out["ate_after_rmse"],
                             "opt_ATE_mean": out["ate_after_mean"],
                             "impr_RMSE_%": out["improvement_rmse_pct"],
                             "impr_mean_%": out["improvement_mean_pct"],
                             "loops": out["n_loops"], "tau_sim": out["tau_sim"]})
    df = pd.DataFrame(rows)
    print("\n=== Table 6: trajectory error after pose-graph optimisation "
          "(per method's own best threshold) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("  'loop_delta=copy' reuses the drifted odometry difference -> the "
          "relaxation is a no-op\n  (improvement exactly 0.00%), so the "
          "paper's +7.0%/+1.7% needs an unstated step.")
    print("  'coincident' treats a detected loop as the same place; "
          "'oracle'/'noisy' imply\n  a registration (ICP/NDT) step that "
          "Sec. 6.4.2(5) says was not implemented.")
    return df


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="JSSS main experiments")
    ap.add_argument("--dataset", default="synthetic", choices=["synthetic", "kitti"])
    ap.add_argument("--sequence", default="05")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--protocol", default="paper_loose",
                    choices=["paper_loose", "paper_pr", "fixed"])
    ap.add_argument("--drift", type=float, nargs="+", default=list(DRIFT_FACTORS))
    ap.add_argument("--weight-mode", default="equal",
                    choices=["equal", "paper662"])
    ap.add_argument("--label-convention", default="auto",
                    choices=["auto", "raw", "learning"])
    ap.add_argument("--pg-iters", type=int, default=None,
                    help="pose-graph iterations (default 500, paper Sec. 4.4)")
    ap.add_argument("--pg-huber", type=float, default=None,
                    help="Huber delta in metres (default 1.0)")
    ap.add_argument("--out", default=RESULTS_DIR)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)

    if args.quick:
        args.frames, args.points = 150, 5000

    cfg = Config()
    cfg.dataset = args.dataset
    cfg.sequence = args.sequence
    cfg.synth.n_frames = args.frames
    cfg.synth.points_per_frame = args.points
    cfg.jsss.n_bands = args.bands
    cfg.retrieval.protocol = args.protocol
    cfg.jsss.weight_mode = args.weight_mode
    cfg.label_convention = args.label_convention
    if args.pg_huber is not None:
        cfg.posegraph.huber_delta = args.pg_huber
    cfg.max_frames = args.frames if args.dataset == "synthetic" else None

    print(f"[config] dataset={cfg.dataset} frames={args.frames} "
          f"points={args.points} bands={args.bands} dim={cfg.jsss.dimension} "
          f"weights={cfg.jsss.resolved_weights()}")

    env = prepare(cfg)
    env["cfg"] = cfg
    ds = env["ds"]
    print(f"[data] frames={ds.n_frames} gt_loops={len(ds.gt_loops)}")

    os.makedirs(args.out, exist_ok=True)
    dfs = {
        "table1_search_radius": table1(env, factors=args.drift[:1]),
        "table4_drift": table4(env, factors=args.drift),
        "table5_pr": table5(env, factors=args.drift),
        "table6_trajectory": table6(env, factors=args.drift,
                                    iterations=args.pg_iters),
    }
    for name, df in dfs.items():
        path = os.path.join(args.out, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  saved {path}")
    return dfs


if __name__ == "__main__":
    main()
