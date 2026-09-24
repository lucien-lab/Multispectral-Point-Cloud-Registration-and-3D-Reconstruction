"""Side-by-side comparison: the paper's reported numbers vs this reimplementation.

Answers the question "would my implementation reproduce the paper's numbers?".
It runs the *paper's protocol* (Tables 1-4: fixed tau_sim = 0.65; Tables 5-6:
tau_odom = 8 m and |i-j| >= 10) and prints a per-table diff, separating

* **definitional** quantities (dimension, drift RMS, TP+FP, accept-all F1) which
  must match exactly, from
* **data/implementation dependent** quantities (precision, recall, F1, AUC)
  which depend on the dataset and on choices the paper leaves unspecified.

Usage
-----
    python compare_with_paper.py --frames 600 --points 20000
    python compare_with_paper.py --quick
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_audit as PA  # noqa: E402
from config import Config, RESULTS_DIR  # noqa: E402
from augment import BandSliceDataset  # noqa: E402
from descriptors import JSSS  # noqa: E402
from pipeline import (  # noqa: E402
    build_methods,
    extract_features,
    poses_for_factor,
    prepare,
    run_trajectory_optimisation,
)
from retrieval import build_problem, evaluate_similarity  # noqa: E402

OUT = []


def say(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    OUT.append(line)


def side_by_side(title, rows, note=""):
    """rows: list of (label, paper_value, repro_value, kind)."""
    say(f"\n### {title}")
    say(f"{'quantity':44s} {'paper':>16s} {'repro':>18s}  kind")
    say("-" * 96)
    for label, pv, rv, kind in rows:
        say(f"{label:44s} {str(pv):>16s} {str(rv):>18s}  {kind}")
    if note:
        say(f"  note: {note}")


# --------------------------------------------------------------------------- #
def t1(env, cfg, factor=1.0):
    ds, feats, ex = env["ds"], env["features"], env["experiment"]
    po, _ = poses_for_factor(ds, cfg, factor)
    out = {}
    for tau in (3.0, 5.0, 8.0, 12.0):
        prob = build_problem(ds, cfg, po, "paper_loose", tau_odom_override=tau)
        for name in ("PureSpace_tight", "JSSS"):
            r = ex.evaluate(ds, feats, po, "paper_loose", names=[name],
                            fixed_tau=cfg.retrieval.tau_sim, problem=prob)[name]
            out[(tau, name)] = r
    rows = []
    for tau, m, loops, tp, fp, prec in PA.T1:
        key = ("JSSS" if m == "JSSS" else "PureSpace_tight",)
        r = out[(tau, "JSSS" if m == "JSSS" else "PureSpace_tight")]
        rows.append((f"{tau:.0f} m / {m}: loops",
                     loops, r["closed_loops"], "data"))
        rows.append((f"{tau:.0f} m / {m}: TP / FP",
                     f"{tp} / {fp}", f"{r['tp']} / {r['fp']}", "data"))
        rows.append((f"{tau:.0f} m / {m}: precision",
                     f"{prec:.3f}", f"{r['precision']:.3f}", "data"))
    return pd.DataFrame([{ "label": a, "paper": b, "repro": c, "kind": d}
                         for a, b, c, d in rows]), out


def t2(env, cfg, bands=(1, 2, 4, 8)):
    ds = env["ds"]
    out = {}
    for B in bands:
        bds = BandSliceDataset(ds, B, n_available=8)
        bcfg = replace(cfg, jsss=replace(cfg.jsss, n_bands=int(B)))
        methods = build_methods(bcfg, include_baselines=False)
        feats = extract_features(bds, cfg, methods, verbose=False)
        po, _ = poses_for_factor(ds, cfg, 1.0)
        prob = build_problem(bds, cfg, po, "paper_loose", tau_odom_override=5.0)
        pairs = prob["pairs"]
        sim = methods["JSSS"].similarity([feats["JSSS"][i] for i in pairs[:, 0]],
                                         [feats["JSSS"][i] for i in pairs[:, 1]])
        out[B] = evaluate_similarity(sim, prob["positive"], cfg,
                                     fixed_tau=cfg.retrieval.tau_sim)
    rows = []
    for b, loops, tp, fp, prec in PA.T2:
        if isinstance(b, int):
            r = out[b]
            rows.append((f"B={b}: loops", loops, r["closed_loops"], "data"))
            rows.append((f"B={b}: TP / FP", f"{tp} / {fp}",
                         f"{r['tp']} / {r['fp']}", "data"))
            rows.append((f"B={b}: precision", f"{prec:.3f}",
                         f"{r['precision']:.3f}", "data"))
    return pd.DataFrame([{"label": a, "paper": b, "repro": c, "kind": d}
                         for a, b, c, d in rows]), out


def t3(env, cfg):
    ds, feats = env["ds"], env["features"]
    comps = feats["JSSS"]
    po, _ = poses_for_factor(ds, cfg, 1.0)
    prob = build_problem(ds, cfg, po, "paper_loose", tau_odom_override=5.0)
    pairs = prob["pairs"]
    res = {}
    mapping = {"quad": ("quad",), "hist": ("hist",), "stats": ("stats",),
               "full": ("quad", "hist", "stats")}
    for name, active in mapping.items():
        desc = JSSS(cfg.jsss, active=active).fit(comps)
        sim = desc.similarity([comps[i] for i in pairs[:, 0]],
                              [comps[i] for i in pairs[:, 1]])
        res[name] = evaluate_similarity(sim, prob["positive"], cfg,
                                        fixed_tau=cfg.retrieval.tau_sim)
    rows = []
    for name, loops, tp, fp, prec in PA.T3:
        r = res[name]
        rows.append((f"{name}: loops", loops, r["closed_loops"], "data"))
        rows.append((f"{name}: TP / FP", f"{tp} / {fp}",
                     f"{r['tp']} / {r['fp']}", "data"))
        rows.append((f"{name}: precision", f"{prec:.3f}",
                     f"{r['precision']:.3f}", "data"))
    return pd.DataFrame([{"label": a, "paper": b, "repro": c, "kind": d}
                         for a, b, c, d in rows]), res


def t4(env, cfg, factors=(1.0, 1.5, 2.0)):
    ds, feats, ex = env["ds"], env["features"], env["experiment"]
    names = ["PureSpace_tight", "RadialHist", "GlobalHist_norm", "JSSS"]
    res = {}
    for f in factors:
        po, info = poses_for_factor(ds, cfg, f)
        prob = build_problem(ds, cfg, po, "paper_loose", tau_odom_override=8.0)
        rr = ex.evaluate(ds, feats, po, "paper_loose", names=names,
                         fixed_tau=cfg.retrieval.tau_sim, problem=prob)
        for n, r in rr.items():
            res[(f, n)] = (r, info["rms_m"])
    alias = {"pure_space_tight": "PureSpace_tight",
             "geometry_descriptor": "RadialHist",
             "global_histogram": "GlobalHist_norm",
             "JSSS": "JSSS"}
    rows = []
    for f, m, loops, tp, fp, prec in PA.T4:
        r, rms = res[(f, alias[m])]
        rows.append((f"{f:.1f}x {m}: precision", f"{prec:.3f}",
                     f"{r['precision']:.3f}", "data"))
        rows.append((f"{f:.1f}x {m}: loops", loops, r["closed_loops"], "data"))
    rows.append(("odometry RMS drift at 1.0x (m)", "7.257",
                 f"{res[(1.0,'JSSS')][1]:.3f}", "definitional"))
    return pd.DataFrame([{"label": a, "paper": b, "repro": c, "kind": d}
                         for a, b, c, d in rows]), res


def t5(env, cfg, factors=(1.0, 2.0)):
    ds, feats, ex = env["ds"], env["features"], env["experiment"]
    names = ["JSSS", "ScanContext", "M2DP", "IntensityHist", "IntensityStats"]
    rows = []
    for f in factors:
        po, _ = poses_for_factor(ds, cfg, f)
        prob = build_problem(ds, cfg, po, "paper_pr")
        rr = ex.evaluate(ds, feats, po, "paper_pr", names=names, problem=prob)
        for f0, m, f1m, auc, tau, nc, npos in PA.T5:
            if f0 != f:
                continue
            r = rr[m]
            rows.append((f"{f:.1f}x {m}: n_cand / n_pos", f"{nc} / {npos}",
                         f"{r['n_cand']} / {r['n_pos']}", "data/protocol"))
            rows.append((f"{f:.1f}x {m}: F1_max", f"{f1m:.3f}",
                         f"{r['f1_max']:.3f}", "data"))
            rows.append((f"{f:.1f}x {m}: AUC", f"{auc:.3f}",
                         f"{r['auc']:.3f}", "data"))
        aa = rr["JSSS"]["accept_all_f1"]
        rows.append((f"{f:.1f}x accept-all F1 (upper bound of F1_max)", "-",
                     f"{aa:.3f}", "definitional"))
    return pd.DataFrame([{"label": a, "paper": b, "repro": c, "kind": d}
                         for a, b, c, d in rows]), None


def t6(env, cfg, factors=(1.0, 2.0), models=("copy", "oracle")):
    ds, feats, ex = env["ds"], env["features"], env["experiment"]
    rows = []
    for f in factors:
        po, _ = poses_for_factor(ds, cfg, f)
        prob = build_problem(ds, cfg, po, "paper_pr")
        rr = ex.evaluate(ds, feats, po, "paper_pr", names=["JSSS"], problem=prob)
        sim = ex.caches["JSSS"].get(prob["pairs"])
        for model in models:
            out = run_trajectory_optimisation(ds, cfg, po, sim, prob,
                                             tau_sim=rr["JSSS"]["best_tau"],
                                             iterations=100,
                                             loop_delta_model=model)
            rows.append((f"{f:.1f}x JSSS improvement [{model}]",
                         "7.0%" if f == 1.0 else "1.7%",
                         f"{out['improvement_rmse_pct']:+.2f}% (RMSE)", "mechanism"))
    return pd.DataFrame([{"label": a, "paper": b, "repro": c, "kind": d}
                         for a, b, c, d in rows]), None


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dataset", default="synthetic",
                    choices=["synthetic", "kitti"])
    ap.add_argument("--sequence", default="05")
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args(argv)
    if args.quick:
        args.frames, args.points = 200, 8000

    cfg = Config()
    cfg.dataset = args.dataset
    cfg.sequence = args.sequence
    cfg.synth.n_frames = args.frames
    cfg.synth.points_per_frame = args.points
    cfg.max_frames = None if args.dataset == "kitti" else cfg.max_frames
    cfg.retrieval.tau_sim = 0.65          # Tables 1-4 working point

    say("# JSSS: paper numbers vs this reimplementation")
    say(f"config: frames={args.frames} points/frame={args.points} "
        f"bands=8 dim={cfg.jsss.dimension} tau_sim={cfg.retrieval.tau_sim}")
    say(f"data: {'KITTI Odometry sequence ' + args.sequence + ' + SemanticKITTI labels' if args.dataset == 'kitti' else 'procedural synthetic scene'}")

    env = prepare(cfg, verbose=False)
    env["cfg"] = cfg
    say(f"frames={env['ds'].n_frames}  gt_loops={len(env['ds'].gt_loops)}")

    tables = {}
    for name, fn in (("T1", lambda: t1(env, cfg)), ("T2", lambda: t2(env, cfg)),
                     ("T3", lambda: t3(env, cfg)), ("T4", lambda: t4(env, cfg)),
                     ("T5", lambda: t5(env, cfg)), ("T6", lambda: t6(env, cfg))):
        df, extra = fn()
        tables[name] = df
        side_by_side(f"{name}: paper vs repro", list(
            df.itertuples(index=False, name=None)))

    say("\n## How to read this")
    say("- kind=definitional : must match exactly (arithmetic on own values).")
    say("- kind=data        : depends on the dataset (synthetic vs KITTI 05) AND")
    say("                     on choices the paper leaves unspecified.")
    say("- kind=protocol    : depends on the evaluation protocol used.")
    say("- kind=mechanism   : depends on the pose-graph formulation.")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "compare_with_paper.md")
    with open(path, "w") as fh:
        fh.write("\n".join(OUT) + "\n")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
