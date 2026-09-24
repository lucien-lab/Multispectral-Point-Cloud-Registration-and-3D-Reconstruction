"""Robustness experiments - three *separated* error sources (修改意见 Sec. 12/20).

The paper labels 1.0x/1.5x/2.0x as "noise robustness" while actually scaling the
odometry drift only.  This script produces the three orthogonal sweeps:

1. ``drift``     - odometry drift factor (candidate-set distortion).
2. ``spectral``  - measurement noise sigma_s in {0, .01, .02, .05, .10}
                   (spectral-descriptor distortion).
3. ``geometric`` - XYZ noise sigma_p in {0, .02, .05, .10} m and point dropout
                   in {0, 10, 20, 30} % (geometric-descriptor distortion).

It also contains the control experiment of 修改意见 Sec. 15: whether JSSS
identifies *place* or merely *material composition* (``--place-variation``),
by regenerating the scene with/without local composition variation.

Usage
-----
    python run_robustness.py --sweep spectral
    python run_robustness.py --sweep geometric --drop 0 0.1 0.2 0.3
    python run_robustness.py --sweep drift --factors 1.0 1.5 2.0
    python run_robustness.py --sweep place --place-variation 0.0 0.5 1.0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augment import GeometricNoiseDataset, SpectralNoiseDataset  # noqa: E402
from config import Config, RESULTS_DIR  # noqa: E402
from data_io import load_dataset  # noqa: E402
from pipeline import (  # noqa: E402
    build_methods,
    extract_features,
    poses_for_factor,
    run_trajectory_optimisation,
)
from retrieval import build_problem, evaluate_similarity  # noqa: E402

METHODS = ["JSSS", "ScanContext", "M2DP", "RadialHist", "GlobalHist_norm",
           "IntensityHist", "IntensityStats"]

METRIC_COLS = ["n_cand", "n_pos", "F1_max", "AUC", "AP", "best_tau", "P@best",
               "R@best", "accept_all_F1", "odom_rms_m", "impr_RMSE_%",
               "impr_mean_%", "loops"]


def eval_dataset(cfg: Config, ds, factor: float = 1.0, protocol: str = "paper_pr",
                 methods=None, with_ate: bool = False,
                 loop_delta_model: str = "noisy"):
    methods = methods or build_methods(cfg)
    warm = load_dataset(cfg) if ds is None else None
    ds = ds or warm
    feats = extract_features(ds, cfg, methods, verbose=False)
    po, info = poses_for_factor(ds, cfg, factor)
    prob = build_problem(ds, cfg, po, protocol)
    rows = []
    for name, desc in methods.items():
        pairs = prob["pairs"]
        left = [feats[name][i] for i in pairs[:, 0]]
        right = [feats[name][i] for i in pairs[:, 1]]
        sim = desc.similarity(left, right)
        r = evaluate_similarity(sim, prob["positive"], cfg)
        row = {"method": name, "n_cand": r["n_cand"], "n_pos": r["n_pos"],
               "F1_max": r["f1_max"], "AUC": r["auc"], "AP": r["ap"],
               "best_tau": r["best_tau"], "P@best": r["precision"],
               "R@best": r["recall"], "accept_all_F1": r["accept_all_f1"],
               "odom_rms_m": info["rms_m"]}
        if with_ate:
            out = run_trajectory_optimisation(ds, cfg, po, sim, prob,
                                             tau_sim=r["best_tau"],
                                             loop_delta_model=loop_delta_model)
            row.update({"impr_RMSE_%": out["improvement_rmse_pct"],
                        "impr_mean_%": out["improvement_mean_pct"],
                        "loops": out["n_loops"]})
        rows.append(row)
    return rows, ds


# --------------------------------------------------------------------------- #
def sweep_drift(cfg: Config, factors, protocol):
    ds = load_dataset(cfg)
    rows = []
    for f in factors:
        r, _ = eval_dataset(cfg, ds, factor=f, protocol=protocol, with_ate=True)
        for x in r:
            x["drift_factor"] = f
        rows.extend(r)
    return pd.DataFrame(rows)


def sweep_spectral(cfg: Config, sigmas, protocol, frames, points):
    ds = load_dataset(cfg)
    rows = []
    for s in sigmas:
        print(f"  [spectral sigma={s}]", flush=True)
        sds = SpectralNoiseDataset(ds, s, cfg.spectral)
        r, _ = eval_dataset(cfg, sds, factor=1.0, protocol=protocol)
        for x in r:
            x["sigma_spectral"] = s
        rows.extend(r)
    return pd.DataFrame(rows)


def sweep_geometric(cfg: Config, sigmas, drops, protocol):
    ds = load_dataset(cfg)
    rows = []
    for s in sigmas:
        for d in drops:
            print(f"  [geometric sigma={s} m, drop={d:.0%}]", flush=True)
            gds = GeometricNoiseDataset(ds, sigma_xyz=s, drop_rate=d)
            r, _ = eval_dataset(cfg, gds, factor=1.0, protocol=protocol)
            for x in r:
                x["sigma_xyz_m"] = s
                x["drop_rate"] = d
            rows.extend(r)
    return pd.DataFrame(rows)


def sweep_place_variation(base: Config, variations, protocol):
    """Control experiment: does JSSS key on place identity or material mixture?"""

    rows = []
    for v in variations:
        print(f"  [composition_variation={v}]", flush=True)
        cfg = Config()
        cfg.synth = base.synth
        cfg.synth.composition_variation = v
        cfg.synth.n_frames = base.synth.n_frames
        cfg.synth.points_per_frame = base.synth.points_per_frame
        cfg.retrieval = base.retrieval
        ds = load_dataset(cfg)
        r, _ = eval_dataset(cfg, ds, factor=1.0, protocol=protocol)
        for x in r:
            x["composition_variation"] = v
        rows.extend(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="drift",
                    choices=["drift", "spectral", "geometric", "place", "all"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--points", type=int, default=12000)
    ap.add_argument("--protocol", default="paper_pr",
                    choices=["paper_loose", "paper_pr", "fixed"])
    ap.add_argument("--factors", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--sigmas", type=float, nargs="+",
                    default=[0.0, 0.01, 0.02, 0.05, 0.10])
    ap.add_argument("--xyz", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10])
    ap.add_argument("--drop", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--place-variation", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0])
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.synth.n_frames = args.frames
    cfg.synth.points_per_frame = args.points
    cfg.retrieval.protocol = args.protocol
    os.makedirs(args.out, exist_ok=True)

    sweeps = {"drift": lambda: sweep_drift(cfg, args.factors, args.protocol),
              "spectral": lambda: sweep_spectral(cfg, args.sigmas, args.protocol,
                                                 args.frames, args.points),
              "geometric": lambda: sweep_geometric(cfg, args.xyz, args.drop,
                                                   args.protocol),
              "place": lambda: sweep_place_variation(cfg, args.place_variation,
                                                     args.protocol)}
    todo = list(sweeps) if args.sweep == "all" else [args.sweep]
    for name in todo:
        print(f"\n=== sweep: {name} ===")
        df = sweeps[name]()
        path = os.path.join(args.out, f"robustness_{name}.csv")
        df.to_csv(path, index=False)
        cond = [c for c in df.columns if c not in METRIC_COLS]
        cols = cond + [c for c in METRIC_COLS if c in df.columns]
        print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"  saved {path}")


if __name__ == "__main__":
    main()
