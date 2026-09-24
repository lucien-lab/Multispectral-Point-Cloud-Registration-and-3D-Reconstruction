"""Component ablation: Q / H / S and all combinations (修改意见 Sec. 7/19).

The paper's Table 3 only reports the three components *individually* plus the
full descriptor, so it cannot establish that the components are complementary
(adds 54 TP but 56 FP -> precision drops).  This script reports the full
2^3 - 1 = 7 combinations with precision, recall, F1, F1-max and AUC, and can
grid-search the similarity weights.

Usage
-----
    python run_ablation_components.py --frames 300 --points 12000
    python run_ablation_components.py --search-weights
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, RESULTS_DIR  # noqa: E402
from descriptors import JSSS  # noqa: E402
from pipeline import poses_for_factor, prepare  # noqa: E402
from retrieval import build_problem  # noqa: E402

COMBOS = [c for k in (1, 2, 3) for c in itertools.combinations(("quad", "hist", "stats"), k)]


def run(env, cfg: Config, protocol: str, factor: float = 1.0):
    ds, feats, ex = env["ds"], env["features"], env["experiment"]
    comps = feats["JSSS"]                       # component dicts, extracted once
    po, _ = poses_for_factor(ds, cfg, factor)
    prob = build_problem(ds, cfg, po, protocol)
    rows = []
    curves = {}
    for combo in COMBOS:
        desc = JSSS(cfg.jsss, active=combo).fit(comps)
        pairs = prob["pairs"]
        left = [comps[i] for i in pairs[:, 0]]
        right = [comps[i] for i in pairs[:, 1]]
        sim = desc.similarity(left, right)
        from retrieval import evaluate_similarity
        r = evaluate_similarity(sim, prob["positive"], cfg)
        # also record behaviour at the paper's fixed working point
        from retrieval import evaluate_similarity as ev
        rfix = ev(sim, prob["positive"], cfg, fixed_tau=cfg.retrieval.tau_sim)
        rows.append({
            "components": "+".join(c[0].upper() for c in combo),
            "dim": (4 * cfg.jsss.n_bands if "quad" in combo else 0)
                   + (cfg.jsss.hist_bins if "hist" in combo else 0)
                   + (3 * cfg.jsss.n_bands if "stats" in combo else 0),
            "n_cand": r["n_cand"], "n_pos": r["n_pos"],
            "loops@tau": rfix["closed_loops"], "TP@tau": rfix["tp"],
            "FP@tau": rfix["fp"], "P@tau": rfix["precision"],
            "R@tau": rfix["recall"],
            "F1@tau": rfix["f1"],
            "F1_max": r["f1_max"], "AUC": r["auc"], "AP": r["ap"],
            "best_tau": r["best_tau"],
            "accept_all_F1": r["accept_all_f1"],
        })
        curves["+".join(c[0].upper() for c in combo)] = r["curve"]
    return pd.DataFrame(rows), curves


def search_weights(env, cfg: Config, protocol: str, factor: float = 1.0,
                   step: float = 0.1):
    """Grid search w_quad / w_hist / w_stats on F1-max (修改意见 Sec. 16)."""

    ds, feats = env["ds"], env["features"]
    comps = feats["JSSS"]
    po, _ = poses_for_factor(ds, cfg, factor)
    prob = build_problem(ds, cfg, po, protocol)
    from retrieval import evaluate_similarity
    rows = []
    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for wq in grid:
        for wh in grid:
            ws = 1.0 - wq - wh
            if ws < -1e-9:
                continue
            ds_cfg = cfg.jsss
            from dataclasses import replace
            jc = replace(ds_cfg, weights=(wq, wh, ws))
            desc = JSSS(jc, active=("quad", "hist", "stats")).fit(comps)
            pairs = prob["pairs"]
            sim = desc.similarity([comps[i] for i in pairs[:, 0]],
                                  [comps[i] for i in pairs[:, 1]])
            r = evaluate_similarity(sim, prob["positive"], cfg)
            rows.append({"w_quad": round(float(wq), 2), "w_hist": round(float(wh), 2),
                         "w_stats": round(max(ws, 0.0), 2),
                         "F1_max": r["f1_max"], "AUC": r["auc"]})
    df = pd.DataFrame(rows).sort_values("F1_max", ascending=False)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--points", type=int, default=12000)
    ap.add_argument("--bands", type=int, default=8)
    ap.add_argument("--protocol", default="paper_pr",
                    choices=["paper_loose", "paper_pr", "fixed"])
    ap.add_argument("--factor", type=float, default=1.0)
    ap.add_argument("--search-weights", action="store_true")
    ap.add_argument("--weight-mode", default="equal", choices=["equal", "paper662"])
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.synth.n_frames = args.frames
    cfg.synth.points_per_frame = args.points
    cfg.jsss.n_bands = args.bands
    cfg.jsss.weight_mode = args.weight_mode
    cfg.retrieval.protocol = args.protocol

    env = prepare(cfg, verbose=False)
    env["cfg"] = cfg
    df, _ = run(env, cfg, args.protocol, args.factor)
    print("\n=== Table 3 (fixed): component ablation, full combination ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("  note: H alone vs Q+H+S shows whether the extra components actually "
          "help (paper Table 3: +54 TP but +56 FP -> precision drops).")

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "table3_components.csv"), index=False)

    if args.search_weights:
        wdf = search_weights(env, cfg, args.protocol, args.factor)
        print("\n=== similarity weight grid search (top 10 by F1_max) ===")
        print(wdf.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        wdf.to_csv(os.path.join(args.out, "component_weights_grid.csv"), index=False)
    return df


if __name__ == "__main__":
    main()
