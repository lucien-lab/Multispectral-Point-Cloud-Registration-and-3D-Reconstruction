"""Band-count ablation (paper Table 2 / 修改意见 Sec. 10-11).

Reports, for B in {1, 2, 4, 8}:

* the full metric set (P / R / F1 / F1-max / AUC) so that a *fixed* threshold is
  no longer used to conclude "4 bands are enough";
* the single-band, spectral-sum, spectral-norm-histogram and native-intensity
  baselines under the four *distinct* names the paper conflates
  (single spectral band vs spectral-sum scalar vs KITTI native intensity vs
  multispectral vector).

Usage
-----
    python run_ablation_bands.py --frames 300 --points 12000
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from augment import BandSliceDataset  # noqa: E402
from config import Config, RESULTS_DIR  # noqa: E402
from pipeline import build_methods, extract_features, poses_for_factor  # noqa: E402
from retrieval import build_problem, evaluate_similarity  # noqa: E402


def run(env, cfg: Config, bands=(1, 2, 4, 8), protocol="paper_pr", factor=1.0,
        verbose=True):
    ds = env["ds"]
    rows = []
    from dataclasses import replace
    for B in bands:
        if verbose:
            print(f"  [bands={B}] extracting ...", flush=True)
        bds = BandSliceDataset(ds, B, n_available=8)
        bcfg = replace(cfg, jsss=replace(cfg.jsss, n_bands=int(B)))
        methods = build_methods(bcfg, include_baselines=False)
        feats = extract_features(bds, cfg, methods, verbose=False)
        po, _ = poses_for_factor(ds, cfg, factor)
        prob = build_problem(bds, cfg, po, protocol)
        pairs = prob["pairs"]
        for name, desc in methods.items():
            left = [feats[name][i] for i in pairs[:, 0]]
            right = [feats[name][i] for i in pairs[:, 1]]
            sim = desc.similarity(left, right)
            r = evaluate_similarity(sim, prob["positive"], cfg)
            rfix = evaluate_similarity(sim, prob["positive"], cfg,
                                       fixed_tau=cfg.retrieval.tau_sim)
            rows.append({"bands": B, "method": name, "dim": getattr(desc, "dimension", None),
                         "n_cand": r["n_cand"], "TP@tau": rfix["tp"], "FP@tau": rfix["fp"],
                         "P@tau": rfix["precision"], "R@tau": rfix["recall"],
                         "F1@tau": rfix["f1"], "F1_max": r["f1_max"],
                         "AUC": r["auc"], "best_tau": r["best_tau"]})
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--points", type=int, default=12000)
    ap.add_argument("--protocol", default="paper_pr",
                    choices=["paper_loose", "paper_pr", "fixed"])
    ap.add_argument("--bands", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--factor", type=float, default=1.0)
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.synth.n_frames = args.frames
    cfg.synth.points_per_frame = args.points
    cfg.retrieval.protocol = args.protocol

    from pipeline import prepare
    env = prepare(cfg, verbose=False)
    cfg.jsss.n_bands = 8          # JSSS is always built with the full stack
    df = run(env, cfg, bands=tuple(args.bands), protocol=args.protocol,
             factor=args.factor)
    print("\n=== Table 2: band-count ablation (per-method optimal threshold) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("  note: at a fixed tau_sim the comparison is confounded because the "
          "descriptor dimension (and therefore the similarity scale) changes.")
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "table2_bands.csv"), index=False)
    return df


if __name__ == "__main__":
    main()
