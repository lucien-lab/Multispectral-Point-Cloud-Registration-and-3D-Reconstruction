"""Diagnose the similarity definition (paper Sec. 3.3 says only
"归一化逆欧氏距离" - the normalisation itself is never specified).

The paper's Table 3 shows the *full* JSSS being worse than its own histogram
component (64.3 % vs 65.1 %) while the quadrant component is identical to pure
space (59.2 %), which is the signature of one component dominating an
*unnormalised* average.  This script evaluates the same features under several
similarity definitions so the effect can be quantified on real data.

Usage
-----
    python diagnose_similarity.py --dataset kitti --frames 1400 --tau-odom 8
    python diagnose_similarity.py --dataset synthetic --frames 400
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, RESULTS_DIR  # noqa: E402
from descriptors.jsss import Normalizer  # noqa: E402
from pipeline import build_methods, extract_features, poses_for_factor  # noqa: E402
from retrieval import build_problem, evaluate_similarity  # noqa: E402

COMBOS = {"Q": ("quad",), "H": ("hist",), "S": ("stats",),
          "QHS": ("quad", "hist", "stats")}


def rms(a, b):
    d = a - b
    return np.sqrt((d ** 2).sum(axis=1) / d.shape[1])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti")
    ap.add_argument("--sequence", default="05")
    ap.add_argument("--frames", type=int, default=1400)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--tau-odom", type=float, default=8.0)
    ap.add_argument("--tau-sim", type=float, default=0.65)
    ap.add_argument("--factor", type=float, default=1.0)
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    cfg = Config()
    cfg.dataset = args.dataset
    cfg.sequence = args.sequence
    cfg.retrieval.tau_sim = args.tau_sim
    cfg.max_frames = args.frames if args.dataset == "synthetic" else None
    if args.dataset == "synthetic":
        cfg.synth.n_frames = args.frames
        cfg.synth.points_per_frame = args.points

    from data_io import load_dataset
    ds = load_dataset(cfg)
    methods = build_methods(cfg, include_baselines=False)
    feats = extract_features(ds, cfg, methods, verbose=False)
    comp = feats["JSSS"]
    po, info = poses_for_factor(ds, cfg, args.factor)
    prob = build_problem(ds, cfg, po, "paper_loose",
                         tau_odom_override=args.tau_odom)
    pairs, pos = prob["pairs"], prob["positive"]
    print(f"dataset={args.dataset} frames={ds.n_frames} tau_odom={args.tau_odom} "
          f"tau_sim={args.tau_sim} cand={len(pairs)} pos={int(pos.sum())}")

    V = {c: np.stack([comp[i][c] for i in range(len(comp))]) for c in COMBOS["QHS"]}
    Z = {c: Normalizer().fit(V[c]).transform(V[c]) for c in V}

    def sim_from(vecs, names, weights=None, normalise=True, mode="inv"):
        total = np.zeros(len(pairs))
        wsum = 0.0
        for k, c in enumerate(names):
            src = Z[c] if normalise else V[c]
            A, B = src[pairs[:, 0]], src[pairs[:, 1]]
            d = rms(A, B)
            s = 1.0 / (1.0 + d) if mode == "inv" else np.exp(-d)
            w = 1.0 if weights is None else weights[k]
            total += w * s
            wsum += w
        return total / wsum

    rows = []

    def record(tag, sim):
        r = evaluate_similarity(sim, pos, cfg, fixed_tau=args.tau_sim)
        rf = evaluate_similarity(sim, pos, cfg)
        rows.append({"similarity": tag, "P@tau": r["precision"],
                     "R@tau": r["recall"], "loops@tau": r["closed_loops"],
                     "F1_max": rf["f1_max"], "AUC": rf["auc"],
                     "best_tau": rf["best_tau"]})
        print(f"  {tag:52s} P@{args.tau_sim}={r['precision']:.3f} "
              f"loops={r['closed_loops']:6d} F1max={rf['f1_max']:.3f} "
              f"best_tau={rf['best_tau']:.3f}", flush=True)

    print("\n[A] component-wise average, z-scored vs raw")
    for combo, names in COMBOS.items():
        record(f"{combo}: z-score + 1/(1+d_rms), equal weights",
               sim_from(V, names, normalise=True))
    for combo, names in COMBOS.items():
        record(f"{combo}: RAW (no normalisation) + 1/(1+d_rms)",
               sim_from(V, names, normalise=False))
    print("\n[B] full descriptor, other valid definitions")
    record("QHS: raw concatenated vector + 1/(1+d_rms)",
           1.0 / (1.0 + rms(np.concatenate([V[c] for c in COMBOS["QHS"]], 1)[pairs[:, 0]],
                            np.concatenate([V[c] for c in COMBOS["QHS"]], 1)[pairs[:, 1]])))
    record("QHS: z-score + exp(-d), equal weights",
           sim_from(V, COMBOS["QHS"], normalise=True, mode="exp"))
    record("QHS: z-score + 1/(1+d), weights 0.5/0.3/0.2",
           sim_from(V, COMBOS["QHS"], weights=(0.5, 0.3, 0.2), normalise=True))
    record("QHS: raw + 1/(1+d), weights 0.5/0.3/0.2",
           sim_from(V, COMBOS["QHS"], weights=(0.5, 0.3, 0.2), normalise=False))

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n  Reading: RAW (fewer scaling choices) makes *every* component "
          "degenerate - loops == candidates, i.e. zero filtering at "
          "tau_sim = 0.65 -\n  because 1/(1+d_rms) barely varies across pairs. "
          "A descriptor whose h_quad behaves like pure space\n  (paper Table 3, "
          "59.2% == pure space exactly) is therefore consistent with a weakly "
          "scaled/unnormalised\n  component average, while the same table shows "
          "h_hist *does* filter - so the two are not\n  on a common scale in "
          "the paper. Per-component z-scoring (this code base) makes all three "
          "components\n  comparable and yields the operating point reported in "
          "the CSV.")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"diagnose_similarity_{args.dataset}.csv")
    df.to_csv(path, index=False)
    print(f"\n  saved {path}")
    return df


if __name__ == "__main__":
    main()
