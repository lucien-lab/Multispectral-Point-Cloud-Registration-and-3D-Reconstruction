"""Figures: PR curves (paper Fig. 1) and the component-ablation summary.

Usage
-----
    python make_figures.py --frames 300 --points 12000
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, RESULTS_DIR  # noqa: E402
from pipeline import build_methods, extract_features, poses_for_factor  # noqa: E402
from data_io import load_dataset  # noqa: E402
from retrieval import build_problem, evaluate_similarity  # noqa: E402

DISPLAY = {"JSSS": "JSSS (ours)", "ScanContext": "ScanContext [7]",
           "M2DP": "M2DP [8]", "IntensityHist": "Intensity-Hist",
           "IntensityStats": "Intensity-Stats", "GlobalHist_norm": "GlobalHist (norm)",
           "RadialHist": "RadialHist", "PureSpace_tight": "Pure-space"}


def pr_curves(cfg: Config, out_dir: str):
    ds = load_dataset(cfg)
    methods = build_methods(cfg)
    feats = extract_features(ds, cfg, methods, verbose=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, factor in zip(axes, (1.0, 2.0)):
        po, info = poses_for_factor(ds, cfg, factor)
        prob = build_problem(ds, cfg, po, "paper_pr")
        for name in ("JSSS", "ScanContext", "M2DP", "IntensityHist", "IntensityStats"):
            desc = methods[name]
            pairs = prob["pairs"]
            sim = desc.similarity([feats[name][i] for i in pairs[:, 0]],
                                  [feats[name][i] for i in pairs[:, 1]])
            r = evaluate_similarity(sim, prob["positive"], cfg)
            c = r["curve"]
            order = np.argsort(c.recall)
            ax.plot(c.recall[order], c.precision[order], marker=".",
                    label=f"{DISPLAY.get(name, name)} (AUC={r['auc']:.3f})")
        ax.set_xlabel("Recall")
        ax.set_title(f"drift {factor:.1f}x (RMS {info['rms_m']:.1f} m)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Precision")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("PR curves - JSSS vs baselines")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_pr_curves.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {path}")


def component_bars(csv_path: str, out_dir: str):
    if not os.path.isfile(csv_path):
        print(f"  skip (missing {csv_path})")
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(df))
    ax.bar(x - 0.2, df["F1_max"], width=0.4, label="F1-max")
    ax.bar(x + 0.2, df["AUC"], width=0.4, label="AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(df["components"], rotation=0)
    ax.set_ylim(0, 1)
    ax.set_title("Component ablation (Q/H/S and combinations)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_component_ablation.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--points", type=int, default=12000)
    ap.add_argument("--out", default=RESULTS_DIR)
    args = ap.parse_args(argv)
    cfg = Config()
    cfg.synth.n_frames = args.frames
    cfg.synth.points_per_frame = args.points
    os.makedirs(args.out, exist_ok=True)
    pr_curves(cfg, args.out)
    component_bars(os.path.join(args.out, "table3_components.csv"), args.out)


if __name__ == "__main__":
    main()
