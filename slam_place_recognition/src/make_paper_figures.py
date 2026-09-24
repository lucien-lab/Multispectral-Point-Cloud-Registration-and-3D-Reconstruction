"""Figures for the rewritten manuscript (v11): built purely from the CSV records.

Reads only CSV outputs so it runs in a second and never re-runs the pipeline.

    python make_paper_figures.py --results ../results

Produces (into <results>/kitti/):
    fig1_unified_pr.png       unified-protocol F1_max / AUC across drift levels
    fig2_fixedtau_tradeoff.png  fixed-threshold R/precision trade-off vs search radius
    fig3_components.png       component-combination ablation with accept-all baseline
    fig4_normalisation.png    z-score vs raw similarity: zero-filtering diagnostic
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

DISPLAY = {
    "JSSS": "JSSS (ours)",
    "ScanContext": "ScanContext [7]",
    "M2DP": "M2DP [8]",
    "GlobalHist_norm": "Spectral-norm hist.",
    "IntensityHist": "Spectral-sum hist.",
    "RadialHist": "Radial hist.",
    "IntensityStats": "Spectral-sum stats",
}
ORDER = ["JSSS", "ScanContext", "M2DP", "GlobalHist_norm", "IntensityHist", "RadialHist"]
COLOR = {"JSSS": "#d62728", "ScanContext": "#1f77b4", "M2DP": "#7f7f7f",
         "GlobalHist_norm": "#2ca02c", "IntensityHist": "#ff7f0e",
         "RadialHist": "#9467bd", "IntensityStats": "#8c564b"}


def _style(ax):
    ax.grid(axis="y", ls=":", lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig1_unified(path: str, out: str):
    df = pd.read_csv(path)
    drifts = ["1.0", "1.5", "2.0"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    x = np.arange(len(drifts))
    w = 0.14
    for ax, col, ylab in ((axes[0], "F1_max", "F1-max"),
                          (axes[1], "AUC", "PR-AUC")):
        for k, m in enumerate(ORDER):
            d = df[df.method == m].set_index(df[df.method == m]["drift"].astype(str))
            vals = [float(d.loc[dd, col]) for dd in drifts]
            ax.bar(x + (k - 2.5) * w, vals, w, label=DISPLAY[m],
                   color=COLOR[m], edgecolor="black", lw=0.4)
        acc = [float(df[(df.method == "PureSpace_tight") &
                        (df["drift"].astype(str) == dd)]["accept_all_F1"].iloc[0])
               for dd in drifts]
        if col == "F1_max":
            for xi, a in zip(x, acc):
                ax.hlines(a, xi - 3.4 * w, xi + 3.4 * w, color="black",
                          ls="--", lw=1.2)
                ax.text(xi + 3.5 * w, a + 0.008, "accept-all", fontsize=7,
                        ha="right", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([f"drift {d}x" for d in drifts])
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.02)
        _style(ax)
    axes[0].legend(fontsize=7.5, ncol=2, loc="lower left", framealpha=0.9)
    axes[0].set_title("(a) Threshold-free F1-max", fontsize=10)
    axes[1].set_title("(b) PR-AUC", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig2_fixedtau(path: str, out: str):
    df = pd.read_csv(path)
    df = df[df["drift"] == 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    taus = sorted(df.tau_odom_m.unique())
    for m, c in (("PureSpace_tight", "#333333"), ("JSSS", "#d62728")):
        sub = df[df.method == m].set_index("tau_odom_m")
        axes[0].plot(taus, [sub.loc[t, "recall"] for t in taus], "o-",
                     color=c, label=DISPLAY.get(m, "pure space (no filter)"))
        axes[0].plot(taus, [sub.loc[t, "precision"] for t in taus], "s--",
                     color=c, alpha=0.55,
                     label=("JSSS precision" if m == "JSSS" else "no-filter precision"))
        axes[1].plot(taus, [sub.loc[t, "f1"] for t in taus], "o-", color=c,
                     label=DISPLAY.get(m, "pure space (no filter)"))
        axes[1].plot(taus, [sub.loc[t, "closed_loops"] for t in taus], "^--",
                     color=c, alpha=0.55,
                     label=("JSSS accepted" if m == "JSSS" else "no-filter accepted"))
    axes[0].set_xlabel(r"search radius $\tau_{odom}$ (m)")
    axes[0].set_ylabel("precision / recall")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("(a) Precision and recall", fontsize=10)
    axes[0].legend(fontsize=7.5)
    axes[1].set_xlabel(r"search radius $\tau_{odom}$ (m)")
    axes[1].set_ylabel("F1 (solid) / accepted pairs (dashed)")
    axes[1].set_title("(b) Net F1 gain vs candidate volume", fontsize=10)
    axes[1].legend(fontsize=7.5)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig3_components(path: str, out: str):
    df = pd.read_csv(path)
    labels = list(df["components"])
    accept = float(df["accept_all_F1"].iloc[0])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    axes[0].bar(x - 0.19, df["F1_max"], 0.36, label="F1-max",
                color="#4c72b0", edgecolor="black", lw=0.4)
    axes[0].bar(x + 0.19, df["F1@tau"], 0.36, label=r"$F1$ at fixed $\tau_{sim}=0.65$",
                color="#dd8452", edgecolor="black", lw=0.4)
    axes[0].axhline(accept, color="black", ls="--", lw=1.2)
    axes[0].text(len(labels) - 0.5, accept + 0.004, "accept-all", fontsize=7.5,
                 ha="right")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title("(a) F1-max and fixed-threshold F1", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].bar(x, df["AUC"], 0.55, color="#55a868", edgecolor="black", lw=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("(b) PR-AUC", fontsize=10)
    for ax in axes:
        _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def fig4_normalisation(path: str, out: str):
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    raw = df["similarity"].str.contains("RAW")
    for mask, tag, c, mk in ((~raw, "z-score normalised", "#1f77b4", "o"),
                             (raw, "raw scale (no normalisation)", "#d62728", "X")):
        sub = df[mask]
        axes[0].scatter(sub["R@tau"], sub["P@tau"], s=64, c=c, marker=mk,
                        edgecolor="black", lw=0.5, label=tag, zorder=3)
        axes[1].scatter(sub["F1_max"], sub["AUC"], s=64, c=c, marker=mk,
                        edgecolor="black", lw=0.5, label=tag, zorder=3)
    acc_f1 = float(df["accept_all_F1"].iloc[0]) if "accept_all_F1" in df else np.nan
    p_acc = float(df[raw]["P@tau"].iloc[0])
    axes[0].scatter([1.0], [p_acc], s=170, facecolor="none", edgecolor="black",
                    marker="*", lw=1.4, zorder=4, label="zero filtering (R=1.000)")
    axes[0].set_xlabel("recall at fixed similarity threshold")
    axes[0].set_ylabel("precision at fixed similarity threshold")
    axes[0].set_title(r"(a) Fixed threshold $\tau_{sim}=0.65$", fontsize=10)
    axes[0].annotate("no filtering:\nall candidates accepted",
                     xy=(1.0, p_acc), xytext=(0.72, p_acc - 0.14),
                     fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    axes[1].axhline(acc_f1, color="black", ls="--", lw=1.2)
    axes[1].text(0.852, acc_f1 + 0.004, "accept-all F1", fontsize=7.5)
    axes[1].set_xlabel("F1-max")
    axes[1].set_ylabel("PR-AUC")
    axes[1].set_title("(b) Threshold-free metrics still look healthy", fontsize=10)
    for ax in axes:
        ax.legend(fontsize=7.5, loc="lower left")
        ax.grid(ls=":", lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "..", "results"))
    args = ap.parse_args()
    k = os.path.join(args.results, "kitti")
    fig1_unified(os.path.join(k, "table5_pr.csv"), os.path.join(k, "fig1_unified_pr.png"))
    fig2_fixedtau(os.path.join(k, "table1_search_radius.csv"),
                  os.path.join(k, "fig2_fixedtau_tradeoff.png"))
    fig3_components(os.path.join(k, "table3_components.csv"),
                    os.path.join(k, "fig3_components.png"))
    fig4_normalisation(os.path.join(args.results, "diagnose_similarity_kitti.csv"),
                       os.path.join(k, "fig4_normalisation.png"))
    print("figures written to", k)


if __name__ == "__main__":
    main()
