#!/usr/bin/env python3
"""Generate report figures comparing geometry-only and geometry+RGB ICP."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parent
BLUE = "#0077BB"
ORANGE = "#EE7733"


def configure_matplotlib() -> None:
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        family = font_manager.FontProperties(fname=font_path).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def equal_3d_axes(ax, points: np.ndarray) -> None:
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2
    radius = max(np.ptp(points, axis=0).max() / 2, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def plot_registration_comparison() -> None:
    source = np.loadtxt(ROOT / "up_t.txt", delimiter=",")[:, :3]
    target = np.loadtxt(ROOT / "down.txt", delimiter=",")[:, :3]
    geometry = np.loadtxt(ROOT / "up_t_aligned_refined.txt", delimiter=",")[:, :3]
    colored = np.loadtxt(ROOT / "up_t_aligned_rgb.txt", delimiter=",")[:, :3]

    panels = [
        (source, "(a) 配准前"),
        (geometry, "(b) 纯几何 ICP"),
        (colored, "(c) 几何+RGB ICP"),
    ]
    fig = plt.figure(figsize=(8.0, 5.7), constrained_layout=True)
    for column, (moving, title) in enumerate(panels, start=1):
        local = np.vstack((moving, target))
        ax = fig.add_subplot(2, 3, column, projection="3d")
        ax.scatter(
            target[:, 0], target[:, 1], target[:, 2],
            s=4, c=BLUE, alpha=0.62, marker="o", label="固定点云 down",
        )
        ax.scatter(
            moving[:, 0], moving[:, 1], moving[:, 2],
            s=5, c=ORANGE, alpha=0.62, marker="^", label="移动点云 up_t",
        )
        equal_3d_axes(ax, local)
        ax.view_init(elev=22, azim=-55)
        ax.set_xlabel("X/m", labelpad=-1)
        ax.set_ylabel("Y/m", labelpad=-1)
        ax.set_zlabel("Z/m", labelpad=-1)
        ax.set_title(title)
        if column == 1:
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02))

        ax2 = fig.add_subplot(2, 3, column + 3)
        ax2.scatter(
            target[:, 0], target[:, 2], s=5, c=BLUE,
            alpha=0.62, marker="o", label="固定点云 down",
        )
        ax2.scatter(
            moving[:, 0], moving[:, 2], s=6, c=ORANGE,
            alpha=0.62, marker="^", label="移动点云 up_t",
        )
        ax2.set_xlim(local[:, 0].min(), local[:, 0].max())
        ax2.set_ylim(local[:, 2].min(), local[:, 2].max())
        ax2.set_xlabel("X/m")
        ax2.set_ylabel("Z/m")
        ax2.set_title(f"{title}：X–Z 投影")
        ax2.grid(True, linewidth=0.4, alpha=0.3)

    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"report_figure_rgb_registration_comparison.{suffix}")
    plt.close(fig)


def load_metrics() -> tuple[dict, dict]:
    with (ROOT / "icp_rgb_metrics.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_method = {row["method"]: row for row in rows}
    return by_method["geometry_icp"], by_method["geometry_rgb_icp"]


def annotate_bars(ax, bars, decimals: int) -> None:
    for bar in bars:
        ax.annotate(
            f"{bar.get_height():.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=7,
        )


def plot_metrics_comparison() -> None:
    geometry, colored = load_metrics()
    geo_keys = ["mean_nn_m", "rmse_nn_m", "reverse_mean_nn_m", "chamfer_m"]
    geo_labels = ["平均最近邻", "最近邻 RMSE", "反向平均最近邻", "双向 Chamfer"]
    rgb_keys = [
        "spatial_nn_rgb_mean",
        "spatial_nn_rgb_rmse",
        "optimization_pair_rgb_mean",
        "optimization_pair_rgb_rmse",
    ]
    rgb_labels = ["空间NN平均", "空间NN RMSE", "优化对应平均", "优化对应RMSE"]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.65), constrained_layout=True)
    width = 0.36

    x = np.arange(len(geo_labels))
    values_geometry = [float(geometry[key]) * 1000 for key in geo_keys]
    values_colored = [float(colored[key]) * 1000 for key in geo_keys]
    bars_g = axes[0].bar(
        x - width / 2, values_geometry, width, label="纯几何 ICP",
        color=BLUE, edgecolor="black", linewidth=0.5, hatch="//",
    )
    bars_c = axes[0].bar(
        x + width / 2, values_colored, width, label="几何+RGB ICP",
        color=ORANGE, edgecolor="black", linewidth=0.5, hatch="..",
    )
    axes[0].set_ylabel("距离/mm")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(geo_labels, rotation=18, ha="right")
    axes[0].set_ylim(0, max(values_geometry + values_colored) * 1.23)
    axes[0].set_title("(a) 独立几何误差")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].grid(axis="y", linewidth=0.4, alpha=0.3)
    annotate_bars(axes[0], bars_g, 3)
    annotate_bars(axes[0], bars_c, 3)

    x = np.arange(len(rgb_labels))
    values_geometry = [float(geometry[key]) for key in rgb_keys]
    values_colored = [float(colored[key]) for key in rgb_keys]
    bars_g = axes[1].bar(
        x - width / 2, values_geometry, width, label="纯几何 ICP",
        color=BLUE, edgecolor="black", linewidth=0.5, hatch="//",
    )
    bars_c = axes[1].bar(
        x + width / 2, values_colored, width, label="几何+RGB ICP",
        color=ORANGE, edgecolor="black", linewidth=0.5, hatch="..",
    )
    axes[1].set_ylabel("RGB 欧氏距离")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(rgb_labels, rotation=18, ha="right")
    axes[1].set_ylim(0, max(values_geometry + values_colored) * 1.22)
    axes[1].set_title("(b) RGB 对应残差")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].grid(axis="y", linewidth=0.4, alpha=0.3)
    annotate_bars(axes[1], bars_g, 1)
    annotate_bars(axes[1], bars_c, 1)

    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"report_figure_rgb_metrics_comparison.{suffix}")
    plt.close(fig)


def main() -> None:
    configure_matplotlib()
    plot_registration_comparison()
    plot_metrics_comparison()
    print("Generated RGB ICP registration and metric comparison figures (PNG/PDF).")


if __name__ == "__main__":
    main()
