#!/usr/bin/env python3
"""Generate report-ready tables and figures for the ICP experiment."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from icp_test import apply_transform, icp


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "up_t.txt"
TARGET_PATH = ROOT / "down.txt"
MAX_ITERATIONS = 500
TOLERANCE = 1e-12
FINAL_QUANTILE = 1.0
QUANTILES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

BLUE = "#0077BB"
ORANGE = "#EE7733"
TEAL = "#009988"
RED = "#CC3311"


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Heiti SC", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.unicode_minus": False,
})


def load_xyz(path):
    data = np.loadtxt(path, delimiter=",")
    return data[:, :3]


def registration_metrics(source, target):
    forward = cKDTree(target).query(source)[0]
    reverse = cKDTree(source).query(target)[0]
    return {
        "mean_nn": float(forward.mean()),
        "median_nn": float(np.median(forward)),
        "rmse_nn": float(np.sqrt(np.mean(forward**2))),
        "reverse_mean_nn": float(reverse.mean()),
        "chamfer": float((forward.mean() + reverse.mean()) / 2),
    }


def equal_3d_axes(ax, points):
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(np.ptp(points, axis=0).max() / 2, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def save_figure(fig, stem):
    fig.savefig(ROOT / f"{stem}.png")
    fig.savefig(ROOT / f"{stem}.pdf")
    plt.close(fig)


def plot_registration(source, aligned, target):
    all_points = np.vstack([source, aligned, target])
    fig = plt.figure(figsize=(7.2, 6.4), constrained_layout=True)

    panels = [
        (source, "(a) 配准前"),
        (aligned, "(b) 配准后"),
    ]
    for column, (moving, title) in enumerate(panels, start=1):
        ax = fig.add_subplot(2, 2, column, projection="3d")
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], s=4, c=BLUE,
                   alpha=0.65, marker="o", label="固定点云 down")
        ax.scatter(moving[:, 0], moving[:, 1], moving[:, 2], s=4, c=ORANGE,
                   alpha=0.65, marker="^", label="移动点云 up_t")
        equal_3d_axes(ax, all_points)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.view_init(elev=22, azim=-55)
        ax.legend(frameon=False, loc="upper right")

        ax2 = fig.add_subplot(2, 2, column + 2)
        ax2.scatter(target[:, 0], target[:, 2], s=5, c=BLUE,
                    alpha=0.65, marker="o", label="固定点云 down")
        ax2.scatter(moving[:, 0], moving[:, 2], s=5, c=ORANGE,
                    alpha=0.65, marker="^", label="移动点云 up_t")
        ax2.set_xlim(all_points[:, 0].min(), all_points[:, 0].max())
        ax2.set_ylim(all_points[:, 2].min(), all_points[:, 2].max())
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.set_title(f"{title}：X-Z 投影")
        ax2.grid(True, linewidth=0.4, alpha=0.3)

    save_figure(fig, "report_figure1_registration")


def plot_metrics(before, after):
    labels = ["平均最近邻", "中位最近邻", "最近邻 RMSE", "双向 Chamfer"]
    before_values = [before["mean_nn"], before["median_nn"], before["rmse_nn"], before["chamfer"]]
    after_values = [after["mean_nn"], after["median_nn"], after["rmse_nn"], after["chamfer"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    bars1 = ax.bar(x - width / 2, before_values, width, color=BLUE, hatch="//",
                   edgecolor="black", linewidth=0.5, label="配准前")
    bars2 = ax.bar(x + width / 2, after_values, width, color=ORANGE, hatch="..",
                   edgecolor="black", linewidth=0.5, label="配准后")
    ax.set_yscale("log")
    ax.set_ylim(min(after_values) * 0.75, max(before_values) * 2.2)
    ax.set_ylabel("距离（对数尺度）")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("ICP 配准前后误差指标对比", pad=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", which="both", linewidth=0.4, alpha=0.3)
    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.4f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, rotation=45)
    fig.tight_layout()
    save_figure(fig, "report_figure2_metrics")


def plot_convergence(history):
    iterations = np.arange(1, len(history) + 1)
    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    ax.semilogy(iterations, history, color=BLUE, linewidth=1.5)
    ax.scatter([iterations[-1]], [history[-1]], color=RED, marker="o", s=30,
               zorder=3, label=f"第 {iterations[-1]} 次收敛")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("平均对应点距离（对数尺度）")
    ax.set_title("ICP 误差收敛曲线")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "report_figure3_convergence")


def plot_sensitivity(rows):
    quantiles = [row["quantile"] for row in rows]
    forward = [row["mean_nn"] for row in rows]
    chamfer = [row["chamfer"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.9, 4.5))
    ax.plot(quantiles, forward, color=BLUE, marker="o", linewidth=1.5,
            label="单向平均最近邻距离")
    ax.plot(quantiles, chamfer, color=ORANGE, marker="s", linestyle="--",
            linewidth=1.5, label="双向 Chamfer 距离")
    best = min(rows, key=lambda row: row["chamfer"])
    ax.scatter([best["quantile"]], [best["chamfer"]], color=RED, marker="*", s=90,
               zorder=3, label=f"最优比例 {best['quantile']:.1f}")
    ax.set_xlabel("对应点保留比例")
    ax.set_ylabel("距离")
    ax.set_title("对应点保留比例的参数敏感性")
    ax.set_xticks(quantiles)
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, "report_figure4_parameter_sensitivity")


def write_csvs(before, after, history, sweep_rows):
    with (ROOT / "report_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "before", "after", "improvement_percent"])
        for key in ("mean_nn", "median_nn", "rmse_nn", "reverse_mean_nn", "chamfer"):
            improvement = (1 - after[key] / before[key]) * 100
            writer.writerow([key, f"{before[key]:.10f}", f"{after[key]:.10f}", f"{improvement:.6f}"])

    with (ROOT / "report_convergence.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "mean_correspondence_distance"])
        writer.writerows((i, f"{value:.12f}") for i, value in enumerate(history, start=1))

    with (ROOT / "report_parameter_sweep.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["quantile", "iterations", "mean_nn", "reverse_mean_nn", "chamfer"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sweep_rows)


def write_report_snippet(before, after, result):
    improvement = (1 - after["mean_nn"] / before["mean_nn"]) * 100
    text = f"""# ICP 实验结果

## 实验数据与参数

| 项目 | 设置 |
|---|---:|
| 移动点云 | up_t.txt（1523 点） |
| 固定点云 | down.txt（1493 点） |
| 初始人工旋转 | 绕 Z 轴 20° |
| 初始人工平移 | (0.10, -0.05, 0.03) |
| RGB 噪声 | N(0, 8²)，随机种子 42 |
| 最大迭代次数 | {MAX_ITERATIONS} |
| 收敛阈值 | {TOLERANCE:.0e} |
| 对应点保留比例 | {FINAL_QUANTILE:.1f} |
| 实际迭代次数 | {result['iterations']} |

## 定量指标

| 指标 | 配准前 | 配准后 | 改善幅度 |
|---|---:|---:|---:|
| 平均最近邻距离 | {before['mean_nn']:.6f} | {after['mean_nn']:.6f} | {(1-after['mean_nn']/before['mean_nn'])*100:.2f}% |
| 最近邻距离中位数 | {before['median_nn']:.6f} | {after['median_nn']:.6f} | {(1-after['median_nn']/before['median_nn'])*100:.2f}% |
| 最近邻 RMSE | {before['rmse_nn']:.6f} | {after['rmse_nn']:.6f} | {(1-after['rmse_nn']/before['rmse_nn'])*100:.2f}% |
| 反向平均最近邻距离 | {before['reverse_mean_nn']:.6f} | {after['reverse_mean_nn']:.6f} | {(1-after['reverse_mean_nn']/before['reverse_mean_nn'])*100:.2f}% |
| 双向 Chamfer 距离 | {before['chamfer']:.6f} | {after['chamfer']:.6f} | {(1-after['chamfer']/before['chamfer'])*100:.2f}% |

## 结果描述

ICP 在第 {result['iterations']} 次迭代后收敛。平均最近邻距离从 {before['mean_nn']:.6f} 降至 {after['mean_nn']:.6f}，改善 {improvement:.2f}%；双向 Chamfer 距离从 {before['chamfer']:.6f} 降至 {after['chamfer']:.6f}。这表明优化后的点到点 ICP 能显著缩小两组点云的几何偏差。RGB 属性未参与对应点搜索，因此本结果反映的是几何配准性能，不代表对颜色噪声的鲁棒性。

## 图题

- 图 1：ICP 配准前后的三维点云及 X-Z 投影对比。蓝色圆点为固定点云 down，橙色三角为移动点云 up_t。
- 图 2：ICP 配准前后距离误差指标对比。纵轴使用对数尺度。
- 图 3：ICP 平均对应点距离的迭代收敛曲线。
- 图 4：对应点保留比例对配准误差的影响。
"""
    (ROOT / "icp_experiment_results.md").write_text(text, encoding="utf-8")


def main():
    source = load_xyz(SOURCE_PATH)
    target = load_xyz(TARGET_PATH)
    result = icp(source, target, max_iterations=MAX_ITERATIONS,
                 tolerance=TOLERANCE, distance_quantile=FINAL_QUANTILE)
    aligned = apply_transform(source, result["rotation"], result["translation"])

    before = registration_metrics(source, target)
    after = registration_metrics(aligned, target)

    sweep_rows = []
    for quantile in QUANTILES:
        sweep = icp(source, target, max_iterations=MAX_ITERATIONS,
                    tolerance=TOLERANCE, distance_quantile=float(quantile))
        moved = apply_transform(source, sweep["rotation"], sweep["translation"])
        metrics = registration_metrics(moved, target)
        sweep_rows.append({
            "quantile": f"{quantile:.1f}",
            "iterations": sweep["iterations"],
            "mean_nn": f"{metrics['mean_nn']:.10f}",
            "reverse_mean_nn": f"{metrics['reverse_mean_nn']:.10f}",
            "chamfer": f"{metrics['chamfer']:.10f}",
        })

    numeric_sweep = [
        {key: (float(value) if key != "iterations" else int(value)) for key, value in row.items()}
        for row in sweep_rows
    ]
    plot_registration(source, aligned, target)
    plot_metrics(before, after)
    plot_convergence(result["error_history"])
    plot_sensitivity(numeric_sweep)
    write_csvs(before, after, result["error_history"], sweep_rows)
    write_report_snippet(before, after, result)

    print(f"iterations={result['iterations']}")
    print(f"mean_nn: {before['mean_nn']:.10f} -> {after['mean_nn']:.10f}")
    print(f"chamfer: {before['chamfer']:.10f} -> {after['chamfer']:.10f}")
    print("Generated 4 PNG figures, 4 PDF figures, 3 CSV files, and 1 Markdown report snippet.")


if __name__ == "__main__":
    main()
