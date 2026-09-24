#!/usr/bin/env python3
"""Visualize the point clouds stored in down.txt and up.txt.

Expected column order: X, Y, Z, R, G, B, label.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASETS = {
    "Down": ROOT / "down.txt",
    "Up": ROOT / "up.txt",
}


def load_points(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(f"{path.name}: expected 7 columns, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path.name}: data contains NaN or infinite values")
    return data


def rgb(data: np.ndarray) -> np.ndarray:
    """Convert columns 4-6 from 8-bit RGB to Matplotlib RGB values."""
    return np.clip(data[:, 3:6] / 255.0, 0.0, 1.0)


def equal_3d_axes(ax, xyz: np.ndarray) -> None:
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    radius = np.ptp(xyz, axis=0).max() / 2
    radius = radius if radius > 0 else 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def style_projection(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.35, alpha=0.28)


def overview(data_by_name: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.15, 1, 1))

    for row, (name, data) in enumerate(data_by_name.items()):
        xyz, colors = data[:, :3], rgb(data)

        ax3d = fig.add_subplot(grid[row, 0], projection="3d")
        ax3d.scatter(
            xyz[:, 0], xyz[:, 1], xyz[:, 2],
            c=colors, s=10, alpha=1.0, linewidths=0, depthshade=False,
        )
        equal_3d_axes(ax3d, xyz)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.set_title(f"{name}: 3D RGB point cloud (n={len(data):,})")
        ax3d.view_init(elev=22, azim=-55)

        ax_xz = fig.add_subplot(grid[row, 1])
        ax_xz.scatter(xyz[:, 0], xyz[:, 2], c=colors, s=12, alpha=1.0, linewidths=0)
        style_projection(ax_xz, "X", "Z")
        ax_xz.set_title(f"{name}: X-Z projection")

        ax_xy = fig.add_subplot(grid[row, 2])
        ax_xy.scatter(xyz[:, 0], xyz[:, 1], c=colors, s=12, alpha=1.0, linewidths=0)
        style_projection(ax_xy, "X", "Y")
        ax_xy.set_title(f"{name}: X-Y projection")

    fig.suptitle("Point-cloud overview (point color = columns 4-6 RGB)", fontsize=16)
    fig.savefig(ROOT / "pointcloud_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def comparison(data_by_name: dict[str, np.ndarray]) -> None:
    down, up = data_by_name["Down"], data_by_name["Up"]
    all_xyz = np.vstack((down[:, :3], up[:, :3]))
    xlim = (all_xyz[:, 0].min(), all_xyz[:, 0].max())
    zlim = (all_xyz[:, 2].min(), all_xyz[:, 2].max())

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)

    # Separate panels preserve each point's original RGB while the shared limits
    # keep the geometry directly comparable.
    for ax, name, data in zip(axes[:2], ("Down", "Up"), (down, up)):
        ax.scatter(
            data[:, 0], data[:, 2],
            s=14, c=rgb(data), alpha=1.0, linewidths=0,
        )
        ax.set_xlim(xlim)
        ax.set_ylim(zlim)
        style_projection(ax, "X", "Z")
        ax.set_title(f"{name}: X-Z projection (original RGB)")

    labels = np.unique(np.concatenate((down[:, 6], up[:, 6])))
    labels = np.sort(labels)
    down_counts = np.array([(down[:, 6] == value).sum() for value in labels])
    up_counts = np.array([(up[:, 6] == value).sum() for value in labels])
    positions = np.arange(len(labels))
    width = 0.36
    axes[2].bar(positions - width / 2, down_counts, width, label="Down", color="#2878B5")
    axes[2].bar(positions + width / 2, up_counts, width, label="Up", color="#D95319")
    axes[2].set_xticks(positions, [f"{value:g}" for value in labels])
    axes[2].set_xlabel("Column 7 value (label)")
    axes[2].set_ylabel("Point count")
    axes[2].set_title("Label distribution")
    axes[2].grid(axis="y", linewidth=0.35, alpha=0.28)
    axes[2].legend(frameon=False)
    for x, count in zip(positions - width / 2, down_counts):
        axes[2].text(x, count, str(count), ha="center", va="bottom", fontsize=8)
    for x, count in zip(positions + width / 2, up_counts):
        axes[2].text(x, count, str(count), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Down vs Up: geometry and labels", fontsize=15)
    fig.savefig(ROOT / "pointcloud_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_summary(data_by_name: dict[str, np.ndarray]) -> None:
    for name, data in data_by_name.items():
        mins = data[:, :3].min(axis=0)
        maxs = data[:, :3].max(axis=0)
        labels, counts = np.unique(data[:, 6], return_counts=True)
        label_text = ", ".join(f"{label:g}: {count}" for label, count in zip(labels, counts))
        print(
            f"{name}: n={len(data)}, "
            f"X=[{mins[0]:.6f}, {maxs[0]:.6f}], "
            f"Y=[{mins[1]:.6f}, {maxs[1]:.6f}], "
            f"Z=[{mins[2]:.6f}, {maxs[2]:.6f}], labels={{ {label_text} }}"
        )


def main() -> None:
    data_by_name = {name: load_points(path) for name, path in DATASETS.items()}
    print_summary(data_by_name)
    overview(data_by_name)
    comparison(data_by_name)
    print("Saved pointcloud_overview.png and pointcloud_comparison.png")


if __name__ == "__main__":
    main()
