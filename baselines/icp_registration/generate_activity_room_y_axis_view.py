#!/usr/bin/env python3
"""Generate the activity-room registration comparison viewed along the Y axis."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE_SCRIPT = Path(
    "/Users/lucien/workspace/graduate-study/papers/3_28/20260328/activity_room_experiment.py"
)
OUTPUT = Path(__file__).resolve().parent / "activity_room_registration_y_axis_view.png"


def load_experiment_module():
    spec = importlib.util.spec_from_file_location("activity_room_experiment_source", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    experiment = load_experiment_module()
    experiment.configure_plots()

    left, _ = experiment.load_side("left")
    right, _ = experiment.load_side("right")
    aligned, _, _ = experiment.overlap_icp(right, left)

    all_points = np.vstack((left, right, aligned))
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()
    x_pad = max((x_max - x_min) * 0.03, 1.0)
    z_pad = max((z_max - z_min) * 0.05, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True, sharey=True, constrained_layout=True)
    panels = ((right, "配准前"), (aligned, "重叠区 ICP 配准后"))

    for ax, (moving, title) in zip(axes, panels):
        ax.scatter(
            left[:, 0],
            left[:, 2],
            s=6,
            c="#1f77b4",
            alpha=0.78,
            linewidths=0,
            label="左侧扫描",
            rasterized=True,
        )
        ax.scatter(
            moving[:, 0],
            moving[:, 2],
            s=6,
            c="#ff7f0e",
            alpha=0.72,
            linewidths=0,
            label="右侧扫描",
            rasterized=True,
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel("X / mm")
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(z_min - z_pad, z_max + z_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.30)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

    axes[0].set_ylabel("Z / mm")
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
