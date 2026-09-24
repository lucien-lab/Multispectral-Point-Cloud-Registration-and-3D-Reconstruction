"""点云PLY和实验对比图的统一保存。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from .evaluation import make_point_cloud

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)

SOURCE_COLOR = np.array([0.0, 0.75, 0.1])
TARGET_COLOR = np.array([0.9, 0.1, 0.1])
METHOD_COLORS = {
    "geometry": "#6B8FB3",
    "rgb": "#D48A6A",
    "spectral6": "#7AAE92",
}


def _equal_axes(ax, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 0.65))


def _scatter_pair(ax, source: np.ndarray, target: np.ndarray, title: str) -> None:
    ax.scatter(*source.T, s=3, c=[SOURCE_COLOR], label="source / left", depthshade=False)
    ax.scatter(*target.T, s=3, c=[TARGET_COLOR], label="target / right", depthshade=False)
    ax.set_title(title)
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.view_init(elev=22, azim=-58)
    _equal_axes(ax, np.vstack((source, target)))
    ax.legend(fontsize=8)


def save_before_after(
    source_perturbed: np.ndarray,
    target: np.ndarray,
    source_registered: np.ndarray,
    output_path: Path,
    method_label: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    _scatter_pair(
        fig.add_subplot(1, 2, 1, projection="3d"),
        source_perturbed,
        target,
        "Before registration",
    )
    _scatter_pair(
        fig.add_subplot(1, 2, 2, projection="3d"),
        source_registered,
        target,
        "After registration",
    )
    fig.suptitle(method_label)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_before_coarse_after(
    source_perturbed: np.ndarray,
    target: np.ndarray,
    source_coarse: np.ndarray,
    source_registered: np.ndarray,
    output_path: Path,
    method_label: str,
) -> None:
    """保存粗配准前、粗配准后和细配准后的三阶段对比图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(16.5, 5.2), constrained_layout=True)
    stages = (
        (source_perturbed, "Before registration"),
        (source_coarse, "After coarse registration"),
        (source_registered, "After fine registration"),
    )
    for index, (source, title) in enumerate(stages, 1):
        _scatter_pair(
            fig.add_subplot(1, 3, index, projection="3d"),
            source,
            target,
            title,
        )
    fig.suptitle(method_label)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_coarse_registration_clouds(
    output_dir: Path,
    source_coarse_aligned: np.ndarray,
    target: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> None:
    """保存粗配准后的source、红绿诊断合并点云和真实RGB合并点云。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(
        str(output_dir / "source_coarse_aligned.ply"),
        make_point_cloud(source_coarse_aligned, source_rgb),
        write_ascii=False,
        compressed=False,
    )
    merged_points = np.vstack((source_coarse_aligned, target))
    merged_colors = np.vstack(
        (
            np.tile(SOURCE_COLOR, (len(source_coarse_aligned), 1)),
            np.tile(TARGET_COLOR, (len(target), 1)),
        )
    )
    o3d.io.write_point_cloud(
        str(output_dir / "source_target_coarse_merged.ply"),
        make_point_cloud(merged_points, merged_colors),
        write_ascii=False,
        compressed=False,
    )
    o3d.io.write_point_cloud(
        str(output_dir / "source_target_coarse_rgb_merged.ply"),
        make_point_cloud(
            merged_points,
            np.vstack((source_rgb, target_rgb)),
        ),
        write_ascii=False,
        compressed=False,
    )


def write_registration_clouds(
    output_dir: Path,
    source_perturbed: np.ndarray,
    source_registered: np.ndarray,
    target: np.ndarray,
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_initial_registration_cloud(
        output_dir,
        source_perturbed,
        target,
    )
    o3d.io.write_point_cloud(
        str(output_dir / "source_perturbed.ply"),
        make_point_cloud(source_perturbed, source_rgb),
        write_ascii=False,
        compressed=False,
    )
    o3d.io.write_point_cloud(
        str(output_dir / "source_registered.ply"),
        make_point_cloud(source_registered, source_rgb),
        write_ascii=False,
        compressed=False,
    )
    merged_points = np.vstack((source_registered, target))
    merged_colors = np.vstack(
        (
            np.tile(SOURCE_COLOR, (len(source_registered), 1)),
            np.tile(TARGET_COLOR, (len(target), 1)),
        )
    )
    o3d.io.write_point_cloud(
        str(output_dir / "source_target_merged.ply"),
        make_point_cloud(merged_points, merged_colors),
        write_ascii=False,
        compressed=False,
    )
    o3d.io.write_point_cloud(
        str(output_dir / "source_target_rgb_merged.ply"),
        make_point_cloud(
            merged_points,
            np.vstack((source_rgb, target_rgb)),
        ),
        write_ascii=False,
        compressed=False,
    )


def write_initial_registration_cloud(
    output_dir: Path,
    source_perturbed: np.ndarray,
    target: np.ndarray,
) -> Path:
    """将配准前的source和target以绿/红诊断色保存到同一个PLY。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "source_target_initial_merged.ply"
    merged_points = np.vstack((source_perturbed, target))
    merged_colors = np.vstack(
        (
            np.tile(SOURCE_COLOR, (len(source_perturbed), 1)),
            np.tile(TARGET_COLOR, (len(target), 1)),
        )
    )
    if not o3d.io.write_point_cloud(
        str(output_path),
        make_point_cloud(merged_points, merged_colors),
        write_ascii=False,
        compressed=False,
    ):
        raise OSError(f"无法写入初始合并点云: {output_path}")
    return output_path


def save_method_comparison(
    registered_by_method: dict[str, np.ndarray],
    target: np.ndarray,
    output_path: Path,
) -> None:
    if not registered_by_method:
        return
    methods = list(registered_by_method)
    fig = plt.figure(figsize=(5.6 * len(methods), 5.0), constrained_layout=True)
    for index, method in enumerate(methods, 1):
        _scatter_pair(
            fig.add_subplot(1, len(methods), index, projection="3d"),
            registered_by_method[method],
            target,
            method,
        )
    fig.suptitle("Trial 001 registration comparison")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_error_boxplots(rows: list[dict[str, object]], output_path: Path) -> None:
    valid = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("rotation_error_deg") is not None
        and row.get("translation_error_m") is not None
    ]
    if not valid:
        return
    methods = list(dict.fromkeys(str(row["method"]) for row in valid))
    rotation = [
        [float(row["rotation_error_deg"]) for row in valid if row["method"] == method]
        for method in methods
    ]
    translation = [
        [1000.0 * float(row["translation_error_m"]) for row in valid if row["method"] == method]
        for method in methods
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].boxplot(rotation, tick_labels=methods)
    axes[0].set_ylabel("Rotation error / deg")
    axes[0].grid(alpha=0.25)
    axes[1].boxplot(translation, tick_labels=methods)
    axes[1].set_ylabel("Translation error / mm")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Registration error over trials")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_coarse_and_fine_error_boxplots(
    rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """以统一方法配色比较粗配准和最终细配准误差。"""
    valid = [
        row
        for row in rows
        if row.get("status") == "ok"
        and row.get("rotation_error_deg") is not None
        and row.get("translation_error_m") is not None
        and row.get("coarse_rotation_error_deg") is not None
        and row.get("coarse_translation_error_m") is not None
    ]
    if not valid:
        return
    methods = [
        method
        for method in ("geometry", "rgb", "spectral6")
        if any(row["method"] == method for row in valid)
    ]
    panels = (
        ("coarse_rotation_error_deg", 1.0, "Coarse rotation error / deg", 2.0),
        (
            "coarse_translation_error_m",
            1000.0,
            "Coarse translation error / mm",
            100.0,
        ),
        ("rotation_error_deg", 1.0, "Fine rotation error / deg", 2.0),
        (
            "translation_error_m",
            1000.0,
            "Fine translation error / mm",
            100.0,
        ),
    )
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
        }
    ):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(7.2, 6.0),
            constrained_layout=True,
        )
        for panel_label, (axis, panel) in zip(
            ("a", "b", "c", "d"),
            zip(axes.flat, panels),
        ):
            key, scale, ylabel, threshold = panel
            values = [
                [
                    scale * float(row[key])
                    for row in valid
                    if row["method"] == method
                ]
                for method in methods
            ]
            artists = axis.boxplot(
                values,
                tick_labels=methods,
                patch_artist=True,
                widths=0.58,
                medianprops={"color": "#2B2B2B", "linewidth": 1.2},
                whiskerprops={"color": "#555555", "linewidth": 0.8},
                capprops={"color": "#555555", "linewidth": 0.8},
                flierprops={
                    "marker": "o",
                    "markersize": 3,
                    "markerfacecolor": "#555555",
                    "markeredgewidth": 0,
                    "alpha": 0.55,
                },
            )
            for box, method in zip(artists["boxes"], methods):
                box.set_facecolor(METHOD_COLORS[method])
                box.set_alpha(0.78)
                box.set_edgecolor("#444444")
                box.set_linewidth(0.8)
            axis.axhline(
                threshold,
                color="#9C3D3D",
                linestyle="--",
                linewidth=0.9,
                alpha=0.8,
            )
            axis.set_ylabel(ylabel)
            axis.grid(axis="y", alpha=0.2, linewidth=0.6)
            axis.text(
                -0.14,
                1.05,
                panel_label,
                transform=axis.transAxes,
                fontweight="bold",
                fontsize=10,
            )
        fig.suptitle(
            "Method-specific coarse initialization and final registration",
            fontsize=10,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(
            output_path.with_suffix(".svg"),
            bbox_inches="tight",
        )
        fig.savefig(
            output_path.with_suffix(".pdf"),
            bbox_inches="tight",
        )
        fig.savefig(
            output_path.with_suffix(".tiff"),
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)
