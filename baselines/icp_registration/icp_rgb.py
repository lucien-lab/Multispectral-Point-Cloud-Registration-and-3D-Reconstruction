#!/usr/bin/env python3
"""Rigid ICP using joint XYZ and RGB correspondence constraints.

RGB does not undergo a rigid transform.  It therefore participates in the
correspondence score and robust correspondence weights, while R and t are
estimated from a weighted 3-D point-to-point objective.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from icp_test import apply_transform, icp, save_aligned, save_merged, save_transform, visualize


def load_cloud(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load X,Y,Z,R,G,B,... and return XYZ, RGB, and the complete array."""
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(f"{path} must contain at least X,Y,Z,R,G,B columns")
    if not np.isfinite(data[:, :6]).all():
        raise ValueError(f"{path} contains NaN or infinite XYZ/RGB values")
    return data[:, :3], np.clip(data[:, 3:6], 0.0, 255.0), data


def weighted_best_fit_transform(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a proper rigid transform using weighted SVD."""
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(weights.sum())
    if len(source) < 3 or weight_sum <= np.finfo(float).eps:
        raise RuntimeError("Too few valid weighted correspondences")

    weights = weights / weight_sum
    src_centroid = np.sum(source * weights[:, None], axis=0)
    tgt_centroid = np.sum(target * weights[:, None], axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid
    covariance = (src_centered * weights[:, None]).T @ tgt_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = tgt_centroid - rotation @ src_centroid
    return rotation, translation


def _annealed_scale(start: float, end: float, iteration: int, anneal_iterations: int) -> float:
    """Exponential coarse-to-fine scale used by joint correspondence search."""
    if anneal_iterations <= 1 or iteration >= anneal_iterations:
        return end
    fraction = (iteration - 1) / (anneal_iterations - 1)
    return float(start * (end / start) ** fraction)


def joint_correspondences(
    moved_xyz: np.ndarray,
    source_rgb: np.ndarray,
    target_xyz: np.ndarray,
    target_rgb: np.ndarray,
    geometry_scale: float,
    rgb_scale: float,
    rgb_weight: float,
    local_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use RGB to rank a local set of geometrically plausible candidates."""
    candidate_count = min(max(int(local_candidates), 1), len(target_xyz))
    candidate_geometry, candidate_indices = cKDTree(target_xyz).query(
        moved_xyz, k=candidate_count
    )
    if candidate_count == 1:
        candidate_geometry = candidate_geometry[:, None]
        candidate_indices = candidate_indices[:, None]
    candidate_rgb = np.linalg.norm(
        source_rgb[:, None, :] - target_rgb[candidate_indices], axis=2
    )
    candidate_scores = (
        (candidate_geometry / geometry_scale) ** 2
        + rgb_weight * (candidate_rgb / rgb_scale) ** 2
    )
    rows = np.arange(len(moved_xyz))
    choices = np.argmin(candidate_scores, axis=1)
    indices = candidate_indices[rows, choices]
    geometry_distances = np.linalg.norm(moved_xyz - target_xyz[indices], axis=1)
    rgb_distances = np.linalg.norm(source_rgb - target_rgb[indices], axis=1)
    joint_squared = (
        (geometry_distances / geometry_scale) ** 2
        + rgb_weight * (rgb_distances / rgb_scale) ** 2
    )
    return indices, geometry_distances, rgb_distances, joint_squared


def rgb_icp(
    source_xyz: np.ndarray,
    source_rgb: np.ndarray,
    target_xyz: np.ndarray,
    target_rgb: np.ndarray,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
    distance_quantile: float = 0.9,
    rgb_weight: float = 1.0,
    rgb_scale: float = 30.0,
    geometry_scale_start: float = 0.008,
    geometry_scale_end: float = 0.003,
    anneal_iterations: int = 40,
    local_candidates: int = 30,
    geometry_init_iterations: int = 500,
    geometry_init_quantile: float = 1.0,
) -> dict:
    """Run coarse-to-fine RGB-constrained rigid ICP.

    At iteration k, correspondences minimize

        ||R p_i + t - q_j||^2 / sigma_g(k)^2
        + lambda_rgb ||c_i - d_j||^2 / sigma_c^2.

    RGB residuals also define Gaussian weights for the weighted SVD update.
    """
    if not 0 < distance_quantile <= 1:
        raise ValueError("distance_quantile must be in (0, 1]")
    if rgb_weight < 0 or rgb_scale <= 0:
        raise ValueError("rgb_weight must be non-negative and rgb_scale positive")
    if geometry_scale_start <= 0 or geometry_scale_end <= 0:
        raise ValueError("geometry scales must be positive")

    # Geometry ICP supplies a stable basin. RGB then disambiguates only local
    # candidates instead of matching similar colors across distant palette cells.
    initial = icp(
        source_xyz,
        target_xyz,
        max_iterations=geometry_init_iterations,
        tolerance=min(tolerance, 1e-12),
        distance_quantile=geometry_init_quantile,
    )
    total_rotation = initial["rotation"]
    total_translation = initial["translation"]
    moved = apply_transform(source_xyz, total_rotation, total_translation)

    previous_error = None
    history: list[dict] = []
    for iteration in range(1, max_iterations + 1):
        geometry_scale = _annealed_scale(
            geometry_scale_start, geometry_scale_end, iteration, anneal_iterations
        )
        indices, geometry_distances, rgb_distances, joint_squared = joint_correspondences(
            moved,
            source_rgb,
            target_xyz,
            target_rgb,
            geometry_scale,
            rgb_scale,
            rgb_weight,
            local_candidates,
        )

        threshold = np.quantile(joint_squared, distance_quantile)
        keep = joint_squared <= threshold
        if keep.sum() < 3:
            raise RuntimeError("Too few correspondences left after joint trimming")

        # RGB Gaussian likelihood plus a Cauchy geometric robust kernel.
        color_weights = np.exp(
            -0.5 * rgb_weight * (rgb_distances[keep] / rgb_scale) ** 2
        )
        median_geometry = max(float(np.median(geometry_distances[keep])), 1e-12)
        geometry_weights = 1.0 / (
            1.0 + (geometry_distances[keep] / (2.5 * median_geometry)) ** 2
        )
        weights = np.maximum(color_weights * geometry_weights, 1e-12)

        rotation, translation = weighted_best_fit_transform(
            moved[keep], target_xyz[indices[keep]], weights
        )
        moved = apply_transform(moved, rotation, translation)
        total_rotation = rotation @ total_rotation
        total_translation = rotation @ total_translation + translation

        mean_joint_error = float(np.average(joint_squared[keep], weights=weights))
        history.append(
            {
                "iteration": iteration,
                "geometry_scale": geometry_scale,
                "mean_joint_error": mean_joint_error,
                "mean_geometry_distance": float(np.average(geometry_distances[keep], weights=weights)),
                "mean_rgb_distance": float(np.average(rgb_distances[keep], weights=weights)),
                "correspondences": int(keep.sum()),
            }
        )

        transform_step = np.linalg.norm(translation) + np.linalg.norm(rotation - np.eye(3))
        scale_finished = iteration >= anneal_iterations
        error_stable = previous_error is not None and abs(previous_error - mean_joint_error) < tolerance
        if scale_finished and error_stable and transform_step < np.sqrt(tolerance):
            break
        previous_error = mean_joint_error

    indices, geometry_distances, rgb_distances, joint_squared = joint_correspondences(
        moved,
        source_rgb,
        target_xyz,
        target_rgb,
        geometry_scale_end,
        rgb_scale,
        rgb_weight,
        local_candidates,
    )
    return {
        "rotation": total_rotation,
        "translation": total_translation,
        "aligned": moved,
        "indices": indices,
        "iterations": iteration,
        "mean_geometry_distance": float(geometry_distances.mean()),
        "rmse_geometry_distance": float(np.sqrt(np.mean(geometry_distances**2))),
        "mean_rgb_distance": float(rgb_distances.mean()),
        "rmse_rgb_distance": float(np.sqrt(np.mean(rgb_distances**2))),
        "mean_joint_error": float(joint_squared.mean()),
        "geometry_initialization": initial,
        "history": history,
    }


def write_history(path: str | Path, history: list[dict]) -> None:
    fieldnames = [
        "iteration",
        "geometry_scale",
        "mean_joint_error",
        "mean_geometry_distance",
        "mean_rgb_distance",
        "correspondences",
    ]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    print(f"saved RGB-ICP convergence history -> {path}")


def evaluate_alignment(
    aligned_xyz: np.ndarray,
    source_rgb: np.ndarray,
    target_xyz: np.ndarray,
    target_rgb: np.ndarray,
) -> dict:
    """Independent geometry metrics and RGB error of spatial NN pairs."""
    forward, indices = cKDTree(target_xyz).query(aligned_xyz, k=1)
    reverse = cKDTree(aligned_xyz).query(target_xyz, k=1)[0]
    rgb = np.linalg.norm(source_rgb - target_rgb[indices], axis=1)
    return {
        "mean_nn_m": float(forward.mean()),
        "rmse_nn_m": float(np.sqrt(np.mean(forward**2))),
        "reverse_mean_nn_m": float(reverse.mean()),
        "chamfer_m": float((forward.mean() + reverse.mean()) / 2),
        "spatial_nn_rgb_mean": float(rgb.mean()),
        "spatial_nn_rgb_rmse": float(np.sqrt(np.mean(rgb**2))),
    }


def write_comparison_metrics(
    path: str | Path, geometry: dict, colored: dict, colored_result: dict
) -> None:
    fields = [
        "method",
        "mean_nn_m",
        "rmse_nn_m",
        "reverse_mean_nn_m",
        "chamfer_m",
        "spatial_nn_rgb_mean",
        "spatial_nn_rgb_rmse",
        "optimization_pair_rgb_mean",
        "optimization_pair_rgb_rmse",
    ]
    rows = [
        {
            "method": "geometry_icp",
            **geometry,
            "optimization_pair_rgb_mean": geometry["spatial_nn_rgb_mean"],
            "optimization_pair_rgb_rmse": geometry["spatial_nn_rgb_rmse"],
        },
        {
            "method": "geometry_rgb_icp",
            **colored,
            "optimization_pair_rgb_mean": colored_result["mean_rgb_distance"],
            "optimization_pair_rgb_rmse": colored_result["rmse_rgb_distance"],
        },
    ]
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved geometry/RGB comparison metrics -> {path}")


def plot_comparison(path: str | Path, geometry: dict, colored: dict, colored_result: dict) -> None:
    """Create a report-ready, explicitly labelled ablation chart."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    chinese_font = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if chinese_font.exists():
        font_manager.fontManager.addfont(chinese_font)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=chinese_font
        ).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    labels = ["纯几何 ICP", "几何+RGB ICP"]
    colors = ["#4C78A8", "#F58518"]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8))

    axes[0].bar(labels, [geometry["mean_nn_m"] * 1000, colored["mean_nn_m"] * 1000], color=colors)
    axes[0].set_title("独立几何评价")
    axes[0].set_ylabel("平均最近邻距离（mm）")

    axes[1].bar(
        labels,
        [geometry["spatial_nn_rgb_mean"], colored["spatial_nn_rgb_mean"]],
        color=colors,
    )
    axes[1].set_title("空间最近邻的颜色一致性")
    axes[1].set_ylabel("RGB 欧氏距离")

    axes[2].bar(
        labels,
        [geometry["spatial_nn_rgb_mean"], colored_result["mean_rgb_distance"]],
        color=colors,
    )
    axes[2].set_title("算法所选对应点颜色残差")
    axes[2].set_ylabel("RGB 欧氏距离")

    for ax in axes:
        ax.grid(axis="y", linewidth=0.4, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", rotation=12)
        for bar in ax.patches:
            ax.annotate(
                f"{bar.get_height():.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved geometry/RGB comparison plot -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rigid ICP with joint XYZ and RGB constraints")
    parser.add_argument("--source", default="up_t.txt", help="moving X,Y,Z,R,G,B,... cloud")
    parser.add_argument("--target", default="down.txt", help="fixed X,Y,Z,R,G,B,... cloud")
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--distance-quantile", type=float, default=0.9)
    parser.add_argument("--rgb-weight", type=float, default=1.0, help="lambda_rgb in the joint objective")
    parser.add_argument("--rgb-scale", type=float, default=30.0, help="robust RGB vector scale sigma_c")
    parser.add_argument("--geometry-scale-start", type=float, default=0.008)
    parser.add_argument("--geometry-scale-end", type=float, default=0.003)
    parser.add_argument("--anneal-iterations", type=int, default=40)
    parser.add_argument("--local-candidates", type=int, default=30)
    parser.add_argument("--geometry-init-iterations", type=int, default=500)
    parser.add_argument("--geometry-init-quantile", type=float, default=1.0)
    parser.add_argument("--aligned-output", default="up_t_aligned_rgb.txt")
    parser.add_argument("--transform-output", default="icp_transform_rgb.txt")
    parser.add_argument("--output", default="icp_merged_rgb.csv")
    parser.add_argument("--history-output", default="icp_rgb_convergence.csv")
    parser.add_argument("--metrics-output", default="icp_rgb_metrics.csv")
    parser.add_argument("--comparison-plot-output", default="icp_rgb_comparison.png")
    parser.add_argument("--plot-output", default="icp_registration_rgb.png")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    source_xyz, source_rgb, source_full = load_cloud(args.source)
    target_xyz, target_rgb, target_full = load_cloud(args.target)
    result = rgb_icp(
        source_xyz,
        source_rgb,
        target_xyz,
        target_rgb,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        distance_quantile=args.distance_quantile,
        rgb_weight=args.rgb_weight,
        rgb_scale=args.rgb_scale,
        geometry_scale_start=args.geometry_scale_start,
        geometry_scale_end=args.geometry_scale_end,
        anneal_iterations=args.anneal_iterations,
        local_candidates=args.local_candidates,
        geometry_init_iterations=args.geometry_init_iterations,
        geometry_init_quantile=args.geometry_init_quantile,
    )

    print(f"source: {args.source} ({len(source_xyz)} points)")
    print(f"target: {args.target} ({len(target_xyz)} points)")
    print(f"iterations: {result['iterations']}")
    print("rotation:")
    print(np.array2string(result["rotation"], precision=8))
    print("translation:")
    print(np.array2string(result["translation"], precision=8))
    print(f"joint-correspondence geometry mean: {result['mean_geometry_distance']:.8f} m")
    print(f"joint-correspondence geometry RMSE: {result['rmse_geometry_distance']:.8f} m")
    print(f"joint-correspondence RGB mean: {result['mean_rgb_distance']:.6f}")
    print(f"joint-correspondence RGB RMSE: {result['rmse_rgb_distance']:.6f}")
    print(f"mean normalized joint error: {result['mean_joint_error']:.8f}")

    save_aligned(args.aligned_output, result["aligned"], source_full)
    save_transform(args.transform_output, result["rotation"], result["translation"])
    save_merged(args.output, result["aligned"], source_full[:, 3:], target_xyz, target_full[:, 3:])
    write_history(args.history_output, result["history"])
    geometry_aligned = apply_transform(
        source_xyz,
        result["geometry_initialization"]["rotation"],
        result["geometry_initialization"]["translation"],
    )
    geometry_metrics = evaluate_alignment(geometry_aligned, source_rgb, target_xyz, target_rgb)
    colored_metrics = evaluate_alignment(result["aligned"], source_rgb, target_xyz, target_rgb)
    write_comparison_metrics(args.metrics_output, geometry_metrics, colored_metrics, result)
    plot_comparison(args.comparison_plot_output, geometry_metrics, colored_metrics, result)
    if args.visualize or args.plot_output:
        visualize(source_xyz, result["aligned"], target_xyz, args.plot_output, args.visualize)


if __name__ == "__main__":
    main()
