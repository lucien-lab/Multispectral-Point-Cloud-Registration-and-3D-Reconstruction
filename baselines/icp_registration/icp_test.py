import argparse
import os

import numpy as np
from scipy.spatial import cKDTree


def load_points(path):
    data = np.loadtxt(path, delimiter=",")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError(f"{path} does not look like a point-cloud text file")
    return data[:, :3], data


def best_fit_transform(source, target):
    src_centroid = source.mean(axis=0)
    tgt_centroid = target.mean(axis=0)
    src_centered = source - src_centroid
    tgt_centered = target - tgt_centroid

    h = src_centered.T @ tgt_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    t = tgt_centroid - r @ src_centroid
    return r, t


def icp(source, target, max_iterations=80, tolerance=1e-8, distance_quantile=0.7):
    moved = source.copy()
    total_r = np.eye(3)
    total_t = np.zeros(3)
    tree = cKDTree(target)
    previous_error = None
    error_history = []

    for iteration in range(1, max_iterations + 1):
        distances, indices = tree.query(moved, k=1)

        # Trim large-distance pairs because these two clouds only partially overlap.
        keep_distance = np.quantile(distances, distance_quantile)
        keep = distances <= keep_distance
        if keep.sum() < 3:
            raise RuntimeError("Too few correspondences left after trimming")

        r, t = best_fit_transform(moved[keep], target[indices[keep]])
        moved = (r @ moved.T).T + t
        total_r = r @ total_r
        total_t = r @ total_t + t

        mean_error = float(distances[keep].mean())
        error_history.append(mean_error)
        if previous_error is not None and abs(previous_error - mean_error) < tolerance:
            break
        previous_error = mean_error

    distances, _ = tree.query(moved, k=1)
    return {
        "rotation": total_r,
        "translation": total_t,
        "iterations": iteration,
        "mean_distance": float(distances.mean()),
        "median_distance": float(np.median(distances)),
        "trimmed_mean_distance": float(np.mean(distances[distances <= np.quantile(distances, distance_quantile)])),
        "error_history": error_history,
    }


def apply_transform(xyz, r, t):
    """Apply rigid transform to N×3 array."""
    return (r @ xyz.T).T + t


def save_aligned(output_path, aligned_xyz, source_full):
    """Save transformed XYZ while preserving all source point attributes."""
    aligned = source_full.copy()
    aligned[:, :3] = aligned_xyz
    formats = ["%.8f", "%.8f", "%.8f"] + ["%.6f"] * (aligned.shape[1] - 3)
    np.savetxt(output_path, aligned, delimiter=",", fmt=formats)
    print(f"saved aligned source ({len(aligned)} points) → {output_path}")


def save_transform(output_path, r, t):
    """Save the source-to-target transform as a 4×4 homogeneous matrix."""
    transform = np.eye(4)
    transform[:3, :3] = r
    transform[:3, 3] = t
    np.savetxt(output_path, transform, fmt="%.10f")
    print(f"saved homogeneous transform → {output_path}")


def save_merged(output_path, source_xyz, source_extra, target_xyz, target_extra):
    """Write merged point cloud to CSV: x,y,z,R,G,B,source_flag"""
    source_flag = np.ones((len(source_xyz), 1))
    target_flag = np.zeros((len(target_xyz), 1))

    source_cols = source_extra[:, :3] if source_extra.shape[1] >= 3 else source_extra
    target_cols = target_extra[:, :3] if target_extra.shape[1] >= 3 else target_extra

    merged = np.hstack([
        np.vstack([source_xyz, target_xyz]),
        np.vstack([source_cols, target_cols]),
        np.vstack([source_flag, target_flag]),
    ])

    header = "x,y,z,R,G,B,source_flag"
    np.savetxt(output_path, merged, delimiter=",", fmt="%.8f,%.8f,%.8f,%d,%d,%d,%d", header=header, comments="")
    print(f"\nsaved merged cloud ({len(merged)} points) → {output_path}")


def _equal_3d_axes(ax, points):
    """Use equal limits so before/after geometry is visually comparable."""
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = (lower + upper) / 2
    radius = max(np.ptp(points, axis=0).max() / 2, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def visualize(source_orig, source_aligned, target_xyz, output_path=None, show=False):
    """Save or show before/after registration with clouds in distinct colors."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 6))
    all_points = np.vstack([source_orig, source_aligned, target_xyz])

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(source_orig[:, 0], source_orig[:, 1], source_orig[:, 2],
                c="#e45756", s=5, alpha=0.75, label="up_t (moving)")
    ax1.scatter(target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2],
                c="#4c78a8", s=5, alpha=0.75, label="down (fixed)")
    ax1.set_title("Before ICP registration")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(loc="upper right")
    _equal_3d_axes(ax1, all_points)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(source_aligned[:, 0], source_aligned[:, 1], source_aligned[:, 2],
                c="#e45756", s=5, alpha=0.75, label="aligned up_t")
    ax2.scatter(target_xyz[:, 0], target_xyz[:, 1], target_xyz[:, 2],
                c="#4c78a8", s=5, alpha=0.75, label="down (fixed)")
    ax2.set_title("After ICP registration")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.legend(loc="upper right")
    _equal_3d_axes(ax2, all_points)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"saved before/after visualization → {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run point-to-point ICP on two comma-separated point-cloud txt files.")
    parser.add_argument("--source", default="up.txt", help="moving cloud, transformed by ICP")
    parser.add_argument("--target", default="down.txt", help="fixed cloud")
    parser.add_argument("--max-iterations", type=int, default=80)
    parser.add_argument("--tolerance", type=float, default=1e-8, help="stop when consecutive trimmed mean errors differ by less than this value")
    parser.add_argument("--distance-quantile", type=float, default=0.7, help="fraction of closest pairs kept each iteration")
    parser.add_argument("--visualize", action="store_true", help="show before/after 3D plots")
    parser.add_argument("--plot-output", default=None, help="save before/after visualization image")
    parser.add_argument("--aligned-output", default=None, help="save transformed source point cloud")
    parser.add_argument("--transform-output", default=None, help="save 4x4 homogeneous transform matrix")
    parser.add_argument("--output", default=None, help="save merged point cloud to CSV")
    args = parser.parse_args()

    source_xyz, source_full = load_points(args.source)
    target_xyz, target_full = load_points(args.target)

    result = icp(
        source_xyz,
        target_xyz,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        distance_quantile=args.distance_quantile,
    )

    source_aligned = apply_transform(source_xyz, result["rotation"], result["translation"])

    print(f"source: {args.source} ({len(source_xyz)} points)")
    print(f"target: {args.target} ({len(target_xyz)} points)")
    print(f"iterations: {result['iterations']}")
    print("rotation:")
    print(np.array2string(result["rotation"], precision=8, suppress_small=False))
    print("translation:")
    print(np.array2string(result["translation"], precision=8, suppress_small=False))
    print(f"mean nearest-neighbor distance: {result['mean_distance']:.8f}")
    print(f"median nearest-neighbor distance: {result['median_distance']:.8f}")
    print(f"trimmed mean distance: {result['trimmed_mean_distance']:.8f}")

    if args.output:
        save_merged(args.output, source_aligned, source_full[:, 3:], target_xyz, target_full[:, 3:])

    if args.aligned_output:
        save_aligned(args.aligned_output, source_aligned, source_full)

    if args.transform_output:
        save_transform(args.transform_output, result["rotation"], result["translation"])

    if args.visualize or args.plot_output:
        visualize(source_xyz, source_aligned, target_xyz, args.plot_output, args.visualize)


if __name__ == "__main__":
    main()
