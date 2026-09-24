#!/usr/bin/env python3
"""活动室左右扫描的 ICP 配准与 Poisson 三维重建实验。"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "pointcloud_output"
OUTPUT_DIR = ROOT / "activity_room_experiment"
SEED = 42

# 与 MATLAB 完整场景绘图范围一致；Y 不裁剪，以保留活动室横向结构。
X_LIMIT_MM = (0.0, 10_000.0)
Z_LIMIT_MM = (1_000.0, 5_000.0)
ICP_MAX_CORRESPONDENCE_MM = 200.0
ICP_MAX_ITERATIONS = 200
ICP_TOLERANCE_MM = 1e-6
TRIM_QUANTILE = 0.70

POISSON_DEPTH = 8
NORMAL_RADIUS_M = 0.30
NORMAL_MAX_NN = 50
SURFACE_SAMPLES = 20_000
COVERAGE_THRESHOLD_M = 0.10


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
        }
    )


def load_side(side: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(INPUT_DIR / f"{side}_pointcloud_all.npz") as data:
        xyz = data["xyz_mm"].astype(np.float64)
        rgb = data["rgb"].astype(np.float64)
        valid = data["valid"].astype(bool)
    mask = (
        valid
        & (xyz[:, 0] >= X_LIMIT_MM[0])
        & (xyz[:, 0] <= X_LIMIT_MM[1])
        & (xyz[:, 2] >= Z_LIMIT_MM[0])
        & (xyz[:, 2] <= Z_LIMIT_MM[1])
    )
    return xyz[mask], np.clip(rgb[mask], 0.0, 1.0)


def rigid_svd(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def overlap_icp(source: np.ndarray, target: np.ndarray):
    aligned = source.copy()
    accumulated = np.eye(4)
    tree = cKDTree(target)
    history: list[tuple[int, float, int]] = []
    previous = np.inf
    for iteration in range(1, ICP_MAX_ITERATIONS + 1):
        distances, indices = tree.query(aligned, k=1)
        inliers = distances <= ICP_MAX_CORRESPONDENCE_MM
        if inliers.sum() < 3:
            raise RuntimeError("ICP 有效对应点少于 3 个")
        increment = rigid_svd(aligned[inliers], target[indices[inliers]])
        aligned = apply_transform(aligned, increment)
        accumulated = increment @ accumulated
        new_distances = tree.query(aligned, k=1)[0]
        new_inliers = new_distances <= ICP_MAX_CORRESPONDENCE_MM
        mean_error = float(new_distances[new_inliers].mean())
        history.append((iteration, mean_error, int(new_inliers.sum())))
        if abs(previous - mean_error) < ICP_TOLERANCE_MM:
            break
        previous = mean_error
    return aligned, accumulated, history


def registration_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    forward = cKDTree(target).query(source, k=1)[0]
    backward = cKDTree(source).query(target, k=1)[0]

    def trimmed_mean(values: np.ndarray) -> float:
        cutoff = np.quantile(values, TRIM_QUANTILE)
        return float(values[values <= cutoff].mean())

    return {
        "forward_mean_mm": float(forward.mean()),
        "backward_mean_mm": float(backward.mean()),
        "symmetric_chamfer_mm": float(0.5 * (forward.mean() + backward.mean())),
        "trimmed_chamfer_70_mm": float(0.5 * (trimmed_mean(forward) + trimmed_mean(backward))),
        "symmetric_median_mm": float(0.5 * (np.median(forward) + np.median(backward))),
        "overlap_100mm": float(0.5 * ((forward <= 100).mean() + (backward <= 100).mean())),
        "overlap_250mm": float(0.5 * ((forward <= 250).mean() + (backward <= 250).mean())),
    }


def make_cloud(points_mm: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_mm / 1000.0))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def reconstruct(points_mm: np.ndarray, colors: np.ndarray) -> o3d.geometry.TriangleMesh:
    cloud = make_cloud(points_mm, colors)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=NORMAL_MAX_NN)
    )
    cloud.orient_normals_towards_camera_location(np.array([0.0, 0.0, 1.9]))
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=POISSON_DEPTH, scale=1.10, linear_fit=False, n_threads=1
    )
    bounds = cloud.get_axis_aligned_bounding_box()
    margin = 0.10
    crop = o3d.geometry.AxisAlignedBoundingBox(
        np.asarray(bounds.min_bound) - margin, np.asarray(bounds.max_bound) + margin
    )
    mesh = mesh.crop(crop)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def exact_point_to_mesh(points_m: np.ndarray, mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(points_m.astype(np.float32))
    return scene.compute_distance(query).numpy().astype(float)


def evaluate_mesh(mesh: o3d.geometry.TriangleMesh, own_input_mm: np.ndarray, reference_mm: np.ndarray):
    own_distance = exact_point_to_mesh(own_input_mm / 1000.0, mesh)
    reference_distance = exact_point_to_mesh(reference_mm / 1000.0, mesh)
    o3d.utility.random.seed(SEED)
    surface = mesh.sample_points_uniformly(number_of_points=SURFACE_SAMPLES)
    surface_points = np.asarray(surface.points)
    surface_colors = np.asarray(surface.colors)
    surface_to_reference = cKDTree(reference_mm / 1000.0).query(surface_points, k=1)[0]
    _, component_counts, _ = mesh.cluster_connected_triangles()
    component_counts = np.asarray(component_counts)
    significant = int(np.sum(component_counts >= max(1, int(len(mesh.triangles) * 0.01))))
    precision = float(np.mean(surface_to_reference <= COVERAGE_THRESHOLD_M))
    completeness = float(np.mean(reference_distance <= COVERAGE_THRESHOLD_M))
    f1 = 2 * precision * completeness / (precision + completeness) if precision + completeness else 0.0
    metrics = {
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "significant_components": significant,
        "input_to_mesh_rmse_mm": float(np.sqrt(np.mean(own_distance**2)) * 1000),
        "reference_to_mesh_mean_mm": float(reference_distance.mean() * 1000),
        "mesh_to_reference_mean_mm": float(surface_to_reference.mean() * 1000),
        "symmetric_chamfer_mm": float(0.5 * (reference_distance.mean() + surface_to_reference.mean()) * 1000),
        "surface_precision_100mm": precision,
        "reference_completeness_100mm": completeness,
        "f1_100mm": f1,
    }
    return metrics, surface_points * 1000.0, surface_colors, surface_to_reference * 1000.0


def equal_axes(ax, points: np.ndarray) -> None:
    mins, maxs = points.min(axis=0), points.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins)) / 2, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 0.55))


def plot_registration(left: np.ndarray, right: np.ndarray, aligned: np.ndarray, before, after, history):
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    for index, (moving, title) in enumerate(((right, "配准前"), (aligned, "重叠区 ICP 配准后")), 1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        ax.scatter(left[:, 0], left[:, 1], left[:, 2], s=2.5, c="#1f77b4", label="左侧扫描")
        ax.scatter(moving[:, 0], moving[:, 1], moving[:, 2], s=2.5, c="#ff7f0e", label="右侧扫描")
        ax.set_title(title)
        ax.set_xlabel("X / mm"); ax.set_ylabel("Y / mm"); ax.set_zlabel("Z / mm")
        ax.view_init(elev=22, azim=-55)
        equal_axes(ax, np.vstack((left, moving)))
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(OUTPUT_DIR / "registration_overview.png", bbox_inches="tight")
    plt.close(fig)

    labels = ["全局双向\nChamfer", "70%截尾\nChamfer", "双向中位数"]
    before_values = [before["symmetric_chamfer_mm"], before["trimmed_chamfer_70_mm"], before["symmetric_median_mm"]]
    after_values = [after["symmetric_chamfer_mm"], after["trimmed_chamfer_70_mm"], after["symmetric_median_mm"]]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    x = np.arange(3); width = 0.36
    axes[0].bar(x - width / 2, before_values, width, label="配准前")
    axes[0].bar(x + width / 2, after_values, width, label="配准后")
    axes[0].set_xticks(x, labels); axes[0].set_ylabel("距离 / mm"); axes[0].legend()
    overlap_before = np.array([before["overlap_100mm"], before["overlap_250mm"]]) * 100
    overlap_after = np.array([after["overlap_100mm"], after["overlap_250mm"]]) * 100
    x2 = np.arange(2)
    axes[1].bar(x2 - width / 2, overlap_before, width, label="配准前")
    axes[1].bar(x2 + width / 2, overlap_after, width, label="配准后")
    axes[1].set_xticks(x2, ["100 mm重叠率", "250 mm重叠率"]); axes[1].set_ylabel("重叠率 / %")
    axes[1].set_ylim(0, 100); axes[1].legend()
    fig.suptitle("活动室左右扫描配准指标")
    fig.savefig(OUTPUT_DIR / "registration_metrics.png", bbox_inches="tight")
    plt.close(fig)

    history_array = np.asarray(history)
    fig, ax = plt.subplots(figsize=(7.6, 4.4), constrained_layout=True)
    ax.plot(history_array[:, 0], history_array[:, 1], lw=2, color="#1f77b4")
    ax.set_xlabel("迭代次数"); ax.set_ylabel("阈值内平均对应距离 / mm")
    ax.grid(alpha=0.3); ax.set_title("重叠区 ICP 收敛曲线")
    fig.savefig(OUTPUT_DIR / "registration_convergence.png", bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction(results):
    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    for index, result in enumerate(results, 1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        colors = result[2] if len(result[2]) == len(result[1]) else np.tile([[0.3, 0.5, 0.7]], (len(result[1]), 1))
        ax.scatter(result[1][:, 0], result[1][:, 1], result[1][:, 2], c=np.clip(colors, 0, 1), s=1.0)
        ax.set_title(result[0])
        ax.set_xlabel("X / mm"); ax.set_ylabel("Y / mm"); ax.set_zlabel("Z / mm")
        ax.view_init(elev=22, azim=-55); equal_axes(ax, result[1])
    fig.savefig(OUTPUT_DIR / "reconstruction_comparison.png", bbox_inches="tight")
    plt.close(fig)

    metrics = [result[4] for result in results]
    names = ["直接融合", "ICP后融合"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    x = np.arange(2); width = 0.36
    axes[0].bar(x - width / 2, [m["symmetric_chamfer_mm"] for m in metrics], width, label="对称Chamfer")
    axes[0].bar(x + width / 2, [m["input_to_mesh_rmse_mm"] for m in metrics], width, label="输入拟合RMSE")
    axes[0].set_xticks(x, names); axes[0].set_ylabel("距离 / mm"); axes[0].legend(fontsize=8)
    axes[1].bar(x - width, [m["surface_precision_100mm"] * 100 for m in metrics], width, label="精确率")
    axes[1].bar(x, [m["reference_completeness_100mm"] * 100 for m in metrics], width, label="完整率")
    axes[1].bar(x + width, [m["f1_100mm"] * 100 for m in metrics], width, label="F1")
    axes[1].set_xticks(x, names); axes[1].set_ylabel("100 mm阈值指标 / %"); axes[1].set_ylim(0, 105); axes[1].legend(fontsize=8)
    axes[2].bar(names, [m["significant_components"] for m in metrics], color=["#7570b3", "#66a61e"])
    axes[2].set_ylabel("显著连通分量数")
    fig.suptitle("活动室 Poisson 三维重建质量指标")
    fig.savefig(OUTPUT_DIR / "reconstruction_metrics.png", bbox_inches="tight")
    plt.close(fig)

    after = results[1]
    vmax = float(np.quantile(after[3], 0.98))
    fig = plt.figure(figsize=(7.5, 5.5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plot = ax.scatter(after[1][:, 0], after[1][:, 1], after[1][:, 2], c=np.clip(after[3], 0, vmax), cmap="turbo", s=1.4, vmin=0, vmax=vmax)
    fig.colorbar(plot, ax=ax, shrink=0.72, label="表面到参考融合点云距离 / mm")
    ax.set_title("ICP 后融合重建表面误差分布")
    ax.set_xlabel("X / mm"); ax.set_ylabel("Y / mm"); ax.set_zlabel("Z / mm")
    ax.view_init(elev=22, azim=-55); equal_axes(ax, after[1])
    fig.savefig(OUTPUT_DIR / "reconstruction_error_map.png", bbox_inches="tight")
    plt.close(fig)


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    configure_plots()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    left, left_rgb = load_side("left")
    right, right_rgb = load_side("right")
    aligned_right, transform, history = overlap_icp(right, left)
    before = registration_metrics(right, left)
    after = registration_metrics(aligned_right, left)
    np.savetxt(OUTPUT_DIR / "right_to_left_transform.txt", transform, fmt="%.12f")
    np.savetxt(OUTPUT_DIR / "registration_convergence.csv", np.asarray(history), delimiter=",", header="iteration,inlier_mean_mm,inlier_count", comments="")
    write_dict_rows(
        OUTPUT_DIR / "registration_metrics.csv",
        [{"condition": "before", **before}, {"condition": "after", **after}],
    )
    plot_registration(left, right, aligned_right, before, after, history)

    before_points = np.vstack((left, right)); after_points = np.vstack((left, aligned_right))
    before_colors = np.vstack((left_rgb, right_rgb)); after_colors = before_colors.copy()
    reference = after_points
    results = []
    for title, points, colors, filename in (
        ("未配准直接融合", before_points, before_colors, "activity_room_mesh_before.ply"),
        ("重叠区ICP后融合", after_points, after_colors, "activity_room_mesh_after.ply"),
    ):
        mesh = reconstruct(points, colors)
        o3d.io.write_triangle_mesh(str(OUTPUT_DIR / filename), mesh, write_ascii=False)
        metrics, surface_points, surface_colors, errors = evaluate_mesh(mesh, points, reference)
        results.append((title, surface_points, surface_colors, errors, metrics))
    write_dict_rows(
        OUTPUT_DIR / "reconstruction_metrics.csv",
        [{"condition": result[0], **result[4]} for result in results],
    )
    plot_reconstruction(results)
    print(f"left={len(left)}, right={len(right)}, ICP iterations={len(history)}")
    print("registration before", before)
    print("registration after", after)
    print("reconstruction", [result[4] for result in results])


if __name__ == "__main__":
    main()
