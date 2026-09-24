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
SEED = 42
POISSON_DEPTH = 8
NORMAL_RADIUS = 0.012
NORMAL_MAX_NN = 40
COVERAGE_THRESHOLD = 0.005
SURFACE_SAMPLES = 20_000


def load_cloud(path: Path) -> o3d.geometry.PointCloud:
    data = np.loadtxt(path, delimiter=",")
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(data[:, :3])
    if data.shape[1] >= 6:
        cloud.colors = o3d.utility.Vector3dVector(
            np.clip(data[:, 3:6] / 255.0, 0.0, 1.0)
        )
    return cloud


def merge_clouds(first: o3d.geometry.PointCloud, second: o3d.geometry.PointCloud):
    return first + second


def reconstruct(cloud: o3d.geometry.PointCloud):
    working = o3d.geometry.PointCloud(cloud)
    working.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=NORMAL_RADIUS, max_nn=NORMAL_MAX_NN
        )
    )
    working.orient_normals_consistent_tangent_plane(30)
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        working, depth=POISSON_DEPTH, scale=1.10, linear_fit=False, n_threads=1
    )

    bounds = working.get_axis_aligned_bounding_box()
    span = np.asarray(bounds.get_extent())
    margin = max(float(np.max(span)) * 0.025, 0.003)
    crop_box = o3d.geometry.AxisAlignedBoundingBox(
        np.asarray(bounds.min_bound) - margin,
        np.asarray(bounds.max_bound) + margin,
    )
    mesh = mesh.crop(crop_box)

    densities = np.asarray(densities)
    # Crop changes vertex indexing, so density pruning is intentionally omitted here.
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    if mesh.has_vertex_colors():
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.clip(np.asarray(mesh.vertex_colors), 0.0, 1.0)
        )
    return mesh


def exact_point_to_mesh(points: np.ndarray, mesh: o3d.geometry.TriangleMesh):
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    query = o3d.core.Tensor(points.astype(np.float32))
    return scene.compute_distance(query).numpy().astype(float)


def sample_mesh(mesh: o3d.geometry.TriangleMesh, count: int):
    o3d.utility.random.seed(SEED)
    sampled = mesh.sample_points_uniformly(number_of_points=count)
    return np.asarray(sampled.points), np.asarray(sampled.colors)


def evaluate(
    label: str,
    mesh: o3d.geometry.TriangleMesh,
    own_input: np.ndarray,
    reference: np.ndarray,
):
    own_dist = exact_point_to_mesh(own_input, mesh)
    reference_dist = exact_point_to_mesh(reference, mesh)
    surface_points, surface_colors = sample_mesh(mesh, SURFACE_SAMPLES)
    surface_to_reference = cKDTree(reference).query(surface_points, k=1)[0]

    triangle_labels, triangle_counts, _ = mesh.cluster_connected_triangles()
    triangle_counts = np.asarray(triangle_counts, dtype=int)
    significant_components = int(
        np.sum(triangle_counts >= max(1, int(len(mesh.triangles) * 0.01)))
    )
    recall = float(np.mean(reference_dist <= COVERAGE_THRESHOLD))
    precision = float(np.mean(surface_to_reference <= COVERAGE_THRESHOLD))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "condition": label,
        "input_points": len(own_input),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "significant_components": significant_components,
        "input_to_mesh_mean": float(np.mean(own_dist)),
        "input_to_mesh_rmse": float(np.sqrt(np.mean(own_dist**2))),
        "reference_to_mesh_mean": float(np.mean(reference_dist)),
        "mesh_to_reference_mean": float(np.mean(surface_to_reference)),
        "symmetric_chamfer": float(
            0.5 * (np.mean(reference_dist) + np.mean(surface_to_reference))
        ),
        "surface_precision_5mm": precision,
        "reference_completeness_5mm": recall,
        "f1_5mm": f1,
    }
    return metrics, reference_dist, surface_points, surface_colors, surface_to_reference


def set_equal_3d(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    radius = max(float(np.max(maxs - mins)) / 2, 1e-5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def save_figures(results):
    plt.rcParams.update(
        {
            "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 160,
        }
    )
    before, after = results

    fig = plt.figure(figsize=(12, 5.2), constrained_layout=True)
    for index, result in enumerate(results, start=1):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        pts = result[2]
        colors = result[3]
        if len(colors) != len(pts):
            colors = np.tile(np.array([[0.22, 0.52, 0.78]]), (len(pts), 1))
        else:
            colors = np.clip(colors, 0.0, 1.0)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=1.2, alpha=0.9)
        ax.set_title(result[0]["condition"] + "：Poisson 重建表面采样")
        ax.set_xlabel("X / m")
        ax.set_ylabel("Y / m")
        ax.set_zlabel("Z / m")
        ax.view_init(elev=24, azim=-58)
        set_equal_3d(ax, pts)
    fig.savefig(ROOT / "reconstruction_figure1_mesh_comparison.png", bbox_inches="tight")
    fig.savefig(ROOT / "reconstruction_figure1_mesh_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    names = ["未配准重建", "ICP 后重建"]
    chamfer = [x[0]["symmetric_chamfer"] * 1000 for x in results]
    rmse = [x[0]["input_to_mesh_rmse"] * 1000 for x in results]
    precision = [x[0]["surface_precision_5mm"] * 100 for x in results]
    completeness = [x[0]["reference_completeness_5mm"] * 100 for x in results]
    f1 = [x[0]["f1_5mm"] * 100 for x in results]
    components = [x[0]["significant_components"] for x in results]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    x = np.arange(2)
    width = 0.36
    axes[0].bar(x - width / 2, chamfer, width, label="对称 Chamfer")
    axes[0].bar(x + width / 2, rmse, width, label="输入拟合 RMSE")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("误差 / mm（对数坐标）")
    axes[0].set_xticks(x, names)
    axes[0].legend(fontsize=8)
    axes[1].bar(x - width, precision, width, label="精确率")
    axes[1].bar(x, completeness, width, label="完整率")
    axes[1].bar(x + width, f1, width, label="F1")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("5 mm 阈值指标 / %")
    axes[1].set_xticks(x, names)
    axes[1].legend(fontsize=8)
    axes[2].bar(names, components, color=["#7570b3", "#66a61e"])
    axes[2].set_ylabel("显著连通分量数")
    axes[2].set_ylim(0, max(components) + 1)
    fig.suptitle("三维重建质量指标对比")
    fig.savefig(ROOT / "reconstruction_figure2_metrics.png", bbox_inches="tight")
    fig.savefig(ROOT / "reconstruction_figure2_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8), constrained_layout=True)
    for result, color in zip(results, ["#d95f02", "#1b9e77"]):
        values = np.sort(result[4] * 1000)
        cdf = np.arange(1, len(values) + 1) / len(values)
        ax.plot(values, cdf, lw=2, label=result[0]["condition"], color=color)
    ax.axvline(COVERAGE_THRESHOLD * 1000, ls="--", color="0.35", label="5 mm 阈值")
    ax.set_xscale("log")
    ax.set_xlabel("重建表面采样点到最终融合点云的距离 / mm（对数坐标）")
    ax.set_ylabel("累计比例")
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_title("重建表面误差累积分布")
    fig.savefig(ROOT / "reconstruction_figure3_error_distribution.png", bbox_inches="tight")
    fig.savefig(ROOT / "reconstruction_figure3_error_distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    pts = after[2]
    errors = after[4] * 1000
    vmax = float(np.quantile(errors, 0.98))
    fig = plt.figure(figsize=(7.4, 5.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plotted = ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2], c=np.clip(errors, 0, vmax),
        s=2.0, cmap="turbo", vmin=0, vmax=vmax
    )
    fig.colorbar(plotted, ax=ax, shrink=0.72, pad=0.08, label="表面采样点到融合点云距离 / mm")
    ax.set_title("ICP 后融合重建表面误差分布")
    ax.set_xlabel("X / m")
    ax.set_ylabel("Y / m")
    ax.set_zlabel("Z / m")
    ax.view_init(elev=24, azim=-58)
    set_equal_3d(ax, pts)
    fig.savefig(ROOT / "reconstruction_figure4_error_map.png", bbox_inches="tight")
    fig.savefig(ROOT / "reconstruction_figure4_error_map.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    moving_before = load_cloud(ROOT / "up_t.txt")
    fixed = load_cloud(ROOT / "down.txt")
    moving_after = load_cloud(ROOT / "up_t_aligned_refined.txt")
    before_cloud = merge_clouds(moving_before, fixed)
    after_cloud = merge_clouds(moving_after, fixed)
    reference = np.asarray(after_cloud.points)

    results = []
    for label, cloud, mesh_name in [
        ("未配准融合重建", before_cloud, "palette_reconstruction_before.ply"),
        ("ICP 后融合重建", after_cloud, "palette_reconstruction_after.ply"),
    ]:
        mesh = reconstruct(cloud)
        o3d.io.write_triangle_mesh(str(ROOT / mesh_name), mesh, write_ascii=False)
        evaluation = evaluate(label, mesh, np.asarray(cloud.points), reference)
        results.append(evaluation)

    fields = list(results[0][0].keys())
    with (ROOT / "reconstruction_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([result[0] for result in results])

    save_figures(results)
    for result in results:
        print(result[0])


if __name__ == "__main__":
    main()
