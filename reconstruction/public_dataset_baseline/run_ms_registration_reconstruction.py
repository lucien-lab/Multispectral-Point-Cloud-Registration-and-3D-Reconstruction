from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
LOG_PATH = ROOT / "experiment_log.md"


@dataclass
class RegistrationResult:
    method: str
    match_inlier_ratio: float
    ransac_inliers: int
    rre_deg: float
    rte: float
    rmse_after_icp: float
    fitness_after_icp: float
    transform: np.ndarray


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {stamp} {message}\n")


def spectral_angle(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = np.sum(a * b, axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return np.arccos(np.clip(num / den, -1.0, 1.0))


def make_transform() -> tuple[np.ndarray, np.ndarray]:
    rx, ry, rz = np.deg2rad([18.0, -11.0, 26.0])
    rot = o3d.geometry.get_rotation_matrix_from_xyz((rx, ry, rz))
    trans = np.array([0.045, -0.028, 0.036])
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T, np.linalg.inv(T)


def simulate_multispectral(points: np.ndarray) -> np.ndarray:
    """Deterministic controlled spectral attributes for ablation only.

    The bands are smooth functions and material-like partitions of geometry.
    They do not represent measured reflectance.
    """
    p = (points - points.min(axis=0)) / (np.ptp(points, axis=0) + 1e-12)
    material = (p[:, 0] > 0.52).astype(float)
    bands = np.stack(
        [
            0.25 + 0.55 * p[:, 0],
            0.35 + 0.45 * p[:, 1],
            0.30 + 0.50 * (1.0 - p[:, 2]),
            0.20 + 0.65 * material + 0.08 * np.sin(6.0 * p[:, 1]),
            0.40 + 0.35 * (p[:, 0] * p[:, 2]) + 0.15 * (1.0 - material),
        ],
        axis=1,
    )
    bands -= bands.min(axis=0)
    bands /= bands.max(axis=0) + 1e-12
    return bands.astype(np.float64)


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    delta = R_est @ R_gt.T
    val = (np.trace(delta) - 1.0) / 2.0
    return float(np.rad2deg(np.arccos(np.clip(val, -1.0, 1.0))))


def compute_fpfh(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.pipelines.registration.Feature:
    radius_normal = voxel * 2.5
    radius_feature = voxel * 5.0
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )


def rigid_from_correspondence(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    src_c = src - src_mean
    tgt_c = tgt - tgt_mean
    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = tgt_mean - R @ src_mean
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def ransac_transform(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    matches: np.ndarray,
    source_spec: np.ndarray,
    target_spec: np.ndarray,
    use_spectral_score: bool,
    seed: int = 7,
    iterations: int = 900,
    threshold: float = 0.006,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(matches)
    best_score = -1.0
    best_inliers = np.zeros(n, dtype=bool)
    best_T = np.eye(4)

    for _ in range(iterations):
        ids = rng.choice(n, size=3, replace=False)
        src = source_xyz[ids]
        tgt = target_xyz[matches[ids]]
        T = rigid_from_correspondence(src, tgt)
        aligned = apply_transform(source_xyz, T)
        d = np.linalg.norm(aligned - target_xyz[matches], axis=1)
        inliers = d < threshold
        if use_spectral_score:
            sam = spectral_angle(source_spec, target_spec[matches])
            score = float(inliers.sum()) - 0.25 * float(np.sum(sam[inliers]))
        else:
            score = float(inliers.sum())
        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_T = T

    if best_inliers.sum() >= 3:
        best_T = rigid_from_correspondence(
            source_xyz[best_inliers], target_xyz[matches[best_inliers]]
        )
    return best_T, best_inliers


def make_point_cloud(points: np.ndarray, colors: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors[:, :3], 0.0, 1.0))
    return pcd


def run_registration(
    method: str,
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    source_spec: np.ndarray,
    target_spec: np.ndarray,
    source_origin: np.ndarray,
    voxel: float,
    T_gt_source_to_target: np.ndarray,
) -> RegistrationResult:
    source_fpfh = np.asarray(compute_fpfh(source_pcd, voxel).data).T
    target_fpfh = np.asarray(compute_fpfh(target_pcd, voxel).data).T

    nbrs = NearestNeighbors(n_neighbors=8, algorithm="auto").fit(target_fpfh)
    fpfh_dist, fpfh_idx = nbrs.kneighbors(source_fpfh)

    if method == "geometry_fpfh":
        matches = fpfh_idx[:, 0]
    elif method == "spatial_spectral_fpfh":
        f_norm = fpfh_dist / (np.median(fpfh_dist[:, [0]]) + 1e-12)
        candidate_spec = target_spec[fpfh_idx]
        sam = spectral_angle(source_spec[:, None, :], candidate_spec)
        sam_norm = sam / (np.median(sam[:, [0]]) + 1e-12)
        combined = 0.70 * f_norm + 0.30 * sam_norm
        matches = fpfh_idx[np.arange(len(source_xyz)), np.argmin(combined, axis=1)]
    else:
        raise ValueError(method)

    match_inlier_ratio = float(np.mean(matches == source_origin))
    T_ransac, inliers = ransac_transform(
        source_xyz,
        target_xyz,
        matches,
        source_spec,
        target_spec,
        use_spectral_score=(method == "spatial_spectral_fpfh"),
    )

    icp = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance=voxel * 2.0,
        init=T_ransac,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60),
    )
    T_final = icp.transformation
    rre = rotation_error_deg(T_final[:3, :3], T_gt_source_to_target[:3, :3])
    rte = float(np.linalg.norm(T_final[:3, 3] - T_gt_source_to_target[:3, 3]))

    return RegistrationResult(
        method=method,
        match_inlier_ratio=match_inlier_ratio,
        ransac_inliers=int(inliers.sum()),
        rre_deg=rre,
        rte=rte,
        rmse_after_icp=float(icp.inlier_rmse),
        fitness_after_icp=float(icp.fitness),
        transform=T_final,
    )


def chamfer_approx(a: np.ndarray, b: np.ndarray) -> float:
    nn_a = NearestNeighbors(n_neighbors=1).fit(a)
    nn_b = NearestNeighbors(n_neighbors=1).fit(b)
    da, _ = nn_b.kneighbors(a)
    db, _ = nn_a.kneighbors(b)
    return float(np.mean(da) + np.mean(db))


def reconstruct_and_score(
    name: str,
    target_xyz: np.ndarray,
    source_xyz: np.ndarray,
    source_spec: np.ndarray,
    T: np.ndarray,
    reference_xyz: np.ndarray,
) -> dict[str, float | int | str]:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    aligned_source = apply_transform(source_xyz, T)
    fused = np.vstack([target_xyz, aligned_source])
    fused_colors = np.vstack([simulate_multispectral(target_xyz)[:, :3], source_spec[:, :3]])

    pcd = make_point_cloud(fused, fused_colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.003)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=40))
    pcd.orient_normals_consistent_tangent_plane(50)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8
    )
    densities = np.asarray(densities)
    keep = densities >= np.quantile(densities, 0.05)
    mesh = mesh.select_by_index(np.where(keep)[0])
    mesh.compute_vertex_normals()

    pcd_path = RESULT_DIR / f"{name}_fused_cloud.ply"
    mesh_path = RESULT_DIR / f"{name}_poisson_mesh.ply"
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)

    sampled = np.asarray(mesh.sample_points_uniformly(number_of_points=4000).points)
    cd = chamfer_approx(reference_xyz, sampled)
    return {
        "method": name,
        "fused_points": int(np.asarray(pcd.points).shape[0]),
        "mesh_vertices": int(np.asarray(mesh.vertices).shape[0]),
        "mesh_triangles": int(np.asarray(mesh.triangles).shape[0]),
        "approx_chamfer": cd,
        "cloud_file": str(pcd_path.relative_to(ROOT)),
        "mesh_file": str(mesh_path.relative_to(ROOT)),
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    log("开始运行空谱配准与三维重建受控验证脚本。")

    bunny_path = DATA_DIR / "stanford_bunny" / "BunnyMesh.ply"
    if not bunny_path.exists():
        raise FileNotFoundError(f"先运行 download_datasets.py: {bunny_path}")

    mesh = o3d.io.read_triangle_mesh(str(bunny_path))
    mesh.compute_vertex_normals()
    base = mesh.sample_points_poisson_disk(number_of_points=6000, init_factor=5)
    target_xyz = np.asarray(base.points)
    target_xyz = target_xyz - target_xyz.mean(axis=0)
    target_spec = simulate_multispectral(target_xyz)

    rng = np.random.default_rng(42)
    partial = np.where(target_xyz[:, 0] > np.quantile(target_xyz[:, 0], 0.18))[0]
    partial = rng.choice(partial, size=3600, replace=False)
    T_source_from_target, T_gt_source_to_target = make_transform()
    source_xyz_clean = apply_transform(target_xyz[partial], T_source_from_target)
    source_xyz = source_xyz_clean + rng.normal(0.0, 0.0007, source_xyz_clean.shape)
    source_spec = np.clip(
        target_spec[partial] + rng.normal(0.0, 0.018, target_spec[partial].shape),
        0.0,
        1.0,
    )

    target_pcd = make_point_cloud(target_xyz, target_spec[:, :3])
    source_pcd = make_point_cloud(source_xyz, source_spec[:, :3])
    voxel = 0.006

    results = [
        run_registration(
            "geometry_fpfh",
            source_pcd,
            target_pcd,
            source_xyz,
            target_xyz,
            source_spec,
            target_spec,
            partial,
            voxel,
            T_gt_source_to_target,
        ),
        run_registration(
            "spatial_spectral_fpfh",
            source_pcd,
            target_pcd,
            source_xyz,
            target_xyz,
            source_spec,
            target_spec,
            partial,
            voxel,
            T_gt_source_to_target,
        ),
    ]

    reg_csv = RESULT_DIR / "registration_metrics.csv"
    with reg_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "match_inlier_ratio",
                "ransac_inliers",
                "rre_deg",
                "rte",
                "rmse_after_icp",
                "fitness_after_icp",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "method": r.method,
                    "match_inlier_ratio": f"{r.match_inlier_ratio:.6f}",
                    "ransac_inliers": r.ransac_inliers,
                    "rre_deg": f"{r.rre_deg:.6f}",
                    "rte": f"{r.rte:.6f}",
                    "rmse_after_icp": f"{r.rmse_after_icp:.6f}",
                    "fitness_after_icp": f"{r.fitness_after_icp:.6f}",
                }
            )

    np.savez(
        RESULT_DIR / "controlled_multispectral_arrays.npz",
        target_xyz=target_xyz,
        target_spectral=target_spec,
        source_xyz=source_xyz,
        source_spectral=source_spec,
        source_origin_index=partial,
        T_gt_source_to_target=T_gt_source_to_target,
        geometry_transform=results[0].transform,
        spatial_spectral_transform=results[1].transform,
    )

    recon_rows = []
    reference = np.asarray(mesh.sample_points_uniformly(number_of_points=5000).points)
    reference = reference - reference.mean(axis=0)
    for r in results:
        recon_rows.append(
            reconstruct_and_score(
                r.method,
                target_xyz,
                source_xyz,
                source_spec,
                r.transform,
                reference,
            )
        )

    recon_csv = RESULT_DIR / "reconstruction_metrics.csv"
    with recon_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(recon_rows[0].keys()))
        writer.writeheader()
        writer.writerows(recon_rows)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data": "Stanford Bunny geometry with controlled synthetic multispectral attributes",
        "warning": "Synthetic spectral attributes are used only for controlled ablation validation.",
        "registration": [r.__dict__ | {"transform": r.transform.tolist()} for r in results],
        "reconstruction": recon_rows,
    }
    summary_path = RESULT_DIR / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"配准指标写入 {reg_csv.relative_to(ROOT)}。")
    log(f"重建指标写入 {recon_csv.relative_to(ROOT)}。")
    log("受控验证完成：结果只用于方法可行性检查，不替代真实多光谱 LiDAR 实验。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
