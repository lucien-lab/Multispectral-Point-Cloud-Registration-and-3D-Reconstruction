#!/usr/bin/env python3
"""发财树 0~300° 六视角点云配准 v2 —— 只利用点间相对几何。

原则（用户指出）：不能利用各帧的绝对坐标（每帧是独立扫描仪坐标系），
只能利用点与点的相对位置关系做配准。

流程（每帧 → 0° 基准帧）：
  1. 下采样 + 法线估计 + FPFH 特征
  2. 全局配准：FPFH 特征匹配 + RANSAC（无任何绝对坐标/角度先验）
  3. 精配准：point-to-plane ICP（初值 = RANSAC 解）
  4. sanity check：打印解的旋转轴/角度（若 cripped 各帧确为"同一树不同
     视角"，解应≈绕竖直轴旋转名义角度差；若有解≈单位阵且 fitness 高，
     则说明数据源已被预对齐——两种情况都会如实报告）
  5. 合并去重、保存、可视化

用法：python register_v2.py [--bench]   # --bench 额外对比 名义角度粗配准
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent
CRIPPED = SCRIPT_DIR / "cripped"
OUT = SCRIPT_DIR / "registered_v2"

VIEWS: list[tuple[str, float]] = [
    ("20260904三维重建发财树0度.txt", 0.0),
    ("20260904三维建模发财树60度.txt", 60.0),
    ("20260904三维建模发财树120度.txt", 120.0),
    ("20260904三维建模发财树180度.txt", 180.0),
    ("20260904三维建模发财树240度.txt", 240.0),
    ("20260904三维建模发财树300度.txt", 300.0),
]
RANSAC_MAX_DIST = 0.01          # 全局配准对应距离阈值
ICP_MAX_DIST = 0.006            # ICP 对应距离阈值


def load_pcd(path: Path, voxel: float = 0.003) -> o3d.geometry.PointCloud:
    a = np.loadtxt(path, delimiter=",")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(a[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.clip(a[:, 3:6], 0.0, 1.0))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    return pcd


def fpfh(pcd: o3d.geometry.PointCloud, radius: float = 0.02) -> o3d.pipelines.registration.Feature:
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))


def transform_analysis(T: np.ndarray) -> tuple[float, np.ndarray]:
    """返回 (旋转角度°, 旋转轴)"""
    R = T[:3, :3]
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if angle < 1e-6:
        return 0.0, np.array([0, 0, 1.0])
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return angle, axis / np.linalg.norm(axis)


def global_registration(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud,
                        f_src, f_tgt) -> np.ndarray:
    """FPFH 特征匹配 + RANSAC 全局配准（纯相对几何，无先验）"""
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src, tgt, f_src, f_tgt, mutual_filter=True,
        max_correspondence_distance=RANSAC_MAX_DIST,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(RANSAC_MAX_DIST),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500))
    return result.transformation


def refine_icp(src: o3d.geometry.PointCloud, tgt: o3d.geometry.PointCloud,
               T0: np.ndarray, max_dist: float = ICP_MAX_DIST) -> o3d.pipelines.registration.RegistrationResult:
    return o3d.pipelines.registration.registration_icp(
        src, tgt, max_dist, T0,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))


def save_txt(pcd: o3d.geometry.PointCloud, path: Path) -> None:
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    np.savetxt(path, np.hstack([pts, cols, np.zeros((len(pts), 2))]),
               delimiter=",", fmt="%.6f")


def render(pcd, path, views, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    n = len(views)
    fig = plt.figure(figsize=(5 * n, 5))
    fig.suptitle(title)
    for i, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c=np.clip(cols * 2, 0, 1))
        ax.view_init(el, az)
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  [渲染] {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true",
                    help="额外用名义角度粗配准(绕质心) + ICP 做对照")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    base_name, _ = VIEWS[0]
    base = load_pcd(CRIPPED / base_name)
    f_base = fpfh(base)
    print(f"[加载] 基准 {base_name} ({len(base.points)}点) + {len(VIEWS)-1} 帧")

    merged_geo = o3d.geometry.PointCloud()
    merged_geo += base
    print("\n[配准] 每帧 → 0° 基准帧（FPFH+RANSAC 全局配准 → ICP 精化）")
    for name, deg in VIEWS[1:]:
        src = load_pcd(CRIPPED / name)
        f_src = fpfh(src)

        T_glob = global_registration(src, base, f_src, f_base)
        ang, axis = transform_analysis(T_glob)
        reg0 = o3d.pipelines.registration.evaluate_registration(src, base, 0.02, T_glob)

        T_icp = refine_icp(src, base, T_glob).transformation
        ang2, axis2 = transform_analysis(T_icp)
        reg1 = o3d.pipelines.registration.evaluate_registration(src, base, 0.02, T_icp)

        print(f"\n  {name}")
        print(f"    RANSAC: rot={ang:.1f}° 轴=({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f}) "
              f"shift={np.linalg.norm(T_glob[:3,3])*1000:.1f}mm fitness={reg0.fitness:.3f} rmse={reg0.inlier_rmse*1000:.1f}mm")
        print(f"    ICP   : rot={ang2:.1f}° 轴=({axis2[0]:+.2f},{axis2[1]:+.2f},{axis2[2]:+.2f}) "
              f"shift={np.linalg.norm(T_icp[:3,3])*1000:.1f}mm fitness={reg1.fitness:.3f} rmse={reg1.inlier_rmse*1000:.1f}mm")
        print(f"    标称角度差={deg:.0f}°（若解≈绕竖直轴转该角 → cripped 为未对齐独立视角；若≈0 → 已预对齐）")

        t = o3d.geometry.PointCloud(src)
        t.transform(T_icp)
        save_txt(t, OUT / f"aligned_{name}")
        merged_geo += t

    merged_geo = merged_geo.voxel_down_sample(0.003)
    save_txt(merged_geo, OUT / "combined.txt")
    o3d.io.write_point_cloud(str(OUT / "combined.pcd"), merged_geo)
    print(f"\n[保存] combined.txt/.pcd ({len(merged_geo.points)}点), aligned_*.txt")
    render(merged_geo, OUT / "combined.png",
           [(22, -60), (22, 30), (70, 30), (0, 0)], "配准合并（FPFH+RANSAC+ICP，纯相对几何）")

    if args.bench:
        print("\n[对照] 名义角度粗配准（绕每帧质心 ±角度）+ ICP —— 仅作 sanity check")
        base_c = np.asarray(base.points).mean(axis=0)
        for name, deg in VIEWS[1:]:
            src = load_pcd(CRIPPED / name)
            src_c = np.asarray(src.points).mean(axis=0)
            best = None
            for sign in (+1, -1):
                r = np.deg2rad(sign * deg)
                c, s = np.cos(r), np.sin(r)
                T = np.eye(4)
                T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
                T[:3, 3] = src_c - T[:3, :3] @ src_c
                reg = o3d.pipelines.registration.evaluate_registration(src, base, 0.02, T)
                if best is None or reg.fitness > best[1]:
                    best = (T, reg.fitness)
            T_icp = refine_icp(src, base, best[0]).transformation
            r = o3d.pipelines.registration.evaluate_registration(src, base, 0.02, T_icp)
            ang, axis = transform_analysis(T_icp)
            print(f"  {name}: 初始 fitness={best[1]:.3f} → ICP后 fitness={r.fitness:.3f} "
                  f"rot={ang:.1f}° shift={np.linalg.norm(T_icp[:3,3])*1000:.1f}mm")
    print("\n[完成] 输出：", OUT)


if __name__ == "__main__":
    main()