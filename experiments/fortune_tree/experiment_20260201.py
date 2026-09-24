#!/usr/bin/env python3
"""发财树 0~300° 六视角配准实验（2026-09-05 重新实验，数据源=data_convert/20260201 完整数据）。

原则（按用户要求）：只利用点间相对几何进行配准，不用绝对坐标先验。

方法：
  1. 每帧加载（8列：id,x,y,z,R,G,B,c），体素下采样
  2. 相对几何配准（两套对照）：
     A) FPFH 特征匹配 + RANSAC 全局配准 → point-to-plane ICP
     B) 恒等变换（同坐标系）→ point-to-plane ICP 微调
     比较两套终态的 fitness/rmse 与几何合理性（旋转/平移量）
  3. 合并去重保存、渲染静态多视角 PNG + ASCII 密度图校验

输出：experiment_20260201/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR / "experiment_20260201"

VIEWS: list[tuple[str, float, str]] = [
    ("data_convert/20260201三维重建发财树0度.txt", 0.0, "重建"),
    ("data_convert/20260201三维建模发财树60度.txt", 60.0, "建模"),
    ("data_convert/20260201三维建模发财树120度.txt", 120.0, "建模"),
    ("data_convert/20260201三维建模发财树180度.txt", 180.0, "建模"),
    ("data_convert/20260201三维建模发财树240度.txt", 240.0, "建模"),
    ("data_convert/20260201三维建模发财树300度.txt", 300.0, "建模"),
]


def load_pcd(path: Path, voxel: float = 0.008) -> o3d.geometry.PointCloud:
    a = np.loadtxt(path, delimiter=",")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(a[:, 1:4])       # 列0是id
    pcd.colors = o3d.utility.Vector3dVector(np.clip(a[:, 4:7], 0.0, 1.0))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    return pcd


def fpfh(pcd):
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=100))


def t_info(T: np.ndarray) -> str:
    R = T[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return f"rot={ang:.2f}° shift={np.linalg.norm(T[:3, 3]) * 1000:.1f}mm"


def save_txt(pcd, path: Path) -> None:
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    np.savetxt(path, np.hstack([pts, cols, np.zeros((len(pts), 2))]),
               delimiter=",", fmt="%.6f")


def render(pcd, path, views, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    n = len(views)
    fig = plt.figure(figsize=(5 * n, 6))
    fig.suptitle(title, fontsize=12)
    for i, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c=np.clip(cols * 2, 0, 1))
        ax.view_init(el, az)
        ax.set_title(f"elev={el} azim={az}", fontsize=9)
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  [渲染] {path}")


def ascii_density(pcd, zaxis=2, nx=64, ny=30, title=""):
    from scipy.spatial import cKDTree
    pts = np.asarray(pcd.points)
    # 用高度轴作 y 向，径向 x / y 作水平
    h = pts[:, zaxis]
    r = np.hypot(pts[:, 0], pts[:, 1])
    grid = np.zeros((ny, nx))
    xi = ((r - r.min()) / (r.max() - r.min()) * (nx - 1)).astype(int)
    yi = ((h - h.min()) / (h.max() - h.min()) * (ny - 1)).astype(int)
    np.add.at(grid, (yi, xi), 1)
    vmax = max(grid.max(), 1)
    print(f"--- {title} (径向 r={r.min():.3f}~{r.max():.3f}, 高度 h={h.min():.3f}~{h.max():.3f}) ---")
    for row in grid:
        print("".join(" .:+*#@"[min(int(v / vmax * 6), 6)] if v else " " for v in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.008)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    base_file, _, _ = VIEWS[0]
    base = load_pcd(SCRIPT_DIR / base_file, args.voxel)
    f_base = fpfh(base)
    print(f"[加载] 基准 {Path(base_file).name} ({len(base.points)}点) + {len(VIEWS)-1} 帧, voxel={args.voxel}")

    merged_A = o3d.geometry.PointCloud()  # A: FPFH+RANSAC+ICP
    merged_B = o3d.geometry.PointCloud()  # B: 恒等初值+ICP 微调
    merged_A += base
    merged_B += base

    print("\n[配准] 每帧 → 0° 基准帧（对照两套方法）")
    for file, deg, tag in VIEWS[1:]:
        src = load_pcd(SCRIPT_DIR / file, args.voxel)
        f_src = fpfh(src)

        # A: 特征全局配准
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src, base, f_src, f_base, mutual_filter=True,
            max_correspondence_distance=0.02,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.02),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(400000, 500))
        icp_A = o3d.pipelines.registration.registration_icp(
            src, base, 0.015, ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        TA = icp_A.transformation
        rA = o3d.pipelines.registration.evaluate_registration(src, base, 0.02, TA)

        # B: 恒等初值 + ICP 微调
        icp_B = o3d.pipelines.registration.registration_icp(
            src, base, 0.015, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        TB = icp_B.transformation
        rB = o3d.pipelines.registration.evaluate_registration(src, base, 0.02, TB)

        print(f"\n  {Path(file).name} ({tag}, 名义 {deg:.0f}°)")
        print(f"    A) RANSAC+ICP: {t_info(TA)}  fitness={rA.fitness:.3f} rmse={rA.inlier_rmse*1000:.1f}mm")
        print(f"    B) 恒等+ICP  : {t_info(TB)}  fitness={rB.fitness:.3f} rmse={rB.inlier_rmse*1000:.1f}mm")
        print(f"    (初值 fitness: RANSAC {rA.fitness - icp_A.fitness:.3f}→.. / 恒等 {o3d.pipelines.registration.evaluate_registration(src, base, 0.02, np.eye(4)).fitness:.3f})")

        pa, pb = o3d.geometry.PointCloud(src), o3d.geometry.PointCloud(src)
        pa.transform(TA)
        pb.transform(TB)
        save_txt(pa, OUT / f"alignedA_{Path(file).stem}.txt")
        save_txt(pb, OUT / f"alignedB_{Path(file).stem}.txt")
        merged_A += pa
        merged_B += pb

    merged_A = merged_A.voxel_down_sample(args.voxel)
    merged_B = merged_B.voxel_down_sample(args.voxel)
    save_txt(merged_A, OUT / "combined_A.txt")
    save_txt(merged_B, OUT / "combined_B.txt")
    o3d.io.write_point_cloud(str(OUT / "combined_A.pcd"), merged_A)
    o3d.io.write_point_cloud(str(OUT / "combined_B.pcd"), merged_B)

    print(f"\n[保存] combined_A/B.txt/.pcd, alignedA/B_*.txt → {OUT}")
    print(f"  A 合并 {len(merged_A.points)} 点, B 合并 {len(merged_B.points)} 点")

    render(merged_B, OUT / "combined_B.png",
           [(22, -60), (22, 30), (70, 30), (0, 0)],
           "配准合并（恒等初值+ICP微调，相对几何）")
    render(merged_A, OUT / "combined_A.png",
           [(22, -60), (22, 30), (70, 30), (0, 0)],
           "配准合并（FPFH+RANSAC+ICP）")
    for which, m in (("A", merged_A), ("B", merged_B)):
        pts = np.asarray(m.points)
        print(f"\n[合并{which}] 点数={len(pts)}  "
              f"x[{pts[:,0].min():+.3f},{pts[:,0].max():+.3f}] "
              f"y[{pts[:,1].min():+.3f},{pts[:,1].max():+.3f}] "
              f"z[{pts[:,2].min():+.3f},{pts[:,2].max():+.3f}]")
        ascii_density(m, zaxis=2, title=f"合并{which} 侧视(径向-高度)")

    print("\n[完成] 输出目录:", OUT)


if __name__ == "__main__":
    main()