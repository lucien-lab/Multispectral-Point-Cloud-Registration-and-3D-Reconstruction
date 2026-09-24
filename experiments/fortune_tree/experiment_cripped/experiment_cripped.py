#!/usr/bin/env python3
"""cripped 六视角配准实验（重新实验 v3）—— 标准相对几何配准流程。

流程（每帧 → 0° 基准帧，只用相对几何）：
  1. 下采样 + 法线 + FPFH
  2. 全局配准：RANSAC（多组参数随机采样多次，取最优 fitness）
  3. 精配准：point-to-plane ICP（多阈值多轮，取最优）
  4. 评估+保存+可视化（PNG 多视角 + GIF 动画 + ASCII 校验）

输出：experiment_cripped/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent
CRIPPED = SCRIPT_DIR / "cripped"
OUT = SCRIPT_DIR / "experiment_cripped"

VIEWS: list[tuple[str, float]] = [
    ("20260904三维重建发财树0度.txt", 0.0),
    ("20260904三维建模发财树60度.txt", 60.0),
    ("20260904三维建模发财树120度.txt", 120.0),
    ("20260904三维建模发财树180度.txt", 180.0),
    ("20260904三维建模发财树240度.txt", 240.0),
    ("20260904三维建模发财树300度.txt", 300.0),
]


def load_pcd(path: Path, voxel: float = 0.003) -> o3d.geometry.PointCloud:
    a = np.loadtxt(path, delimiter=",")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(a[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.clip(a[:, 3:6], 0.0, 1.0))
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    return pcd


def fpfh(pcd) -> o3d.pipelines.registration.Feature:
    return o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=100))


def global_reg(src, tgt, f_src, f_tgt, voxel: float) -> np.ndarray:
    """RANSAC 全局配准，多参数多尝试取最优"""
    best_T, best_f = np.eye(4), -1.0
    for max_d in (0.01, 0.02, 0.03):
        for ransac_n in (3, 4):
            for iters in (200000, 400000):
                res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    src, tgt, f_src, f_tgt, mutual_filter=True,
                    max_correspondence_distance=max_d,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    ransac_n=ransac_n,
                    checkers=[
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_d),
                    ],
                    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(iters, 500))
                f = o3d.pipelines.registration.evaluate_registration(
                    src, tgt, 0.02, res.transformation).fitness
                if f > best_f:
                    best_f = f
                    best_T = res.transformation
    return best_T, best_f


def refine_icp(src, tgt, T0) -> o3d.pipelines.registration.RegistrationResult:
    """多阈值 ICP，取最优 fitness"""
    best = None
    for max_d in (0.004, 0.006, 0.008, 0.012):
        r = o3d.pipelines.registration.registration_icp(
            src, tgt, max_d, T0,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        if best is None or r.fitness > best.fitness:
            best = r
    return best


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


def render_gif(pcd, path: Path, n_frames: int = 36) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5, c=np.clip(cols * 2, 0, 1))
    def upd(f):
        ax.view_init(elev=22, azim=f * 360 / n_frames)
        return (ax,)
    anim = FuncAnimation(fig, upd, frames=n_frames, interval=90)
    anim.save(path, writer="pillow", dpi=90)
    plt.close(fig)
    print(f"  [渲染] {path}")


def ascii_density(pcd, nx=64, ny=30):
    pts = np.asarray(pcd.points)
    h, r = pts[:, 2], np.hypot(pts[:, 0], pts[:, 1])
    grid = np.zeros((ny, nx))
    xi = ((r - r.min()) / (r.max() - r.min()) * (nx - 1)).astype(int)
    yi = ((h - h.min()) / (h.max() - h.min()) * (ny - 1)).astype(int)
    np.add.at(grid, (yi, xi), 1)
    vmax = max(grid.max(), 1)
    for row in grid:
        print("".join(" .:+*#@"[min(int(v / vmax * 6), 6)] if v else " " for v in row))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voxel", type=float, default=0.003)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    base_f, _ = VIEWS[0]
    base = load_pcd(CRIPPED / base_f, args.voxel)
    f_base = fpfh(base)
    print(f"[加载] 基准={base_f} ({len(base.points)}点) + {len(VIEWS)-1} 帧")

    merged = o3d.geometry.PointCloud()
    merged += base
    save_txt(base, OUT / f"aligned_{base_f}")
    print("\n[配准] RANSAC 全局配准(多参数最优) → ICP 精配准(多阈值最优)")
    for name, deg in VIEWS[1:]:
        src = load_pcd(CRIPPED / name, args.voxel)
        f_src = fpfh(src)
        T0, f0 = global_reg(src, base, f_src, f_base, args.voxel)
        reg = refine_icp(src, base, T0)
        T1 = reg.transformation
        fe = o3d.pipelines.registration.evaluate_registration(src, base, 0.02, T1)
        print(f"\n  {name}")
        print(f"    RANSAC: {t_info(T0)} fitness={f0:.3f}")
        print(f"    ICP   : {t_info(T1)} fitness={fe.fitness:.3f} rmse={fe.inlier_rmse*1000:.1f}mm")
        cp = o3d.geometry.PointCloud(src)
        cp.transform(T1)
        save_txt(cp, OUT / f"aligned_{name}")
        merged += cp

    merged = merged.voxel_down_sample(args.voxel)
    save_txt(merged, OUT / "combined.txt")
    o3d.io.write_point_cloud(str(OUT / "combined.pcd"), merged)
    pts = np.asarray(merged.points)
    print(f"\n[保存] combined.txt/.pcd ({len(merged.points)}点), aligned_*.txt")
    print(f"   bbox: x[{pts[:,0].min():+.3f},{pts[:,0].max():+.3f}] "
          f"y[{pts[:,1].min():+.3f},{pts[:,1].max():+.3f}] z[{pts[:,2].min():+.3f},{pts[:,2].max():+.3f}]")
    render(merged, OUT / "combined.png", [(22, -60), (22, 30), (70, 30), (0, 0)],
           "cripped 六视角配准合并（FPFH+RANSAC+ICP）")
    render_gif(merged, OUT / "combined_rot.gif")
    print("--- 侧视(径向-高度)密度 ---")
    ascii_density(merged)
    print("\n[完成] 输出：", OUT)


if __name__ == "__main__":
    main()