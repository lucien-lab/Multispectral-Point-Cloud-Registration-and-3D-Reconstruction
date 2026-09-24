#!/usr/bin/env python3
"""发财树 0~300° 六视角点云配准 → 同一棵树。

背景（已用数值诊断确认）：cripped/ 下 6 帧点云**本就在同一全局坐标系**
（扫描仪固定+转台旋转，镜像重合 fitness≈0.9），因此：
  - 初值 = 单位阵（不做名义角度粗旋转，避免球形树冠 ICP 跑飞）
  - 精配准 = 保守 point-to-plane ICP，仅当变换量小时采用，
    否则回退初值（如 60° 帧曾漂移 154mm 属局部极小）

输出 registered/：
  aligned_*.txt      各帧对齐后的点云（含 RGB）
  combined.txt/pcd   合并去样板后的完整树点云
  combined_*.png     多视角可视化渲染
  rotation.gif       绕竖轴旋转动画（直观展示完整性）

用法：python register.py [--mode both|raw|icp] [--gif]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

SCRIPT_DIR = Path(__file__).resolve().parent
CRIPPED = SCRIPT_DIR / "cripped"
OUT = SCRIPT_DIR / "registered"

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
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
    return pcd


def save_txt(pcd: o3d.geometry.PointCloud, path: Path) -> None:
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    np.savetxt(path, np.hstack([pts, cols, np.zeros((len(pts), 2))]),
               delimiter=",", fmt="%.6f")


def render_static(pcd: o3d.geometry.PointCloud, path: Path,
                  views: list[tuple[float, float]], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(views)
    fig = plt.figure(figsize=(5 * n, 5))
    fig.suptitle(title, fontsize=12)
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    for i, (el, az) in enumerate(views):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5,
                   c=np.clip(cols * 2.0, 0, 1))
        ax.view_init(elev=el, azim=az)
        ax.set_title(f"elev={el}° azim={az}°", fontsize=9)
        ax.set_box_aspect((1, 1, 1))
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  [渲染] {path}")


def render_gif(pcd: o3d.geometry.PointCloud, path: Path, n_frames: int = 36) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("  [跳过] 缺少 matplotlib.animation（pillow）")
        return
    pts, cols = np.asarray(pcd.points), np.asarray(pcd.colors)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.5,
                    c=np.clip(cols * 2.0, 0, 1))

    def update(frame: int):
        ax.view_init(elev=22, azim=frame * 360 / n_frames)
        return (sc,)
    anim = FuncAnimation(fig, update, frames=n_frames, interval=90, blit=False)
    anim.save(path, writer="pillow", dpi=90)
    plt.close(fig)
    print(f"  [渲染] {path}（{n_frames} 帧旋转动画）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["both", "raw", "icp"], default="both",
                    help="raw=仅初值直接合并；icp=保守ICP精调；both=两者对比")
    ap.add_argument("--voxel", type=float, default=0.003)
    ap.add_argument("--icp-max", type=float, default=0.006, help="ICP最大对应距离")
    ap.add_argument("--icp-iters", type=int, default=60)
    ap.add_argument("--max-shift-mm", type=float, default=50.0,
                    help="ICP变换超此平移量则视为跑飞，回退初值")
    ap.add_argument("--gif", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)

    base_name, _ = VIEWS[0]
    base = load_pcd(CRIPPED / base_name, args.voxel)
    frames = [(n, d, load_pcd(CRIPPED / n, args.voxel)) for n, d in VIEWS[1:]]
    print(f"[加载] 基准={base_name} ({len(base.points)}点) + {len(frames)} 帧, voxel={args.voxel}")

    # ---- 评估初值（单位阵，即 cripped 全局坐标）----
    print(f"[初值评估] {base_name} 为基准（fitness: 对应距离<20mm 的点占比）")
    raw_T: dict[str, np.ndarray] = {}
    for name, _, pcd in frames:
        reg = o3d.pipelines.registration.evaluate_registration(pcd, base, 0.02, np.eye(4))
        raw_T[name] = np.eye(4)
        print(f"  {name}: fitness={reg.fitness:.3f} rmse={reg.inlier_rmse*1000:.1f}mm")

    # ---- 保守 ICP 精调 ----
    icp_T: dict[str, np.ndarray] = {}
    for name, _, pcd in frames:
        reg = o3d.pipelines.registration.registration_icp(
            pcd, base, args.icp_max, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=args.icp_iters))
        T = reg.transformation
        shift = np.linalg.norm(T[:3, 3]) * 1000
        ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        if shift > args.max_shift_mm:
            print(f"  [回退] {name}: ICP 平移 {shift:.0f}mm > {args.max_shift_mm:.0f}mm，"
                  f"判定局部极小，使用初值")
            T = np.eye(4)
        icp_T[name] = T
        print(f"  [ICP] {name}: rot={ang:.1f}° shift={shift:.1f}mm "
              f"fitness={reg.fitness:.3f}→{o3d.pipelines.registration.evaluate_registration(pcd.transform(T), base, 0.02, np.eye(4)).fitness:.3f}")

    def build(tag: str, Tmap: dict[str, np.ndarray]) -> o3d.geometry.PointCloud:
        merged = o3d.geometry.PointCloud()
        merged += base
        for name, _, pcd in frames:
            cp = o3d.geometry.PointCloud(pcd)
            cp.transform(Tmap[name])
            save_txt(cp, OUT / f"aligned_{tag}_{name}")
            merged += cp
        merged = merged.voxel_down_sample(args.voxel)
        save_txt(merged, OUT / f"combined_{tag}.txt")
        o3d.io.write_point_cloud(str(OUT / f"combined_{tag}.pcd"), merged)
        print(f"[保存] combined_{tag}: {len(merged.points)} 点 → "
              f"combined_{tag}.txt/.pcd、aligned_{tag}_*.txt")
        return merged

    if args.mode in ("raw", "both"):
        raw = build("raw", raw_T)
        render_static(raw, OUT / "combined_raw.png",
                      [(22, -60), (22, 30), (70, 30), (0, 0)],
                      "合并点云（初值=原坐标系，未ICP）")
        if args.gif:
            render_gif(raw, OUT / "rotation_raw.gif")
    if args.mode in ("icp", "both"):
        icp = build("icp", icp_T)
        render_static(icp, OUT / "combined_icp.png",
                      [(22, -60), (22, 30), (70, 30), (0, 0)],
                      "合并点云（保守ICP精调）")
        if args.gif:
            render_gif(icp, OUT / "rotation_icp.gif")
    print("[完成] 输出目录：", OUT)


if __name__ == "__main__":
    main()