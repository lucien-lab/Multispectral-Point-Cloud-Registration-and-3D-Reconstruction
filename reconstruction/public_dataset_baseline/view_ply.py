from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser(description="View a PLY point cloud or mesh with Open3D.")
    parser.add_argument("ply", help="Path to .ply file")
    parser.add_argument("--no-window", action="store_true", help="Only print file statistics.")
    args = parser.parse_args()

    path = Path(args.ply).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
        mesh.compute_vertex_normals()
        print(f"mesh: {path}")
        print(f"vertices: {len(mesh.vertices)}")
        print(f"triangles: {len(mesh.triangles)}")
        if not args.no_window:
            o3d.visualization.draw_geometries([mesh], window_name=path.name)
        return

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points)
    print(f"point cloud: {path}")
    print(f"points: {len(points)}")
    print(f"has colors: {pcd.has_colors()}")
    print(f"has normals: {pcd.has_normals()}")
    if len(points) > 0:
        print(f"bbox min: {points.min(axis=0)}")
        print(f"bbox max: {points.max(axis=0)}")

    if not args.no_window:
        o3d.visualization.draw_geometries([pcd], window_name=path.name)


if __name__ == "__main__":
    main()
