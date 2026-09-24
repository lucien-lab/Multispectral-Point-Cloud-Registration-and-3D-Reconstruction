from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LOG_PATH = ROOT / "experiment_log.md"


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {stamp} {message}\n")


def copy_if_needed(src: str | Path, dst: Path) -> int:
    src = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst.stat().st_size


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    open3d_root = DATA_DIR / "open3d_cache"

    log("开始下载/缓存 Open3D 示例数据：BunnyMesh 与 DemoICPPointClouds。")

    bunny = o3d.data.BunnyMesh(data_root=str(open3d_root))
    bunny_dst = DATA_DIR / "stanford_bunny" / "BunnyMesh.ply"
    bunny_size = copy_if_needed(bunny.path, bunny_dst)

    demo_icp = o3d.data.DemoICPPointClouds(data_root=str(open3d_root))
    icp_paths = []
    for i, src in enumerate(demo_icp.paths):
        dst = DATA_DIR / "open3d_demo_icp" / f"cloud_bin_{i}.pcd"
        size = copy_if_needed(src, dst)
        icp_paths.append({"file": str(dst.relative_to(ROOT)), "bytes": size})

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": [
            {
                "name": "Stanford Bunny via Open3D BunnyMesh",
                "source": "Stanford 3D Scanning Repository / Open3D data module",
                "local_file": str(bunny_dst.relative_to(ROOT)),
                "bytes": bunny_size,
                "purpose": "surface reconstruction baseline and controlled registration validation",
            },
            {
                "name": "Open3D DemoICPPointClouds",
                "source": "Open3D data module",
                "local_files": icp_paths,
                "purpose": "real scanned point cloud fragments for ICP tutorial comparison",
            },
        ],
        "note": (
            "The multispectral attributes used in the validation script are controlled "
            "synthetic attributes attached to public geometry. They are used only for "
            "method sanity checks and are not reported as measured multispectral LiDAR data."
        ),
    }
    manifest_path = DATA_DIR / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"数据集清单写入 {manifest_path.relative_to(ROOT)}。")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
