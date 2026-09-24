"""Prepare / validate the ``dat/`` directory.

Two modes
---------
``synthetic`` (default)
    Build the procedural sequence metadata and cache it under
    ``dat/synthetic/`` (poses, route, material raster, ground-truth loops).
    Frames themselves are generated on demand, so nothing large is written.

``kitti``
    Validate that a real KITTI Odometry + (optional) SemanticKITTI tree is laid
    out as expected and report what was found.  No download is attempted.

Usage
-----
    python prepare_data.py                       # build synthetic metadata
    python prepare_data.py --check kitti
    python prepare_data.py --frames 600 --points 20000
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DAT_DIR  # noqa: E402

KITTI_LAYOUT = """
dat/
└── kitti/                                  (KITTI Odometry benchmark, ~22 GB)
    ├── dataset/
    │   ├── poses/05.txt                    (3x4 pose matrices, one per line)
    │   └── sequences/05/
    │       ├── velodyne/000000.bin ...      (float32 x,y,z,intensity)
    │       └── times.txt
└── semantickitti/                          (optional labels, ~1 GB for seq 05)
    └── dataset/sequences/05/labels/000000.label

Download hints
--------------
  KITTI odometry : https://www.cvlibs.net/datasets/kitti/eval_odometry.php
                   (odometry data set - velodyne laser data, 22 GB)
  SemanticKITTI  : http://www.semantic-kitti.org/dataset.html
                   (labels + poses; the velodyne scans are shared with KITTI)
"""


def check_kitti(root: str | None = None, sequence: str = "05") -> bool:
    root = root or os.path.join(DAT_DIR, "kitti")
    ok = True
    checks = [
        ("poses", os.path.join(root, "dataset", "poses", f"{sequence}.txt"),
         "file"),
        ("velodyne dir", os.path.join(root, "dataset", "sequences", sequence,
                                      "velodyne"), "dir"),
    ]
    print(f"checking KITTI layout under {root}")
    for label, path, kind in checks:
        exists = os.path.isfile(path) if kind == "file" else os.path.isdir(path)
        print(f"  [{'OK ' if exists else 'MISS'}] {label:14s} {path}")
        ok &= exists
    if ok:
        bins = sorted(glob.glob(os.path.join(
            root, "dataset", "sequences", sequence, "velodyne", "*.bin")))
        print(f"  found {len(bins)} scan files")
    labels = os.path.join(DAT_DIR, "semantickitti", "dataset", "sequences",
                          sequence, "labels")
    has_labels = os.path.isdir(labels) and bool(glob.glob(os.path.join(labels, "*.label")))
    print(f"  [{'OK ' if has_labels else 'MISS'}] SemanticKITTI labels {labels}")
    if not has_labels:
        print("       (without labels every point is treated as class 0 / 'soil')")
    if not ok:
        print(KITTI_LAYOUT)
    return ok


def build_synthetic(frames: int, points: int, out: str | None = None) -> str:
    from config import SyntheticConfig
    import synth

    cfg = SyntheticConfig(n_frames=frames, points_per_frame=points)
    data = synth.build_sequence(cfg)
    out = out or os.path.join(DAT_DIR, "synthetic")
    os.makedirs(out, exist_ok=True)
    synth.save_metadata(out, data)
    poses = data["poses"]
    loops = data["gt_loops"]
    print(f"synthetic metadata -> {os.path.join(out, 'meta.npz')}")
    print(f"  frames            : {len(poses)}")
    print(f"  route length (m)  : {np.linalg.norm(np.diff(data['route'], axis=0), axis=1).sum():.0f}")
    print(f"  material raster   : {data['map'].shape} @ {data['map'].res} m")
    print(f"  ground-truth loops: {len(loops)}  (same place, |i-j| > 1)")
    unique_laps = np.unique(loops[:, 1] - loops[:, 0]) if len(loops) else []
    if len(unique_laps):
        print(f"  loop frame gaps   : min {unique_laps.min()}, max {unique_laps.max()}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="synthetic", choices=["synthetic", "kitti"])
    ap.add_argument("--check", dest="mode2", default=None,
                    choices=["kitti", "synthetic"])
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--sequence", default="05")
    args = ap.parse_args(argv)

    mode = args.mode2 or args.mode
    os.makedirs(DAT_DIR, exist_ok=True)
    if mode == "kitti":
        ok = check_kitti(sequence=args.sequence)
        sys.exit(0 if ok else 1)
    build_synthetic(args.frames, args.points)


if __name__ == "__main__":
    main()
