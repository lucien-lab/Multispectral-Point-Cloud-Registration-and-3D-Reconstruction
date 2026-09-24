"""Dataset access layer.

Two back-ends
-------------
``kitti``     - real KITTI Odometry + optional SemanticKITTI labels, expected as

                    dat/kitti/dataset/sequences/05/velodyne/*.bin
                    dat/kitti/dataset/sequences/05/times.txt
                    dat/kitti/dataset/poses/05.txt
                    dat/semantickitti/dataset/sequences/05/labels/*.label

``synthetic`` - procedural sequence (see :mod:`synth`), used when KITTI is not
                present so that the pipeline stays executable.

Both back-ends expose the same interface::

    ds = load_dataset(cfg)
    ds.n_frames
    ds.poses_gt           # (n, 3) [x, y, yaw]
    ds.gt_loops           # (k, 2) ground-truth loop pairs
    ds.frame(i) -> dict(xyz=(M,3), spectra=(M,B), scene_ids=(M,), material_ids=(M,))
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List

import numpy as np

from config import Config, DAT_DIR
from spectral_model import (
    LABEL_CONVENTIONS,
    MATERIAL_INDEX,
    SEMANTICKITTI_TO_MATERIAL,
    SpectralSimulator,
)


def build_label_lut(convention: str) -> np.ndarray:
    """65536-entry lookup: raw SemanticKITTI id -> material index."""

    lut = np.full(65536, MATERIAL_INDEX["soil"], dtype=np.int16)
    for lab, mat in LABEL_CONVENTIONS[convention].items():
        if 0 <= lab < 65536:
            lut[lab] = MATERIAL_INDEX[mat]
    return lut


_LABEL_LUTS = {c: build_label_lut(c) for c in LABEL_CONVENTIONS}


# --------------------------------------------------------------------------- #
def _pose_from_matrix(T: np.ndarray) -> np.ndarray:
    """KITTI 3x4 pose matrix -> planar ``[x, y, yaw]``.

    KITTI odometry poses are expressed in the left-camera frame (x right,
    y down, z forward), so the driving plane is the **x-z** plane.  Projecting
    onto (x, y) would map the trajectory to a vertical plane where almost every
    frame looks co-located - a mistake that inflated the ground-truth loop count
    of sequence 05 from 11,786 to 86,947 in this pipeline before it was fixed.

    Note: the *point clouds* are already in the Velodyne frame (x forward,
    y left, z up) and are therefore used as-is; only the poses need the planar
    projection.  Distances are invariant to this rotation/reflection, so the
    remaining pipeline is unaffected.
    """
    return np.array([T[0, 3], T[2, 3], np.arctan2(T[0, 2], T[2, 2])],
                    dtype=np.float64)


def _load_kitti_poses(path: str, n: int | None = None) -> np.ndarray:
    rows = []
    with open(path) as fh:
        for line in fh:
            v = np.array([float(x) for x in line.split()])
            rows.append(_pose_from_matrix(v.reshape(3, 4)))
    poses = np.stack(rows)
    return poses[:n] if n else poses


# --------------------------------------------------------------------------- #
class KittiDataset:
    """Real-data back-end."""

    def __init__(self, cfg: Config, root: str | None = None):
        self.cfg = cfg
        self.root = root or os.path.join(DAT_DIR, "kitti")
        seq = cfg.sequence
        self.velo_dir = os.path.join(self.root, "dataset", "sequences", seq, "velodyne")
        pose_file = os.path.join(self.root, "dataset", "poses", f"{seq}.txt")
        self.label_dir = os.path.join(DAT_DIR, "semantickitti", "dataset",
                                      "sequences", seq, "labels")
        if not os.path.isdir(self.velo_dir):
            raise FileNotFoundError(
                f"KITTI velodyne directory not found: {self.velo_dir}\n"
                "See dat/README.md for the expected layout.")
        self.files = sorted(glob.glob(os.path.join(self.velo_dir, "*.bin")))
        self.poses_gt_all = _load_kitti_poses(pose_file)
        self.label_convention = self._detect_label_convention(cfg)
        if self.label_convention:
            print(f"  SemanticKITTI label convention: {self.label_convention}")
        # Scan file names are not necessarily contiguous (a subset may have been
        # downloaded), so map every scan to its pose row by the integer in the
        # file name instead of by position.
        self.frame_ids = np.array([int(os.path.basename(f)[:6]) for f in self.files],
                                  dtype=np.int64)
        valid = self.frame_ids < len(self.poses_gt_all)
        if not np.all(valid):
            print(f"  warning: dropped {int((~valid).sum())} scan(s) without a pose")
        self.files = [f for f, ok in zip(self.files, valid) if ok]
        self.frame_ids = self.frame_ids[valid]
        n = len(self.files)
        if cfg.max_frames and cfg.max_frames < n:
            keep = np.linspace(0, n - 1, cfg.max_frames).round().astype(int)
            self.files = [self.files[i] for i in keep]
            self.frame_ids = self.frame_ids[keep]
        self.poses_gt = self.poses_gt_all[self.frame_ids]
        self.n_frames = len(self.files)
        self.gt_loops = self._gt_loops()
        self.sim = SpectralSimulator(cfg.spectral, cfg.jsss.n_bands, seed=11)
        self._rng = np.random.default_rng(12)

    # ------------------------------------------------------------------ #
    def _gt_loops(self, radius: float = 8.0) -> np.ndarray:
        """All consecutive-indexed frame pairs within ``radius`` (KD-tree)."""

        from scipy.spatial import cKDTree

        xy = self.poses_gt[:, :2]
        if len(xy) < 2:
            return np.zeros((0, 2), dtype=np.int64)
        pairs = cKDTree(xy).query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return pairs[np.argsort(pairs[:, 0] * 10 ** 6 + pairs[:, 1])]

    # ------------------------------------------------------------------ #
    def _detect_label_convention(self, cfg: Config) -> str:
        """Detect which SemanticKITTI id space the label files use.

        The official odometry label archive stores *raw* ids (40 = road,
        70 = vegetation, ...), while the SemanticKITTI API's learning map
        produces ids 0-19.  Override with ``cfg.label_convention`` ("raw" /
        "learning").
        """

        forced = getattr(cfg, "label_convention", "auto")
        if forced in ("raw", "learning"):
            return forced
        files = sorted(glob.glob(os.path.join(self.label_dir, "*.label")))
        if not files:
            return "raw"                      # labels absent -> irrelevant
        sem = np.fromfile(files[0], dtype=np.uint32) & 0xFFFF
        if len(sem) == 0:
            return "raw"
        frac_learning = float((sem <= 19).mean())
        return "learning" if frac_learning > 0.99 else "raw"

    # ------------------------------------------------------------------ #
    def _labels(self, i: int, m: int) -> np.ndarray:
        """Map a SemanticKITTI ``.label`` file to material indices (vectorised)."""

        fid = int(self.frame_ids[i])
        path = os.path.join(self.label_dir, f"{fid:06d}.label")
        if not os.path.isfile(path):
            return np.zeros(m, dtype=np.int16)   # unlabeled
        lab = np.fromfile(path, dtype=np.uint32)
        sem = (lab & 0xFFFF).astype(np.int64)
        sem = sem[:m] if len(sem) >= m else np.pad(sem, (0, m - len(sem)))
        return _LABEL_LUTS[self.label_convention][sem]

    # ------------------------------------------------------------------ #
    def frame(self, i: int) -> Dict[str, np.ndarray]:
        scan = np.fromfile(self.files[i], dtype=np.float32).reshape(-1, 4)
        # KITTI velodyne .bin files are already in the Velodyne frame
        # (x forward, y left, z up) - no conversion needed.  Only the poses
        # (camera frame) require the planar projection, see _pose_from_matrix.
        xyz = scan[:, :3].astype(np.float32)
        material_ids = self._labels(i, len(xyz))
        spectra = self.sim.simulate(material_ids, xyz, self._rng)
        return {"xyz": xyz, "spectra": spectra,
                "material_ids": material_ids, "scene_ids": material_ids,
                "intensity_raw": scan[:, 3].astype(np.float32),
                "frame_id": np.int64(self.frame_ids[i])}


# --------------------------------------------------------------------------- #
class SyntheticDataset:
    """Procedural back-end (see :mod:`synth`)."""

    def __init__(self, cfg: Config):
        import synth  # local import: heavy module only needed here

        sconf = cfg.synth
        if cfg.max_frames:
            sconf.n_frames = int(cfg.max_frames)
        self._synth = synth
        built = synth.build_sequence(sconf)
        self.route = built["route"]
        self.poses_gt = built["poses"]
        self.gt_loops = built["gt_loops"]
        self.map = built["map"]
        self.sequence = built["sequence"]
        self.n_frames = len(self.poses_gt)
        self.sim = SpectralSimulator(cfg.spectral, cfg.jsss.n_bands, seed=13)
        self._rng = np.random.default_rng(14)
        self._cache: Dict[int, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------ #
    def frame(self, i: int) -> Dict[str, np.ndarray]:
        if i in self._cache:
            return self._cache[i]
        rng = np.random.default_rng(self.cfg_seed + i)
        xyz, material_ids, scene_ids = self.sequence.frame_points(self.poses_gt[i], rng)
        spectra = self.sim.simulate(material_ids, xyz, rng)
        out = {"xyz": xyz, "spectra": spectra, "material_ids": material_ids,
               "scene_ids": scene_ids}
        self._cache[i] = out
        return out

    @property
    def cfg_seed(self) -> int:
        return 1234


# --------------------------------------------------------------------------- #
def load_dataset(cfg: Config):
    """Factory: real KITTI when available, procedural synthetic otherwise."""

    if cfg.dataset == "kitti":
        return KittiDataset(cfg)
    if cfg.dataset == "synthetic":
        return SyntheticDataset(cfg)
    raise ValueError(f"unknown dataset backend: {cfg.dataset}")


def ground_truth_loop_set(poses: np.ndarray, radius: float = 8.0,
                          delta_min: int = 10) -> set:
    """Explicit ground-truth loop set honouring the temporal gap |i-j| > Delta_min."""

    xy = np.asarray(poses)[:, :2]
    out = set()
    n = len(xy)
    for i in range(n):
        d = np.linalg.norm(xy[i + 1:] - xy[i], axis=1)
        for off in np.nonzero(d <= radius)[0]:
            j = i + 1 + int(off)
            if abs(j - i) > delta_min:
                out.add((i, j))
    return out
