"""Procedural synthetic sequence that mimics KITTI Sequence 05.

Why this exists
---------------
KITTI Odometry Sequence 05 (plus SemanticKITTI labels) is not redistributable
and is not present in ``dat/``.  To keep the whole pipeline executable and
testable, this module generates a *structurally equivalent* sequence:

* a closed urban route driven twice, so that genuine revisits exist;
* a material map (asphalt road, concrete sidewalk/buildings, vegetation
  patches, metal poles, car paint) laid out along the route;
* a smooth composition field that makes **different parts of the route have
  different material mixtures** -> this is what gives a place a spectral
  identity (set ``composition_variation = 0`` to remove it, which is exactly
  the "same material, different place" control of 修改意见 Sec. 15);
* frames are generated on demand (no large files on disk); only poses, ground
  truth loops and the scene raster are persisted into ``dat/synthetic/``.

Material ids follow :data:`spectral_model.MATERIALS`; scene categories are the
keys of :data:`spectral_model.SCENE_TO_MATERIAL`.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from config import SyntheticConfig
from spectral_model import MATERIAL_INDEX, SCENE_TO_MATERIAL

SCENE_CATEGORIES = tuple(SCENE_TO_MATERIAL.keys())
SCENE_INDEX = {c: i for i, c in enumerate(SCENE_CATEGORIES)}


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #
def build_route(n_laps: float, jitter: float, seed: int) -> np.ndarray:
    """Return an (n, 2) polyline of waypoints for the driven route."""

    rng = np.random.default_rng(seed)
    # rounded rectangle with two straights that are traversed repeatedly
    corners = np.array([[0.0, 0.0], [180.0, 0.0], [180.0, 120.0],
                        [0.0, 120.0], [0.0, 0.0]])
    pts = []
    for i in range(len(corners) - 1):
        a, b = corners[i], corners[i + 1]
        n = max(int(np.linalg.norm(b - a) / 2.0), 4)
        seg = np.linspace(a, b, n, endpoint=False)
        # smooth lateral perturbation so that the two laps are not identical
        t = np.linspace(0, np.pi, n)
        seg = seg + jitter * np.stack([np.sin(t + i), np.cos(t + i)], axis=1)
        pts.append(seg)
    loop = np.concatenate(pts, axis=0)
    return np.tile(loop, (int(np.ceil(n_laps)), 1))


def poses_along_route(route: np.ndarray, n_frames: int) -> np.ndarray:
    """Sample ``n_frames`` poses [x, y, yaw] uniformly along the route."""

    seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    samples = np.linspace(0.0, total * 0.999, n_frames)
    x = np.interp(samples, s, route[:, 0])
    y = np.interp(samples, s, route[:, 1])
    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)
    return np.stack([x, y, yaw], axis=1)


# --------------------------------------------------------------------------- #
# Material / height raster
# --------------------------------------------------------------------------- #
def _smooth_field(X: np.ndarray, Y: np.ndarray, length: float,
                  rng: np.random.Generator) -> np.ndarray:
    acc = np.zeros_like(X)
    for _ in range(4):
        theta = rng.uniform(0, np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        k = 2 * np.pi / (length * rng.uniform(0.6, 1.6))
        acc += np.sin(k * (X * np.cos(theta) + Y * np.sin(theta)) + phase)
    acc /= 4.0
    return np.clip(acc, -1.0, 1.0)


class MaterialMap:
    """Rasterised scene: material id + surface height for every cell."""

    def __init__(self, cfg: SyntheticConfig, route: np.ndarray):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed + 1)
        res = cfg.map_resolution
        margin = 60.0
        lo = route.min(axis=0) - margin
        hi = route.max(axis=0) + margin
        self.lo, self.hi, self.res = lo, hi, res
        nx = int(np.ceil((hi[0] - lo[0]) / res))
        ny = int(np.ceil((hi[1] - lo[1]) / res))
        self.shape = (nx, ny)
        xs = lo[0] + (np.arange(nx) + 0.5) * res
        ys = lo[1] + (np.arange(ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys, indexing="ij")

        # --- distance to the route polyline (road / sidewalk bands) -------- #
        d_route = self._distance_to_polyline(X, Y, route)

        mat = np.full(self.shape, SCENE_INDEX["soil"], dtype=np.int8)
        mat[d_route < 4.0] = SCENE_INDEX["road"]
        mat[(d_route >= 4.0) & (d_route < 7.0)] = SCENE_INDEX["sidewalk"]

        # --- composition field: varies along the route --------------------- #
        v = cfg.composition_variation
        f1 = _smooth_field(X, Y, 70.0, rng)
        f2 = _smooth_field(X, Y, 45.0, rng)
        comp = 0.5 + 0.5 * v * f1          # building density
        veg = 0.5 + 0.5 * v * f2           # vegetation density

        far = d_route >= 7.0
        buildings = far & (comp > 0.55)
        veg_patch = far & (comp <= 0.55) & (veg > 0.40)
        mat[buildings] = SCENE_INDEX["building"]
        mat[veg_patch] = SCENE_INDEX["vegetation"]
        mat[far & ~buildings & ~veg_patch] = SCENE_INDEX["soil"]

        # --- street furniture: poles and parked cars ----------------------- #
        self._scatter_objects(mat, route, rng, origin=lo, res=res)

        # --- heights -------------------------------------------------------- #
        h = np.zeros(self.shape, dtype=np.float32)
        is_b = mat == SCENE_INDEX["building"]
        h[is_b] = 9.0 + 6.0 * (0.5 + 0.5 * comp[is_b])
        is_v = mat == SCENE_INDEX["vegetation"]
        h[is_v] = 3.0 + 5.0 * (0.5 + 0.5 * veg[is_v])
        is_pole = mat == SCENE_INDEX["pole"]
        h[is_pole] = 6.0
        is_car = mat == SCENE_INDEX["car"]
        h[is_car] = 1.5
        h += rng.normal(0.0, 0.01, size=h.shape).astype(np.float32)

        self.material = mat
        self.height = h
        # per-cell within-class spectral gain (specimen variation)
        self.specimen = _smooth_field(X, Y, cfg.map_resolution * 60, rng)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _distance_to_polyline(X, Y, route, chunk: int = 200_000) -> np.ndarray:
        """Minimum distance from every grid cell to the route polyline."""
        xs = X.ravel()
        ys = Y.ravel()
        out = np.empty(xs.shape, dtype=np.float32)
        seg_a = route[:-1]
        seg_b = route[1:]
        v = seg_b - seg_a
        vv = (v ** 2).sum(axis=1) + 1e-12
        for start in range(0, len(xs), chunk):
            sl = slice(start, start + chunk)
            px = xs[sl][:, None]
            py = ys[sl][:, None]
            t = ((px - seg_a[:, 0]) * v[:, 0] + (py - seg_a[:, 1]) * v[:, 1]) / vv
            t = np.clip(t, 0.0, 1.0)
            dx = seg_a[:, 0] + t * v[:, 0] - px
            dy = seg_a[:, 1] + t * v[:, 1] - py
            out[sl] = np.sqrt((dx ** 2 + dy ** 2).min(axis=1))
        return out.reshape(X.shape)

    @staticmethod
    def _scatter_objects(mat: np.ndarray, route: np.ndarray,
                         rng: np.random.Generator, origin, res: float,
                         spacing: float = 22.0):
        """Poles along both kerbs and parked cars near the road edge."""
        seg = np.linalg.norm(np.diff(route, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        # de-duplicate the repeated laps so objects are stamped only once
        n = int(s[-1] / spacing)
        for k in range(n):
            t = (k + 0.5) * spacing
            idx = int(np.searchsorted(s, t))
            idx = min(idx, len(route) - 2)
            p = route[idx]
            d = route[idx + 1] - route[idx]
            d = d / (np.linalg.norm(d) + 1e-9)
            nrm = np.array([-d[1], d[0]])
            for side in (+1.0, -1.0):
                x, y = p + side * 4.6 * nrm
                _stamp(mat, (x, y), 0.4, SCENE_INDEX["pole"], rng,
                       res=res, origin=origin)
            if k % 3 == 1:
                side = float(rng.choice([-1.0, 1.0]))
                x, y = p + side * 3.0 * nrm
                _stamp(mat, (x, y), 1.0, SCENE_INDEX["car"], rng,
                       res=res, origin=origin)


def _stamp(mat: np.ndarray, xy, radius: float, category: int,
           rng: np.random.Generator, res: float = 0.5, origin=(0.0, 0.0)):
    """Mark a disc of cells with a scene category (helper for objects)."""
    nx, ny = mat.shape
    cx = int((xy[0] - origin[0]) / res)
    cy = int((xy[1] - origin[1]) / res)
    r = max(int(round(radius / res)), 1)
    x0, x1 = max(cx - r, 0), min(cx + r + 1, nx)
    y0, y1 = max(cy - r, 0), min(cy + r + 1, ny)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    sub = mat[x0:x1, y0:y1]
    sub[mask] = category


# --------------------------------------------------------------------------- #
# Frame generation
# --------------------------------------------------------------------------- #
class SyntheticSequence:
    """On-demand frame generator (points + material ids)."""

    def __init__(self, cfg: SyntheticConfig, material_map: MaterialMap):
        self.cfg = cfg
        self.map = material_map

    def frame_points(self, pose: np.ndarray, rng: np.random.Generator):
        """Sample a LiDAR-like scan around ``pose``.

        Returns
        -------
        xyz : (M, 3) float32 in the sensor frame (x forward, y left, z up)
        material_ids : (M,) int16, indices into ``spectral_model.MATERIALS``
        scene_ids : (M,) int8, scene categories (for diagnostics / Sec. 15)
        """
        cfg = self.cfg
        M = cfg.points_per_frame
        nx, ny = self.map.shape
        # ---- candidate cells inside the sensor window ---------------------- #
        n_cand = M * 4
        dx = rng.uniform(-cfg.range_m, cfg.range_m, size=n_cand)
        dy = rng.uniform(-cfg.range_m, cfg.range_m, size=n_cand)
        keep = (dx ** 2 + dy ** 2) <= cfg.range_m ** 2
        dx, dy = dx[keep], dy[keep]

        # rotate into world frame
        c, s = np.cos(pose[2]), np.sin(pose[2])
        wx = pose[0] + c * dx - s * dy
        wy = pose[1] + s * dx + c * dy

        ix = ((wx - self.map.lo[0]) / self.map.res).astype(np.int64)
        iy = ((wy - self.map.lo[1]) / self.map.res).astype(np.int64)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ix, iy = ix[ok], iy[ok]
        dx, dy, wx, wy = dx[ok], dy[ok], wx[ok], wy[ok]

        # ---- LiDAR-like angular sampling: density ~ 1 / r^2 ---------------- #
        r = np.sqrt(dx ** 2 + dy ** 2)
        w = 1.0 / (r ** 2 + 1.0)
        p = w / w.sum()
        if len(p) < M:
            idx = rng.choice(len(p), size=M, p=p)
        else:
            idx = rng.choice(len(p), size=M, replace=False, p=p) \
                if M <= len(p) else rng.choice(len(p), size=M, p=p)

        scene = self.map.material[ix[idx], iy[idx]].astype(np.int8)
        h = self.map.height[ix[idx], iy[idx]].astype(np.float64)

        # ---- z: ground returns at ~0, structures sampled over their height - #
        is_ground = np.isin(scene, [SCENE_INDEX["road"], SCENE_INDEX["sidewalk"],
                                    SCENE_INDEX["soil"]])
        z = np.where(is_ground, 0.0, rng.uniform(0.0, 1.0, size=len(h)) * h)
        z = z - 1.8   # sensor mounted ~1.8 m above ground

        xyz_sensor = np.stack([dx[idx], dy[idx], z + 1.8], axis=1)
        xyz_sensor += rng.normal(0.0, cfg.point_sigma_xyz, size=xyz_sensor.shape)

        mat_names = [SCENE_TO_MATERIAL[SCENE_CATEGORIES[i]] for i in scene]
        material_ids = np.array([MATERIAL_INDEX[m] for m in mat_names],
                                dtype=np.int16)
        return (xyz_sensor.astype(np.float32), material_ids,
                scene.astype(np.int8))


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def build_sequence(cfg: SyntheticConfig):
    """Build route, poses, material map, sequence object and GT loops."""

    route = build_route(cfg.n_laps, cfg.lateral_jitter_m, cfg.seed)
    poses = poses_along_route(route, cfg.n_frames)
    matmap = MaterialMap(cfg, route)
    seq = SyntheticSequence(cfg, matmap)
    gt_loops = ground_truth_loops(poses, cfg.revisit_distance_m)
    return {
        "route": route,
        "poses": poses,
        "map": matmap,
        "sequence": seq,
        "gt_loops": gt_loops,
        "config": asdict(cfg),
    }


def ground_truth_loops(poses: np.ndarray, radius: float) -> np.ndarray:
    """All frame pairs closer than ``radius`` (pairwise, small sequences only)."""

    xy = np.asarray(poses)[:, :2]
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
    iu = np.triu_indices(len(xy), k=1)
    m = d[iu] <= radius
    return np.stack([iu[0][m], iu[1][m]], axis=1)


def save_metadata(path: str, data: Dict) -> None:
    os.makedirs(path, exist_ok=True)
    np.savez_compressed(
        os.path.join(path, "meta.npz"),
        poses=data["poses"],
        route=data["route"],
        map_material=data["map"].material,
        map_height=data["map"].height,
        map_lo=data["map"].lo,
        map_hi=data["map"].hi,
        map_res=np.array([data["map"].res]),
        gt_loops=data["gt_loops"],
    )
