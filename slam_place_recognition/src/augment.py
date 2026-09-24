"""Dataset wrappers used by the robustness and ablation experiments.

All wrappers keep the ``n_frames / poses_gt / gt_loops / frame(i)`` interface of
:mod:`data_io`, so the rest of the pipeline is unchanged.

* :class:`BandSliceDataset`      - band-count ablation (paper Table 2).
* :class:`SpectralNoiseDataset`  - real spectral-noise sweep (修改意见 Sec. 12).
* :class:`GeometricNoiseDataset` - XYZ noise + point dropout (修改意见 Sec. 20).
* :class:`DriftInjectedPointsDataset` - bakes pose drift into the points, used to
  test the paper's M2DP/"PCA principal axis" causal claim (Sec. 5.5/6.2).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict

import numpy as np

from config import SpectralModelConfig
from spectral_model import SpectralSimulator


class _Wrapper:
    def __init__(self, ds):
        self.ds = ds
        self.n_frames = ds.n_frames
        self.poses_gt = ds.poses_gt
        self.gt_loops = ds.gt_loops

    def frame(self, i: int) -> Dict[str, np.ndarray]:
        return self.ds.frame(i)


# --------------------------------------------------------------------------- #
class BandSliceDataset(_Wrapper):
    """Keep ``n_bands`` of the available bands (evenly spaced indices)."""

    def __init__(self, ds, n_bands: int, n_available: int = 8):
        super().__init__(ds)
        self.idx = np.linspace(0, n_available - 1, n_bands).round().astype(int)
        self.n_bands = n_bands

    def frame(self, i: int) -> Dict[str, np.ndarray]:
        f = dict(self.ds.frame(i))
        f["spectra"] = f["spectra"][:, self.idx]
        f["band_indices"] = self.idx
        return f


# --------------------------------------------------------------------------- #
class SpectralNoiseDataset(_Wrapper):
    """Re-sample the spectra with a different measurement noise sigma."""

    def __init__(self, ds, sigma: float, base_cfg: SpectralModelConfig | None = None,
                 seed: int = 99):
        super().__init__(ds)
        cfg = replace(base_cfg or SpectralModelConfig(), sigma_spectral=float(sigma))
        self.sim = SpectralSimulator(cfg, ds.frame(0)["spectra"].shape[1], seed=seed)
        self.sigma = sigma

    def frame(self, i: int) -> Dict[str, np.ndarray]:
        f = dict(self.ds.frame(i))
        rng = np.random.default_rng(1000 + i)
        f["spectra"] = self.sim.simulate(f["material_ids"], f["xyz"], rng)
        return f


# --------------------------------------------------------------------------- #
class GeometricNoiseDataset(_Wrapper):
    """Perturb point coordinates and/or drop a fraction of the points."""

    def __init__(self, ds, sigma_xyz: float = 0.0, drop_rate: float = 0.0,
                 seed: int = 7):
        super().__init__(ds)
        self.sigma_xyz = float(sigma_xyz)
        self.drop_rate = float(drop_rate)
        self.seed = seed

    def frame(self, i: int) -> Dict[str, np.ndarray]:
        f = dict(self.ds.frame(i))
        rng = np.random.default_rng(self.seed * 100000 + i)
        xyz = f["xyz"].astype(np.float64)
        if self.sigma_xyz > 0:
            xyz = xyz + rng.normal(0.0, self.sigma_xyz, size=xyz.shape)
        keep = np.ones(len(xyz), dtype=bool)
        if self.drop_rate > 0:
            keep = rng.random(len(xyz)) >= self.drop_rate
        f["xyz"] = xyz[keep].astype(np.float32)
        f["spectra"] = f["spectra"][keep]
        for k in ("material_ids", "scene_ids"):
            if k in f:
                f[k] = np.asarray(f[k])[keep]
        return f


# --------------------------------------------------------------------------- #
class DriftInjectedPointsDataset(_Wrapper):
    """Apply an interpolated odometry drift directly to the scan coordinates.

    Provided *only* to test the causal explanation given for M2DP in
    Sec. 5.5/6.2; it is not part of the described pipeline.
    """

    def __init__(self, ds, delta: np.ndarray, scale: float = 1.0):
        super().__init__(ds)
        self.delta = np.asarray(delta, dtype=np.float64)
        self.scale = float(scale)

    def frame(self, i: int) -> Dict[str, np.ndarray]:
        f = dict(self.ds.frame(i))
        d = self.scale * self.delta[i, :3]
        p = f["xyz"].astype(np.float64)
        # a rigid-ish shift growing with distance from the sensor models the
        # "principal axis rotation" the paper describes
        r = np.linalg.norm(p[:, :2], axis=1, keepdims=True) / 40.0
        p = p + r * d[None, :] + d[None, :]
        f["xyz"] = p.astype(np.float32)
        return f
