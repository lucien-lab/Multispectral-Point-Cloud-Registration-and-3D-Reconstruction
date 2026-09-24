"""M2DP descriptor (He et al., IROS 2016) - baseline of Sec. 3.4/5.5.

Pipeline: PCA on the scan -> project onto the plane spanned by the first two
principal axes -> 2-D density histogram (radial x angular) -> repeat for
``n_azimuth`` rotation angles -> L2-normalised concatenated vector.

Audit note (修改意见 Sec. 13 / P0 in the multi-model review): this encoding is a
function of the point cloud **only**.  A pose-only odometry drift cannot change
``M2DP(X_i)``; therefore the paper's explanation "里程计漂移改变了点的空间分布，
导致主轴方向偏移" is only valid if the drift is additionally baked into the
points (see :func:`drift.apply_drift_to_points`).
"""

from __future__ import annotations

import numpy as np

from config import M2DPConfig
from descriptors.jsss import Normalizer, rms_similarity


class M2DP:
    name = "M2DP"

    def __init__(self, cfg: M2DPConfig):
        self.cfg = cfg
        self._norm: Normalizer | None = None

    # ------------------------------------------------------------------ #
    def extract(self, frame: dict) -> np.ndarray:
        p = np.asarray(frame["xyz"], dtype=np.float64)
        cfg = self.cfg
        nb = cfg.n_radial * cfg.n_angular
        if len(p) < cfg.min_points:
            return np.zeros(cfg.n_azimuth * nb)
        c = p.mean(axis=0)
        q = p - c
        try:
            _, _, vt = np.linalg.svd(q, full_matrices=False)
        except np.linalg.LinAlgError:
            return np.zeros(cfg.n_azimuth * nb)
        # the 2-D plane spanned by the two leading principal axes; the third
        # axis (normal) defines the four projection directions.
        # PCA axis signs are arbitrary -> canonicalise them deterministically,
        # otherwise the descriptor is not reproducible across frames (classic
        # M2DP pitfall).
        u = self._canonical_axis(q, vt[0])
        v = self._canonical_axis(q, vt[1])
        n = np.cross(u, v)
        out = []
        for k in range(cfg.n_azimuth):
            ang = k * np.pi / cfg.n_azimuth
            e1 = np.cos(ang) * u + np.sin(ang) * n
            e2 = np.cos(ang) * v + np.sin(ang) * n
            x = q @ e1
            y = q @ e2
            r = np.sqrt(x ** 2 + y ** 2)
            th = np.arctan2(y, x)
            h, _, _ = np.histogram2d(
                r, th, bins=[cfg.n_radial, cfg.n_angular],
                range=[[0.0, cfg.max_range], [-np.pi, np.pi]])
            out.append(h.ravel())
        vec = np.concatenate(out)
        nrm = np.linalg.norm(vec)
        return vec / nrm if nrm > 1e-12 else vec

    # ------------------------------------------------------------------ #
    @staticmethod
    def _canonical_axis(points: np.ndarray, axis: np.ndarray) -> np.ndarray:
        """Fix the sign ambiguity of a principal axis.

        Convention: the projected third moment along the axis is non-negative;
        ties are broken by making the largest-|projection| point positive.
        """

        proj = points @ axis
        m3 = float((proj ** 3).mean())
        if abs(m3) < 1e-12:
            m3 = float(proj[np.argmax(np.abs(proj))])
        return axis if m3 >= 0 else -axis

    def fit(self, comps):
        self._norm = Normalizer().fit(np.stack(comps))
        return self

    def similarity(self, left, right) -> np.ndarray:
        A = self._norm.transform(np.stack(left))
        B = self._norm.transform(np.stack(right))
        return rms_similarity(A, B)
