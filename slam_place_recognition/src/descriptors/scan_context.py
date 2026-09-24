"""ScanContext descriptor (Kim & Kim, IROS 2019) - baseline of Sec. 3.4/5.5.

Implementation choices (the paper only states Nr=15, Ns=40):

* bin value = maximum z in the (ring, sector) cell, as in the original paper;
* distance = minimum over column shifts of the cosine distance,
  ``sim = 1 - dist`` clipped to [0, 1].

The column-shift alignment is the only place where the paper's odometry search
radius enters for a geometry-only descriptor, which is exactly why the drift
model changes a descriptor's *candidate set* but not its per-frame encoding
(audit finding P0 / 修改意见 Sec. 13).
"""

from __future__ import annotations

import numpy as np

from config import ScanContextConfig


class ScanContext:
    name = "ScanContext"

    def __init__(self, cfg: ScanContextConfig):
        self.cfg = cfg

    # ------------------------------------------------------------------ #
    def extract(self, frame: dict) -> np.ndarray:
        p = frame["xyz"]
        Nr, Ns = self.cfg.n_rings, self.cfg.n_sectors
        sc = np.zeros((Nr, Ns), dtype=np.float64)
        if len(p) == 0:
            return sc
        r = np.linalg.norm(p[:, :2], axis=1)
        theta = np.arctan2(p[:, 1], p[:, 0])
        ring = np.floor(r / (self.cfg.max_range / Nr)).astype(np.int64)
        sector = np.floor((theta + np.pi) / (2 * np.pi) * Ns).astype(np.int64)
        valid = (ring >= 0) & (ring < Nr) & (sector >= 0) & (sector < Ns)
        ring, sector, z = ring[valid], sector[valid], p[valid, 2]
        np.maximum.at(sc, (ring, sector), z)
        return sc

    # ------------------------------------------------------------------ #
    @staticmethod
    def _column_shift_distance(a: np.ndarray, b: np.ndarray) -> float:
        Nr, Ns = a.shape
        best = np.inf
        for shift in range(Ns):
            bb = np.roll(b, shift, axis=1)
            num = (a * bb).sum()
            den = np.linalg.norm(a) * np.linalg.norm(bb)
            if den < 1e-12:
                continue
            best = min(best, 1.0 - num / den)
        return best if np.isfinite(best) else 1.0

    def similarity(self, left: list, right: list, chunk: int = 4096) -> np.ndarray:
        """Batch column-shift cosine distance, vectorised over pairs.

        For long sequences the candidate set can contain tens of thousands of
        pairs; a per-pair Python loop over ``Ns`` shifts is far too slow, so the
        shift search is vectorised over the pair axis in blocks.
        """

        if len(left) == 0:
            return np.zeros(0)
        Nr, Ns = self.cfg.n_rings, self.cfg.n_sectors
        out = np.empty(len(left), dtype=np.float64)
        for start in range(0, len(left), chunk):
            stop = min(start + chunk, len(left))
            A = np.asarray(left[start:stop], dtype=np.float64).reshape(-1, Nr, Ns)
            B = np.asarray(right[start:stop], dtype=np.float64).reshape(-1, Nr, Ns)
            na = np.linalg.norm(A.reshape(len(A), -1), axis=1)
            nb = np.linalg.norm(B.reshape(len(B), -1), axis=1)
            den = na * nb
            best = np.full(len(A), np.inf)          # min over shifts of (1 - cos)
            for shift in range(Ns):
                Bs = np.roll(B, shift, axis=2)
                num = (A * Bs).sum(axis=(1, 2))
                with np.errstate(divide="ignore", invalid="ignore"):
                    dist = 1.0 - num / np.where(den < 1e-12, np.nan, den)
                best = np.fmin(best, dist)
            # similarity = 1 - min_shift(cosine distance) = max_shift(cosine)
            out[start:stop] = np.nan_to_num(1.0 - best, nan=0.0)
        return np.clip(out, 0.0, 1.0)

    def fit(self, comps):
        return self
