"""JSSS descriptor (paper Sec. 3.3).

    h = [ h_quad^T | h_hist^T | h_stats^T ]
    D = 4B + N_bins + 3B            (8 bands, 20 bins -> 76)

* ``h_quad``  : per-quadrant mean spectrum (4B) capturing the spatial
                distribution of reflectance w.r.t. the sensor.
* ``h_hist``  : 20-bin histogram of the spectral norm r = ||s||_2 (global
                reflectance profile of the scan).
* ``h_stats`` : per-band mean, standard deviation and skewness (3B).

Similarity (Sec. 3.3): each component is compared with a *normalised inverse
Euclidean distance* and the three similarities are combined.  The combination
weights are configurable because the paper is internally inconsistent
(Sec. 3.3 "取平均" vs Sec. 6.6.2 "权重 0.5/0.3/0.2").
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from config import JSSSConfig

COMPONENTS = ("quad", "hist", "stats")


# --------------------------------------------------------------------------- #
def _quadrant_means(xyz: np.ndarray, spectra: np.ndarray) -> np.ndarray:
    """h_quad: 4B vector of per-quadrant mean spectra."""

    B = spectra.shape[1]
    cx, cy = xyz[:, 0].mean(), xyz[:, 1].mean()
    q = (xyz[:, 0] < cx).astype(np.int64) + 2 * (xyz[:, 1] < cy).astype(np.int64)
    global_mean = spectra.mean(axis=0)
    out = np.empty(4 * B, dtype=np.float64)
    for k in range(4):
        m = q == k
        v = spectra[m].mean(axis=0) if m.any() else global_mean
        out[k * B:(k + 1) * B] = v
    return out


def _norm_histogram(spectra: np.ndarray, n_bins: int) -> np.ndarray:
    """h_hist: histogram of the spectral norm, fixed range [0, sqrt(B)]."""

    B = spectra.shape[1]
    r = np.linalg.norm(spectra, axis=1)
    hist, _ = np.histogram(r, bins=n_bins, range=(0.0, float(np.sqrt(B))))
    return hist.astype(np.float64) / max(len(r), 1)


def _band_statistics(spectra: np.ndarray) -> np.ndarray:
    """h_stats: per-band mean, std and skewness."""

    mu = spectra.mean(axis=0)
    sd = spectra.std(axis=0)
    z = (spectra - mu) / np.maximum(sd, 1e-9)
    skew = (z ** 3).mean(axis=0)
    return np.concatenate([mu, sd, skew])


# --------------------------------------------------------------------------- #
def jsss_feature(xyz: np.ndarray, spectra: np.ndarray,
                 cfg: JSSSConfig) -> Dict[str, np.ndarray]:
    """Compute the three JSSS components for one frame."""

    if len(xyz) == 0:
        B = spectra.shape[1] if spectra.ndim == 2 else cfg.n_bands
        return {"quad": np.zeros(4 * B),
                "hist": np.zeros(cfg.hist_bins),
                "stats": np.zeros(3 * B)}
    return {
        "quad": _quadrant_means(xyz, spectra),
        "hist": _norm_histogram(spectra, cfg.hist_bins),
        "stats": _band_statistics(spectra),
    }


def stack_feature(comp: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the components into h (Sec. 3.3)."""

    return np.concatenate([comp["quad"], comp["hist"], comp["stats"]])


def component_slices(cfg: JSSSConfig) -> Dict[str, slice]:
    B, nb = cfg.n_bands, cfg.hist_bins
    return {"quad": slice(0, 4 * B),
            "hist": slice(4 * B, 4 * B + nb),
            "stats": slice(4 * B + nb, 4 * B + nb + 3 * B)}


# --------------------------------------------------------------------------- #
class Normalizer:
    """Per-dimension z-score using sequence statistics (no labels involved)."""

    def __init__(self, mean: np.ndarray | None = None, std: np.ndarray | None = None):
        self.mean = mean
        self.std = std

    def fit(self, F: np.ndarray) -> "Normalizer":
        self.mean = np.asarray(F).mean(axis=0)
        std = np.asarray(F).std(axis=0)
        self.std = np.where(std < 1e-9, 1.0, std)
        return self

    def transform(self, F: np.ndarray) -> np.ndarray:
        return (np.asarray(F) - self.mean) / self.std


def rms_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Root-mean-square per-dimension Euclidean distance for stacked pairs."""

    d = a - b
    return np.sqrt((d ** 2).sum(axis=1) / d.shape[1])


def rms_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised inverse Euclidean distance: sim = 1 / (1 + d_rms) in (0, 1]."""

    return 1.0 / (1.0 + rms_distance(a, b))


class JSSS:
    """Descriptor wrapper: extraction, normalisation and weighted similarity."""

    name = "JSSS"

    def __init__(self, cfg: JSSSConfig, active: Iterable[str] = COMPONENTS):
        self.cfg = cfg
        self.active = tuple(active)
        self.slices = component_slices(cfg)
        self._norm: Dict[str, Normalizer] = {}

    # ------------------------------------------------------------------ #
    @property
    def dimension(self) -> int:
        return self.cfg.dimension

    def extract(self, frame: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return jsss_feature(frame["xyz"], frame["spectra"], self.cfg)

    def fit(self, comps: list[Dict[str, np.ndarray]]) -> "JSSS":
        for c in self.active:
            self._norm[c] = Normalizer().fit(np.stack([x[c] for x in comps]))
        return self

    def similarity(self, left: list, right: list) -> np.ndarray:
        """Weighted mean of per-component similarities for paired frames."""

        w = self.cfg.resolved_weights()
        wmap = dict(zip(COMPONENTS, w))
        total = np.zeros(len(left), dtype=np.float64)
        wsum = 0.0
        for c in self.active:
            A = self._norm[c].transform(np.stack([x[c] for x in left]))
            B = self._norm[c].transform(np.stack([x[c] for x in right]))
            total += wmap[c] * rms_similarity(A, B)
            wsum += wmap[c]
        return total / max(wsum, 1e-12)
