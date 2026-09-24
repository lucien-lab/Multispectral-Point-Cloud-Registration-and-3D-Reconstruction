"""Baseline descriptors (paper Sec. 3.4).

Implemented
-----------
* ``PureSpace``        - accept every candidate inside tau_odom (tight/wide).
* ``RadialHistogram``  - 10-bin radial-distance histogram ("几何描述子").
* ``GlobalHistogram``  - 20-bin histogram of the spectral norm.  NOTE (audit):
                         Sec. 3.4 defines exactly this, which is identical to
                         JSSS's ``h_hist``; ``source="sum"`` switches to the
                         summed-intensity histogram instead.
* ``IntensityStats``   - 7-D statistics of the scalar intensity
                         (mean, std, max, min, median, skew, kurtosis).
* ``SpectralSumDescriptor`` - the "8 波段求和强度" baseline of Table 2/5.

All non-trivial baselines reuse the same normalised inverse Euclidean distance
as JSSS so that the comparison is not confounded by the metric.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats as sps

from config import BaselineConfig, JSSSConfig
from descriptors.jsss import Normalizer, rms_similarity


class _ScalarDescriptor:
    """Common behaviour for descriptors producing one vector per frame."""

    name = "scalar"

    def __init__(self):
        self._norm: Normalizer | None = None

    def extract(self, frame: Dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def fit(self, comps):
        self._norm = Normalizer().fit(np.stack(comps))
        return self

    def similarity(self, left, right) -> np.ndarray:
        A = self._norm.transform(np.stack(left))
        B = self._norm.transform(np.stack(right))
        return rms_similarity(A, B)


class PureSpace(_ScalarDescriptor):
    """No filtering at all: every candidate pair is accepted (sim = 1)."""

    name = "PureSpace"

    def extract(self, frame: Dict[str, np.ndarray]) -> np.ndarray:
        return np.zeros(1)

    def fit(self, comps):
        return self

    def similarity(self, left, right) -> np.ndarray:
        return np.ones(len(left))


class RadialHistogram(_ScalarDescriptor):
    """10-bin radial-distance histogram (paper's "几何描述子")."""

    name = "RadialHist"

    def __init__(self, cfg: BaselineConfig, max_range: float = 40.0):
        super().__init__()
        self.cfg = cfg
        self.max_range = max_range

    def extract(self, frame: Dict[str, np.ndarray]) -> np.ndarray:
        r = np.linalg.norm(frame["xyz"][:, :2], axis=1)
        h, _ = np.histogram(r, bins=self.cfg.radial_bins, range=(0.0, self.max_range))
        return h.astype(np.float64) / max(len(r), 1)


class GlobalHistogram(_ScalarDescriptor):
    """20-bin histogram of the spectral norm (== JSSS ``h_hist``) or of the sum."""

    name = "GlobalHist"

    def __init__(self, cfg: BaselineConfig, jsss_cfg: JSSSConfig,
                 source: str | None = None):
        super().__init__()
        self.cfg = cfg
        self.jsss_cfg = jsss_cfg
        self.source = source or cfg.global_hist_source

    def extract(self, frame: Dict[str, np.ndarray]) -> np.ndarray:
        s = frame["spectra"]
        if self.source == "sum":
            r = s.sum(axis=1)
            hi = float(s.shape[1])
        else:
            r = np.linalg.norm(s, axis=1)
            hi = float(np.sqrt(s.shape[1]))
        h, _ = np.histogram(r, bins=self.cfg.hist_bins, range=(0.0, hi))
        return h.astype(np.float64) / max(len(r), 1)


class IntensityStats(_ScalarDescriptor):
    """7-D statistics of the scalar intensity (mean/std/max/min/median/skew/kurt)."""

    name = "IntensityStats"

    def __init__(self, cfg: BaselineConfig, source: str = "sum"):
        super().__init__()
        self.cfg = cfg
        self.source = source

    def _scalar(self, frame: Dict[str, np.ndarray]) -> np.ndarray:
        if self.source == "native" and "intensity_raw" in frame:
            return frame["intensity_raw"].astype(np.float64)
        return frame["spectra"].sum(axis=1).astype(np.float64)

    def extract(self, frame: Dict[str, np.ndarray]) -> np.ndarray:
        v = self._scalar(frame)
        return np.array([v.mean(), v.std(), v.max(), v.min(), np.median(v),
                         float(sps.skew(v)), float(sps.kurtosis(v))])


class SpectralSumDescriptor(_ScalarDescriptor):
    """Scalar sum over bands -> 1-D "single channel intensity" baseline (Table 2)."""

    name = "IntensitySum"

    def extract(self, frame: Dict[str, np.ndarray]) -> np.ndarray:
        return np.array([float(frame["spectra"].sum(axis=1).mean())])


class MeanSpectrumDescriptor(_ScalarDescriptor):
    """Mean spectrum over the whole scan (B-D vector; used for diagnostics)."""

    name = "MeanSpectrum"

    def extract(self, frame: Dict[str, np.ndarray]) -> np.ndarray:
        return frame["spectra"].mean(axis=0).astype(np.float64)
