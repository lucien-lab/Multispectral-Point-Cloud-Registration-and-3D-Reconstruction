"""Spectral simulation for multispectral LiDAR (paper Sec. 4.1).

Pipeline
--------
1. Map every point to a material class (from SemanticKITTI labels if available,
   otherwise from the procedural scene in :mod:`synth`).
2. Assign an 8-band reflectance spectrum from published material optics.
3. Optional enhancement over the paper's model (see 修改意见 Sec. 14):

       s_i = g_i * mu_{c(i)} + delta_{c,i} + eps_i,   eps ~ N(0, sigma_s^2)

   with the paper's model recovered by setting ``within_class_std = 0`` and
   ``gain_std = 0``.
4. Per-band normalisation to [0, 1].

The reflectance curves below are qualitative reconstructions of the values
quoted in the paper (Sec. 6.4.1(a)): asphalt 0.08-0.10, concrete 0.20-0.35,
vegetation red-edge jump from ~0.05-0.10 to ~0.45-0.55 beyond 700 nm and a
water-absorption drop near 1550 nm, metal car paint 0.25-0.45.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from config import N_BANDS_DEFAULT, WAVELENGTHS_NM, SpectralModelConfig

# --------------------------------------------------------------------------- #
# Material catalogue
# --------------------------------------------------------------------------- #
#: Material classes used by the simulation.
MATERIALS: Tuple[str, ...] = (
    "asphalt",
    "concrete",
    "metal",
    "glass",
    "vegetation",
    "soil",
    "car_paint",
    "cloth",
)
MATERIAL_INDEX: Dict[str, int] = {m: i for i, m in enumerate(MATERIALS)}

#: Representative 8-band reflectance for each material
#: (bands: 450, 550, 650, 750, 850, 950, 1050, 1550 nm).
REFLECTANCE: Dict[str, Tuple[float, ...]] = {
    # flat, very dark: 0.08-0.10 across the spectrum (paper Sec. 6.4.1(a))
    "asphalt":   (0.090, 0.092, 0.095, 0.099, 0.100, 0.098, 0.095, 0.082),
    # bright and nearly neutral: 0.20-0.35
    "concrete":  (0.280, 0.300, 0.320, 0.340, 0.342, 0.332, 0.318, 0.255),
    # specular, moderately high and slightly increasing with wavelength
    "metal":     (0.350, 0.380, 0.400, 0.420, 0.450, 0.440, 0.430, 0.380),
    # dark, low NIR (glass is transparent / low diffuse return)
    "glass":     (0.072, 0.070, 0.064, 0.052, 0.048, 0.048, 0.055, 0.045),
    # red edge: strong NIR jump, 1550 nm water-absorption dip
    "vegetation": (0.052, 0.062, 0.050, 0.452, 0.551, 0.523, 0.451, 0.180),
    # dry soil / gravel: moderately bright, reddish, less NIR jump
    "soil":      (0.150, 0.180, 0.225, 0.262, 0.275, 0.280, 0.268, 0.215),
    # car paint: 0.25-0.45, viewed as metal body in the paper
    "car_paint": (0.250, 0.280, 0.345, 0.395, 0.420, 0.412, 0.402, 0.340),
    # clothing / misc small objects
    "cloth":     (0.180, 0.205, 0.230, 0.260, 0.272, 0.265, 0.258, 0.210),
}

#: SemanticKITTI *learning-map* label id (0-19) -> material.  This is the id
#: space produced by applying ``semantic-kitti.yaml``'s learning map.
SEMANTICKITTI_TO_MATERIAL: Dict[int, str] = {
    0: "soil",          # unlabeled -> treat as ground clutter
    1: "car_paint",     # car
    2: "metal",         # bicycle
    3: "metal",         # motorcycle
    4: "car_paint",     # truck
    5: "car_paint",     # other-vehicle
    6: "cloth",         # person
    7: "cloth",         # bicyclist
    8: "cloth",         # motorcyclist
    9: "asphalt",       # road
    10: "asphalt",      # parking
    11: "concrete",     # sidewalk
    12: "soil",         # other-ground
    13: "concrete",     # building
    14: "metal",        # fence
    15: "vegetation",   # vegetation
    16: "vegetation",   # trunk
    17: "soil",         # terrain
    18: "metal",        # pole
    19: "metal",        # traffic-sign
}

#: SemanticKITTI *raw* label id -> material.  This is the id space of the files
#: distributed as ``data_odometry_labels.zip`` on semantic-kitti.org (verified:
#: sequence 05 contains ids 40/48/51/70/72/50/49/80/60/20/10...).
SEMANTICKITTI_RAW_TO_MATERIAL: Dict[int, str] = {
    0: "soil",           # unlabeled
    10: "car_paint",     # car
    11: "metal",         # bicycle
    13: "car_paint",     # bus
    15: "cloth",         # motorcyclist
    16: "metal",         # on-rails
    18: "car_paint",     # truck
    20: "car_paint",     # other-vehicle
    30: "cloth",         # person
    31: "cloth",         # bicyclist
    32: "cloth",         # motorcyclist
    40: "asphalt",       # road
    44: "asphalt",       # parking
    48: "concrete",      # sidewalk
    49: "soil",          # other-ground
    50: "concrete",      # building
    51: "metal",         # fence
    52: "concrete",      # other-structure
    60: "concrete",      # lane-marking (bright paint/concrete)
    70: "vegetation",    # vegetation
    71: "vegetation",    # trunk
    72: "soil",          # terrain
    80: "metal",         # pole
    81: "metal",         # traffic-sign
    99: "cloth",         # other-object
    252: "car_paint",    # moving-car
    253: "cloth",        # moving-bicyclist
    254: "cloth",        # moving-person
    255: "cloth",        # moving-motorcyclist
    256: "metal",        # moving-on-rails
    257: "car_paint",    # moving-bus
    258: "car_paint",    # moving-truck
    259: "car_paint",    # moving-other-vehicle
}

LABEL_CONVENTIONS = {
    "learning": SEMANTICKITTI_TO_MATERIAL,
    "raw": SEMANTICKITTI_RAW_TO_MATERIAL,
}

#: Coarse labels used by the procedural scene generator.
SCENE_TO_MATERIAL: Dict[str, str] = {
    "road": "asphalt",
    "sidewalk": "concrete",
    "building": "concrete",
    "vegetation": "vegetation",
    "soil": "soil",
    "pole": "metal",
    "car": "car_paint",
    "glass": "glass",
}


def material_spectra(scale: np.ndarray | None = None) -> np.ndarray:
    """Return the (n_materials, n_bands) matrix of reference reflectances.

    ``scale`` optionally supplies a per-material multiplicative scale so that
    different "specimens" of the same material can be generated (within-class
    variability, 修改意见 Sec. 14).
    """

    mat = np.array([REFLECTANCE[m] for m in MATERIALS], dtype=np.float64)
    if scale is not None:
        mat = mat * np.asarray(scale, dtype=np.float64)[:, None]
    return mat


class SpectralSimulator:
    """Adds 8-band spectra to a labelled point cloud."""

    def __init__(self, cfg: SpectralModelConfig, n_bands: int = N_BANDS_DEFAULT,
                 seed: int = 0, wavelengths=WAVELENGTHS_NM):
        self.cfg = cfg
        self.n_bands = n_bands
        self.wavelengths = wavelengths
        self.material_spectra = material_spectra()
        if n_bands != len(wavelengths):
            self.material_spectra = self._resample_spectra(n_bands)
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    def _resample_spectra(self, n_bands: int) -> np.ndarray:
        """Interpolate the 8-band catalogue onto ``n_bands`` wavelengths.

        Used by the band-count ablation (Sec. 5.2): 1 / 2 / 4 / 8 bands.
        For a subset we take evenly spaced wavelengths out of the 8 available.
        """
        src_nm = np.asarray(WAVELENGTHS_NM, dtype=np.float64)
        idx = np.linspace(0, len(src_nm) - 1, n_bands).round().astype(int)
        return material_spectra()[:, idx]

    # ------------------------------------------------------------------ #
    def spectra_for(self, material_ids: np.ndarray,
                    gain: np.ndarray | None = None,
                    delta: np.ndarray | None = None) -> np.ndarray:
        """Sample spectra for points with the given material ids.

        Parameters
        ----------
        material_ids : (M,) int array of indices into :data:`MATERIALS`.
        gain : optional (M,) or (M, 1) multiplicative gain g_i.
        delta : optional (M, B) additive within-class deviation.
        """
        mus = self.material_spectra[material_ids]                # (M, B)
        if gain is not None:
            gain = np.asarray(gain, dtype=np.float64).reshape(-1, 1)
            mus = mus * gain
        if delta is not None:
            mus = mus + delta
        sig = self.cfg.sigma_spectral
        if sig > 0:
            mus = mus + self.rng.normal(0.0, sig, size=mus.shape)
        if self.cfg.normalize_bands:
            lo, hi = 0.0, 1.0
            # Per-band normalisation of the *catalogue* range keeps the physics;
            # clipping the noise keeps the [0, 1] target range of Sec. 4.1.
            mus = np.clip(mus, lo, hi)
        return mus.astype(np.float32)

    # ------------------------------------------------------------------ #
    def simulate(self, material_ids: np.ndarray,
                 point_xyz: np.ndarray | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
        """Full simulation step: gain (distance) + within-class + noise.

        ``point_xyz`` is used for the distance-dependent gain, matching the
        "distance / incidence angle / calibration" factors of 修改意见 Sec. 14.
        """
        rng = rng or self.rng
        M = len(material_ids)
        gain = None
        if self.cfg.gain_std > 0:
            # per-point gain, weakly tied to distance (longer range -> lower
            # received power and larger calibration uncertainty)
            base = np.ones(M)
            if point_xyz is not None:
                r = np.linalg.norm(point_xyz[:, :2], axis=1)
                base = 1.0 - 0.15 * np.clip(r / (r.max() + 1e-9), 0.0, 1.0)
            gain = base * (1.0 + rng.normal(0.0, self.cfg.gain_std, size=M))
        delta = None
        if self.cfg.within_class_std > 0:
            # patch-correlated within-class deviation (same material, different
            # specimen / ageing / moisture)
            loc = _smooth_field_seed(point_xyz, self.cfg.patch_correlation, rng)
            delta = loc[:, None] * rng.normal(0.0, self.cfg.within_class_std,
                                              size=(M, self.n_bands))
        return self.spectra_for(material_ids, gain=gain, delta=delta)


def _smooth_field_seed(point_xyz: np.ndarray | None, length: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Cheap spatially-correlated field in [-1, 1] for within-class variation."""
    if point_xyz is None or len(point_xyz) == 0:
        return rng.uniform(-1.0, 1.0)
    xy = np.asarray(point_xyz[:, :2], dtype=np.float64)
    L = max(float(length), 1e-6)
    acc = np.zeros(len(xy))
    for _ in range(3):
        theta = rng.uniform(0, np.pi)
        phase = rng.uniform(0, 2 * np.pi)
        k = 2 * np.pi / (L * rng.uniform(0.5, 2.0))
        acc += np.sin(k * (xy[:, 0] * np.cos(theta) + xy[:, 1] * np.sin(theta)) + phase)
    acc /= 3.0
    return np.clip(acc, -1.0, 1.0)
