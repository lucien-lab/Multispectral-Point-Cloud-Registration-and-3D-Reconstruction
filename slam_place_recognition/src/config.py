"""Global configuration for the JSSS (Joint Spatial-Spectral Signatures) pipeline.

Every numeric constant that the paper mentions is exposed here so that the
reported tables can be reproduced (or deliberately re-run under a corrected
protocol, see `retrieval.EvalProtocol`).

References to the paper (JSSS_LiDAR_SLAM_v3(修改).docx):
  * Sec. 3.3  -> descriptor definition, D = 4B + N_bins + 3B
  * Sec. 4.1  -> 8 bands, sigma_spectral = 0.02, 1400 frames, ~120k points/frame
  * Sec. 4.2  -> bounded sinusoid odometry drift, factors {1.0, 1.5, 2.0}x
  * Sec. 4.3  -> TP definition (GT distance < 8 m), 40 similarity thresholds
  * Sec. 4.4  -> pose-graph relaxation, 500 iters (main) / 100 iters (ablation)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
DAT_DIR = os.path.join(PROJECT_DIR, "dat")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# --------------------------------------------------------------------------- #
# Spectral configuration
# --------------------------------------------------------------------------- #
# Wavelengths used in the paper (Sec. 4.1/4.2): [450 ... 1550] nm.
WAVELENGTHS_NM: Tuple[int, ...] = (450, 550, 650, 750, 850, 950, 1050, 1550)
N_BANDS_DEFAULT = 8

# Measurement noise on the per-point spectral vector (Sec. 4.1: sigma = 0.02).
SIGMA_SPECTRAL_DEFAULT = 0.02

# Reflectance [0, 1] target range after per-band normalisation (Sec. 4.1).
REFLECTANCE_RANGE = (0.0, 1.0)


@dataclass
class SpectralModelConfig:
    """Sec. 4.1 spectral simulation.

    The paper describes:  s_i = mu_{c(i)} + eps_i,  eps ~ N(0, sigma^2).

    Two extra terms are implemented because the review (修改意见 Sec. 14/15)
    identified the lack of *within-class* variability as a validity threat:

        s_i = g_i * mu_{c(i)} + delta_{c,i} + eps_i

    with ``g_i`` a smooth multiplicative gain (distance / incidence / gain
    drift) and ``delta`` a per-patch within-class deviation.  Set
    ``within_class_std = 0`` and ``gain_std = 0`` to recover the paper's model.
    """

    sigma_spectral: float = SIGMA_SPECTRAL_DEFAULT
    within_class_std: float = 0.0     # delta: same material, different place/specimen
    gain_std: float = 0.0             # g_i: distance / incidence / calibration gain
    patch_correlation: float = 40.0   # metres, spatial correlation length of delta
    normalize_bands: bool = True     # per-band min-max normalisation to [0, 1]


# --------------------------------------------------------------------------- #
# Descriptor configuration
# --------------------------------------------------------------------------- #
@dataclass
class JSSSConfig:
    n_bands: int = N_BANDS_DEFAULT
    hist_bins: int = 20            # N_bins (Sec. 3.3)
    quad_bins: int = 4             # 4 quadrants (Sec. 3.3)
    # Similarity weights.  NOTE (audit finding): Sec. 3.3 says "取平均" (equal
    # average) while Sec. 6.6.2 says "固定相似度权重(0.5/0.3/0.2)".
    # Both are selectable; `weights` wins unless "auto" is requested.
    weight_mode: str = "equal"     # {"equal", "paper662"}
    weights: Tuple[float, float, float] | None = None

    def resolved_weights(self) -> Tuple[float, float, float]:
        if self.weights is not None:
            w = tuple(float(x) for x in self.weights)
        elif self.weight_mode == "paper662":
            w = (0.5, 0.3, 0.2)
        else:
            w = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        s = sum(w)
        return tuple(x / s for x in w)

    @property
    def dimension(self) -> int:
        return 4 * self.n_bands + self.hist_bins + 3 * self.n_bands

    def dimension_without_quad(self) -> int:
        return self.hist_bins + 3 * self.n_bands


# --------------------------------------------------------------------------- #
# ScanContext / M2DP (baselines, Sec. 3.4)
# --------------------------------------------------------------------------- #
@dataclass
class ScanContextConfig:
    n_rings: int = 15       # Nr (Sec. 3.4)
    n_sectors: int = 40     # Ns (Sec. 3.4)
    max_range: float = 40.0


@dataclass
class M2DPConfig:
    n_azimuth: int = 4      # 4 projections (0/45/90/135 deg)
    n_radial: int = 8
    n_angular: int = 16
    max_range: float = 40.0
    min_points: int = 50


@dataclass
class BaselineConfig:
    radial_bins: int = 10       # "几何描述子（径向距离直方图）"
    hist_bins: int = 20         # "全局直方图" (== JSSS h_hist, see audit note)
    stats_dims: int = 7         # mean, std, max, min, median, skew, kurtosis
    # "全局直方图" is defined in Sec. 3.4 as a histogram of the *spectral norm*,
    # i.e. literally JSSS's h_hist.  Set to "sum" to use the summed intensity.
    global_hist_source: str = "spectral_norm"   # {"spectral_norm", "sum"}


# --------------------------------------------------------------------------- #
# Drift / robustness configuration (Sec. 4.2)
# --------------------------------------------------------------------------- #
@dataclass
class DriftConfig:
    frequencies: Tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)
    factor: float = 1.0                     # 1.0x / 1.5x / 2.0x
    amplitude_m: float = 1.0                # base amplitude before scaling
    target_rms_m: float | None = 7.257      # Table 6 reports 7.257 m at 1.0x
    seed: int = 20240517


@dataclass
class GeometricNoiseConfig:
    """Point-cloud (XYZ) perturbation - absent in the paper (audit finding)."""

    sigma_xyz: float = 0.0        # metres, 0 / 0.02 / 0.05 / 0.10
    drop_rate: float = 0.0        # fraction of points dropped, 0 / 0.1 / 0.2 / 0.3


# --------------------------------------------------------------------------- #
# Retrieval / evaluation configuration
# --------------------------------------------------------------------------- #
@dataclass
class RetrievalConfig:
    tau_odom: float = 5.0           # odometry search radius (Sec. 3.3/5.1)
    tau_sim: float = 0.65           # similarity threshold (Tables 1-4)
    delta_min_frames: int = 10      # |i-j| > Delta_min (Sec. 3.1)
    n_thresholds: int = 40          # 40 uniform thresholds (Sec. 4.3)
    tp_distance_m: float = 8.0      # TP if GT distance < 8 m (Sec. 4.3)
    # Evaluation protocol.  The paper mixes three of them (audit finding P0-6):
    #   "paper_loose" - Tables 1-4: no temporal gap, TP by GT distance only
    #   "paper_pr"    - Tables 5-6: tau_odom = 8 m and gap >= 10 frames
    #   "fixed"       - recommended: explicit ground-truth loop set + gap
    protocol: str = "fixed"
    # Mode for Tables 1-4 style comparison: "fixed_threshold" uses tau_sim as
    # given; "per_method_opt" reports every method at its own F1-max threshold.
    threshold_mode: str = "per_method_opt"

    def tau_odom_for(self, protocol: str | None = None) -> float:
        p = protocol or self.protocol
        return 8.0 if p == "paper_pr" else self.tau_odom


# --------------------------------------------------------------------------- #
# Pose-graph optimisation (Sec. 4.3/4.4)
# --------------------------------------------------------------------------- #
@dataclass
class PoseGraphConfig:
    iterations: int = 500          # main results
    iterations_ablation: int = 100
    huber_delta: float = 1.0       # metres
    odom_weight: float = 1.0
    loop_weight: float = 1.0
    seed_pose_weight: float = 100.0   # anchors the gauge
    # "jacobi" = vectorised (fast, default); "gauss_seidel" = sequential
    # reference implementation.  Both use the same Huber-robust cost.
    solver: str = "jacobi"


# --------------------------------------------------------------------------- #
# Synthetic dataset configuration (used when real KITTI is unavailable)
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticConfig:
    n_frames: int = 600
    points_per_frame: int = 20000
    range_m: float = 40.0            # LiDAR range
    point_sigma_xyz: float = 0.02    # sensor geometric noise
    map_resolution: float = 0.5
    n_laps: float = 2.0              # 2nd lap creates genuine revisits
    lateral_jitter_m: float = 1.0    # small offset between laps
    # 0.0 => homogeneous composition all around the route (only class-level
    # differences). 1.0 => strong local composition change (place identity).
    composition_variation: float = 1.0
    revisit_distance_m: float = 2.0  # GT loop if two frames closer than this
    seed: int = 7


# --------------------------------------------------------------------------- #
# Top-level config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    dataset: str = "synthetic"       # {"synthetic", "kitti"}
    sequence: str = "05"
    # SemanticKITTI id space of the label files: "auto" detects it by reading
    # the first label file (raw ids 40/70/... vs learning-map ids 0-19).
    label_convention: str = "auto"    # {"auto", "raw", "learning"}
    max_frames: int | None = None    # subsample long sequences (1400 in paper)
    jsss: JSSSConfig = field(default_factory=JSSSConfig)
    spectral: SpectralModelConfig = field(default_factory=SpectralModelConfig)
    sc: ScanContextConfig = field(default_factory=ScanContextConfig)
    m2dp: M2DPConfig = field(default_factory=M2DPConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    geom: GeometricNoiseConfig = field(default_factory=GeometricNoiseConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    posegraph: PoseGraphConfig = field(default_factory=PoseGraphConfig)
    synth: SyntheticConfig = field(default_factory=SyntheticConfig)
    out_dir: str = RESULTS_DIR


def default_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
