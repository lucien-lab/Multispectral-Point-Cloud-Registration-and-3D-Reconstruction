"""Odometry drift model (paper Sec. 4.2).

    delta_x(t) = sum_k a_k * sin(2*pi*f_k*t + phi_k)

with the amplitude scaled by the noise factor {1.0x, 1.5x, 2.0x}.  The paper
reports ~7 m RMS at 1.0x (Table 6: 7.257 m odometry ATE RMSE) and 14.514 m at
2.0x, i.e. the ATE scales linearly with the factor.

IMPORTANT (audit finding): this model perturbs *poses only*.  It does **not**
perturb the point cloud XYZ, therefore the claim in Sec. 5.5/6.2 that M2DP's
"PCA principal axis shifts because of the noise" can only hold if the drift is
additionally applied to the scans (see :func:`apply_drift_to_points`).
"""

from __future__ import annotations

import numpy as np

from config import DriftConfig


def _base_waveform(n: int, frequencies, rng: np.random.Generator):
    """Unit-amplitude multi-sine waveform, shape (n, 3) for x, y, yaw."""

    t = np.arange(n, dtype=np.float64)
    # frequencies are given in cycles per metre of travel in the original
    # implementation; treat them as cycles per frame for a stable realisation.
    out = np.zeros((n, 3), dtype=np.float64)
    for axis in range(3):
        for f in frequencies:
            phase = rng.uniform(0.0, 2 * np.pi)
            out[:, axis] += np.sin(2 * np.pi * f * t + phase)
    # normalise each axis so that RMS == 1 -> amplitude has a physical meaning
    rms = np.sqrt((out ** 2).mean(axis=0, keepdims=True))
    return out / np.maximum(rms, 1e-9)


def drift_trajectory(poses_gt: np.ndarray, cfg: DriftConfig):
    """Return (poses_odom, drift_delta) given ground-truth poses (n, 3+).

    Poses are ``[x, y, yaw]`` (extra columns such as z are preserved).
    """

    poses_gt = np.asarray(poses_gt, dtype=np.float64)
    n = len(poses_gt)
    rng = np.random.default_rng(cfg.seed)
    wave = _base_waveform(n, cfg.frequencies, rng)

    amp = cfg.amplitude_m
    if cfg.target_rms_m is not None:
        # scale so that the planar positional error has the requested RMS
        pos_rms = np.sqrt((wave[:, :2] ** 2).sum(axis=1).mean())
        amp = cfg.target_rms_m / max(pos_rms, 1e-9)

    delta = np.zeros_like(poses_gt)
    delta[:, :2] = amp * cfg.factor * wave[:, :2]
    if poses_gt.shape[1] >= 3:
        # yaw drift: bounded, smaller angular amplitude (rad)
        yaw_rms = 0.01 * cfg.factor * amp / max(amp, 1e-9)
        delta[:, 2] = yaw_rms * wave[:, 2]

    poses_odom = poses_gt.copy()
    poses_odom[:, :2] += delta[:, :2]
    if poses_gt.shape[1] >= 3:
        poses_odom[:, 2] += delta[:, 2]
    return poses_odom, delta


def apply_drift_to_points(points: np.ndarray, delta_xyz: np.ndarray,
                          weights: np.ndarray | None = None) -> np.ndarray:
    """Optional: bake the (interpolated) drift into the scan coordinates.

    This is provided *only* to let the user test the causal explanation given
    for M2DP in Sec. 5.5/6.2.  It is NOT part of the paper's described pipeline.
    """

    p = np.asarray(points, dtype=np.float64).copy()
    if weights is None:
        p[:, :3] += np.asarray(delta_xyz[:3], dtype=np.float64)[None, :]
    else:
        p[:, :3] += np.asarray(weights, dtype=np.float64)[:, None] * \
            np.asarray(delta_xyz[:3], dtype=np.float64)[None, :]
    return p


def odometry_ate_rmse(poses_est: np.ndarray, poses_gt: np.ndarray) -> float:
    """RMSE of the planar position error (Table 6 'odometry ATE')."""

    d = np.asarray(poses_est)[:, :2] - np.asarray(poses_gt)[:, :2]
    return float(np.sqrt((d ** 2).sum(axis=1).mean()))
