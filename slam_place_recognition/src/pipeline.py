"""End-to-end pipeline: dataset -> drift -> candidates -> descriptors -> metrics.

Design notes
------------
*Descriptors do not see the odometry.*  Their per-frame features are functions of
the point cloud only, so features are extracted **once** per (dataset, band
count, geometric-noise setting) and reused across every drift factor.  This is
also the concrete demonstration of the audit finding that odometry drift can
only change the *candidate set*, not the descriptor values.

A :class:`SimilarityCache` memoises pairwise similarities for the (i, j) pairs
that different drift factors request, so overlapping candidates are only
evaluated once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from config import Config
from data_io import load_dataset
from descriptors import (
    JSSS,
    M2DP,
    GlobalHistogram,
    IntensityStats,
    MeanSpectrumDescriptor,
    PureSpace,
    RadialHistogram,
    ScanContext,
    SpectralSumDescriptor,
)
from descriptors.jsss import COMPONENTS
from drift import drift_trajectory
from pose_graph import ate_metrics, build_loop_edges, optimize_pose_graph
from retrieval import build_problem, evaluate_similarity


# --------------------------------------------------------------------------- #
class SimilarityCache:
    """Memoised pairwise similarities for one descriptor."""

    def __init__(self, desc, features: List, name: str):
        self.desc = desc
        self.features = features
        self.name = name
        self._cache: Dict[Tuple[int, int], float] = {}
        self.n_computed = 0

    def get(self, pairs: np.ndarray) -> np.ndarray:
        if len(pairs) == 0:
            return np.zeros(0)
        missing = [k for k, p in enumerate(pairs)
                   if (int(p[0]), int(p[1])) not in self._cache]
        if missing:
            mp = pairs[missing]
            left = [self.features[int(i)] for i in mp[:, 0]]
            right = [self.features[int(i)] for i in mp[:, 1]]
            sims = self.desc.similarity(left, right)
            for k, s in zip(missing, np.asarray(sims)):
                self._cache[(int(pairs[k, 0]), int(pairs[k, 1]))] = float(s)
            self.n_computed += len(missing)
        return np.array([self._cache[(int(i), int(j))] for i, j in pairs])


# --------------------------------------------------------------------------- #
def build_methods(cfg: Config, active: Sequence[str] = COMPONENTS,
                  include_baselines: bool = True) -> Dict[str, object]:
    """Instantiate every descriptor of Sec. 3.3/3.4 (plus component ablations)."""

    methods: Dict[str, object] = {}
    methods["JSSS"] = JSSS(cfg.jsss, active=active)
    if include_baselines:
        methods["ScanContext"] = ScanContext(cfg.sc)
        methods["M2DP"] = M2DP(cfg.m2dp)
        methods["PureSpace_tight"] = PureSpace()
        methods["RadialHist"] = RadialHistogram(cfg.baseline)
        methods["GlobalHist_norm"] = GlobalHistogram(cfg.baseline, cfg.jsss,
                                                    source="spectral_norm")
        methods["IntensityHist"] = GlobalHistogram(cfg.baseline, cfg.jsss,
                                                   source="sum")
        methods["IntensityStats"] = IntensityStats(cfg.baseline, source="sum")
        methods["IntensitySum"] = SpectralSumDescriptor()
        methods["MeanSpectrum"] = MeanSpectrumDescriptor()
    return methods


def extract_features(ds, cfg: Config, methods: Dict[str, object],
                     verbose: bool = True) -> Dict[str, List]:
    """Extract per-frame features for every method (drift-independent)."""

    n = ds.n_frames
    acc: Dict[str, List] = {k: [] for k in methods}
    t0 = time.time()
    for i in range(n):
        frame = ds.frame(i)
        for name, desc in methods.items():
            acc[name].append(desc.extract(frame))
        if verbose and (i + 1) % max(1, n // 10) == 0:
            print(f"    features {i + 1}/{n}  ({time.time() - t0:.1f}s)", flush=True)
    for name, desc in methods.items():
        desc.fit(acc[name])
    return acc


# --------------------------------------------------------------------------- #
@dataclass
class Experiment:
    cfg: Config
    descriptors: Dict[str, object] = field(default_factory=dict)
    caches: Dict[str, SimilarityCache] = field(default_factory=dict)

    def evaluate(self, ds, features: Dict[str, List], poses_odom: np.ndarray,
                 protocol: str | None = None, names: Iterable[str] | None = None,
                 fixed_tau: float | None = None,
                 problem: Dict | None = None) -> Dict[str, Dict]:
        problem = problem or build_problem(ds, self.cfg, poses_odom, protocol)
        out: Dict[str, Dict] = {}
        for name in (names or self.descriptors.keys()):
            cache = self.caches.get(name)
            if cache is None:
                cache = SimilarityCache(self.descriptors[name], features[name], name)
                self.caches[name] = cache
            sim = cache.get(problem["pairs"])
            res = evaluate_similarity(sim, problem["positive"], self.cfg, fixed_tau)
            res["protocol"] = problem["protocol"]
            res["tau_odom"] = problem["tau_odom"]
            out[name] = res
        return out


# --------------------------------------------------------------------------- #
def prepare(cfg: Config, verbose: bool = True,
            methods: Dict[str, object] | None = None) -> Dict:
    """Load data, extract features and set up the evaluation helper."""

    if verbose:
        print(f"  loading dataset ({cfg.dataset}, seq {cfg.sequence}) ...", flush=True)
    ds = load_dataset(cfg)
    if verbose:
        print(f"  frames={ds.n_frames}  gt_loops={len(ds.gt_loops)}", flush=True)
        print("  extracting features ...", flush=True)
    methods = methods or build_methods(cfg)
    features = extract_features(ds, cfg, methods, verbose=verbose)
    ex = Experiment(cfg=cfg, descriptors=methods)
    return {"ds": ds, "features": features, "experiment": ex, "methods": methods}


def poses_for_factor(ds, cfg: Config, factor: float) -> Tuple[np.ndarray, dict]:
    """Drifted odometry poses for one noise factor (Sec. 4.2)."""

    from dataclasses import replace
    dcfg = replace(cfg.drift, factor=float(factor))
    poses_odom, delta = drift_trajectory(ds.poses_gt, dcfg)
    d = poses_odom[:, :2] - ds.poses_gt[:, :2]
    rms = float(np.sqrt((d ** 2).sum(axis=1).mean()))
    return poses_odom, {"factor": factor, "rms_m": rms, "delta": delta}


def run_trajectory_optimisation(ds, cfg: Config, poses_odom: np.ndarray,
                                sim: np.ndarray, problem: Dict,
                                tau_sim: float | None = None,
                                iterations: int | None = None,
                                loop_delta_model: str = "copy",
                                sigma_reg: float = 0.2) -> Dict:
    """Detect loops at threshold, optimise the pose graph, report ATE.

    ``loop_delta_model`` selects the loop-constraint measurement model; see
    :mod:`pose_graph`.  With the default ``"copy"`` the relaxation is provably
    a no-op, which is exactly the audit finding about the paper's Table 6.
    """

    pairs, pos = problem["pairs"], problem["positive"]
    if tau_sim is None or not np.isfinite(tau_sim):
        tau_sim = float(problem.get("tau_sim", cfg.retrieval.tau_sim))
    sel = sim >= tau_sim
    loops = pairs[sel]
    from dataclasses import replace
    pg = replace(cfg.posegraph, iterations=iterations or cfg.posegraph.iterations)
    edges = build_loop_edges(loops, poses_odom, ds.poses_gt,
                             model=loop_delta_model, sigma_reg=sigma_reg)
    before = ate_metrics(np.asarray(poses_odom)[:, :2], ds.poses_gt[:, :2])
    est = optimize_pose_graph(poses_odom, edges, pg)
    after = ate_metrics(est, ds.poses_gt[:, :2])
    return {"n_loops": int(len(loops)), "tau_sim": float(tau_sim),
            "loop_delta_model": loop_delta_model,
            "ate_before_rmse": before["ate_rmse"],
            "ate_before_mean": before["ate_mean"],
            "ate_after_rmse": after["ate_rmse"],
            "ate_after_mean": after["ate_mean"],
            "improvement_rmse_pct": (before["ate_rmse"] - after["ate_rmse"]) /
                                    before["ate_rmse"] * 100.0,
            "improvement_mean_pct": (before["ate_mean"] - after["ate_mean"]) /
                                    before["ate_mean"] * 100.0}
