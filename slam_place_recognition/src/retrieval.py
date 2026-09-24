"""Loop-closure retrieval, evaluation protocols and PR metrics (paper Sec. 4.3).

The paper mixes three evaluation protocols, which is one of the P0 findings of
the multi-model review.  All three are implemented and selectable:

===============  ==============================  ==============================
protocol         candidate rule                  positive rule
===============  ==============================  ==============================
``paper_loose``  ||P_i - P_j|| <= tau_odom       GT distance <= 8 m  (Sec. 4.3)
                 (no temporal gap)               -> Tables 1-4
``paper_pr``     ||P_i - P_j|| <= 8 m,           GT distance <= 8 m  -> Tables 5-6
                 |i-j| >= 10
``fixed``        ||P_i - P_j|| <= tau_odom,      explicit GT loop set
                 |i-j| > Delta_min               (distance + gap)   -> recommended
===============  ==============================  ==============================

Metrics returned for every method: ``n_cand``, ``n_pos``, ``tp``, ``fp``,
``fn``, ``precision``, ``recall``, ``f1``, ``f1_max``, ``auc``, ``ap``,
``best_tau`` and the *accept-all* upper bound F1 (used by the audit script).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from config import Config


# --------------------------------------------------------------------------- #
def candidate_pairs(poses_odom: np.ndarray, tau_odom: float,
                    delta_min: int, chunk: int = 512) -> np.ndarray:
    """All pairs (i<j) within ``tau_odom``; temporal gap already applied."""

    xy = np.asarray(poses_odom)[:, :2]
    n = len(xy)
    out_i, out_j = [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = xy[start:stop]
        d = np.linalg.norm(block[:, None, :] - xy[None, :, :], axis=2)
        rows, cols = np.nonzero(d <= tau_odom)
        # `rows` indexes the block, `cols` is already an absolute frame index
        ri = rows + start
        cj = cols
        keep = (cj - ri) > delta_min      # implies j > i, so no duplicates
        out_i.append(ri[keep])
        out_j.append(cj[keep])
    if not out_i:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.stack([np.concatenate(out_i), np.concatenate(out_j)], axis=1)
    pairs = np.unique(pairs, axis=0)
    return pairs[pairs[:, 0] < pairs[:, 1]]


# --------------------------------------------------------------------------- #
def build_problem(ds, cfg: Config, poses_odom: np.ndarray,
                  protocol: str | None = None,
                  tau_odom_override: float | None = None) -> Dict:
    """Assemble candidate pairs + positive labels for one protocol."""

    proto = protocol or cfg.retrieval.protocol
    rc = cfg.retrieval
    if proto == "paper_loose":
        tau = rc.tau_odom
        gap = 0
    elif proto == "paper_pr":
        tau = 8.0
        gap = rc.delta_min_frames
    else:
        tau = rc.tau_odom
        gap = rc.delta_min_frames
    if tau_odom_override is not None:
        tau = float(tau_odom_override)

    pairs = candidate_pairs(poses_odom, tau, gap)

    xy_gt = ds.poses_gt[:, :2]
    xy_odom = np.asarray(poses_odom)[:, :2]
    if len(pairs):
        d_gt = np.linalg.norm(xy_gt[pairs[:, 0]] - xy_gt[pairs[:, 1]], axis=1)
        d_odom = np.linalg.norm(xy_odom[pairs[:, 0]] - xy_odom[pairs[:, 1]], axis=1)
    else:
        d_gt = np.zeros(0)
        d_odom = np.zeros(0)

    if proto == "fixed":
        gt_set = set(map(tuple, np.asarray(ds.gt_loops).tolist()))
        pos = np.array([tuple(p) in gt_set for p in pairs.tolist()], dtype=bool) \
            if len(pairs) else np.zeros(0, bool)
    else:
        pos = d_gt <= rc.tp_distance_m

    return {"protocol": proto, "tau_odom": tau, "gap": gap,
            "pairs": pairs, "d_gt": d_gt, "d_odom": d_odom, "positive": pos}


# --------------------------------------------------------------------------- #
@dataclass
class CurveResult:
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    accept_all_f1: float
    n_cand: int
    n_pos: int


def pr_curve(similarity: np.ndarray, positive: np.ndarray,
             n_thresholds: int = 40,
             thresholds: Sequence[float] | None = None) -> CurveResult:
    """Precision/recall/F1 versus the similarity threshold (Sec. 4.3)."""

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, n_thresholds)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    n_pos = int(positive.sum())
    n_cand = len(positive)
    P = np.zeros_like(thresholds)
    R = np.zeros_like(thresholds)
    F = np.zeros_like(thresholds)
    for k, t in enumerate(thresholds):
        sel = similarity >= t
        tp = int((sel & positive).sum())
        fp = int(sel.sum()) - tp
        p = tp / (tp + fp) if (tp + fp) else 1.0
        r = tp / n_pos if n_pos else 1.0
        P[k], R[k] = p, r
        F[k] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    aa = (n_pos / n_cand) if n_cand else 0.0
    aa_f1 = 2 * aa * 1.0 / (aa + 1.0) if aa > 0 else 0.0
    return CurveResult(thresholds, P, R, F, aa_f1, n_cand, n_pos)


def pr_auc(recall: np.ndarray, precision: np.ndarray) -> float:
    """Area under the PR curve (trapezoidal), recall ascending."""

    order = np.argsort(recall)
    r, p = recall[order], precision[order]
    if len(r) < 2:
        return float(p.mean()) if len(p) else 0.0
    return float(np.trapezoid(p, r) if hasattr(np, "trapezoid")
                 else np.trapz(p, r))


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    order = np.argsort(recall)
    r, p = recall[order], precision[order]
    dr = np.diff(np.concatenate([[0.0], r]))
    return float((p * dr).sum())


# --------------------------------------------------------------------------- #
def evaluate_similarity(sim: np.ndarray, pos: np.ndarray, cfg: Config,
                        fixed_tau: float | None = None) -> Dict:
    """Metrics for a precomputed similarity vector (caching-friendly entry point)."""

    rc = cfg.retrieval
    if len(pos) == 0:
        return {"n_cand": 0, "n_pos": 0, "tp": 0, "fp": 0, "fn": 0,
                "precision": float("nan"), "recall": float("nan"),
                "f1": 0.0, "f1_max": 0.0, "auc": 0.0, "ap": 0.0,
                "best_tau": float("nan"), "accept_all_f1": 0.0,
                "closed_loops": 0, "similarity": np.asarray(sim),
                "curve": None}

    curve = pr_curve(sim, pos, rc.n_thresholds)
    k_best = int(np.argmax(curve.f1)) if len(curve.f1) else 0

    tau = fixed_tau if fixed_tau is not None else float(curve.thresholds[k_best])
    sel = sim >= tau
    tp = int((sel & pos).sum())
    fp = int(sel.sum()) - tp
    fn = int(pos.sum()) - tp
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / int(pos.sum()) if pos.sum() else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
          else 0.0)

    return {
        "n_cand": curve.n_cand, "n_pos": curve.n_pos,
        "closed_loops": int(sel.sum()),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "f1_max": float(curve.f1.max()) if len(curve.f1) else 0.0,
        "auc": pr_auc(curve.recall, curve.precision),
        "ap": average_precision(curve.recall, curve.precision),
        "best_tau": float(curve.thresholds[k_best]),
        "accept_all_f1": curve.accept_all_f1,
        "similarity": np.asarray(sim), "curve": curve,
    }


def evaluate_method(desc, features: List, problem: Dict, cfg: Config,
                    fixed_tau: float | None = None) -> Dict:
    """Run one descriptor on one problem (compatibility wrapper)."""

    pairs = problem["pairs"]
    if len(pairs) == 0:
        return evaluate_similarity(np.zeros(0), np.zeros(0, bool), cfg, fixed_tau)
    left = [features[i] for i in pairs[:, 0]]
    right = [features[i] for i in pairs[:, 1]]
    sim = desc.similarity(left, right)
    return evaluate_similarity(sim, problem["positive"], cfg, fixed_tau)
