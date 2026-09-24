"""Pose-graph optimisation by Huber-robust relaxation (paper Sec. 4.3/4.4).

The paper describes "带Huber鲁棒损失的松弛法，迭代100次" (Sec. 4.3) and
"主要结果迭代500次，消融实验迭代100次" (Sec. 4.4); both are configurable.

Loop-edge measurement models (IMPORTANT - audit finding)
-------------------------------------------------------
A pose graph only helps if the loop constraints carry information that the
odometry chain does not already have.  The paper never mentions a registration
(ICP/NDT) step - it explicitly lists "仅做闭环检测，未做闭环验证" as a limitation
(Sec. 6.4.2(5)).  The consequences are executable and are implemented here:

``copy``        delta_ij := P_odom[j] - P_odom[i]  (i.e. reuse the drifted
                odometry difference).  The odometry trajectory is then an exact
                fixed point of the relaxation => **the ATE cannot improve**
                (improvement is exactly 0 %).  This is what a literal reading of
                the paper's method produces, so the reported +7.0 % / +1.7 %
                improvement cannot come from this scheme alone.
``coincident``  delta_ij := 0, i.e. a detected loop is treated as "the same
                place".  This is the most plausible reading of a "simple
                relaxation" without registration.
``oracle``      delta_ij := P_gt[j] - P_gt[i] (perfect registration) - an upper
                bound for what loop constraints can contribute.
``noisy``       oracle + N(0, sigma_reg^2) - a realistic registration error.

Usage
-----
    edges = build_loop_edges(pairs, poses_odom, poses_gt, model="noisy",
                             sigma_reg=0.2)
    est = optimize_pose_graph(poses_odom, edges, cfg.posegraph)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from config import PoseGraphConfig

LOOP_DELTA_MODELS = ("copy", "coincident", "oracle", "noisy")


# --------------------------------------------------------------------------- #
def build_loop_edges(pairs: Iterable[Sequence[int]], poses_odom: np.ndarray,
                     poses_gt: np.ndarray | None = None,
                     model: str = "copy", sigma_reg: float = 0.2,
                     seed: int = 0) -> List[Tuple[int, int, np.ndarray]]:
    """Resolve detected loop pairs into relative-position measurements."""

    if model not in LOOP_DELTA_MODELS:
        raise ValueError(f"unknown loop-delta model: {model}")
    poses_odom = np.asarray(poses_odom, dtype=np.float64)
    rng = np.random.default_rng(seed)
    edges: List[Tuple[int, int, np.ndarray]] = []
    for i, j in pairs:
        i, j = int(i), int(j)
        if model == "copy":
            d = poses_odom[j, :2] - poses_odom[i, :2]
        elif model == "coincident":
            d = np.zeros(2)
        else:
            if poses_gt is None:
                raise ValueError(f"model '{model}' requires ground-truth poses")
            d = np.asarray(poses_gt, dtype=np.float64)[j, :2] - \
                np.asarray(poses_gt, dtype=np.float64)[i, :2]
            if model == "noisy":
                d = d + rng.normal(0.0, sigma_reg, size=2)
        edges.append((i, j, d))
    return edges


# --------------------------------------------------------------------------- #
def _to_edges(loop_edges, poses: np.ndarray) -> List[Tuple[int, int, np.ndarray]]:
    """Accept either (i, j) pairs or (i, j, delta) triples."""

    out: List[Tuple[int, int, np.ndarray]] = []
    for e in loop_edges:
        if len(e) == 2:
            i, j = int(e[0]), int(e[1])
            out.append((i, j, poses[j, :2] - poses[i, :2]))
        else:
            out.append((int(e[0]), int(e[1]), np.asarray(e[2], dtype=np.float64)))
    return out


def optimize_pose_graph(poses_odom: np.ndarray, loop_edges,
                        cfg: PoseGraphConfig) -> np.ndarray:
    """Return optimised planar poses (n, 2).

    ``solver="jacobi"`` (default) flattens odometry + loop constraints into
    edge arrays and updates every node with vectorised scatter-adds - this is
    what makes long sequences (1400+ nodes, tens of thousands of loop edges)
    tractable.  ``"gauss_seidel"`` is the original sequential implementation,
    kept for cross-checking.
    """

    if getattr(cfg, "solver", "jacobi") == "gauss_seidel":
        return _gauss_seidel(poses_odom, loop_edges, cfg)
    return _jacobi(poses_odom, loop_edges, cfg)


# --------------------------------------------------------------------------- #
def _edge_arrays(poses_odom: np.ndarray, loop_edges, cfg: PoseGraphConfig):
    """Flatten odometry + loop constraints into (src, dst, delta, weight)."""

    poses = np.asarray(poses_odom, dtype=np.float64)[:, :2].copy()
    n = len(poses)
    src, dst, delta, w = [], [], [], []
    if n > 1:
        src.append(np.arange(n - 1))
        dst.append(np.arange(1, n))
        delta.append(poses[1:] - poses[:-1])
        w.append(np.full(n - 1, cfg.odom_weight))
    loop_list = _to_edges(loop_edges, poses)
    if loop_list:
        L = np.array([[i, j] for i, j, _ in loop_list], dtype=np.int64)
        D = np.array([d for _, _, d in loop_list], dtype=np.float64)
        keep = L[:, 0] != L[:, 1]
        src.append(L[keep, 0])
        dst.append(L[keep, 1])
        delta.append(D[keep])
        w.append(np.full(int(keep.sum()), cfg.loop_weight))
    cat = lambda xs, d=0: np.concatenate(xs) if xs else np.zeros((0,) if d == 1 else (0, 2))  # noqa: E731
    return (poses, cat(src), cat(dst), cat(delta, 2) if delta else np.zeros((0, 2)), cat(w))


def _jacobi(poses_odom: np.ndarray, loop_edges,
            cfg: PoseGraphConfig) -> np.ndarray:
    """Huber-robust Jacobi relaxation, vectorised over all edges."""

    poses, src, dst, delta, w_base = _edge_arrays(poses_odom, loop_edges, cfg)
    n = len(poses)
    if n == 0 or len(src) == 0:
        return poses
    anchor = poses[0].copy()
    for _ in range(cfg.iterations):
        # residual of every constraint under the current estimate
        r = np.linalg.norm(poses[dst] - delta - poses[src], axis=1)
        hub = np.where(r <= cfg.huber_delta, 1.0,
                       cfg.huber_delta / np.maximum(r, 1e-9))
        w = w_base * hub
        num = np.zeros((n, 2))
        den = np.zeros(n)
        np.add.at(num, src, w[:, None] * (poses[dst] - delta))
        np.add.at(den, src, w)
        np.add.at(num, dst, w[:, None] * (poses[src] + delta))
        np.add.at(den, dst, w)
        num[0] += cfg.seed_pose_weight * anchor
        den[0] += cfg.seed_pose_weight
        new = num / np.maximum(den, 1e-12)[:, None]
        shift = float(np.linalg.norm(new - poses, axis=1).max())
        poses = new
        if shift < 1e-9:
            break
    return poses


def _gauss_seidel(poses_odom: np.ndarray, loop_edges,
                  cfg: PoseGraphConfig) -> np.ndarray:
    """Sequential Huber-robust Gauss-Seidel relaxation (reference solver)."""

    poses = np.asarray(poses_odom, dtype=np.float64)[:, :2].copy()
    n = len(poses)
    if n == 0:
        return poses

    adj: List[List[Tuple[int, np.ndarray, float]]] = [[] for _ in range(n)]
    for i in range(n - 1):
        d = poses[i + 1] - poses[i]
        adj[i].append((i + 1, d, cfg.odom_weight))
        adj[i + 1].append((i, -d, cfg.odom_weight))
    for i, j, d in _to_edges(loop_edges, poses):
        if i == j:
            continue
        adj[i].append((j, d, cfg.loop_weight))
        adj[j].append((i, -d, cfg.loop_weight))

    anchor = poses[0].copy()
    for _ in range(cfg.iterations):
        max_shift = 0.0
        for i in range(n):
            cand, wts = [], []
            for other, delta, w in adj[i]:
                cand.append(poses[other] - delta)
                wts.append(w)
            if i == 0:
                cand.append(anchor)
                wts.append(cfg.seed_pose_weight)
            cand = np.asarray(cand)
            wts = np.asarray(wts)
            r = np.linalg.norm(cand - poses[i], axis=1)
            hub = np.where(r <= cfg.huber_delta, 1.0,
                           cfg.huber_delta / np.maximum(r, 1e-9))
            w = wts * hub
            new = (w[:, None] * cand).sum(axis=0) / w.sum()
            max_shift = max(max_shift, float(np.linalg.norm(new - poses[i])))
            poses[i] = new
        if max_shift < 1e-9:
            break
    return poses


# --------------------------------------------------------------------------- #
def ate_metrics(poses_est: np.ndarray, poses_gt: np.ndarray) -> dict:
    """ATE RMSE and mean error (Table 6)."""

    e = np.asarray(poses_est)[:, :2] - np.asarray(poses_gt)[:, :2]
    d = np.linalg.norm(e, axis=1)
    return {"ate_rmse": float(np.sqrt((d ** 2).mean())),
            "ate_mean": float(d.mean()),
            "n": int(len(d))}


def improvement_rate(ate_before: float, ate_after: float) -> float:
    return float((ate_before - ate_after) / ate_before * 100.0) if ate_before else 0.0
