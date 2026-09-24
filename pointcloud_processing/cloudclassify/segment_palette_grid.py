#!/usr/bin/env python3
"""Segment the 24-square colour palette in a CloudCompare ASCII point cloud."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, RANSACRegressor
from scipy.optimize import linear_sum_assignment


ROWS = 6
COLS = 4


@dataclass(frozen=True)
class SquareGrid:
    """A row-major 4-by-6 grid of equally sized squares in the X-Z plane."""

    x0: float
    z_top: float
    side: float
    pitch_x: float
    pitch_z: float

    def cell_bounds(self, row: int, col: int) -> tuple[float, float, float, float]:
        if not 0 <= row < ROWS or not 0 <= col < COLS:
            raise ValueError(f"cell index out of bounds: row={row}, col={col}")
        xmin = self.x0 + col * self.pitch_x
        xmax = xmin + self.side
        zmax = self.z_top - row * self.pitch_z
        zmin = zmax - self.side
        return xmin, xmax, zmin, zmax


@dataclass(frozen=True)
class FrontPlane:
    """Front colour-card surface represented as y = ax + bz + c."""

    x_coefficient: float
    z_coefficient: float
    intercept: float
    tolerance: float

    def predict(self, x: float | np.ndarray, z: float | np.ndarray) -> float | np.ndarray:
        return self.x_coefficient * x + self.z_coefficient * z + self.intercept


def load_palette(path: str | Path) -> tuple[str, np.ndarray]:
    """Read an eight-column CloudCompare ASCII point cloud."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        point_count = int(handle.readline().strip())
    points = np.loadtxt(path, comments="//", skiprows=2)
    if points.shape != (point_count, 8):
        raise ValueError(
            f"expected ({point_count}, 8) data values, found {points.shape}"
        )
    return header, points


def grid_label(row: int, col: int) -> int:
    """Return the 1-based row-major label for a 4-by-6 grid cell."""
    if not 0 <= row < ROWS or not 0 <= col < COLS:
        raise ValueError(f"cell index out of bounds: row={row}, col={col}")
    return row * COLS + col + 1


def point_to_grid_cell(
    grid: SquareGrid, x: float, z: float
) -> tuple[int, int] | None:
    """Return the containing cell, or None for a gap or outside point."""
    for row in range(ROWS):
        for col in range(COLS):
            xmin, xmax, zmin, zmax = grid.cell_bounds(row, col)
            if xmin <= x <= xmax and zmin <= z <= zmax:
                return row, col
    return None


def _colour_candidate_mask(points: np.ndarray) -> np.ndarray:
    rgb = points[:, 3:6]
    brightness = rgb.mean(axis=1)
    saturation = rgb.max(axis=1) - rgb.min(axis=1)
    return (brightness > 45.0) | (saturation > 35.0)


def _cluster_centres(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) < count:
        raise ValueError(f"need at least {count} points to fit the grid")
    centres = KMeans(n_clusters=count, n_init=50, random_state=0).fit(
        values.reshape(-1, 1)
    ).cluster_centers_.ravel()
    return np.sort(centres)


def fit_palette_grid(points: np.ndarray) -> tuple[SquareGrid, FrontPlane]:
    """Fit a 4-by-6, equal-square grid to the palette front surface."""
    colour_points = points[_colour_candidate_mask(points)]
    if len(colour_points) == 0:
        raise ValueError("no coloured points available to locate the palette")

    model = RANSACRegressor(
        LinearRegression(), residual_threshold=0.0015, random_state=0, max_trials=1000
    ).fit(colour_points[:, [0, 2]], colour_points[:, 1])
    estimator = model.estimator_
    residuals = np.abs(
        colour_points[:, 1] - model.predict(colour_points[:, [0, 2]])
    )
    inlier_residuals = residuals[model.inlier_mask_]
    tolerance = max(0.0015, float(np.percentile(inlier_residuals, 99.5) * 1.25))
    plane = FrontPlane(
        x_coefficient=float(estimator.coef_[0]),
        z_coefficient=float(estimator.coef_[1]),
        intercept=float(estimator.intercept_),
        tolerance=tolerance,
    )
    front_colour_points = colour_points[
        np.abs(colour_points[:, 1] - plane.predict(colour_points[:, 0], colour_points[:, 2]))
        <= plane.tolerance
    ]
    x_centres = _cluster_centres(front_colour_points[:, 0], COLS)
    z_centres = _cluster_centres(front_colour_points[:, 2], ROWS)
    pitch_x = float(np.median(np.diff(x_centres)))
    pitch_z = float(np.median(np.diff(z_centres)))
    side = 0.8 * min(pitch_x, pitch_z)
    if side <= 0:
        raise ValueError("failed to obtain a positive square side length")
    return (
        SquareGrid(
            x0=float(x_centres[0] - side / 2),
            z_top=float(z_centres[-1] + side / 2),
            side=float(side),
            pitch_x=pitch_x,
            pitch_z=pitch_z,
        ),
        plane,
    )


def assign_labels(
    points: np.ndarray, grid: SquareGrid, plane: FrontPlane
) -> np.ndarray:
    """Assign 0 to background and 1..24 to points inside the front-face squares."""
    labels = np.zeros(len(points), dtype=np.int64)
    front_mask = np.abs(points[:, 1] - plane.predict(points[:, 0], points[:, 2])) <= plane.tolerance
    for index in np.flatnonzero(front_mask):
        cell = point_to_grid_cell(grid, points[index, 0], points[index, 2])
        if cell is not None:
            labels[index] = grid_label(*cell)
    return labels


def ideal_sample_positions(
    grid: SquareGrid, row: int, col: int, size: int = 7
) -> np.ndarray:
    """Return the centre positions of a size-by-size lattice in one square."""
    if size < 1:
        raise ValueError("sample-grid size must be positive")
    xmin, xmax, zmin, zmax = grid.cell_bounds(row, col)
    offsets = (np.arange(size) + 0.5) / size
    x_positions = xmin + (xmax - xmin) * offsets
    z_positions = zmax - (zmax - zmin) * offsets
    x_mesh, z_mesh = np.meshgrid(x_positions, z_positions)
    return np.column_stack((x_mesh.ravel(), z_mesh.ravel()))


def assign_regular_labels(
    points: np.ndarray, grid: SquareGrid, size: int = 7
) -> np.ndarray:
    """Label a maximum of size² original points in each square grid cell."""
    labels = np.zeros(len(points), dtype=np.int64)
    for row in range(ROWS):
        for col in range(COLS):
            candidate_indices = np.array(
                [
                    index
                    for index, point in enumerate(points)
                    if point_to_grid_cell(grid, point[0], point[2]) == (row, col)
                ],
                dtype=int,
            )
            if len(candidate_indices) == 0:
                continue
            target_positions = ideal_sample_positions(grid, row, col, size)
            if len(candidate_indices) <= len(target_positions):
                chosen_indices = candidate_indices
            else:
                candidate_positions = points[candidate_indices][:, [0, 2]]
                distance_cost = np.linalg.norm(
                    target_positions[:, None, :] - candidate_positions[None, :, :], axis=2
                )
                _, chosen_columns = linear_sum_assignment(distance_cost)
                chosen_indices = candidate_indices[chosen_columns]
            labels[chosen_indices] = grid_label(row, col)
    return labels


def labels_for_output(
    points: np.ndarray, grid: SquareGrid, regular_size: int = 7
) -> np.ndarray:
    """Return the regular per-square labels used in the exported point cloud."""
    return assign_regular_labels(points, grid, size=regular_size)


def write_labeled_cloud(
    path: str | Path, header: str, points: np.ndarray, labels: np.ndarray
) -> None:
    """Write CloudCompare ASCII data with one trailing integer label column."""
    path = Path(path)
    if labels.shape != (len(points),):
        raise ValueError("labels must contain exactly one value per point")
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{header} label\n{len(points)}\n")
        for point, label in zip(points, labels, strict=True):
            handle.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(point[3])} {int(point[4])} {int(point[5])} "
                f"{point[6]:.6f} {point[7]:.6f} {int(label)}\n"
            )


def write_label_counts(path: str | Path, labels: np.ndarray) -> None:
    """Write all 25 label counts, including empty labels if any."""
    counts = np.bincount(labels, minlength=ROWS * COLS + 1)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "point_count"])
        writer.writerows(enumerate(counts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="palette.txt", help="input CloudCompare ASCII file")
    parser.add_argument("--output", default="palette_labeled.txt", help="labeled output file")
    parser.add_argument("--counts", default="palette_label_counts.csv", help="label-count CSV")
    parser.add_argument(
        "--regular-size", type=int, default=7, help="points per colour-block axis"
    )
    parser.add_argument("--dry-run", action="store_true", help="print fitted grid without writing")
    args = parser.parse_args()

    header, points = load_palette(args.input)
    grid, plane = fit_palette_grid(points)
    print(
        "grid: "
        f"plane=(y={plane.x_coefficient:.6f}x + {plane.z_coefficient:.6f}z + "
        f"{plane.intercept:.6f}, tolerance={plane.tolerance:.6f}), "
        f"x0={grid.x0:.6f}, z_top={grid.z_top:.6f}, side={grid.side:.6f}, "
        f"pitch_x={grid.pitch_x:.6f}, pitch_z={grid.pitch_z:.6f}"
    )
    if args.dry_run:
        return
    labels = labels_for_output(points, grid, regular_size=args.regular_size)
    write_labeled_cloud(args.output, header, points, labels)
    write_label_counts(args.counts, labels)
    print("label_counts=", np.bincount(labels, minlength=ROWS * COLS + 1).tolist())


if __name__ == "__main__":
    main()
