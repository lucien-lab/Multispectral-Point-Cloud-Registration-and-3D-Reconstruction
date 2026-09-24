#!/usr/bin/env python3
"""Build a unified class-id point cloud from cropped_point_cloud.txt."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


OBJECT_CLASS_NAMES = (
    "bottles",
    "car",
    "cardbox",
    "column",
    "copper",
    "cup",
    "desk_background",
    "flowers",
    "gray_copper",
    "leaves",
    "mud_column",
    "root",
    "silver_cppper",
    "stone",
    "trash_bin",
)


def load_cropped_point_cloud(path: str | Path) -> np.ndarray:
    """Load the headerless comma-separated source point cloud."""
    points = np.loadtxt(path, delimiter=",")
    if points.ndim != 2 or points.shape[1] != 8:
        raise ValueError(f"expected an eight-column source point cloud, found {points.shape}")
    point_ids = points[:, 0].astype(int)
    if not np.all(points[:, 0] == point_ids) or len(np.unique(point_ids)) != len(point_ids):
        raise ValueError("source point_id values must be unique integers")
    return points


def _load_cloudcompare_columns(path: Path, columns: tuple[int, ...]) -> np.ndarray:
    data = np.loadtxt(path, comments="//", skiprows=2, usecols=columns)
    if len(columns) == 1:
        return np.asarray(data).reshape(-1, 1)
    return np.atleast_2d(data)


def _register_ids(
    labels_by_id: dict[int, int], ids: np.ndarray, class_id: int, source_name: str
) -> None:
    for raw_id in ids:
        point_id = int(raw_id)
        if point_id in labels_by_id:
            raise ValueError(f"duplicate point_id {point_id} in {source_name}")
        labels_by_id[point_id] = class_id


def build_classification(
    points: np.ndarray, class_directory: str | Path
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    """Map source point IDs to 25 palette and 15 object classes."""
    class_directory = Path(class_directory)
    labels_by_id: dict[int, int] = {}

    palette_data = _load_cloudcompare_columns(
        class_directory / "palette_labeled.txt", (6, 8)
    )
    palette_ids = palette_data[:, 0].astype(int)
    palette_labels = palette_data[:, 1].astype(int)
    if not np.all((0 <= palette_labels) & (palette_labels <= 24)):
        raise ValueError("palette labels must be in the inclusive range 0..24")
    for point_id, class_id in zip(palette_ids, palette_labels, strict=True):
        _register_ids(labels_by_id, np.array([point_id]), int(class_id), "palette_labeled.txt")

    for offset, class_name in enumerate(OBJECT_CLASS_NAMES, start=25):
        ids = _load_cloudcompare_columns(class_directory / f"{class_name}.txt", (6,))[:, 0]
        _register_ids(labels_by_id, ids, offset, f"{class_name}.txt")

    source_ids = points[:, 0].astype(int)
    source_id_set = set(source_ids.tolist())
    unknown_ids = set(labels_by_id) - source_id_set
    if unknown_ids:
        raise ValueError(f"classification files contain {len(unknown_ids)} unknown point IDs")
    missing_ids = source_id_set - set(labels_by_id)
    if missing_ids:
        raise ValueError(f"classification files omit {len(missing_ids)} source point IDs")

    labels = np.fromiter((labels_by_id[int(point_id)] for point_id in source_ids), dtype=int)
    mapping: list[dict[str, int | str]] = [
        {
            "class_id": 0,
            "class_name": "palette_background",
            "source_file": "palette_labeled.txt",
            "point_count": int(np.count_nonzero(labels == 0)),
        }
    ]
    mapping.extend(
        {
            "class_id": class_id,
            "class_name": f"palette_color_{class_id:02d}",
            "source_file": "palette_labeled.txt",
            "point_count": int(np.count_nonzero(labels == class_id)),
        }
        for class_id in range(1, 25)
    )
    mapping.extend(
        {
            "class_id": class_id,
            "class_name": class_name,
            "source_file": f"{class_name}.txt",
            "point_count": int(np.count_nonzero(labels == class_id)),
        }
        for class_id, class_name in enumerate(OBJECT_CLASS_NAMES, start=25)
    )
    return labels, mapping


def write_classified_point_cloud(
    path: str | Path, points: np.ndarray, labels: np.ndarray
) -> None:
    """Write source columns plus a final class_id column."""
    if labels.shape != (len(points),):
        raise ValueError("labels must contain one class ID for every source point")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["point_id", "x", "y", "z", "r", "g", "b", "class_id"]
        )
        for point, class_id in zip(points, labels, strict=True):
            writer.writerow(
                [
                    int(point[0]),
                    f"{point[1]:.6f}",
                    f"{point[2]:.6f}",
                    f"{point[3]:.6f}",
                    f"{point[4]:.6f}",
                    f"{point[5]:.6f}",
                    f"{point[6]:.6f}",
                    int(class_id),
                ]
            )


def write_class_mapping(path: str | Path, mapping: list[dict[str, int | str]]) -> None:
    """Write class IDs, names, source files and point counts as CSV."""
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["class_id", "class_name", "source_file", "point_count"]
        )
        writer.writeheader()
        writer.writerows(mapping)


def main() -> None:
    project_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default=project_directory.parent / "cropped_point_cloud.txt", type=Path
    )
    parser.add_argument("--class-dir", default=project_directory, type=Path)
    parser.add_argument(
        "--output", default=project_directory / "cropped_point_cloud_classified.txt", type=Path
    )
    parser.add_argument(
        "--mapping", default=project_directory / "class_id_mapping.csv", type=Path
    )
    args = parser.parse_args()

    points = load_cropped_point_cloud(args.source)
    labels, mapping = build_classification(points, args.class_dir)
    write_classified_point_cloud(args.output, points, labels)
    write_class_mapping(args.mapping, mapping)
    print(f"points={len(points)} classes={len(mapping)} output={args.output}")


if __name__ == "__main__":
    main()
