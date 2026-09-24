"""Export data for CloudCompare labeling."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    PointCloudLoader,
    SpectralDataLoader,
    FeatureExtractor,
    Classifier,
)


def export_point_cloud_with_spectral(
    features: list,
    labels: np.ndarray,
    output_path: str,
) -> None:
    """Export point cloud with spectral features as PLY file.

    Args:
        features: List of feature dictionaries.
        labels: Array of cluster labels.
        output_path: Path to save the PLY file.
    """
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(features)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float intensity_mean\n")
        f.write("property float intensity_std\n")
        f.write("end_header\n")

        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, feature in enumerate(features):
            label = labels[i]
            r = int(colors[label][0] * 255)
            g = int(colors[label][1] * 255)
            b = int(colors[label][2] * 255)

            spectral = feature["spectral"]
            mean_int = np.mean(spectral) if len(spectral) > 0 else 0
            std_int = np.std(spectral) if len(spectral) > 0 else 0

            f.write(
                f"{feature['x']:.4f} {feature['y']:.4f} 0.0 "
                f"{r} {g} {b} {mean_int:.2f} {std_int:.2f}\n"
            )


import matplotlib.pyplot as plt


def export_point_cloud_xyz(
    features: list,
    labels: np.ndarray,
    output_path: str,
) -> None:
    """Export point cloud as XYZ file with label.

    Args:
        features: List of feature dictionaries.
        labels: Array of cluster labels.
        output_path: Path to save the XYZ file.
    """
    with open(output_path, "w") as f:
        f.write("X Y Z Label\n")
        for i, feature in enumerate(features):
            f.write(
                f"{feature['x']:.6f} {feature['y']:.6f} 0 {labels[i]}\n"
            )


def export_spectral_data(
    features: list,
    output_dir: str,
    max_points: int = 100,
) -> None:
    """Export spectral data as individual files.

    Args:
        features: List of feature dictionaries.
        output_dir: Directory to save spectral files.
        max_points: Maximum number of files to export.
    """
    os.makedirs(output_dir, exist_ok=True)

    step = max(1, len(features) // max_points)
    for i in range(0, len(features), step):
        feature = features[i]
        point_id = feature["id"]
        spectral = feature["spectral"]

        filepath = os.path.join(output_dir, f"spectral_{point_id}.txt")
        with open(filepath, "w") as f:
            for j, intensity in enumerate(spectral):
                f.write(f"{j} {intensity}\n")

        if i // step >= max_points - 1:
            break


def main() -> None:
    """Export data for CloudCompare."""
    data_dir = r"E:\11-control_system_of_lader\datas\20260130分类场景八万点"
    output_dir = os.path.join(data_dir, "export")

    print("Loading data...")
    point_loader = PointCloudLoader()
    point_cloud = point_loader.load(os.path.join(data_dir, "el_re.txt"))

    spectral_loader = SpectralDataLoader()
    spectral_data = spectral_loader.load(data_dir, "gp/specData")

    print("Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract(point_cloud, spectral_data)

    print("Classifying...")
    classifier = Classifier(num_classes=5)
    labels = classifier.fit_predict(features)

    print(f"\nExporting to {output_dir}...")

    ply_path = os.path.join(output_dir, "point_cloud.ply")
    print(f"  Exporting PLY: {ply_path}")
    export_point_cloud_with_spectral(features, labels, ply_path)

    xyz_path = os.path.join(output_dir, "point_cloud.xyz")
    print(f"  Exporting XYZ: {xyz_path}")
    export_point_cloud_xyz(features, labels, xyz_path)

    spectral_dir = os.path.join(output_dir, "spectral")
    print(f"  Exporting spectral samples: {spectral_dir}")
    export_spectral_data(features, spectral_dir, max_points=100)

    print("\nDone!")
    print(f"  - PLY file: {ply_path} (open in CloudCompare)")
    print(f"  - XYZ file: {xyz_path}")
    print(f"  - Spectral samples: {spectral_dir}")


if __name__ == "__main__":
    main()