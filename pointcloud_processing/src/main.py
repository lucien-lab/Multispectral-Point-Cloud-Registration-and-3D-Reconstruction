"""Main application for point cloud and spectral data classification."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    PointCloudLoader,
    SpectralDataLoader,
    FeatureExtractor,
    Classifier,
    Evaluator,
)
from ui.visualizer import Visualizer


def main() -> None:
    """Main entry point for the application."""
    data_dir = (
        r"E:\11-control_system_of_lader\datas\20260130分类场景八万点"
    )

    print("Loading point cloud data...")
    point_loader = PointCloudLoader()
    point_cloud = point_loader.load(os.path.join(data_dir, "el_re.txt"))
    print(f"  Loaded {point_loader.get_count()} points")

    print("\nLoading spectral data...")
    spectral_loader = SpectralDataLoader()
    spectral_data = spectral_loader.load(data_dir, "gp/specData")
    print(f"  Loaded {spectral_loader.get_count()} spectral files")

    print("\nExtracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract(point_cloud, spectral_data)
    print(f"  Extracted features for {len(features)} points")

    print("\nClassifying...")
    classifier = Classifier(num_classes=5)
    labels = classifier.fit_predict(features)
    print(f"  Classification complete")

    label_counts = classifier.get_label_counts()
    print("\nClass distribution:")
    for label, count in label_counts.items():
        print(f"  Class {label}: {count} points")

    print("\nEvaluating...")
    evaluator = Evaluator()
    metrics = evaluator.evaluate(features, labels)
    evaluator.print_report()

    print("\nVisualizing...")
    visualizer = Visualizer()
    visualizer.plot_classification(features, labels, "classification_result.png")
    print("  Plot saved to classification_result.png")

    print("\nClassification complete!")


if __name__ == "__main__":
    main()