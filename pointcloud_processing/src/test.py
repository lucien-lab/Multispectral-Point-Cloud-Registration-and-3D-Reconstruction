"""Quick test script for evaluation."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    PointCloudLoader,
    SpectralDataLoader,
    FeatureExtractor,
    Classifier,
    Evaluator,
)


def main():
    start = time.time()
    
    data_dir = r"E:\11-control_system_of_lader\datas\20260130分类场景八万点"

    print("Loading point cloud...")
    t0 = time.time()
    point_loader = PointCloudLoader()
    point_cloud = point_loader.load(os.path.join(data_dir, "el_re.txt"))
    print(f"  {point_loader.get_count()} points ({time.time()-t0:.1f}s)")

    print("Loading spectral data...")
    t0 = time.time()
    spectral_loader = SpectralDataLoader()
    spectral_data = spectral_loader.load(data_dir, "gp/specData")
    print(f"  {spectral_loader.get_count()} files ({time.time()-t0:.1f}s)")

    print("Extracting features...")
    t0 = time.time()
    extractor = FeatureExtractor()
    features = extractor.extract(point_cloud, spectral_data)
    print(f"  {len(features)} points ({time.time()-t0:.1f}s)")

    print("Classifying...")
    t0 = time.time()
    classifier = Classifier(num_classes=5)
    labels = classifier.fit_predict(features)
    print(f"  Done ({time.time()-t0:.1f}s)")

    print("\nClass distribution:")
    for label, count in classifier.get_label_counts().items():
        print(f"  Class {label}: {count} points")

    print("\nEvaluating...")
    evaluator = Evaluator()
    metrics = evaluator.evaluate(features, labels)
    evaluator.print_report()

    print(f"Total time: {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()