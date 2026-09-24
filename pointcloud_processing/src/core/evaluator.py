"""Evaluation metrics for clustering results."""

import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class Evaluator:
    """Evaluator for clustering results."""

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.metrics = {}

    def evaluate(self, features: list, labels: np.ndarray) -> dict:
        """Evaluate clustering results using multiple metrics.

        Args:
            features: List of feature dictionaries.
            labels: Array of cluster labels.

        Returns:
            Dictionary of evaluation metrics.
        """
        feature_matrix = np.array([f["features"] for f in features])

        self.metrics = {}

        if len(np.unique(labels)) > 1:
            self.metrics["silhouette"] = silhouette_score(
                feature_matrix, labels
            )
            self.metrics["calinski_harabasz"] = calinski_harabasz_score(
                feature_matrix, labels
            )
            self.metrics["davies_bouldin"] = davies_bouldin_score(
                feature_matrix, labels
            )

        return self.metrics

    def get_metrics(self) -> dict:
        """Get evaluation metrics.

        Returns:
            Dictionary of metrics.
        """
        return self.metrics

    def print_report(self) -> None:
        """Print evaluation report."""
        print("\n===== Clustering Evaluation Report =====")
        
        if "silhouette" in self.metrics:
            print(f"Silhouette Score: {self.metrics['silhouette']:.4f}")
            print("  (Range: -1 to 1, higher is better)")
        
        if "calinski_harabasz" in self.metrics:
            print(f"Calinski-Harabasz Index: {self.metrics['calinski_harabasz']:.2f}")
            print("  (Higher is better)")
        
        if "davies_bouldin" in self.metrics:
            print(f"Davies-Bouldin Index: {self.metrics['davies_bouldin']:.4f}")
            print("  (Lower is better)")
        
        print("=========================================\n")