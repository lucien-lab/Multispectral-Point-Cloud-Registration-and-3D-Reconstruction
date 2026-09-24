"""Classification module."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class Classifier:
    """K-Means classifier for point cloud and spectral data."""

    def __init__(self, num_classes: int = 5) -> None:
        """Initialize the classifier.

        Args:
            num_classes: Number of clusters for classification.
        """
        self.num_classes = num_classes
        self.labels = None
        self.scaler = StandardScaler()
        self.kmeans = None

    def fit_predict(self, features: list) -> np.ndarray:
        """Fit the classifier and predict labels.

        Args:
            features: List of feature dictionaries.

        Returns:
            Array of class labels.
        """
        feature_matrix = np.array([f["features"] for f in features])

        feature_scaled = self.scaler.fit_transform(feature_matrix)

        self.kmeans = KMeans(
            n_clusters=self.num_classes,
            random_state=42,
            n_init=10,
        )
        self.labels = self.kmeans.fit_predict(feature_scaled)

        return self.labels

    def get_labels(self) -> np.ndarray:
        """Get classification labels.

        Returns:
            Array of labels.
        """
        return self.labels

    def get_unique_labels(self) -> np.ndarray:
        """Get unique label values.

        Returns:
            Array of unique labels.
        """
        return np.unique(self.labels)

    def get_label_counts(self) -> dict:
        """Get count of points in each class.

        Returns:
            Dictionary mapping labels to counts.
        """
        if self.labels is None:
            return {}
        unique_labels = np.unique(self.labels)
        return {int(label): int(np.sum(self.labels == label))
                for label in unique_labels}