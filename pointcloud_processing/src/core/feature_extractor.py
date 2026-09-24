"""Feature extraction module."""

import numpy as np


class FeatureExtractor:
    """Extractor for geometric and spectral features."""

    def __init__(self) -> None:
        """Initialize the feature extractor."""
        self.features = []

    def extract(
        self,
        point_cloud: list,
        spectral_data: dict,
    ) -> list:
        """Extract geometric and spectral features.

        Args:
            point_cloud: List of point dictionaries.
            spectral_data: Dictionary of spectral data.

        Returns:
            List of feature dictionaries.
        """
        self.features = []

        for point in point_cloud:
            point_id = point["id"]

            if point_id not in spectral_data:
                continue

            intensities = spectral_data[point_id]["intensities"]

            if len(intensities) == 0:
                continue

            spectral_features = self._extract_spectral_features(intensities)

            feature_vector = [point["x"], point["y"]]
            feature_vector.extend(spectral_features)

            self.features.append({
                "id": point_id,
                "x": point["x"],
                "y": point["y"],
                "features": feature_vector,
                "spectral": intensities,
            })

        return self.features

    def _extract_spectral_features(self, intensities: np.ndarray) -> list:
        """Extract statistical features from spectral intensities.

        Args:
            intensities: Spectral intensity array.

        Returns:
            List of spectral features.
        """
        features = [
            np.mean(intensities),
            np.std(intensities),
            np.max(intensities),
            np.min(intensities),
            np.median(intensities),
        ]

        if len(intensities) > 10:
            features.append(np.percentile(intensities, 25))
            features.append(np.percentile(intensities, 75))

        num_segments = min(5, len(intensities) // 10)
        for i in range(num_segments):
            start_idx = i * (len(intensities) // num_segments)
            end_idx = (i + 1) * (len(intensities) // num_segments)
            features.append(np.mean(intensities[start_idx:end_idx]))

        return features

    def get_features(self) -> list:
        """Get extracted features.

        Returns:
            List of feature dictionaries.
        """
        return self.features

    def get_feature_matrix(self) -> np.ndarray:
        """Get feature matrix as numpy array.

        Returns:
            Feature matrix.
        """
        return np.array([f["features"] for f in self.features])