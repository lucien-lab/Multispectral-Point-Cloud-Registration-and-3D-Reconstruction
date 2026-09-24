"""Spectral data loader."""

import os
import re

import numpy as np


class SpectralDataLoader:
    """Loader for spectral data from text files."""

    def __init__(self) -> None:
        """Initialize the spectral data loader."""
        self.spectral_data = {}

    def load(self, data_dir: str, spec_dir: str = "gp/specData") -> dict:
        """Load spectral data from files.

        Args:
            data_dir: Path to data directory.
            spec_dir: Name of the spectral data subdirectory.

        Returns:
            Dictionary mapping point IDs to spectral data.
        """
        spec_path = os.path.join(data_dir, spec_dir)

        self.spectral_data = {}
        for filename in os.listdir(spec_path):
            if filename.startswith("spec_") and filename.endswith(".txt"):
                match = re.search(r"spec_(\d+)\.txt", filename)
                if match:
                    point_id = int(match.group(1))
                    filepath = os.path.join(spec_path, filename)

                    wavelengths = []
                    intensities = []
                    with open(filepath, "r") as file:
                        for line in file:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    wavelengths.append(float(parts[0]))
                                    intensities.append(float(parts[1]))
                                except ValueError:
                                    pass

                    self.spectral_data[point_id] = {
                        "wavelengths": np.array(wavelengths),
                        "intensities": np.array(intensities),
                    }

        return self.spectral_data

    def get_data(self) -> dict:
        """Get loaded spectral data.

        Returns:
            Dictionary of spectral data.
        """
        return self.spectral_data

    def get_count(self) -> int:
        """Get number of loaded spectral data files.

        Returns:
            Number of spectral data entries.
        """
        return len(self.spectral_data)

    def get_spectral(self, point_id: int) -> dict:
        """Get spectral data for a specific point.

        Args:
            point_id: ID of the point.

        Returns:
            Spectral data dictionary or None.
        """
        return self.spectral_data.get(point_id)