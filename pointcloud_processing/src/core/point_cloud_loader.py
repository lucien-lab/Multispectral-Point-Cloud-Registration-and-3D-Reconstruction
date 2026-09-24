"""Point cloud data loader."""

import os
import re

import numpy as np


class PointCloudLoader:
    """Loader for point cloud data from text files."""

    def __init__(self) -> None:
        """Initialize the point cloud loader."""
        self.points = []

    def load(self, filepath: str) -> list:
        """Load point cloud data from file.

        Args:
            filepath: Path to the point cloud file.

        Returns:
            List of point dictionaries with id, x, y.
        """
        pattern = re.compile(
            r"(\d+)\s+X[:\uff1a](-?\d+\.?\d*)\s+Y[:\uff1a](-?\d+\.?\d*)"
        )

        self.points = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    point_id = int(match.group(1))
                    x_coord = float(match.group(2))
                    y_coord = float(match.group(3))
                    self.points.append({
                        "id": point_id,
                        "x": x_coord,
                        "y": y_coord,
                    })

        return self.points

    def get_points(self) -> list:
        """Get loaded points.

        Returns:
            List of point dictionaries.
        """
        return self.points

    def get_count(self) -> int:
        """Get number of loaded points.

        Returns:
            Number of points.
        """
        return len(self.points)