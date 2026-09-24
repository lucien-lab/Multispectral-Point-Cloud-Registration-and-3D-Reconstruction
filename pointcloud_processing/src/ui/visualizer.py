"""Visualizer module for plotting results."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


class Visualizer:
    """Visualizer for classification results."""

    def __init__(self) -> None:
        """Initialize the visualizer."""
        self.figure = None

    def plot_classification(
        self,
        features: list,
        labels: np.ndarray,
        save_path: str = None,
    ) -> None:
        """Plot point cloud classification results.

        Args:
            features: List of feature dictionaries.
            labels: Array of class labels.
            save_path: Path to save the plot.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        points = np.array([[f["x"], f["y"]] for f in features])

        axes[0].scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            cmap="tab10",
            s=5,
            alpha=0.7,
        )
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].set_title("Point Cloud Classification")
        axes[0].set_aspect("equal")

        unique_labels = np.unique(labels)
        legend_elements = [
            Patch(
                facecolor=plt.cm.tab10(label / max(unique_labels.max(), 1)),
                label=f"Class {label}",
            )
            for label in unique_labels
        ]
        axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)

        sample_indices = np.random.choice(
            len(features),
            min(20, len(features)),
            replace=False,
        )
        for idx in sample_indices:
            spectral = features[idx]["spectral"]
            label = labels[idx]
            axes[1].plot(
                range(len(spectral)),
                spectral,
                alpha=0.5,
                label=f"Class {label}" if idx == sample_indices[0] else "",
            )

        axes[1].set_xlabel("Spectral Index")
        axes[1].set_ylabel("Intensity")
        axes[1].set_title("Spectral Data Samples")
        axes[1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def plot_gui(
        self,
        features: list,
        labels: np.ndarray,
    ) -> plt.Figure:
        """Create figure for GUI display.

        Args:
            features: List of feature dictionaries.
            labels: Array of class labels.

        Returns:
            Matplotlib figure object.
        """
        figure = plt.figure(figsize=(12, 8))

        axes_point = figure.add_subplot(1, 2, 1)

        points = np.array([[f["x"], f["y"]] for f in features])
        axes_point.scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            cmap="tab10",
            s=3,
            alpha=0.7,
        )
        axes_point.set_xlabel("X")
        axes_point.set_ylabel("Y")
        axes_point.set_title("Point Cloud Classification")
        axes_point.set_aspect("equal")

        unique_labels = np.unique(labels)
        legend_elements = [
            Patch(
                facecolor=plt.cm.tab10(label / max(unique_labels.max(), 1)),
                label=f"Class {label}",
            )
            for label in unique_labels
        ]
        axes_point.legend(handles=legend_elements, loc="upper right", fontsize=8)

        axes_spectral = figure.add_subplot(1, 2, 2)

        label_colors = {
            label: plt.cm.tab10(label / max(unique_labels.max(), 1))
            for label in unique_labels
        }

        for idx, feature in enumerate(features[:50]):
            label = labels[idx]
            spectral = feature["spectral"]
            axes_spectral.plot(
                range(len(spectral)),
                spectral,
                alpha=0.4,
                color=label_colors[label],
                linewidth=0.5,
            )

        axes_spectral.set_xlabel("Spectral Index")
        axes_spectral.set_ylabel("Intensity")
        axes_spectral.set_title("Spectral Data (Sampled)")

        figure.tight_layout()

        self.figure = figure
        return figure

    def close(self) -> None:
        """Close the plot."""
        if self.figure:
            plt.close(self.figure)
        self.figure = None