"""GUI application with interactive labeling for point cloud and spectral data."""

import os
import sys
import threading

import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    PointCloudLoader,
    SpectralDataLoader,
    FeatureExtractor,
    Classifier,
    Evaluator,
)
from ui.visualizer import Visualizer


class InteractiveLabelingApp:
    """GUI application with interactive labeling for classification."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the application."""
        self.root = root
        self.root.title("Point Cloud Classifier with Interactive Labeling")
        self.root.geometry("1400x900")

        self.point_cloud = []
        self.spectral_data = {}
        self.features = []
        self.labels = None
        self.num_classes = 5

        self.labeled_points = {}
        self.current_label = 0

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="Data Directory:").grid(
            row=0, column=0, sticky=tk.W, padx=5
        )
        self.dir_entry = ttk.Entry(control_frame, width=45)
        self.dir_entry.grid(row=0, column=1, padx=5)
        self.dir_entry.insert(
            0, r"E:\11-control_system_of_lader\datas\20260130分类场景八万点"
        )

        ttk.Button(control_frame, text="Browse", command=self._browse_dir).grid(
            row=0, column=2, padx=5
        )

        ttk.Label(control_frame, text="Classes:").grid(
            row=1, column=0, sticky=tk.W, padx=5
        )
        self.n_classes_var = tk.IntVar(value=5)
        ttk.Spinbox(
            control_frame,
            from_=2,
            to=10,
            textvariable=self.n_classes_var,
            width=10,
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        self.run_button = ttk.Button(
            control_frame,
            text="Run Classification",
            command=self._run_classification,
        )
        self.run_button.grid(row=2, column=0, columnspan=3, pady=5)

        label_frame = ttk.LabelFrame(control_frame, text="Labeling", padding="5")
        label_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)

        ttk.Label(label_frame, text="Current Label:").pack(side=tk.LEFT)
        
        self.label_buttons = []
        for i in range(10):
            btn = ttk.Button(
                label_frame,
                text=f"{i}",
                width=3,
                command=lambda idx=i: self._set_label(idx),
            )
            btn.pack(side=tk.LEFT, padx=2)
            self.label_buttons.append(btn)

        ttk.Button(
            label_frame,
            text="Clear Labels",
            command=self._clear_labels,
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            label_frame,
            text="Calculate Accuracy",
            command=self._calculate_accuracy,
        ).pack(side=tk.LEFT)

        self.status_label = ttk.Label(
            control_frame, text="Ready", foreground="blue"
        )
        self.status_label.grid(row=4, column=0, columnspan=3)

        self.progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.progress.grid(row=5, column=0, columnspan=3, sticky=tk.EW)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.figure = Figure(figsize=(14, 9))
        self.canvas = FigureCanvasTkAgg(self.figure, main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.figure.canvas.mpl_connect("button_press_event", self._on_click)

        self.info_frame = ttk.LabelFrame(
            self.root, text="Info", padding="10"
        )
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.info_text = tk.Text(self.info_frame, height=6, state=tk.DISABLED)
        self.info_text.pack(fill=tk.X)

    def _browse_dir(self) -> None:
        """Open directory browser dialog."""
        directory = filedialog.askdirectory()
        if directory:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, directory)

    def _set_label(self, label: int) -> None:
        """Set current label for marking."""
        self.current_label = label
        for i, btn in enumerate(self.label_buttons):
            if i == label:
                btn.state(["pressed"])
            else:
                btn.state(["!pressed"])
        self._update_info(f"Current label set to: {label}")

    def _clear_labels(self) -> None:
        """Clear all labeled points."""
        self.labeled_points = {}
        self._redraw_plot()
        self._update_info("All labels cleared")

    def _on_click(self, event) -> None:
        """Handle click event on plot."""
        if not self.features:
            return

        if event.xdata is None or event.ydata is None:
            return

        x, y = event.xdata, event.ydata
        threshold = 0.5

        closest_idx = None
        min_dist = float("inf")

        for idx, feature in enumerate(self.features):
            dist = np.sqrt((feature["x"] - x) ** 2 + (feature["y"] - y) ** 2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_idx = idx

        if closest_idx is not None:
            point_id = self.features[closest_idx]["id"]
            self.labeled_points[point_id] = {
                "index": closest_idx,
                "true_label": self.current_label,
                "predicted": int(self.labels[closest_idx]),
            }
            self._redraw_plot()
            self._update_info(
                f"Point {point_id} labeled as {self.current_label} "
                f"(predicted: {self.labels[closest_idx]})"
            )

    def _redraw_plot(self) -> None:
        """Redraw the plot with labeled points highlighted."""
        if not self.features or self.labels is None:
            return

        self.figure.clear()

        axes_point_cloud = self.figure.add_subplot(1, 2, 1)

        points_array = np.array([[f["x"], f["y"]] for f in self.features])

        unlabeled_mask = np.ones(len(self.features), dtype=bool)
        for point_id in self.labeled_points:
            idx = self.labeled_points[point_id]["index"]
            unlabeled_mask[idx] = False

        axes_point_cloud.scatter(
            points_array[unlabeled_mask, 0],
            points_array[unlabeled_mask, 1],
            c=self.labels[unlabeled_mask],
            cmap="tab10",
            s=3,
            alpha=0.6,
        )

        if self.labeled_points:
            labeled_indices = [v["index"] for v in self.labeled_points.values()]
            true_labels = [v["true_label"] for v in self.labeled_points.values()]

            axes_point_cloud.scatter(
                points_array[labeled_indices, 0],
                points_array[labeled_indices, 1],
                c=true_labels,
                cmap="Set1",
                s=30,
                edgecolors="black",
                linewidths=1.5,
                marker="s",
            )

        axes_point_cloud.set_xlabel("X")
        axes_point_cloud.set_ylabel("Y")
        axes_point_cloud.set_title("Point Cloud (Click to label)")
        axes_point_cloud.set_aspect("equal")

        axes_spectral = self.figure.add_subplot(1, 2, 2)

        unique_labels = np.unique(self.labels)
        label_colors = {
            label: plt.cm.tab10(label / max(unique_labels.max(), 1))
            for label in unique_labels
        }

        for idx, feature in enumerate(self.features[:30]):
            label = self.labels[idx]
            spectral = feature["spectral"]
            alpha = 0.3
            color = label_colors[label]
            axes_spectral.plot(
                range(len(spectral)),
                spectral,
                alpha=alpha,
                color=color,
                linewidth=0.5,
            )

        axes_spectral.set_xlabel("Spectral Index")
        axes_spectral.set_ylabel("Intensity")
        axes_spectral.set_title("Spectral Data (Sampled)")

        self.figure.tight_layout()
        self.canvas.draw()

    def _calculate_accuracy(self) -> None:
        """Calculate classification accuracy based on labeled points."""
        if not self.labeled_points:
            messagebox.showwarning("Warning", "No points labeled yet!")
            return

        correct = 0
        total = len(self.labeled_points)

        for point_id, data in self.labeled_points.items():
            if data["true_label"] == data["predicted"]:
                correct += 1

        accuracy = correct / total if total > 0 else 0

        self._update_info(
            f"Accuracy: {accuracy*100:.1f}% ({correct}/{total} correct)\n"
            f"True label = Predicted label"
        )

        messagebox.showinfo(
            "Accuracy Result",
            f"Labeled points: {total}\n"
            f"Correct: {correct}\n"
            f"Accuracy: {accuracy*100:.1f}%"
        )

    def _run_classification(self) -> None:
        """Run classification in a separate thread."""
        data_dir = self.dir_entry.get()
        if not os.path.exists(data_dir):
            messagebox.showerror("Error", "Directory does not exist")
            return

        self.run_button.config(state=tk.DISABLED)
        self.progress.start()
        self.status_label.config(text="Loading data...")

        worker_thread = threading.Thread(
            target=self._classify_data, args=(data_dir,)
        )
        worker_thread.daemon = True
        worker_thread.start()

    def _load_point_cloud(self, data_dir: str) -> list:
        """Load point cloud data from file."""
        filepath = os.path.join(data_dir, "el_re.txt")
        loader = PointCloudLoader()
        return loader.load(filepath)

    def _load_spectral_data(self, data_dir: str) -> dict:
        """Load spectral data from files."""
        loader = SpectralDataLoader()
        return loader.load(data_dir, "gp/specData")

    def _extract_features(self, point_cloud: list, spectral_data: dict) -> list:
        """Extract geometric and spectral features."""
        extractor = FeatureExtractor()
        return extractor.extract(point_cloud, spectral_data)

    def _classify_features(self, features: list, num_classes: int) -> np.ndarray:
        """Classify features using K-Means clustering."""
        classifier = Classifier(num_classes=num_classes)
        return classifier.fit_predict(features)

    def _classify_data(self, data_dir: str) -> None:
        """Main classification workflow."""
        try:
            self._update_status("Loading point cloud...")
            self.point_cloud = self._load_point_cloud(data_dir)

            self._update_status("Loading spectral data...")
            self.spectral_data = self._load_spectral_data(data_dir)

            self._update_status("Extracting features...")
            self.features = self._extract_features(
                self.point_cloud, self.spectral_data
            )

            self._update_status("Classifying...")
            self.num_classes = self.n_classes_var.get()
            self.labels = self._classify_features(self.features, self.num_classes)

            self._update_status("Visualizing...")
            self.labeled_points = {}
            self._redraw_plot()

            self._update_status("Complete! Click on points to label them.")

        except Exception as error:
            messagebox.showerror("Error", str(error))

        finally:
            self.progress.stop()
            self.run_button.config(state=tk.NORMAL)

    def _update_status(self, message: str) -> None:
        """Update status label."""
        self.status_label.config(text=message)
        self.root.update()

    def _update_info(self, message: str) -> None:
        """Update info panel."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        info = f"Labeled points: {len(self.labeled_points)}\n"
        info += f"Current label: {self.current_label}\n"
        info += f"\n{message}\n"
        
        if self.labeled_points:
            label_counts = {}
            for data in self.labeled_points.values():
                l = data["true_label"]
                label_counts[l] = label_counts.get(l, 0) + 1
            info += f"\nLabel distribution: {label_counts}"
        
        self.info_text.insert(1.0, info)
        self.info_text.config(state=tk.DISABLED)


def main() -> None:
    """Main entry point for the application."""
    root = tk.Tk()
    app = InteractiveLabelingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()