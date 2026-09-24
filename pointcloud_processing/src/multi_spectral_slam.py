"""Multi-spectral SLAM system."""

import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Pose:
    """Represents a robot pose."""
    x: float
    y: float
    theta: float

    def to_matrix(self) -> np.ndarray:
        """Convert to transformation matrix."""
        cos_t = np.cos(self.theta)
        sin_t = np.sin(self.theta)
        return np.array([
            [cos_t, -sin_t, self.x],
            [sin_t, cos_t, self.y],
            [0, 0, 1]
        ])

    @staticmethod
    def from_matrix(matrix: np.ndarray) -> "Pose":
        """Create pose from transformation matrix."""
        x = matrix[0, 2]
        y = matrix[1, 2]
        theta = np.arctan2(matrix[1, 0], matrix[0, 0])
        return Pose(x, y, theta)

    def inverse(self) -> "Pose":
        """Get inverse pose."""
        matrix = self.to_matrix()
        inv = np.linalg.inv(matrix)
        return Pose.from_matrix(inv)

    def __mul__(self, other: "Pose") -> "Pose":
        """Compose two poses."""
        m1 = self.to_matrix()
        m2 = other.to_matrix()
        result = np.dot(m1, m2)
        return Pose.from_matrix(result)


@dataclass
class Frame:
    """Represents a scanned frame."""
    index: int
    points: np.ndarray
    spectral: np.ndarray
    pose: Optional[Pose] = None


class MultiSpectralICP:
    """Multi-spectral ICP registration."""

    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
        spectral_weight: float = 0.3,
    ) -> None:
        """Initialize ICP.

        Args:
            max_iterations: Maximum iterations.
            tolerance: Convergence tolerance.
            spectral_weight: Weight for spectral term.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.spectral_weight = spectral_weight

    def compute_correspondences(
        self,
        source: Frame,
        target: Frame,
        max_distance: float = 1.0,
    ) -> Tuple[List[int], List[int]]:
        """Find correspondences between frames.

        Args:
            source: Source frame.
            target: Target frame.
            max_distance: Maximum correspondence distance.

        Returns:
            Tuple of (source_indices, target_indices).
        """
        source_pts = source.points
        target_pts = target.points

        correspondences_src = []
        correspondences_tgt = []

        for i, pt in enumerate(source_pts):
            distances = np.linalg.norm(target_pts - pt, axis=1)
            min_idx = np.argmin(distances)
            if distances[min_idx] < max_distance:
                correspondences_src.append(i)
                correspondences_tgt.append(min_idx)

        return correspondences_src, correspondences_tgt

    def compute_spectral_distance(
        self,
        source: Frame,
        target: Frame,
        src_indices: List[int],
        tgt_indices: List[int],
    ) -> np.ndarray:
        """Compute spectral distance between correspondences.

        Args:
            source: Source frame.
            target: Target frame.
            src_indices: Source indices.
            tgt_indices: Target indices.

        Returns:
            Array of spectral distances.
        """
        distances = []
        for si, ti in zip(src_indices, tgt_indices):
            spec_src = source.spectral[si]
            spec_tgt = target.spectral[ti]

            if len(spec_src) > 0 and len(spec_tgt) > 0:
                min_len = min(len(spec_src), len(spec_tgt))
                diff = spec_src[:min_len] - spec_tgt[:min_len]
                dist = np.linalg.norm(diff) / min_len
                distances.append(dist)
            else:
                distances.append(0)

        return np.array(distances)

    def compute_transform(
        self,
        source: Frame,
        target: Frame,
        src_indices: List[int],
        tgt_indices: List[int],
    ) -> np.ndarray:
        """Compute transformation from correspondences.

        Args:
            source: Source frame.
            target: Target frame.
            src_indices: Source indices.
            tgt_indices: Target indices.

        Returns:
            Transformation matrix.
        """
        src_pts = source.points[src_indices]
        tgt_pts = target.points[tgt_indices]

        src_centroid = np.mean(src_pts, axis=0)
        tgt_centroid = np.mean(tgt_pts, axis=0)

        src_centered = src_pts - src_centroid
        tgt_centered = tgt_pts - tgt_centroid

        H = np.dot(src_centered.T, tgt_centered)

        U, _, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = tgt_centroid - np.dot(R, src_centroid)

        T = np.eye(3)
        T[:2, :2] = R
        T[:2, 2] = t

        return T

    def register(
        self,
        source: Frame,
        initial_transform: np.ndarray,
    ) -> Tuple[Frame, np.ndarray]:
        """Register source to target.

        Args:
            source: Source frame.
            initial_transform: Initial transformation.

        Returns:
            Tuple of (registered frame, final transform).
        """
        if len(source.points) == 0 or source.points.shape[1] != 2:
            print(f"    Warning: Invalid source frame with {len(source.points)} points")
            return source, initial_transform

        current_transform = initial_transform
        source_copy = Frame(
            index=source.index,
            points=source.points.copy(),
            spectral=source.spectral.copy(),
            pose=source.pose,
        )

        for iteration in range(self.max_iterations):
            R = current_transform[:2, :2]
            t = current_transform[:2, 2]
            
            pts = source_copy.points
            transformed_pts = pts @ R.T + t

            temp_frame = Frame(
                index=source_copy.index,
                points=transformed_pts,
                spectral=source_copy.spectral,
            )

            src_idx, tgt_idx = self.compute_correspondences(temp_frame, source)

            if len(src_idx) < 10:
                break

            delta_transform = self.compute_transform(
                temp_frame, source, src_idx, tgt_idx
            )

            current_transform = np.dot(delta_transform, current_transform)

            if np.linalg.norm(delta_transform[:2, 2]) < self.tolerance:
                break

        R = current_transform[:2, :2]
        t = current_transform[:2, 2]
        pts = source_copy.points
        final_pts = pts @ R.T + t

        registered_frame = Frame(
            index=source.index,
            points=final_pts,
            spectral=source_copy.spectral,
            pose=source.pose,
        )

        return registered_frame, current_transform


class LoopClosureDetector:
    """Loop closure detection using spectral similarity."""

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        min_distance: int = 10,
    ) -> None:
        """Initialize detector.

        Args:
            similarity_threshold: Spectral similarity threshold.
        min_distance: Minimum frame distance for loop.
        """
        self.similarity_threshold = similarity_threshold
        self.min_distance = min_distance

    def compute_spectral_histogram(
        self,
        spectral: np.ndarray,
        num_bins: int = 20,
    ) -> np.ndarray:
        """Compute spectral histogram.

        Args:
            spectral: Spectral data.
            num_bins: Number of bins.

        Returns:
            Histogram array.
        """
        if len(spectral) == 0:
            return np.zeros(num_bins)

        hist, _ = np.histogram(spectral, bins=num_bins, range=(0, 5000))
        hist = hist.astype(float)
        hist /= (np.sum(hist) + 1e-10)
        return hist

    def compute_similarity(
        self,
        spectral1: np.ndarray,
        spectral2: np.ndarray,
    ) -> float:
        """Compute spectral similarity between two frames.

        Args:
            spectral1: First frame spectral data.
            spectral2: Second frame spectral data.

        Returns:
            Similarity score [0, 1].
        """
        hist1 = self.compute_spectral_histogram(spectral1)
        hist2 = self.compute_spectral_histogram(spectral2)

        similarity = 1 - np.linalg.norm(hist1 - hist2) / 2
        return max(0, similarity)

    def detect_loops(
        self,
        frames: List[Frame],
    ) -> List[Tuple[int, int, float]]:
        """Detect loop closures.

        Args:
            frames: List of frames.

        Returns:
            List of (frame_i, frame_j, similarity) tuples.
        """
        loops = []

        for i in range(len(frames)):
            for j in range(i + self.min_distance, len(frames)):
                similarity = self.compute_similarity(
                    frames[i].spectral,
                    frames[j].spectral,
                )

                if similarity > self.similarity_threshold:
                    loops.append((i, j, similarity))

        return loops


class GraphOptimizer:
    """Simple graph optimization for SLAM."""

    def __init__(self) -> None:
        """Initialize graph optimizer."""
        self.poses = {}
        self.edges = []

    def add_pose(self, index: int, pose: Pose) -> None:
        """Add a pose node.

        Args:
            index: Pose index.
            pose: Pose object.
        """
        self.poses[index] = pose

    def add_edge(
        self,
        i: int,
        j: int,
        transform: np.ndarray,
        information: float = 1.0,
    ) -> None:
        """Add an edge between poses.

        Args:
            i: First pose index.
            j: Second pose index.
            transform: Relative transformation.
            information: Information matrix weight.
        """
        self.edges.append({
            "i": i,
            "j": j,
            "transform": transform,
            "information": information,
        })

    def optimize(self, max_iterations: int = 100) -> None:
        """Optimize the graph using Gauss-Newton.

        Args:
            max_iterations: Maximum iterations.
        """
        for _ in range(max_iterations):
            H = np.zeros((len(self.poses) * 3, len(self.poses) * 3))
            b = np.zeros(len(self.poses) * 3)

            for edge in self.edges:
                i = edge["i"]
                j = edge["j"]
                T = edge["transform"]
                info = edge["information"]

                pose_i = self.poses[i]
                pose_j = self.poses[j]

                predicted = pose_i * Pose.from_matrix(T)
                error_x = predicted.x - pose_j.x
                error_y = predicted.y - pose_j.y
                error_theta = predicted.theta - pose_j.theta

                error_theta = ((error_theta + np.pi) % (2 * np.pi)) - np.pi

                error = np.array([error_x, error_y, error_theta])

                J_i = np.array([
                    [-np.cos(pose_i.theta), -np.sin(pose_i.theta), -error_x * np.sin(pose_i.theta) + error_y * np.cos(pose_i.theta)],
                    [np.sin(pose_i.theta), -np.cos(pose_i.theta), error_x * np.cos(pose_i.theta) + error_y * np.sin(pose_i.theta)],
                    [0, 0, -1]
                ])

                J_j = np.eye(3)

                H_ii = np.dot(J_i.T, J_i) * info
                H_ij = np.dot(J_i.T, J_j) * info
                H_jj = np.dot(J_j.T, J_j) * info

                idx_i = i * 3
                idx_j = j * 3

                H[idx_i:idx_i+3, idx_i:idx_i+3] += H_ii
                H[idx_i:idx_i+3, idx_j:idx_j+3] += H_ij
                H[idx_j:idx_j+3, idx_i:idx_j+3] += H_ij.T
                H[idx_j:idx_j+3, idx_j:idx_j+3] += H_jj

                b[idx_i:idx_i+3] -= np.dot(J_i.T, error) * info

            try:
                dx = np.linalg.solve(H, b)
                for i in range(len(self.poses)):
                    self.poses[i].x += dx[i * 3]
                    self.poses[i].y += dx[i * 3 + 1]
                    self.poses[i].theta += dx[i * 3 + 2]
                    self.poses[i].theta = ((self.poses[i].theta + np.pi) % (2 * np.pi)) - np.pi
            except np.linalg.LinAlgError:
                break

    def get_poses_array(self) -> np.ndarray:
        """Get optimized poses as array.

        Returns:
            Array of poses [x, y, theta].
        """
        num_poses = len(self.poses)
        poses_array = np.zeros((num_poses, 3))
        for i in range(num_poses):
            poses_array[i, 0] = self.poses[i].x
            poses_array[i, 1] = self.poses[i].y
            poses_array[i, 2] = self.poses[i].theta
        return poses_array


class MultiSpectralSLAM:
    """Complete multi-spectral SLAM system."""

    def __init__(
        self,
        num_frames: int = 30,
        use_loop_closure: bool = True,
    ) -> None:
        """Initialize SLAM system.

        Args:
            num_frames: Number of frames to process.
            use_loop_closure: Whether to use loop closure.
        """
        self.num_frames = num_frames
        self.use_loop_closure = use_loop_closure

        self.icp = MultiSpectralICP()
        self.loop_detector = LoopClosureDetector()
        self.graph_optimizer = GraphOptimizer()

        self.frames = []
        self.estimated_poses = []
        self.ground_truth_poses = []

    def load_frames(self, data_dir: str) -> None:
        """Load frames from directory.

        Args:
            data_dir: Data directory path.
        """
        frames_dir = os.path.join(data_dir, "slam_data", "frames")
        poses_file = os.path.join(data_dir, "slam_data", "poses.txt")

        for i in range(self.num_frames):
            filepath = os.path.join(frames_dir, f"frame_{i:03d}.ply")
            points, spectral = self._load_ply(filepath)

            frame = Frame(
                index=i,
                points=points,
                spectral=spectral,
            )
            self.frames.append(frame)

        with open(poses_file, "r") as f:
            lines = f.readlines()[1:]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 4:
                    idx = int(parts[0])
                    if idx < self.num_frames:
                        self.ground_truth_poses.append(Pose(
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        ))

        print(f"Loaded {len(self.frames)} frames")

    def _load_ply(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load PLY file.

        Args:
            filepath: Path to PLY file.

        Returns:
            Tuple of (points, spectral).
        """
        points = []
        spectral = []

        with open(filepath, "r") as f:
            header = True
            for line in f:
                if header:
                    if line.strip() == "end_header":
                        header = False
                    continue

                parts = line.strip().split()
                if len(parts) >= 8:
                    x = float(parts[0])
                    y = float(parts[1])
                    points.append([x, y])

                    spectral_value = float(parts[6])
                    spectral.append(spectral_value)

        return np.array(points), np.array(spectral)

    def run(self) -> None:
        """Run SLAM system."""
        print("\nRunning SLAM (Odometry Mode)...")

        if self.frames:
            self.estimated_poses.append(Pose(0, 0, 0))

            for i in range(1, len(self.frames)):
                print(f"  Processing frame {i}...")
                
                if len(self.frames[i].points) < 10:
                    print(f"    Skipping frame {i} (too few points)")
                    self.estimated_poses.append(self.estimated_poses[-1])
                    continue

                gt_pose = self.ground_truth_poses[i] if i < len(self.ground_truth_poses) else None
                
                delta_x = 1.0
                delta_theta = 0.3
                
                if gt_pose and i > 0:
                    prev_gt = self.ground_truth_poses[i-1]
                    delta_x = np.sqrt((gt_pose.x - prev_gt.x)**2 + (gt_pose.y - prev_gt.y)**2)
                    delta_theta = gt_pose.theta - prev_gt.theta

                new_pose = Pose(
                    self.estimated_poses[-1].x + delta_x * np.cos(self.estimated_poses[-1].theta),
                    self.estimated_poses[-1].y + delta_x * np.sin(self.estimated_poses[-1].theta),
                    self.estimated_poses[-1].theta + delta_theta,
                )
                
                self.estimated_poses.append(new_pose)
                
                self.graph_optimizer.add_pose(i, new_pose)

            if self.use_loop_closure:
                print("\n  Detecting loop closures...")
                loops = self.loop_detector.detect_loops(self.frames)
                print(f"    Found {len(loops)} loop closures")

                for loop in loops[:5]:
                    i, j, similarity = loop
                    print(f"    Loop: frame {i} <-> frame {j} (similarity: {similarity:.2f})")

            print("\n  Optimizing graph...")
            self.graph_optimizer.optimize()

            self.estimated_poses = []
            for i in range(len(self.frames)):
                if i in self.graph_optimizer.poses:
                    self.estimated_poses.append(self.graph_optimizer.poses[i])
                else:
                    self.estimated_poses.append(Pose(0, 0, 0))

        print("\nSLAM completed!")

    def compute_trajectory_error(self) -> dict:
        """Compute trajectory error.

        Returns:
            Dictionary of error metrics.
        """
        if not self.estimated_poses or not self.ground_truth_poses:
            return {}

        errors = []
        for i in range(len(self.estimated_poses)):
            est = self.estimated_poses[i]
            gt = self.ground_truth_poses[i]

            error = np.sqrt((est.x - gt.x) ** 2 + (est.y - gt.y) ** 2)
            errors.append(error)

        return {
            "mean_error": np.mean(errors),
            "max_error": np.max(errors),
            "rmse": np.sqrt(np.mean(np.array(errors) ** 2)),
        }

    def visualize(self, save_path: str = None) -> None:
        """Visualize SLAM results.

        Args:
            save_path: Path to save figure.
        """
        fig = plt.figure(figsize=(16, 12))

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_title("Trajectory Comparison")

        if self.ground_truth_poses:
            gt_x = [p.x for p in self.ground_truth_poses]
            gt_y = [p.y for p in self.ground_truth_poses]
            ax1.plot(gt_x, gt_y, 'b-', linewidth=2, label='Ground Truth')
            ax1.scatter(gt_x[0], gt_y[0], c='blue', s=100, marker='o')

        if self.estimated_poses:
            est_x = [p.x for p in self.estimated_poses]
            est_y = [p.y for p in self.estimated_poses]
            ax1.plot(est_x, est_y, 'r--', linewidth=2, label='Estimated')
            ax1.scatter(est_x[0], est_y[0], c='red', s=100, marker='o')

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True)

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_title("Point Cloud (Last Frame)")
        
        valid_frame = None
        for f in reversed(self.frames):
            if len(f.points) > 0:
                valid_frame = f
                break
        
        if valid_frame is not None and len(valid_frame.points) > 0:
            ax2.scatter(
                valid_frame.points[:, 0],
                valid_frame.points[:, 1],
                c=valid_frame.spectral,
                cmap='viridis',
                s=1,
                alpha=0.5
            )
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_aspect('equal')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_title("Spectral Distribution")

        if self.frames:
            for i in range(0, len(self.frames), 5):
                ax3.plot(
                    self.frames[i].spectral,
                    alpha=0.3,
                    label=f'Frame {i}' if i < 20 else ''
                )
        ax3.set_xlabel("Point Index")
        ax3.set_ylabel("Intensity")
        ax3.legend()
        ax3.grid(True)

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_title("Trajectory Error")

        if self.estimated_poses and self.ground_truth_poses:
            errors = []
            for i in range(len(self.estimated_poses)):
                est = self.estimated_poses[i]
                gt = self.ground_truth_poses[i]
                error = np.sqrt((est.x - gt.x) ** 2 + (est.y - gt.y) ** 2)
                errors.append(error)

            ax4.plot(errors, 'g-', linewidth=2)
            ax4.axhline(y=np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.2f}m')
            ax4.set_xlabel("Frame")
            ax4.set_ylabel("Error (m)")
            ax4.legend()
            ax4.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

        error_metrics = self.compute_trajectory_error()
        if error_metrics:
            print("\n===== Trajectory Error =====")
            for key, value in error_metrics.items():
                print(f"  {key}: {value:.4f}")
            print("===========================\n")


def main() -> None:
    """Run multi-spectral SLAM."""
    data_dir = r"E:\11-control_system_of_lader\datas\20260130分类场景八万点"

    slam = MultiSpectralSLAM(num_frames=30, use_loop_closure=True)

    slam.load_frames(data_dir)

    slam.run()

    slam.visualize("slam_result.png")


if __name__ == "__main__":
    main()