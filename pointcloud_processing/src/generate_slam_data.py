"""Generate simulated multi-frame point cloud data for multi-spectral SLAM testing."""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    PointCloudLoader,
    SpectralDataLoader,
    FeatureExtractor,
)


def generate_scan_trajectory(
    num_frames: int = 20,
    path_type: str = "spiral",
) -> list:
    """Generate sensor trajectory.

    Args:
        num_frames: Number of frames to generate.
        path_type: Type of trajectory ("spiral", "line", "circle").

    Returns:
        List of poses [x, y, theta].
    """
    poses = []
    if path_type == "spiral":
        for i in range(num_frames):
            angle = i * 0.3
            radius = 5 + i * 0.5
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            theta = angle + np.pi / 2
            poses.append([x, y, theta])
    elif path_type == "line":
        for i in range(num_frames):
            x = -10 + i * 1.0
            y = 0
            theta = 0
            poses.append([x, y, theta])
    elif path_type == "circle":
        for i in range(num_frames):
            angle = i * (2 * np.pi / num_frames)
            radius = 8
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            theta = angle + np.pi / 2
            poses.append([x, y, theta])
    return poses


def simulate_scan(
    original_features: list,
    pose: list,
    fov_angle: float = 60.0,
    max_range: float = 15.0,
    noise_std: float = 0.05,
) -> list:
    """Simulate a scan from a given pose.

    Args:
        original_features: Original feature list.
        pose: Sensor pose [x, y, theta].
        fov_angle: Field of view angle in degrees.
        max_range: Maximum range of sensor.
        noise_std: Standard deviation of noise.

    Returns:
        List of visible points.
    """
    sensor_x, sensor_y, sensor_theta = pose
    cos_t = np.cos(sensor_theta)
    sin_t = np.sin(sensor_theta)

    visible_points = []
    half_fov = fov_angle / 2

    for feature in original_features:
        dx = feature["x"] - sensor_x
        dy = feature["y"] - sensor_y

        range_val = np.sqrt(dx * dx + dy * dy)
        if range_val > max_range or range_val < 0.1:
            continue

        angle_to_point = np.arctan2(dy, dx)
        relative_angle = (angle_to_point - sensor_theta + np.pi) % (2 * np.pi) - np.pi

        if abs(relative_angle) > half_fov * np.pi / 180:
            continue

        local_x = dx * cos_t + dy * sin_t
        local_y = -dx * sin_t + dy * cos_t

        noisy_x = local_x + np.random.normal(0, noise_std)
        noisy_y = local_y + np.random.normal(0, noise_std)

        new_x = sensor_x + noisy_x * cos_t - noisy_y * sin_t
        new_y = sensor_y + noisy_x * sin_t + noisy_y * cos_t

        noisy_spectral = feature["spectral"] + np.random.normal(0, 50, len(feature["spectral"]))
        noisy_spectral = np.clip(noisy_spectral, 0, 10000)

        visible_points.append({
            "id": feature["id"],
            "x": new_x,
            "y": new_y,
            "spectral": noisy_spectral,
            "range": range_val,
            "angle": relative_angle,
        })

    return visible_points


def export_frame(
    frame_data: list,
    frame_idx: int,
    output_dir: str,
) -> None:
    """Export a single frame as PLY file.

    Args:
        frame_data: List of points in the frame.
        frame_idx: Frame index.
        output_dir: Output directory.
    """
    filepath = os.path.join(output_dir, f"frame_{frame_idx:03d}.ply")
    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(frame_data)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float range\n")
        f.write("property float angle\n")
        f.write("end_header\n")

        colors = plt.cm.viridis(np.linspace(0, 1, len(frame_data)))
        for i, point in enumerate(frame_data):
            r = int(colors[i][0] * 255)
            g = int(colors[i][1] * 255)
            b = int(colors[i][2] * 255)
            f.write(
                f"{point['x']:.4f} {point['y']:.4f} 0.0 "
                f"{r} {g} {b} {point['range']:.4f} {point['angle']:.4f}\n"
            )


import matplotlib.pyplot as plt


def export_poses(poses: list, output_path: str) -> None:
    """Export poses as text file.

    Args:
        poses: List of poses.
        output_path: Output file path.
    """
    with open(output_path, "w") as f:
        f.write("frame x y theta\n")
        for i, pose in enumerate(poses):
            f.write(f"{i} {pose[0]:.6f} {pose[1]:.6f} {pose[2]:.6f}\n")


def export_slam_dataset(
    features: list,
    num_frames: int = 20,
    output_dir: str = None,
    path_type: str = "spiral",
) -> None:
    """Export complete SLAM dataset.

    Args:
        features: Feature list with spectral data.
        num_frames: Number of frames to generate.
        output_dir: Output directory.
        path_type: Trajectory type.
    """
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "slam_data"
        )

    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print(f"Generating {num_frames} frames...")
    poses = generate_scan_trajectory(num_frames, path_type)

    for i, pose in enumerate(poses):
        frame_data = simulate_scan(features, pose)
        export_frame(frame_data, i, frames_dir)
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{num_frames} frames")

    poses_path = os.path.join(output_dir, "poses.txt")
    export_poses(poses, poses_path)

    print(f"\nSLAM dataset saved to: {output_dir}")
    print(f"  Frames: {frames_dir}")
    print(f"  Poses: {poses_path}")
    print(f"  Trajectory: {path_type}")
    print(f"  Total frames: {num_frames}")


def main() -> None:
    """Generate SLAM dataset."""
    data_dir = r"E:\11-control_system_of_lader\datas\20260130分类场景八万点"

    print("Loading original data...")
    point_loader = PointCloudLoader()
    point_cloud = point_loader.load(os.path.join(data_dir, "el_re.txt"))

    spectral_loader = SpectralDataLoader()
    spectral_data = spectral_loader.load(data_dir, "gp/specData")

    print("Extracting features...")
    extractor = FeatureExtractor()
    features = extractor.extract(point_cloud, spectral_data)

    print(f"\nGenerating SLAM dataset...")
    export_slam_dataset(
        features,
        num_frames=30,
        path_type="spiral",
    )

    print("\nYou can use this data to test:")
    print("  - Multi-spectral ICP registration")
    print("  - Loop closure detection")
    print("  - Graph-based SLAM")
    print("  - Spectral BA optimization")


if __name__ == "__main__":
    main()