import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from joint_registration_core import (
    CameraModel,
    apply_transform,
    initialize_camera,
    lift_angular_controls,
    photo_edge_distance,
    project_camera,
    refine_camera_multiscale,
    register_right_to_left,
    symmetric_nearest_metrics,
    zbuffer_visible,
)


class RigidTransformTest(unittest.TestCase):
    def test_apply_transform_uses_homogeneous_right_to_left_transform(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        transform = np.eye(4)
        transform[:3, 3] = [10.0, -2.0, 3.0]

        np.testing.assert_allclose(
            apply_transform(points, transform),
            [[10.0, -2.0, 3.0], [11.0, -2.0, 3.0]],
        )

    def test_symmetric_metrics_are_zero_for_identical_clouds(self):
        points = np.array(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [0.0, 100.0, 0.0]]
        )

        metrics = symmetric_nearest_metrics(points, points)

        self.assertEqual(metrics["symmetric_median_mm"], 0.0)
        self.assertEqual(metrics["symmetric_rmse_mm"], 0.0)
        self.assertEqual(metrics["source_count"], 3)
        self.assertEqual(metrics["target_count"], 3)

    def test_multiscale_icp_recovers_partial_overlap_with_outliers(self):
        rng = np.random.default_rng(20260722)
        grid = np.linspace(-1200.0, 1200.0, 22)
        a, b = np.meshgrid(grid, grid, indexing="ij")
        floor = np.column_stack((a.ravel(), b.ravel(), np.zeros(a.size)))
        wall_x = np.column_stack(
            (np.full(a.size, 650.0), a.ravel(), b.ravel() + 1200.0)
        )
        wall_y = np.column_stack(
            (a.ravel(), np.full(a.size, -900.0), b.ravel() + 900.0)
        )
        left = np.vstack((floor, wall_x, wall_y))
        left += rng.normal(0.0, 2.0, size=left.shape)

        true_transform = np.eye(4)
        true_transform[:3, :3] = Rotation.from_euler(
            "xyz", [1.0, -1.5, 2.0], degrees=True
        ).as_matrix()
        true_transform[:3, 3] = [120.0, -80.0, 35.0]
        inverse = np.linalg.inv(true_transform)
        overlap = left[rng.choice(len(left), size=int(len(left) * 0.68), replace=False)]
        right = overlap @ inverse[:3, :3].T + inverse[:3, 3]
        outliers = rng.uniform(
            low=[-4500.0, -4500.0, -1000.0],
            high=[4500.0, 4500.0, 4500.0],
            size=(int(len(right) * 0.15), 3),
        )
        right = np.vstack((right, outliers))

        result = register_right_to_left(left, right)

        self.assertTrue(result.accepted, result.reason)
        self.assertGreaterEqual(result.correspondence_count, 50)
        self.assertLess(
            result.final_metrics["symmetric_median_mm"],
            result.initial_metrics["symmetric_median_mm"],
        )
        np.testing.assert_allclose(
            result.transform[:3, 3], true_transform[:3, 3], atol=35.0
        )
        rotation_error = Rotation.from_matrix(
            result.transform[:3, :3] @ true_transform[:3, :3].T
        ).magnitude()
        self.assertLess(np.degrees(rotation_error), 1.0)

    def test_prealigned_parallel_surfaces_do_not_slide_into_non_overlap(self):
        x = np.linspace(1500.0, 9000.0, 45)
        y_left = np.linspace(-1200.0, 1800.0, 30)
        y_right = np.linspace(-4200.0, 500.0, 42)
        xx_l, yy_l = np.meshgrid(x, y_left, indexing="ij")
        xx_r, yy_r = np.meshgrid(x, y_right, indexing="ij")
        left_wall = np.column_stack((xx_l.ravel(), yy_l.ravel(), np.full(xx_l.size, 1800.0)))
        right_wall = np.column_stack((xx_r.ravel(), yy_r.ravel(), np.full(xx_r.size, 1800.0)))
        left_floor = np.column_stack((xx_l.ravel(), yy_l.ravel(), np.zeros(xx_l.size)))
        right_floor = np.column_stack((xx_r.ravel(), yy_r.ravel(), np.zeros(xx_r.size)))
        end_x, end_z = np.meshgrid(
            np.linspace(2500.0, 7500.0, 30),
            np.linspace(200.0, 2200.0, 18),
            indexing="ij",
        )
        end_wall = np.column_stack(
            (end_x.ravel(), np.full(end_x.size, 350.0), end_z.ravel())
        )
        left = np.vstack((left_wall, left_floor, end_wall))
        right_reference = np.vstack((right_wall, right_floor, end_wall))
        true_transform = np.eye(4)
        true_transform[:3, :3] = Rotation.from_euler("z", 0.35, degrees=True).as_matrix()
        true_transform[:3, 3] = [0.0, 30.0, 10.0]
        inverse = np.linalg.inv(true_transform)
        right = right_reference @ inverse[:3, :3].T + inverse[:3, 3]

        result = register_right_to_left(left, right)

        self.assertTrue(result.accepted, result.reason)
        self.assertLess(np.linalg.norm(result.transform[:3, 3]), 100.0)
        self.assertLess(
            result.final_metrics["symmetric_median_mm"],
            result.initial_metrics["symmetric_median_mm"],
        )


class CameraInitializationTest(unittest.TestCase):
    def test_project_camera_returns_pixels_and_signed_depth(self):
        camera = CameraModel(
            rotation_vector=np.zeros(3),
            translation_mm=np.array([0.0, 0.0, 2000.0]),
            focal_px=1200.0,
            principal_point_px=np.array([500.0, 400.0]),
            radial_k1=0.0,
        )
        points = np.array(
            [[0.0, 0.0, 0.0], [100.0, 50.0, 0.0], [0.0, 0.0, -3000.0]]
        )

        pixels, depth = project_camera(points, camera)

        np.testing.assert_allclose(pixels[:2], [[500.0, 400.0], [560.0, 430.0]])
        np.testing.assert_allclose(depth, [2000.0, 2000.0, -1000.0])

    def test_lift_angular_controls_selects_nearest_valid_xyz(self):
        azimuth = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        elevation = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        xyz = np.column_stack((azimuth * 10.0, elevation * 20.0, np.arange(5)))
        controls = np.array([[0.0, 0.0], [0.51, 0.49], [1.0, 1.0]])

        lifted = lift_angular_controls(azimuth, elevation, xyz, controls)

        np.testing.assert_allclose(lifted, xyz[[0, 2, 4]])

    def test_initialize_camera_recovers_noisy_non_coplanar_correspondences(self):
        rng = np.random.default_rng(20260722)
        points = rng.uniform(
            low=[-900.0, -700.0, -500.0],
            high=[900.0, 700.0, 600.0],
            size=(36, 3),
        )
        known = CameraModel(
            rotation_vector=Rotation.from_euler(
                "xyz", [5.0, -8.0, 2.0], degrees=True
            ).as_rotvec(),
            translation_mm=np.array([120.0, -60.0, 3800.0]),
            focal_px=1850.0,
            principal_point_px=np.array([1000.0, 750.0]),
            radial_k1=0.0,
        )
        pixels, depth = project_camera(points, known)
        self.assertTrue(np.all(depth > 0.0))
        noisy = pixels + rng.normal(0.0, 0.35, size=pixels.shape)

        result = initialize_camera(
            points,
            noisy,
            image_size=(2000, 1500),
            focal_bounds_px=(1000.0, 3500.0),
        )

        self.assertTrue(result.accepted, result.reason)
        self.assertLess(result.reprojection_median_px, 2.0)
        self.assertGreater(result.positive_depth_fraction, 0.8)
        self.assertLess(abs(result.camera.focal_px - known.focal_px), 100.0)


class VisibilityAndRefinementTest(unittest.TestCase):
    def test_zbuffer_keeps_nearer_point_at_same_pixel(self):
        projected = np.array([[10.2, 8.8], [10.3, 8.7], [4.0, 5.0]])
        depth = np.array([1000.0, 1500.0, 800.0])

        visible = zbuffer_visible(
            projected, depth, image_size=(20, 20), depth_tolerance_mm=25.0
        )

        np.testing.assert_array_equal(visible, [True, False, True])

    def test_photo_edge_distance_is_small_on_rectangle_boundary(self):
        photo = np.ones((100, 120, 3), dtype=float)
        photo[25:75, 30:90] = 0.0

        distance = photo_edge_distance(photo, output_size=(120, 100))

        self.assertLess(distance[25, 50], 1.1)
        self.assertGreater(distance[10, 10], 10.0)

    def test_multiscale_refinement_reduces_control_error(self):
        x = np.linspace(-700.0, 700.0, 18)
        y = np.linspace(-500.0, 500.0, 14)
        top = np.column_stack((x, np.full_like(x, -500.0), np.zeros_like(x)))
        bottom = np.column_stack((x, np.full_like(x, 500.0), np.zeros_like(x)))
        left = np.column_stack((np.full_like(y, -700.0), y, np.zeros_like(y)))
        right = np.column_stack((np.full_like(y, 700.0), y, np.zeros_like(y)))
        points = np.vstack((top, bottom, left, right))
        known = CameraModel(
            rotation_vector=np.zeros(3),
            translation_mm=np.array([0.0, 0.0, 3000.0]),
            focal_px=800.0,
            principal_point_px=np.array([320.0, 240.0]),
        )
        pixels, _ = project_camera(points, known)
        photo = np.ones((480, 640, 3), dtype=float)
        rounded = np.rint(pixels).astype(int)
        photo[np.clip(rounded[:, 1], 0, 479), np.clip(rounded[:, 0], 0, 639)] = 0.0
        from scipy.ndimage import binary_dilation
        ink = np.any(photo < 0.5, axis=2)
        photo[binary_dilation(ink, iterations=2)] = 0.0
        perturbed = CameraModel(
            rotation_vector=Rotation.from_euler("y", 1.0, degrees=True).as_rotvec(),
            translation_mm=np.array([60.0, -25.0, 3050.0]),
            focal_px=760.0,
            principal_point_px=np.array([320.0, 240.0]),
        )
        before_pixels, _ = project_camera(points, perturbed)
        before = np.median(np.linalg.norm(before_pixels - pixels, axis=1))

        result = refine_camera_multiscale(
            points,
            photo,
            controls_3d=points[::5],
            controls_2d=pixels[::5],
            initial_camera=perturbed,
            pyramid_widths=(320, 640),
        )
        after_pixels, _ = project_camera(points, result.camera)
        after = np.median(np.linalg.norm(after_pixels - pixels, axis=1))

        self.assertTrue(result.accepted, result.reason)
        self.assertLess(after, before)
        self.assertGreaterEqual(result.positive_depth_fraction, 0.95)


if __name__ == "__main__":
    unittest.main()
