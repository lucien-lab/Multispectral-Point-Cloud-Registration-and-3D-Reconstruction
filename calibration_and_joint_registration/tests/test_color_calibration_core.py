import base64
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageCms

from color_calibration_core import (
    CalibrationSamples,
    CURRENT_WHITE,
    MatrixFitResult,
    SOURCE_WHITE,
    VISIBLE_INDICES,
    WAVELENGTHS_NM,
    WhiteFitResult,
    apply_matrix_fit,
    build_anchor_samples,
    delta_e76,
    fit_regularized_matrix,
    fit_homography,
    linear_srgb_to_lab,
    load_camera_srgb,
    normalized_scan_coordinates,
    optimize_white,
    project_homogeneous,
    registration_error,
    reflectance_from_white,
    restore_amplitudes,
    spectral_to_linear_srgb,
    white_objective,
)


DISPLAY_P3_ICC = base64.b64decode(
    "AAACGGFwcGwEAAAAbW50clJHQiBYWVogB+YAAQABAAAAAAAAYWNzcEFQUEwAAAAAQVBQTAAAAAAAAAAAAAAAAAAAAAAAAPbWAAEAAAAA0y1hcHBsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKZGVzYwAAAPwAAAAwY3BydAAAASwAAABQd3RwdAAAAXwAAAAUclhZWgAAAZAAAAAUZ1hZWgAAAaQAAAAUYlhZWgAAAbgAAAAUclRSQwAAAcwAAAAgY2hhZAAAAewAAAAsYlRSQwAAAcwAAAAgZ1RSQwAAAcwAAAAgbWx1YwAAAAAAAAABAAAADGVuVVMAAAAUAAAAHABEAGkAcwBwAGwAYQB5ACAAUAAzbWx1YwAAAAAAAAABAAAADGVuVVMAAAA0AAAAHABDAG8AcAB5AHIAaQBnAGgAdAAgAEEAcABwAGwAZQAgAEkAbgBjAC4ALAAgADIAMAAyADJYWVogAAAAAAAA9tUAAQAAAADTLFhZWiAAAAAAAACD3wAAPb////+7WFlaIAAAAAAAAEq/AACxNwAACrlYWVogAAAAAAAAKDgAABELAADIuXBhcmEAAAAAAAMAAAACZmYAAPKnAAANWQAAE9AAAApbc2YzMgAAAAAAAQxCAAAF3v//8yYAAAeTAAD9kP//+6L///2jAAAD3AAAwG4="
)


class CalibrationConstantsTest(unittest.TestCase):
    def test_fixed_wavelength_order_and_visible_indices(self):
        np.testing.assert_array_equal(
            WAVELENGTHS_NM, [495, 696, 600, 803, 545, 642]
        )
        self.assertEqual(VISIBLE_INDICES, (0, 1, 2, 4, 5))

    def test_source_reflectance_round_trip(self):
        rfl = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=float)
        amplitudes = restore_amplitudes(rfl)
        recovered = reflectance_from_white(amplitudes, SOURCE_WHITE)
        np.testing.assert_allclose(recovered, rfl, rtol=0, atol=1e-12)

    def test_current_white_is_positive_and_803_is_fixed(self):
        self.assertTrue(np.all(CURRENT_WHITE > 0))
        self.assertAlmostEqual(float(CURRENT_WHITE[3]), 361.2250)


class ColorMathTest(unittest.TestCase):
    def test_black_maps_to_black(self):
        rgb = spectral_to_linear_srgb(np.zeros((1, 6)))
        np.testing.assert_allclose(rgb, 0.0, atol=0.0)

    def test_lab_delta_is_zero_for_identical_colors(self):
        lab = linear_srgb_to_lab(np.array([[0.2, 0.3, 0.4]]))
        np.testing.assert_allclose(delta_e76(lab, lab), 0.0, atol=1e-12)

    def test_803_does_not_change_visible_rgb(self):
        a = np.array([[0.2, 0.2, 0.2, 0.0, 0.2, 0.2]])
        b = a.copy()
        b[:, 3] = 1000.0
        np.testing.assert_allclose(
            spectral_to_linear_srgb(a), spectral_to_linear_srgb(b), atol=0.0
        )


class CameraRegistrationTest(unittest.TestCase):
    def test_load_camera_converts_embedded_display_p3_to_srgb(self):
        pixels = np.array(
            [[[128, 64, 32], [40, 128, 210]], [[200, 80, 20], [16, 180, 96]]],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "display-p3-camera.png"
            Image.fromarray(pixels, mode="RGB").save(
                image_path, icc_profile=DISPLAY_P3_ICC
            )
            with Image.open(image_path) as source:
                reference = ImageCms.profileToProfile(
                    source.convert("RGB"),
                    ImageCms.ImageCmsProfile(BytesIO(source.info["icc_profile"])),
                    ImageCms.createProfile("sRGB"),
                    outputMode="RGB",
                )
            expected = np.asarray(reference, dtype=np.float64) / 255.0
            image = load_camera_srgb(image_path)

        self.assertEqual(image.shape, (2, 2, 3))
        self.assertEqual(image.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(image)))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        np.testing.assert_allclose(image, expected, rtol=0.0, atol=0.0)
        self.assertGreater(float(np.max(np.abs(expected - pixels / 255.0))), 0.0)
        self.assertFalse(np.array_equal(image, pixels / 255.0))

    def test_normalized_scan_coordinates_maps_extents_to_unit_square(self):
        azimuth = np.array([-10.0, 0.0, 10.0])
        elevation = np.array([5.0, 10.0, 15.0])
        np.testing.assert_allclose(
            normalized_scan_coordinates(azimuth, elevation),
            [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        )

    def test_fit_homography_projects_control_points_and_has_low_error(self):
        scan = np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], dtype=float
        )
        photo = np.array(
            [[100, 200], [900, 180], [920, 700], [80, 720], [500, 450]], dtype=float
        )
        transform = fit_homography(scan[:4], photo[:4])
        predicted = project_homogeneous(scan, transform)
        np.testing.assert_allclose(predicted[:4], photo[:4], atol=1e-8)
        self.assertLess(registration_error(transform, scan[:4], photo[:4]), 1e-8)


class AnchorExtractionTest(unittest.TestCase):
    PHOTO_COLORS = np.array(
        [
            [0.90, 0.90, 0.90],  # white
            [0.80, 0.05, 0.05],  # red
            [0.05, 0.50, 0.05],  # green
            [0.45, 0.25, 0.10],  # brown
            [0.02, 0.02, 0.02],  # black
        ],
        dtype=np.float64,
    )

    @classmethod
    def _striped_photo_samples(cls, counts):
        height = 80
        strip_width = 120
        photo = np.empty((height, strip_width * len(counts), 3), dtype=np.float64)
        amplitudes = []
        projected_xy = []
        input_class_ids = []
        for class_id, count in enumerate(counts):
            photo[:, class_id * strip_width : (class_id + 1) * strip_width] = (
                cls.PHOTO_COLORS[class_id]
            )
            positions = [
                (x + 0.25, y + 0.5)
                for y in range(5, height - 5)
                for x in range(class_id * strip_width + 5, (class_id + 1) * strip_width - 5)
            ]
            for x, y in positions[:count]:
                row_id = len(amplitudes) + 1
                amplitudes.append([row_id, 2, 3, 4, 5, 6])
                projected_xy.append([x, y])
                input_class_ids.append(class_id)
        return (
            np.asarray(amplitudes, dtype=np.float64),
            np.asarray(projected_xy, dtype=np.float64),
            photo,
            np.asarray(input_class_ids, dtype=np.int32),
        )

    def test_balances_all_photo_classes_at_500_with_fixed_sampling_seed(self):
        counts = [520, 530, 540, 550, 560]
        amplitudes, projected_xy, photo, input_class_ids = self._striped_photo_samples(
            counts
        )

        first = build_anchor_samples(amplitudes, projected_xy, photo)
        second = build_anchor_samples(amplitudes, projected_xy, photo)

        self.assertIsInstance(first, CalibrationSamples)
        np.testing.assert_array_equal(first.amplitudes, second.amplitudes)
        np.testing.assert_array_equal(first.target_linear_rgb, second.target_linear_rgb)
        np.testing.assert_array_equal(first.class_ids, second.class_ids)
        unique, selected_counts = np.unique(first.class_ids, return_counts=True)
        np.testing.assert_array_equal(unique, np.arange(5))
        np.testing.assert_array_equal(selected_counts, np.full(5, 500))

        rng = np.random.default_rng(20260722)
        expected_rows = []
        for class_id in range(5):
            candidates = np.flatnonzero(input_class_ids == class_id)
            expected_rows.extend(rng.choice(candidates, size=500, replace=False))
        np.testing.assert_array_equal(
            first.amplitudes[:, 0], amplitudes[np.asarray(expected_rows), 0]
        )

    def test_rejects_class_with_fewer_than_15_valid_point_cloud_rows(self):
        amplitudes, projected_xy, photo, input_class_ids = self._striped_photo_samples(
            [20, 20, 20, 20, 20]
        )
        black_rows = np.flatnonzero(input_class_ids == 4)
        amplitudes[black_rows[:6], 2] = -1.0

        samples = build_anchor_samples(amplitudes, projected_xy, photo)

        unique, selected_counts = np.unique(samples.class_ids, return_counts=True)
        np.testing.assert_array_equal(unique, np.arange(4))
        np.testing.assert_array_equal(selected_counts, np.full(4, 20))


class WhiteOptimizationTest(unittest.TestCase):
    def test_recovers_visible_channel_ratios_and_keeps_803_fixed(self):
        rng = np.random.default_rng(20260722)
        known = np.array([420.0, 330.0, 370.0, 361.2250, 480.0, 410.0])
        reflectance = rng.uniform(0.05, 0.8, size=(500, 6))
        amplitudes = reflectance * known
        target_linear_rgb = spectral_to_linear_srgb(reflectance)
        class_ids = np.repeat(np.arange(5), 100)
        samples = CalibrationSamples(amplitudes, target_linear_rgb, class_ids)

        result = optimize_white(samples)

        visible = np.array(VISIBLE_INDICES, dtype=np.int32)
        result_visible = result.white[visible]
        known_visible = known[visible]
        result_ratio = result_visible / np.exp(np.mean(np.log(result_visible)))
        known_ratio = known_visible / np.exp(np.mean(np.log(known_visible)))
        np.testing.assert_allclose(result_ratio, known_ratio, rtol=0.03)
        self.assertIsInstance(result, WhiteFitResult)
        self.assertEqual(float(result.white[3]), 361.2250)
        self.assertTrue(np.all(np.isfinite(result.white)))
        self.assertTrue(np.all(result.white > 0.0))
        self.assertTrue(np.isfinite(result.exposure))
        self.assertGreater(result.exposure, 0.0)
        self.assertTrue(np.isfinite(result.objective))
        self.assertEqual(set(result.class_metrics), {"white", "red", "green", "brown", "black"})
        self.assertAlmostEqual(
            result.objective,
            white_objective(np.log(result_visible), samples, CURRENT_WHITE),
            places=8,
        )


class RegularizedMatrixFitTest(unittest.TestCase):
    @staticmethod
    def _synthetic_samples(seed, count_per_class):
        rng = np.random.default_rng(seed)
        class_ids = np.repeat(np.arange(5), count_per_class)
        reflectance = rng.uniform(0.05, 0.8, size=(class_ids.size, 6))
        visible = reflectance[:, VISIBLE_INDICES]
        known_matrix = np.array(
            [
                [0.25, 0.04, 0.01],
                [0.03, 0.20, 0.02],
                [0.10, 0.10, 0.30],
                [0.05, 0.30, 0.04],
                [0.20, 0.05, 0.25],
            ],
            dtype=np.float64,
        )
        known_bias = np.array([[0.02, 0.01, 0.03]], dtype=np.float64)
        target_linear_rgb = visible @ known_matrix + known_bias
        return (
            CalibrationSamples(
                amplitudes=reflectance * CURRENT_WHITE,
                target_linear_rgb=target_linear_rgb,
                class_ids=class_ids,
            ),
            reflectance,
        )

    def test_recovers_linear_rgb_for_held_out_synthetic_reflectance(self):
        training, _ = self._synthetic_samples(seed=20260722, count_per_class=80)
        _, held_out_reflectance = self._synthetic_samples(
            seed=20260723, count_per_class=20
        )
        known_matrix = np.array(
            [
                [0.25, 0.04, 0.01],
                [0.03, 0.20, 0.02],
                [0.10, 0.10, 0.30],
                [0.05, 0.30, 0.04],
                [0.20, 0.05, 0.25],
            ],
            dtype=np.float64,
        )
        known_bias = np.array([[0.02, 0.01, 0.03]], dtype=np.float64)

        result = fit_regularized_matrix(training, CURRENT_WHITE)
        predicted = apply_matrix_fit(held_out_reflectance, result)
        expected = held_out_reflectance[:, VISIBLE_INDICES] @ known_matrix + known_bias

        self.assertIsInstance(result, MatrixFitResult)
        self.assertEqual(result.matrix.shape, (5, 3))
        self.assertEqual(result.bias.shape, (1, 3))
        self.assertIn(result.ridge_coefficient, (1e-4, 1e-3, 1e-2, 1e-1, 1.0))
        self.assertTrue(np.isfinite(result.validation_error))
        self.assertLess(float(np.mean(np.abs(predicted - expected))), 0.01)

    def test_803_nm_never_enters_matrix_prediction(self):
        samples, _ = self._synthetic_samples(seed=20260722, count_per_class=80)
        result = fit_regularized_matrix(samples, CURRENT_WHITE)
        reflectance = np.full((3, 6), 0.2, dtype=np.float64)
        altered_803 = reflectance.copy()
        altered_803[:, 3] = 1_000_000.0

        np.testing.assert_allclose(
            apply_matrix_fit(reflectance, result),
            apply_matrix_fit(altered_803, result),
            rtol=0.0,
            atol=0.0,
        )

    def test_rejects_nonfinite_reflectance(self):
        samples, _ = self._synthetic_samples(seed=20260722, count_per_class=80)
        result = fit_regularized_matrix(samples, CURRENT_WHITE)
        reflectance = np.full((1, 6), 0.2, dtype=np.float64)
        reflectance[0, 0] = np.nan

        with self.assertRaisesRegex(ValueError, "reflectance.*finite"):
            apply_matrix_fit(reflectance, result)

    def test_rejects_nonfinite_matrix(self):
        samples, _ = self._synthetic_samples(seed=20260722, count_per_class=80)
        result = fit_regularized_matrix(samples, CURRENT_WHITE)
        matrix = result.matrix.copy()
        matrix[0, 0] = np.inf
        invalid_result = MatrixFitResult(
            matrix=matrix,
            bias=result.bias,
            ridge_coefficient=result.ridge_coefficient,
            validation_error=result.validation_error,
        )

        with self.assertRaisesRegex(ValueError, "matrix.*finite"):
            apply_matrix_fit(np.full((1, 6), 0.2), invalid_result)

    def test_rejects_nonfinite_bias(self):
        samples, _ = self._synthetic_samples(seed=20260722, count_per_class=80)
        result = fit_regularized_matrix(samples, CURRENT_WHITE)
        bias = result.bias.copy()
        bias[0, 0] = np.nan
        invalid_result = MatrixFitResult(
            matrix=result.matrix,
            bias=bias,
            ridge_coefficient=result.ridge_coefficient,
            validation_error=result.validation_error,
        )

        with self.assertRaisesRegex(ValueError, "bias.*finite"):
            apply_matrix_fit(np.full((1, 6), 0.2), invalid_result)
