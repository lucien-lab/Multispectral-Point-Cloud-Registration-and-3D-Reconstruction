import dataclasses
import json
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from joint_registration_core import CameraModel, project_camera
from manual_registration_core import (
    ManualRegistrationConfig,
    compose_manual_camera,
    default_manual_config,
    manual_config_document,
    manual_delta_transform,
    parse_manual_config,
)


class ManualRegistrationConfigurationTest(unittest.TestCase):
    def test_default_config_is_immutable_and_has_neutral_geometry(self):
        config = default_manual_config()

        self.assertIsInstance(config, ManualRegistrationConfig)
        self.assertEqual(config.schema_version, 1)
        np.testing.assert_array_equal(config.translation_mm, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(config.rotation_euler_xyz_degrees, [0.0, 0.0, 0.0])
        self.assertEqual(config.projection_zoom, 1.0)
        self.assertTrue(config.display.show_left)
        self.assertTrue(config.display.show_right)
        self.assertEqual(config.display.point_opacity, 0.75)
        self.assertEqual(config.display.point_size, 3.0)
        self.assertEqual(config.display.photo_brightness, 1.0)
        self.assertIsInstance(config.translation_mm, tuple)
        self.assertIsInstance(config.rotation_euler_xyz_degrees, tuple)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.projection_zoom = 1.1
        with self.assertRaises(AttributeError):
            config.translation_mm.setflags(write=True)
        with self.assertRaises(TypeError):
            config.translation_mm[0] = 2000.1

    def test_document_round_trip_preserves_config_and_base_names(self):
        source = {
            "schema_version": 1,
            "translation_delta_mm": [100.0, -20.0, 5.0],
            "rotation_delta_deg": [2.0, -1.0, 0.5],
            "manual_transform_4x4": np.block(
                [
                    [Rotation.from_euler("xyz", [2.0, -1.0, 0.5], degrees=True).as_matrix(), np.array([[100.0], [-20.0], [5.0]])],
                    [np.zeros((1, 3)), np.ones((1, 1))],
                ]
            ).tolist(),
            "projection_zoom": 1.25,
            "show_left": False,
            "show_right": True,
            "point_opacity": 0.6,
            "point_size": 3.5,
            "photo_brightness": 1.2,
            "base_camera_model": "camera_model.json",
            "base_right_transform": "right_to_left_transform.json",
        }

        config = parse_manual_config(source)
        document = manual_config_document(
            config, "camera_model.json", "right_to_left_transform.json"
        )

        self.assertEqual(document, source)
        self.assertEqual(json.loads(json.dumps(document, allow_nan=False)), source)
        reparsed = parse_manual_config(document)
        np.testing.assert_allclose(reparsed.translation_mm, config.translation_mm)
        np.testing.assert_allclose(
            reparsed.rotation_euler_xyz_degrees,
            config.rotation_euler_xyz_degrees,
        )
        self.assertEqual(reparsed.projection_zoom, config.projection_zoom)
        self.assertEqual(reparsed.display, config.display)

    def test_parse_rejects_unknown_keys(self):
        document = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        document["unexpected"] = 1

        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_manual_config(document)

    def test_parse_rejects_missing_top_level_keys_and_legacy_nested_config(self):
        for key in ("translation_delta_mm", "manual_transform_4x4", "point_size"):
            with self.subTest(key=key):
                document = manual_config_document(
                    default_manual_config(), "camera.json", "transform.json"
                )
                del document[key]
                with self.assertRaisesRegex(ValueError, "missing"):
                    parse_manual_config(document)

        legacy = {
            "schema_version": 1,
            "camera_delta": {
                "translation_mm": [0.0, 0.0, 0.0],
                "rotation_euler_xyz_degrees": [0.0, 0.0, 0.0],
            },
            "projection_zoom": 1.0,
            "display": {
                "show_left": True,
                "show_right": True,
                "point_opacity": 0.75,
                "point_size": 3.0,
                "photo_brightness": 1.0,
            },
            "base_camera_model": "camera.json",
            "base_right_transform": "transform.json",
        }
        with self.assertRaisesRegex(ValueError, "unknown"):
            parse_manual_config(legacy)

    def test_parse_requires_persisted_base_artifact_fields(self):
        for key in ("base_camera_model", "base_right_transform"):
            with self.subTest(key=key):
                document = manual_config_document(
                    default_manual_config(), "camera.json", "transform.json"
                )
                del document[key]
                with self.assertRaisesRegex(ValueError, "missing"):
                    parse_manual_config(document)

    def test_parse_rejects_non_finite_numbers_and_boolean_numbers(self):
        for value in (float("nan"), float("inf"), float("-inf"), True, False):
            with self.subTest(value=value):
                document = manual_config_document(
                    default_manual_config(), "camera.json", "transform.json"
                )
                document["projection_zoom"] = value
                with self.assertRaises(ValueError):
                    parse_manual_config(document)

    def test_parse_rejects_out_of_range_geometry_and_display_values(self):
        invalid_documents = []
        for value in (0.49, 2.01):
            document = manual_config_document(
                default_manual_config(), "camera.json", "transform.json"
            )
            document["projection_zoom"] = value
            invalid_documents.append(document)
        for value in (-2000.1, 2000.1):
            document = manual_config_document(
                default_manual_config(), "camera.json", "transform.json"
            )
            document["translation_delta_mm"][0] = value
            invalid_documents.append(document)
        for value in (-30.1, 30.1):
            document = manual_config_document(
                default_manual_config(), "camera.json", "transform.json"
            )
            document["rotation_delta_deg"][0] = value
            invalid_documents.append(document)
        for value in (-0.01, 1.01):
            document = manual_config_document(
                default_manual_config(), "camera.json", "transform.json"
            )
            document["point_opacity"] = value
            invalid_documents.append(document)
        document = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        document["point_size"] = 0.0
        invalid_documents.append(document)
        document = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        document["photo_brightness"] = 0.0
        invalid_documents.append(document)

        for document in invalid_documents:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    parse_manual_config(document)

    def test_parse_rejects_non_boolean_display_flags(self):
        for key in ("show_left", "show_right"):
            document = manual_config_document(
                default_manual_config(), "camera.json", "transform.json"
            )
            document[key] = 1
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    parse_manual_config(document)

    def test_parse_copies_input_arrays(self):
        document = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        config = parse_manual_config(document)
        document["translation_delta_mm"][0] = 999.0

        self.assertEqual(config.translation_mm[0], 0.0)

    def test_parse_rejects_non_rigid_or_inconsistent_manual_matrix(self):
        non_rigid = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        non_rigid["manual_transform_4x4"][0][0] = 2.0
        with self.assertRaisesRegex(ValueError, "rigid"):
            parse_manual_config(non_rigid)

        inconsistent = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        inconsistent["manual_transform_4x4"][0][3] = 1.0
        with self.assertRaisesRegex(ValueError, "consistent"):
            parse_manual_config(inconsistent)

        non_finite = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        non_finite["manual_transform_4x4"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            parse_manual_config(non_finite)


class ManualRegistrationCompositionTest(unittest.TestCase):
    def test_identity_base_applies_delta_translation_in_camera_coordinates(self):
        base = CameraModel(
            rotation_vector=np.zeros(3),
            translation_mm=np.zeros(3),
            focal_px=800.0,
            principal_point_px=np.array([320.0, 240.0]),
            radial_k1=0.04,
        )
        document = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        document["translation_delta_mm"] = [25.0, -50.0, 75.0]
        document["manual_transform_4x4"][0][3] = 25.0
        document["manual_transform_4x4"][1][3] = -50.0
        document["manual_transform_4x4"][2][3] = 75.0
        document["projection_zoom"] = 1.2
        config = parse_manual_config(document)

        delta = manual_delta_transform(config)
        camera = compose_manual_camera(base, config)

        np.testing.assert_allclose(delta[:3, :3], np.eye(3))
        np.testing.assert_allclose(delta[:3, 3], [25.0, -50.0, 75.0])
        np.testing.assert_allclose(camera.translation_mm, [25.0, -50.0, 75.0])
        np.testing.assert_allclose(camera.rotation_vector, [0.0, 0.0, 0.0])
        self.assertEqual(camera.focal_px, 960.0)
        np.testing.assert_allclose(camera.principal_point_px, base.principal_point_px)
        self.assertEqual(camera.radial_k1, base.radial_k1)

    def test_composition_left_multiplies_camera_frame_rotation_and_translation(self):
        base_rotation = Rotation.from_euler("xyz", [8.0, -4.0, 12.0], degrees=True)
        delta_rotation = Rotation.from_euler("xyz", [-3.0, 7.0, 2.5], degrees=True)
        base = CameraModel(
            rotation_vector=base_rotation.as_rotvec(),
            translation_mm=np.array([100.0, -40.0, 2500.0]),
            focal_px=1400.0,
            principal_point_px=np.array([1000.0, 750.0]),
            radial_k1=-0.02,
        )
        document = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        document["translation_delta_mm"] = [30.0, -10.0, 20.0]
        document["rotation_delta_deg"] = [-3.0, 7.0, 2.5]
        document["manual_transform_4x4"] = np.block(
            [
                [delta_rotation.as_matrix(), np.array([[30.0], [-10.0], [20.0]])],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        ).tolist()
        document["projection_zoom"] = 0.75
        config = parse_manual_config(document)

        camera = compose_manual_camera(base, config)
        expected_rotation = delta_rotation.as_matrix() @ base_rotation.as_matrix()
        expected_translation = (
            delta_rotation.as_matrix() @ base.translation_mm
            + config.translation_mm
        )

        np.testing.assert_allclose(
            Rotation.from_rotvec(camera.rotation_vector).as_matrix(), expected_rotation
        )
        np.testing.assert_allclose(camera.translation_mm, expected_translation)
        self.assertEqual(camera.focal_px, base.focal_px * config.projection_zoom)
        np.testing.assert_allclose(camera.principal_point_px, base.principal_point_px)
        self.assertEqual(camera.radial_k1, base.radial_k1)

    def test_projection_zoom_changes_intrinsics_without_scaling_xyz_geometry(self):
        base = CameraModel(
            rotation_vector=Rotation.from_euler(
                "xyz", [4.0, -3.0, 1.0], degrees=True
            ).as_rotvec(),
            translation_mm=np.array([20.0, -30.0, 1800.0]),
            focal_px=1000.0,
            principal_point_px=np.array([640.0, 480.0]),
            radial_k1=0.01,
        )
        document = manual_config_document(
            default_manual_config(), "camera.json", "transform.json"
        )
        document["projection_zoom"] = 1.5
        manual = compose_manual_camera(base, parse_manual_config(document))
        xyz = np.array([[0.0, 0.0, 0.0], [150.0, -50.0, 100.0]])

        base_pixels, base_depth = project_camera(xyz, base)
        manual_pixels, manual_depth = project_camera(xyz, manual)

        np.testing.assert_allclose(manual_depth, base_depth)
        np.testing.assert_allclose(manual.translation_mm, base.translation_mm)
        np.testing.assert_allclose(manual.rotation_vector, base.rotation_vector)
        np.testing.assert_allclose(
            manual_pixels - base.principal_point_px,
            1.5 * (base_pixels - base.principal_point_px),
        )


if __name__ == "__main__":
    unittest.main()
