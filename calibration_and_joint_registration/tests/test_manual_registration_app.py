import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms

from joint_registration_core import CameraModel, project_camera, zbuffer_visible
from manual_registration_core import (
    compose_manual_camera,
    default_manual_config,
    manual_config_document,
    parse_manual_config,
)
from manual_registration_app import ManualRegistrationPaths, create_app
import manual_registration_app


FRONTEND_ROOT = Path(__file__).resolve().parents[1] / "manual_registration_web"


class ManualRegistrationFrontendContractTest(unittest.TestCase):
    def _read_frontend(self, name):
        path = FRONTEND_ROOT / name
        self.assertTrue(path.is_file(), f"missing bundled frontend file: {path}")
        return path.read_text(encoding="utf-8")

    def test_frontend_bundles_semantic_canvas_and_local_assets(self):
        html = self._read_frontend("index.html")

        self.assertIn('<canvas id="registration-canvas"', html)
        self.assertIn('<link rel="stylesheet" href="/manual_registration_web/styles.css"', html)
        self.assertIn('<script type="module" src="/manual_registration_web/app.js"', html)
        self.assertIn('aria-label="手动照片配准画布"', html)
        self.assertIn('aria-disabled="true"', html)
        self.assertIn('id="base-status"', html)
        self.assertIn('id="config-warning"', html)
        self.assertIn("手动点云—照片配准", html)
        self.assertNotIn("<script src=\"http", html)
        self.assertNotIn("<link href=\"http", html)

    def test_frontend_exposes_geometry_display_history_and_export_controls(self):
        html = self._read_frontend("index.html")

        required_ids = (
            "translation-x",
            "translation-y",
            "translation-z",
            "rotation-x",
            "rotation-y",
            "rotation-z",
            "translation-x-slider",
            "translation-y-slider",
            "translation-z-slider",
            "rotation-x-slider",
            "rotation-y-slider",
            "rotation-z-slider",
            "projection-zoom",
            "show-left",
            "show-right",
            "point-opacity",
            "point-size",
            "photo-brightness",
            "undo-button",
            "redo-button",
            "reset-button",
            "save-button",
            "export-button",
            "save-status",
        )
        for control_id in required_ids:
            self.assertIn(f'id="{control_id}"', html)

        for metric_id in (
            "metric-in-frame",
            "metric-visible",
            "metric-coverage",
            "metric-control-median",
            "metric-control-max",
            "metric-focal",
        ):
            self.assertIn(f'id="{metric_id}"', html)

        for control_id in required_ids[:18]:
            self.assertIn(f'for="{control_id}"', html)

        for control_id in required_ids[:-6]:
            marker = f'id="{control_id}"'
            start = html.index(marker)
            self.assertIn("disabled", html[start : start + 240], control_id)

        for parameter in ("tx", "ty", "tz", "rx", "ry", "rz", "zoom"):
            self.assertIn(f'data-reset-parameter="{parameter}"', html)

    def test_frontend_implementation_contains_projection_history_and_api_hooks(self):
        css = self._read_frontend("styles.css")
        javascript = self._read_frontend("app.js")
        core = self._read_frontend("registration_core.mjs")
        combined = "\n".join((css, javascript, core))

        for remote_marker in ("http://", "https://", "@import url"):
            self.assertNotIn(remote_marker, combined)
        for api_path in (
            "/api/session",
            "/api/photo",
            "/api/save",
            "/api/metrics",
            "/api/export",
        ):
            self.assertIn(api_path, javascript)
        self.assertIn("导出点云", self._read_frontend("index.html"))
        for implementation_marker in (
            "rodriguesMatrix",
            "eulerXYZMatrix",
            "composeCamera",
            "projectPoints",
        ):
            self.assertIn(implementation_marker, core)
        for implementation_marker in (
            "requestAnimationFrame",
            "lostpointercapture",
            "base_status",
            "config_warning",
            "createDebouncedTask",
            "parseCommittedNumber",
        ):
            self.assertIn(implementation_marker, javascript)


class ManualRegistrationAppTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "joint_registration_output"
        self.output.mkdir()
        self.paths = self._write_fixture()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_cloud(self, path, offset):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("x_mm", "y_mm", "z_mm", "azimuth_rad", "elevation_rad", "valid"),
            )
            writer.writeheader()
            for index in range(8):
                writer.writerow(
                    {
                        "x_mm": -90.0 + index * 25.0 + offset,
                        "y_mm": -50.0 + index * 14.0,
                        "z_mm": 20.0 + index * 8.0,
                        "azimuth_rad": -0.4 + index * 0.1,
                        "elevation_rad": -0.3 + index * 0.08,
                        "valid": 1,
                    }
                )

    def _write_fixture(self):
        left_csv = self.inputs / "left.csv"
        right_csv = self.inputs / "right.csv"
        self._write_cloud(left_csv, 0.0)
        self._write_cloud(right_csv, 10.0)
        photo = self.inputs / "photo.png"
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
        Image.new("RGB", (80, 60), (120, 130, 140)).save(
            photo, icc_profile=profile.tobytes()
        )
        camera = self.output / "camera_model.json"
        camera.write_text(
            json.dumps(
                {
                    "model": "pinhole_equal_focal_with_k1",
                    "rotation_vector": [0.0, 0.0, 0.0],
                    "translation_mm": [0.0, 0.0, 1000.0],
                    "focal_px": 240.0,
                    "principal_point_px": [40.0, 30.0],
                    "radial_k1": 0.0,
                }
            ),
            encoding="utf-8",
        )
        transform = self.output / "right_to_left_transform.json"
        right_to_left = np.eye(4)
        right_to_left[:3, 3] = [7.0, -3.0, 2.0]
        transform.write_text(
            json.dumps(
                {"accepted": True, "reason": "accepted", "transform": right_to_left.tolist()}
            ),
            encoding="utf-8",
        )
        controls = self.inputs / "controls.json"
        controls.write_text(
            json.dumps(
                {
                    "scan_control_points": [[index / 7.0, index / 7.0] for index in range(6)],
                    "photo_control_points_px": [[40.0, 30.0] for _ in range(6)],
                }
            ),
            encoding="utf-8",
        )
        report = self.output / "registration_report.json"
        report.write_text(json.dumps({"accepted": True, "reason": "accepted"}), encoding="utf-8")
        return ManualRegistrationPaths(
            left_csv=left_csv,
            right_csv=right_csv,
            photo_path=photo,
            camera_model_path=camera,
            right_transform_path=transform,
            controls_path=controls,
            report_path=report,
            output_root=self.output,
        )

    def _client(self):
        app = create_app(self.paths)
        app.config.update(TESTING=True)
        return app.test_client()

    def _document(self):
        return manual_config_document(
            default_manual_config(), self.paths.camera_model_path.name, self.paths.right_transform_path.name
        )

    @staticmethod
    def _sha256(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def test_export_writes_authoritative_artifacts_and_preserves_inputs(self):
        document = self._document()
        document["translation_delta_mm"] = [12.0, -7.0, 18.0]
        document["manual_transform_4x4"][0][3] = 12.0
        document["manual_transform_4x4"][1][3] = -7.0
        document["manual_transform_4x4"][2][3] = 18.0
        sources = (
            self.paths.left_csv,
            self.paths.right_csv,
            self.paths.photo_path,
            self.paths.camera_model_path,
            self.paths.right_transform_path,
            self.paths.controls_path,
            self.paths.report_path,
        )
        hashes_before = {str(path.resolve()): self._sha256(path) for path in sources}

        response = self._client().post("/api/export", json=document)

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertTrue(result["exported"])
        expected_names = (
            "manual_photo_colored.ply",
            "manual_projection_overlay.png",
            "manual_camera_model.json",
            "manual_registration.json",
            "manual_export_report.json",
        )
        for name in expected_names:
            self.assertTrue((self.output / name).is_file(), name)
        ply = (self.output / "manual_photo_colored.ply").read_text(encoding="ascii")
        self.assertIn("element vertex 16", ply)
        self.assertIn("property uchar side", ply)
        self.assertIn("property uchar photo_color_valid", ply)
        expected_camera = compose_manual_camera(
            CameraModel(
                rotation_vector=np.array([0.0, 0.0, 0.0]),
                translation_mm=np.array([0.0, 0.0, 1000.0]),
                focal_px=240.0,
                principal_point_px=np.array([40.0, 30.0]),
                radial_k1=0.0,
            ),
            parse_manual_config(document),
        )
        camera = json.loads((self.output / "manual_camera_model.json").read_text(encoding="utf-8"))
        np.testing.assert_allclose(camera["rotation_vector"], expected_camera.rotation_vector)
        np.testing.assert_allclose(camera["translation_mm"], expected_camera.translation_mm)
        self.assertEqual(camera["focal_px"], expected_camera.focal_px)
        report = json.loads((self.output / "manual_export_report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["accepted"])
        self.assertEqual(report["input_hashes"], hashes_before)
        self.assertIn("config_hash", report)
        self.assertTrue(all(np.isfinite(value) for value in report["metrics"].values()))
        hashes_after = {str(path.resolve()): self._sha256(path) for path in sources}
        self.assertEqual(hashes_after, hashes_before)

    def test_export_failure_keeps_the_previous_complete_report(self):
        previous = b'{"accepted": true, "complete": true}\n'
        marker = self.output / "manual_export_report.json"
        marker.write_bytes(previous)
        original_atomic_json = manual_registration_app._atomic_json

        def fail_only_completion_marker(root_fd, name, document):
            if name == "manual_export_report.json":
                raise ValueError("simulated report failure")
            return original_atomic_json(root_fd, name, document)

        with patch.object(manual_registration_app, "_atomic_json", side_effect=fail_only_completion_marker):
            response = self._client().post("/api/export", json=self._document())

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["exported"])
        self.assertEqual(marker.read_bytes(), previous)
        for name in (
            "manual_photo_colored.ply",
            "manual_projection_overlay.png",
            "manual_camera_model.json",
            "manual_registration.json",
        ):
            self.assertTrue((self.output / name).is_file(), name)

    def test_export_rejects_input_changed_after_startup_before_writing_artifacts(self):
        client = self._client()
        marker = self.output / "manual_export_report.json"
        previous = b'{"accepted": true, "complete": true}\n'
        marker.write_bytes(previous)
        self.paths.left_csv.write_text(
            self.paths.left_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8"
        )

        response = client.post("/api/export", json=self._document())

        self.assertEqual(response.status_code, 409)
        self.assertFalse(response.get_json()["exported"])
        self.assertEqual(marker.read_bytes(), previous)
        for name in (
            "manual_photo_colored.ply",
            "manual_projection_overlay.png",
            "manual_camera_model.json",
            "manual_registration.json",
        ):
            self.assertFalse((self.output / name).exists(), name)

    def test_export_rejects_input_changed_before_completion_marker(self):
        client = self._client()
        marker = self.output / "manual_export_report.json"
        previous = b'{"accepted": true, "complete": true}\n'
        marker.write_bytes(previous)
        original_atomic_bytes = manual_registration_app._atomic_bytes

        def mutate_input_after_config(root_fd, name, content):
            result = original_atomic_bytes(root_fd, name, content)
            if name == "manual_registration.json":
                self.paths.controls_path.write_text(
                    self.paths.controls_path.read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )
            return result

        with patch.object(manual_registration_app, "_atomic_bytes", side_effect=mutate_input_after_config):
            response = client.post("/api/export", json=self._document())

        self.assertEqual(response.status_code, 409)
        self.assertFalse(response.get_json()["exported"])
        self.assertEqual(marker.read_bytes(), previous)

    def test_ply_contract_uses_bilinear_color_zbuffer_and_neutral_gray(self):
        photo = np.zeros((3, 3, 3), dtype=np.float64)
        for y in range(3):
            for x in range(3):
                photo[y, x] = [x / 10.0, y / 10.0, (x + y) / 10.0]
        projected = np.array([[0.25, 0.5], [1.2, 1.3], [1.2, 1.3], [-1.0, 0.0]])
        depth = np.array([10.0, 100.0, 5.0, 10.0])
        colors, valid = manual_registration_app._sample_photo(photo, projected, depth)
        ply = manual_registration_app._ply_bytes(
            np.arange(12, dtype=np.float64).reshape(4, 3),
            colors,
            np.array([0, 0, 1, 1], dtype=np.uint8),
            valid,
        ).decode("ascii")

        lines = ply.splitlines()
        header_end = lines.index("end_header")
        vertices = [line.split() for line in lines[header_end + 1 :]]
        self.assertEqual(len(vertices), 4)
        self.assertEqual(vertices[0][3:], ["6", "13", "19", "0", "1"])
        self.assertEqual(vertices[1][3:], ["115", "115", "115", "0", "0"])
        self.assertEqual(vertices[2][3:], ["31", "33", "64", "1", "1"])
        self.assertEqual(vertices[3][3:], ["115", "115", "115", "1", "0"])

    def test_session_contains_base_data_aligned_clouds_and_accepted_status(self):
        response = self._client().get("/api/session")

        self.assertEqual(response.status_code, 200)
        session = response.get_json()
        self.assertEqual(session["photo"], {"width": 80, "height": 60})
        self.assertTrue(session["base_status"]["accepted"])
        self.assertEqual(session["config"], self._document())
        self.assertEqual(len(session["left_points"]), 8)
        self.assertEqual(len(session["right_points"]), 8)
        self.assertEqual(len(session["controls"]["points_xyz"]), 6)
        self.assertEqual(len(session["controls"]["pixels_xy"]), 6)
        self.assertEqual(session["base_camera"]["focal_px"], 240.0)
        expected_right = np.array(
            [
                [-90.0 + index * 25.0 + 10.0 + 7.0, -50.0 + index * 14.0 - 3.0, 20.0 + index * 8.0 + 2.0]
                for index in range(8)
            ]
        )
        np.testing.assert_allclose(session["right_points"], expected_right)

    def test_startup_rejects_missing_inputs_rejected_report_and_output_inside_cloud_input(self):
        missing = self.paths.__class__(
            **{**self.paths.__dict__, "photo_path": self.inputs / "absent.png"}
        )
        with self.assertRaisesRegex(ValueError, "missing input"):
            create_app(missing)
        self.paths.report_path.write_text(json.dumps({"accepted": False, "reason": "bad fit"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bad fit"):
            create_app(self.paths)
        diagnostic = create_app(self.paths, diagnostic_mode=True)
        self.assertFalse(diagnostic.test_client().get("/api/session").get_json()["base_status"]["accepted"])
        nested = self.paths.__class__(
            **{**self.paths.__dict__, "output_root": self.inputs / "joint_registration_output"}
        )
        with self.assertRaisesRegex(ValueError, "input point-cloud directory"):
            create_app(nested, diagnostic_mode=True)

    def test_save_is_atomic_and_reloadable_and_rejects_bad_requests(self):
        client = self._client()
        document = self._document()
        document["translation_delta_mm"] = [10.0, 0.0, 0.0]
        document["manual_transform_4x4"][0][3] = 10.0
        document["photo_brightness"] = 1.25
        document["show_right"] = False
        saved = client.post("/api/save", json=document)
        self.assertEqual(saved.status_code, 200)
        target = self.output / "manual_registration.json"
        original = target.read_text(encoding="utf-8")
        self.assertEqual(create_app(self.paths).test_client().get("/api/session").get_json()["config"], document)
        self.assertEqual(client.post("/api/save", data="{}", content_type="text/plain").status_code, 415)
        self.assertEqual(client.post("/api/save", data="{", content_type="application/json").status_code, 400)
        self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_invalid_saved_config_warns_until_a_valid_repair_is_saved(self):
        target = self.output / "manual_registration.json"
        target.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "camera_delta": {
                        "translation_mm": [0, 0, 0],
                        "rotation_euler_xyz_degrees": [0, 0, 0],
                    },
                    "projection_zoom": 1,
                    "display": {
                        "show_left": True,
                        "show_right": True,
                        "point_opacity": 0.75,
                        "point_size": 3,
                        "photo_brightness": 1,
                    },
                    "base_camera_model": "camera_model.json",
                    "base_right_transform": "right_to_left_transform.json",
                }
            ),
            encoding="utf-8",
        )
        client = self._client()

        warned = client.get("/api/session").get_json()
        self.assertIn("config_warning", warned)
        self.assertEqual(warned["config"], self._document())

        repaired = client.post("/api/save", json=self._document())

        self.assertEqual(repaired.status_code, 200)
        self.assertNotIn("config_warning", client.get("/api/session").get_json())

    def test_save_refuses_a_symlink_target(self):
        client = self._client()
        target = self.output / "manual_registration.json"
        alternate = self.root / "alternate.json"
        alternate.write_text("unchanged", encoding="utf-8")
        target.symlink_to(alternate)

        response = client.post("/api/save", json=self._document())

        self.assertEqual(response.status_code, 400)
        self.assertEqual(alternate.read_text(encoding="utf-8"), "unchanged")

    def test_startup_requires_a_dedicated_non_symlink_output_root(self):
        not_dedicated = self.root / "manual-output"
        not_dedicated.mkdir()
        with self.assertRaisesRegex(ValueError, "joint_registration_output"):
            create_app(self.paths.__class__(**{**self.paths.__dict__, "output_root": not_dedicated}))

        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(self.root, target_is_directory=True)
        through_ancestor_link = linked_parent / "joint_registration_output"
        with self.assertRaisesRegex(ValueError, "symlink"):
            create_app(
                self.paths.__class__(
                    **{**self.paths.__dict__, "output_root": through_ancestor_link}
                )
            )

    def test_startup_rejects_symlink_or_directory_manual_target(self):
        real_output = self.root / "real-output"
        self.output.rename(real_output)
        self.output.symlink_to(real_output, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            create_app(self.paths)

        self.output.unlink()
        real_output.rename(self.output)
        target = self.output / "manual_registration.json"
        target.mkdir()
        with self.assertRaisesRegex(ValueError, "directory"):
            create_app(self.paths)

    def test_save_uses_the_opened_output_root_after_path_is_replaced(self):
        client = self._client()
        trusted_root = self.output
        moved_root = self.root / "moved-output-root"
        trusted_root.rename(moved_root)
        outside = self.root / "outside"
        outside.mkdir()
        trusted_root.symlink_to(outside, target_is_directory=True)

        response = client.post("/api/save", json=self._document())

        self.assertEqual(response.status_code, 200)
        self.assertTrue((moved_root / "manual_registration.json").is_file())
        self.assertFalse((outside / "manual_registration.json").exists())

    def test_save_rejects_a_directory_target_with_a_client_error(self):
        client = self._client()
        (self.output / "manual_registration.json").mkdir()

        response = client.post("/api/save", json=self._document())

        self.assertEqual(response.status_code, 400)

    def test_metrics_rejects_wrong_content_type_and_invalid_configuration(self):
        client = self._client()
        self.assertEqual(
            client.post("/api/metrics", data="{}", content_type="text/plain").status_code,
            415,
        )
        invalid = self._document()
        invalid["projection_zoom"] = 9.0
        self.assertEqual(client.post("/api/metrics", json=invalid).status_code, 400)

    def test_metrics_are_authoritative_for_the_submitted_configuration(self):
        document = self._document()
        document["translation_delta_mm"] = [20.0, -10.0, 0.0]
        document["manual_transform_4x4"][0][3] = 20.0
        document["manual_transform_4x4"][1][3] = -10.0
        client = self._client()
        session = client.get("/api/session").get_json()
        response = client.post("/api/metrics", json=document)

        self.assertEqual(response.status_code, 200)
        metrics = response.get_json()
        self.assertIn("in_frame_count", metrics)
        self.assertIn("visible_count", metrics)
        self.assertIn("coverage_fraction", metrics)
        self.assertIn("control_median_px", metrics)
        self.assertIn("control_max_px", metrics)
        base = CameraModel(
            rotation_vector=np.asarray(session["base_camera"]["rotation_vector"]),
            translation_mm=np.asarray(session["base_camera"]["translation_mm"]),
            focal_px=session["base_camera"]["focal_px"],
            principal_point_px=np.asarray(session["base_camera"]["principal_point_px"]),
            radial_k1=session["base_camera"]["radial_k1"],
        )
        camera = compose_manual_camera(base, parse_manual_config(document))
        points = np.vstack((session["left_points"], session["right_points"]))
        projected, depth = project_camera(points, camera)
        in_frame = (
            (depth > 0.0)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < 80.0)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < 60.0)
        )
        visible = zbuffer_visible(projected, depth, (80, 60), 40.0)
        control_projected, _ = project_camera(
            np.asarray(session["controls"]["points_xyz"]), camera
        )
        control_error = np.linalg.norm(
            control_projected - np.asarray(session["controls"]["pixels_xy"]), axis=1
        )
        self.assertEqual(metrics["in_frame_count"], int(np.count_nonzero(in_frame)))
        self.assertEqual(metrics["visible_count"], int(np.count_nonzero(visible)))
        self.assertEqual(metrics["coverage_fraction"], np.count_nonzero(visible) / len(points))
        self.assertEqual(metrics["control_median_px"], float(np.median(control_error)))
        self.assertEqual(metrics["control_max_px"], float(np.max(control_error)))
        self.assertEqual(metrics["focal_px"], camera.focal_px)


if __name__ == "__main__":
    unittest.main()
