import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageCms

from color_calibration_core import CURRENT_WHITE, SOURCE_WHITE, WAVELENGTHS_NM


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_DIR / "calibrate_white_from_photo.py"
POINTCLOUD_OUTPUT = PROJECT_DIR / "pointcloud_output"

REFLECTANCE_COLUMNS = [f"rfl_{value}_nm" for value in WAVELENGTHS_NM]
REQUIRED_COLUMNS = [
    "scan_index",
    "x_mm",
    "y_mm",
    "z_mm",
    "azimuth_rad",
    "elevation_rad",
    "valid",
    *REFLECTANCE_COLUMNS,
]
PRESERVED_COLUMNS = [
    "scan_index",
    "x_mm",
    "y_mm",
    "z_mm",
    "azimuth_rad",
    "elevation_rad",
    "valid",
]
RGB_COLUMNS = [
    "rgb_physical_r",
    "rgb_physical_g",
    "rgb_physical_b",
    "rgb_photo_matched_r",
    "rgb_photo_matched_g",
    "rgb_photo_matched_b",
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_white_array(name, values):
    canonical_value = {
        "name": name,
        "wavelength_order_nm": [int(value) for value in WAVELENGTHS_NM],
        "values": [float(value) for value in values],
    }
    canonical_json = json.dumps(
        canonical_value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return canonical_value, canonical_json, hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()


class CalibrationCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.input_dir = self.root / "inputs"
        self.input_dir.mkdir()
        self.left_csv = self.input_dir / "left.csv"
        self.right_csv = self.input_dir / "right.csv"
        self.camera = self.input_dir / "camera.png"
        self.registration = self.input_dir / "registration.json"
        self.metadata = self.input_dir / "metadata.json"
        self.output = self.root / "calibration_output"
        self._write_camera()
        self._write_registration()
        self.metadata.write_text(
            json.dumps({"wavelength_order_nm": WAVELENGTHS_NM.tolist()}),
            encoding="utf-8",
        )
        rows = self._make_rows()
        self._write_csv(self.left_csv, rows[::2])
        self._write_csv(self.right_csv, rows[1::2])

    def tearDown(self):
        self.temporary.cleanup()

    def _write_camera(self):
        colors = np.array(
            [
                [230, 230, 230],
                [204, 13, 13],
                [13, 128, 13],
                [115, 64, 26],
                [5, 5, 5],
            ],
            dtype=np.uint8,
        )
        pixels = np.empty((100, 500, 3), dtype=np.uint8)
        for index, color in enumerate(colors):
            pixels[:, index * 100 : (index + 1) * 100] = color
        profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        Image.fromarray(pixels, mode="RGB").save(self.camera, icc_profile=profile)

    def _write_registration(self):
        self.registration.write_text(
            json.dumps(
                {
                    "wavelength_order_nm": WAVELENGTHS_NM.tolist(),
                    "orientation": "same",
                    "scan_control_points": [
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0],
                    ],
                    "photo_control_points_px": [
                        [0.0, 0.0],
                        [499.0, 0.0],
                        [499.0, 99.0],
                        [0.0, 99.0],
                    ],
                    "max_median_reprojection_error_px": 30.0,
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _make_rows():
        rows = []
        scan_index = 1
        for azimuth, elevation in ((0.0, 0.0), (1.0, 1.0)):
            rows.append(
                {
                    "scan_index": str(scan_index),
                    "x_mm": str(scan_index * 10),
                    "y_mm": str(scan_index * -3),
                    "z_mm": str(1000 + scan_index),
                    "azimuth_rad": str(azimuth),
                    "elevation_rad": str(elevation),
                    "valid": "0",
                    **{column: "0" for column in REFLECTANCE_COLUMNS},
                }
            )
            scan_index += 1
        class_positions = (0.1, 0.3, 0.5, 0.7, 0.9)
        for class_id, azimuth in enumerate(class_positions):
            for repetition in range(16):
                reflectance = [
                    0.12 + class_id * 0.04 + repetition * 0.0005,
                    0.09 + class_id * 0.02 + repetition * 0.0003,
                    0.15 + class_id * 0.03 + repetition * 0.0004,
                    0.10 + repetition * 0.0002,
                    0.11 + class_id * 0.025 + repetition * 0.0003,
                    0.13 + class_id * 0.035 + repetition * 0.0004,
                ]
                rows.append(
                    {
                        "scan_index": str(scan_index),
                        "x_mm": f"{scan_index * 10}.25",
                        "y_mm": f"{-scan_index * 3}.5",
                        "z_mm": f"{1000 + scan_index}.75",
                        "azimuth_rad": str(azimuth),
                        "elevation_rad": "0.5",
                        "valid": "1",
                        **{
                            column: f"{value:.9f}"
                            for column, value in zip(REFLECTANCE_COLUMNS, reflectance)
                        },
                    }
                )
                scan_index += 1
        return rows

    @staticmethod
    def _write_csv(path, rows, columns=None):
        columns = list(columns or REQUIRED_COLUMNS)
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in columns})

    def _command(self, mode="--fit", output=None, *extra):
        return [
            sys.executable,
            str(SCRIPT),
            mode,
            "--left-csv",
            str(self.left_csv),
            "--right-csv",
            str(self.right_csv),
            "--camera-photo",
            str(self.camera),
            "--registration-json",
            str(self.registration),
            "--metadata-json",
            str(self.metadata),
            "--output-root",
            str(output or self.output),
            *extra,
        ]

    def _run(self, mode="--fit", output=None, *extra):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_DIR)
        return subprocess.run(
            self._command(mode, output, *extra),
            cwd=PROJECT_DIR,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    @staticmethod
    def _accepted_document(rejected):
        accepted = json.loads(json.dumps(rejected))
        accepted.update(
            {
                "accepted": True,
                "automated_acceptance_passed": True,
                "visual_acceptance_explicitly_approved": True,
                "classification": "accepted_photo_fit",
                "status": [
                    "optimized_from_rgb",
                    "optimized_from_rgb",
                    "optimized_from_rgb",
                    "fixed_unobservable",
                    "optimized_from_rgb",
                    "optimized_from_rgb",
                ],
                "acceptance_reasons": [],
            }
        )
        accepted["automated_checks"] = {
            name: {**check, "passed": True}
            for name, check in rejected["automated_checks"].items()
        }
        return accepted

    def test_missing_required_column_names_the_column(self):
        rows = self._make_rows()[::2]
        columns = [column for column in REQUIRED_COLUMNS if column != "elevation_rad"]
        self._write_csv(self.left_csv, rows, columns)

        result = self._run()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("elevation_rad", result.stderr)
        self.assertFalse(self.output.exists())

    def test_rejects_cli_and_metadata_wavelength_order_mismatches(self):
        cli_result = self._run(
            "--prepare-registration",
            None,
            "--wavelength-order-nm",
            "495",
            "696",
            "600",
            "803",
            "545",
            "641",
        )
        self.assertNotEqual(cli_result.returncode, 0)
        self.assertIn("wavelength", cli_result.stderr.lower())

        self.metadata.write_text(
            json.dumps({"wavelength_order_nm": [495, 696, 600, 803, 642, 545]}),
            encoding="utf-8",
        )
        metadata_result = self._run("--prepare-registration")
        self.assertNotEqual(metadata_result.returncode, 0)
        self.assertIn("metadata", metadata_result.stderr.lower())
        self.assertIn("wavelength", metadata_result.stderr.lower())
        self.assertFalse(self.output.exists())

    def test_rejects_output_root_resolving_to_pointcloud_output(self):
        result = self._run("--prepare-registration", POINTCLOUD_OUTPUT)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pointcloud_output", result.stderr)

    def test_existing_output_requires_explicit_overwrite_and_keeps_unknown_files(self):
        self.output.mkdir()
        sentinel = self.output / "keep-me.txt"
        sentinel.write_text("unchanged", encoding="utf-8")

        rejected = self._run("--prepare-registration")
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("--overwrite-calibration-output", rejected.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")

        accepted = self._run(
            "--prepare-registration", None, "--overwrite-calibration-output"
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
        self.assertTrue((self.output / "common" / "angular_projection.png").is_file())

    def test_accept_fit_cannot_override_failed_automated_checks(self):
        result = self._run("--fit", None, "--accept-fit")

        self.assertEqual(result.returncode, 0, result.stderr)
        final = json.loads(
            (self.output / "white_fit" / "final_white_amplitudes.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(final["automated_acceptance_passed"])
        self.assertTrue(final["visual_acceptance_explicitly_approved"])
        self.assertFalse(final["accepted"])
        self.assertEqual(final["classification"], "rejected_photo_fit_candidate")
        self.assertNotIn(
            "visual_acceptance_not_explicitly_approved",
            final["acceptance_reasons"],
        )
        self.assertTrue(
            all(
                reason.startswith("automated_check_failed:")
                for reason in final["acceptance_reasons"]
            )
        )

    def test_writeback_guard_requires_consistent_explicit_acceptance(self):
        from calibrate_white_from_photo import require_accepted_white_amplitudes

        result = self._run("--fit")
        self.assertEqual(result.returncode, 0, result.stderr)
        rejected = json.loads(
            (self.output / "white_fit" / "final_white_amplitudes.json").read_text(
                encoding="utf-8"
            )
        )
        with self.assertRaisesRegex(ValueError, "accepted.*true"):
            require_accepted_white_amplitudes(rejected)

        top_level_tampered = {
            **rejected,
            "accepted": True,
            "automated_acceptance_passed": True,
            "visual_acceptance_explicitly_approved": True,
            "classification": "accepted_photo_fit",
        }
        with self.assertRaisesRegex(
            ValueError, "automated_checks|status|acceptance_reasons"
        ):
            require_accepted_white_amplitudes(top_level_tampered)

        accepted = self._accepted_document(rejected)
        geometry_without_actual_verification = json.loads(json.dumps(accepted))
        del geometry_without_actual_verification["automated_checks"][
            "geometry_and_row_counts_preserved"
        ]["verification"]
        with self.assertRaisesRegex(ValueError, "actual_output_csvs"):
            require_accepted_white_amplitudes(geometry_without_actual_verification)
        geometry_without_actual_structure = json.loads(json.dumps(accepted))
        del geometry_without_actual_structure["automated_checks"][
            "geometry_and_row_counts_preserved"
        ]["sides"]["left"]["white_fit"]["actual_output_rows"]
        with self.assertRaisesRegex(ValueError, "actual output geometry"):
            require_accepted_white_amplitudes(geometry_without_actual_structure)
        for field in (
            "automated_acceptance_passed",
            "visual_acceptance_explicitly_approved",
        ):
            inconsistent = {**accepted, field: False}
            with self.assertRaisesRegex(ValueError, field):
                require_accepted_white_amplitudes(inconsistent)
        with self.assertRaisesRegex(ValueError, "classification"):
            require_accepted_white_amplitudes(
                {**accepted, "classification": "rejected_photo_fit_candidate"}
            )
        with self.assertRaisesRegex(ValueError, "wavelength"):
            require_accepted_white_amplitudes(
                {**accepted, "wavelength_order_nm": [495, 696, 600, 803, 642, 545]}
            )
        invalid_white = list(accepted["white_amplitudes"])
        invalid_white[0] = -1.0
        with self.assertRaisesRegex(ValueError, "positive"):
            require_accepted_white_amplitudes(
                {**accepted, "white_amplitudes": invalid_white}
            )
        invalid_803 = list(accepted["white_amplitudes"])
        invalid_803[3] = 999.0
        with self.assertRaisesRegex(ValueError, "803"):
            require_accepted_white_amplitudes(
                {**accepted, "white_amplitudes": invalid_803}
            )
        values = require_accepted_white_amplitudes(accepted)
        np.testing.assert_array_equal(values, accepted["white_amplitudes"])

    def test_plot_failure_replaces_old_accepted_marker_with_in_progress_state(self):
        import calibrate_white_from_photo as calibration_cli

        initial = self._run("--fit")
        self.assertEqual(initial.returncode, 0, initial.stderr)
        final_path = self.output / "white_fit" / "final_white_amplitudes.json"
        rejected = json.loads(final_path.read_text(encoding="utf-8"))
        accepted = self._accepted_document(rejected)
        final_path.write_text(json.dumps(accepted), encoding="utf-8")
        calibration_cli.require_accepted_white_amplitudes(accepted)

        args = calibration_cli.parse_args(
            self._command(
                "--fit", None, "--overwrite-calibration-output"
            )[2:]
        )
        with mock.patch.object(
            calibration_cli,
            "_plot_comparison",
            side_effect=RuntimeError("injected final plotting failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected final plotting failure"):
                calibration_cli.fit(args)

        interrupted = json.loads(final_path.read_text(encoding="utf-8"))
        self.assertFalse(interrupted["accepted"])
        self.assertEqual(interrupted["classification"], "fit_in_progress")
        with self.assertRaisesRegex(ValueError, "accepted.*true"):
            calibration_cli.require_accepted_white_amplitudes(interrupted)

    def test_successful_fit_writes_final_completion_marker_after_last_plot(self):
        import calibrate_white_from_photo as calibration_cli

        events = []
        original_plot_comparison = calibration_cli._plot_comparison

        def record_state(output, document):
            events.append(document["classification"])
            calibration_cli._write_json(
                output,
                "white_fit/final_white_amplitudes.json",
                document,
            )

        def record_plot(*args, **kwargs):
            events.append("final_plot")
            return original_plot_comparison(*args, **kwargs)

        args = calibration_cli.parse_args(self._command("--fit")[2:])
        with mock.patch.object(
            calibration_cli, "_write_fit_state", side_effect=record_state, create=True
        ), mock.patch.object(
            calibration_cli, "_plot_comparison", side_effect=record_plot
        ):
            calibration_cli.fit(args)

        self.assertEqual(
            events,
            ["fit_in_progress", "final_plot", "rejected_photo_fit_candidate"],
        )
        final = json.loads(
            (self.output / "white_fit" / "final_white_amplitudes.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(final["completion_marker"], "fit_outputs_complete")

    def test_actual_output_verifier_detects_dropped_rows_and_changed_geometry(self):
        from calibrate_white_from_photo import (
            SafeOutput,
            _load_cloud,
            verify_exported_geometry_and_counts,
        )

        result = self._run("--fit")
        self.assertEqual(result.returncode, 0, result.stderr)

        white_left = self.output / "white_fit" / "left_pointcloud.csv"
        with white_left.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            white_fields = list(reader.fieldnames or [])
            white_rows = list(reader)
        with white_left.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=white_fields)
            writer.writeheader()
            writer.writerows(white_rows[:-1])

        matrix_right = self.output / "matrix_fit" / "right_pointcloud.csv"
        with matrix_right.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            matrix_fields = list(reader.fieldnames or [])
            matrix_rows = list(reader)
        matrix_rows[0]["x_mm"] = "tampered"
        with matrix_right.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=matrix_fields)
            writer.writeheader()
            writer.writerows(matrix_rows)

        clouds = [
            _load_cloud(self.left_csv, "left", SOURCE_WHITE),
            _load_cloud(self.right_csv, "right", SOURCE_WHITE),
        ]
        check = verify_exported_geometry_and_counts(
            clouds, SafeOutput(self.output)
        )

        self.assertFalse(check["passed"])
        self.assertEqual(check["verification"], "actual_output_csvs")
        self.assertEqual(
            check["sides"]["left"]["white_fit"]["actual_output_rows"],
            len(white_rows) - 1,
        )
        self.assertFalse(
            check["sides"]["left"]["white_fit"]["row_count_matches"]
        )
        self.assertFalse(
            check["sides"]["right"]["matrix_fit"]["preserved_fields_exact"]
        )

    def test_prepare_registration_accepts_finite_negative_reflectance_noise(self):
        rows = self._make_rows()[::2]
        valid_row = next(row for row in rows if row["valid"] == "1")
        valid_row["rfl_495_nm"] = "-0.01"
        self._write_csv(self.left_csv, rows)

        result = self._run("--prepare-registration")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.output / "common" / "angular_projection.png").is_file())

    def test_overwrite_rejects_nested_output_directory_symlink_escape(self):
        self.output.mkdir()
        external = self.root / "external-directory"
        external.mkdir()
        sentinel = external / "sentinel.txt"
        sentinel.write_text("unchanged", encoding="utf-8")
        (self.output / "common").symlink_to(external, target_is_directory=True)

        result = self._run(
            "--prepare-registration", None, "--overwrite-calibration-output"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr.lower())
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")
        self.assertFalse((external / "source_provenance.json").exists())

    def test_overwrite_rejects_output_leaf_symlink_to_source_file(self):
        self.output.mkdir()
        common = self.output / "common"
        common.mkdir()
        source_hash = sha256(self.left_csv)
        (common / "source_provenance.json").symlink_to(self.left_csv)

        result = self._run(
            "--prepare-registration", None, "--overwrite-calibration-output"
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr.lower())
        self.assertEqual(sha256(self.left_csv), source_hash)

    def test_fit_writes_complete_outputs_with_provenance_and_preserved_rows(self):
        input_hashes = {path.name: sha256(path) for path in (self.left_csv, self.right_csv)}

        result = self._run()

        self.assertEqual(result.returncode, 0, result.stderr)
        required_files = [
            "common/source_provenance.json",
            "common/registration.json",
            "common/anchor_statistics.json",
            "common/angular_projection.png",
            "white_fit/final_white_amplitudes.json",
            "white_fit/parameters.json",
            "white_fit/left_pointcloud.csv",
            "white_fit/right_pointcloud.csv",
            "white_fit/left_pointcloud.ply",
            "white_fit/right_pointcloud.ply",
            "white_fit/preview.png",
            "matrix_fit/parameters.json",
            "matrix_fit/left_pointcloud.csv",
            "matrix_fit/right_pointcloud.csv",
            "matrix_fit/left_pointcloud.ply",
            "matrix_fit/right_pointcloud.ply",
            "matrix_fit/preview.png",
            "comparison/camera_white_matrix.png",
            "comparison/metrics.json",
        ]
        for relative in required_files:
            path = self.output / relative
            self.assertTrue(path.is_file(), relative)
            self.assertGreater(path.stat().st_size, 0, relative)

        self.assertEqual(
            {path.name: sha256(path) for path in (self.left_csv, self.right_csv)},
            input_hashes,
        )
        for scheme in ("white_fit", "matrix_fit"):
            for side, source in (("left", self.left_csv), ("right", self.right_csv)):
                with source.open(newline="", encoding="utf-8") as handle:
                    source_rows = list(csv.DictReader(handle))
                with (self.output / scheme / f"{side}_pointcloud.csv").open(
                    newline="", encoding="utf-8"
                ) as handle:
                    output_rows = list(csv.DictReader(handle))
                self.assertEqual(len(output_rows), len(source_rows))
                for source_row, output_row in zip(source_rows, output_rows):
                    self.assertEqual(
                        [output_row[column] for column in PRESERVED_COLUMNS],
                        [source_row[column] for column in PRESERVED_COLUMNS],
                    )
                    for column in RGB_COLUMNS:
                        value = int(output_row[column])
                        self.assertGreaterEqual(value, 0)
                        self.assertLessEqual(value, 255)

        final = json.loads(
            (self.output / "white_fit" / "final_white_amplitudes.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(final["wavelength_order_nm"], WAVELENGTHS_NM.tolist())
        self.assertEqual(len(final["white_amplitudes"]), 6)
        self.assertEqual(len(final["status"]), 6)
        self.assertEqual(final["status"][3], "fixed_unobservable")
        self.assertEqual(final["white_amplitudes"][3], 361.225)
        self.assertTrue(np.all(np.isfinite(final["white_amplitudes"])))
        self.assertTrue(np.all(np.asarray(final["white_amplitudes"]) > 0.0))
        self.assertIs(type(final["accepted"]), bool)
        self.assertFalse(final["accepted"])
        self.assertEqual(final["classification"], "rejected_photo_fit_candidate")
        self.assertEqual(
            final["candidate_classification"],
            "photo_fitted_white_divisor_candidate",
        )
        self.assertTrue(final["visual_acceptance_required"])
        self.assertFalse(final["visual_acceptance_explicitly_approved"])
        self.assertFalse(final["automated_acceptance_passed"])
        required_checks = {
            "minimum_valid_classes",
            "minimum_improved_classes",
            "white_wall_channel_spread",
            "black_region_linear_luminance",
            "white_fit_gamut_clipping_fraction",
            "white_values_positive_finite",
            "fixed_803_nm",
            "geometry_and_row_counts_preserved",
        }
        self.assertEqual(set(final["automated_checks"]), required_checks)
        for check in final["automated_checks"].values():
            self.assertIs(type(check["passed"]), bool)
        geometry_check = final["automated_checks"][
            "geometry_and_row_counts_preserved"
        ]
        self.assertTrue(geometry_check["passed"])
        self.assertEqual(geometry_check["verification"], "actual_output_csvs")
        for side, source in (("left", self.left_csv), ("right", self.right_csv)):
            with source.open(newline="", encoding="utf-8") as handle:
                source_rows = list(csv.DictReader(handle))
            expected_valid = sum(float(row["valid"]) == 1.0 for row in source_rows)
            self.assertEqual(
                geometry_check["sides"][side]["source_rows"], len(source_rows)
            )
            self.assertEqual(
                geometry_check["sides"][side]["source_valid_count"],
                expected_valid,
            )
            for scheme in ("white_fit", "matrix_fit"):
                actual = geometry_check["sides"][side][scheme]
                self.assertEqual(actual["actual_output_rows"], len(source_rows))
                self.assertEqual(actual["actual_output_valid_count"], expected_valid)
                self.assertTrue(actual["preserved_fields_exact"])
        self.assertTrue(final["acceptance_reasons"])
        self.assertIn(
            "visual_acceptance_not_explicitly_approved",
            final["acceptance_reasons"],
        )
        self.assertTrue(
            any(reason.startswith("automated_check_failed:") for reason in final["acceptance_reasons"])
        )
        for index in (0, 1, 2, 4, 5):
            self.assertNotEqual(final["status"][index], "optimized_from_rgb")

        provenance = json.loads(
            (self.output / "common" / "source_provenance.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(provenance["source_white_amplitudes"], SOURCE_WHITE.tolist())
        self.assertEqual(provenance["optimization_center_white"], CURRENT_WHITE.tolist())
        for name, values in (
            ("source_white_amplitudes", SOURCE_WHITE),
            ("optimization_center_white", CURRENT_WHITE),
        ):
            canonical_value, canonical_json, expected_hash = canonical_white_array(
                name, values
            )
            entry = provenance["white_array_provenance"][name]
            self.assertEqual(entry["canonical_value"], canonical_value)
            self.assertEqual(entry["canonical_json_utf8"], canonical_json)
            self.assertEqual(entry["sha256"], expected_hash)
            self.assertEqual(entry["encoding"], "UTF-8 canonical JSON")
        expected_hashes = {
            "camera_photo": sha256(self.camera),
            "left_csv": sha256(self.left_csv),
            "right_csv": sha256(self.right_csv),
            "registration_json": sha256(self.registration),
            "script": sha256(SCRIPT),
        }
        self.assertEqual(
            {name: provenance["files"][name]["sha256"] for name in expected_hashes},
            expected_hashes,
        )

        render = self._run("--render-only")
        self.assertEqual(render.returncode, 0, render.stderr)


if __name__ == "__main__":
    unittest.main()
