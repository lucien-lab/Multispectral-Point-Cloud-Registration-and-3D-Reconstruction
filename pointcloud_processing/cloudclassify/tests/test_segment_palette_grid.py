import tempfile
import unittest
from pathlib import Path

import numpy as np


class GridBasicsTests(unittest.TestCase):
    def test_grid_labels_are_row_major_and_one_based(self):
        from segment_palette_grid import grid_label

        self.assertEqual(grid_label(0, 0), 1)
        self.assertEqual(grid_label(0, 3), 4)
        self.assertEqual(grid_label(5, 0), 21)
        self.assertEqual(grid_label(5, 3), 24)

    def test_each_cell_has_equal_x_and_z_side_length(self):
        from segment_palette_grid import SquareGrid

        grid = SquareGrid(
            x0=-0.07, z_top=0.01, side=0.012, pitch_x=0.015, pitch_z=0.015
        )
        xmin, xmax, zmin, zmax = grid.cell_bounds(2, 1)
        self.assertAlmostEqual(xmax - xmin, 0.012)
        self.assertAlmostEqual(zmax - zmin, 0.012)

    def test_load_palette_validates_declared_count_and_column_count(self):
        from segment_palette_grid import load_palette

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.txt"
            path.write_text("//X Y Z R G B A B\n2\n0 0 0 1 2 3 4 5\n1 1 1 6 7 8 9 10\n")
            header, points = load_palette(path)
        self.assertEqual(header, "//X Y Z R G B A B")
        self.assertEqual(points.shape, (2, 8))

    def test_grid_cell_rejects_gaps_and_accepts_square_interior(self):
        from segment_palette_grid import SquareGrid, point_to_grid_cell

        grid = SquareGrid(
            x0=0.0, z_top=0.04, side=0.01, pitch_x=0.012, pitch_z=0.012
        )
        self.assertEqual(point_to_grid_cell(grid, 0.005, 0.035), (0, 0))
        self.assertIsNone(point_to_grid_cell(grid, 0.0108, 0.035))
        self.assertIsNone(point_to_grid_cell(grid, -0.001, 0.035))

    def test_fit_palette_grid_recovers_a_four_by_six_square_layout(self):
        from segment_palette_grid import fit_palette_grid, point_to_grid_cell

        points = []
        for row in range(6):
            for col in range(4):
                points.append(
                    [
                        0.01 + 0.015 * col,
                        1.0,
                        0.05 - 0.015 * row,
                        40 + 15 * col,
                        80 + 10 * row,
                        180 - 10 * col,
                        0,
                        0,
                    ]
                )
        points.extend([[0.4, 0.7, 0.4, 0, 0, 0, 0, 0]] * 10)
        grid, plane = fit_palette_grid(np.asarray(points, dtype=float))
        self.assertAlmostEqual(plane.predict(0.01, 0.05), 1.0)
        self.assertGreater(plane.tolerance, 0.0)
        self.assertAlmostEqual(grid.side, 0.012, places=6)
        self.assertEqual(point_to_grid_cell(grid, 0.01, 0.05), (0, 0))
        self.assertEqual(point_to_grid_cell(grid, 0.055, -0.025), (5, 3))

    def test_assignment_uses_zero_for_background_and_one_to_24_for_cells(self):
        from segment_palette_grid import FrontPlane, SquareGrid, assign_labels

        grid = SquareGrid(
            x0=0.0, z_top=0.04, side=0.01, pitch_x=0.012, pitch_z=0.012
        )
        points = np.array(
            [
                [0.005, 1.007, 0.035, 10, 20, 30, 0, 0],
                [0.029, 1.0022, 0.011, 40, 50, 60, 0, 0],
                [0.0108, 1.007, 0.035, 10, 20, 30, 0, 0],
                [0.005, 1.0, 0.035, 10, 20, 30, 0, 0],
            ]
        )
        plane = FrontPlane(x_coefficient=0.0, z_coefficient=0.2,
                           intercept=1.0, tolerance=0.001)
        labels = assign_labels(points, grid, plane)
        np.testing.assert_array_equal(labels, np.array([1, 11, 0, 0]))

    def test_writer_adds_one_label_column(self):
        from segment_palette_grid import write_labeled_cloud

        header = "//X Y Z R G B A B"
        points = np.array([[0, 0, 0, 1, 2, 3, 4, 5]], dtype=float)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.txt"
            write_labeled_cloud(output, header, points, np.array([7]))
            with output.open(encoding="utf-8") as handle:
                self.assertEqual(handle.readline().strip(), header + " label")
                self.assertEqual(handle.readline().strip(), "1")
            loaded = np.loadtxt(output, comments="//", skiprows=2)
        self.assertEqual(loaded.shape, (9,))
        self.assertEqual(int(loaded[-1]), 7)

    def test_ideal_sample_positions_form_a_seven_by_seven_square_grid(self):
        from segment_palette_grid import SquareGrid, ideal_sample_positions

        grid = SquareGrid(
            x0=0.0, z_top=0.08, side=0.07, pitch_x=0.08, pitch_z=0.08
        )
        targets = ideal_sample_positions(grid, 0, 0)
        self.assertEqual(targets.shape, (49, 2))
        self.assertEqual(len(np.unique(targets[:, 0])), 7)
        self.assertEqual(len(np.unique(targets[:, 1])), 7)

    def test_regular_assignment_caps_a_full_cell_at_49_points(self):
        from segment_palette_grid import SquareGrid, assign_regular_labels

        grid = SquareGrid(
            x0=0.0, z_top=0.08, side=0.07, pitch_x=0.08, pitch_z=0.08
        )
        axis = np.linspace(0.001, 0.069, 8)
        x, z = np.meshgrid(axis, 0.08 - axis)
        points = np.zeros((64, 8))
        points[:, 0] = x.ravel()
        points[:, 2] = z.ravel()
        labels = assign_regular_labels(points, grid)
        self.assertEqual(np.count_nonzero(labels == 1), 49)
        self.assertEqual(np.count_nonzero(labels), 49)

    def test_real_regular_labels_have_requested_counts(self):
        from segment_palette_grid import fit_palette_grid, labels_for_output, load_palette

        _, points = load_palette(Path(__file__).parents[1] / "palette.txt")
        grid, _ = fit_palette_grid(points)
        counts = np.bincount(labels_for_output(points, grid), minlength=25)
        self.assertTrue(np.all(counts[1:] <= 49))
        self.assertTrue(np.all((counts[1:] >= 47) | (counts[1:] == 44)))
        self.assertEqual(counts[20], 44)


if __name__ == "__main__":
    unittest.main()
