import unittest

import numpy as np

from src.fine_tuning_validation import compute_intersection_grid, validate_corner_points


class FineTuningValidationTests(unittest.TestCase):
    def test_compute_intersection_grid_returns_expected_points(self):
        cols_points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 0.0]),
        ]
        rows_points = [
            np.array([0.0, 0.5, 0.0]),
            np.array([2.0, 0.5, 0.0]),
            np.array([0.0, 1.5, 0.0]),
            np.array([2.0, 1.5, 0.0]),
        ]

        points = compute_intersection_grid(cols_points, rows_points)

        expected = np.array(
            [
                [0.0, 0.5, 0.0],
                [0.0, 1.5, 0.0],
                [1.0, 0.5, 0.0],
                [1.0, 1.5, 0.0],
            ]
        )
        np.testing.assert_allclose(points, expected)

    def test_compute_intersection_grid_raises_when_lines_do_not_intersect(self):
        cols_points = [
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 2.0, 0.0]),
        ]
        rows_points = [
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 2.0, 0.0]),
        ]

        with self.assertRaisesRegex(RuntimeError, "交点"):
            compute_intersection_grid(cols_points, rows_points)

    def test_validate_corner_points_rejects_mismatched_count(self):
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])

        with self.assertRaisesRegex(RuntimeError, "角点数量"):
            validate_corner_points(points, expected_count=4)


if __name__ == "__main__":
    unittest.main()
