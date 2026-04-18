import unittest

import numpy as np

from src.calibration_ordering import build_order_candidates, reorder_points


class CalibrationOrderingTests(unittest.TestCase):
    def setUp(self):
        self.points = np.arange(12, dtype=np.float64).reshape(6, 2)

    def test_reorder_points_keeps_order_without_transforms(self):
        reordered = reorder_points(self.points, cols=3, rows=2, transpose=False, flip_x=False, flip_y=False)

        np.testing.assert_allclose(reordered, self.points)

    def test_reorder_points_supports_transpose(self):
        reordered = reorder_points(self.points, cols=3, rows=2, transpose=True, flip_x=False, flip_y=False)

        expected = np.array(
            [
                [0.0, 1.0],
                [6.0, 7.0],
                [2.0, 3.0],
                [8.0, 9.0],
                [4.0, 5.0],
                [10.0, 11.0],
            ]
        )
        np.testing.assert_allclose(reordered, expected)

    def test_build_order_candidates_includes_identity_candidate(self):
        candidates = build_order_candidates(self.points, pattern_cols=3, pattern_rows=2)

        identity = next(
            c
            for c in candidates
            if not c["transpose"] and not c["flip_x"] and not c["flip_y"]
        )
        np.testing.assert_allclose(identity["points"], self.points)

    def test_build_order_candidates_falls_back_when_shape_does_not_match(self):
        points = np.arange(10, dtype=np.float64).reshape(5, 2)

        candidates = build_order_candidates(points, pattern_cols=3, pattern_rows=2)

        self.assertEqual(len(candidates), 1)
        np.testing.assert_allclose(candidates[0]["points"], points)


if __name__ == "__main__":
    unittest.main()
