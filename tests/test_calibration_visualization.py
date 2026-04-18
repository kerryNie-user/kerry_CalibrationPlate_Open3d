import unittest

import numpy as np

from src.calibration_visualization import build_point_colors


class CalibrationVisualizationTests(unittest.TestCase):
    def test_build_point_colors_returns_empty_array_for_empty_input(self):
        colors = build_point_colors(np.array([], dtype=np.float64))

        self.assertEqual(colors.shape, (0, 3))

    def test_build_point_colors_returns_uint8_rgb_triplets(self):
        colors = build_point_colors(np.array([0.0, 0.5, 1.0], dtype=np.float64))

        self.assertEqual(colors.shape, (3, 3))
        self.assertEqual(colors.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
