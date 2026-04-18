import unittest

import numpy as np

from src.calibration_solver import compute_metrics


class CalibrationSolverTests(unittest.TestCase):
    def test_compute_metrics_uses_all_points_when_inlier_mask_is_none(self):
        image_points = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float64)
        reprojected_points = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        cam_points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float64)

        metrics = compute_metrics(image_points, reprojected_points, None, cam_points)

        self.assertEqual(metrics["num_points"], 2)
        self.assertEqual(metrics["num_inliers"], 2)
        self.assertAlmostEqual(metrics["rmse_px"], np.sqrt((0.0**2 + 5.0**2) / 2.0))

    def test_compute_metrics_returns_zero_rmse_for_perfect_projection(self):
        image_points = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        cam_points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float64)

        metrics = compute_metrics(image_points, image_points.copy(), np.array([True, True]), cam_points)

        self.assertEqual(metrics["rmse_px"], 0.0)
        self.assertEqual(metrics["max_px"], 0.0)
        self.assertEqual(metrics["depth_positive_ratio"], 1.0)

    def test_compute_metrics_returns_expected_keys(self):
        image_points = np.array([[1.0, 1.0]], dtype=np.float64)
        reprojected_points = np.array([[1.0, 1.0]], dtype=np.float64)
        cam_points = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

        metrics = compute_metrics(image_points, reprojected_points, np.array([True]), cam_points)

        self.assertEqual(
            set(metrics.keys()),
            {
                "rmse_px",
                "mean_px",
                "median_px",
                "p90_px",
                "max_px",
                "inlier_ratio_lt1px",
                "inlier_ratio_lt2px",
                "inlier_ratio_lt3px",
                "depth_positive_ratio",
                "match_score_0_100",
                "num_points",
                "num_inliers",
            },
        )


if __name__ == "__main__":
    unittest.main()
