import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.calibration_reporting import classify_score, save_results


class CalibrationReportingTests(unittest.TestCase):
    def test_classify_score_maps_score_ranges(self):
        self.assertEqual(classify_score(90.0), "优秀")
        self.assertEqual(classify_score(75.0), "良好")
        self.assertEqual(classify_score(60.0), "可用")
        self.assertEqual(classify_score(10.0), "偏弱")

    def test_save_results_writes_report_and_detail_csv(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            transform = np.eye(4, dtype=np.float64)
            metrics = {
                "match_score_0_100": 88.0,
                "rmse_px": 0.5,
                "mean_px": 0.5,
                "median_px": 0.5,
                "p90_px": 0.5,
                "max_px": 0.5,
                "inlier_ratio_lt1px": 1.0,
                "inlier_ratio_lt2px": 1.0,
                "inlier_ratio_lt3px": 1.0,
                "depth_positive_ratio": 1.0,
                "num_points": 2,
                "num_inliers": 2,
            }
            inlier_mask = np.array([True, False])
            reprojected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
            order_config = {
                "source_cols": 9,
                "source_rows": 6,
                "transpose": False,
                "flip_x": False,
                "flip_y": False,
            }
            ranking = [{"rmse_px": 0.5, "score_0_100": 88.0}]

            save_results(output_dir, transform, metrics, inlier_mask, reprojected, order_config, ranking)

            report = json.loads((output_dir / "calibration_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["quality_level"], "优秀")
            self.assertEqual(report["best_order_config"]["source_cols"], 9)
            self.assertTrue((output_dir / "cam_lidar_extrinsic.npz").exists())
            self.assertTrue((output_dir / "reprojection_detail.csv").exists())


if __name__ == "__main__":
    unittest.main()
