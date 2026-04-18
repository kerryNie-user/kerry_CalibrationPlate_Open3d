import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.io_contract import (
    read_camera_corners,
    read_camera_params,
    read_lidar_cloud,
    read_lidar_corners,
    write_corner_csv,
)


class IoContractTests(unittest.TestCase):
    def test_read_camera_corners_supports_header_and_xy_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "camera_corner.csv"
            path.write_text("idx,u,v\n1,10.5,20.5\n2,30.0,40.0\n", encoding="utf-8")

            corners = read_camera_corners(path)

            np.testing.assert_allclose(corners, np.array([[10.5, 20.5], [30.0, 40.0]]))

    def test_read_lidar_corners_reads_xyz_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "lidar_corner.csv"
            path.write_text("x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n", encoding="utf-8")

            corners = read_lidar_corners(path)

            np.testing.assert_allclose(corners, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_read_lidar_cloud_supports_12_column_csv_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "lidar.csv"
            rows = [
                ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "x", "y", "z", "r"],
                ["0", "0", "0", "0", "0", "0", "0", "0", "1.0", "2.0", "3.0", "9.0"],
                ["0", "0", "0", "0", "0", "0", "0", "0", "4.0", "5.0", "6.0", "8.0"],
            ]
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)

            points, reflectivities = read_lidar_cloud(path)

            np.testing.assert_allclose(points, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            np.testing.assert_allclose(reflectivities, np.array([9.0, 8.0]))

    def test_read_lidar_cloud_supports_4_column_csv_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "lidar.csv"
            rows = [
                ["x", "y", "z", "r"],
                ["1.0", "2.0", "3.0", "9.0"],
                ["4.0", "5.0", "6.0", "8.0"],
            ]
            with path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)

            points, reflectivities = read_lidar_cloud(path)

            np.testing.assert_allclose(points, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
            np.testing.assert_allclose(reflectivities, np.array([9.0, 8.0]))

    def test_read_camera_params_supports_npz_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "camera_params.npz"
            camera_matrix = np.array([[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]])
            dist_coeffs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
            np.savez(path, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

            loaded_matrix, loaded_dist = read_camera_params(path)

            np.testing.assert_allclose(loaded_matrix, camera_matrix)
            np.testing.assert_allclose(loaded_dist, dist_coeffs)

    def test_read_camera_params_supports_yaml_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "camera_params.yaml"
            path.write_text(
                "camera_matrix: [[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]\n"
                "distortion_coefficients: [0.1, 0.2, 0.3, 0.4, 0.5]\n",
                encoding="utf-8",
            )

            loaded_matrix, loaded_dist = read_camera_params(path)

            np.testing.assert_allclose(
                loaded_matrix,
                np.array([[100.0, 0.0, 50.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]),
            )
            np.testing.assert_allclose(loaded_dist, np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]))

    def test_write_corner_csv_writes_xyz_header_and_points(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "lidar_corner.csv"

            write_corner_csv(path, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

            self.assertEqual(path.read_text(encoding="utf-8"), "x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n")


if __name__ == "__main__":
    unittest.main()
