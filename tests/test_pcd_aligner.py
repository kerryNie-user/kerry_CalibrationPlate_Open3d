import unittest

import numpy as np
import open3d.geometry as geometry
import open3d.utility as utility

from src.PcdAligner import PcdAligner


def make_point_cloud(points):
    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = utility.Vector3dVector(np.tile([[0.5, 0.5, 0.5]], (len(points), 1)))
    return pcd


class PcdAlignerTests(unittest.TestCase):
    def test_align_and_inverse_align_restore_points_with_explicit_transform(self):
        aligner = PcdAligner()
        pcd = make_point_cloud(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.1],
                [0.2, 0.4, 0.05],
            ]
        )

        aligned_pcd, transform = aligner.align(pcd)
        restored_pcd = aligner.inverse_align(aligned_pcd, transform)

        np.testing.assert_allclose(np.asarray(restored_pcd.points), np.asarray(pcd.points), atol=1e-6)

    def test_second_alignment_does_not_depend_on_first_alignment_history(self):
        aligner = PcdAligner()
        pcd_a = make_point_cloud(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.1],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.4],
                [1.0, 0.3, 0.8],
            ]
        )
        pcd_b = make_point_cloud(
            [
                [10.0, 5.0, 1.0],
                [11.0, 5.0, 1.7],
                [11.0, 6.0, 1.1],
                [10.0, 6.0, 2.1],
                [10.4, 5.2, 1.05],
            ]
        )

        _, _ = aligner.align(pcd_a)
        aligned_b, transform_b = aligner.align(pcd_b)
        restored_b = aligner.inverse_align(aligned_b, transform_b)

        np.testing.assert_allclose(np.asarray(restored_b.points), np.asarray(pcd_b.points), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
