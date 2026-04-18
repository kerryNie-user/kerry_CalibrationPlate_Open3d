import unittest

from src.board_spec import BoardSpec


class BoardSpecTests(unittest.TestCase):
    def test_inner_corners_map_to_grid_lines(self):
        spec = BoardSpec(inner_cols=9, inner_rows=6)

        self.assertEqual(spec.grid_cols, 10)
        self.assertEqual(spec.grid_rows, 7)
        self.assertEqual(spec.point_count, 54)

    def test_validate_point_count_accepts_matching_corner_count(self):
        spec = BoardSpec(inner_cols=9, inner_rows=6)

        self.assertEqual(spec.validate_point_count(54), 54)

    def test_validate_point_count_rejects_mismatch(self):
        spec = BoardSpec(inner_cols=9, inner_rows=6)

        with self.assertRaisesRegex(RuntimeError, "54"):
            spec.validate_point_count(53)


if __name__ == "__main__":
    unittest.main()
