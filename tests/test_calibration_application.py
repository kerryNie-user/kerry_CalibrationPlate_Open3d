import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from CalibrationApplicator import CalibrationApplicator  # noqa: E402


class _FailingFineTuningWidget:
    def get_corner_points(self):
        raise RuntimeError("bad corners")


class _PlateSelectionWidget:
    def __init__(self):
        self.chosen_points = None

    def set_chosen_points(self, points):
        self.chosen_points = points


class CalibrationApplicationTests(unittest.TestCase):
    def test_failed_close_clears_fine_tuning_window_references(self):
        app = CalibrationApplicator.__new__(CalibrationApplicator)
        app.fine_tuning_widget_ = _FailingFineTuningWidget()
        app.fine_tuning_window_ = object()
        app.fine_funing_widget_ = app.fine_tuning_widget_
        app.fine_funing_window_ = app.fine_tuning_window_
        app.plate_selection_widget_ = _PlateSelectionWidget()
        app.chosen_points_ = ["keep-existing"]
        app.choosed_points_ = app.chosen_points_

        result = app._callback_fine_funing_window_closed()

        self.assertTrue(result)
        self.assertEqual(app.chosen_points_, ["keep-existing"])
        self.assertIsNone(app.fine_tuning_widget_)
        self.assertIsNone(app.fine_tuning_window_)
        self.assertIsNone(app.fine_funing_widget_)
        self.assertIsNone(app.fine_funing_window_)


if __name__ == "__main__":
    unittest.main()
