import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from CalibrationApplicator import CalibrationApplicator, CalibrationApplication  # noqa: E402
from FineTuningWidgetor import FineTuningWidget, FineTuningWidgetor  # noqa: E402
from OperativeWidgetor import OperativeWidget, OperativeWidgetor  # noqa: E402
from PlateSelectionWidgetor import PlateSelectionWidget, PlateSelectionWidgetor  # noqa: E402


class GuiNamingTests(unittest.TestCase):
    def test_new_class_names_alias_existing_classes(self):
        self.assertIs(CalibrationApplication, CalibrationApplicator)
        self.assertIs(FineTuningWidget, FineTuningWidgetor)
        self.assertIs(OperativeWidget, OperativeWidgetor)
        self.assertIs(PlateSelectionWidget, PlateSelectionWidgetor)

    def test_plate_selection_supports_new_and_legacy_chosen_point_setters(self):
        widget = PlateSelectionWidget.__new__(PlateSelectionWidget)

        widget.set_chosen_points([1, 2, 3])
        self.assertEqual(widget.chosen_points_, [1, 2, 3])

        widget.set_choosed_points([4, 5, 6])
        self.assertEqual(widget.chosen_points_, [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
