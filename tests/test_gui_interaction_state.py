import unittest

import numpy as np

from src.gui_interaction_state import FineTuningInteractionState


class GuiInteractionStateTests(unittest.TestCase):
    def test_reset_returns_to_idle_state(self):
        state = FineTuningInteractionState(mouse_over_color=[1.0, 0.0, 0.0, 1.0], selected_color=[0.0, 0.0, 1.0, 1.0])
        state.begin_drag(0, 2, np.array([10.0, 20.0]), np.array([1.0, 2.0]), [0.0, 1.0, 0.0, 1.0])
        state.select_hovered()

        state.reset()

        self.assertFalse(state.rotation_locked)
        self.assertEqual(state.highlighted_type, -1)
        self.assertEqual(state.highlighted_idx, -1)
        self.assertEqual(state.selected_type, -1)
        self.assertEqual(state.selected_idx, -1)
        self.assertIsNone(state.mouse_origin_position)
        self.assertIsNone(state.sphere_origin_position)

    def test_begin_drag_records_drag_context(self):
        state = FineTuningInteractionState(mouse_over_color=[1.0, 0.0, 0.0, 1.0], selected_color=[0.0, 0.0, 1.0, 1.0])

        state.begin_drag(1, 3, np.array([10.0, 20.0]), np.array([4.0, 5.0]), [0.0, 1.0, 0.0, 1.0])

        self.assertTrue(state.rotation_locked)
        self.assertEqual(state.highlighted_type, 1)
        self.assertEqual(state.highlighted_idx, 3)
        np.testing.assert_allclose(state.mouse_origin_position, np.array([10.0, 20.0]))
        np.testing.assert_allclose(state.sphere_origin_position, np.array([4.0, 5.0]))

    def test_select_hovered_promotes_hover_to_selection(self):
        state = FineTuningInteractionState(mouse_over_color=[1.0, 0.0, 0.0, 1.0], selected_color=[0.0, 0.0, 1.0, 1.0])
        state.highlighted_type = 0
        state.highlighted_idx = 1

        state.select_hovered()

        self.assertEqual(state.selected_type, 0)
        self.assertEqual(state.selected_idx, 1)
        self.assertEqual(state.highlighted_type, -1)
        self.assertEqual(state.highlighted_idx, -1)


if __name__ == "__main__":
    unittest.main()
