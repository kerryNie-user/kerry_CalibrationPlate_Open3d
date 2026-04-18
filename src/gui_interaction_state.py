from dataclasses import dataclass


@dataclass
class FineTuningInteractionState:
    mouse_over_color: list
    selected_color: list
    highlighted_type: int = -1
    highlighted_idx: int = -1
    highlighted_color: list | None = None
    selected_type: int = -1
    selected_idx: int = -1
    mouse_origin_position: object = None
    sphere_origin_position: object = None
    rotation_locked: bool = False

    def __post_init__(self):
        if self.highlighted_color is None:
            self.highlighted_color = self.mouse_over_color

    def reset(self):
        self.highlighted_type = -1
        self.highlighted_idx = -1
        self.highlighted_color = self.mouse_over_color
        self.selected_type = -1
        self.selected_idx = -1
        self.mouse_origin_position = None
        self.sphere_origin_position = None
        self.rotation_locked = False

    def begin_drag(self, hit_type, hit_idx, mouse_origin_position, sphere_origin_position, click_color):
        self.highlighted_type = hit_type
        self.highlighted_idx = hit_idx
        self.highlighted_color = click_color
        self.mouse_origin_position = mouse_origin_position
        self.sphere_origin_position = sphere_origin_position
        self.rotation_locked = True

    def stop_drag_and_refresh_hover(self, hit_type, hit_idx):
        self.highlighted_type = hit_type
        self.highlighted_idx = hit_idx
        self.highlighted_color = self.mouse_over_color
        self.mouse_origin_position = None
        self.sphere_origin_position = None
        self.rotation_locked = False

    def clear_selection(self):
        self.selected_type = -1
        self.selected_idx = -1

    def select_hovered(self):
        self.selected_type = self.highlighted_type
        self.selected_idx = self.highlighted_idx
        self.highlighted_type = -1
        self.highlighted_idx = -1
