import open3d.visualization.gui as gui

from logger import info
from ObbSelectWidget import ObbSelectWidget
from ObbSelectedWidget import ObbSelectedWidget

# <<< 应用：负责窗口和运行 >>>
class ObbSelectApplication:
    def __init__(self, pcd, patches):
        self.app_ = gui.Application.instance
        self.app_.initialize()

        self.pcd_ = pcd
        self.patches_ = patches

        select_window = self.app_.create_window("Choosing program -- by kerry", 1024, 768)
        select_window.set_on_close(self._callback_select_window_closed)
        self.widget_ = ObbSelectWidget(self.pcd_, self.patches_, select_window.renderer)
        self.widget_.set_on_select(self.open_new_window)
        select_window.add_child(self.widget_)

        self.selected_window_ = None
        self.selected_widget_ = None

        self.choosed_points_ = None

    def run(self):
        info("Application started.")
        self.app_.run()

    def open_new_window(self, obb):
        # 如果没有选择窗口，则创建
        if self.selected_window_ is None:
            info("Creating new selected window...")
            self.selected_window_ = self.app_.create_window("Selected OBB", 800, 600)
            self.selected_window_.set_on_close(self._callback_selected_window_closed)
            self.selected_widget_ = ObbSelectedWidget(self.pcd_, self.patches_)
            self.selected_window_.add_child(self.selected_widget_)
            info("Selected window has been created.")

        self.selected_widget_.config_window(self.selected_window_.renderer)
        self.selected_widget_.open_window(obb)
        self.selected_window_.show(True)

    def _callback_select_window_closed(self):
        info("Select window closed.")
        return True

    def _callback_selected_window_closed(self):
        info("Selected window closed.")
        self.choosed_points_ = self.selected_widget_.get_corner_points()
        self.widget_.set_choosed_points(self.choosed_points_)
        self.selected_window_ = None
        self.selected_widget_.clear()
        self.selected_widget_ = None
        return True

    def get_corner_points(self):
        return self.choosed_points_
# >>> ----------------- <<<
