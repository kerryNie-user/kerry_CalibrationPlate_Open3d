import numpy as np
import open3d.visualization.gui as gui
import open3d.geometry as geometry
import open3d.utility as utility

from logger import info
from ObbSelectWidgetor import ObbSelectWidgetor
from ObbSelectedWidgetor import ObbSelectedWidgetor

# <<< 应用：负责窗口和运行 >>>
class ObbSelectApplicator:
    def __init__(self, points: np.array, colors: np.array):
        self.app_ = gui.Application.instance
        self.app_.initialize()

        tensities = (colors - colors.min()) / (np.ptp(colors) + 1e-8)
        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(points)
        pcd.colors = utility.Vector3dVector([(tensity, tensity, tensity) for tensity in tensities])

        self.pcd_, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        self.pcd_.estimate_normals(search_param=geometry.KDTreeSearchParamKNN(knn=30))
        self.patches_ = self.pcd_.detect_planar_patches(
            normal_variance_threshold_deg=20,
            coplanarity_deg=75,
            outlier_ratio=0.2,
            min_plane_edge_length=0,
            min_num_points=0)

        select_window = self.app_.create_window("Choosing program -- by kerry", 1024, 768)

        self.select_widget_ = ObbSelectWidgetor(self.pcd_, self.patches_, select_window.renderer)
        self.select_widget_.set_on_select(self.open_new_window)

        select_window.set_on_close(self._callback_select_window_closed)
        select_window.add_child(self.select_widget_)

        self.selected_widget_ = None
        self.selected_window_ = None

        self.choosed_points_ = None

    def run(self):
        info("Application started.")
        self.app_.run()

    def open_new_window(self, obb_idx):
        # 如果没有选择窗口，则创建
        if self.selected_window_ is None:
            info("Creating new selected window...")
            self.selected_widget_ = ObbSelectedWidgetor(self.pcd_, self.patches_)

            self.selected_window_ = self.app_.create_window("Selected OBB", 800, 600)
            self.selected_window_.set_on_close(self._callback_selected_window_closed)
            self.selected_window_.add_child(self.selected_widget_)
            
        self.selected_widget_.config_window(self.selected_window_.renderer)
        self.selected_widget_.open_window(obb_idx)

        self.selected_window_.show(True)

    def _callback_select_window_closed(self):
        info("Select window has been closed.")
        return True

    def _callback_selected_window_closed(self):
        info("Selected window has been closed.")
        self.choosed_points_ = self.selected_widget_.get_corner_points()
        self.select_widget_.set_choosed_points(self.choosed_points_)
        self.selected_widget_ = None
        self.selected_window_ = None
        return True

    def get_corner_points(self):
        return self.choosed_points_
# >>> ----------------- <<<
