# -*- coding: utf-8 -*-
import numpy as np
import open3d.visualization.gui as gui
import open3d.geometry as geometry
import open3d.utility as utility

import logger
from PlateSelectionWidgetor import PlateSelectionWidgetor
from FineTuningWidgetor import FineTuningWidgetor

class CalibrationApplicator:
    def __init__(self, points: np.array, colors: np.array):
        self.app_ = gui.Application.instance
        self.app_.initialize()

        # 把颜色归一化道 0 - 1 上
        colors = (colors - colors.min()) / (np.ptp(colors) + 1e-8)
        # 创造初始点云
        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(points)
        pcd.colors = utility.Vector3dVector([(color, color, color) for color in colors])

        # 对离群杂点剔除，用最近的 20 个点进行计算
        self.pcd_, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        # 用最近的 30 个点计算法线（用于 kdtree 搜索）
        self.pcd_.estimate_normals(search_param=geometry.KDTreeSearchParamKNN(knn=30))
        # 宽松拟合平面，粗略找寻所有平面
        self.obbs_ = self.pcd_.detect_planar_patches(normal_variance_threshold_deg=20,
                                                     coplanarity_deg=75,
                                                     outlier_ratio=0.2,
                                                     min_plane_edge_length=0,
                                                     min_num_points=0)

        # 创建 ‘挑选’ 窗口
        plate_selection_window = self.app_.create_window("Choosing", 1250, 1000)

        # 用于选择标定板所在平面的窗口
        self.plate_selection_widget_ = PlateSelectionWidgetor(self.pcd_, self.obbs_, plate_selection_window.renderer)
        # 设置选择事件会掉函数
        self.plate_selection_widget_.set_on_select(self.open_new_window)

        # 设置窗口关闭会掉函数
        plate_selection_window.set_on_close(self._callback_plate_selection_window_closed)
        # 把窗口添加到 window 持有者中
        plate_selection_window.add_child(self.plate_selection_widget_)

        # 微调窗口
        self.fine_funing_widget_ = None
        self.fine_funing_window_ = None

        # 被选择的点
        self.choosed_points_ = None

    def open_new_window(self, obb_idx):
        # 如果没有选择窗口，则创建
        if self.fine_funing_window_ is None:
            logger.info("Creating new selected window...")
            # 创建窗口内容
            self.fine_funing_widget_ = FineTuningWidgetor(self.pcd_, self.obbs_)

            # 创建窗口
            self.fine_funing_window_ = self.app_.create_window("Selected OBB", 800, 600)
            # 设置窗口关闭回调函数
            self.fine_funing_window_.set_on_close(self._callback_fine_funing_window_closed)
            # 把窗口内容添加到窗口中
            self.fine_funing_window_.add_child(self.fine_funing_widget_)
        
        # 配置窗口
        self.fine_funing_widget_.config_window(self.fine_funing_window_.renderer)
        self.fine_funing_widget_.open_window(obb_idx)

        # 设置窗口弹出模式
        self.fine_funing_window_.show(True)

    # 关闭窗口时返回允许
    def _callback_plate_selection_window_closed(self):
        logger.info("Select window has been closed.")
        return True

    # 关闭窗口时获取所有标定点并储存
    def _callback_fine_funing_window_closed(self):
        logger.info("Selected window has been closed.")
        self.choosed_points_ = self.fine_funing_widget_.get_corner_points()
        # 把标定点传入到选择窗口中显示在标定板所在位置上
        self.plate_selection_widget_.set_choosed_points(self.choosed_points_)
        # 清空窗口元素
        self.fine_funing_widget_ = None
        self.fine_funing_window_ = None
        return True

    # 提取角点
    def get_corner_points(self):
        return self.choosed_points_