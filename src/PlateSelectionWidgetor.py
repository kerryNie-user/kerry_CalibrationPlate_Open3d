import time
import numpy as np

import open3d.geometry as geometry
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import logger
from OperativeWidgetor import OperativeWidgetor

class PlateSelectionWidgetor(OperativeWidgetor):
    # 鼠标悬停时显示灰色
    MOUSE_OVER_COLOR = OperativeWidgetor.GREY
    # 鼠标点击时显示绿色
    MOUSE_CLICK_COLOR = OperativeWidgetor.GREEN
    # 被选择的角点会在最后显示为蓝色
    MOUSE_SELECT_COLOR = OperativeWidgetor.BLUE

    def __init__(self, pcd, patches, renderer):
        super().__init__()

        self.pcd_ = pcd
        self.obbs_ = patches

        # Scene 初始化
        self.scene = rendering.Open3DScene(renderer)
        # 背景设为白色
        self.scene.set_background([1, 1, 1, 1])
        # 不现实天空背景
        self.scene.show_skybox(False)
        # 不现实坐标轴
        self.scene.show_axes(False)

        # 加载点云
        self.visualize_pcd(self.pcd_, "total")
        logger.info("Point cloud has been added to the scene.")

        # 添加 OBB 线框
        for i, obb in enumerate(self.obbs_):
            self.visualize_obb(obb, str(i))
        logger.info(f"{len(self.obbs_)} OBB(s) have been added to the scene.")

        # 鼠标悬停的命中索引
        self.highlight_idx_ = self.NONE_HITTED_STATE
        # 被选择的索引
        self.selected_idx_ = self.NONE_HITTED_STATE

        # 选择角点后运行的回调函数
        self._callback_on_select = None
        # 被选择的角点
        self.choosed_points_ = None

        # 初始相机
        aabb = self.pcd_.get_axis_aligned_bounding_box()
        self.setup_camera(60.0, aabb, aabb.get_center())

        # 鼠标交互
        self.set_on_mouse(self._callback_mouse)
        logger.suggestion("Click on a calibration plate to select it.")

    # 鼠标事件处理
    def _callback_mouse(self, event):
        logger.debug(f"[{time.process_time()}]--------------------")
        logger.debug(f"Mouse event type: {event.type}")
        logger.debug(f"Selected obb == OBB[{self.selected_idx_}]")
        logger.debug(f"Highlighted obb == OBB[{self.highlight_idx_}]")

        # 鼠标点位
        mouse_position = np.array([event.x, event.y])

        # 如果发生拖动事件，不处理高亮和选择
        if event.type == gui.MouseEvent.Type.DRAG:
            self.highlight_idx_ = self.NONE_HITTED_STATE
            self.selected_idx_ = self.NONE_HITTED_STATE

        # 点按后记住点击的索引
        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self.selected_idx_ = self.highlight_idx_

        # 如果点击索引依然存在，运行鼠标回调函数
        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            if self.selected_idx_ != self.NONE_HITTED_STATE:
                self._callback_on_select(self.selected_idx_)
        
        # 鼠标悬停命中检测
        self.highlight_idx_ = self.detect_intersects(mouse_position)
        # 刷新高亮显示
        self.update_obb_highlight(self.highlight_idx_, self.MOUSE_OVER_COLOR, "highlighted_obb")
        self.update_obb_highlight(self.selected_idx_, self.MOUSE_CLICK_COLOR, "selected_obb")

        # 刷新角点显示
        if self.choosed_points_ is not None:
            self.visualize_spheres(self.choosed_points_, 0.01, self.MOUSE_SELECT_COLOR, "selected_spheres")

        return gui.Widget.EventCallbackResult.HANDLED

    # 高亮显示 OBB
    def update_obb_highlight(self, idx, color, name):
        self.scene.remove_geometry(name)

        if idx != self.NONE_HITTED_STATE:
            self.visualize_rectangle(self.obbs_[idx], color, name)
            logger.debug(f"Added {color} highlight to OBB[{idx}].")

    # 检测射线是否与 obb 有交集
    def ray_intersects_obb(self, ray_origin, ray_direct, obb: geometry.OrientedBoundingBox):
        R = obb.R.T
        ray_origin_local = R @ (ray_origin - obb.center)
        ray_direct_local = R @ ray_direct
        extents = obb.extent / 2.0
        
        # 检测每条坐标轴方向上射线与 OBB 边是否有交点，如果都有才算相交
        tmin, tmax = [], []
        for org, dir, ext in zip(ray_origin_local, ray_direct_local, extents):
            if abs(dir) < 1e-8:
                if abs(org) > ext:
                    return False, None
            else:
                t1 = (-ext - org) / dir
                t2 = (ext - org) / dir
                tmin.append(min(t1, t2))
                tmax.append(max(t1, t2))
        t_enter = max(tmin)
        t_exit = min(tmax)
        return (t_enter <= t_exit and t_exit >= 0), t_enter

    # 鼠标悬停命中检测
    def detect_intersects(self, mouse_position):
        # 获取鼠标位置的射线
        ray_origin, ray_direct = self.get_ray_of_mouse(mouse_position)

        # 记录最近的 OBB 索引
        hit_idx = self.NONE_HITTED_STATE
        min_distance = float("inf")
        for i, obb in enumerate(self.obbs_):
            hit, tmin = self.ray_intersects_obb(ray_origin, ray_direct, obb)
            if hit:
                logger.debug(f"Ray hit OBB[{i}] at distance {tmin}.")
                if tmin < min_distance:
                    hit_idx = i
                    min_distance = tmin
        return hit_idx

    # 设置选择回调函数
    def set_on_select(self, callback):
        self._callback_on_select = callback

    # 设置被选择的角点
    def set_choosed_points(self, points):
        self.choosed_points_ = points