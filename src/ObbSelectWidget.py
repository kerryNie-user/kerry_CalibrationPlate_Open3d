import time
import numpy as np
import threading

import open3d.geometry as geometry
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from logger import suggestion, info, debug
from OperativeWidget import OperativeWidget

# <<< 场景部件：用于从模型所有平面中人工选择标定板所在平面 >>>
class ObbSelectWidget(OperativeWidget):
    NONE_COLOR = [-1, -1, -1, -1]
    MOUSE_OVER_COLOR = [0.9, 0.9, 0.9, 1.0]
    MOUSE_CLICK_COLOR = [0.0, 1.0, 0.0, 1.0]
    MOUSE_SELECT_COLOR = [0.0, 0.0, 1.0, 1.0]

    def __init__(self, pcd, patches, renderer):
        super().__init__()
        self.pcd_ = pcd
        self.patches_ = patches

        # Scene 初始化
        self.scene = rendering.Open3DScene(renderer)
        self.scene.set_background([1, 1, 1, 1])
        self.scene.show_skybox(False)
        self.scene.show_axes(False)

        # 加载点云
        self.visualize_pcd(self.pcd_, "total")
        info("Point cloud has been added to the scene.")

        # 添加 OBB 线框
        for i, obb in enumerate(self.patches_):
            self.visualize_obb(obb, str(i))
        info(f"{len(self.patches_)} OBB(s) have been added to the scene.")

        # 高亮状态
        self.highlight_idx_ = self.NONE_HITTED_STATE
        self.selected_idx_ = self.NONE_HITTED_STATE

        self._callback_on_select = None

        self.choosed_points_ = None

        # 初始相机
        aabb = self.pcd_.get_axis_aligned_bounding_box()
        self.setup_camera(60.0, aabb, aabb.get_center())

        # 鼠标交互
        self.set_on_mouse(self._callback_mouse)
        suggestion("Click on a calibration plate to select it.")

    # 鼠标事件处理
    def _callback_mouse(self, event):
        debug(f"----------[Window Total]----------")
        debug(f"-------[Time : {time.process_time()}]-------")
        debug(f"Mouse event type: {event.type}")

        debug(f"Selected obb == OBB[{self.selected_idx_}]")
        debug(f"Highlighted obb == OBB[{self.highlight_idx_}]")

        mouse_position = np.array([event.x, event.y])

        if event.type == gui.MouseEvent.Type.DRAG:
            self.highlight_idx_ = self.NONE_HITTED_STATE
            self.selected_idx_ = self.NONE_HITTED_STATE

        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self.selected_idx_ = self.highlight_idx_

        elif event.type == gui.MouseEvent.Type.BUTTON_UP:
            if self.selected_idx_ != self.NONE_HITTED_STATE:
                self._callback_on_select(self.selected_idx_)
                    
        self.highlight_idx_ = self.detect_intersects(mouse_position)
        self.update_obb_highlight(self.highlight_idx_, self.MOUSE_OVER_COLOR, "highlighted_obb")
        self.update_obb_highlight(self.selected_idx_, self.MOUSE_CLICK_COLOR, "selected_obb")

        if self.choosed_points_ is not None:
            self.visualize_spheres(self.choosed_points_, 0.01, self.MOUSE_SELECT_COLOR, "selected_spheres")

        return gui.Widget.EventCallbackResult.HANDLED

    # 高亮显示 OBB
    def update_obb_highlight(self, idx, color, name):
        self.scene.remove_geometry(name)

        if idx != self.NONE_HITTED_STATE:
            self.visualize_rectangle(self.patches_[idx], color, name)
            debug(f"Added {color} highlight to OBB[{idx}].")

    # 检测射线是否与 obb 有交集
    def ray_intersects_obb(self, ray_origin, ray_direct, obb: geometry.OrientedBoundingBox):
        R = obb.R.T
        ray_origin_local = R @ (ray_origin - obb.center)
        ray_direct_local = R @ ray_direct
        extents = obb.extent / 2.0

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
        ray_origin, ray_direct = self.get_ray_of_mouse(mouse_position)

        hit_idx = self.NONE_HITTED_STATE
        min_distance = float("inf")
        for i, obb in enumerate(self.patches_):
            hit, tmin = self.ray_intersects_obb(ray_origin, ray_direct, obb)
            if hit:
                debug(f"Ray hit OBB[{i}] at distance {tmin}.")
                if tmin < min_distance:
                    hit_idx = i
                    min_distance = tmin
        return hit_idx

    def set_on_select(self, callback):
        self._callback_on_select = callback

    def set_choosed_points(self, points):
        self.choosed_points_ = points
# >>> ------------------------------------------- <<<
