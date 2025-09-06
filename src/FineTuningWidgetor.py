import time
import numpy as np

import open3d.utility as utility
import open3d.geometry as geometry
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import logger
from PcdAligner import PcdAligner
import PlateExtraction
import PcdDimension
from OperativeWidgetor import OperativeWidgetor

PLATE_COLS = 10
PLATE_ROWS = 7

# <<< 场景部件：用于处理被选中的标定板展示界面 >>>
class FineTuningWidgetor(OperativeWidgetor):
    # 列索引
    COLS_HITTEN_STATE = 0
    # 行索引
    ROWS_HITTEN_STATE = 1

    # 鼠标悬停颜色
    MOUSE_OVER_COLOR = OperativeWidgetor.RED
    # 鼠标点击颜色
    MOUSE_CLICK_COLOR = OperativeWidgetor.GREEN
    # 鼠标选择颜色
    MOUSE_SELECT_COLOR = OperativeWidgetor.BLUE

    # 不允许旋转点云
    ROTATION_LOCK = 0
    # 允许旋转点云
    ROTATION_UNLOCK = 1

    # 增加键（ASICII）
    ADDER_KEY = 61
    # 减少键（ASICII）
    SUBER_KEY = 45

    def __init__(self, pcd: geometry.PointCloud, obbs: list[geometry.OrientedBoundingBox]):
        super().__init__()

        self.pcd_ = pcd
        self.obbs_ = obbs
        # 点云对齐器
        self.aligner_ = PcdAligner()

        # 所有标定板的角点，结构为 [2][num_point]
        self.sphere_centers_ = None

        # 点云旋转状态
        self.rotation_state_ = self.ROTATION_UNLOCK

        # 鼠标悬停索引
        self.highlighted_type_ = self.NONE_HITTED_STATE
        self.highlighted_idx_ = self.NONE_HITTED_STATE
        self.highlighted_color_ = self.NONE_COLOR

        # 选中索引
        self.selected_type_ = self.NONE_HITTED_STATE
        self.selected_idx_ = self.NONE_HITTED_STATE
        self.selected_color_ = self.MOUSE_SELECT_COLOR

         # 注释：本函数索引操纵球的方式为为 -> 先索引行列，再索引点，且相对行列的点在列表中交叉排列

        # 在拖动过程中，记录鼠标和拖动球的初始位置
        self.mouse_origin_position_ = None
        self.sphere_origin_position_ = None

        # 拖动球半径
        self.sphere_radius_ = 0.01
        # 拖动球离标定板的距离
        self.blank_distance_ = 0.05
        # 键盘操作缩放系数
        self.key_manu_scale_ = self.sphere_radius_ / 10

        self.set_on_mouse(self._callback_mouse)
        self.set_on_key(self._callback_key)

        logger.suggestion("Drag the sphere to coarsely tune it.")
        logger.suggestion("Use the keyboard to fine-tune:")
        logger.suggestion("    '+' : going right or up.")
        logger.suggestion("    '-' : going down or left.")

    # 配置窗口
    def config_window(self, renderer: rendering.OffscreenRenderer):
        if self.scene is None:
            # 如果不存在场景，创建场景
            if renderer is None:
                raise RuntimeError("Renderer is None, cannot initialize the scene.")
            logger.info("Selected window has been created.")
            self.scene = rendering.Open3DScene(renderer)
            self.scene.set_background([1, 1, 1, 1])
            self.scene.show_skybox(False)
            self.scene.show_axes(False)
        else:
            # 如果存在场景，清理场景内的运算信息
            logger.info("Selected window has been cleaned.")
            self.clean()

    # 处理点云并显示画面
    def open_window(self, obb_idx: int):
        if self.scene is None:
            raise RuntimeError("Scene has not been initialized, please use 'config_window' function before.")

        obb = self.obbs_[obb_idx]
        # 提取平面
        cropped_pcd, obb = PlateExtraction.extract_plate(self.pcd_, obb)
        # 把平面对齐到原点
        aligned_pcd = self.aligner_.align(cropped_pcd, obb)
        # 把点云压缩到 2D
        pcd2d = PcdDimension.compress_to_2d(aligned_pcd)
        # 移除边界点
        _, bounding_removed_obb = PlateExtraction.remove_boundary(aligned_pcd)
        # 创建操作操纵杆点
        cols_points, rows_points = self.create_operating_lever(pcd2d, bounding_removed_obb, PLATE_COLS, PLATE_ROWS, self.blank_distance_)
        # 把操纵杆点合并到球体中心数组
        self.sphere_centers_ = [cols_points, rows_points]

        # 可视化二维点
        self.visualize_pcd(pcd2d, "aligned")
        # 可视化移除边界点后的 OBB
        self.visualize_obb(bounding_removed_obb, "bounding_removed")
        # 可视化操作操纵杆
        cols_points = np.hstack((cols_points, np.zeros((len(cols_points), 1))))
        rows_points = np.hstack((rows_points, np.zeros((len(rows_points), 1))))
        self.visualize_operation_lever(self.sphere_centers_[self.COLS_HITTEN_STATE], self.sphere_centers_[self.ROWS_HITTEN_STATE], self.sphere_radius_)

        aabb = aligned_pcd.get_axis_aligned_bounding_box()
        self.setup_camera(60.0, aabb, aabb.get_center())

    # 清理场景内的运算信息
    def clean(self):
        self.scene.clear_geometry()
        self.aligner_.clear()

        self.sphere_centers_ = None

        self.rotation_state_ = self.ROTATION_UNLOCK

        self.highlighted_type_ = self.NONE_HITTED_STATE
        self.highlighted_idx_ = self.NONE_HITTED_STATE
        self.highlighted_color_ = self.NONE_COLOR

        self.selected_type_ = self.NONE_HITTED_STATE
        self.selected_idx_ = self.NONE_HITTED_STATE

        self.mouse_origin_position_ = None
        self.sphere_origin_position_ = None

    # 鼠标事件处理
    def _callback_mouse(self, event: gui.MouseEvent):
        logger.debug(f"[{time.process_time()}]--------------------")
        logger.debug(f"Mouse event type: {event.type}")
        logger.debug(f"Highlighted geometry == [{self.highlighted_type_}][{self.highlighted_idx_}]")
        logger.debug(f"Selected geometry == [{self.selected_type_}][{self.selected_idx_}]")

        # 当前鼠标点位
        mouse_current_position = np.array([event.x, event.y])

        # 鼠标拖动事件
        if event.type == gui.MouseEvent.Type.DRAG:
            # 如果拖动被上锁，操纵球被命中
            if self.rotation_state_ == self.ROTATION_LOCK:
                # 计算移动向量
                displacement_vector = self.move_sphere(self.mouse_origin_position_, mouse_current_position)
                # 计算拖动后球的位置（原始坐标 + 移动向量）
                sphere_position = [x + y for (x, y) in zip(self.sphere_origin_position_, displacement_vector)] + [0.0]
                self.sphere_centers_[self.highlighted_type_][self.highlighted_idx_] = sphere_position
                # 可视化刷新
                self.visualize_operation_lever(self.sphere_centers_[self.COLS_HITTEN_STATE], self.sphere_centers_[self.ROWS_HITTEN_STATE], self.sphere_radius_)
                self.update_sphere_highlight(self.highlighted_type_, self.highlighted_idx_, self.highlighted_color_, "highlighted_sphere")
                # 不再处理拖动事件
                return gui.Widget.EventCallbackResult.CONSUMED

        # 鼠标抬起视为命中
        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            self.selected_type_ = self.highlighted_type_
            self.selected_idx_ = self.highlighted_idx_
            self.highlighted_type_ = self.NONE_HITTED_STATE
            self.highlighted_idx_ = self.NONE_HITTED_STATE
        # 鼠标点击视为等待下一个命中
        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self.selected_type_ = self.NONE_HITTED_STATE
            self.selected_idx_ = self.NONE_HITTED_STATE

        # 高亮处理模块
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and self.highlighted_type_ != self.NONE_HITTED_STATE and self.highlighted_idx_ != self.NONE_HITTED_STATE:
             # 按下鼠标且命中触发绿色高亮，锁定拖动
            self.highlighted_color_ = self.MOUSE_CLICK_COLOR
            self.mouse_origin_position_ = mouse_current_position
            self.sphere_origin_position_ = np.array(self.sphere_centers_[self.highlighted_type_][self.highlighted_idx_])[:2]
            self.rotation_state_ = self.ROTATION_LOCK
        else:
            # 否则触发红色的悬停高亮
            self.highlighted_type_ = self.NONE_HITTED_STATE
            self.highlighted_idx_ = self.NONE_HITTED_STATE
            self.highlighted_color_ = self.MOUSE_OVER_COLOR
            self.mouse_origin_position_ = None
            self.sphere_origin_position_ = None
            self.rotation_state_ = self.ROTATION_UNLOCK
            self.highlighted_type_, self.highlighted_idx_ = self.detect_intersects(mouse_current_position)

        # 刷新行列高亮
        self.update_sphere_highlight(self.highlighted_type_, self.highlighted_idx_, self.highlighted_color_, "highlighted_sphere")
        self.update_sphere_highlight(self.selected_type_, self.selected_idx_, self.selected_color_, "selected_sphere")

        return gui.Widget.EventCallbackResult.HANDLED

    # 键盘事件处理
    def _callback_key(self, event: gui.KeyEvent):
        logger.debug(f"[{time.process_time()}]--------------------")
        logger.debug(f"Key event type: {event.type}, Key: {event.key}")
        logger.debug(f"Selected geometry == [{self.selected_type_}][{self.selected_idx_}]")

        # 有选择命中则处理鼠标微调
        if self.selected_type_ != self.NONE_HITTED_STATE and self.selected_idx_ != self.NONE_HITTED_STATE:
            displacement_vector = [0.0, 0.0]
            if event.type == gui.KeyEvent.Type.DOWN:
                # 这里的加减使用 ASICII 码
                if event.key == self.ADDER_KEY:
                    displacement_vector[self.selected_type_] = self.key_manu_scale_
                elif event.key == self.SUBER_KEY:
                    displacement_vector[self.selected_type_] = - self.key_manu_scale_

            logger.debug(f"Fine tuning vector == {displacement_vector}")

            # 更新偏移量
            if self.mouse_origin_position_ is not None:
                self.mouse_origin_position_ += displacement_vector

            # 更新球坐标
            sphere_center = np.array(self.sphere_centers_[self.selected_type_][self.selected_idx_]) + np.array(displacement_vector + [0.0])
            self.sphere_centers_[self.selected_type_][self.selected_idx_] = sphere_center

            # 刷新操纵系统变化
            self.visualize_operation_lever(self.sphere_centers_[self.COLS_HITTEN_STATE], self.sphere_centers_[self.ROWS_HITTEN_STATE], self.sphere_radius_)
            self.update_sphere_highlight(self.highlighted_type_, self.highlighted_idx_, self.highlighted_color_, "highlighted_sphere")
            self.update_sphere_highlight(self.selected_type_, self.selected_idx_, self.selected_color_, "selected_sphere")
            
        return gui.Widget.EventCallbackResult.HANDLED



    # 计算射线与球的交点和距离
    def ray_intersects_sphere(self, line_origin: list, line_direct: list, sphere_center: list, sphere_radius: float):
        # d: 直线的方向向量 (dx, dy, dz)
        dx, dy, dz = line_direct
        # P: 直线上的点 P (x1, y1, z1)
        x1, y1, z1 = line_origin
        # O: 球心 (x0, y0, z0)
        x0, y0, z0 = sphere_center

        # r: 球的半径
        # 计算一元二次方程的系数 A, B, C
        A = dx ** 2 + dy ** 2 + dz ** 2
        B = 2 * (dx * (x1 - x0) + dy * (y1 - y0) + dz * (z1 - z0))
        C = (x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2 - sphere_radius ** 2
        
        # 计算判别式
        delta = B**2 - 4 * A * C
        
        if delta < 0:
            hit = False
            tmin = None
            tmax = None
        else:
            hit = True
            tmin = - (B + np.sqrt(delta)) / (2 * A)
            tmax = - (B - np.sqrt(delta)) / (2 * A)
        return hit, tmin, tmax

    # 检测相交球
    def detect_intersects(self, mouse_position: list):
        # 求鼠标射线的起点和方向
        ray_origin, ray_direct = self.get_ray_of_mouse(mouse_position)

        # 查找最近的交点，设为命中
        hit_type = self.NONE_HITTED_STATE
        hit_idx = self.NONE_HITTED_STATE
        min_distance = float("inf")
        for i, centers in enumerate(self.sphere_centers_):
            for j, center in enumerate(centers):
                hit, tmin, _ = self.ray_intersects_sphere(ray_origin, ray_direct, center, self.sphere_radius_)
                if hit:
                    logger.debug(f"Ray hit SPHERE[{i}][{j}] at distance {tmin}.")
                    if tmin < min_distance:
                        hit_type = i
                        hit_idx = j
                        min_distance = tmin
        return hit_type, hit_idx

    # 更新球的高亮显示
    def update_sphere_highlight(self, type: int, idx: int, color: list, name: str):
        self.scene.remove_geometry(name)
        if type != self.NONE_HITTED_STATE and idx != self.NONE_HITTED_STATE and color != self.NONE_COLOR:
            self.visualize_sphere(self.sphere_centers_[type][idx], self.sphere_radius_, color, name)
            logger.debug(f"Added color {color} highlight to SPHERE[{type}][{idx}].")

    # 计算球的移动向量
    def move_sphere(self, mouse_original_position: list, mouse_current_position: list):
        # 计算鼠标移动前后的方向向量
        ray_origin, ray_original_direct = self.get_ray_of_mouse(mouse_original_position)
        _, ray_current_direct = self.get_ray_of_mouse(mouse_current_position)
        
        # 鼠标移动前后在 “2D点云” 上的交点
        original_intersection_point = self.line_intersection_plane(ray_origin, ray_original_direct)
        current_intersection_point = self.line_intersection_plane(ray_origin, ray_current_direct)

        # 计算球的移动向量
        ray_displacement_vector = np.array(current_intersection_point) - np.array(original_intersection_point)

        # 列上的点只允许横移，行上的点只允许竖移
        if self.highlighted_type_ == self.COLS_HITTEN_STATE:
            sphere_displacement_vector = [ray_displacement_vector[0], 0.0]
        elif self.highlighted_type_ == self.ROWS_HITTEN_STATE:
            sphere_displacement_vector = [0.0, ray_displacement_vector[1]]
        else:
            raise RuntimeError("Can not move a invisible thing.")

        return sphere_displacement_vector

    # 计算射线与平面 z=0 的交点
    def line_intersection_plane(self, line_origin: list, line_direct: list):
        x0, y0, z0 = line_origin
        xd, yd, zd = line_direct

        if zd == 0:
            raise ValueError("The line is parallel to the plane z=0, no intersection.")
        
        t = - z0 / zd
        x = x0 + t * xd
        y = y0 + t * yd
        
        return [x, y, 0.0]



    # 从点云中拿到边界并扩展
    def get_boundary(self, pcd: geometry.PointCloud, expand_distance: float=0.0):
        points = np.asarray(pcd.points)
        left = np.min(points[:, 0]) - expand_distance
        right = np.max(points[:, 0]) + expand_distance
        bottom = np.min(points[:, 1]) - expand_distance
        top = np.max(points[:, 1]) + expand_distance
        return left, right, bottom, top

    # 计算 OBB 角点
    def create_corner_lines(self, obb, cols, rows):
        extents = sorted(obb.extent, reverse=True)
        width, height, _ = extents
        x, y, _ = obb.center

        # 计算边界点位
        obb_left_bound = x - width / 2
        obb_right_bound = x + width / 2
        obb_bottom_bound = y - height / 2
        obb_top_bound = y + height / 2

        # 计算棋盘格边长
        col_width = width / cols
        row_height = height / rows

        # 计算行列坐标
        rows_coordinates = [obb_bottom_bound + row_height * i  for i in range(1, rows)]
        cols_coordinates = [obb_left_bound + col_width * i  for i in range(1, cols)]

        # 生成角点 3D 坐标
        left_points = np.asarray([[obb_left_bound, row_coordinate, 0] for row_coordinate in rows_coordinates])
        right_points = np.asarray([[obb_right_bound, row_coordinate, 0] for row_coordinate in rows_coordinates])
        bottom_points = np.asarray([[col_coordinate, obb_bottom_bound, 0] for col_coordinate in cols_coordinates])
        top_points = np.asarray([[col_coordinate, obb_top_bound, 0] for col_coordinate in cols_coordinates])

        return left_points, right_points, bottom_points, top_points

    # 生成操纵杆点，即行列操纵球的中心
    def create_operating_lever(self, pcd, obb, cols, rows, distance):
        # 计算边界点位
        left, right, bottom, top = self.get_boundary(pcd, distance)
        left_points, right_points, bottom_points, top_points = self.create_corner_lines(obb, cols, rows)

        # 扩展边界点位
        left_points[:, 0] = left
        right_points[:, 0] = right
        bottom_points[:, 1] = bottom
        top_points[:, 1] = top

        # 生成操纵杆点
        cols_points = [point for pair in zip(bottom_points, top_points) for point in pair]
        rows_points = [point for pair in zip(left_points, right_points) for point in pair]

        return cols_points, rows_points

    # 可视化操作操纵杆
    def visualize_operation_lever(self, cols_points, rows_points, radius):
        # 可视化经纬线
        logger.debug("Cols of points and lines:")
        self.visualize_points_lines(cols_points, "cols")
        logger.debug("Rows of points and lines:")
        self.visualize_points_lines(rows_points, "rows")
        
        # 可视化操纵球
        logger.debug("Cols of spheres:")
        self.visualize_spheres(cols_points, radius, [0.9, 0.9, 0.9, 1.0], "cols_spheres")
        logger.debug("Rows of spheres:")
        self.visualize_spheres(rows_points, radius, [0.9, 0.9, 0.9, 1.0], "rows_spheres")



    # 计算线段与线段的交点
    def line_intersection(self, line1_start: list, line1_end: list, line2_start: list, line2_end: list):
        # 线段方向向量
        direct1 = np.array(line1_end) - np.array(line1_start)
        direct2 = np.array(line2_end) - np.array(line2_start)
        
        # 法向量叉乘
        denom = direct1[0] * direct2[1] - direct1[1] * direct2[0]
        
        if denom != 0:
            direct_lines = np.array(line2_start) - np.array(line1_start)
            t = (direct_lines[0] * direct2[1] - direct_lines[1] * direct2[0]) / denom
            u = (direct_lines[0] * direct1[1] - direct_lines[1] * direct1[0]) / denom
        
            if 0 <= t <= 1 and 0 <= u <= 1:
                intersection = np.array(line1_start) + t * direct1
                return intersection
        return None

    # 返回角点
    def get_corner_points(self):
        # 拿到行列坐标
        cols_points = self.sphere_centers_[self.COLS_HITTEN_STATE]
        rows_points = self.sphere_centers_[self.ROWS_HITTEN_STATE]

        # 对于每一根经线，计算所有纬线的交点
        points = []
        for i in range(int(len(cols_points) / 2)):
            col_start = cols_points[2 * i]
            col_end = cols_points[2 * i + 1]
            for j in range(int(len(rows_points) / 2)):
                row_strat = rows_points[2 * j]
                row_end = rows_points[2 * j + 1]
                intersection_point = self.line_intersection(col_start, col_end, row_strat, row_end)
                points.append(intersection_point)

        # 构成点云
        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(np.array(points))
        # 逆转向得到原点
        restored_pcd = self.aligner_.inverse_align(pcd)

        logger.info(f"Restored {len(restored_pcd.points)} points.")

        return restored_pcd.points
