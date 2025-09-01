import time
import numpy as np

import open3d.utility as utility
import open3d.geometry as geometry
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from logger import info, debug, suggestion
from PointCloudAligner import PointCloudAligner
from PlateExtracter import PlateExtracter
from PcdDimensionOperator import PcdDimensionOperator
from OperativeWidget import OperativeWidget

PLATH_COLS = 10
PLATH_ROWS = 7

# <<< 场景部件：用于处理被选中的标定板展示界面 >>>
class ObbSelectedWidget(OperativeWidget):
    COLS_HITTEN_STATE = 0
    ROWS_HITTEN_STATE = 1

    ROTATION_LOCK = 0
    ROTATION_UNLOCK = 1

    NONE_COLOR = [-1, -1, -1, -1]
    MOUSE_OVER_COLOR = [1.0, 0.0, 0.0, 1.0]
    MOUSE_CLICK_COLOR = [0.0, 1.0, 0.0, 1.0]
    MOUSE_SELECT_COLOR = [0.0, 0.0, 1.0, 1.0]

    ADDER_KEY = 61
    SUBER_KEY = 45

    def __init__(self, pcd, patches):
        super().__init__()
        self.pcd_ = pcd
        self.patches_ = patches
        self.aligner_ = PointCloudAligner()

        self.sphere_centers_ = None

        self.rotation_state_ = self.ROTATION_UNLOCK

        self.highlighted_type_ = self.NONE_HITTED_STATE
        self.highlighted_idx_ = self.NONE_HITTED_STATE
        self.highlighted_color_ = self.NONE_COLOR

        self.selected_type_ = self.NONE_HITTED_STATE
        self.selected_idx_ = self.NONE_HITTED_STATE
        self.selected_color_ = self.MOUSE_SELECT_COLOR

        self.mouse_origin_position_ = None
        self.sphere_origin_position_ = None

        self.sphere_radius_ = 0.01
        self.blank_distance_ = 0.05
        self.key_manu_scale_ = self.sphere_radius_ / 10

        self.set_on_mouse(self._callback_mouse)
        self.set_on_key(self._callback_key)

        suggestion("Drag the sphere to coarsely tune it.")
        suggestion("Use the keyboard to fine-tune:")
        suggestion("    '+' : going right or up.")
        suggestion("    '-' : going down or left.")

    def config_window(self, renderer):
        if self.scene is None:
            if renderer is None:
                raise RuntimeError("Renderer is None, cannot initialize the scene.")
            info("Selected window has been created.")
            self.scene = rendering.Open3DScene(renderer)
            self.scene.set_background([1, 1, 1, 1])
            self.scene.show_skybox(False)
            self.scene.show_axes(False)
        else:
            info("Selected window has been cleaned.")
            self.clear()

    def open_window(self, obb):
        if self.scene is None:
            raise RuntimeError("Scene has not been initialized, please use 'config_window' function before.")

        cropped_pcd, obb = PlateExtracter().extract_plate(self.pcd_, obb)
        aligned_pcd = self.aligner_.align(cropped_pcd, obb)
        pcd2d = PcdDimensionOperator().compress_to_2d(aligned_pcd)
        _, bounding_removed_obb = PlateExtracter().remove_boundary(aligned_pcd)
        rows_points, cols_points = self.create_operating_lever(pcd2d, bounding_removed_obb, PLATH_COLS, PLATH_ROWS, self.blank_distance_)
        self.sphere_centers_ = [cols_points, rows_points]

        self.visualize_pcd(pcd2d, "aligned")
        self.visualize_obb(bounding_removed_obb, "bounding_removed")
        cols_points = np.hstack((cols_points, np.zeros((len(cols_points), 1))))
        rows_points = np.hstack((rows_points, np.zeros((len(rows_points), 1))))
        self.visualize_operation_lever(self.sphere_centers_[self.COLS_HITTEN_STATE], self.sphere_centers_[self.ROWS_HITTEN_STATE], self.sphere_radius_)

        aabb = aligned_pcd.get_axis_aligned_bounding_box()
        self.setup_camera(60.0, aabb, aabb.get_center())

    def clear(self):
        self.scene.clear_geometry()

        self.pcd_ = None
        self.patches_ = None
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



    def _callback_mouse(self, event):
        debug(f"----------[Window Plate]----------")
        debug(f"-------[Time : {time.process_time()}]-------")
        debug(f"Mouse event type: {event.type}")

        debug(f"Highlighted geometry == [{self.highlighted_type_}][{self.highlighted_idx_}]")
        debug(f"Selected geometry == [{self.selected_type_}][{self.selected_idx_}]")

        mouse_current_position = np.array([event.x, event.y])

        if event.type == gui.MouseEvent.Type.DRAG:
            if self.rotation_state_ == self.ROTATION_LOCK:
                displacement_vector = self.move_sphere(self.mouse_origin_position_, mouse_current_position)
                sphere_position = [x + y for (x, y) in zip(self.sphere_origin_position_, displacement_vector)] + [0.0]
                self.sphere_centers_[self.highlighted_type_][self.highlighted_idx_] = sphere_position
                self.visualize_operation_lever(self.sphere_centers_[self.COLS_HITTEN_STATE], self.sphere_centers_[self.ROWS_HITTEN_STATE], self.sphere_radius_)
                self.update_sphere_highlight(self.highlighted_type_, self.highlighted_idx_, self.highlighted_color_, "highlighted_sphere")
                return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.Type.BUTTON_UP:
            self.selected_type_ = self.highlighted_type_
            self.selected_idx_ = self.highlighted_idx_
        elif event.type == gui.MouseEvent.Type.BUTTON_DOWN:
            self.selected_type_ = self.NONE_HITTED_STATE
            self.selected_idx_ = self.NONE_HITTED_STATE

        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and self.highlighted_type_ != self.NONE_HITTED_STATE and self.highlighted_idx_ != self.NONE_HITTED_STATE:
            self.highlighted_color_ = self.MOUSE_CLICK_COLOR
            self.mouse_origin_position_ = mouse_current_position
            self.sphere_origin_position_ = np.array(self.sphere_centers_[self.highlighted_type_][self.highlighted_idx_])[:2]
            self.rotation_state_ = self.ROTATION_LOCK
        else:
            self.highlighted_type_ = self.NONE_HITTED_STATE
            self.highlighted_idx_ = self.NONE_HITTED_STATE
            self.highlighted_color_ = self.MOUSE_OVER_COLOR
            self.mouse_origin_position_ = None
            self.sphere_origin_position_ = None
            self.rotation_state_ = self.ROTATION_UNLOCK
            self.highlighted_type_, self.highlighted_idx_ = self.detect_intersects(mouse_current_position)

        self.update_sphere_highlight(self.highlighted_type_, self.highlighted_idx_, self.highlighted_color_, "highlighted_sphere")
        self.update_sphere_highlight(self.selected_type_, self.selected_idx_, self.selected_color_, "selected_sphere")

        return gui.Widget.EventCallbackResult.HANDLED

    def _callback_key(self, event):
        debug(f"----------[Key Event]----------")
        debug(f"-------[Time : {time.process_time()}]-------")
        debug(f"Key event type: {event.type}, Key: {event.key}")

        debug(f"Selected geometry == [{self.selected_type_}][{self.selected_idx_}]")

        if self.selected_type_ != self.NONE_HITTED_STATE and self.selected_idx_ != self.NONE_HITTED_STATE:
            displacement_vector = [0.0, 0.0]
            if event.type == gui.KeyEvent.Type.DOWN:
                if event.key == self.ADDER_KEY:
                    displacement_vector[self.selected_type_] = self.key_manu_scale_
                elif event.key == self.SUBER_KEY:
                    displacement_vector[self.selected_type_] = - self.key_manu_scale_

            debug(f"Fine tuning vector == {displacement_vector}")

            if self.mouse_origin_position_ is not None:
                self.mouse_origin_position_ += displacement_vector

            sphere_center = np.array(self.sphere_centers_[self.selected_type_][self.selected_idx_]) + np.array(displacement_vector + [0.0])
            self.sphere_centers_[self.selected_type_][self.selected_idx_] = sphere_center

            self.visualize_operation_lever(self.sphere_centers_[self.COLS_HITTEN_STATE], self.sphere_centers_[self.ROWS_HITTEN_STATE], self.sphere_radius_)
            self.update_sphere_highlight(self.highlighted_type_, self.highlighted_idx_, self.highlighted_color_, "highlighted_sphere")
            self.update_sphere_highlight(self.selected_type_, self.selected_idx_, self.selected_color_, "selected_sphere")
            
        return gui.Widget.EventCallbackResult.HANDLED

    def get_corner_points(self):
        cols_points = self.sphere_centers_[self.COLS_HITTEN_STATE]
        rows_points = self.sphere_centers_[self.ROWS_HITTEN_STATE]

        points = []
        for i in range(int(len(cols_points) / 2)):
            col_start = cols_points[2 * i]
            col_end = cols_points[2 * i + 1]
            for j in range(int(len(rows_points) / 2)):
                row_strat = rows_points[2 * j]
                row_end = rows_points[2 * j + 1]
                intersection_point = self.line_intersection(col_start, col_end, row_strat, row_end)
                points.append(intersection_point)

        pcd = geometry.PointCloud()
        pcd.points = utility.Vector3dVector(np.array(points))
        restored_pcd = self.aligner_.inverse_align(pcd)

        info(f"Restored {len(restored_pcd.points)} points.")

        return restored_pcd.points



    def line_intersection(self, line1_start, line1_end, line2_start, line2_end):
        line1_start = np.array(line1_start)
        line1_end = np.array(line1_end)
        line2_start = np.array(line2_start)
        line2_end = np.array(line2_end)
        
        direct1 = line1_end - line1_start
        direct2 = line2_end - line2_start
        
        denom = direct1[0] * direct2[1] - direct1[1] * direct2[0]
        
        if denom == 0:
            return None
        
        direct_lines = line2_start - line1_start
        t = (direct_lines[0] * direct2[1] - direct_lines[1] * direct2[0]) / denom
        u = (direct_lines[0] * direct1[1] - direct_lines[1] * direct1[0]) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            intersection = line1_start + t * direct1
            return intersection
        else:
            return None

    def ray_intersects_sphere(self, line_origin, line_direct, sphere_center, sphere_radius):
        # P: 直线上的点 P (x1, y1, z1)
        # d: 直线的方向向量 (dx, dy, dz)
        # O: 球心 (x0, y0, z0)
        # r: 球的半径
        # 计算一元二次方程的系数 A, B, C
        dx, dy, dz = line_direct
        x1, y1, z1 = line_origin
        x0, y0, z0 = sphere_center
        
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

    def detect_intersects(self, mouse_position):
        ray_origin, ray_direct = self.get_ray_of_mouse(mouse_position)

        hit_type = self.NONE_HITTED_STATE
        hit_idx = self.NONE_HITTED_STATE
        min_distance = float("inf")
        for i, centers in enumerate(self.sphere_centers_):
            for j, center in enumerate(centers):
                hit, tmin, _ = self.ray_intersects_sphere(ray_origin, ray_direct, center, self.sphere_radius_)
                if hit:
                    debug(f"Ray hit SPHERE[{i}][{j}] at distance {tmin}.")
                    if tmin < min_distance:
                        hit_type = i
                        hit_idx = j
                        min_distance = tmin
        return hit_type, hit_idx

    def update_sphere_highlight(self, type, idx, color, name):
        self.scene.remove_geometry(name)
        if type != self.NONE_HITTED_STATE and idx != self.NONE_HITTED_STATE and color != self.NONE_COLOR:
            self.visualize_sphere(self.sphere_centers_[type][idx], self.sphere_radius_, color, name)
            debug(f"Added color {color} highlight to SPHERE[{type}][{idx}].")

    def move_sphere(self, mouse_original_position, mouse_current_position):
        ray_origin, ray_original_direct = self.get_ray_of_mouse(mouse_original_position)
        _, ray_current_direct = self.get_ray_of_mouse(mouse_current_position)
        
        original_intersection_point = self.line_intersection_plane(ray_origin, ray_original_direct)
        current_intersection_point = self.line_intersection_plane(ray_origin, ray_current_direct)

        ray_displacement_vector = np.array(current_intersection_point) - np.array(original_intersection_point)

        sphere_displacement_vector = [0.0, 0.0]

        if self.highlighted_type_ == self.COLS_HITTEN_STATE:
            sphere_displacement_vector[0] = ray_displacement_vector[0]
        elif self.highlighted_type_ == self.ROWS_HITTEN_STATE:
            sphere_displacement_vector[1] = ray_displacement_vector[1]
        else:
            raise RuntimeError("Can not move a invisible thing.")

        return sphere_displacement_vector

    def line_intersection_plane(self, line_origin, line_direct):
        xo, yo, zo = line_origin
        xd, yd, zd = line_direct

        if zd == 0:
            raise ValueError("The line is parallel to the plane z=0, no intersection.")
        
        t = - zo / zd
        
        x = xo + t * xd
        y = yo + t * yd
        z = 0

        intersection_point = [x, y, z]
        
        return intersection_point



    def get_boundary(self, pcd: geometry.PointCloud):
        points = np.asarray(pcd.points)
        left = np.min(points[:, 0])
        right = np.max(points[:, 0])
        bottom = np.min(points[:, 1])
        top = np.max(points[:, 1])
        return left, right, bottom, top

    def expand_boundary(self, left, right, bottom, top, distance):
        left -= distance
        right += distance
        bottom -= distance
        top += distance
        return left, right, bottom, top

    def create_corner_lines(self, obb, cols, rows):
        extents = sorted(obb.extent, reverse=True)
        width, height, _ = extents
        x, y, _ = obb.center

        obb_left_bound = x - width / 2
        obb_right_bound = x + width / 2
        obb_bottom_bound = y - height / 2
        obb_top_bound = y + height / 2

        col_width = width / cols
        row_height = height / rows

        rows_coordinates = [obb_bottom_bound + row_height * i  for i in range(1, rows)]
        cols_coordinates = [obb_left_bound + col_width * i  for i in range(1, cols)]

        left_points = np.asarray([[obb_left_bound, row_coordinate, 0] for row_coordinate in rows_coordinates])
        right_points = np.asarray([[obb_right_bound, row_coordinate, 0] for row_coordinate in rows_coordinates])
        bottom_points = np.asarray([[col_coordinate, obb_bottom_bound, 0] for col_coordinate in cols_coordinates])
        top_points = np.asarray([[col_coordinate, obb_top_bound, 0] for col_coordinate in cols_coordinates])

        return left_points, right_points, bottom_points, top_points

    def create_operating_lever(self, pcd, obb, cols, rows, distance):
        left, right, bottom, top = self.get_boundary(pcd)
        left, right, bottom, top = self.expand_boundary(left, right, bottom, top, distance)
        left_points, right_points, bottom_points, top_points = self.create_corner_lines(obb, cols, rows)

        left_points[:, 0] = left
        right_points[:, 0] = right
        bottom_points[:, 1] = bottom
        top_points[:, 1] = top

        rows_points = [point for pair in zip(left_points, right_points) for point in pair]
        cols_points = [point for pair in zip(bottom_points, top_points) for point in pair]

        return rows_points, cols_points

    def visualize_operation_lever(self, cols_points, rows_points, radius):
        debug("Cols of points and lines:")
        self.visualize_points_lines(cols_points, "cols")
        debug("Rows of points and lines:")
        self.visualize_points_lines(rows_points, "rows")
        
        debug("Cols of spheres:")
        self.visualize_spheres(cols_points, radius, [0.9, 0.9, 0.9, 1.0], "cols_spheres")
        debug("Rows of spheres:")
        self.visualize_spheres(rows_points, radius, [0.9, 0.9, 0.9, 1.0], "rows_spheres")

# >>> --------------------------------- <<<