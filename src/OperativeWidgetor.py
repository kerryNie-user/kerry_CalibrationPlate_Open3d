import numpy as np

import open3d.utility as utility
import open3d.geometry as geometry
import open3d.visualization as visualization
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from logger import debug

# <<< 场景父类：包含了鼠标坐标转化和简单模型一键显示 >>>
class OperativeWidgetor(gui.SceneWidget):
    NONE_HITTED_STATE = -1

    NONE_COLOR = [-1, -1, -1, -1]
    GREY = [0.9, 0.9, 0.9, 1.0]
    RED = [1.0, 0.0, 0.0, 1.0]
    GREEN = [0.0, 1.0, 0.0, 1.0]
    BLUE = [0.0, 0.0, 1.0, 1.0]

    def __init__(self):
        super().__init__()
        self.scene = None



    def get_ray_of_mouse(self, mouse_position):
        if len(mouse_position) != 2:
            raise RuntimeError(f"Mouse has only 2 position information x, y. but you have given {len(mouse_position)}")

        x = mouse_position[0]
        y = mouse_position[1]

        cam = self.scene.camera
        view = np.asarray(cam.get_view_matrix())
        proj = np.asarray(cam.get_projection_matrix())
        view_inv = np.linalg.inv(view)
        cam_pos = view_inv[:3, 3]

        ndc_x = 2.0 * x / self.frame.width - 1.0
        ndc_y = 1.0 - 2.0 * y / self.frame.height
        ndc_near = np.array([ndc_x, ndc_y, -1.0, 1.0])
        p_world = np.linalg.inv(proj @ view) @ ndc_near
        p_world /= p_world[3]

        ray_origin = cam_pos
        ray_direct = p_world[:3] - cam_pos
        debug(f"Ray origin: {ray_origin}")
        debug(f"Ray direction: {ray_direct}")
        return ray_origin, ray_direct / np.linalg.norm(ray_direct)



    def visualize_pcd(self, pcd: geometry.PointCloud, name: str):
        if self.scene is None:
            raise RuntimeError("self.scene is None when rendering.")

        self.scene.remove_geometry(name)

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 3.0

        self.scene.add_geometry(name, pcd, material)

    def visualize_lines(self, points: list, lines: list, name: str):
        if self.scene is None:
            raise RuntimeError("self.scene is None when rendering.")

        if len(points) == 0 or len(lines) == 0 or 2 * len(lines) != len(points):
            raise RuntimeError("Length of points and line set is invalid.")

        self.scene.remove_geometry(name)

        points = np.asarray(points)
        lines = np.asarray(lines)

        if np.array(points).shape[1] != 3:
            raise RuntimeError(f"Points need to have 3 dimensions, but {np.array(points).shape[1]} dimensions have been given.")

        if np.array(lines).shape[1] != 2:
            raise RuntimeError(f"Lines need to have 3 dimensions, but {np.array(lines).shape[1]} dimensions have been given.")

        debug("Create Line with:")
        debug(f"{len(points)} points == \n{points}")
        debug(f"{len(lines)} lines == \n{lines}")
        
        line = geometry.LineSet()
        line.points = utility.Vector3dVector(points)
        line.lines = utility.Vector2iVector(lines)
        vertex_colors = np.tile([1.0, 0.0, 0.0], (lines.shape[0], 1))  # 每个点的颜色
        line.colors = utility.Vector3dVector(vertex_colors.astype(np.float64))

        debug(f"LineSet created with {len(line.points)} points and {len(line.lines)} lines.")

        if line.is_empty():
            raise RuntimeError("Line geometry is empty. Please check your points and lines.")

        material = visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"  # 使用无光照着色器
        material.line_width = 5.0  # 设置线宽
        material.base_color = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float64)  # 设置颜色为红色

        self.scene.add_geometry(name, line, material)

    def visualize_rectangle(self, obb: geometry.OrientedBoundingBox, color: list, name: str):
        self.scene.remove_geometry(name)

        obb_mesh = geometry.TriangleMesh.create_box(width=obb.extent[0], height=obb.extent[1], depth=obb.extent[2])
        obb_mesh.translate(-obb_mesh.get_center())
        obb_mesh.rotate(obb.R, center=np.zeros(3))
        obb_mesh.translate(obb.center)

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = color
        material.point_size = 3.0
        self.scene.add_geometry(name, obb_mesh, material)

    def visualize_sphere(self, center: list, radius: float, color: list, name: str):
        if self.scene is None:
            raise RuntimeError("self.scene is None when rendering.")

        self.scene.remove_geometry(name)

        sphere = geometry.TriangleMesh.create_sphere(radius)
        sphere.paint_uniform_color(np.array(color)[:3])
        sphere.translate(center)

        material = rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.base_color = color

        self.scene.add_geometry(name, sphere, material)

    def visualize_spheres(self, centers: list, radius: float, color: list, name: str):
        if self.scene is None:
            raise RuntimeError("self.scene is None when rendering.")

        for i, center in enumerate(centers):
            self.visualize_sphere(center, radius, color, name + str(i))

    def visualize_points_lines(self, points: list, name: str):
        if self.scene is None:
            raise RuntimeError("self.scene is None when rendering.")
        elif len(points) % 2 != 0:
            raise RuntimeError("Asymmetric number of points.")

        points_num = int(len(points) / 2)
        lines = [[2 * i, 2 * i + 1] for i in range(points_num)]
        self.visualize_lines(points, lines, name)

    def visualize_obb(self, obb: geometry.OrientedBoundingBox, name: str):
        if self.scene is None:
            raise RuntimeError("self.scene is None when rendering.")

        self.scene.remove_geometry(name)

        line_set = geometry.LineSet.create_from_oriented_bounding_box(obb)

        material = rendering.MaterialRecord()
        material.shader = "unlitLine"
        material.line_width = 2.0
        material.base_color = (1.0, 0.0, 0.0, 1.0)

        self.scene.add_geometry(name, line_set, material)
# >>> --------------------------------- <<<