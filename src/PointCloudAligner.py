import numpy as np
import open3d.geometry as geometry
import open3d.utility as utility

from logger import info

class PointCloudAligner:
    def __init__(self):
        self.rotate_list_ = []
        self.translate_list_ = []

    def align(self, pcd: geometry.PointCloud, obb: geometry.OrientedBoundingBox = None):
        if len(pcd.points) < 4:
            raise RuntimeError("Too less points, 3 at least.")

        aligning_pcd = geometry.PointCloud(pcd)
        aligning_obb = geometry.OrientedBoundingBox(obb) if obb != None else pcd.get_oriented_bounding_box()

        face_direct = self.area_face_direct(aligning_obb, 0)  # 获取最大面法向量
        origin_direct = np.array([face_direct[0], face_direct[1], face_direct[2]])
        target_direct = np.array([0, 0, 1])
        aligning_pcd, aligning_obb = self.rotate_to_direct(aligning_pcd, aligning_obb, origin_direct, target_direct)
        self.rotate_list_.append((origin_direct, target_direct))

        up_direct = self.area_face_direct(aligning_obb, 1)
        origin_direct = np.array([up_direct[0], up_direct[1], up_direct[2]])
        target_direct = np.array([0, 1, 0])
        aligning_pcd, aligning_obb = self.rotate_to_direct(aligning_pcd, aligning_obb, origin_direct, target_direct)
        self.rotate_list_.append((origin_direct, target_direct))

        origin_point = aligning_pcd.get_oriented_bounding_box().get_center()
        target_point = np.array([0, 0, 0])
        aligning_pcd = self.translate_to_point(aligning_pcd, origin_point, target_point)
        self.translate_list_.append((origin_point, target_point))

        info(f"Done {len(self.rotate_list_)} rotate and {len(self.translate_list_)} translate as align")
        return aligning_pcd

    def inverse_align(self, pcd: geometry.PointCloud):
        if len(self.rotate_list_) == 0 and len(self.translate_list_) == 0:
            raise RuntimeError("You have to do some align first.")

        # 获取所有平移操作的历史
        for translation in reversed(self.translate_list_):
            target_point, origin_point = translation
            pcd = self.translate_to_point(pcd, origin_point, target_point)

        # 获取所有旋转操作的历史
        for rotation in reversed(self.rotate_list_):
            target_direct, origin_direct = rotation
            pcd, _ = self.rotate_to_direct(pcd, None, origin_direct, target_direct)

        info(f"Done {len(self.translate_list_)} inverse translate and {len(self.rotate_list_)} inverse rotate as inverse align.")
        return pcd

    def area_face_direct(self, obb, area_idx: int):
        if area_idx not in (0, 1, 2):
            raise RuntimeError(f"Area idx need to be 0, 1 or 2, but {area_idx} is given.")

        extents = obb.extent
        areas = [extents[1]*extents[2], extents[0]*extents[2], extents[0]*extents[1]]
        sorted_indices = sorted(range(3), key=lambda i: areas[i], reverse=True)
        target_idx = sorted_indices[area_idx]
        normal = np.zeros(3)
        normal[target_idx] = 1.0
        return obb.R @ normal

    def rotate_to_direct(self, pcd: geometry.PointCloud, obb: geometry.OrientedBoundingBox, origin_direct: np.array, target_direct: np.array) -> geometry.PointCloud:
        # 旋转矩阵
        def rotation_matrix_from_vectors(vec1, vec2):
            a = vec1 / np.linalg.norm(vec1)
            b = vec2 / np.linalg.norm(vec2)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            if s == 0:
                return np.eye(3)
            kmat = np.array([[0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]])
            R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))
            return R

        origin_direct = origin_direct / np.linalg.norm(origin_direct)  # 法向量单位化
        target_direct = target_direct / np.linalg.norm(target_direct)

        R = rotation_matrix_from_vectors(origin_direct, target_direct)

        # 旋转点云
        points_rot = np.asarray(pcd.points) @ R.T
        # 更新 OBB
        if obb != None:
            obb = obb.rotate(R)

        # 构建新点云
        rotated_pcd = geometry.PointCloud()
        rotated_pcd.points = utility.Vector3dVector(points_rot)
        rotated_pcd.colors = pcd.colors
        return rotated_pcd, obb

    def translate_to_point(self, pcd: geometry.PointCloud, origin_point: np.array, target_point: np.array) -> geometry.PointCloud:
        points = np.asarray(pcd.points)
        translation_vector = target_point - origin_point  # 计算平移向量
        points_translated = points + translation_vector  # 平移点云

        # 构建新点云
        translated_pcd = geometry.PointCloud()
        translated_pcd.points = utility.Vector3dVector(points_translated)
        translated_pcd.colors = pcd.colors  # 保持颜色不变
        return translated_pcd

    def test_align_inverse_align(self, pcd):
        # 获取原始点云的所有点
        original_points = np.asarray(pcd.points)
        
        # 对整个点云进行 align 操作
        aligned_pcd = self.align(pcd)

        # 对 align 后的点云进行 inverse_align 操作
        restored_pcd = self.inverse_align(aligned_pcd)

        # 获取恢复后的点云的所有点
        restored_points = np.asarray(restored_pcd.points)

         # 标记是否所有点都一致
        all_points_equal = True

        # 比较原始点云和恢复后的点云
        for i, (original_point, restored_point) in enumerate(zip(original_points, restored_points)):
            if not np.allclose(original_point, restored_point):
                all_points_equal = False
                raise RuntimeError(f"Point[{i}] is different, origin from {original_point} but restored at {restored_point}")

        # 如果所有点都一致
        if all_points_equal:
            info("Align and its inverse function can work well.")
        
        self.clear()
    
    def clear(self):
        self.rotate_list_.clear()
        self.translate_list_.clear()