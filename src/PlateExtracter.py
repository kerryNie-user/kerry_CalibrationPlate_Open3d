import numpy as np
from sklearn.cluster import KMeans
import open3d.geometry as geometry
import open3d.utility as utility

from logger import info, debug
from DownSampler import DownSampler
from PcdDimensionOperator import PcdDimensionOperator

class PlateExtracter:
    def __init__(self):
        info("Start to extract plate from point clould.")

    def extract_plate(self, pcd: geometry.PointCloud, obb: geometry.OrientedBoundingBox):
        indices, _ = self.expand_obb(pcd, obb)
        selected_pcd = pcd.select_by_index(indices)

        reflectivities = np.asarray(selected_pcd.colors).mean(axis=1).reshape(-1, 1)
        clustered_pcds = self.cluster_pcd(selected_pcd, reflectivities)
        cropped_pcd, obb = self.extract_plate_by_clustered_pcd(selected_pcd, clustered_pcds)
        return cropped_pcd, obb

    def expand_obb(self, pcd: geometry.PointCloud, obb: geometry.OrientedBoundingBox, scale_limit_threshold: float = 0.05):
        # 获取选择的点云区域
        obb_extent = np.array(obb.extent)  # 转换为 NumPy 数组以便操作
        min_dim_idx = np.argmin(obb_extent)  # 获取最小维度的索引
        previous_indices = None

        expanded_obb = geometry.OrientedBoundingBox()
        expanded_obb.center = obb.center  # 保持原始的中心
        expanded_obb.R = obb.R  # 保持原始的旋转矩阵
        expanded_obb.extent = obb_extent.copy()  # 保持原始的尺寸

        largest_scale = 10

        for scale_factor in range(1, largest_scale):
            debug(f"Trying scale factor == {scale_factor}")
            new_extent = obb_extent.copy()
            new_extent[min_dim_idx] *= scale_factor  # 只扩大最小维度
            expanded_obb.extent = new_extent  # 更新 OBB 尺寸
            indices = expanded_obb.get_point_indices_within_bounding_box(pcd.points)
            if previous_indices != None:
                if len(indices) - len(previous_indices) < len(indices) / scale_factor * scale_limit_threshold:
                    debug("Not so good, break it.")
                    info(f"Obb has been expanded for {scale_factor - 1} scales.")
                    break
            debug("A good try, continue.")
            previous_indices = indices
            if scale_factor == largest_scale:
                info(f"Obb has been expanded for {scale_factor} scales.")
        return previous_indices, expanded_obb

    def cluster_pcd(self, pcd: geometry.PointCloud, pcd_property: np.array, normalize_color: bool = True):
        clustered_values = self.kmeans_clustering(pcd_property.reshape(-1, 1))
        unique_labels = np.unique(clustered_values)

        clustered_pcds = []
        for label in unique_labels:
            # 获取当前簇的点索引
            clustered_indices = np.where(clustered_values == label)[0]
            clustered_pcd = pcd.select_by_index(clustered_indices)
            if normalize_color:
                color = float(label / (len(unique_labels) - 1))
                colors = [[color, color, color] for _ in range(len(clustered_pcd.points))]
                clustered_pcd.colors = utility.Vector3dVector(colors)
            clustered_pcds.append(clustered_pcd)
        return clustered_pcds

    def kmeans_clustering(self, X, min_clusters_number: int = 2, min_cluster_size_ratio: float = 0.1):
        max_clusters_number = int(1.0 / min_cluster_size_ratio)
        if min_clusters_number > max_clusters_number:
            raise RuntimeError(f"Invalid for cluster number config, when {min_clusters_number} at leat but {max_clusters_number} at most.")

        previous_n_clusters = min_clusters_number
        previous_labels = KMeans(n_clusters=previous_n_clusters, random_state=0).fit_predict(X)
        info("Start to clusting.")
        
        for n_clusters in range(min_clusters_number + 1, max_clusters_number):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            labels = kmeans.fit_predict(X)
            
            # 统计每个簇的样本数
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            # 打印当前聚类情况
            debug(f"Trying {n_clusters} clusters ...")
            debug(f"       Clusters to members: {dict(zip(unique_labels, counts))}")
            
            # 检查是否有簇的样本数少于阈值
            if any(count < len(X) * min_cluster_size_ratio for count in counts):
                debug(f"The new cluster has a untenable cluster, return previous {previous_n_clusters} clusters.\n")
                return previous_labels  # 如果某个簇的点数过少，返回之前有效的聚类结果
            else:
                debug(f"{n_clusters} clusters is tenable, continue to try more kinds.\n")
                previous_labels = labels
                previous_n_clusters = n_clusters  # 更新有效的簇数
        
        print(f"The last clusters is tenable, return the last result with {max_clusters_number} clusters.")
        return previous_labels  # 如果所有聚类数都有效，返回最后的结果

    def extract_plate_by_clustered_pcd(self, pcd: geometry.PointCloud, clustered_pcds: list[geometry.PointCloud]):
        obbs = []
        for clustered_pcd in clustered_pcds:
            clustered_pcd.estimate_normals(search_param=geometry.KDTreeSearchParamKNN(knn=30))
            patches = clustered_pcd.detect_planar_patches(
                normal_variance_threshold_deg=20,
                coplanarity_deg=10,
                outlier_ratio=0.2,
                min_plane_edge_length=0,
                min_num_points=0)
            obbs.extend(patches)
        biggest_obb, _ = self.find_biggest_obb(obbs)

        indices, _ = self.expand_obb(pcd, biggest_obb)
        cropped_pcd = pcd.select_by_index(indices)
        return cropped_pcd, biggest_obb

    def find_biggest_obb(self, obbs: list[geometry.OrientedBoundingBox]):
        info(f"Start to find obb from {len(obbs)} obbs.")
        extents = [obb.extent for obb in obbs]
        areas = []
        for extent in extents:
            sorted_extent = sorted(extent)
            area = sorted_extent[1] * sorted_extent[2]  # 计算剩下两个维度的面积
            areas.append(area)
        sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
        biggest_obb = obbs[sorted_indices[0]]
        other_obbs = [obbs[i] for i in sorted_indices[1:]]
        info(f"Biggest obb has been founded with area of {areas[sorted_indices[0]]}.")
        info(f"With {len(other_obbs)} other obbs.")
        return biggest_obb, other_obbs



    def remove_boundary(self, pcd, pre_remove_percent=0.02):
        pcd2d = PcdDimensionOperator().compress_to_2d(pcd)
        bound_removed_pcd = self.remove_bounding(pcd2d, pre_remove_percent)
        downsampled_pcd = DownSampler().voxel_pcd(bound_removed_pcd)

        reflectivities = np.asarray(downsampled_pcd.colors).mean(axis=1).reshape(-1, 1)
        clustered_pcds = self.cluster_pcd(downsampled_pcd, reflectivities)
        clustered_pcds.sort(key=lambda pcd: len(pcd.points), reverse=True)

        max_points_num = len(clustered_pcds[0].points)
        min_points_num = len(clustered_pcds[1].points)

        if max_points_num - min_points_num > min_points_num / 20:
            info("Cut the bounding.")
            min_clustered_pcd = clustered_pcds[1]
            min_clustered_pcd = PcdDimensionOperator().convert_to_layers(min_clustered_pcd,  1e-8)

            obbs = []
            for clustered_pcd in [min_clustered_pcd]:
                clustered_pcd.estimate_normals(search_param=geometry.KDTreeSearchParamKNN(knn=10))
                patches = clustered_pcd.detect_planar_patches(
                    normal_variance_threshold_deg=1,
                    coplanarity_deg=10,
                    outlier_ratio=0.05,
                    min_plane_edge_length=0,
                    min_num_points=0)
                obbs.extend(patches)
            biggest_obb, _ = self.find_biggest_obb(obbs)

            indices = biggest_obb.get_point_indices_within_bounding_box(min_clustered_pcd.points)
            bounding_removed_min_pcd = min_clustered_pcd.select_by_index(indices)
            x_min, x_max, y_min, y_max = self.get_min_max_xy(bounding_removed_min_pcd)
            boundary_removed_obb = self.create_obb_from_bounds(x_min, x_max, y_min, y_max)

            indices = boundary_removed_obb.get_point_indices_within_bounding_box(pcd2d.points)
            boundary_removed_pcd = pcd2d.select_by_index(indices)
            return boundary_removed_pcd, boundary_removed_obb
        else:
            return pcd2d, pcd.get_oriented_bounding_box()

    def remove_bounding(self, pcd, thickness_percent=0.01):
        # 获取点云的坐标
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        # 计算点云的边界（最大最小坐标）
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        
        # 计算厚度为1%的距离
        x_thickness = (max_coords[0] - min_coords[0]) * thickness_percent
        y_thickness = (max_coords[1] - min_coords[1]) * thickness_percent
        
        # 过滤掉距离边界1%距离范围内的点
        mask = ((points[:, 0] > min_coords[0] + x_thickness) & (points[:, 0] < max_coords[0] - x_thickness) &
                (points[:, 1] > min_coords[1] + y_thickness) & (points[:, 1] < max_coords[1] - y_thickness))
        
        # 根据mask选择点
        filtered_points = points[mask]
        filtered_colors = colors[mask]
        filtered_pcd = geometry.PointCloud()
        filtered_pcd.points = utility.Vector3dVector(filtered_points)
        filtered_pcd.colors = utility.Vector3dVector(filtered_colors)
        
        return filtered_pcd

    def get_min_max_xy(self, pcd):
        # 获取点云的所有点
        points = np.asarray(pcd.points)

        # 获取 x 和 y 坐标
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # 获取 x 和 y 的最小值和最大值
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        return x_min, x_max, y_min, y_max

    def create_obb_from_bounds(self, x_min=-1e-8, x_max=1e-8, y_min=-1e-8, y_max=1e-8, z_min=-1e-8, z_max=1e-8):
        # OBB 的中心在 (x_center, y_center, z=0)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        width  = x_max - x_min
        height = y_max - y_min
        depth  = z_max - z_min

        # 创建一个 OBB
        obb = geometry.OrientedBoundingBox(center=[x_center, y_center, z_center], R=np.eye(3), extent=[width, height, depth])
        return obb