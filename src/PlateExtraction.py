# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cluster import KMeans
import open3d.geometry as geometry
import open3d.utility as utility

import logger
import PcdDimension
import PcdDownsample

# 提取平面
def extract_plate(pcd: geometry.PointCloud, obb: geometry.OrientedBoundingBox):
    # 先扩展 OBB，确保包含所有点
    indices, _ = _expand_obb(pcd, obb)
    # 把扩展后的点取出
    selected_pcd = pcd.select_by_index(indices)

    colors = np.asarray(selected_pcd.colors).mean(axis=1).reshape(-1, 1)
    # 对颜色进行聚类
    clustered_pcds = _cluster_pcd(selected_pcd, colors)
    cropped_pcd, obb = _extract_plate_by_clustered_pcd(selected_pcd, clustered_pcds)
    return cropped_pcd, obb


# 扩展 OBB
def _expand_obb(pcd: geometry.PointCloud, obb: geometry.OrientedBoundingBox, scale_limit_threshold: float = 0.05):
    # 找到 OBB 的厚度面
    obb_extent = np.array(obb.extent)
    min_dim_idx = np.argmin(obb_extent)
    previous_indices = None

    expanded_obb = geometry.OrientedBoundingBox()
    expanded_obb.center = obb.center
    expanded_obb.R = obb.R
    expanded_obb.extent = obb_extent.copy()

    # 最大的扩展系数
    largest_scale = 10

    # 自动扩展 OBB，使得尽可能包含更多点却不会引入过量杂点
    for scale_factor in range(1, largest_scale):
        logger.debug(f"Trying scale factor == {scale_factor}")
        # 扩展 OBB
        new_extent = obb_extent.copy()
        new_extent[min_dim_idx] *= scale_factor
        expanded_obb.extent = new_extent
        # 找到扩展后的点
        indices = expanded_obb.get_point_indices_within_bounding_box(pcd.points)
        if previous_indices is not None:
            # 如果扩展后的点数量并没有显著提升，说明扩展过头了
            if len(indices) - len(previous_indices) < len(indices) / scale_factor * scale_limit_threshold:
                logger.debug("Not so good, break it.")
                logger.info(f"Obb has been expanded for {scale_factor - 1} scales.")
                break
        logger.debug("A good try, continue.")
        # 记录验证后的扩展索引
        previous_indices = indices
        if scale_factor == largest_scale:
            logger.info(f"Obb has been expanded for {scale_factor} scales.")
    return previous_indices, expanded_obb


# 对点云进行聚类
def _cluster_pcd(pcd: geometry.PointCloud, pcd_property: np.array, normalize_color: bool = True):
    # 获取聚类结果
    clustered_values = _kmeans_clustering(pcd_property.reshape(-1, 1))
    # 所有的聚类标签
    unique_labels = np.unique(clustered_values)

    clustered_pcds = []
    # 对每个聚类标签进行处理
    for label in unique_labels:
        # 所有属于当前聚类标签的点的索引
        clustered_indices = np.where(clustered_values == label)[0]
        clustered_pcd = pcd.select_by_index(clustered_indices)
        if normalize_color:
            # 归一化颜色
            color = float(label / (len(unique_labels) - 1))
            colors = [[color, color, color] for _ in range(len(clustered_pcd.points))]
            clustered_pcd.colors = utility.Vector3dVector(colors)
        clustered_pcds.append(clustered_pcd)
    return clustered_pcds


# 聚类
def _kmeans_clustering(X, min_clusters_number: int = 2, min_cluster_size_ratio: float = 0.1):
    # 最多类数
    max_clusters_number = int(1.0 / min_cluster_size_ratio)
    if min_clusters_number > max_clusters_number:
        raise RuntimeError(f"Invalid for cluster number config, when {min_clusters_number} at least but {max_clusters_number} at most.")
    
    if min_clusters_number < 2:
        raise RuntimeError(f"Invalid for cluster number config, when 2 at least.")

    previous_n_clusters = min_clusters_number
    previous_labels = KMeans(n_clusters=previous_n_clusters, random_state=0).fit_predict(X)
    logger.info("Start to clustering.")
    
    # 不断尝试增加聚类数量
    for n_clusters in range(min_clusters_number + 1, max_clusters_number):
        # 进行聚类，并总结所有类
        labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
        unique_labels, counts = np.unique(labels, return_counts=True)

        logger.debug(f"Trying {n_clusters} clusters ...")
        logger.debug(f"       Clusters to members: {dict(zip(unique_labels, counts))}")
        
        # 如果新增类过于小众，说明聚类效果不佳
        if any(count < len(X) * min_cluster_size_ratio for count in counts):
            break
        else:
            # 否则储存当前聚类结果
            logger.debug(f"{n_clusters} clusters is tenable, continue to try more kinds.\n")
            previous_labels = labels
            previous_n_clusters = n_clusters
    
    logger.debug(f"The new cluster has a untenable cluster, return previous {previous_n_clusters} clusters.\n")
    return previous_labels


# 通过聚类结果提取平面
def _extract_plate_by_clustered_pcd(pcd: geometry.PointCloud, clustered_pcds: list[geometry.PointCloud]):
    obbs = []
    for clustered_pcd in clustered_pcds:
        # 对每个聚类结果进行平面检测
        clustered_pcd.estimate_normals(search_param=geometry.KDTreeSearchParamKNN(knn=30))
        patches = clustered_pcd.detect_planar_patches(normal_variance_threshold_deg=20,
                                                      coplanarity_deg=10,
                                                      outlier_ratio=0.2,
                                                      min_plane_edge_length=0,
                                                      min_num_points=0)
        # 把所有检测到的平面合并起来
        obbs.extend(patches)
    
    # 找到最大的 OBB（避免与标定板相同材质的杂质干扰）
    biggest_obb, _ = _find_biggest_obb(obbs)

    # 扩展 OBB
    indices, _ = _expand_obb(pcd, biggest_obb)
    cropped_pcd = pcd.select_by_index(indices)
    return cropped_pcd, biggest_obb


# 找到最大的 OBB
def _find_biggest_obb(obbs: list[geometry.OrientedBoundingBox]):
    logger.info(f"Start to find obb from {len(obbs)} obbs.")

    # 拿出每个边的边长
    extents = [obb.extent for obb in obbs]
    areas = []
    # 计算每个 OBB 的面积
    for extent in extents:
        sorted_extent = sorted(extent)
        area = sorted_extent[1] * sorted_extent[2]  # Calculate area of remaining two dimensions
        areas.append(area)

    # 通过面积构建索引
    sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i], reverse=True)
    # 找到最大的 OBB 和其他的 OBB
    biggest_obb = obbs[sorted_indices[0]]
    other_obbs = [obbs[i] for i in sorted_indices[1:]]
    logger.info(f"Biggest obb has been founded with area of {areas[sorted_indices[0]]}.")
    logger.info(f"With {len(other_obbs)} other obbs.")
    return biggest_obb, other_obbs


# 移除标定板边界
def remove_boundary(pcd, pre_remove_percent=0.02):
    # 先压缩成 2D，避免 z 轴的干扰
    pcd2d = PcdDimension.compress_to_2d(pcd)
    # 先移除固定的 pre_remove_percent 比例的边界，抹去标定板边界处可能有的聚集性的杂质干扰
    bound_removed_pcd = _remove_bound(pcd2d, pre_remove_percent)
    # 再进行一次下采样，避免密度对标定板平面检测的影响
    downsampled_pcd = PcdDownsample.downsample_pcd_by_voxel(bound_removed_pcd)

    # 用颜色因素聚类
    colors = np.asarray(downsampled_pcd.colors).mean(axis=1).reshape(-1, 1)
    clustered_pcds = _cluster_pcd(downsampled_pcd, colors)
    clustered_pcds.sort(key=lambda pcd: len(pcd.points), reverse=True)

    # 最多颜色和最少颜色分别的点数
    max_points_num = len(clustered_pcds[0].points)
    min_points_num = len(clustered_pcds[1].points)

    # 如果相差不大则视为标定板没有边框，不用去除
    if max_points_num - min_points_num > min_points_num / 20:
        logger.info("Cut the bounding.")
        min_clustered_pcd = clustered_pcds[1]
        min_clustered_pcd = PcdDimension.convert_to_layers(min_clustered_pcd, 1e-8)

        # 对最小的聚类的点进行平面检测
        obbs = []
        for clustered_pcd in [min_clustered_pcd]:
            clustered_pcd.estimate_normals(search_param=geometry.KDTreeSearchParamKNN(knn=10))
            patches = clustered_pcd.detect_planar_patches(normal_variance_threshold_deg=1,
                                                          coplanarity_deg=10,
                                                          outlier_ratio=0.05,
                                                          min_plane_edge_length=0,
                                                          min_num_points=0)
            obbs.extend(patches)
        biggest_obb, _ = _find_biggest_obb(obbs)

        # 直接获取最大平面视为标定板平面
        indices = biggest_obb.get_point_indices_within_bounding_box(min_clustered_pcd.points)
        bounding_removed_min_pcd = min_clustered_pcd.select_by_index(indices)
        # 手动构建 OBB（不知道为什么，自动生成的 OBB 是歪的）
        x_min, x_max, y_min, y_max = _get_min_max_xy(bounding_removed_min_pcd)
        boundary_removed_obb = _create_obb_from_bounds(x_min, x_max, y_min, y_max)

        # 提取边界内的点云
        indices = boundary_removed_obb.get_point_indices_within_bounding_box(pcd2d.points)
        boundary_removed_pcd = pcd2d.select_by_index(indices)
        return boundary_removed_pcd, boundary_removed_obb
    else:
        return pcd2d, pcd.get_oriented_bounding_box()


# 按固定比例移除边界
def _remove_bound(pcd, thickness_percent=0.01):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # 最小坐标和最大坐标
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # 需要移除的厚度
    x_thickness = (max_coords[0] - min_coords[0]) * thickness_percent
    y_thickness = (max_coords[1] - min_coords[1]) * thickness_percent
    
    # 构建掩码
    mask = ((points[:, 0] > min_coords[0] + x_thickness) & (points[:, 0] < max_coords[0] - x_thickness) &
            (points[:, 1] > min_coords[1] + y_thickness) & (points[:, 1] < max_coords[1] - y_thickness))
    
    # 提取点云并储存在新对象中
    filtered_points = points[mask]
    filtered_colors = colors[mask]
    filtered_pcd = geometry.PointCloud()
    filtered_pcd.points = utility.Vector3dVector(filtered_points)
    filtered_pcd.colors = utility.Vector3dVector(filtered_colors)
    
    return filtered_pcd


# 获取最小最大 x, y 坐标
def _get_min_max_xy(pcd):
    points = np.asarray(pcd.points)

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    return x_min, x_max, y_min, y_max


# 通过 x, y 坐标创建 OBB
def _create_obb_from_bounds(x_min=-1e-8, x_max=1e-8, y_min=-1e-8, y_max=1e-8, z_min=-1e-8, z_max=1e-8):
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # 计算 OBB 边长
    width  = x_max - x_min
    height = y_max - y_min
    depth  = z_max - z_min

    # 手动构建 OBB
    obb = geometry.OrientedBoundingBox(center=[x_center, y_center, z_center], R=np.eye(3), extent=[width, height, depth])
    return obb