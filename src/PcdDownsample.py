# -*- coding: utf-8 -*-
import numpy as np
import open3d.geometry as geometry

import logger

# 基于体素下采样的点云下采样
def downsample_pcd_by_voxel(pcd: geometry.PointCloud, initial_voxel_size: int = 0.01, max_deviation_threshold: float = 150):
    if max_deviation_threshold <= 0:
        raise ValueError("max_deviation_threshold must be greater than 0.")

    if initial_voxel_size <= 0:
        raise ValueError("initial_voxel_size must be greater than 0.")

    logger.debug(f"Start to search for voxel size starting at {initial_voxel_size}")

    # 下采样大小（相当于从 voxel_size 大小的体素中采样一个点），所以值越大，采样的点越稀疏
    voxel_size = initial_voxel_size
    # 点云密度方差
    density_deviation = 0

    while density_deviation < max_deviation_threshold:
        # 如果可以，会尽可能多地采集点，只要密度方差不会过高
        voxel_size *= 0.8

        # 对点云进行下采样
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        # 计算点云密度方差
        density_deviation = _compute_density_deviation(downsampled_pcd)
        logger.debug(f"Voxel size: {voxel_size}, Density deviation: {density_deviation}")
        

    logger.info(f"Optimal voxel size is {voxel_size}.")
    logger.info("If this is not satisfactory, consider reducing the deviation threshold.")
    
    return downsampled_pcd

# 计算点云密度方差
def _compute_density_deviation(pcd: geometry.PointCloud, num_neighbors: int = 10):
    # 计算点云密度
    densities = _compute_local_density(pcd, num_neighbors)
    # 计算点云平均密度
    mean_density = np.mean(densities)
    # 计算点云方差
    variance = np.abs(densities - mean_density)
    # 计算密度方差平均值
    return np.mean(variance)

# 计算点云局部密度
def _compute_local_density(pcd: geometry.PointCloud, num_neighbors: int = 10):
    points = np.asarray(pcd.points)
    num_points = len(points)

    # 构建 KD 树
    kdtree = geometry.KDTreeFlann(pcd)

    # 获取所有点的邻居索引
    densities = np.zeros(num_points)
    for i in range(num_points):
        _, idx, _ = kdtree.search_knn_vector_3d(points[i], num_neighbors)

        # 跳过第一个点（即点本身）
        neighbors = points[idx[1:], :]
        # 获取该点和其邻居之间的距离
        distances = np.linalg.norm(neighbors - points[i], axis=1)

        # 计算局部密度（使用距离的倒数）
        densities[i] = np.sum(1.0 / distances[distances != 0])

    return densities