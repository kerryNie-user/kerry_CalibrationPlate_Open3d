import numpy as np
import open3d.geometry as geometry

from logger import info, debug

class DownSampler:
    def __init__(self):
        info("Down Sampler has been created.")

    def voxel_pcd(self, pcd: geometry.PointCloud, voxel_size: int = 0.01, deviation_threshod=250):
        debug(f"Start to search for voxel size from {voxel_size}")
        previous_voxel_size = 0
        density_deviation = 0
        while density_deviation < deviation_threshod:
            previous_voxel_size = voxel_size
            downsampled_pcd = pcd.voxel_down_sample(voxel_size)
            density_deviation = self.compute_density_mean_deviation(downsampled_pcd)
            debug(f"voxel_size == {voxel_size}, density_deviation == {density_deviation}")
            voxel_size *= 0.9

        if previous_voxel_size == 0:
            raise RuntimeError("Your pcd is too sparse, try to larger voxel_size.")
        else:
            info(f"Voxel_size {previous_voxel_size} is fixed to this pcd.")
            info("If still not so good, try to lower the threshod.")
            downsampled_pcd = pcd.voxel_down_sample(previous_voxel_size)
            return downsampled_pcd

    def compute_density_mean_deviation(self, pcd: geometry.PointCloud, num_neighbors: int = 10):
        densities = self.compute_local_density(pcd, num_neighbors)
        mean_density = np.mean(densities)
        mean_deviation = np.mean(np.abs(densities - mean_density))
        return mean_deviation

    def compute_local_density(self, pcd: geometry.PointCloud, num_neighbors: int = 10):
        points = np.asarray(pcd.points)
        kdtree = geometry.KDTreeFlann(pcd)
        densities = []
        for i, point in enumerate(points):
            _, idx, _ = kdtree.search_knn_vector_3d(point, num_neighbors)
            local_density = 0
            for j in range(1, len(idx)):  # 跳过第一个索引（是它自己）
                dist = np.linalg.norm(points[i] - points[idx[j]])  # 计算两点之间的欧几里得距离
                local_density += 1 / dist if dist != 0 else 0  # 取倒数，避免分母为零
            densities.append(local_density)
        return np.array(densities)
