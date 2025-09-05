import numpy as np
import open3d.geometry as geometry
from logger import info, debug

def downsample_pcd_by_voxel(pcd: geometry.PointCloud, initial_voxel_size: int = 0.01, max_deviation_threshold: float = 150):
    """
    Downsamples the point cloud using voxel size, optimizing for uniform point density.

    Args:
        pcd: The point cloud to downsample.
        initial_voxel_size: The starting voxel size for downsampling.
        max_deviation_threshold: The maximum allowable density deviation to stop downsampling.

    Returns:
        A downsampled point cloud with optimized voxel size.
    """
    debug(f"Start to search for voxel size starting at {initial_voxel_size}")
    voxel_size = initial_voxel_size
    prev_voxel_size = 0
    density_deviation = 0

    while density_deviation < max_deviation_threshold:
        prev_voxel_size = voxel_size
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        density_deviation = _compute_density_deviation(downsampled_pcd)
        debug(f"Voxel size: {voxel_size}, Density deviation: {density_deviation}")
        voxel_size *= 0.8

    if prev_voxel_size == 0:
        raise RuntimeError("Point cloud is too sparse. Try using a larger voxel size.")
    else:
        info(f"Optimal voxel size is {prev_voxel_size}.")
        info("If this is not satisfactory, consider reducing the deviation threshold.")
        # Final downsampling with the selected voxel size
        return pcd.voxel_down_sample(prev_voxel_size)

def _compute_density_deviation(pcd: geometry.PointCloud, num_neighbors: int = 10):
    """
    Computes the mean deviation of local point density in the point cloud.

    Args:
        pcd: The point cloud.
        num_neighbors: The number of neighbors to consider when calculating local density.

    Returns:
        The mean deviation of local density values.
    """
    densities = _compute_local_density(pcd, num_neighbors)
    mean_density = np.mean(densities)
    return np.mean(np.abs(densities - mean_density))

def _compute_local_density(pcd: geometry.PointCloud, num_neighbors: int = 10):
    """
    Computes the local density for each point in the point cloud.

    Args:
        pcd: The point cloud.
        num_neighbors: The number of neighbors to consider when calculating local density.

    Returns:
        A numpy array containing the local densities for each point.
    """
    points = np.asarray(pcd.points)
    kdtree = geometry.KDTreeFlann(pcd)
    densities = []

    for i, point in enumerate(points):
        _, idx, _ = kdtree.search_knn_vector_3d(point, num_neighbors)
        local_density = 0
        for j in range(1, len(idx)):  # Skip the first index (the point itself)
            dist = np.linalg.norm(points[i] - points[idx[j]])  # Calculate Euclidean distance
            if dist != 0:
                local_density += 1 / dist  # Use inverse distance to measure density
        densities.append(local_density)

    return np.array(densities)