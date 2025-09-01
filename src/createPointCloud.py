import open3d.geometry as geometry
import open3d.utility as utility
import numpy as np

from logger import info

def createPointCloud(points: np.ndarray, reflectivities: np.ndarray):
    info("Start to create Point Cloud.")
    tensities = (reflectivities - reflectivities.min()) / (np.ptp(reflectivities) + 1e-8)

    pcd = geometry.PointCloud()
    pcd.points = utility.Vector3dVector(points)
    pcd.colors = utility.Vector3dVector([(tensity, tensity, tensity) for tensity in tensities])
    info("Point Cloud has been created.")
    return pcd