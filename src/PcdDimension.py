import numpy as np
import open3d.utility as utility
import open3d.geometry as geometry

from logger import info

def compress_to_2d(pcd):
    points2d = np.asarray(pcd.points)
    points2d[:, 2] = 0
    pcd2d = geometry.PointCloud(pcd)
    pcd2d.points = utility.Vector3dVector(points2d)
    info("Pcd has been compressed to 2 dimensions.")
    return pcd2d

def convert_to_layers(pcd2d, z=1e-8):
    points = np.asarray(pcd2d.points)
            
    points_layer1 = points.copy()
    points_layer1[:, 2] = z

    points_layer2 = points.copy()
    points_layer2[:, 2] = -z
    
    pcd_layer1 = geometry.PointCloud()
    pcd_layer1.points = utility.Vector3dVector(points_layer1)

    pcd_layer2 = geometry.PointCloud()
    pcd_layer2.points = utility.Vector3dVector(points_layer2)

    pcd_combined = pcd2d + pcd_layer1 + pcd_layer2
    info(f"Pcd has been convert to 3 layers with {2 * z} depth.")
    return pcd_combined
