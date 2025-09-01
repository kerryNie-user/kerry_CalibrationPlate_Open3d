# main.py  — Open3D 0.19, 鼠标射线检测 OBB
import open3d.geometry as geometry
from logger import *
from readCSV import readCSV, writeCSV
from createPointCloud import createPointCloud
from ObbSelectApplication import ObbSelectApplication

if __name__ == "__main__":
    info("<<< Calibration Plate App >>>")

    logging.getLogger().setLevel(INFO)

    points, reflectivities = readCSV("res", "map.csv")
    pcd = createPointCloud(points, reflectivities)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

    # 检测平面并得到 OBB 列表
    pcd.estimate_normals(search_param=geometry.KDTreeSearchParamKNN(knn=30))
    patches = pcd.detect_planar_patches(
        normal_variance_threshold_deg=20,
        coplanarity_deg=75,
        outlier_ratio=0.2,
        min_plane_edge_length=0,
        min_num_points=0)

    app = ObbSelectApplication(pcd, patches)
    app.run()

    points = app.get_corner_points()
    writeCSV("output", "corner.csv", points)

    info(">>> Calibration Plate App <<<")