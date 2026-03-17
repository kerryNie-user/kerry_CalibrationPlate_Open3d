# -*- coding: utf-8 -*-
import csv
import numpy as np
import open3d as o3d
from pathlib import Path

import logger

# 列索引（CSV 格式）
X = 8
Y = 9
Z = 10
R = 11

# 从 csv 读取点和反射率
def readCSV(file_path: str):
    csv_path = Path(file_path)

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"File {file_path} is not a CSV file")

    logger.info(f"Start to read points from {csv_path}.")

    points = []
    reflectivities = []
    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        # 跳过表头
        next(reader)
        for row in reader:
            try:
                point = [float(row[X]), float(row[Y]), float(row[Z])]
                reflectivity = float(row[R])
                points.append(point)
                reflectivities.append(reflectivity)
            except ValueError as e:
                logger.debug(f"Skipping invalid row {row}: {e}")

    points = np.asarray(points)
    reflectivities = np.asarray(reflectivities)
    logger.info(f"All {len(points)} points has been read from {csv_path}.")
    return points, reflectivities

# 从 ply 读取点和反射率（intensity 字段，无则补零）
def readPLY(file_path: str):
    ply_path = Path(file_path)

    if ply_path.suffix.lower() != ".ply":
        raise ValueError(f"File {file_path} is not a PLY file")

    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    logger.info(f"Start to read points from {ply_path}.")

    # 尝试用张量 API 读取，以支持自定义 intensity 字段
    try:
        t_pcd = o3d.t.io.read_point_cloud(str(ply_path))
        points = t_pcd.point.positions.numpy()

        # 按优先级查找反射率字段
        for field in ("intensity", "reflectivity", "scalar_intensity"):
            if field in t_pcd.point:
                reflectivities = t_pcd.point[field].numpy().flatten().astype(np.float64)
                logger.info(f"Using '{field}' as reflectivity.")
                break
        else:
            # PLY 无强度字段时用颜色灰度代替，仍无则补零
            if t_pcd.point.colors is not None and len(t_pcd.point.colors) > 0:
                colors = t_pcd.point.colors.numpy()
                reflectivities = colors[:, 0].astype(np.float64)
                logger.info("No intensity field found; using color channel as reflectivity.")
            else:
                reflectivities = np.zeros(len(points), dtype=np.float64)
                logger.info("No intensity field found; using zeros as reflectivity.")
    except Exception as e:
        logger.debug(f"Tensor API failed ({e}), falling back to legacy reader.")
        pcd = o3d.io.read_point_cloud(str(ply_path))
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            reflectivities = colors[:, 0].astype(np.float64)
        else:
            reflectivities = np.zeros(len(points), dtype=np.float64)

    logger.info(f"All {len(points)} points has been read from {ply_path}.")
    return points, reflectivities

# 根据文件扩展名自动选择读取方式
def readPoints(file_path: str):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return readCSV(file_path)
    elif suffix == ".ply":
        return readPLY(file_path)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Supported: .csv, .ply")

# 向 csv 写入标定点的坐标值
def writeCSV(file_path: str, points: list):
    if len(points) == 0:
        raise RuntimeError("You have not choose any points")

    csv_path = Path(file_path)

    # 如果路径不存在，自动创建目录
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open(mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])

        for point in points:
            if len(point) != 3:
                raise RuntimeError(f"Point size should be 3, but {len(point)} has been given")
            writer.writerow(point)

    logger.suggestion(f"All {len(points)} points have been written to {csv_path}.")