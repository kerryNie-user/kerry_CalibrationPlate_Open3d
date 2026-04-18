# -*- coding: utf-8 -*-
from pathlib import Path

import logger
from io_contract import read_lidar_cloud, write_corner_csv

# 从 csv 读取点和反射率
def readCSV(file_path: str):
    csv_path = Path(file_path)
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"File {file_path} is not a CSV file")
    return read_lidar_cloud(csv_path)

# 从 ply 读取点和反射率（intensity 字段，无则补零）
def readPLY(file_path: str):
    ply_path = Path(file_path)
    if ply_path.suffix.lower() != ".ply":
        raise ValueError(f"File {file_path} is not a PLY file")
    return read_lidar_cloud(ply_path)

# 根据文件扩展名自动选择读取方式
def readPoints(file_path: str):
    return read_lidar_cloud(file_path)

# 向 csv 写入标定点的坐标值
def writeCSV(file_path: str, points: list):
    write_corner_csv(file_path, points)
