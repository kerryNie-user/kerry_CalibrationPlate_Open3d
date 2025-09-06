# -*- coding: utf-8 -*-
import csv
import numpy as np
from pathlib import Path

import logger

# 列索引
X = 8
Y = 9
Z = 10
R = 11

# 工作空间路径
work_space = Path(__file__).parent.parent

# 从 csv 读取点和颜色
def readCSV(directory_path: str, file_name: str):
    # csv 文件存储路径
    csv_path = work_space / directory_path / file_name

    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"File {file_name} is not a CSV file")

    logger.info(f"Start to read points from {csv_path}.")

    # 点列表
    points = []
    # 反射率列表
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

# 向 csv 写入标定点的坐标值
def writeCSV(directory_path, file_name, points: list):
    if len(points) == 0:
        raise RuntimeError("You have not choose any points")
        
    # csv 文件写入路径
    csv_path = work_space / directory_path / file_name

    # 如果路径不存在，新建 output 文件夹
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 把所有点都写入文件夹
    with csv_path.open(mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y', 'z'])

        for point in points:
            if len(point) != 3:
                raise RuntimeError(f"Point size should be 3, but {len(point)} has been given")

            writer.writerow(point)

    logger.suggestion(f"All {len(points)} points have been written to {csv_path}.")