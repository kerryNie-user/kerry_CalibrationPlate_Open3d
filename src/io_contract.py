# -*- coding: utf-8 -*-
import ast
import csv
from pathlib import Path

import numpy as np

try:
    from . import logger
except ImportError:
    import logger

POINT_CLOUD_12_COLS = (8, 9, 10, 11)
POINT_CLOUD_4_COLS = (0, 1, 2, 3)


def _as_path(path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def read_camera_corners(path) -> np.ndarray:
    csv_path = _as_path(path)
    points = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"空文件: {csv_path}")
        for row in reader:
            if len(row) >= 3:
                x, y = float(row[-2]), float(row[-1])
            elif len(row) == 2:
                x, y = float(row[0]), float(row[1])
            else:
                continue
            points.append([x, y])
    if len(points) == 0:
        raise RuntimeError(f"未读取到相机角点: {csv_path}")
    return np.asarray(points, dtype=np.float64)


def read_lidar_corners(path) -> np.ndarray:
    csv_path = _as_path(path)
    points = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"空文件: {csv_path}")
        for row in reader:
            if len(row) < 3:
                continue
            points.append([float(row[0]), float(row[1]), float(row[2])])
    if len(points) == 0:
        raise RuntimeError(f"未读取到雷达角点: {csv_path}")
    return np.asarray(points, dtype=np.float64)


def _read_lidar_cloud_csv(path: Path):
    points = []
    reflectivities = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"空文件: {path}")
        for row in reader:
            try:
                if len(row) >= 12:
                    x_idx, y_idx, z_idx, r_idx = POINT_CLOUD_12_COLS
                elif len(row) >= 4:
                    x_idx, y_idx, z_idx, r_idx = POINT_CLOUD_4_COLS
                elif len(row) >= 3:
                    x_idx, y_idx, z_idx = POINT_CLOUD_4_COLS[:3]
                    points.append([float(row[x_idx]), float(row[y_idx]), float(row[z_idx])])
                    reflectivities.append(float(row[z_idx]))
                    continue
                else:
                    continue
                points.append([float(row[x_idx]), float(row[y_idx]), float(row[z_idx])])
                reflectivities.append(float(row[r_idx]))
            except ValueError as exc:
                logger.debug(f"Skipping invalid row {row}: {exc}")
    if len(points) == 0:
        raise RuntimeError(f"未读取到雷达点云: {path}")
    return np.asarray(points, dtype=np.float64), np.asarray(reflectivities, dtype=np.float64)


def _read_lidar_cloud_ply(path: Path):
    import open3d as o3d

    if not path.exists():
        raise FileNotFoundError(f"PLY file not found: {path}")

    logger.info(f"Start to read points from {path}.")
    try:
        t_pcd = o3d.t.io.read_point_cloud(str(path))
        points = t_pcd.point.positions.numpy()
        for field in ("intensity", "reflectivity", "scalar_intensity"):
            if field in t_pcd.point:
                reflectivities = t_pcd.point[field].numpy().flatten().astype(np.float64)
                logger.info(f"Using '{field}' as reflectivity.")
                break
        else:
            if t_pcd.point.colors is not None and len(t_pcd.point.colors) > 0:
                colors = t_pcd.point.colors.numpy()
                reflectivities = colors[:, 0].astype(np.float64)
                logger.info("No intensity field found; using color channel as reflectivity.")
            else:
                reflectivities = np.zeros(len(points), dtype=np.float64)
                logger.info("No intensity field found; using zeros as reflectivity.")
    except Exception as exc:
        logger.debug(f"Tensor API failed ({exc}), falling back to legacy reader.")
        pcd = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pcd.points)
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
            reflectivities = colors[:, 0].astype(np.float64)
        else:
            reflectivities = np.zeros(len(points), dtype=np.float64)

    logger.info(f"All {len(points)} points has been read from {path}.")
    return points, reflectivities


def read_lidar_cloud(path):
    lidar_path = _as_path(path)
    suffix = lidar_path.suffix.lower()
    if suffix == ".csv":
        return _read_lidar_cloud_csv(lidar_path)
    if suffix == ".ply":
        return _read_lidar_cloud_ply(lidar_path)
    raise ValueError(f"Unsupported file format '{suffix}'. Supported: .csv, .ply")


def _read_camera_params_npz(path: Path):
    data = np.load(path)
    return data["camera_matrix"].astype(np.float64), data["dist_coeffs"].astype(np.float64)


def _read_camera_params_yaml(path: Path):
    text = path.read_text(encoding="utf-8")
    matrix_line = None
    dist_line = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("camera_matrix:"):
            tail = s.split(":", 1)[1].strip()
            if tail:
                matrix_line = tail
        if s.startswith("distortion_coefficients:"):
            tail = s.split(":", 1)[1].strip()
            if tail:
                dist_line = tail
    if matrix_line is None or dist_line is None:
        lines = [ln.rstrip("\n") for ln in text.splitlines()]
        for i, line in enumerate(lines):
            if line.strip() == "camera_matrix:" and i + 3 < len(lines):
                block = []
                for row in lines[i + 1 : i + 4]:
                    block.append(row.replace("-", "", 1).strip())
                matrix_line = "[" + ",".join(block) + "]"
            if line.strip() == "distortion_coefficients:" and i + 1 < len(lines):
                dist_line = lines[i + 1].replace("-", "", 1).strip()
    if matrix_line is None or dist_line is None:
        raise RuntimeError(f"无法解析相机参数文件: {path}")
    camera_matrix = np.asarray(ast.literal_eval(matrix_line), dtype=np.float64)
    dist_coeffs = np.asarray(ast.literal_eval(dist_line), dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise RuntimeError(f"camera_matrix 维度错误: {camera_matrix.shape}")
    return camera_matrix, dist_coeffs.reshape(1, -1)


def read_camera_params(path):
    params_path = _as_path(path)
    suffix = params_path.suffix.lower()
    if suffix == ".npz":
        return _read_camera_params_npz(params_path)
    if suffix in {".yaml", ".yml"}:
        return _read_camera_params_yaml(params_path)
    raise RuntimeError(f"不支持的相机参数格式: {params_path.suffix}")


def write_corner_csv(path, points):
    csv_path = _as_path(path)
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        raise RuntimeError("You have not choose any points")
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for point in points:
            if len(point) != 3:
                raise RuntimeError(f"Point size should be 3, but {len(point)} has been given")
            writer.writerow(point.tolist())
    logger.suggestion(f"All {len(points)} points have been written to {csv_path}.")
