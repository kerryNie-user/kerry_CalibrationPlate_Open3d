import numpy as np


def line_intersection(line1_start, line1_end, line2_start, line2_end):
    direct1 = np.array(line1_end) - np.array(line1_start)
    direct2 = np.array(line2_end) - np.array(line2_start)
    denom = direct1[0] * direct2[1] - direct1[1] * direct2[0]

    if denom != 0:
        direct_lines = np.array(line2_start) - np.array(line1_start)
        t = (direct_lines[0] * direct2[1] - direct_lines[1] * direct2[0]) / denom
        u = (direct_lines[0] * direct1[1] - direct_lines[1] * direct1[0]) / denom
        if 0 <= t <= 1 and 0 <= u <= 1:
            return np.array(line1_start) + t * direct1
    return None


def compute_intersection_grid(cols_points, rows_points):
    points = []
    for i in range(int(len(cols_points) / 2)):
        col_start = cols_points[2 * i]
        col_end = cols_points[2 * i + 1]
        for j in range(int(len(rows_points) / 2)):
            row_start = rows_points[2 * j]
            row_end = rows_points[2 * j + 1]
            intersection_point = line_intersection(col_start, col_end, row_start, row_end)
            if intersection_point is None:
                raise RuntimeError(f"交点计算失败: col={i}, row={j}")
            points.append(intersection_point)
    return np.asarray(points, dtype=np.float64)


def validate_corner_points(points, expected_count):
    if points is None or len(points) == 0:
        raise RuntimeError("没有可导出的角点")
    if len(points) != expected_count:
        raise RuntimeError(f"角点数量不正确: expected={expected_count}, actual={len(points)}")
    return points
