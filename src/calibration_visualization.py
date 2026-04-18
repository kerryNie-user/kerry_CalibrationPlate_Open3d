from pathlib import Path

import cv2
import numpy as np


def build_point_colors(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    v_min = float(np.percentile(values, 2))
    v_max = float(np.percentile(values, 98))
    if v_max <= v_min:
        norm = np.zeros(len(values), dtype=np.float64)
    else:
        norm = np.clip((values - v_min) / (v_max - v_min), 0.0, 1.0)
    idx = (norm * 255.0).astype(np.uint8)
    colors = cv2.applyColorMap(255 - idx.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return colors


def render_lidar_projection(
    image,
    projected_points,
    depths,
    color_values,
    output_overlay_path: Path,
    output_canvas_path: Path,
    point_radius: int,
):
    h, w = image.shape[:2]
    valid = depths > 0
    u = projected_points[:, 0]
    v = projected_points[:, 1]
    valid &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv = projected_points[valid]
    depth_valid = depths[valid]
    color_values_valid = color_values[valid]
    colors = build_point_colors(color_values_valid)
    overlay = image.copy()
    canvas = np.zeros_like(image)
    for (px, py), color in zip(uv, colors):
        center = (int(round(px)), int(round(py)))
        bgr = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(overlay, center, point_radius, bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, center, point_radius, bgr, -1, lineType=cv2.LINE_AA)
    blended = cv2.addWeighted(image, 0.55, overlay, 0.95, 0.0)
    output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_overlay_path), blended)
    cv2.imwrite(str(output_canvas_path), canvas)
    return {
        "num_total": int(len(projected_points)),
        "num_visible": int(len(uv)),
        "depth_min": float(np.min(depth_valid)) if len(depth_valid) else float("nan"),
        "depth_max": float(np.max(depth_valid)) if len(depth_valid) else float("nan"),
    }
