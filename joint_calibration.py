import argparse
import ast
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def load_camera_corners(path: Path) -> np.ndarray:
    points = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"空文件: {path}")
        for row in reader:
            if len(row) >= 3:
                x, y = float(row[-2]), float(row[-1])
            elif len(row) == 2:
                x, y = float(row[0]), float(row[1])
            else:
                continue
            points.append([x, y])
    if len(points) == 0:
        raise RuntimeError(f"未读取到相机角点: {path}")
    return np.asarray(points, dtype=np.float64)


def load_lidar_corners(path: Path) -> np.ndarray:
    points = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"空文件: {path}")
        for row in reader:
            if len(row) < 3:
                continue
            x, y, z = float(row[0]), float(row[1]), float(row[2])
            points.append([x, y, z])
    if len(points) == 0:
        raise RuntimeError(f"未读取到雷达角点: {path}")
    return np.asarray(points, dtype=np.float64)


def load_lidar_cloud(path: Path):
    points = []
    values = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"空文件: {path}")
        for row in reader:
            if len(row) >= 12:
                x, y, z = float(row[8]), float(row[9]), float(row[10])
                v = float(row[11])
            elif len(row) >= 4:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
                v = float(row[3])
            elif len(row) >= 3:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
                v = z
            else:
                continue
            points.append([x, y, z])
            values.append(v)
    if len(points) == 0:
        raise RuntimeError(f"未读取到雷达点云: {path}")
    return np.asarray(points, dtype=np.float64), np.asarray(values, dtype=np.float64)


def load_camera_params_npz(path: Path):
    data = np.load(path)
    return data["camera_matrix"].astype(np.float64), data["dist_coeffs"].astype(np.float64)


def load_camera_params_yaml(path: Path):
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
    dist_coeffs = dist_coeffs.reshape(1, -1)
    return camera_matrix, dist_coeffs


def load_camera_params(path: Path):
    if path.suffix.lower() == ".npz":
        return load_camera_params_npz(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_camera_params_yaml(path)
    raise RuntimeError(f"不支持的相机参数格式: {path.suffix}")


def compute_metrics(image_points, reprojected_points, inliers_mask, cam_points):
    errors = np.linalg.norm(reprojected_points - image_points, axis=1)
    inliers_mask = inliers_mask.astype(bool) if inliers_mask is not None else np.ones(len(errors), dtype=bool)
    inlier_errors = errors[inliers_mask] if np.any(inliers_mask) else errors
    depth_positive_ratio = float(np.mean(cam_points[:, 2] > 0))
    rmse = float(np.sqrt(np.mean(inlier_errors**2)))
    mean = float(np.mean(inlier_errors))
    median = float(np.median(inlier_errors))
    p90 = float(np.percentile(inlier_errors, 90))
    maxe = float(np.max(inlier_errors))
    inlier1 = float(np.mean(inlier_errors < 1.0))
    inlier2 = float(np.mean(inlier_errors < 2.0))
    inlier3 = float(np.mean(inlier_errors < 3.0))
    stability = np.exp(-rmse / 2.0)
    score = 100.0 * (0.45 * stability + 0.25 * inlier2 + 0.20 * inlier3 + 0.10 * depth_positive_ratio)
    return {
        "rmse_px": rmse,
        "mean_px": mean,
        "median_px": median,
        "p90_px": p90,
        "max_px": maxe,
        "inlier_ratio_lt1px": inlier1,
        "inlier_ratio_lt2px": inlier2,
        "inlier_ratio_lt3px": inlier3,
        "depth_positive_ratio": depth_positive_ratio,
        "match_score_0_100": float(score),
        "num_points": int(len(errors)),
        "num_inliers": int(np.sum(inliers_mask)),
    }


def classify_score(score: float) -> str:
    if score >= 85:
        return "优秀"
    if score >= 70:
        return "良好"
    if score >= 55:
        return "可用"
    return "偏弱"


def reorder_points(points: np.ndarray, cols: int, rows: int, transpose: bool, flip_x: bool, flip_y: bool) -> np.ndarray:
    grid = points.reshape(rows, cols, -1)
    if transpose:
        grid = np.transpose(grid, (1, 0, 2))
    if flip_x:
        grid = grid[:, ::-1, :]
    if flip_y:
        grid = grid[::-1, :, :]
    return grid.reshape(-1, points.shape[1])


def build_order_candidates(points: np.ndarray, pattern_cols: int, pattern_rows: int):
    n = len(points)
    candidates = []
    source_shapes = [(pattern_cols, pattern_rows)]
    if pattern_cols != pattern_rows:
        source_shapes.append((pattern_rows, pattern_cols))
    seen = set()
    for cols, rows in source_shapes:
        if cols * rows != n:
            continue
        for transpose in (False, True):
            out_cols, out_rows = (rows, cols) if transpose else (cols, rows)
            if out_cols * out_rows != n:
                continue
            if out_cols != pattern_cols or out_rows != pattern_rows:
                continue
            for flip_x in (False, True):
                for flip_y in (False, True):
                    key = (cols, rows, transpose, flip_x, flip_y)
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "source_cols": cols,
                            "source_rows": rows,
                            "transpose": transpose,
                            "flip_x": flip_x,
                            "flip_y": flip_y,
                            "points": reorder_points(points, cols, rows, transpose, flip_x, flip_y),
                        }
                    )
    if len(candidates) == 0:
        candidates.append(
            {
                "source_cols": pattern_cols,
                "source_rows": pattern_rows,
                "transpose": False,
                "flip_x": False,
                "flip_y": False,
                "points": points.copy(),
            }
        )
    return candidates


def calibrate(lidar_points, image_points, camera_matrix, dist_coeffs):
    object_points = lidar_points.reshape(-1, 1, 3).astype(np.float64)
    image_points_cv = image_points.reshape(-1, 1, 2).astype(np.float64)
    ok_ransac, rvec_ransac, tvec_ransac, inliers = cv2.solvePnPRansac(
        object_points,
        image_points_cv,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=3.0,
        confidence=0.99,
        iterationsCount=200,
    )
    if ok_ransac:
        use_inliers = inliers.reshape(-1)
        ok_refine, rvec_refine, tvec_refine = cv2.solvePnP(
            object_points[use_inliers],
            image_points_cv[use_inliers],
            camera_matrix,
            dist_coeffs,
            rvec_ransac,
            tvec_ransac,
            True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok_refine:
            rvec, tvec = rvec_refine, tvec_refine
        else:
            rvec, tvec = rvec_ransac, tvec_ransac
        inlier_mask = np.zeros(len(lidar_points), dtype=bool)
        inlier_mask[use_inliers] = True
    else:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points_cv,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("solvePnP 求解失败")
        inlier_mask = np.ones(len(lidar_points), dtype=bool)
    rmat, _ = cv2.Rodrigues(rvec)
    reprojected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reprojected = reprojected.reshape(-1, 2)
    cam_points = (rmat @ lidar_points.T + tvec).T
    metrics = compute_metrics(image_points, reprojected, inlier_mask, cam_points)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rmat
    transform[:3, 3] = tvec.reshape(3)
    return transform, metrics, reprojected, inlier_mask


def project_lidar_to_image(lidar_points, transform, camera_matrix, dist_coeffs):
    rmat = transform[:3, :3]
    tvec = transform[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(rmat)
    object_points = lidar_points.reshape(-1, 1, 3).astype(np.float64)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected = projected.reshape(-1, 2)
    cam_points = (rmat @ lidar_points.T + tvec).T
    return projected, cam_points[:, 2]


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


def save_results(output_dir: Path, transform, metrics, inlier_mask, reprojected, order_config, ranking):
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "cam_lidar_extrinsic.npz",
        T_cam_lidar=transform,
        R=transform[:3, :3],
        t=transform[:3, 3],
    )
    payload = {
        "T_cam_lidar": transform.tolist(),
        "R": transform[:3, :3].tolist(),
        "t": transform[:3, 3].tolist(),
        "metrics": metrics,
        "best_order_config": order_config,
        "order_ranking_top": ranking,
        "quality_level": classify_score(metrics["match_score_0_100"]),
    }
    (output_dir / "calibration_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "reprojection_detail.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "u_proj", "v_proj", "inlier"])
        for i, (p, m) in enumerate(zip(reprojected, inlier_mask), start=1):
            writer.writerow([i, float(p[0]), float(p[1]), int(m)])


def main():
    parser = argparse.ArgumentParser(description="相机-雷达联合标定与匹配评估")
    parser.add_argument("--camera-csv", default="output/camera_corner.csv")
    parser.add_argument("--lidar-csv", default="output/lidar_corner.csv")
    parser.add_argument("--camera-params", default="output/camera_params.npz")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--pattern-cols", type=int, default=9)
    parser.add_argument("--pattern-rows", type=int, default=6)
    parser.add_argument("--image", default="res/chess.jpg")
    parser.add_argument("--lidar-cloud", default="res/lidar.csv")
    parser.add_argument("--vis-overlay", default="output/lidar_projection_overlay.jpg")
    parser.add_argument("--vis-canvas", default="output/lidar_projection_canvas.jpg")
    parser.add_argument("--vis-point-radius", type=int, default=1)
    parser.add_argument("--vis-max-points", type=int, default=120000)
    args = parser.parse_args()

    camera_csv = Path(args.camera_csv)
    lidar_csv = Path(args.lidar_csv)
    params_path = Path(args.camera_params)
    output_dir = Path(args.output_dir)

    image_points = load_camera_corners(camera_csv)
    lidar_points = load_lidar_corners(lidar_csv)
    if len(image_points) != len(lidar_points):
        raise RuntimeError(f"角点数量不一致: camera={len(image_points)}, lidar={len(lidar_points)}")
    camera_matrix, dist_coeffs = load_camera_params(params_path)
    candidates = build_order_candidates(lidar_points, args.pattern_cols, args.pattern_rows)
    best = None
    ranking = []
    for candidate in candidates:
        transform, metrics, reprojected, inlier_mask = calibrate(candidate["points"], image_points, camera_matrix, dist_coeffs)
        ranking.append(
            {
                "source_cols": candidate["source_cols"],
                "source_rows": candidate["source_rows"],
                "transpose": candidate["transpose"],
                "flip_x": candidate["flip_x"],
                "flip_y": candidate["flip_y"],
                "rmse_px": metrics["rmse_px"],
                "score_0_100": metrics["match_score_0_100"],
            }
        )
        if best is None or metrics["rmse_px"] < best["metrics"]["rmse_px"]:
            best = {
                "transform": transform,
                "metrics": metrics,
                "reprojected": reprojected,
                "inlier_mask": inlier_mask,
                "order": {
                    "source_cols": candidate["source_cols"],
                    "source_rows": candidate["source_rows"],
                    "transpose": candidate["transpose"],
                    "flip_x": candidate["flip_x"],
                    "flip_y": candidate["flip_y"],
                },
            }
    ranking = sorted(ranking, key=lambda x: x["rmse_px"])[:8]
    transform = best["transform"]
    metrics = best["metrics"]
    reprojected = best["reprojected"]
    inlier_mask = best["inlier_mask"]
    order = best["order"]
    save_results(output_dir, transform, metrics, inlier_mask, reprojected, order, ranking)

    image_path = Path(args.image)
    lidar_cloud_path = Path(args.lidar_cloud)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"图像读取失败: {image_path}")
    lidar_cloud_points, lidar_values = load_lidar_cloud(lidar_cloud_path)
    if len(lidar_cloud_points) > args.vis_max_points > 0:
        pick = np.linspace(0, len(lidar_cloud_points) - 1, args.vis_max_points, dtype=np.int64)
        lidar_cloud_points = lidar_cloud_points[pick]
        lidar_values = lidar_values[pick]
    projected_cloud, depths = project_lidar_to_image(lidar_cloud_points, transform, camera_matrix, dist_coeffs)
    vis_stats = render_lidar_projection(
        image=image,
        projected_points=projected_cloud,
        depths=depths,
        color_values=lidar_values,
        output_overlay_path=Path(args.vis_overlay),
        output_canvas_path=Path(args.vis_canvas),
        point_radius=max(1, int(args.vis_point_radius)),
    )

    print("联合标定完成")
    print(f"最佳角点顺序: {order}")
    print(f"质量等级: {classify_score(metrics['match_score_0_100'])}")
    print(f"匹配评分: {metrics['match_score_0_100']:.2f}/100")
    print(f"重投影 RMSE: {metrics['rmse_px']:.4f} px")
    print(f"P90 误差: {metrics['p90_px']:.4f} px")
    print(f"最大误差: {metrics['max_px']:.4f} px")
    print(f"2px 内比例: {metrics['inlier_ratio_lt2px']:.4f}")
    print(f"深度为正比例: {metrics['depth_positive_ratio']:.4f}")
    print("输出文件:")
    print(f"  {output_dir / 'cam_lidar_extrinsic.npz'}")
    print(f"  {output_dir / 'calibration_report.json'}")
    print(f"  {output_dir / 'reprojection_detail.csv'}")
    print(f"  {Path(args.vis_overlay)}")
    print(f"  {Path(args.vis_canvas)}")
    print(f"可视化可见点: {vis_stats['num_visible']}/{vis_stats['num_total']}")


if __name__ == "__main__":
    main()
