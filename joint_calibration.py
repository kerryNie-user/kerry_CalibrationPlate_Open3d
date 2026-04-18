import argparse
from pathlib import Path

import cv2
import numpy as np
from src.board_spec import BoardSpec
from src.calibration_ordering import build_order_candidates
from src.calibration_reporting import classify_score, save_mat_results, save_results
from src.calibration_solver import calibrate, project_lidar_to_image
from src.calibration_visualization import render_lidar_projection
from src.io_contract import (
    read_camera_corners as load_camera_corners,
    read_camera_params as load_camera_params,
    read_lidar_cloud as load_lidar_cloud,
    read_lidar_corners as load_lidar_corners,
)


def main():
    parser = argparse.ArgumentParser(description="相机-雷达联合标定与匹配评估")
    parser.add_argument("--camera-csv", default="output/camera_corner.csv")
    parser.add_argument("--lidar-csv", default="output/lidar_corner.csv")
    parser.add_argument("--camera-params", default="output/camera_params.npz")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--pattern-cols", type=int, default=9, help="棋盘内角点列数")
    parser.add_argument("--pattern-rows", type=int, default=6, help="棋盘内角点行数")
    parser.add_argument("--image", default="res/chess.jpg")
    parser.add_argument("--lidar-cloud", default="res/lidar.csv")
    parser.add_argument("--vis-overlay", default="output/lidar_projection_overlay.jpg")
    parser.add_argument("--vis-canvas", default="output/lidar_projection_canvas.jpg")
    parser.add_argument("--vis-point-radius", type=int, default=1)
    parser.add_argument("--vis-max-points", type=int, default=120000)
    parser.add_argument("--output-mat", default="output/cam_lidar_extrinsic.mat")
    args = parser.parse_args()

    camera_csv = Path(args.camera_csv)
    lidar_csv = Path(args.lidar_csv)
    params_path = Path(args.camera_params)
    output_dir = Path(args.output_dir)
    board_spec = BoardSpec(inner_cols=args.pattern_cols, inner_rows=args.pattern_rows)

    image_points = load_camera_corners(camera_csv)
    lidar_points = load_lidar_corners(lidar_csv)
    if len(image_points) != len(lidar_points):
        raise RuntimeError(f"角点数量不一致: camera={len(image_points)}, lidar={len(lidar_points)}")
    board_spec.validate_point_count(len(image_points))
    board_spec.validate_point_count(len(lidar_points))
    camera_matrix, dist_coeffs = load_camera_params(params_path)
    candidates = build_order_candidates(lidar_points, board_spec.inner_cols, board_spec.inner_rows)
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
    save_mat_results(
        mat_path=Path(args.output_mat),
        transform=transform,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        order_config=order,
        metrics=metrics,
    )

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
    print(f"  {Path(args.output_mat)}")
    print(f"  {Path(args.vis_overlay)}")
    print(f"  {Path(args.vis_canvas)}")
    print(f"可视化可见点: {vis_stats['num_visible']}/{vis_stats['num_total']}")


if __name__ == "__main__":
    main()
