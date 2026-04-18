import cv2
import numpy as np


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
