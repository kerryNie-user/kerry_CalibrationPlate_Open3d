import csv
import json
from pathlib import Path

import numpy as np


def classify_score(score: float) -> str:
    if score >= 85:
        return "优秀"
    if score >= 70:
        return "良好"
    if score >= 55:
        return "可用"
    return "偏弱"


def save_mat_results(
    mat_path: Path,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    order_config: dict,
    metrics: dict,
):
    def write_mat_v4_matrix(f, name: str, arr):
        data = np.asarray(arr, dtype=np.float64)
        if data.ndim == 0:
            data = data.reshape(1, 1)
        elif data.ndim == 1:
            data = data.reshape(-1, 1)
        rows, cols = data.shape
        type_code = np.int32(0)
        imagf = np.int32(0)
        name_bytes = name.encode("ascii", errors="ignore") + b"\x00"
        header = np.array([type_code, rows, cols, imagf, len(name_bytes)], dtype=np.int32)
        f.write(header.tobytes(order="C"))
        f.write(name_bytes)
        f.write(np.asfortranarray(data).tobytes(order="F"))

    mat_path.parent.mkdir(parents=True, exist_ok=True)
    with mat_path.open("wb") as f:
        write_mat_v4_matrix(f, "R", transform[:3, :3].astype(np.float64))
        write_mat_v4_matrix(f, "T", transform[:3, 3].reshape(3, 1).astype(np.float64))
        write_mat_v4_matrix(f, "T_cam_lidar", transform.astype(np.float64))
        write_mat_v4_matrix(f, "R_cam_lidar", transform[:3, :3].astype(np.float64))
        write_mat_v4_matrix(f, "t_cam_lidar", transform[:3, 3].reshape(3, 1).astype(np.float64))
        write_mat_v4_matrix(f, "camera_matrix", camera_matrix.astype(np.float64))
        write_mat_v4_matrix(f, "dist_coeffs", dist_coeffs.astype(np.float64))
        write_mat_v4_matrix(f, "pattern_source_cols", [[float(order_config["source_cols"])]])
        write_mat_v4_matrix(f, "pattern_source_rows", [[float(order_config["source_rows"])]])
        write_mat_v4_matrix(f, "pattern_transpose", [[float(int(order_config["transpose"]))]])
        write_mat_v4_matrix(f, "pattern_flip_x", [[float(int(order_config["flip_x"]))]])
        write_mat_v4_matrix(f, "pattern_flip_y", [[float(int(order_config["flip_y"]))]])
        for k, v in metrics.items():
            if isinstance(v, (int, float, np.integer, np.floating)):
                write_mat_v4_matrix(f, f"metric_{k}", [[float(v)]])


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
