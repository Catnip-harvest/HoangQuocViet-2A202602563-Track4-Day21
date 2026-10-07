"""Hai metric kiểm tra calibration LiDAR-camera, dùng chung cho mọi script trong src/.

1. hit_ratio (cần label): % điểm LiDAR nằm trong 3D box của vật thể (theo calib gốc) mà khi
   chiếu bằng calib cần kiểm tra vẫn rơi vào 2D box của vật thể đó.
2. edge_score (không cần label): % điểm "mép" của LiDAR (chỗ khoảng cách nhảy đột ngột giữa
   2 điểm liền kề trong một vòng quét) rơi gần một cạnh Canny của ảnh (<= EDGE_TOL_PX pixel).
"""
from __future__ import annotations

import cv2
import numpy as np

from starter.projection import project_velo_to_image, velo_to_cam

CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Cyclist")
RANGE_BINS = ((0, 15, "0-15m"), (15, 30, "15-30m"), (30, 1e9, ">30m"))
EDGE_JUMP_M = 0.5    # bước nhảy khoảng cách tối thiểu để coi điểm là "mép" vật thể
EDGE_TOL_PX = 3.0    # điểm mép cách cạnh ảnh không quá chừng này pixel thì tính là khớp


def finite_points(points: np.ndarray) -> np.ndarray:
    return points[np.isfinite(points).all(axis=1)]


def points_in_box(points_cam: np.ndarray, obj) -> np.ndarray:
    """Mask (N,) các điểm (đã ở camera frame) nằm trong 3D box của label (location = tâm đáy)."""
    h, w, l = obj.dimensions
    c, s = np.cos(obj.rotation_y), np.sin(obj.rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (points_cam - obj.location) @ R        # toạ độ của điểm trong hệ trục gắn với box
    return ((np.abs(local[:, 0]) <= l / 2) & (local[:, 1] <= 0) & (local[:, 1] >= -h)
            & (np.abs(local[:, 2]) <= w / 2))


def range_bin(distance_m: float) -> str:
    return next(name for lo, hi, name in RANGE_BINS if lo <= distance_m < hi)


def object_hits_for_frame(fr: dict, calib_test, points: np.ndarray | None = None) -> list[dict]:
    """Một dòng cho mỗi vật thể: số điểm trên vật thể và số điểm rơi đúng vào 2D box khi chiếu bằng calib_test.

    Điểm thuộc vật thể được xác định bằng calib gốc fr["calib"], vì drift không đổi vị trí thật của vật."""
    pts = finite_points(fr["points"] if points is None else points)
    cam_true = velo_to_cam(pts[:, :3], fr["calib"])
    uv, _, mask = project_velo_to_image(pts, calib_test, fr["image"].shape)
    uv_all = np.full((len(pts), 2), np.nan)
    uv_all[mask] = uv
    rows = []
    for i, obj in enumerate(fr["labels"]):
        if obj.type not in CLASSES:
            continue
        sel = points_in_box(cam_true, obj) & mask
        u, v = uv_all[sel, 0], uv_all[sel, 1]
        x1, y1, x2, y2 = obj.bbox
        hits = int(((u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)).sum())
        dist = float(np.hypot(obj.location[0], obj.location[2]))
        rows.append({"object_id": i, "type": obj.type, "distance_m": round(dist, 1),
                     "range_bin": range_bin(dist), "box_width_px": round(float(x2 - x1), 1),
                     "object_points": int(sel.sum()), "hits": hits})
    return rows


def hit_ratio(rows: list[dict]) -> float:
    n = sum(r["object_points"] for r in rows)
    return sum(r["hits"] for r in rows) / n if n else float("nan")


def image_edge_distance(image: np.ndarray) -> np.ndarray:
    """Bản đồ (H, W): khoảng cách (pixel) từ mỗi pixel tới cạnh Canny gần nhất."""
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    return cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)


def lidar_edge_mask(points: np.ndarray) -> np.ndarray:
    """Điểm mép: gần hơn điểm liền kề (trong thứ tự quét) ít nhất EDGE_JUMP_M mét.

    File .bin lưu điểm theo thứ tự quét, nên 2 điểm liền nhau trong mảng là 2 tia kề nhau
    trên cùng một vòng. Lấy phía gần (foreground) vì đó là đường viền của vật thể."""
    r = np.linalg.norm(points[:, :3], axis=1)
    jump_prev = np.r_[0.0, r[:-1] - r[1:]]
    jump_next = np.r_[r[1:] - r[:-1], 0.0]
    return np.maximum(jump_prev, jump_next) > EDGE_JUMP_M


def edge_score(fr: dict, calib_test, points: np.ndarray | None = None,
               edge_dist: np.ndarray | None = None) -> float:
    pts = finite_points(fr["points"] if points is None else points)
    pts = pts[lidar_edge_mask(pts)]
    uv, _, _ = project_velo_to_image(pts, calib_test, fr["image"].shape)
    if len(uv) == 0:
        return float("nan")
    if edge_dist is None:
        edge_dist = image_edge_distance(fr["image"])
    d = edge_dist[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    return float((d <= EDGE_TOL_PX).mean())
