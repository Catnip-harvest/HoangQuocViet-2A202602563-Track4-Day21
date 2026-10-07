"""Ảnh failure case: chỉ vẽ điểm LiDAR thuộc vật thể, xanh = rơi đúng vào 2D box, đỏ = rơi ra ngoài.

    python -m src.make_failure_figs          # cần chạy src.exp_ego_motion trước để chọn frame nuScenes
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.calib_qa import CLASSES, finite_points, points_in_box
from starter.datasets import load_frame
from starter.projection import draw_box2d, perturb_extrinsic, project_velo_to_image, velo_to_cam

FIG = Path("results/figures")


def caption(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def hit_miss_overlay(fr: dict, calib_test, title: str, types=CLASSES) -> tuple[np.ndarray, str]:
    pts = finite_points(fr["points"])
    cam_true = velo_to_cam(pts[:, :3], fr["calib"])
    uv, _, mask = project_velo_to_image(pts, calib_test, fr["image"].shape)
    uv_all = np.full((len(pts), 2), np.nan)
    uv_all[mask] = uv
    out = (fr["image"] * 0.6).astype(np.uint8)
    hits = total = 0
    for obj in fr["labels"]:
        if obj.type not in types:
            continue
        sel = points_in_box(cam_true, obj) & mask
        x1, y1, x2, y2 = obj.bbox
        for u, v in uv_all[sel]:
            inside = x1 <= u <= x2 and y1 <= v <= y2
            hits += inside
            total += 1
            cv2.circle(out, (int(u), int(v)), 2, (0, 255, 0) if inside else (0, 0, 255), -1)
        out = draw_box2d(out, obj.bbox, color=(255, 255, 0), label=obj.type)
    return out, f"{title}  hit_ratio={hits / max(total, 1):.1%} ({hits}/{total})"


def crop_to(img: np.ndarray, boxes: list, pad: int = 60) -> np.ndarray:
    """Cắt theo chiều ngang quanh các box (giữ nguyên chiều cao vì ảnh KITTI chỉ cao 375 pixel)."""
    b = np.array(boxes)
    x1 = int(max(b[:, 0].min() - pad, 0))
    x2 = int(min(b[:, 2].max() + pad, img.shape[1]))
    return img[:, x1:x2]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # fail_01 (Geometry): yaw 2° trên frame nhiều người đi bộ.
    fr = load_frame("data/kitti_mini", "000011")
    peds = [o.bbox for o in fr["labels"] if o.type == "Pedestrian"]
    panels = []
    for yaw in (0.0, 1.0, 2.0):
        img, text = hit_miss_overlay(fr, perturb_extrinsic(fr["calib"], yaw_deg=yaw), f"yaw {yaw} deg", ("Pedestrian",))
        panels.append(caption(crop_to(img, peds), text))
    width = min(p.shape[1] for p in panels)
    cv2.imwrite(str(FIG / "fail_01_yaw_2deg_pedestrian_000011.png"), np.vstack([p[:, :width] for p in panels]))
    print(f"-> {FIG / 'fail_01_yaw_2deg_pedestrian_000011.png'}")

    # fail_02 (Time): nuScenes keyframe bị ảnh hưởng nặng nhất khi bỏ bù ego motion.
    ego = Path("results/ego_motion.csv")
    if ego.exists():
        df = pd.read_csv(ego)
        df = df[df["object_points"] >= 200]
        frame = df.loc[(df["hit_ratio_ego"] - df["hit_ratio_no_ego"]).idxmax(), "frame"]
        fr = load_frame("data/nuscenes_mini_subset", frame, use_ego_motion=True)
        raw = load_frame("data/nuscenes_mini_subset", frame, use_ego_motion=False)
        a = caption(*hit_miss_overlay(fr, fr["calib"], f"{frame} co bu ego motion"))
        b = caption(*hit_miss_overlay(fr, raw["calib"], f"{frame} KHONG bu ego motion"))
        both = np.vstack([a, b])
        both = cv2.resize(both, (both.shape[1] // 2, both.shape[0] // 2), interpolation=cv2.INTER_AREA)
        name = f"fail_02_no_ego_motion_{frame}.png"
        cv2.imwrite(str(FIG / name), both)
        print(f"-> {FIG / name}")


if __name__ == "__main__":
    main()
