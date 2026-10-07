"""Failure case lớp Time: bỏ bù chuyển động (ego motion) giữa thời điểm chụp LiDAR và camera trên nuScenes.

Với mỗi keyframe: lệch thời gian camera - LiDAR, quãng đường xe đi trong khoảng đó, hit_ratio có bù / không bù.
    python -m src.exp_ego_motion
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.calib_qa import hit_ratio, object_hits_for_frame
from starter.datasets import list_frames, load_frame


def main() -> None:
    ap = argparse.ArgumentParser(description="So sánh hit_ratio có / không bù ego motion trên nuScenes")
    ap.add_argument("--data-root", default="data/nuscenes_mini_subset")
    ap.add_argument("--out", default="results/ego_motion.csv")
    args = ap.parse_args()

    rows = []
    for frame in list_frames(args.data_root):
        fr = load_frame(args.data_root, frame, use_ego_motion=True)
        fr_raw = load_frame(args.data_root, frame, use_ego_motion=False)
        # Cùng label (theo pose camera) và cùng calib gốc để chọn điểm thuộc vật thể; chỉ đổi calib dùng để chiếu.
        with_ego = object_hits_for_frame(fr, fr["calib"])
        no_ego = object_hits_for_frame(fr, fr_raw["calib"])
        shift = np.linalg.norm(fr["calib"].Tr_velo_to_cam[:, 3] - fr_raw["calib"].Tr_velo_to_cam[:, 3])
        dt_ms = (fr["timestamp_camera_us"] - fr["timestamp_lidar_us"]) / 1000
        rows.append({"frame": frame, "scene": frame.rsplit("_", 1)[0], "dt_cam_minus_lidar_ms": round(dt_ms, 1),
                     "ego_shift_m": round(float(shift), 3),
                     "ego_speed_mps": round(float(shift / abs(dt_ms) * 1000), 2) if dt_ms else 0.0,
                     "object_points": sum(r["object_points"] for r in with_ego),
                     "hit_ratio_ego": round(hit_ratio(with_ego), 4), "hit_ratio_no_ego": round(hit_ratio(no_ego), 4)})
        print(rows[-1])

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.groupby("scene")[["dt_cam_minus_lidar_ms", "ego_speed_mps", "hit_ratio_ego", "hit_ratio_no_ego"]]
          .mean().round(3).to_string())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
