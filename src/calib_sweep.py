"""Quét độ lệch calibration (và tuỳ chọn suy giảm point cloud), đo hit_ratio và edge_score.

Chạy từ gốc repo, ví dụ:
    python -m src.calib_sweep                                         # yaw 0-3 độ trên 3 frame KITTI
    python -m src.calib_sweep --param pitch --out results/pitch_sweep.csv
    python -m src.calib_sweep --param tx --levels 0 0.02 0.05 0.1 --out results/tx_sweep.csv
    python -m src.calib_sweep --frames all --levels 0 0.5 1 2 --edge --out results/kitti_all_metrics.csv
    python -m src.calib_sweep --levels 0 1 --degrade random_dropout --degrade-levels 1 0.7 0.5 0.3
    python -m src.calib_sweep --data-root data/nuscenes_mini_subset --frames scene-0103_010 scene-1094_010
    python -m src.calib_sweep --help

Mỗi dòng của CSV tổng là một cấu hình (frame x mức lệch x mức suy giảm). File *_objects.csv đi kèm
có một dòng cho mỗi vật thể, dùng để tách theo class và khoảng cách.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.calib_qa import edge_score, finite_points, hit_ratio, image_edge_distance, object_hits_for_frame
from starter import perturb
from starter.datasets import list_frames, load_frame
from starter.projection import perturb_extrinsic, project_velo_to_image

ANGLE_PARAMS = ("roll", "pitch", "yaw")
SHIFT_PARAMS = ("tx", "ty", "tz")


def perturbed_calib(calib, param: str, level: float):
    if param in ANGLE_PARAMS:
        return perturb_extrinsic(calib, **{f"{param}_deg": level})
    shift = [0.0, 0.0, 0.0]
    shift[SHIFT_PARAMS.index(param)] = level
    return perturb_extrinsic(calib, t_xyz_m=tuple(shift))


def degrade(points, kind: str, level: float | None, seed: int):
    if kind == "none" or level is None:
        return points
    if kind == "random_dropout":
        return perturb.random_dropout(points, keep_ratio=level, seed=seed)
    if kind == "gaussian_noise":
        return perturb.gaussian_noise(points, sigma_xyz_m=level, seed=seed)
    if kind == "beam_dropout":
        return perturb.beam_dropout(points, keep_every=int(level))
    raise ValueError(kind)


def class_ratio(rows: list[dict], types: tuple[str, ...]) -> float:
    return hit_ratio([r for r in rows if r["type"] in types])


def main() -> None:
    ap = argparse.ArgumentParser(description="Quét lệch calibration LiDAR-camera, đo hit_ratio (cần label) "
                                             "và edge_score (không cần label).")
    ap.add_argument("--data-root", default="data/kitti_mini",
                    help="data/synthetic, data/kitti_mini hoặc data/nuscenes_mini_subset")
    ap.add_argument("--frames", nargs="+", default=["000008", "000011", "000049"],
                    help="danh sách frame id, hoặc 'all' để chạy mọi frame của dataset")
    ap.add_argument("--param", choices=ANGLE_PARAMS + SHIFT_PARAMS, default="yaw",
                    help="thông số calibration bị làm lệch: góc (độ) hoặc dịch chuyển (mét)")
    ap.add_argument("--levels", nargs="+", type=float, default=[0, 0.5, 1, 2, 3],
                    help="các mức lệch; luôn nên có 0 làm mốc")
    ap.add_argument("--degrade", choices=("none", "random_dropout", "gaussian_noise", "beam_dropout"),
                    default="none", help="làm suy giảm point cloud trước khi đo (bonus B2)")
    ap.add_argument("--degrade-levels", nargs="+", type=float, default=None,
                    help="keep_ratio cho random_dropout, sigma (m) cho gaussian_noise, keep_every cho beam_dropout")
    ap.add_argument("--seed", type=int, default=0, help="seed cho mọi phép ngẫu nhiên")
    ap.add_argument("--edge", action="store_true", help="tính thêm edge_score (chậm hơn một chút)")
    ap.add_argument("--out", default="results/yaw_sweep_detail.csv", help="CSV tổng, một dòng mỗi cấu hình")
    args = ap.parse_args()

    frames = list_frames(args.data_root) if args.frames == ["all"] else args.frames
    degrade_levels = args.degrade_levels if args.degrade != "none" else [None]
    dataset = Path(args.data_root).name

    rows, object_rows = [], []
    for frame in frames:
        fr = load_frame(args.data_root, frame)
        edge_dist = image_edge_distance(fr["image"]) if args.edge else None
        for deg_level in degrade_levels:
            pts = finite_points(degrade(fr["points"], args.degrade, deg_level, args.seed))
            for level in args.levels:
                calib = perturbed_calib(fr["calib"], args.param, level)
                obj = object_hits_for_frame(fr, calib, pts)
                _, _, mask = project_velo_to_image(pts, calib, fr["image"].shape)
                config = {"dataset": dataset, "frame": frame, "param": args.param, "level": level,
                          "degrade": args.degrade, "degrade_level": deg_level}
                row = {**config, "n_points": len(pts), "inside_image": int(mask.sum()),
                       "object_points": sum(r["object_points"] for r in obj), "hits": sum(r["hits"] for r in obj),
                       "hit_ratio": round(hit_ratio(obj), 4),
                       "hit_ratio_ped_cyc": round(class_ratio(obj, ("Pedestrian", "Cyclist")), 4),
                       "hit_ratio_vehicle": round(class_ratio(obj, ("Car", "Van", "Truck")), 4)}
                if args.edge:
                    row["edge_score"] = round(edge_score(fr, calib, pts, edge_dist), 4)
                rows.append(row)
                object_rows += [{**config, **r} for r in obj]
                print({k: row[k] for k in ("frame", "level", "degrade_level", "object_points", "hit_ratio")}
                      | ({"edge_score": row["edge_score"]} if args.edge else {}))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(object_rows).to_csv(out.with_name(out.stem + "_objects.csv"), index=False)
    print(f"-> {out} ({len(rows)} dòng), {out.with_name(out.stem + '_objects.csv')} ({len(object_rows)} vật thể)")


if __name__ == "__main__":
    main()
