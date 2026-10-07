"""Bonus B3: đo latency của một lần kiểm tra calibration (hit_ratio và edge_score) trên 1 frame.

Bỏ lần chạy đầu (warm-up), chạy --runs lần, ghi mỗi lần một dòng CSV, in p50/p95.
    python -m src.latency --data-root data/kitti_mini --frame 000011
"""
from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.calib_qa import edge_score, finite_points, image_edge_distance, object_hits_for_frame
from starter.datasets import load_frame
from starter.projection import perturb_extrinsic


def main() -> None:
    ap = argparse.ArgumentParser(description="Đo latency p50/p95 của hit_ratio và edge_score")
    ap.add_argument("--data-root", default="data/kitti_mini")
    ap.add_argument("--frame", default="000011")
    ap.add_argument("--runs", type=int, default=21, help="tổng số lần chạy, kể cả lần warm-up bị bỏ")
    ap.add_argument("--out", default="results/latency.csv")
    args = ap.parse_args()

    fr = load_frame(args.data_root, args.frame)
    calib = perturb_extrinsic(fr["calib"], yaw_deg=1.0)
    stages = {
        "hit_ratio": lambda: object_hits_for_frame(fr, calib),
        "edge_score": lambda: edge_score(fr, calib, finite_points(fr["points"]), image_edge_distance(fr["image"])),
    }
    rows = []
    for stage, fn in stages.items():
        for i in range(args.runs):
            t0 = time.perf_counter()
            fn()
            ms = (time.perf_counter() - t0) * 1000
            rows.append({"stage": stage, "run": i, "warmup": i == 0, "ms": round(ms, 3)})

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"CPU: {platform.processor()} ({os.cpu_count()} logical cores), Python {platform.python_version()}")
    for stage, g in df[~df["warmup"]].groupby("stage"):
        ms = g["ms"].to_numpy()
        print(f"{stage}: n={len(ms)} p50={np.percentile(ms, 50):.1f} ms p95={np.percentile(ms, 95):.1f} ms")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
