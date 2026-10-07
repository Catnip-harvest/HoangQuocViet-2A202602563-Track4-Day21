"""Bonus B6: tìm lỗi cài sẵn trong data/synthetic bằng 3 phép kiểm tra, in bảng lỗi và ghi CSV.

    python -m src.find_synthetic_bugs

1. Điểm NaN/Inf trong point cloud (lớp I/O).
2. Timestamp: khoảng cách giữa các frame, đối chiếu với chuyển động của người đi bộ trong label (lớp Time).
3. Sector bị thưa: chia 360° thành ô 1°, so mỗi frame với trung vị các frame khác. Quy tắc gốc
   empty_azimuth_bins (ô 10° rỗng hoàn toàn) bỏ sót lỗi này vì sector chỉ bị thưa, không rỗng.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from starter.datasets import list_frames, load_frame


def azimuth_hist(points: np.ndarray) -> np.ndarray:
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    return np.histogram(az, bins=360, range=(-180, 180))[0]


def main() -> None:
    ap = argparse.ArgumentParser(description="Tìm lỗi cài sẵn trong một thư mục KITTI-format (mặc định data/synthetic)")
    ap.add_argument("--data-root", default="data/synthetic")
    ap.add_argument("--sector-drop", type=float, default=0.5,
                    help="ô 1° bị coi là thưa nếu có ít hơn tỉ lệ này so với trung vị các frame")
    ap.add_argument("--min-run-deg", type=int, default=5, help="số ô thưa liên tiếp tối thiểu để báo lỗi")
    ap.add_argument("--out", default="results/synthetic_bugs.csv")
    args = ap.parse_args()

    frames = list_frames(args.data_root)
    data = {f: load_frame(args.data_root, f) for f in frames}
    bugs = []

    for f, fr in data.items():
        bad = int((~np.isfinite(fr["points"]).all(axis=1)).sum())
        if bad:
            bugs.append({"bug": "Điểm NaN/Inf trong point cloud", "frame": f,
                         "evidence": f"{bad}/{len(fr['points'])} điểm ({bad / len(fr['points']):.2%})", "layer": "I/O"})

    ts_path = Path(args.data_root) / "training" / "timestamps.txt"
    if ts_path.exists():
        ts = np.loadtxt(ts_path)
        dt = np.diff(ts)
        ped_z = [next(o.location[2] for o in data[f]["labels"] if o.type == "Pedestrian") for f in frames]
        step = np.diff(ped_z)
        nominal = np.median(dt)
        for i in np.flatnonzero(np.abs(dt - nominal) > 0.5 * nominal):
            bugs.append({"bug": "Timestamp sai: khoảng cách giữa 2 frame không đều", "frame": frames[i + 1],
                         "evidence": f"dt {frames[i]}->{frames[i + 1]} = {dt[i]:.3f} s (các cặp khác {nominal:.3f} s), "
                                     f"nhưng người đi bộ vẫn đi {step[i]:.2f} m như mọi frame khác "
                                     f"({', '.join(f'{s:.2f}' for s in step)} m) -> frame không bị rơi, timestamp ghi sai",
                         "layer": "Time"})

    hists = {f: azimuth_hist(fr["points"][np.isfinite(fr["points"]).all(axis=1)]) for f, fr in data.items()}
    for f in frames:
        ref = np.median(np.stack([hists[g] for g in frames if g != f]), axis=0)
        thin = hists[f] < args.sector_drop * np.maximum(ref, 1)
        runs, start = [], None
        for i, t in enumerate(np.r_[thin, False]):
            if t and start is None:
                start = i
            elif not t and start is not None:
                if i - start >= args.min_run_deg:
                    runs.append((start, i))
                start = None
        for a, b in runs:
            got, want = int(hists[f][a:b].sum()), int(ref[a:b].sum())
            bugs.append({"bug": "Sector LiDAR bị thưa (che một phần cảm biến)", "frame": f,
                         "evidence": f"azimuth {a - 180}°..{b - 180}° có {got} điểm, trung vị các frame khác {want} "
                                     f"(mất {1 - got / want:.0%}); empty_azimuth_bins vẫn = 0",
                         "layer": "I/O (sensor)"})

    df = pd.DataFrame(bugs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"-> {out} ({len(df)} dòng)")


if __name__ == "__main__":
    main()
