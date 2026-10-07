"""Vẽ mọi biểu đồ từ các CSV trong results/. Chạy từ gốc repo: python -m src.plot_results

Biểu đồ nào thiếu CSV thì bỏ qua. Ngoài ảnh, script ghi thêm results/metric_compare_auc.csv (bonus B1).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = Path("results")
FIG = RES / "figures"


def read(name: str) -> pd.DataFrame | None:
    path = RES / name
    return pd.read_csv(path, dtype={"frame": str}) if path.exists() else None


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)
    print(f"-> {FIG / name}")


def style(ax, xlabel: str, ylabel: str = "% điểm của vật thể nằm trong 2D box") -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def pooled(objects: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = objects.groupby(by)[["hits", "object_points"]].sum().reset_index()
    g["pct"] = 100 * g["hits"] / g["object_points"]
    return g


def plot_yaw_sweep() -> None:
    df = read("yaw_perturb_sweep.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for frame, g in df.groupby("frame"):
        ax.plot(g["yaw_deg"], 100 * g["hit_ratio"], marker="o", label=f"frame {frame}")
    style(ax, "Lệch yaw (độ)")
    ax.set_title("KITTI: hit_ratio theo lệch yaw")
    save(fig, "yaw_sweep.png")


def plot_breakdown() -> None:
    obj = read("yaw_sweep_detail_objects.csv")
    if obj is None:
        return
    obj["group"] = np.where(obj["type"].isin(["Pedestrian", "Cyclist"]), "Người đi bộ + cyclist", "Xe (Car/Van/Truck)")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in pooled(obj, ["group", "level"]).groupby("group"):
        axes[0].plot(g["level"], g["pct"], marker="o", label=name)
    style(axes[0], "Lệch yaw (độ)")
    axes[0].set_title("Tách theo class (3 frame gộp)")
    for name, g in pooled(obj, ["range_bin", "level"]).groupby("range_bin"):
        axes[1].plot(g["level"], g["pct"], marker="o", label=name)
    style(axes[1], "Lệch yaw (độ)")
    axes[1].set_title("Tách theo khoảng cách vật thể")
    save(fig, "yaw_breakdown_class_range.png")


def plot_param_compare() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    found = False
    for param, ax in [("yaw", 0), ("pitch", 0), ("roll", 0), ("tx", 1), ("ty", 1), ("tz", 1)]:
        df = read("yaw_sweep_detail.csv" if param == "yaw" else f"{param}_sweep.csv")
        if df is None:
            continue
        found = True
        g = df.groupby("level")[["hits", "object_points"]].sum()
        x = g.index * (100 if ax == 1 else 1)
        axes[ax].plot(x, 100 * g["hits"] / g["object_points"], marker="o", label=param)
    if not found:
        plt.close(fig)
        return
    style(axes[0], "Lệch góc (độ)")
    axes[0].set_title("Xoay: yaw / pitch / roll")
    style(axes[1], "Dịch chuyển (cm)")
    axes[1].set_title("Dịch: tx (trước) / ty (trái) / tz (lên)")
    save(fig, "param_compare.png")


def auc(clean: np.ndarray, drifted: np.ndarray) -> float:
    """P(metric của frame bị lệch < metric của frame sạch). 0.5 = đoán mò, 1.0 = tách hoàn toàn."""
    diff = clean[:, None] - drifted[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def plot_metric_compare() -> None:
    df = read("kitti_all_metrics.csv")
    if df is None:
        return
    rows = []
    clean = df[df["level"] == 0]
    for level in sorted(df["level"].unique()):
        if level == 0:
            continue
        cur = df[df["level"] == level]
        for metric in ("hit_ratio", "edge_score"):
            base = clean.set_index("frame")[metric]
            paired = (cur.set_index("frame")[metric] < base).mean()
            rows.append({"yaw_deg": level, "metric": metric, "auc_unpaired": round(auc(base.to_numpy(), cur[metric].to_numpy()), 3),
                         "frames_lower_than_own_baseline": round(float(paired), 3),
                         "mean_clean": round(base.mean(), 4), "mean_drifted": round(cur[metric].mean(), 4),
                         "std_clean": round(base.std(), 4)})
    table = pd.DataFrame(rows)
    table.to_csv(RES / "metric_compare_auc.csv", index=False)
    print(table.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in (("hit_ratio", axes[0]), ("edge_score", axes[1])):
        for frame, g in df.groupby("frame"):
            ax.plot(g["level"], 100 * g[metric], color="tab:blue", alpha=0.25, lw=1)
        m = df.groupby("level")[metric].mean()
        ax.plot(m.index, 100 * m, color="black", marker="o", lw=2, label=f"trung bình 20 frame")
        ax.set_title(metric + (" (cần label)" if metric == "hit_ratio" else " (không cần label)"))
        style(ax, "Lệch yaw (độ)", "%")
    save(fig, "metric_compare_hit_vs_edge.png")


def plot_degradation() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    found = False
    for name, ax, xlabel in (("degrade_dropout.csv", axes[0], "keep_ratio (tỉ lệ điểm giữ lại)"),
                             ("degrade_noise.csv", axes[1], "sigma nhiễu xyz (m)")):
        df = read(name)
        if df is None:
            continue
        found = True
        for level, g in df.groupby("level"):
            s = g.groupby("degrade_level")[["hits", "object_points"]].sum()
            ax.plot(s.index, 100 * s["hits"] / s["object_points"], marker="o", label=f"yaw {level}°")
        style(ax, xlabel)
        ax.set_title(name.replace(".csv", ""))
    if found:
        save(fig, "degradation_stress.png")
    else:
        plt.close(fig)


def plot_datasets() -> None:
    kitti, nusc = read("kitti_all_metrics.csv"), read("nusc_yaw_sweep.csv")
    if kitti is None or nusc is None:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, df in (("KITTI (20 frame)", kitti), ("nuScenes (80 keyframe)", nusc)):
        g = df.groupby("level")[["hits", "object_points"]].sum()
        ax.plot(g.index, 100 * g["hits"] / g["object_points"], marker="o", label=name)
    nusc["scene"] = nusc["frame"].str.rsplit("_", n=1).str[0]
    for scene, g in nusc.groupby("scene"):
        s = g.groupby("level")[["hits", "object_points"]].sum()
        ax.plot(s.index, 100 * s["hits"] / s["object_points"], ls="--", marker=".", label=f"nuScenes {scene}")
    style(ax, "Lệch yaw (độ)")
    ax.set_title("KITTI so với nuScenes")
    save(fig, "kitti_vs_nuscenes.png")


def plot_inside_image_metric() -> None:
    df = read("yaw_perturb_sweep.csv")
    if df is None:
        return
    g = df[df["frame"] == "000011"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g["yaw_deg"], 100 * g["inside_image"] / g["inside_image"].iloc[0], marker="s",
            label="số điểm trong ảnh (so với 0°)")
    ax.plot(g["yaw_deg"], 100 * g["hit_ratio"], marker="o", label="hit_ratio")
    style(ax, "Lệch yaw (độ)", "%")
    ax.set_title("Frame 000011: metric 'điểm trong ảnh' không thấy drift")
    save(fig, "fail_03_metric_inside_image_blind.png")


def main() -> None:
    plot_yaw_sweep()
    plot_breakdown()
    plot_param_compare()
    plot_metric_compare()
    plot_degradation()
    plot_datasets()
    plot_inside_image_metric()


if __name__ == "__main__":
    main()
