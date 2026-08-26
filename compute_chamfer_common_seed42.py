#!/usr/bin/env python3

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree


# ============================================================
# 路徑與實驗設定
# ============================================================

DUST3R_ROOT = Path("/tmp2/b12504107/dust3r-semcom")

REFERENCE_PATH = (
    DUST3R_ROOT
    / "outputs/chateau_baseline/chateau_baseline.ply"
)

OUTPUT_DIR = Path(
    "/tmp2/b12504107/semcom-dust3r-bridge/"
    "figures/common_seed42"
)

RUN_TIMESTAMPS = {
    "awgn": {
        -5: "20260826_110846",
        0: "20260826_110859",
        5: "20260826_110912",
        10: "20260826_110923",
        15: "20260826_110936",
        20: "20260826_110950",
    },
    "rayleigh": {
        -5: "20260826_111003",
        0: "20260826_111014",
        5: "20260826_111025",
        10: "20260826_111039",
        15: "20260826_111052",
        20: "20260826_111106",
    },
}

SNRS = [-5, 0, 5, 10, 15, 20]

# 每個點雲最後用相同數量的點計算
MAX_POINTS = 50_000

# 移除距離中心最遠的 0.5% 點
OUTLIER_QUANTILE = 0.995

# 使用 95% 半徑作為尺度正規化基準
SCALE_QUANTILE = 0.95

# ICP 使用的點數
ICP_POINTS = 20_000
ICP_MAX_ITERATIONS = 50

# ICP 每輪保留距離較近的 90% correspondence
ICP_KEEP_RATIO = 0.90

RANDOM_SEED = 42


def build_point_cloud_path(channel: str, snr: int) -> Path:
    """根據本次固定時間戳記建立 standard PLY 路徑。"""
    experiment_name = f"omdma_{channel}_snr{snr}"
    timestamp = RUN_TIMESTAMPS[channel][snr]

    return (
        DUST3R_ROOT
        / "outputs/semcom_dust3r"
        / experiment_name
        / timestamp
        / f"{experiment_name}_standard.ply"
    )


def load_ply_points(path: Path) -> np.ndarray:
    """讀取 PLY，回傳 N×3 點座標。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到點雲：{path}")

    loaded = trimesh.load(path, process=False)

    if isinstance(loaded, trimesh.Scene):
        point_arrays = []

        for geometry in loaded.geometry.values():
            if hasattr(geometry, "vertices"):
                vertices = np.asarray(
                    geometry.vertices,
                    dtype=np.float64,
                )

                if len(vertices) > 0:
                    point_arrays.append(vertices)

        if not point_arrays:
            raise ValueError(f"Scene 中沒有頂點：{path}")

        points = np.concatenate(point_arrays, axis=0)

    elif hasattr(loaded, "vertices"):
        points = np.asarray(
            loaded.vertices,
            dtype=np.float64,
        )

    else:
        raise TypeError(
            f"不支援的 PLY 類型：{type(loaded)}，檔案：{path}"
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"點雲維度錯誤：{points.shape}，檔案：{path}"
        )

    # 移除 NaN 與 Inf
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if len(points) == 0:
        raise ValueError(f"移除 NaN/Inf 後沒有有效點：{path}")

    return points


def preprocess_points(
    points: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, dict]:
    """
    1. 移除極端離群點
    2. 固定數量隨機下採樣
    3. 中心化
    4. 尺度正規化
    """
    rng = np.random.default_rng(random_seed)

    original_count = len(points)

    # 使用座標中位數估計中心，降低離群點影響
    robust_center = np.median(points, axis=0)
    radius = np.linalg.norm(
        points - robust_center,
        axis=1,
    )

    radius_threshold = np.quantile(
        radius,
        OUTLIER_QUANTILE,
    )

    points = points[radius <= radius_threshold]
    filtered_count = len(points)

    if len(points) > MAX_POINTS:
        indices = rng.choice(
            len(points),
            size=MAX_POINTS,
            replace=False,
        )
        points = points[indices]

    # 中心正規化
    center = np.mean(points, axis=0)
    points = points - center

    # 尺度正規化
    normalized_radius = np.linalg.norm(points, axis=1)
    scale = np.quantile(
        normalized_radius,
        SCALE_QUANTILE,
    )

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"無效的正規化尺度：{scale}")

    points = points / scale

    information = {
        "original_count": original_count,
        "finite_and_filtered_count": filtered_count,
        "sampled_count": len(points),
        "normalization_scale": float(scale),
    }

    return points, information


def estimate_rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    以 SVD 找出 source 到 target 的最佳 rigid transform。

    target ≈ R @ source + t
    """
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)

    source_centered = source - source_center
    target_centered = target - target_center

    covariance = source_centered.T @ target_centered

    u, _, vt = np.linalg.svd(covariance)

    rotation = vt.T @ u.T

    # 避免產生 reflection
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    translation = (
        target_center
        - rotation @ source_center
    )

    return rotation, translation


def apply_transform(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """對 row-vector 格式的點套用 rigid transform。"""
    return points @ rotation.T + translation


def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, float, int]:
    """使用 trimmed point-to-point ICP 將 source 對齊 target。"""
    rng = np.random.default_rng(random_seed)

    source_sample_size = min(ICP_POINTS, len(source))
    target_sample_size = min(ICP_POINTS, len(target))

    source_indices = rng.choice(
        len(source),
        size=source_sample_size,
        replace=False,
    )

    target_indices = rng.choice(
        len(target),
        size=target_sample_size,
        replace=False,
    )

    aligned_source = source.copy()
    source_for_icp = source[source_indices].copy()
    target_for_icp = target[target_indices]

    target_tree = cKDTree(target_for_icp)

    previous_error = np.inf
    completed_iterations = 0

    for iteration in range(ICP_MAX_ITERATIONS):
        distances, nearest_indices = target_tree.query(
            source_for_icp,
            k=1,
            workers=-1,
        )

        cutoff = np.quantile(
            distances,
            ICP_KEEP_RATIO,
        )

        keep_mask = distances <= cutoff

        matched_source = source_for_icp[keep_mask]
        matched_target = target_for_icp[
            nearest_indices[keep_mask]
        ]

        if len(matched_source) < 3:
            raise RuntimeError(
                "ICP correspondence 少於 3 點，無法估計 rigid transform"
            )

        rotation, translation = estimate_rigid_transform(
            matched_source,
            matched_target,
        )

        source_for_icp = apply_transform(
            source_for_icp,
            rotation,
            translation,
        )

        aligned_source = apply_transform(
            aligned_source,
            rotation,
            translation,
        )

        new_distances, _ = target_tree.query(
            source_for_icp,
            k=1,
            workers=-1,
        )

        current_error = float(
            np.sqrt(np.mean(new_distances**2))
        )

        completed_iterations = iteration + 1

        if abs(previous_error - current_error) < 1e-7:
            break

        previous_error = current_error

    return (
        aligned_source,
        current_error,
        completed_iterations,
    )


def symmetric_squared_chamfer(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float]:
    """
    回傳：
    1. symmetric squared Chamfer
    2. prediction → reference
    3. reference → prediction
    """
    reference_tree = cKDTree(reference)
    prediction_tree = cKDTree(prediction)

    prediction_to_reference, _ = reference_tree.query(
        prediction,
        k=1,
        workers=-1,
    )

    reference_to_prediction, _ = prediction_tree.query(
        reference,
        k=1,
        workers=-1,
    )

    forward = float(
        np.mean(prediction_to_reference**2)
    )

    backward = float(
        np.mean(reference_to_prediction**2)
    )

    chamfer = 0.5 * (forward + backward)

    return chamfer, forward, backward


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("載入 clean reference")
    print(REFERENCE_PATH)

    reference_raw = load_ply_points(REFERENCE_PATH)

    reference, reference_info = preprocess_points(
        reference_raw,
        RANDOM_SEED,
    )

    print(
        "Reference points:",
        reference_info["original_count"],
        "->",
        reference_info["finite_and_filtered_count"],
        "->",
        reference_info["sampled_count"],
    )

    rows = []

    for channel_index, channel in enumerate(
        ["awgn", "rayleigh"]
    ):
        for snr_index, snr in enumerate(SNRS):
            point_cloud_path = build_point_cloud_path(
                channel,
                snr,
            )

            print("=" * 70)
            print(f"Channel={channel}, SNR={snr} dB")
            print(point_cloud_path)

            prediction_raw = load_ply_points(
                point_cloud_path
            )

            experiment_seed = (
                RANDOM_SEED
                + 1000 * (channel_index + 1)
                + snr_index
            )

            prediction, prediction_info = preprocess_points(
                prediction_raw,
                experiment_seed,
            )

            aligned_prediction, icp_rmse, icp_iterations = (
                icp_align(
                    prediction,
                    reference,
                    experiment_seed,
                )
            )

            chamfer, forward, backward = (
                symmetric_squared_chamfer(
                    reference,
                    aligned_prediction,
                )
            )

            print(f"ICP iterations : {icp_iterations}")
            print(f"ICP RMSE       : {icp_rmse:.8e}")
            print(f"Chamfer        : {chamfer:.8e}")
            print(f"Pred -> Ref    : {forward:.8e}")
            print(f"Ref -> Pred    : {backward:.8e}")

            rows.append(
                {
                    "channel": channel,
                    "snr_db": snr,
                    "chamfer_distance": chamfer,
                    "prediction_to_reference": forward,
                    "reference_to_prediction": backward,
                    "icp_rmse": icp_rmse,
                    "icp_iterations": icp_iterations,
                    "reference_original_points":
                        reference_info["original_count"],
                    "reference_sampled_points":
                        reference_info["sampled_count"],
                    "prediction_original_points":
                        prediction_info["original_count"],
                    "prediction_filtered_points":
                        prediction_info[
                            "finite_and_filtered_count"
                        ],
                    "prediction_sampled_points":
                        prediction_info["sampled_count"],
                    "reference_normalization_scale":
                        reference_info["normalization_scale"],
                    "prediction_normalization_scale":
                        prediction_info["normalization_scale"],
                    "point_cloud_path": str(point_cloud_path),
                }
            )

    csv_path = OUTPUT_DIR / "chamfer_distance.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    plt.figure(figsize=(8, 5.5))

    styles = {
        "awgn": {
            "label": "AWGN",
            "marker": "o",
            "color": "tab:blue",
        },
        "rayleigh": {
            "label": "Rayleigh",
            "marker": "s",
            "color": "tab:orange",
        },
    }

    for channel in ["awgn", "rayleigh"]:
        channel_rows = [
            row for row in rows
            if row["channel"] == channel
        ]

        channel_rows.sort(
            key=lambda row: row["snr_db"]
        )

        x_values = [
            row["snr_db"]
            for row in channel_rows
        ]

        y_values = [
            row["chamfer_distance"]
            for row in channel_rows
        ]

        plt.plot(
            x_values,
            y_values,
            linewidth=2,
            markersize=7,
            marker=styles[channel]["marker"],
            color=styles[channel]["color"],
            label=styles[channel]["label"],
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel(
        "Normalized symmetric squared Chamfer distance"
    )
    plt.title(
        "DUSt3R 3D Reconstruction Quality "
        "(Common Seed 42)"
    )
    plt.xticks(SNRS)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    figure_path = (
        OUTPUT_DIR
        / "chamfer_distance_comparison.png"
    )

    plt.savefig(
        figure_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("=" * 70)
    print("Chamfer distance 計算完成")
    print(f"CSV：{csv_path}")
    print(f"Figure：{figure_path}")


if __name__ == "__main__":
    main()