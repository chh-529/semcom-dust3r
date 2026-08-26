import time

program_start = time.perf_counter()

import csv
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import trimesh

from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.inference import inference
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode


timings = {
    "00_import_modules": time.perf_counter() - program_start
}


@contextmanager
def timer(name, use_cuda=False):
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    print(f"\n[{name}] 開始")

    yield

    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    timings[name] = elapsed

    print(f"[{name}] 完成：{elapsed:.4f} 秒")


def to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def convert_colors(colors):
    if colors.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    if colors.max() <= 1.0:
        colors = colors * 255.0

    return np.clip(colors, 0, 255).astype(np.uint8)


device = "cuda"

checkpoint = Path(
    "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
)

image_paths = [
    "croco/assets/Chateau1.png",
    "croco/assets/Chateau2.png",
]

run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("outputs/chateau_timed_demo") / run_name
output_dir.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("找不到 CUDA GPU")

if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

for image_path in image_paths:
    if not Path(image_path).is_file():
        raise FileNotFoundError(image_path)

torch.cuda.reset_peak_memory_stats()

print("GPU:", torch.cuda.get_device_name(0))
print("Output directory:", output_dir)


# 1. 模型與 checkpoint
with timer("01_model_loading", use_cuda=True):
    model = AsymmetricCroCo3DStereo.from_pretrained(
        str(checkpoint)
    )
    model = model.to(device).eval()


# 2. 讀取與縮放圖片
with timer("02_image_loading"):
    images = load_images(image_paths, size=512)


# 3. 建立影像配對
with timer("03_pair_creation"):
    pairs = make_pairs(
        images,
        scene_graph="complete",
        prefilter=None,
        symmetrize=True,
    )


# 4. DUSt3R GPU 推論
with timer("04_dust3r_inference", use_cuda=True):
    with torch.no_grad():
        inference_output = inference(
            pairs,
            model,
            device,
            batch_size=1,
            verbose=True,
        )


# 5. 建立 Global Aligner
with timer("05_scene_creation", use_cuda=True):
    scene = global_aligner(
        inference_output,
        device=device,
        mode=GlobalAlignerMode.PointCloudOptimizer,
    )


# 6. Global alignment
with timer("06_global_alignment", use_cuda=True):
    alignment_loss = scene.compute_global_alignment(
        init="mst",
        niter=300,
        schedule="cosine",
        lr=0.01,
    )


# 7. 將深度、點雲和圖片取回 CPU
with timer("07_result_readback", use_cuda=True):
    points_list = scene.get_pts3d()
    masks_list = scene.get_masks()
    depth_list = scene.get_depthmaps()
    images_rgb = scene.imgs

    view_data = []

    for points, mask, depth, image in zip(
        points_list,
        masks_list,
        depth_list,
        images_rgb,
    ):
        view_data.append(
            (
                to_numpy(points),
                to_numpy(mask).astype(bool),
                to_numpy(depth),
                to_numpy(image),
            )
        )


# 8. 建立完整 baseline 點雲
with timer("08_baseline_generation"):
    baseline_points = []
    baseline_colors = []

    for points, mask, depth, image in view_data:
        valid = (
            mask
            & np.isfinite(points).all(axis=-1)
            & np.isfinite(depth)
            & (depth > 0)
        )

        baseline_points.append(points[valid])
        baseline_colors.append(
            convert_colors(image[valid])
        )

    baseline_points = np.concatenate(
        baseline_points, axis=0
    )
    baseline_colors = np.concatenate(
        baseline_colors, axis=0
    )


# 9. 深度感知近密遠疏取樣
with timer("09_depth_aware_sampling"):
    rng = np.random.default_rng(seed=42)

    adaptive_points = []
    adaptive_colors = []

    depth_statistics = []

    for view_index, (points, mask, depth, image) in enumerate(
        view_data
    ):
        valid = (
            mask
            & np.isfinite(points).all(axis=-1)
            & np.isfinite(depth)
            & (depth > 0)
        )

        valid_depth = depth[valid]

        near_threshold = np.percentile(valid_depth, 30)
        far_threshold = np.percentile(valid_depth, 70)

        keep_probability = np.zeros_like(
            depth, dtype=np.float32
        )

        near_region = depth <= near_threshold

        middle_region = (
            (depth > near_threshold)
            & (depth <= far_threshold)
        )

        far_region = depth > far_threshold

        # 近景 100%、中景 50%、遠景 15%
        keep_probability[near_region] = 1.00
        keep_probability[middle_region] = 0.50
        keep_probability[far_region] = 0.15

        random_values = rng.random(depth.shape)
        sampling_mask = random_values < keep_probability
        final_mask = valid & sampling_mask

        adaptive_points.append(points[final_mask])
        adaptive_colors.append(
            convert_colors(image[final_mask])
        )

        depth_statistics.append(
            (
                view_index,
                near_threshold,
                far_threshold,
                int(valid.sum()),
                int(final_mask.sum()),
            )
        )

    adaptive_points = np.concatenate(
        adaptive_points, axis=0
    )
    adaptive_colors = np.concatenate(
        adaptive_colors, axis=0
    )


# 10. 輸出 baseline
with timer("10_baseline_export"):
    baseline_cloud = trimesh.points.PointCloud(
        vertices=baseline_points,
        colors=baseline_colors,
    )

    baseline_cloud.export(
        output_dir / "chateau_baseline.ply"
    )
    baseline_cloud.export(
        output_dir / "chateau_baseline.glb"
    )


# 11. 輸出深度感知結果
with timer("11_adaptive_export"):
    adaptive_cloud = trimesh.points.PointCloud(
        vertices=adaptive_points,
        colors=adaptive_colors,
    )

    adaptive_cloud.export(
        output_dir / "chateau_depth_adaptive.ply"
    )
    adaptive_cloud.export(
        output_dir / "chateau_depth_adaptive.glb"
    )


total_time = time.perf_counter() - program_start
timings["12_total_runtime"] = total_time

baseline_count = len(baseline_points)
adaptive_count = len(adaptive_points)
retention_ratio = adaptive_count / baseline_count
peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)


# 儲存各階段時間
timing_path = output_dir / "timings.csv"

with timing_path.open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["stage", "seconds", "percentage"])

    for stage, seconds in timings.items():
        percentage = seconds / total_time * 100
        writer.writerow([
            stage,
            f"{seconds:.6f}",
            f"{percentage:.2f}",
        ])


print("\n" + "=" * 65)
print("Demo 執行完成")
print("=" * 65)

for stage, seconds in timings.items():
    percentage = seconds / total_time * 100
    print(
        f"{stage:28s}: "
        f"{seconds:9.4f} 秒 "
        f"({percentage:6.2f}%)"
    )

print("-" * 65)
print("Alignment loss:", float(alignment_loss))
print("Baseline points:", baseline_count)
print("Adaptive points:", adaptive_count)
print(f"Point retention ratio: {retention_ratio:.2%}")
print(f"Point reduction ratio: {1-retention_ratio:.2%}")
print(f"Peak GPU memory: {peak_memory_gb:.3f} GB")

for (
    view_index,
    near_threshold,
    far_threshold,
    original_count,
    sampled_count,
) in depth_statistics:
    print(
        f"View {view_index}: "
        f"near threshold={near_threshold:.4f}, "
        f"far threshold={far_threshold:.4f}, "
        f"points={original_count} -> {sampled_count}"
    )

print("Output directory:", output_dir)
print("Timing CSV:", timing_path)
