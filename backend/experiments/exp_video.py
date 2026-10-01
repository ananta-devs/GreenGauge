"""
GreenGauge Experiment: Video Motion & Frame Processing Benchmark
================================================================
File Modality: Video (.gif / animated frame sequence)
Tracks energy consumption and carbon emissions for computer vision video processing & temporal analytics.

Pipeline stages:
1. Synthetic video generation / loading (60 frames, 128x128 resolution, animated motion)
2. Frame decoding into 4D tensor (T, H, W, C)
3. Temporal frame differencing (motion energy detection)
4. Temporal moving-average smoothing (temporal denoising)
5. Spatial Sobel gradient filtering on every frame
6. Keyframe extraction based on peak motion thresholds
"""

import os
import time
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from scipy.signal import convolve2d
from codecarbon import EmissionsTracker

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
VIDEO_PATH = os.path.join(DATA_DIR, "sample_video.gif")


def ensure_sample_video():
    """Generates a small synthetic video clip (60 frames, ~200 KB) if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(VIDEO_PATH):
        return VIDEO_PATH

    print(f"[*] Generating synthetic sample video at {VIDEO_PATH}...")
    num_frames = 60
    width, height = 128, 128
    frames = []

    # Simulate a bouncing ball across a moving gradient background
    ball_x, ball_y = 20.0, 20.0
    vel_x, vel_y = 3.5, 4.2
    radius = 12

    for frame_idx in range(num_frames):
        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)

        # Dynamic background pattern
        shift = int((frame_idx * 4) % 255)
        for y in range(0, height, 8):
            color = ((y * 2 + shift) % 255, 60, (255 - shift) % 255)
            draw.rectangle([0, y, width, y + 8], fill=color)

        # Update bouncing ball physics
        ball_x += vel_x
        ball_y += vel_y
        if ball_x - radius <= 0 or ball_x + radius >= width:
            vel_x = -vel_x
        if ball_y - radius <= 0 or ball_y + radius >= height:
            vel_y = -vel_y

        draw.ellipse(
            [ball_x - radius, ball_y - radius, ball_x + radius, ball_y + radius],
            fill=(255, 255, 0),
            outline=(255, 140, 0),
            width=2
        )

        frames.append(np.array(img))

    imageio.mimsave(VIDEO_PATH, frames, fps=20, loop=0)
    size_kb = os.path.getsize(VIDEO_PATH) / 1024
    print(f"[OK] Generated {VIDEO_PATH} ({size_kb:.1f} KB, {num_frames} frames)")
    return VIDEO_PATH


def process_video(video_path, repetitions=2):
    """Executes a compute-controlled video frame analytics pipeline."""
    # 1. Read frames into 4D numpy array: (T, H, W, C)
    frames = imageio.mimread(video_path)
    video_tensor = np.array(frames, dtype=np.float32)
    t, h, w, c = video_tensor.shape

    results = {}
    for _ in range(repetitions):
        # 2. Grayscale conversion across all frames: (T, H, W)
        gray_frames = (
            0.2989 * video_tensor[:, :, :, 0] +
            0.5870 * video_tensor[:, :, :, 1] +
            0.1140 * video_tensor[:, :, :, 2]
        )

        # 3. Temporal Frame Differencing: |F_t - F_{t-1}|
        diffs = np.abs(np.diff(gray_frames, axis=0))
        motion_energy = np.mean(diffs, axis=(1, 2))

        # 4. Temporal 3-Frame Smoothing across time axis
        smoothed_temporal = np.zeros_like(gray_frames)
        smoothed_temporal[0] = gray_frames[0]
        smoothed_temporal[-1] = gray_frames[-1]
        smoothed_temporal[1:-1] = (gray_frames[:-2] + gray_frames[1:-1] + gray_frames[2:]) / 3.0

        # 5. Spatial Sobel Edge Detection on each frame
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

        edge_frames = []
        for i in range(t):
            gx = convolve2d(smoothed_temporal[i], sobel_x, mode="same")
            gy = convolve2d(smoothed_temporal[i], sobel_y, mode="same")
            edge_frames.append(np.sqrt(gx**2 + gy**2))

        edge_tensor = np.array(edge_frames)

        # 6. Keyframe detection (frames where motion energy is in the top 15%)
        threshold = np.percentile(motion_energy, 85)
        keyframe_indices = np.where(motion_energy >= threshold)[0]

    results["total_frames"] = t
    results["resolution"] = f"{w}x{h}"
    results["mean_motion_delta"] = float(np.mean(motion_energy))
    results["peak_motion_delta"] = float(np.max(motion_energy))
    results["keyframe_count"] = int(len(keyframe_indices))
    results["mean_edge_intensity"] = float(np.mean(edge_tensor))

    return results


def main():
    print("=" * 80)
    print("  GREENGAUGE: VIDEO PROCESSING EXPERIMENT (CodeCarbon)")
    print("=" * 80)

    video_path = ensure_sample_video()
    file_size_kb = os.path.getsize(video_path) / 1024
    print(f"Target Video: {video_path} ({file_size_kb:.2f} KB)")

    tracker = EmissionsTracker(
        project_name="exp_video_processing",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )

    print("\nStarting video benchmark (Temporal Differencing, Sobel filtering, Keyframes)...")
    tracker.start()
    t0 = time.perf_counter()

    metrics = process_video(video_path, repetitions=3)

    duration = time.perf_counter() - t0
    emissions = tracker.stop()
    em_data = tracker.final_emissions_data

    total_frames_processed = metrics["total_frames"] * 3
    fps_throughput = total_frames_processed / duration

    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS: VIDEO WORKLOAD")
    print("=" * 80)
    print(f"  Video Frames        : {metrics['total_frames']} frames @ {metrics['resolution']}")
    print(f"  File Size           : {file_size_kb:.2f} KB")
    print(f"  Execution Time      : {duration:.4f} s")
    print(f"  Processing Speed    : {fps_throughput:.2f} FPS")
    print(f"  Motion Energy Delta : {metrics['mean_motion_delta']:.2f} (Peak: {metrics['peak_motion_delta']:.2f})")
    print(f"  Detected Keyframes  : {metrics['keyframe_count']}")
    print(f"  Energy Consumed     : {em_data.energy_consumed:.8f} kWh")
    print(f"  Carbon Emissions    : {emissions:.10f} kg CO2e")
    print(f"  CPU Average Power   : {em_data.cpu_power or 0.0:.2f} W")
    print(f"  RAM Average Power   : {em_data.ram_power or 0.0:.2f} W")
    print("=" * 80)


if __name__ == "__main__":
    main()
