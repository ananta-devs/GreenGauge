"""
GreenGauge Experiment: Image Processing Benchmark
=================================================
File Modality: Image (.png)
Tracks energy consumption and carbon emissions for computer vision & image processing workloads.

Pipeline stages:
1. Synthetic image generation / loading (512x512 RGB)
2. Grayscale conversion & color space manipulation
3. 2D Gaussian convolution filtering
4. Sobel edge detection (gradient magnitude computation)
5. Histogram equalization & contrast enhancement
6. Multi-scale resizing & spatial rotation
"""

import os
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.signal import convolve2d
from codecarbon import EmissionsTracker

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
IMAGE_PATH = os.path.join(DATA_DIR, "sample_image.png")


def ensure_sample_image():
    """Generates a small synthetic test image (512x512, ~80 KB) if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(IMAGE_PATH):
        return IMAGE_PATH

    print(f"[*] Generating synthetic sample image at {IMAGE_PATH}...")
    width, height = 512, 512
    img = Image.new("RGB", (width, height), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)

    # Draw color gradients and geometric shapes
    for y in range(height):
        r = int(255 * (y / height))
        g = int(200 * (1 - y / height))
        b = 180
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Add geometric shapes for edge detection
    draw.rectangle([50, 50, 200, 200], fill=(30, 144, 255), outline=(0, 0, 128), width=3)
    draw.ellipse([250, 100, 450, 300], fill=(255, 99, 71), outline=(178, 34, 34), width=4)
    draw.polygon([(100, 450), (250, 280), (400, 450)], fill=(50, 205, 50), outline=(0, 100, 0), width=3)

    img.save(IMAGE_PATH, format="PNG", optimize=True)
    size_kb = os.path.getsize(IMAGE_PATH) / 1024
    print(f"[OK] Generated {IMAGE_PATH} ({size_kb:.1f} KB)")
    return IMAGE_PATH


def process_image(image_path, iterations=5):
    """Executes a compute-controlled image processing pipeline."""
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img, dtype=np.float32)

    # 1. Grayscale conversion (Luminance formula)
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]

    # 2. 2D Gaussian Kernel Convolution
    gaussian_kernel = np.array([
        [1,  4,  7,  4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1,  4,  7,  4, 1]
    ], dtype=np.float32) / 273.0

    blurred = gray
    for _ in range(iterations):
        blurred = convolve2d(blurred, gaussian_kernel, mode="same", boundary="symm")

    # 3. Sobel Edge Detection (Horizontal & Vertical Gradients)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    gx = convolve2d(blurred, sobel_x, mode="same")
    gy = convolve2d(blurred, sobel_y, mode="same")
    edges = np.sqrt(gx**2 + gy**2)

    # 4. Histogram Equalization
    hist, bins = np.histogram(gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    equalized = np.ma.filled(cdf_m, 0).astype("uint8")[gray.astype("uint8")]

    # 5. Multi-scale transformations & rotations
    pil_gray = Image.fromarray(np.clip(edges, 0, 255).astype(np.uint8))
    transformed = pil_gray.rotate(45).resize((256, 256)).filter(ImageFilter.SHARPEN)

    return {
        "original_shape": arr.shape,
        "processed_shape": np.array(transformed).shape,
        "edge_mean_magnitude": float(np.mean(edges)),
        "equalized_mean": float(np.mean(equalized))
    }


def main():
    print("=" * 80)
    print("  GREENGAUGE: IMAGE PROCESSING EXPERIMENT (CodeCarbon)")
    print("=" * 80)

    image_path = ensure_sample_image()
    file_size_kb = os.path.getsize(image_path) / 1024
    print(f"Target Image: {image_path} ({file_size_kb:.2f} KB)")

    tracker = EmissionsTracker(
        project_name="exp_image_processing",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )

    print("\nStarting image benchmark (Gaussian blur, Sobel convolution, Histogram EQ)...")
    tracker.start()
    t0 = time.perf_counter()

    metrics = process_image(image_path, iterations=8)

    duration = time.perf_counter() - t0
    emissions = tracker.stop()
    em_data = tracker.final_emissions_data

    pixels_processed = metrics["original_shape"][0] * metrics["original_shape"][1] * 8
    throughput_kpx = (pixels_processed / duration) / 1000.0

    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS: IMAGE WORKLOAD")
    print("=" * 80)
    print(f"  Input Resolution    : {metrics['original_shape'][0]}x{metrics['original_shape'][1]} px (RGB)")
    print(f"  File Size           : {file_size_kb:.2f} KB")
    print(f"  Execution Time      : {duration:.4f} s")
    print(f"  Throughput          : {throughput_kpx:.2f} kPixels/s")
    print(f"  Energy Consumed     : {em_data.energy_consumed:.8f} kWh")
    print(f"  Carbon Emissions    : {emissions:.10f} kg CO2e")
    print(f"  CPU Average Power   : {em_data.cpu_power or 0.0:.2f} W")
    print(f"  RAM Average Power   : {em_data.ram_power or 0.0:.2f} W")
    print("=" * 80)


if __name__ == "__main__":
    main()
