"""
GreenGauge Experiment: Audio Signal Processing Benchmark
========================================================
File Modality: Audio (.wav)
Tracks energy consumption and carbon emissions for digital signal processing & audio feature extraction.

Pipeline stages:
1. Synthetic audio waveform generation / loading (44.1 kHz, 16-bit PCM, multi-frequency + chirp)
2. Normalization & dynamic range adjustments
3. Fast Fourier Transform (FFT) frequency spectrum analysis
4. Short-Time Fourier Transform (STFT) & Spectrogram extraction
5. Acoustic feature computation:
   - Root-Mean-Square (RMS) Energy
   - Zero-Crossing Rate (ZCR)
   - Spectral Centroid & Spectral Rolloff
6. Digital IIR Bandpass Filtering (Butterworth filter)
"""

import os
import time
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
from codecarbon import EmissionsTracker

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
AUDIO_PATH = os.path.join(DATA_DIR, "sample_audio.wav")


def ensure_sample_audio():
    """Generates a small synthetic audio file (3s @ 44.1 kHz, ~260 KB) if not present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(AUDIO_PATH):
        return AUDIO_PATH

    print(f"[*] Generating synthetic sample audio at {AUDIO_PATH}...")
    sample_rate = 44100
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)

    # Multi-harmonic tones (A4: 440 Hz, E5: 659.25 Hz, A5: 880 Hz)
    tone_1 = 0.4 * np.sin(2 * np.pi * 440 * t)
    tone_2 = 0.3 * np.sin(2 * np.pi * 659.25 * t)
    tone_3 = 0.2 * np.sin(2 * np.pi * 880 * t)

    # Linear frequency chirp sweep from 200 Hz to 3000 Hz
    chirp = 0.2 * signal.chirp(t, f0=200, t1=duration_s, f1=3000, method="linear")

    # Modulated noise
    noise = 0.05 * np.random.normal(0, 1, len(t))

    composite = tone_1 + tone_2 + tone_3 + chirp + noise
    # Normalize to 16-bit integer range [-32768, 32767]
    normalized = np.int16((composite / np.max(np.abs(composite))) * 32767)

    wavfile.write(AUDIO_PATH, sample_rate, normalized)
    size_kb = os.path.getsize(AUDIO_PATH) / 1024
    print(f"[OK] Generated {AUDIO_PATH} ({size_kb:.1f} KB, duration={duration_s}s)")
    return AUDIO_PATH


def process_audio(audio_path, passes=10):
    """Executes a compute-controlled audio signal processing pipeline."""
    sample_rate, data = wavfile.read(audio_path)
    # Convert to float64 in range [-1.0, 1.0]
    waveform = data.astype(np.float64) / 32768.0

    results = {}
    for _ in range(passes):
        # 1. FFT Frequency Spectrum
        fft_complex = np.fft.rfft(waveform)
        fft_magnitude = np.abs(fft_complex)
        freqs = np.fft.rfftfreq(len(waveform), 1.0 / sample_rate)

        # 2. Short-Time Fourier Transform (STFT)
        f_stft, t_stft, zxx = signal.stft(waveform, fs=sample_rate, nperseg=1024, noverlap=512)
        spectrogram = np.abs(zxx)

        # 3. Time-domain features: RMS energy and Zero-Crossing Rate
        frame_len = 1024
        hop_len = 512
        frames = [waveform[i:i + frame_len] for i in range(0, len(waveform) - frame_len, hop_len)]
        rms_energy = np.array([np.sqrt(np.mean(f**2)) for f in frames])
        zcr = np.array([np.mean(np.abs(np.diff(np.sign(f)))) / 2 for f in frames])

        # 4. Spectral Centroid
        spectral_centroid = np.sum(freqs * fft_magnitude) / (np.sum(fft_magnitude) + 1e-12)

        # 5. Spectral Rolloff (85% energy threshold)
        cumulative_energy = np.cumsum(fft_magnitude)
        threshold = 0.85 * cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= threshold)[0][0]
        spectral_rolloff = freqs[rolloff_idx]

        # 6. Butterworth Bandpass Filter (300 Hz - 3400 Hz speech band)
        sos = signal.butter(6, [300, 3400], btype="bandpass", fs=sample_rate, output="sos")
        filtered_waveform = signal.sosfilt(sos, waveform)

    results["sample_rate"] = sample_rate
    results["samples"] = len(waveform)
    results["duration_s"] = len(waveform) / sample_rate
    results["stft_shape"] = spectrogram.shape
    results["mean_rms"] = float(np.mean(rms_energy))
    results["mean_zcr"] = float(np.mean(zcr))
    results["spectral_centroid_hz"] = float(spectral_centroid)
    results["spectral_rolloff_hz"] = float(spectral_rolloff)
    results["filtered_std"] = float(np.std(filtered_waveform))

    return results


def main():
    print("=" * 80)
    print("  GREENGAUGE: AUDIO PROCESSING EXPERIMENT (CodeCarbon)")
    print("=" * 80)

    audio_path = ensure_sample_audio()
    file_size_kb = os.path.getsize(audio_path) / 1024
    print(f"Target Audio: {audio_path} ({file_size_kb:.2f} KB)")

    tracker = EmissionsTracker(
        project_name="exp_audio_processing",
        tracking_mode="process",
        save_to_file=False,
        log_level="warning"
    )

    print("\nStarting audio benchmark (FFT, STFT Spectrogram, Centroid, Butterworth Filter)...")
    tracker.start()
    t0 = time.perf_counter()

    metrics = process_audio(audio_path, passes=15)

    duration = time.perf_counter() - t0
    emissions = tracker.stop()
    em_data = tracker.final_emissions_data

    total_samples_processed = metrics["samples"] * 15
    throughput_ksamples = (total_samples_processed / duration) / 1000.0

    print("\n" + "=" * 80)
    print("  BENCHMARK RESULTS: AUDIO WORKLOAD")
    print("=" * 80)
    print(f"  Audio Duration      : {metrics['duration_s']:.2f} s @ {metrics['sample_rate']} Hz")
    print(f"  File Size           : {file_size_kb:.2f} KB")
    print(f"  Execution Time      : {duration:.4f} s")
    print(f"  Throughput          : {throughput_ksamples:.2f} kSamples/s")
    print(f"  Spectrogram Shape   : {metrics['stft_shape']}")
    print(f"  Spectral Centroid   : {metrics['spectral_centroid_hz']:.1f} Hz")
    print(f"  Spectral Rolloff    : {metrics['spectral_rolloff_hz']:.1f} Hz")
    print(f"  Energy Consumed     : {em_data.energy_consumed:.8f} kWh")
    print(f"  Carbon Emissions    : {emissions:.10f} kg CO2e")
    print(f"  CPU Average Power   : {em_data.cpu_power or 0.0:.2f} W")
    print(f"  RAM Average Power   : {em_data.ram_power or 0.0:.2f} W")
    print("=" * 80)


if __name__ == "__main__":
    main()
