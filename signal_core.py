"""Vibration signal generation, analysis, CSV input and result export."""
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import csv
import json
import math
import os
import shutil
import tempfile
import numpy as np
from scipy import signal, ndimage

MAX_SAMPLES = 1_000_000


@dataclass(frozen=True)
class GeneratorConfig:
    fs: float = 2000.0
    duration: float = 2.0
    amplitude1: float = 1.0
    frequency1: float = 50.0
    amplitude2: float = 0.5
    frequency2: float = 120.0
    interference_amplitude: float = 0.3
    interference_frequency: float = 400.0
    noise_std: float = 0.2
    pulse_enabled: bool = False
    pulse_amplitude: float = 3.0
    pulse_probability: float = 0.005
    seed: int = 42


@dataclass(frozen=True)
class ProcessingConfig:
    remove_mean: bool = True
    filter_method: str = "butterworth"
    cutoff: float = 180.0
    order: int = 4
    window_size: int = 11


@dataclass
class AnalysisResult:
    time: np.ndarray
    raw: np.ndarray
    processed: np.ndarray
    frequency: np.ndarray
    raw_spectrum: np.ndarray
    processed_spectrum: np.ndarray
    raw_metrics: dict
    processed_metrics: dict
    fs: float
    source: dict
    processing: ProcessingConfig


def positive(value, label):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label}必须是有限正数")
    return value


def integer(value, label, low, high):
    value = float(value)
    if not math.isfinite(value) or not value.is_integer() or not low <= value <= high:
        raise ValueError(f"{label}必须为 {low}～{high} 的整数")
    return int(value)


def checked_signal(values, fs):
    positive(fs, "采样率")
    data = np.array(values, dtype=float, copy=True)
    if data.ndim != 1 or not 4 <= data.size <= MAX_SAMPLES:
        raise ValueError(f"信号须为一维，样本数在 4～{MAX_SAMPLES} 之间")
    if not np.all(np.isfinite(data)) or np.max(np.abs(data)) > 1e50:
        raise ValueError("数据包含空值、无穷大或超过 1e50 的幅值")
    return data


def generate(config):
    fs = positive(config.fs, "采样率")
    count = fs * positive(config.duration, "时长")
    if not math.isfinite(count) or not 4 <= count <= MAX_SAMPLES:
        raise ValueError(f"采样率×时长须在 4～{MAX_SAMPLES} 之间")
    count = int(round(count))
    pairs = [(config.amplitude1, config.frequency1), (config.amplitude2, config.frequency2),
             (config.interference_amplitude, config.interference_frequency)]
    for amplitude, frequency in pairs:
        if not math.isfinite(amplitude) or abs(amplitude) > 1e50:
            raise ValueError("正弦幅值须有限且绝对值不大于 1e50")
        if not math.isfinite(frequency) or not 0 < frequency < fs / 2:
            raise ValueError("正弦频率须大于零且小于采样率的一半")
    if not math.isfinite(config.noise_std) or not 0 <= config.noise_std <= 1e50:
        raise ValueError("噪声标准差须在 0～1e50 之间")
    if not math.isfinite(config.pulse_amplitude) or abs(config.pulse_amplitude) > 1e50:
        raise ValueError("脉冲幅值须有限且绝对值不大于 1e50")
    if not math.isfinite(config.pulse_probability) or not 0 <= config.pulse_probability <= 1:
        raise ValueError("脉冲概率须在 0～1 之间")
    rng = np.random.default_rng(integer(config.seed, "随机种子", 0, 2**32 - 1))
    time = np.arange(count) / fs
    data = sum(amplitude * np.sin(2 * np.pi * frequency * time) for amplitude, frequency in pairs)
    data += rng.normal(0, config.noise_std, count)
    if config.pulse_enabled:
        mask = rng.random(count) < config.pulse_probability
        data[mask] += config.pulse_amplitude * rng.choice([-1, 1], int(mask.sum()))
    return time, checked_signal(data, fs)


def load_csv(path, fs, value_column=1, time_column=0):
    """One-based column selection. A zero time column uses the supplied fs."""
    value_column = integer(value_column, "幅值列", 1, 10000)
    time_column = integer(time_column, "时间列", 0, 10000)
    if value_column == time_column:
        raise ValueError("幅值列与时间列必须不同")
    values, times = [], []
    first = True
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.reader(stream), 1):
            if not row or all(not item.strip() for item in row):
                continue
            if len(row) < max(value_column, time_column):
                raise ValueError(f"CSV 第 {line_number} 行列数不足")
            cells = [row[value_column - 1]]
            if time_column:
                cells.append(row[time_column - 1])
            try:
                numbers = [float(cell) for cell in cells]
            except ValueError:
                if first and all(any(char.isalpha() for char in cell) for cell in cells):
                    first = False
                    continue
                raise ValueError(f"CSV 第 {line_number} 行包含非数值或空单元格") from None
            first = False
            values.append(numbers[0])
            if time_column:
                times.append(numbers[1])
            if len(values) > MAX_SAMPLES:
                raise ValueError(f"样本数超过 {MAX_SAMPLES}")
    if time_column:
        time = np.asarray(times, dtype=float)
        if len(time) < 4 or not np.all(np.isfinite(time)):
            raise ValueError("时间列须至少有 4 个有限数值，单位为秒")
        steps = np.diff(time)
        dt = float(np.median(steps))
        if dt <= 0 or not np.allclose(steps, dt, rtol=1e-4, atol=abs(dt) * 1e-6):
            raise ValueError("时间列须严格递增且等间隔")
        fs = positive(1 / dt, "推算采样率")
    else:
        fs = positive(fs, "采样率")
        time = np.arange(len(values)) / fs
    data = checked_signal(values, fs)
    return time, data, fs, {"type": "csv", "path": str(Path(path).resolve()),
                            "value_column": value_column, "time_column": time_column}


def process(data, fs, config):
    output = checked_signal(data, fs)
    if config.remove_mean:
        output -= output.mean()
    if config.filter_method == "butterworth":
        cutoff = positive(config.cutoff, "截止频率")
        if cutoff >= fs / 2:
            raise ValueError("截止频率须小于采样率的一半")
        order = integer(config.order, "滤波阶数", 1, 12)
        sos = signal.butter(order, cutoff, fs=fs, output="sos")
        try:
            output = signal.sosfiltfilt(sos, output)
        except ValueError as error:
            raise ValueError("样本过少，无法执行双向低通滤波，请增加样本或降低阶数") from error
    elif config.filter_method in ("moving_average", "median"):
        size = integer(config.window_size, "窗口点数", 1, min(len(output), 1001))
        if size % 2 == 0:
            raise ValueError("窗口点数须为奇数")
        if config.filter_method == "moving_average":
            output = ndimage.uniform_filter1d(output, size=size, mode="reflect")
        else:
            output = ndimage.median_filter(output, size=size, mode="reflect")
    elif config.filter_method != "none":
        raise ValueError("未知滤波方法")
    return checked_signal(output, fs)


def amplitude_spectrum(data, fs):
    n = len(data)
    amplitude = np.abs(np.fft.rfft(data)) / n
    amplitude[1:-1 if n % 2 == 0 else None] *= 2
    return np.fft.rfftfreq(n, 1 / fs), amplitude


def metrics(data, frequency, spectrum):
    centered = data - data.mean()
    variance = float(np.mean(centered**2))
    rms = float(np.sqrt(np.mean(data**2)))
    mean_abs = float(np.mean(np.abs(data)))
    index = int(np.argmax(spectrum[1:])) + 1
    return {"mean": float(data.mean()), "std": float(np.sqrt(variance)), "rms": rms,
            "absolute_peak": float(np.max(np.abs(data))), "peak_to_peak": float(np.ptp(data)),
            "kurtosis": float(np.mean(centered**4) / variance**2) if variance > 0 else math.nan,
            "shape_factor": rms / mean_abs if mean_abs > 0 else math.nan,
            "dominant_frequency_hz": float(frequency[index]) if variance > 0 and spectrum[index] > 0 else math.nan,
            "dominant_amplitude": float(spectrum[index]) if variance > 0 else 0.0}


def analyze(time, data, fs, source, config):
    raw = checked_signal(data, fs)
    time = np.array(time, dtype=float, copy=True)
    if time.shape != raw.shape or not np.all(np.isfinite(time)):
        raise ValueError("时间轴与信号不匹配")
    if not np.allclose(np.diff(time), 1 / fs, rtol=1e-4, atol=1e-10 / fs):
        raise ValueError("时间轴与采样率不匹配")
    processed = process(raw, fs, config)
    frequency, raw_spectrum = amplitude_spectrum(raw, fs)
    _, processed_spectrum = amplitude_spectrum(processed, fs)
    return AnalysisResult(time, raw, processed, frequency, raw_spectrum, processed_spectrum,
                          metrics(raw, frequency, raw_spectrum), metrics(processed, frequency, processed_spectrum),
                          fs, source, config)


def export_result(result, parent):
    """Publish a complete result directory after all four files have been written."""
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".vibration-", dir=parent))
    destination = parent / ("result_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    try:
        np.savetxt(temporary / "waveform.csv", np.column_stack([result.time, result.raw, result.processed]),
                   delimiter=",", header="time_s,raw,processed", comments="", fmt="%.17g", encoding="utf-8-sig")
        np.savetxt(temporary / "spectrum.csv",
                   np.column_stack([result.frequency, result.raw_spectrum, result.processed_spectrum]),
                   delimiter=",", header="frequency_hz,raw_peak_amplitude,processed_peak_amplitude",
                   comments="", fmt="%.17g", encoding="utf-8-sig")
        with (temporary / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["metric", "raw", "processed"])
            for name in result.raw_metrics:
                writer.writerow([name, result.raw_metrics[name], result.processed_metrics[name]])
        parameters = {"created_at": datetime.now().astimezone().isoformat(), "fs_hz": result.fs,
                      "sample_count": len(result.raw), "record_duration_s": len(result.raw) / result.fs,
                      "source": result.source, "processing": asdict(result.processing),
                      "definitions": {"std": "population, ddof=0", "kurtosis": "central moment, Gaussian=3",
                                      "shape_factor": "RMS / mean(abs(x))", "undefined": "nan",
                                      "spectrum": "rectangular window, one-sided peak amplitude",
                                      "dominant_frequency": "largest bin excluding DC, no interpolation",
                                      "butterworth": "zero-phase forward/backward sosfiltfilt",
                                      "window_boundary": "reflect"}}
        (temporary / "parameters.txt").write_text(json.dumps(parameters, ensure_ascii=False, indent=2), encoding="utf-8")
        os.rename(temporary, destination)
    except Exception:
        shutil.rmtree(temporary)
        raise
    return destination
