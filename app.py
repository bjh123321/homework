"""Desktop entry point for the vibration-signal virtual instrument."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np

from signal_core import (
    GeneratorConfig, ProcessingConfig, generate, load_csv,
    analyze, export_result, integer,
)

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

FILTERS = {
    "不滤波": "none",
    "Butterworth 低通": "butterworth",
    "移动平均": "moving_average",
    "中值滤波": "median",
}
METRICS = {
    "mean": "均值",
    "std": "标准差",
    "rms": "均方根值 RMS",
    "absolute_peak": "绝对峰值",
    "peak_to_peak": "峰峰值",
    "kurtosis": "峭度",
    "shape_factor": "波形因子",
    "dominant_frequency_hz": "主频 (Hz)",
    "dominant_amplitude": "主频幅值",
}


class VibrationApp:
    def __init__(self, root):
        self.root = root
        self.data = None
        self.result = None
        self.revision = 0
        self.busy = False
        self.closed = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.variables = {}
        self.axes = []
        self.figures = []
        self.canvases = []
        self.colorbars = [None] * 6

        root.title("振动信号采集、滤波与频谱分析虚拟仪器")
        root.geometry("1380x920")
        root.minsize(1050, 720)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.status = tk.StringVar(value="请生成仿真信号或导入 CSV 文件")
        self.source_text = tk.StringVar(value="尚未载入数据")
        self.build_controls()
        self.build_results()
        ttk.Label(root, textvariable=self.status, padding=8, wraplength=1250).pack(
            fill="x", side="bottom"
        )
        self.refresh_buttons()

    def entry(self, parent, label, key, default, row, column=0, source=True):
        ttk.Label(parent, text=label).grid(
            row=row, column=column * 2, sticky="w", padx=5, pady=3
        )
        variable = tk.StringVar(value=str(default))
        self.variables[key] = variable
        ttk.Entry(parent, textvariable=variable, width=12).grid(
            row=row, column=column * 2 + 1, padx=5, pady=3
        )
        variable.trace_add("write", lambda *_: self.invalidate(source))

    def build_controls(self):
        controls = ttk.Notebook(self.root)
        controls.pack(fill="x", padx=10, pady=8)
        generation = ttk.Frame(controls, padding=8)
        importing = ttk.Frame(controls, padding=8)
        processing = ttk.Frame(controls, padding=8)
        controls.add(generation, text="信号生成")
        controls.add(importing, text="CSV 导入")
        controls.add(processing, text="预处理与滤波")

        config = GeneratorConfig()
        fields = [
            ("采样率 (Hz)", "fs"), ("时长 (s)", "duration"),
            ("正弦1 幅值", "amplitude1"), ("正弦1 频率 (Hz)", "frequency1"),
            ("正弦2 幅值", "amplitude2"), ("正弦2 频率 (Hz)", "frequency2"),
            ("干扰幅值", "interference_amplitude"),
            ("干扰频率 (Hz)", "interference_frequency"),
            ("噪声标准差", "noise_std"), ("随机种子", "seed"),
            ("脉冲幅值", "pulse_amplitude"), ("每点脉冲概率", "pulse_probability"),
        ]
        for index, (label, key) in enumerate(fields):
            self.entry(
                generation, label, key, getattr(config, key),
                index // 4, index % 4
            )

        self.pulse = tk.BooleanVar(value=False)
        self.pulse.trace_add("write", lambda *_: self.invalidate(True))
        ttk.Checkbutton(
            generation, text="加入随机正负脉冲", variable=self.pulse
        ).grid(row=3, column=0, columnspan=2, sticky="w")
        self.generate_button = ttk.Button(
            generation, text="生成并分析", command=self.generate_action
        )
        self.generate_button.grid(row=3, column=6, columnspan=2, pady=5)

        self.entry(importing, "采样率 (Hz)", "csv_fs", 2000, 0)
        self.entry(importing, "幅值列 (从1开始)", "value_column", 1, 0, 1)
        self.entry(importing, "时间列 (0为无)", "time_column", 0, 0, 2)
        ttk.Label(
            importing,
            text="支持 UTF-8 CSV，可有一行表头。无时间列时使用填写的采样率；"
                 "有时间列时要求单位为秒、严格递增且等间隔，程序据此推算采样率。",
            wraplength=1050,
        ).grid(row=1, column=0, columnspan=6, sticky="w", padx=5, pady=8)
        self.import_button = ttk.Button(
            importing, text="选择 CSV 并分析", command=self.import_action
        )
        self.import_button.grid(row=2, column=0, columnspan=2, sticky="w", padx=5)
        ttk.Label(
            importing,
            text="导入导出文件 waveform.csv：时间列填 1，幅值列填 2（原始）或 3（处理后）。",
        ).grid(row=2, column=2, columnspan=4, sticky="w")

        self.demean = tk.BooleanVar(value=True)
        self.demean.trace_add("write", lambda *_: self.invalidate(False))
        ttk.Checkbutton(processing, text="去均值", variable=self.demean).grid(
            row=0, column=0, sticky="w", padx=5
        )
        self.detrend = tk.BooleanVar(value=False)
        self.detrend.trace_add("write", lambda *_: self.invalidate(False))
        ttk.Checkbutton(processing, text="线性去趋势", variable=self.detrend).grid(
            row=0, column=1, sticky="w", padx=5
        )
        self.method = tk.StringVar(value="Butterworth 低通")
        self.method.trace_add("write", lambda *_: self.invalidate(False))
        ttk.Combobox(
            processing, textvariable=self.method, values=list(FILTERS),
            state="readonly", width=20
        ).grid(row=0, column=2, padx=5)
        self.entry(processing, "截止频率 (Hz)", "cutoff", 180, 1, source=False)
        self.entry(processing, "低通阶数 (1～12)", "order", 4, 1, 1, source=False)
        self.entry(processing, "窗口点数 (奇数)", "window_size", 11, 1, 2, source=False)
        ttk.Label(
            processing,
            text="低通采用 SOS 双向零相位滤波；移动平均和中值滤波采用反射边界。"
                 "FFT 为单边幅值谱；Welch 使用 Hann 窗和 50% 重叠；"
                 "STFT 使用 Hann 窗和 75% 重叠。",
            wraplength=1100,
        ).grid(row=2, column=0, columnspan=6, sticky="w", padx=5, pady=8)

        actions = ttk.Frame(self.root, padding=(12, 0))
        actions.pack(fill="x")
        self.analyze_button = ttk.Button(
            actions, text="按当前参数重新分析", command=self.analyze_action
        )
        self.analyze_button.pack(side="left")
        self.save_button = ttk.Button(
            actions, text="保存全部结果", command=self.save_action
        )
        self.save_button.pack(side="left", padx=10)
        ttk.Label(actions, textvariable=self.source_text).pack(side="left", padx=10)

    def build_results(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        titles = [
            "原始波形", "处理后波形", "波形对比",
            "FFT 幅值谱", "Welch 功率谱", "STFT 时频图",
        ]
        for title in titles:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=title)
            figure = Figure(figsize=(9, 4), dpi=100, layout="constrained")
            axis = figure.add_subplot(111)
            canvas = FigureCanvasTkAgg(figure, master=frame)
            toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.axes.append(axis)
            self.figures.append(figure)
            self.canvases.append(canvas)

        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="分析指标")
        self.table = ttk.Treeview(
            frame, columns=("metric", "raw", "processed"),
            show="headings", height=10
        )
        for key, label in [
            ("metric", "指标"), ("raw", "原始信号"), ("processed", "处理后信号")
        ]:
            self.table.heading(key, text=label)
            self.table.column(key, width=230, anchor="center")
        self.table.pack(fill="x")
        ttk.Label(
            frame,
            text="标准差采用总体定义；峭度为中心四阶矩/方差平方，高斯信号理论值为3。"
                 "波形因子为 RMS/平均绝对值。FFT 主频取非直流最大幅值频点。"
                 "Welch 纵轴为功率谱密度，STFT 颜色表示局部幅值强度。",
            padding=12, wraplength=1050,
        ).pack(anchor="w")

    def refresh_buttons(self):
        state = "disabled" if self.busy else "normal"
        self.generate_button.configure(state=state)
        self.import_button.configure(state=state)
        self.analyze_button.configure(
            state="normal" if self.data is not None and not self.busy else "disabled"
        )
        self.save_button.configure(
            state="normal" if self.result is not None and not self.busy else "disabled"
        )

    def invalidate(self, source):
        self.revision += 1
        self.result = None
        if source:
            self.data = None
            self.source_text.set("数据参数已变更，请重新生成或导入")
        self.table.delete(*self.table.get_children())
        for index, (axis, canvas) in enumerate(zip(self.axes, self.canvases)):
            if self.colorbars[index] is not None:
                self.colorbars[index].remove()
                self.colorbars[index] = None
            axis.clear()
            canvas.draw_idle()
        self.status.set("参数已变更，结果待重新计算")
        self.refresh_buttons()

    def processing_config(self):
        method = FILTERS[self.method.get()]
        cutoff, order, window = 180.0, 4, 11
        if method == "butterworth":
            cutoff = float(self.variables["cutoff"].get())
            order = integer(self.variables["order"].get(), "滤波阶数", 1, 12)
        if method in ("moving_average", "median"):
            window = integer(self.variables["window_size"].get(), "窗口点数", 1, 1001)
        return ProcessingConfig(
            self.demean.get(), method, cutoff, order, window, self.detrend.get()
        )

    def run_job(self, work, completed):
        if self.busy:
            return
        self.busy = True
        revision = self.revision
        self.refresh_buttons()
        self.status.set("正在计算，请稍候…")
        future = self.executor.submit(work)

        def poll():
            if self.closed:
                return
            if not future.done():
                self.root.after(80, poll)
                return
            self.busy = False
            try:
                value = future.result()
                if revision == self.revision:
                    completed(value)
                else:
                    self.status.set("计算期间参数已变更，请按当前参数重新计算")
            except Exception as error:
                self.status.set(f"操作失败：{error}")
                messagebox.showerror("操作失败", str(error), parent=self.root)
            self.refresh_buttons()

        self.root.after(80, poll)

    def generate_action(self):
        try:
            settings = {
                key: float(self.variables[key].get())
                for key in asdict(GeneratorConfig())
                if key not in ("pulse_enabled", "seed")
            }
            settings["seed"] = integer(
                self.variables["seed"].get(), "随机种子", 0, 2**32 - 1
            )
            config = GeneratorConfig(**settings, pulse_enabled=self.pulse.get())
            processing = self.processing_config()
        except ValueError as error:
            self.input_error(error)
            return
        self.invalidate(True)

        def work():
            time, data = generate(config)
            source = {"type": "generated", "parameters": asdict(config)}
            return (
                (time, data, config.fs, source),
                analyze(time, data, config.fs, source, processing),
            )

        self.run_job(work, self.accept_data)

    def import_action(self):
        path = filedialog.askopenfilename(
            title="导入单通道 CSV",
            filetypes=[("CSV", "*.csv"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            fs = float(self.variables["csv_fs"].get())
            value_column = integer(
                self.variables["value_column"].get(), "幅值列", 1, 10000
            )
            time_column = integer(
                self.variables["time_column"].get(), "时间列", 0, 10000
            )
            processing = self.processing_config()
        except ValueError as error:
            self.input_error(error)
            return
        self.invalidate(True)

        def work():
            data = load_csv(path, fs, value_column, time_column)
            return data, analyze(*data, processing)

        self.run_job(work, self.accept_data)

    def analyze_action(self):
        if self.data is None:
            return
        try:
            config = self.processing_config()
        except ValueError as error:
            self.input_error(error)
            return
        self.invalidate(False)
        data = self.data
        self.run_job(lambda: analyze(*data, config), self.show_result)

    def accept_data(self, value):
        self.data, result = value
        source = (
            "生成信号" if result.source["type"] == "generated"
            else Path(result.source["path"]).name
        )
        self.source_text.set(
            f"{source}    {len(result.raw)} 点    {result.fs:g} Hz"
        )
        self.show_result(result)

    def show_result(self, result):
        self.result = result
        indices = np.linspace(
            0, len(result.raw) - 1, min(20000, len(result.raw)), dtype=int
        )
        for index, (axis, canvas) in enumerate(zip(self.axes, self.canvases)):
            if self.colorbars[index] is not None:
                self.colorbars[index].remove()
                self.colorbars[index] = None
            axis.clear()

            if index in (0, 1, 2):
                if index in (0, 2):
                    axis.plot(
                        result.time[indices], result.raw[indices],
                        label="原始", linewidth=0.8,
                    )
                if index in (1, 2):
                    axis.plot(
                        result.time[indices], result.processed[indices],
                        label="处理后", linewidth=0.8,
                    )
                axis.set_xlabel("时间 (s)")
                axis.set_ylabel("幅值")
                axis.legend(loc="upper right")
                axis.grid(True, alpha=0.25)

            elif index == 3:
                axis.plot(
                    result.frequency, result.raw_spectrum,
                    label="原始", linewidth=1,
                )
                axis.plot(
                    result.frequency, result.processed_spectrum,
                    label="处理后", linewidth=1,
                )
                axis.set_xlabel("频率 (Hz)")
                axis.set_ylabel("单边峰值幅度")
                axis.set_title("FFT 单边幅值谱")
                axis.legend(loc="upper right")
                axis.grid(True, alpha=0.25)

            elif index == 4:
                floor = np.finfo(float).tiny
                axis.semilogy(
                    result.welch_frequency, np.maximum(result.raw_psd, floor),
                    label="原始", linewidth=1,
                )
                axis.semilogy(
                    result.welch_frequency, np.maximum(result.processed_psd, floor),
                    label="处理后", linewidth=1,
                )
                axis.set_xlabel("频率 (Hz)")
                axis.set_ylabel("功率谱密度")
                axis.set_title("Welch PSD（Hann 窗，50% 重叠）")
                axis.legend(loc="upper right")
                axis.grid(True, alpha=0.25)

            else:
                magnitude_db = 20 * np.log10(
                    np.maximum(result.processed_stft_magnitude, 1e-12)
                )
                mesh = axis.pcolormesh(
                    result.stft_time, result.stft_frequency, magnitude_db,
                    shading="auto", cmap="viridis",
                )
                self.colorbars[index] = canvas.figure.colorbar(
                    mesh, ax=axis, pad=0.02, label="幅值 (dB)"
                )
                axis.set_xlabel("时间 (s)")
                axis.set_ylabel("频率 (Hz)")
                axis.set_title("处理后信号 STFT 时频图（Hann 窗，75% 重叠）")

            canvas.draw_idle()

        self.table.delete(*self.table.get_children())
        for key, label in METRICS.items():
            values = [result.raw_metrics[key], result.processed_metrics[key]]
            self.table.insert(
                "", "end",
                values=[label] + [
                    f"{value:.8g}" if math.isfinite(value) else "—"
                    for value in values
                ],
            )
        self.status.set(
            f"分析完成。FFT 频率分辨率 {result.fs / len(result.raw):g} Hz；"
            "可查看六类图形并保存全部数值结果。"
            + ("时域绘图抽取至20000点，计算和导出保留全部样本。"
               if len(result.raw) > 20000 else "")
        )

    def save_action(self):
        if self.result is None:
            return
        parent = filedialog.askdirectory(title="选择结果保存目录")
        if not parent:
            return
        result = self.result
        self.run_job(
            lambda: export_result(result, parent),
            lambda path: self.status.set(f"已保存全部分析结果至 {path}"),
        )

    def input_error(self, error):
        self.invalidate(False)
        self.status.set(f"参数错误：{error}")
        messagebox.showerror("参数错误", str(error), parent=self.root)

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=False)
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    VibrationApp(root)
    root.mainloop()
