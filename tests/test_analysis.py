import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal_core import (GeneratorConfig, ProcessingConfig, generate, analyze, process,
                         amplitude_spectrum, load_csv, export_result)


class AnalysisTests(unittest.TestCase):
    def test_even_and_odd_fft_amplitudes(self):
        for n in (1000, 1001):
            t = np.arange(n) / n
            x = 3 + 2 * np.sin(2 * np.pi * 30 * t)
            f, a = amplitude_spectrum(x, n)
            self.assertAlmostEqual(a[0], 3)
            self.assertAlmostEqual(a[30], 2)
            if n % 2 == 0:
                _, nyquist = amplitude_spectrum((-1.)**np.arange(n), n)
                self.assertAlmostEqual(nyquist[-1], 1)
            else:
                _, highest = amplitude_spectrum(np.cos(2 * np.pi * (n // 2) * t), n)
                self.assertAlmostEqual(highest[-1], 1)

    def test_metrics_against_sine_values(self):
        fs = 1000
        t = np.arange(fs) / fs
        x = 2 * np.sin(2 * np.pi * 10 * t)
        result = analyze(t, x, fs, {}, ProcessingConfig(False, "none"))
        m = result.raw_metrics
        for key, expected in {"mean": 0, "std": np.sqrt(2), "rms": np.sqrt(2),
                              "absolute_peak": 2, "peak_to_peak": 4, "kurtosis": 1.5,
                              "dominant_frequency_hz": 10, "dominant_amplitude": 2}.items():
            self.assertAlmostEqual(m[key], expected, places=10)
        self.assertAlmostEqual(m["shape_factor"], np.sqrt(2) / np.mean(np.abs(x)))
        np.testing.assert_array_equal(result.raw, result.processed)

    def test_generation_is_reproducible_and_pulses_are_optional(self):
        config = GeneratorConfig()
        t, x = generate(config)
        np.testing.assert_array_equal(x, generate(config)[1])
        self.assertEqual(len(t), 4000)
        pulse = GeneratorConfig(pulse_enabled=True, pulse_probability=1)
        difference = generate(pulse)[1] - x
        np.testing.assert_allclose(np.abs(difference), 3)

    def test_lowpass_attenuates_high_frequency(self):
        fs = 2000
        t = np.arange(8000) / fs
        x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 400 * t)
        y = process(x, fs, ProcessingConfig(False))
        _, a = amplitude_spectrum(y[1000:-1000], fs)
        self.assertGreater(a[150], 0.98)
        self.assertLess(a[1200], 0.01)

    def test_window_filters_and_demean(self):
        x = np.zeros(101)
        x[50] = 9
        median = process(x, 1000, ProcessingConfig(False, "median", window_size=5))
        np.testing.assert_array_equal(median, np.zeros(101))
        average = process(x, 1000, ProcessingConfig(False, "moving_average", window_size=5))
        np.testing.assert_allclose(average[48:53], 1.8)
        self.assertAlmostEqual(average.sum(), 9)
        y = process(np.arange(100) + 4, 1000, ProcessingConfig(True, "none"))
        self.assertAlmostEqual(y.mean(), 0)

    def test_degenerate_signals(self):
        for level in (0, 5):
            x = np.full(100, level)
            result = analyze(np.arange(100) / 100, x, 100, {}, ProcessingConfig(False, "none"))
            self.assertTrue(np.isnan(result.raw_metrics["kurtosis"]))
            self.assertTrue(np.isnan(result.raw_metrics["dominant_frequency_hz"]))
            if level == 0:
                self.assertTrue(np.isnan(result.raw_metrics["shape_factor"]))

    def test_invalid_parameters(self):
        for config in [GeneratorConfig(fs=0), GeneratorConfig(fs=float("nan")),
                       GeneratorConfig(frequency1=1000), GeneratorConfig(noise_std=-1),
                       GeneratorConfig(duration=1e30), GeneratorConfig(seed=2.5)]:
            with self.assertRaises(ValueError):
                generate(config)
        for config in [ProcessingConfig(filter_method="bad"), ProcessingConfig(cutoff=500),
                       ProcessingConfig(order=1.5), ProcessingConfig(filter_method="median", window_size=4)]:
            with self.assertRaises(ValueError):
                process(np.ones(100), 1000, config)
        with self.assertRaises(ValueError):
            process(np.ones(4), 1000, ProcessingConfig())
        with self.assertRaises(ValueError):
            process([1, 2, np.nan, 4], 1000, ProcessingConfig(filter_method="none"))

    def test_csv_and_export_round_trip(self):
        t, x = generate(GeneratorConfig())
        result = analyze(t, x, 2000, {"type": "test"}, ProcessingConfig())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            destination = export_result(result, directory)
            self.assertEqual({p.name for p in destination.iterdir()},
                             {"waveform.csv", "spectrum.csv", "psd.csv", "stft.csv",
                              "metrics.csv", "parameters.txt"})
            restored_t, restored_x, fs, source = load_csv(destination / "waveform.csv", 1, 3, 1)
            np.testing.assert_array_equal(restored_x, result.processed)
            np.testing.assert_array_equal(restored_t, result.time)
            self.assertAlmostEqual(fs, 2000)
            second = export_result(result, directory)
            self.assertNotEqual(destination, second)
            path = Path(directory) / "one.csv"
            path.write_text("amplitude\n1\n2\n3\n4\n", encoding="utf-8")
            np.testing.assert_array_equal(load_csv(path, 20)[1], [1, 2, 3, 4])
            for text in ["1\n2\nbad\n4\n", "1\n2\nnan\n4\n", "1\n2\n3\n",
                         "t,x\n0,1\n0.1,2\n0.3,3\n0.4,4\n"]:
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_csv(path, 20, 2, 1) if text.startswith("t,") else load_csv(path, 20)


if __name__ == "__main__":
    unittest.main()


class AdvancedAnalysisTests(unittest.TestCase):
    def test_linear_detrend_removes_linear_component(self):
        fs = 1000
        time = np.arange(1000) / fs
        data = np.sin(2 * np.pi * 10 * time) + 0.5 * time
        result = process(
            data, fs,
            ProcessingConfig(remove_mean=True, filter_method="none", remove_trend=True)
        )
        slope = np.polyfit(time, result, 1)[0]
        self.assertAlmostEqual(slope, 0.0, places=8)

    def test_welch_and_stft_products(self):
        fs = 2000
        time = np.arange(4000) / fs
        data = np.sin(2 * np.pi * 50 * time) + 0.3 * np.sin(2 * np.pi * 400 * time)
        result = analyze(
            time, data, fs, {"type": "test"}, ProcessingConfig(False, "none")
        )
        peak = result.welch_frequency[np.argmax(result.raw_psd)]
        self.assertLess(abs(peak - 50), 5)
        self.assertEqual(
            result.raw_stft_magnitude.shape, result.processed_stft_magnitude.shape
        )
        self.assertEqual(
            result.raw_stft_magnitude.shape,
            (len(result.stft_frequency), len(result.stft_time))
        )
        self.assertGreater(result.raw_psd.max(), 0)
        self.assertGreater(result.processed_psd.max(), 0)

    def test_export_contains_psd_stft_and_parameters(self):
        time, data = generate(GeneratorConfig())
        result = analyze(
            time, data, 2000, {"type": "test"}, ProcessingConfig()
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = export_result(result, directory)
            expected = {
                "waveform.csv", "spectrum.csv", "psd.csv",
                "stft.csv", "metrics.csv", "parameters.txt",
            }
            self.assertEqual({p.name for p in destination.iterdir()}, expected)
            parameters = (destination / "parameters.txt").read_text(encoding="utf-8")
            self.assertIn('"sample_count": 4000', parameters)
            self.assertIn('"stft"', parameters)
