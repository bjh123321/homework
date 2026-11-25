import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal_core import (
    GeneratorConfig,
    ProcessingConfig,
    generate,
    analyze,
    process,
    amplitude_spectrum,
    load_csv,
    export_result,
)


class AnalysisTests(unittest.TestCase):
    def test_even_and_odd_fft_amplitudes(self):
        for n in (1000, 1001):
            t = np.arange(n) / n
            x = 3 + 2 * np.sin(2 * np.pi * 30 * t)
            _, amplitude = amplitude_spectrum(x, n)
            self.assertAlmostEqual(amplitude[0], 3)
            self.assertAlmostEqual(amplitude[30], 2)
            if n % 2 == 0:
                _, nyquist = amplitude_spectrum((-1.0) ** np.arange(n), n)
                self.assertAlmostEqual(nyquist[-1], 1)
            else:
                _, highest = amplitude_spectrum(
                    np.cos(2 * np.pi * (n // 2) * t), n
                )
                self.assertAlmostEqual(highest[-1], 1)

    def test_metrics_against_sine_values(self):
        fs = 1000
        time = np.arange(fs) / fs
        data = 2 * np.sin(2 * np.pi * 10 * time)
        result = analyze(
            time, data, fs, {}, ProcessingConfig(False, "none")
        )
        metrics = result.raw_metrics
        expected = {
            "mean": 0,
            "std": np.sqrt(2),
            "rms": np.sqrt(2),
            "absolute_peak": 2,
            "peak_to_peak": 4,
            "kurtosis": 1.5,
            "dominant_frequency_hz": 10,
            "dominant_amplitude": 2,
        }
        for key, value in expected.items():
            self.assertAlmostEqual(metrics[key], value, places=10)
        self.assertAlmostEqual(
            metrics["shape_factor"], np.sqrt(2) / np.mean(np.abs(data))
        )
        np.testing.assert_array_equal(result.raw, result.processed)

    def test_generation_is_reproducible_and_pulses_are_optional(self):
        config = GeneratorConfig()
        time, data = generate(config)
        np.testing.assert_array_equal(data, generate(config)[1])
        self.assertEqual(len(time), 4000)

        pulse_config = GeneratorConfig(pulse_enabled=True, pulse_probability=1)
        difference = generate(pulse_config)[1] - data
        np.testing.assert_allclose(np.abs(difference), 3)

    def test_lowpass_attenuates_high_frequency(self):
        fs = 2000
        time = np.arange(8000) / fs
        data = (
            np.sin(2 * np.pi * 50 * time)
            + np.sin(2 * np.pi * 400 * time)
        )
        processed = process(data, fs, ProcessingConfig(False))
        _, amplitude = amplitude_spectrum(processed[1000:-1000], fs)
        self.assertGreater(amplitude[150], 0.98)
        self.assertLess(amplitude[1200], 0.01)

    def test_window_filters_and_demean(self):
        data = np.zeros(101)
        data[50] = 9
        median = process(
            data, 1000,
            ProcessingConfig(False, "median", window_size=5)
        )
        np.testing.assert_array_equal(median, np.zeros(101))

        average = process(
            data, 1000,
            ProcessingConfig(False, "moving_average", window_size=5)
        )
        np.testing.assert_allclose(average[48:53], 1.8)
        self.assertAlmostEqual(average.sum(), 9)

        demeaned = process(
            np.arange(100) + 4, 1000,
            ProcessingConfig(True, "none")
        )
        self.assertAlmostEqual(demeaned.mean(), 0)

    def test_degenerate_signals(self):
        for level in (0, 5):
            data = np.full(100, level)
            result = analyze(
                np.arange(100) / 100, data, 100, {},
                ProcessingConfig(False, "none")
            )
            self.assertTrue(np.isnan(result.raw_metrics["kurtosis"]))
            self.assertTrue(
                np.isnan(result.raw_metrics["dominant_frequency_hz"])
            )
            if level == 0:
                self.assertTrue(
                    np.isnan(result.raw_metrics["shape_factor"])
                )

    def test_invalid_parameters(self):
        generator_configs = [
            GeneratorConfig(fs=0),
            GeneratorConfig(fs=float("nan")),
            GeneratorConfig(frequency1=1000),
            GeneratorConfig(noise_std=-1),
            GeneratorConfig(duration=1e30),
            GeneratorConfig(seed=2.5),
        ]
        for config in generator_configs:
            with self.assertRaises(ValueError):
                generate(config)

        processing_configs = [
            ProcessingConfig(filter_method="bad"),
            ProcessingConfig(cutoff=500),
            ProcessingConfig(order=1.5),
            ProcessingConfig(filter_method="median", window_size=4),
        ]
        for config in processing_configs:
            with self.assertRaises(ValueError):
                process(np.ones(100), 1000, config)

        with self.assertRaises(ValueError):
            process(np.ones(4), 1000, ProcessingConfig())
        with self.assertRaises(ValueError):
            process(
                [1, 2, np.nan, 4], 1000,
                ProcessingConfig(filter_method="none")
            )

    def test_csv_and_export_round_trip(self):
        time, data = generate(GeneratorConfig())
        result = analyze(
            time, data, 2000, {"type": "test"}, ProcessingConfig()
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = export_result(result, directory)
            expected_files = {
                "waveform.csv",
                "spectrum.csv",
                "psd.csv",
                "stft.csv",
                "metrics.csv",
                "parameters.txt",
            }
            self.assertEqual(
                {path.name for path in destination.iterdir()}, expected_files
            )

            restored_time, restored_data, fs, _ = load_csv(
                destination / "waveform.csv", 1, 3, 1
            )
            np.testing.assert_array_equal(restored_data, result.processed)
            np.testing.assert_array_equal(restored_time, result.time)
            self.assertAlmostEqual(fs, 2000)

            second = export_result(result, directory)
            self.assertNotEqual(destination, second)

            csv_path = Path(directory) / "one.csv"
            csv_path.write_text(
                "amplitude\n1\n2\n3\n4\n", encoding="utf-8"
            )
            np.testing.assert_array_equal(
                load_csv(csv_path, 20)[1], [1, 2, 3, 4]
            )

            invalid_inputs = [
                "1\n2\nbad\n4\n",
                "1\n2\nnan\n4\n",
                "1\n2\n3\n",
                "t,x\n0,1\n0.1,2\n0.3,3\n0.4,4\n",
            ]
            for content in invalid_inputs:
                csv_path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    if content.startswith("t,"):
                        load_csv(csv_path, 20, 2, 1)
                    else:
                        load_csv(csv_path, 20)

    def test_linear_detrend_removes_linear_component(self):
        fs = 1000
        time = np.arange(1000) / fs
        data = np.sin(2 * np.pi * 10 * time) + 0.5 * time
        processed = process(
            data, fs,
            ProcessingConfig(
                remove_mean=True,
                filter_method="none",
                remove_trend=True,
            ),
        )
        slope = np.polyfit(time, processed, 1)[0]
        self.assertAlmostEqual(slope, 0.0, places=8)

    def test_welch_and_stft_products(self):
        fs = 2000
        time = np.arange(4000) / fs
        data = (
            np.sin(2 * np.pi * 50 * time)
            + 0.3 * np.sin(2 * np.pi * 400 * time)
        )
        result = analyze(
            time, data, fs, {"type": "test"},
            ProcessingConfig(False, "none")
        )
        peak = result.welch_frequency[np.argmax(result.raw_psd)]
        self.assertLess(abs(peak - 50), 5)
        self.assertEqual(
            result.raw_stft_magnitude.shape,
            result.processed_stft_magnitude.shape,
        )
        self.assertEqual(
            result.raw_stft_magnitude.shape,
            (len(result.stft_frequency), len(result.stft_time)),
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
            expected_files = {
                "waveform.csv",
                "spectrum.csv",
                "psd.csv",
                "stft.csv",
                "metrics.csv",
                "parameters.txt",
            }
            self.assertEqual(
                {path.name for path in destination.iterdir()}, expected_files
            )
            parameters = (destination / "parameters.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn('"sample_count": 4000', parameters)
            self.assertIn('"stft"', parameters)


if __name__ == "__main__":
    unittest.main()
