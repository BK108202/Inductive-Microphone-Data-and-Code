"""Global IM peak versus independently recorded background at that exact frequency.

The original in-band SNR and full-record RMS ratio remain separate metrics.
This module reads no files and writes no analysis outputs.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal.windows import hann


METHODS = {
    "metric": "IM global peak / same-frequency background spectral power ratio; measured peak power, not background-subtracted excess-power SNR.",
    "signal": "Use existing global_padded_parabolic_hz f* and global_padded_parabolic_amplitude A from the all-positive-frequency maximum; peak_power=A**2/2, conditional on a resolved sinusoid. Source interpolation must be available; no target or bin fallback.",
    "noise": "In each complete noise window evaluate C=sum(w*(y-mean(y))*exp(-2j*pi*f*arange(N)/fs))/sum(w) at the SOURCE f*, with periodic Hann w and the same linearly resampled samples as the signal FFT pipeline. background_power=2*abs(C)**2 for 0<f*<fs/2. This is a direct Fourier projection, not a noise maximum, nearest FFT bin, or interpolated noise peak. The signal uses its existing interpolated peak estimate while noise uses direct evaluation without maximization.",
    "windows": "Each configured independent IM noise interval is divided into consecutive, nonoverlapping complete window_s windows (default 6 s), anchored at the configured start. Tail is omitted and its duration recorded; no shortening or padding. Signal requested duration must equal window_s. Per-window actual timestamps, sample counts and effective N/fs durations are retained.",
    "preprocessing": "Reuse analyze_segment for timestamp integrity, boundary coverage, gap policy, and linear interpolation on N points spanning retained first-to-last timestamps; then remove each interpolated window's mean and apply periodic Hann. No additional filter, outlier removal or detrending. Original data are unchanged.",
    "bandwidth": "No added +/-1 Hz band. Nominal Hann ENBW=fs*sum(w**2)/sum(w)**2=1.5*fs/N, approximately 0.25 Hz for 6 s. background_power equals the one-sided PSD evaluated at f* times that window's nominal ENBW. Mean removal and timestamp resampling can additionally change the effective response. Exact source/noise nominal ENBWs are retained; these are estimator bandwidths, not sensor bandwidth or calibrated analog bandwidth. Zero padding does not narrow ENBW.",
    "aggregation": "Average finite nonnegative per-window BACKGROUND POWERS in linear units, separately within each explicitly selected noise recording. Do not average amplitudes or dB and do not pool noise controls. Nonoverlapping windows are not presumed statistically independent. All planned complete windows must be supported; otherwise the comparison is unavailable while partial per-window powers remain for audit.",
    "definition": "peak_background_power_ratio=(A**2/2)/mean(background_power); peak_background_ratio_db=10*log10(peak_background_power_ratio). No background subtraction, artificial floor or uncertainty inferred from a single spectral estimate. Both powers have units (Data28 counts)**2; their ratio is dimensionless.",
    "quality": "Missing/ambiguous references, invalid units (must explicitly be Data28 counts), wrong window duration, unsupported FFTs, unavailable source interpolation, frequencies outside any required sampling support, and nonpositive mean background power give explicit unavailable comparisons. Computed timing-warning windows remain included in diagnostic mode; all quality metadata are preserved. Off-nominal frequency tolerance only flags, never restricts peak selection.",
    "limitations": "Matched acquisition gain, filtering, analog bandwidth and independent-background representativeness remain unconfirmed. Global maximum selection can select interference/harmonics or bias weak signals; noise contains any same-frequency interference. Finite-window spectral leakage includes neighboring frequencies. Sampling and interpolation can bias high-frequency results. Ratios do not establish detection, calibrated sensitivity, sensor bandwidth, phase or a synchronized transfer function.",
}


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _text(value):
    return "" if value is None or pd.isna(value) else str(value)


def _true(value):
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _positive(value):
    return np.isfinite(value) and value > 0


def _file(row):
    return _text(row.get("file_id")) or _text(row.get("file_label"))


def _settings(config):
    settings = config.get("same_frequency_background", {})
    width = _number(settings.get("window_s", 6.0))
    tolerance = _number(settings.get("frequency_tolerance_hz", 1.0))
    if not _positive(width) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("same_frequency_background requires positive finite window_s and nonnegative finite frequency_tolerance_hz")
    return settings, width, tolerance


def _noise_projection_supported(row):
    # The shared analyzer builds valid uniform arrays for a constant record but
    # leaves fft_computed=False because a global peak cannot be estimated.
    return (_true(row.get("fft_computed"))
            or (row.get("global_interpolation_status") == "zero_signal"
                and _true(row.get("window_coverage_passed"))
                and (row.get("gap_policy") == "diagnostic" or _true(row.get("strict_gap_check_passed")))))


def prepare_noise_windows(record, noise_row, config, analyze_segment):
    """Prepare independently audited windows while their raw record is loaded."""
    _, width, _ = _settings(config)
    start, end = (_number(noise_row.get(key)) for key in ("analysis_start_s", "analysis_end_s"))
    reference = dict(row=dict(noise_row), windows=[], window_s=width,
                     planned_window_count=0, unused_tail_s=np.nan,
                     preparation_status="unavailable", preparation_error="")
    if not np.isfinite(start) or not np.isfinite(end) or end <= start:
        reference["preparation_error"] = "invalid_configured_noise_interval"
        return reference
    count = int(np.floor((end - start) / width + 1e-10))
    reference.update(planned_window_count=count,
                     unused_tail_s=max(0.0, float(end - start - count * width)))
    for index in range(count):
        a, b = start + index * width, start + (index + 1) * width
        row = {**noise_row, "window_index": index + 1, "analysis_start_s": a,
               "analysis_end_s": b, "requested_duration_s": width,
               "window_status": "unavailable", "window_error": ""}
        uniform = np.array([], dtype=float)
        try:
            analyzed, spectrum = analyze_segment(record, a, b, None, config.get("fft", {}))
            row.update(analyzed)
            uniform = np.asarray(spectrum.get("uniform_y", []), dtype=float).copy()
            if not _noise_projection_supported(row):
                row["window_status"] = "fft_unavailable"
            elif uniform.ndim != 1 or len(uniform) < 4 or not np.isfinite(uniform).all():
                row["window_status"] = "invalid_uniform_samples"
            else:
                row["window_status"] = "available"
        except ValueError as exc:
            row.update(window_status="unsupported", window_error=str(exc), fft_computed=False)
        reference["windows"].append({"row": row, "uniform_y": uniform})
    reference["preparation_status"] = "prepared" if count else "no_complete_windows"
    return reference


def evaluate_frequency_power(y, fs, frequency):
    """Direct one-sided Hann spectral power at a specified interior frequency."""
    y = np.asarray(y, dtype=float)
    if (y.ndim != 1 or y.size < 4 or not np.isfinite(y).all() or not _positive(fs)
            or not _positive(frequency) or frequency >= fs / 2):
        raise ValueError("Finite samples and fs with 0 < frequency < fs/2 are required")
    weights = hann(len(y), sym=False)
    coefficient = np.sum(weights * (y - y.mean()) * np.exp(-2j * np.pi * frequency * np.arange(len(y)) / fs)) / weights.sum()
    return float(2 * abs(coefficient)**2)


def _window_power(window, frequency, width):
    """Evaluate the same physical source frequency; zero power is a valid sample."""
    row = window["row"]
    y = np.asarray(window.get("uniform_y", []), dtype=float)
    fs = _number(row.get("fs_hz"))
    duration = _number(row.get("analysis_end_s")) - _number(row.get("analysis_start_s"))
    output = dict(background_power=np.nan, noise_enbw_hz=np.nan,
                  noise_effective_fft_duration_s=np.nan, noise_samples_per_peak_cycle=np.nan,
                  noise_window_status="unavailable")
    problem = None
    if row.get("window_status") != "available" or not _noise_projection_supported(row):
        problem = "noise_window_unsupported"
    elif _text(row.get("unit")) != "Data28 counts":
        problem = "noise_window_units_invalid"
    elif not np.isclose(duration, width, rtol=0, atol=1e-8):
        problem = "noise_window_duration_mismatch"
    elif not _positive(fs):
        problem = "noise_window_sampling_support_invalid"
    elif y.ndim != 1 or y.size < 4 or not np.isfinite(y).all() or y.size != _number(row.get("sample_count")):
        problem = "noise_window_uniform_samples_invalid"
    elif not _positive(frequency) or frequency >= fs / 2:
        problem = "peak_outside_noise_window_frequency_support"
    if problem:
        output["noise_window_status"] = problem
        return output
    weights = hann(len(y), sym=False)
    enbw = fs * np.sum(weights**2) / weights.sum()**2
    power = evaluate_frequency_power(y, fs, frequency)
    output.update(noise_enbw_hz=float(enbw), noise_effective_fft_duration_s=float(len(y) / fs),
                  noise_samples_per_peak_cycle=float(fs / frequency))
    if not np.isfinite(power) or power < 0:
        output["noise_window_status"] = "noise_window_power_invalid"
    else:
        output.update(background_power=power, noise_window_status="computed")
    return output


def build_same_frequency_background(summary, references, config):
    """Return comparison and per-window audit tables without modifying inputs."""
    settings, width, tolerance = _settings(config)
    if not settings.get("enabled", True) or summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    names = config.get("snr", {}).get("noise_files", {}).get("IM", [])
    if (not isinstance(names, (list, tuple)) or any(not isinstance(n, str) or not n for n in names)
            or len(names) != len(set(names))):
        raise ValueError("snr.noise_files.IM must contain distinct explicit file IDs")
    required = {"sensor", "role", "window_variant"}
    if not required.issubset(summary):
        raise ValueError(f"Missing window metadata: {sorted(required - set(summary))}")
    sources = summary.loc[summary.sensor.eq("IM") & summary.role.eq("sweep") & summary.window_variant.eq("primary")]
    if "selected" in sources:
        sources = sources.loc[sources.selected.map(_true)]
    references = [r for r in references if r["row"].get("sensor") == "IM"
                  and r["row"].get("role") == "noise" and r["row"].get("window_variant") == "primary"
                  and ("selected" not in r["row"] or _true(r["row"]["selected"]))]
    comparisons, per_windows = [], []
    for source in sources.to_dict("records"):
        for name in names or [""]:
            frequency = _number(source.get("global_padded_parabolic_hz"))
            amplitude = _number(source.get("global_padded_parabolic_amplitude"))
            nominal = _number(source.get("target_hz"))
            row = {"excited_" + k: v for k, v in source.items()}
            row.update(sensor="IM", excited_file=_file(source), noise_file=name, noise_condition="",
                       segment_index=source.get("segment_index"), target_hz=nominal,
                       global_peak_hz=frequency, global_peak_amplitude=amplitude,
                       peak_power=np.nan, mean_background_power=np.nan, peak_background_power_ratio=np.nan,
                       peak_background_ratio_db=np.nan, window_s=width,
                       frequency_delta_hz=frequency - nominal, frequency_tolerance_hz=tolerance,
                       frequency_mismatch=bool(np.isfinite(frequency - nominal) and abs(frequency - nominal) > tolerance),
                       noise_window_count_planned=0, noise_window_count_available=0,
                       noise_enbw_min_hz=np.nan, noise_enbw_max_hz=np.nan, unused_tail_s=np.nan,
                       availability_status="unavailable", comparison_status="unavailable",
                       matched_acquisition_conditions_confirmed=False,
                       background_representativeness_confirmed=False, quality_flags="")
            flags = ["matched_acquisition_gain_filter_bandwidth_unverified", "independent_background_representativeness_unverified",
                     "global_peak_sinusoidal_component_assumption"]
            problems = []
            if row["frequency_mismatch"]:
                flags.append("global_peak_differs_from_nominal")
            if not _positive(nominal):
                problems.append("nominal_frequency_invalid")
            if _text(source.get("unit")) != "Data28 counts":
                problems.append("excited_documented_im_count_units_required")
            if not _true(source.get("fft_computed")):
                problems.append("excited_fft_unavailable")
            if _text(source.get("global_padded_interpolation_status")) != "ok":
                problems.append("global_padded_interpolation_unavailable")
            if not _positive(amplitude):
                problems.append("global_peak_amplitude_invalid")
            else:
                with np.errstate(over="ignore", under="ignore"):
                    row["peak_power"] = float(np.float64(amplitude)**2 / 2)
                if not _positive(row["peak_power"]):
                    problems.append("global_peak_power_invalid")
            source_fs = _number(source.get("fs_hz"))
            if not _positive(source_fs) or not _positive(frequency) or frequency >= source_fs / 2:
                problems.append("peak_outside_excited_frequency_support")
            duration = _number(source.get("analysis_end_s")) - _number(source.get("analysis_start_s"))
            if not np.isclose(duration, width, rtol=0, atol=1e-8):
                problems.append("excited_window_duration_mismatch")
            if _text(source.get("quality_flags")):
                flags.append("excited_quality_flags_present")
            if not _true(source.get("strict_gap_check_passed")):
                flags.append("excited_strict_timestamp_check_not_passed")
            matches = [ref for ref in references if _file(ref["row"]) == name]
            reference = matches[0] if len(matches) == 1 else None
            powers, enbws = [], []
            if not name:
                problems.append("noise_reference_not_configured")
            elif reference is None:
                problems.append("noise_reference_missing" if not matches else "noise_reference_ambiguous")
            else:
                noise = reference["row"]
                row.update({"noise_" + k: v for k, v in noise.items()})
                row.update(noise_condition=_text(noise.get("noise_condition")),
                           unused_tail_s=reference["unused_tail_s"],
                           noise_window_count_planned=reference["planned_window_count"])
                if _text(noise.get("unit")) != "Data28 counts":
                    problems.append("noise_documented_im_count_units_required")
                if not np.isclose(_number(reference.get("window_s")), width, rtol=0, atol=1e-8):
                    problems.append("noise_reference_window_duration_mismatch")
                if not reference["planned_window_count"]:
                    problems.append("noise_no_complete_windows")
                if len(reference["windows"]) != reference["planned_window_count"]:
                    problems.append("noise_planned_windows_missing")
                for window in reference["windows"]:
                    original = window["row"]
                    detail = {"excited_" + k: v for k, v in source.items()}
                    detail.update({"noise_" + k: v for k, v in original.items()})
                    detail.update(excited_file=_file(source), noise_file=name, noise_condition=row["noise_condition"],
                                  segment_index=source.get("segment_index"), target_hz=nominal,
                                  evaluation_frequency_hz=frequency, unused_tail_s=reference["unused_tail_s"],
                                  noise_window_count_planned=reference["planned_window_count"])
                    estimate = _window_power(window, frequency, width)
                    detail.update(estimate)
                    window_flags = []
                    if _text(original.get("quality_flags")):
                        window_flags.append("noise_window_quality_flags_present")
                    if not _true(original.get("strict_gap_check_passed")):
                        window_flags.append("noise_window_strict_timestamp_check_not_passed")
                    if not np.isclose(_number(original.get("fs_hz")), source_fs, rtol=1e-9, atol=1e-9):
                        window_flags.append("signal_noise_digital_sampling_support_differs")
                    if estimate["noise_samples_per_peak_cycle"] < 4:
                        window_flags.append("noise_fewer_than_four_samples_per_peak_cycle")
                    if estimate["noise_window_status"] != "computed":
                        window_flags.append(estimate["noise_window_status"])
                    else:
                        powers.append(estimate["background_power"])
                        enbws.append(estimate["noise_enbw_hz"])
                    detail["quality_flags"] = ";".join(window_flags)
                    flags.extend(window_flags)
                    per_windows.append(detail)
                row["noise_window_count_available"] = len(powers)
                if enbws:
                    row.update(noise_enbw_min_hz=min(enbws), noise_enbw_max_hz=max(enbws))
                if len(powers) != reference["planned_window_count"]:
                    problems.append("noise_complete_window_set_unavailable")
                elif powers:
                    row["mean_background_power"] = float(np.mean(powers))
                    if not _positive(row["mean_background_power"]):
                        problems.append("mean_background_power_invalid_or_nonpositive")
            if not problems:
                with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                    ratio = row["peak_power"] / row["mean_background_power"]
                    db = 10 * np.log10(ratio)
                if _positive(ratio) and np.isfinite(db):
                    row.update(peak_background_power_ratio=float(ratio), peak_background_ratio_db=float(db),
                               availability_status="computed", comparison_status="computed_conditional")
                else:
                    problems.append("ratio_not_representable")
            if problems:
                row["comparison_status"] = problems[0]
                flags.extend(problems)
            row["quality_flags"] = ";".join(dict.fromkeys(flags))
            comparisons.append(row)
    return pd.DataFrame(comparisons), pd.DataFrame(per_windows)


def plot_same_frequency_background(table):
    """Separate control curves and actual global frequencies; no fitted curves."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    axes[0].set(title="IM global peak / same-frequency background spectral power\nSeparate noise recordings; no background subtraction",
                ylabel="Peak / mean background spectral power [dB]")
    axes[1].set(title="Actual global FFT peak frequency used", ylabel="Global peak frequency [Hz]",
                xlabel="Nominal excitation frequency [Hz]", yscale="log")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
    axes[0].axhline(0, color="0.6", lw=0.8)
    if table.empty:
        axes[0].text(0.5, 0.5, "No primary IM sweep / noise comparisons", transform=axes[0].transAxes, ha="center")
    else:
        for (source, noise), group in table.groupby(["excited_file", "noise_file"], sort=False):
            group = group.sort_values("target_hz")
            condition = _text(group["noise_condition"].iloc[0])
            label = f"{noise}: {condition}" if condition else str(noise)
            if table.excited_file.nunique() > 1:
                label = f"{source} vs {label}"
            axes[0].plot(group.target_hz, group.peak_background_ratio_db, "o-", ms=4, label=label)
        peaks = table.drop_duplicates(["excited_file", "segment_index", "target_hz"])
        for name, group in peaks.groupby("excited_file", sort=False):
            group = group.sort_values("target_hz")
            axes[1].plot(group.target_hz, group.global_peak_hz, "o-", ms=4, label=f"{name} global peak")
        nominal = pd.to_numeric(peaks.target_hz, errors="coerce")
        nominal = nominal.loc[np.isfinite(nominal) & nominal.gt(0)]
        if len(nominal):
            bounds = [nominal.min(), nominal.max()]
            axes[1].plot(bounds, bounds, "--", color="0.4", label="Nominal identity")
        for ax in axes:
            ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
