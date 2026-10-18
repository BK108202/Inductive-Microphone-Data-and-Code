"""Three sweep summary views, using existing estimators without reading raw data.

These helpers do not synchronize the two recordings or estimate a transfer
function. They retain invalid/missing pairs and the exact upstream measurements.
"""
from pathlib import PureWindowsPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = {
    "frequency": (
        "Estimated frequency is the global padded parabolic spectral peak, selected "
        "without conditioning on nominal frequency. No target or bin estimate is "
        "substituted when interpolation is unavailable. Error = estimated - nominal; "
        "relative error = 100 * error / nominal. Bias, MAE, RMSE and maximum absolute "
        "errors aggregate finite estimates per sensor, with total/estimated counts."
    ),
    "snr": (
        "Primary SNR = 10 log10[(P_excited - P_noise) / P_noise], defined only for "
        "positive excess and positive reference noise power. P values are existing "
        "PSD integrals over the same configured nominal band (default +/- 1 Hz), clipped to common "
        "frequency support. Measured ratio = 10 log10[P_excited / P_noise] is "
        "retained separately. Separate control recordings assume comparable, "
        "additive background; in-band interference is included. No artificial "
        "dB floor, regression, R-squared, or pooling of the two noise controls. "
        "Upstream tables document actual windows, preprocessing, and bandwidth."
    ),
    "reference": (
        "The two explicitly selected files are outer-joined one-to-one at equal "
        "nominal excitation frequencies. Ratio = IM FFT peak amplitude in Data28 "
        "counts / accelerometer FFT peak amplitude in g, using the same specified "
        "spectral estimator for both sensors. These are coherent-gain corrected "
        "one-sided peak amplitudes after mean removal and periodic Hann windowing; "
        "they are not RMS or peak-to-peak amplitudes. No synchronized transfer "
        "function, phase, coherence, or pulse pairing is calculated."
    ),
    "reference_guards": (
        "Finite positive amplitudes and proper units are required for the raw "
        "arithmetic counts/g ratio. Missing FFTs, invalid peak estimates, a peak "
        "more than the configured tolerance from nominal, an IM/reference peak "
        "difference above that tolerance, or repeated reference extrema at/above "
        "the configured occupancy threshold exclude the point from the response "
        "curve. The raw ratio is retained for audit. Frequency tolerance is an "
        "analysis compatibility guard, not a calibration or detection threshold. "
        "Unknown experimental comparability or accelerometer full scale leaves "
        "a descriptive, explicitly conditional ratio, not calibrated sensitivity."
    ),
    "timing": (
        "Computed spectra with failed strict timestamp-gap checks remain "
        "diagnostic in the CSV tables; interpolation over gaps can bias results. "
        "Main plots omit warning rings, warning legends and footer annotations "
        "by default (show_annotations=False); quality flags and metrics are retained. "
        "Fewer than four samples per target cycle is flagged as sparse sampling; "
        "high-frequency amplitudes may be biased by sampling and interpolation."
    ),
}


def _numbers(frame, name):
    return pd.to_numeric(frame.get(name, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _true(frame, name):
    values = frame.get(name, pd.Series(False, index=frame.index))
    return values.map(lambda value: value is True or isinstance(value, np.bool_) and bool(value))


def _sweep(summary):
    table = summary.copy()
    if "role" in table:
        table = table.loc[table["role"].eq("sweep")]
    if "window_variant" in table:
        table = table.loc[table["window_variant"].eq("primary")]
    return table.copy()


def build_frequency_accuracy(summary):
    """Return per-window global-peak errors and finite-value per-sensor metrics."""
    table = _sweep(summary)
    for required in ("sensor", "target_hz"):
        if required not in table:
            raise ValueError(f"Frequency summary requires {required}.")
    table["target_hz"] = _numbers(table, "target_hz")
    estimate = _numbers(table, "global_padded_parabolic_hz")
    eligible = (_true(table, "fft_computed") & np.isfinite(estimate) & estimate.gt(0)
                & np.isfinite(table["target_hz"]) & table["target_hz"].gt(0))
    table["estimated_frequency_hz"] = estimate.where(eligible)
    table["error_hz"] = table["estimated_frequency_hz"] - table["target_hz"]
    table["relative_error_percent"] = 100 * table["error_hz"] / table["target_hz"]
    table["timing_warning"] = _true(table, "fft_computed") & ~_true(table, "strict_gap_check_passed")
    table["frequency_estimator"] = "global_padded_parabolic_hz"
    rows = []
    for sensor, part in table.groupby("sensor", sort=False):
        errors = part.loc[np.isfinite(part["error_hz"]), "error_hz"]
        relative = part.loc[np.isfinite(part["relative_error_percent"]), "relative_error_percent"]
        rows.append(dict(sensor=sensor, n_total=len(part), n_estimated=len(errors),
                         bias_hz=errors.mean(), mae_hz=errors.abs().mean(),
                         rmse_hz=np.sqrt((errors ** 2).mean()),
                         max_abs_error_hz=errors.abs().max(),
                         max_abs_relative_error_percent=relative.abs().max(),
                         n_timing_warnings=int(part["timing_warning"].sum())))
    columns = ["sensor", "n_total", "n_estimated", "bias_hz", "mae_hz", "rmse_hz",
               "max_abs_error_hz", "max_abs_relative_error_percent", "n_timing_warnings"]
    metrics = pd.DataFrame(rows, columns=columns)
    table.attrs["methods"] = {"frequency": METHODS["frequency"], "timing": METHODS["timing"]}
    return table.sort_values(["sensor", "target_hz"]).reset_index(drop=True), metrics


def _file_mask(table, stem):
    matches = pd.Series(False, index=table.index)
    # File stems are exact identifiers: IM_16 never matches IM_160.
    for column in ("file_label", "path"):
        if column in table:
            matches |= table[column].map(lambda value: PureWindowsPath(str(value)).stem == stem)
    return matches


def build_reference_response(summary, settings):
    """Retain every selected nominal frequency and guard the descriptive ratio."""
    settings = dict(settings)
    for key in ("IM_file", "accel_file"):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f"Reference response requires an explicit {key} stem.")
    amplitude_key = settings.get("amplitude_key", "global_padded_parabolic_amplitude")
    frequency_key = settings.get("frequency_key", "global_padded_parabolic_hz")
    valid_estimators = {
        "global_padded_parabolic_amplitude": "global_padded_parabolic_hz",
        "target_padded_parabolic_amplitude": "target_padded_parabolic_hz",
    }
    if valid_estimators.get(amplitude_key) != frequency_key:
        raise ValueError("Use a matching global or target padded parabolic amplitude/frequency pair; RMS is not interchangeable.")
    tolerance = float(settings.get("frequency_tolerance_hz", 1.0))
    threshold = float(settings.get("plateau_fraction_threshold", 0.01))
    if not np.isfinite(tolerance) or tolerance <= 0 or not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("Frequency tolerance must be positive and occupancy threshold in (0, 1].")
    table = _sweep(summary)
    for required in ("sensor", "target_hz"):
        if required not in table:
            raise ValueError(f"Reference summary requires {required}.")
    table["target_hz"] = _numbers(table, "target_hz")
    parts = []
    for prefix, sensor, key in (("im", "IM", "IM_file"), ("accel", "accel", "accel_file")):
        part = table.loc[table["sensor"].eq(sensor) & _file_mask(table, settings[key])].copy()
        if "path" in part and part["path"].dropna().nunique() > 1:
            raise ValueError(f"Ambiguous file stem for {settings[key]}: multiple input paths.")
        if not (np.isfinite(part["target_hz"]) & part["target_hz"].gt(0)).all():
            raise ValueError(f"{settings[key]} contains a missing or invalid nominal frequency.")
        if part["target_hz"].duplicated().any():
            raise ValueError(f"Duplicate nominal frequencies in {settings[key]}; one-to-one pairing required.")
        part[prefix + "_present"] = True
        part = part.rename(columns={c: prefix + "_" + c for c in part if c not in ("target_hz", prefix + "_present")})
        parts.append(part)
    result = parts[0].merge(parts[1], on="target_hz", how="outer", validate="one_to_one", sort=True)
    result["IM_file"] = settings["IM_file"]
    result["accel_file"] = settings["accel_file"]
    result["amplitude_key"] = amplitude_key
    result["frequency_key"] = frequency_key
    result["frequency_tolerance_hz"] = tolerance
    result["plateau_fraction_threshold"] = threshold
    result["conditions_confirmed"] = settings.get("conditions_confirmed", False) is True
    result["conditions_source"] = str(settings.get("conditions_source", ""))
    result["accel_full_scale_confirmed"] = settings.get("accel_full_scale_confirmed", False) is True
    invalid = pd.Series(False, index=result.index)
    flags = [[] for _ in range(len(result))]

    def flag(mask, name, excludes=False):
        nonlocal invalid
        mask = mask.fillna(True).astype(bool)
        for index in np.flatnonzero(mask.to_numpy()):
            flags[index].append(name)
        if excludes:
            invalid |= mask

    for prefix in ("im", "accel"):
        present = _true(result, prefix + "_present")
        flag(~present, prefix + "_missing_frequency", True)
        result[prefix + "_amplitude"] = _numbers(result, prefix + "_" + amplitude_key)
        result[prefix + "_frequency_hz"] = _numbers(result, prefix + "_" + frequency_key)
        amplitude = result[prefix + "_amplitude"]
        frequency = result[prefix + "_frequency_hz"]
        flag(~np.isfinite(amplitude) | amplitude.le(0), prefix + "_invalid_amplitude", True)
        flag(~_true(result, prefix + "_fft_computed"), prefix + "_fft_unavailable", True)
        flag(~np.isfinite(frequency) | frequency.le(0), prefix + "_invalid_peak_frequency", True)
        flag((frequency - result["target_hz"]).abs().gt(tolerance), prefix + "_peak_not_near_nominal", True)
        if amplitude_key.startswith("target_"):
            status = result.get(prefix + "_target_padded_status", pd.Series("missing", index=result.index))
            flag(~status.eq("available"), prefix + "_target_peak_unavailable", True)
        result[prefix + "_timing_warning"] = (_true(result, prefix + "_fft_computed")
                                               & ~_true(result, prefix + "_strict_gap_check_passed"))
        flag(result[prefix + "_timing_warning"], prefix + "_timing_warning")
        inherited = result.get(prefix + "_quality_flags", pd.Series("", index=result.index))
        flag(inherited.fillna("").astype(str).str.strip().ne(""), prefix + "_input_quality_flags_present")
        samples = _numbers(result, prefix + "_samples_per_target_cycle")
        result[prefix + "_sparse_sampling_warning"] = np.isfinite(samples) & samples.lt(4)
        flag(result[prefix + "_sparse_sampling_warning"], prefix + "_fewer_than_four_samples_per_cycle")
        rank = _numbers(result, prefix + "_target_padded_rank")
        flag(rank.gt(1), prefix + "_target_not_dominant")
    result["peak_frequency_difference_hz"] = (result["im_frequency_hz"] - result["accel_frequency_hz"]).abs()
    flag(result["peak_frequency_difference_hz"].gt(tolerance), "im_reference_frequency_mismatch", True)
    correct_units = (result.get("im_unit", pd.Series("", index=result.index)).eq("Data28 counts")
                     & result.get("accel_unit", pd.Series("", index=result.index)).eq("g"))
    flag(~correct_units, "units_not_counts_and_g", True)
    occupancy = _numbers(result, "accel_raw_extrema_occupancy_fraction")
    result["reference_plateau_flag"] = occupancy.ge(threshold)
    flag(result["reference_plateau_flag"], "reference_repeated_extrema", True)
    flag(~np.isfinite(occupancy) | occupancy.lt(0) | occupancy.gt(1), "reference_extrema_occupancy_unknown_or_invalid", True)
    amplitude_valid = (np.isfinite(result["im_amplitude"]) & result["im_amplitude"].gt(0)
                       & np.isfinite(result["accel_amplitude"]) & result["accel_amplitude"].gt(0))
    result["raw_counts_per_g"] = (result["im_amplitude"] / result["accel_amplitude"]).where(amplitude_valid & correct_units)
    flag(~result["conditions_confirmed"], "conditions_not_confirmed")
    flag(~result["accel_full_scale_confirmed"], "accel_full_scale_not_confirmed")
    result["timing_warning"] = result["im_timing_warning"] | result["accel_timing_warning"]
    result["sparse_sampling_warning"] = result["im_sparse_sampling_warning"] | result["accel_sparse_sampling_warning"]
    result["response_valid"] = ~invalid & np.isfinite(result["raw_counts_per_g"])
    result["counts_per_g"] = result["raw_counts_per_g"].where(result["response_valid"])
    result["quality_flags"] = [";".join(values) for values in flags]
    conditional_flags = {"conditions_not_confirmed", "accel_full_scale_not_confirmed"}
    result["measurement_warning"] = [any(value not in conditional_flags for value in values) for values in flags]
    result["response_status"] = np.where(result["response_valid"],
                                         np.where(result["quality_flags"].eq(""), "descriptive_ratio", "descriptive_ratio_with_warnings"),
                                         "excluded_from_response_curve")
    result["ratio_unit"] = "counts/g"
    result.attrs["methods"] = {key: METHODS[key] for key in ("reference", "reference_guards", "timing")}
    result.attrs["effective_settings"] = dict(settings, amplitude_key=amplitude_key, frequency_key=frequency_key,
                                              frequency_tolerance_hz=tolerance, plateau_fraction_threshold=threshold)
    return result.reset_index(drop=True)


def _axes(xlabel, ylabel, title):
    figure, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, which="both", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    return figure, ax


def _finish(figure, ax, footer, *, show_annotations=False):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=9)
    if show_annotations:
        figure.text(0.5, 0.025, footer, ha="center", va="bottom", fontsize=9)
        figure.tight_layout(rect=(0, 0.12, 1, 1))
    else:
        figure.tight_layout()
    return figure


def _warning_rings(ax, x, y, mask, label="Timestamp-gap warning", *, show_annotations=False):
    if not show_annotations:
        return
    mask = np.asarray(mask, bool) & np.isfinite(x) & np.isfinite(y)
    if mask.any():
        ax.scatter(np.asarray(x)[mask], np.asarray(y)[mask], facecolors="none", edgecolors="#c87500",
                   s=95, linewidths=1.4, zorder=5, label=label)


def plot_frequency_accuracy(table, metrics, *, show_annotations=False):
    """Plot independent global-peak estimates against nominal input and y=x."""
    figure, ax = _axes("Nominal excitation frequency (Hz)", "Estimated frequency (Hz)",
                       "Measured frequency versus nominal excitation")
    for sensor, part in table.groupby("sensor", sort=False):
        part = part.sort_values("target_hz")
        labels = part["file_label"].dropna().astype(str).unique() if "file_label" in part else []
        label = labels[0] if len(labels) == 1 else sensor
        is_im = sensor == "IM"
        ax.plot(part["target_hz"], part["estimated_frequency_hz"], marker="o" if is_im else "x",
                linestyle="-", label=label, markersize=6 if is_im else 5,
                markerfacecolor="none" if is_im else None, zorder=3)
        _warning_rings(ax, part["target_hz"], part["estimated_frequency_hz"], part["timing_warning"],
                       show_annotations=show_annotations)
    nominal = _numbers(table, "target_hz")
    nominal = nominal[np.isfinite(nominal)]
    if len(nominal):
        lower, upper = min(0, nominal.min()), nominal.max() * 1.03
        ax.plot([lower, upper], [lower, upper], "--", color="0.35", linewidth=1.2, label="y = x", zorder=1)
    n_estimated = int(metrics["n_estimated"].sum()) if "n_estimated" in metrics else 0
    warnings = int(_true(table, "timing_warning").sum())
    errors = []
    for metric in metrics.to_dict("records"):
        if metric.get("n_estimated", 0):
            errors.append(f"{metric['sensor']}: MAE {metric['mae_hz']:.3g} Hz, max |error| {metric['max_abs_error_hz']:.3g} Hz")
    footer = (f"Global spectral peak; {n_estimated}/{len(table)} estimates. "
              f"Orange rings: {warnings} timestamp-gap warnings.\n"
              + "; ".join(errors))
    return _finish(figure, ax, footer, show_annotations=show_annotations)


def plot_main_snr(snr_table, sensor="IM", *, show_annotations=False):
    """Plot excess-power SNR with both separate noise references and no floor."""
    figure, ax = _axes("Excitation frequency (Hz)", "SNR (dB)", f"{sensor}: nominal-band excess-power SNR")
    ax.set_xscale("log")
    if snr_table.empty:
        return _finish(figure, ax, "No SNR comparisons available (SNR disabled or no selected inputs).",
                       show_annotations=show_annotations)
    table = snr_table.loc[snr_table["sensor"].eq(sensor)].copy()
    if table.empty:
        return _finish(figure, ax, f"No SNR comparisons available for {sensor}.",
                       show_annotations=show_annotations)
    for (noise_file, noise_condition), part in table.groupby(["noise_file", "noise_condition"], sort=False, dropna=False):
        part = part.sort_values("target_hz")
        x, y = _numbers(part, "target_hz"), _numbers(part, "snr_db")
        # NaN is retained in the line to interrupt unsupported values, not replaced.
        y = y.where(np.isfinite(y))
        ax.plot(x, y, "o-", markersize=4, label=f"{PureWindowsPath(str(noise_file)).stem}: {noise_condition}")
        _warning_rings(ax, x, y, _true(part, "either_timing_warning"), show_annotations=show_annotations)
    undefined = int((~np.isfinite(_numbers(table, "snr_db"))).sum())
    warnings = int(_true(table, "either_timing_warning").sum())
    bandwidth = _numbers(table, "effective_bandwidth_hz")
    finite = bandwidth[np.isfinite(bandwidth)]
    band_text = f"Actual bandwidth: {finite.min():g}–{finite.max():g} Hz." if len(finite) else "Actual bandwidth is recorded upstream."
    halfwidths = _numbers(table, "requested_band_halfwidth_hz")
    halfwidths = halfwidths[np.isfinite(halfwidths)].unique()
    if len(halfwidths) == 1:
        nominal_band = f"Nominal ±{halfwidths[0]:g} Hz"
    elif len(halfwidths):
        nominal_band = f"Nominal half-width {halfwidths.min():g}–{halfwidths.max():g} Hz"
    else:
        nominal_band = "Configured nominal band"
    footer = (f"{nominal_band}, clipped to common support. {band_text}\n"
              f"Orange rings: {warnings} timestamp-gap warnings; {undefined} undefined values omitted without a dB floor.")
    return _finish(figure, ax, footer, show_annotations=show_annotations)


def plot_reference_response(table, *, show_annotations=False):
    """Connect valid ratios, retaining gaps and visibly identifying exclusions."""
    table = table.sort_values("target_hz")
    figure, ax = _axes("Excitation frequency (Hz)", r"$A_{\rm IM}/a_{\rm ref}$ (counts/g)",
                       "Reference-normalized spectral amplitude")
    ax.set_xscale("log")
    x, y = _numbers(table, "target_hz"), _numbers(table, "counts_per_g")
    label = (f"{table.iloc[0]['IM_file']} / {table.iloc[0]['accel_file']}" if len(table) else "Selected pair")
    ax.plot(x, y, "o-", markersize=4, label=label)
    _warning_rings(ax, x, y, _true(table, "measurement_warning"), label="Timestamp-gap / quality warning",
                   show_annotations=show_annotations)
    raw = _numbers(table, "raw_counts_per_g")
    excluded = ~_true(table, "response_valid") & np.isfinite(raw)
    if excluded.any():
        ax.scatter(x[excluded], raw[excluded], marker="x", color="0.45", s=40,
                   label="Excluded arithmetic ratio", zorder=4)
    caveats = []
    if len(table) and not _true(table, "conditions_confirmed").all():
        caveats.append("excitation comparability unconfirmed")
    if len(table) and not _true(table, "accel_full_scale_confirmed").all():
        caveats.append("accelerometer full scale unconfirmed")
    missing = int((~np.isfinite(raw)).sum())
    excluded_count = int((~_true(table, "response_valid")).sum())
    note = ("Conditional ratio: " + "; ".join(caveats) + ".") if caveats else "Descriptive amplitude ratio; no synchronized transfer function."
    footer = (note + "\n" + f"{excluded_count} points excluded from curve ({missing} without a finite ratio). "
              "Orange rings: timestamp-gap / quality warnings.")
    sparse_count = int(_true(table, "sparse_sampling_warning").sum())
    if sparse_count:
        footer += (f"\n{sparse_count} pairs have <4 samples/cycle: high-frequency amplitudes may be biased "
                   "by sparse sampling/interpolation.")
    return _finish(figure, ax, footer, show_annotations=show_annotations)
