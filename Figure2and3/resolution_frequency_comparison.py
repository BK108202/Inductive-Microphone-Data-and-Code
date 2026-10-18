"""Seven frequency estimators on identical retained resolution windows.

FFT methods use notebook-supplied spectra and peak selection. The independent
sine fit uses original recorded times/values and never the nominal frequency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from resolution_sine_fit import fit_sine_frequency


RESOLUTION_METHODS = (
    ("global_bin", "Largest FFT\nnative bin"),
    ("global_3point", "Native\n3-point fit"),
    ("global_5point", "Native\n5-point fit"),
    ("global_padded", "Largest FFT\npadded bin"),
    ("global_padded_3point", "Padded\n3-point fit"),
    ("global_padded_5point", "Padded\n5-point fit"),
    ("raw_timestamp_sine", "Original timestamps\nsine fit"),
)

DEFAULT_SINE_FIT = {
    "search_halfwidth_native_bins": 2.0,
    "coarse_points": 17,
    "frequency_tolerance_hz": 1e-9,
    "timestamp_basis": "recorded Time_us; relation to actual acquisition instants unconfirmed",
}

COMPARISON_METHODS = {
    "primary_frequency_estimator": "global_padded_3point; not selected by proximity to nominal",
    "largest_fft": "Actual global argmax over all positive native rFFT bins, excluding DC; preserved as global_bin_hz. global_padded_hz repeats the full search on the padded grid.",
    "fft_comparison": "Compute both three-point and five-point unweighted quadratic fits on the same native and padded Hann amplitude spectra, in the configured log/linear domain; retain all results and rejection statuses. Existing global_parabolic aliases use configured interpolation_points (default 3).",
    "sine_fit": "Unweighted variable-projection least squares of c+a*cos(2*pi*f*(t-tcenter))+b*sin(2*pi*f*(t-tcenter)) on ALL original retained samples. No resampling, Hann weighting, filtering, outlier removal, or nominal-target constraint. The search is centered on the largest native FFT bin with configured halfwidth in native bins; a coarse frequency grid selects a local bracket for bounded refinement. Frequency is unavailable on a rejected fit; candidate diagnostics retained.",
    "sine_amplitude": "sqrt(a*a+b*b), a fitted sinusoidal peak amplitude in original signal units, distinct from FFT peak amplitude and raw RMS/peak-to-peak. Offset fitted jointly. Residual RMS includes model mismatch, harmonics and noise; it is not a measured noise floor or SNR. No calibrated uncertainty inferred.",
    "time_assumption": "Raw timestamp fit conditionally treats recorded Time_us as acquisition times. That relationship is not documented; receive/logging timestamps need not be accurate sample times. Agreement with FFT does not calibrate a shared timebase or prove generator frequency.",
    "strict_view": "The same existing strict-gap eligibility mask is applied to every estimator only to compare a common subset; nonuniform-time sine fitting does not intrinsically require uniform intervals.",
    "comparison_aggregation": "Each record/window remains separate. Frequency error means estimate minus nominal setting, not independently measured truth; ppm=1e6*(estimate-target)/target. Group means and sample SD use independent records only; 90s and 100s windows are not repeats. Mean offsets/ppm are descriptive and are never subtracted as a calibration.",
}


def sine_fit_settings(settings=None):
    supplied = {} if settings is None else dict(settings)
    if set(supplied) - set(DEFAULT_SINE_FIT):
        raise ValueError(f"Unknown sine_fit settings: {sorted(set(supplied) - set(DEFAULT_SINE_FIT))}")
    effective = {**DEFAULT_SINE_FIT, **supplied}
    width, tolerance, count = (effective[name] for name in
                              ("search_halfwidth_native_bins", "frequency_tolerance_hz", "coarse_points"))
    if isinstance(width, bool) or not np.isfinite(width) or width <= 0:
        raise ValueError("sine_fit.search_halfwidth_native_bins must be positive")
    if isinstance(tolerance, bool) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("sine_fit.frequency_tolerance_hz must be positive")
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 5 or count % 2 != 1:
        raise ValueError("sine_fit.coarse_points must be an odd integer >=5")
    if not isinstance(effective["timestamp_basis"], str) or not effective["timestamp_basis"].strip():
        raise ValueError("sine_fit.timestamp_basis needs an explicit assumption")
    return effective


def add_frequency_estimators(row, spectrum, settings, peak_function):
    """Return extra row fields; use only spectra/sample arrays, not input files."""
    settings = sine_fit_settings(settings)
    out = {"primary_frequency_estimator": "global_padded_3point",
           "sine_fit_timestamp_basis": settings["timestamp_basis"],
           "sine_minus_padded_3point_hz": np.nan,
           "padded_5point_minus_3point_hz": np.nan,
           "native_5point_minus_3point_hz": np.nan}
    extra = [name for name, _ in RESOLUTION_METHODS if name not in ("global_bin", "global_padded")]
    for name in extra:
        out.update({name + "_hz": np.nan, name + "_amplitude": np.nan,
                    name + "_error_hz": np.nan, name + "_status": "fft_unavailable"})
        for suffix in ("_hz", "_amplitude", "_error_hz"):
            out["strict_" + name + suffix] = np.nan
    if not row.get("fft_computed", False):
        return out
    for prefix, fkey, akey in (("global", "f", "amplitude"),
                                ("global_padded", "padded_f", "padded_amplitude")):
        for points in (3, 5):
            name = f"{prefix}_{points}point"
            fit = peak_function(spectrum[fkey], spectrum[akey], row["interpolation_scale"], points)
            out[name + "_hz"] = fit["parabolic_hz"]
            out[name + "_amplitude"] = fit["parabolic_amplitude"]
            out[name + "_status"] = fit["interpolation_status"]
    # The nominal target is deliberately absent from this call and its bounds.
    seed = row.get("global_bin_hz", np.nan)
    halfwidth = settings["search_halfwidth_native_bins"] * row["native_df_hz"]
    if np.isfinite(seed) and seed > 0:
        fit = fit_sine_frequency(spectrum["t"], spectrum["y"], seed, halfwidth,
                                 coarse_points=settings["coarse_points"],
                                 frequency_tolerance_hz=settings["frequency_tolerance_hz"])
        for key, value in fit.items():
            key = {"frequency_hz": "hz", "amplitude_peak": "amplitude"}.get(key, key)
            if isinstance(value, list):
                value = ";".join(str(item) for item in value)
            out["raw_timestamp_sine_" + key] = np.nan if value is None else value
        if fit["status"] != "ok":
            out["raw_timestamp_sine_candidate_amplitude"] = out["raw_timestamp_sine_amplitude"]
            out["raw_timestamp_sine_amplitude"] = np.nan
    else:
        out["raw_timestamp_sine_status"] = "no_valid_fft_seed"
    target = row.get("target_hz", np.nan)
    for name in extra:
        out[name + "_error_hz"] = out[name + "_hz"] - target
        if row.get("strict_gap_check_passed", False):
            for suffix in ("_hz", "_amplitude", "_error_hz"):
                out["strict_" + name + suffix] = out[name + suffix]
    out["sine_minus_padded_3point_hz"] = out["raw_timestamp_sine_hz"] - out["global_padded_3point_hz"]
    out["padded_5point_minus_3point_hz"] = out["global_padded_5point_hz"] - out["global_padded_3point_hz"]
    out["native_5point_minus_3point_hz"] = out["global_5point_hz"] - out["global_3point_hz"]
    return out


def frequency_estimator_table(resolution):
    """One row per record/window/method; nominal error never used to select a fit."""
    records = []
    for row in resolution.to_dict("records"):
        for method, label in RESOLUTION_METHODS:
            hz = row.get(method + "_hz", np.nan)
            target = row["target_hz"]
            if method in ("global_bin", "global_padded"):
                status = "available" if np.isfinite(hz) else "unavailable"
            else:
                status = row.get(method + "_status", "unavailable")
            records.append({**{key: row.get(key) for key in
                               ("sensor", "path", "file_label", "base_hz", "target_hz", "window_variant",
                                "analysis_start_s", "analysis_end_s", "sample_count", "native_df_hz",
                                "padded_df_hz", "dt_cv", "gap_interval_count", "strict_gap_check_passed", "quality_flags")},
                            "method": method, "method_label": label.replace("\n", " "),
                            "estimate_hz": hz, "nominal_error_mhz": (hz - target) * 1000,
                            "nominal_error_ppm": (hz - target) / target * 1e6,
                            "status": status,
                            "interpretation": "error relative to nominal setting; not calibrated truth"})
    return pd.DataFrame(records)


def plot_estimator_errors(table, variant):
    """Display absolute nominal errors and method agreement without correcting them."""
    selected = table[table["window_variant"] == variant]
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    palette = plt.get_cmap("tab10")
    for i, (method, label) in enumerate(RESOLUTION_METHODS):
        part = selected[selected["method"] == method]
        for ax, column in zip(axes, ("nominal_error_mhz", "nominal_error_ppm")):
            ax.scatter(part["target_hz"] + (i - 3) * .8, part[column],
                       label=label.replace("\n", " "), color=palette(i),
                       marker=("o", "s", "^", "v", "D", "P", "x")[i], s=35, alpha=.85)
    for ax in axes:
        ax.axhline(0, c="black", lw=1, alpha=.6)
        ax.set_xlabel("Nominal frequency group (Hz; horizontal offsets separate methods)")
        ax.set_xticks(sorted(selected["base_hz"].unique()))
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Estimate minus nominal setting (mHz)")
    axes[1].set_ylabel("Relative difference from nominal setting (ppm)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Frequency-estimator comparison | {variant}")
    fig.text(.02, .015, "Every record is shown; unavailable estimates are omitted. No timebase correction applied.\n"
             "Recorded timestamps are not independently calibrated; residual/nominal differences do not establish physical resolution.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .94))
    return fig


def plot_sine_fit_agreement(resolution, variant):
    part = resolution[resolution["window_variant"] == variant]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    for base, group in part.groupby("base_hz"):
        axes[0].scatter(group["target_hz"], group["sine_minus_padded_3point_hz"] * 1000, label=f"{base:g} Hz")
        axes[1].scatter(group["target_hz"], group["padded_5point_minus_3point_hz"] * 1000, label=f"{base:g} Hz")
    for ax in axes:
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("Nominal setting (Hz)")
        ax.grid(alpha=.25)
    axes[0].set_ylabel("Original-time sine fit minus padded 3-point (mHz)")
    axes[1].set_ylabel("Padded 5-point minus padded 3-point (mHz)")
    axes[0].legend()
    fig.suptitle(f"Method agreement | {variant}")
    fig.text(.02, .02, "Agreement is a processing check, not independent timebase calibration. All methods use the same recorded time scale.", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .94))
    return fig
