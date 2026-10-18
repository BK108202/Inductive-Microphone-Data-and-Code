"""Global IM tone amplitude versus each independent full-window AC noise RMS.

Consume existing window summaries only: no data reads, spectral recalculation,
unit conversion, frequency-local noise selection, or changes to in-band SNR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


AMPLITUDE_KEY = "global_padded_parabolic_amplitude"
FREQUENCY_KEY = "global_padded_parabolic_hz"

METHODS = {
    "metric": "IM global FFT peak / independent full-record AC background SNR; supplementary to nominal-band excess-power SNR.",
    "signal_estimator": "Use the existing all-strictly-positive-bin global maximum of the padded FFT and its three-point interpolated one-sided sinusoidal peak amplitude. Signal RMS = peak amplitude / sqrt(2), conditional on the selected component being a resolved sinusoid. No target-conditioned search or fallback estimator.",
    "signal_preprocessing": "Existing FFT pipeline: linearly interpolate N original samples across their retained timestamp span, remove mean, periodic Hann with coherent-gain amplitude normalization, configured zero padding and three-point interpolation scale. No additional filtering. Exact windows and settings are retained in excited_* fields.",
    "noise_estimator": "Use raw_rms = sqrt(mean((original retained noise samples - their mean)**2)) across the entire configured noise window (default full recording). No interpolation, Hann weighting, Welch averaging, frequency selection, outlier removal or filtering for this denominator; noise FFT availability is not required.",
    "windows": "One existing primary IM sweep window per nominal setting; one separately configured IM noise window per explicit noise_file. Actual and requested windows and sample counts are preserved. No matching of individual samples across recordings is assumed.",
    "bandwidth": "Noise RMS contains all AC fluctuations in its original recorded samples. Report each record's fs/2 as nominal digital support (DC removed), not calibrated analog bandwidth or an exact filter cutoff for irregular samples. No +/-1 Hz integration, common-band filtering, or fs-based RMS scaling. Different signal/noise sampling supports remain explicitly flagged diagnostics.",
    "linear_and_db": "amplitude_ratio_linear = (A_global/sqrt(2))/noise_rms; snr_power_linear = amplitude_ratio_linear**2; snr_db = 20*log10(amplitude_ratio_linear) = 10*log10(snr_power_linear). No subtraction of broadband noise power from the peak power; the peak estimate includes any unresolved background at that frequency.",
    "aggregation": "Separate value for every primary IM sweep window and each configured IM noise recording; no pooling, averaging, or fabricated uncertainty between controls/windows.",
    "frequency": "Report actual interpolated global frequency and its difference from nominal. frequency_tolerance_hz (default 1 Hz) flags disagreement only; it is not a search restriction, SNR integration band or noise bandwidth. A mismatched peak can be interference or a harmonic and is not called the nominal response.",
    "units": "Both source and reference must explicitly have unit Data28 counts. No conversion is inferred. Ratio is dimensionless; signal/noise RMS remain Data28 counts.",
    "quality": "Preserve all excited/noise metadata with prefixes, including timing gaps and strict FFT eligibility. Missing/ambiguous controls, invalid/nonpositive amplitudes or RMS, missing interpolation, unavailable signal FFT, unit mismatch, or invalid frequency give NaN ratios with explicit flags. No substitute estimator or artificial floor. Usable raw noise RMS does not require a valid noise FFT.",
    "assumptions": "All finite ratios are conditional diagnostics: matched acquisition gain/filter/analog bandwidth and representativeness of the independent background are unverified. Global maximum selection can favor interference/harmonics and bias weak-signal amplitude. Nothing-on and amplifier-on controls answer separate questions. Results do not establish synchronized transfer, target detection, calibrated sensitivity, detection limit, or sensor bandwidth.",
}


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _true(value):
    return value is not None and not pd.isna(value) and bool(value)


def _text(value):
    return "" if value is None or pd.isna(value) else str(value)


def _file(row):
    return _text(row.get("file_id")) or _text(row.get("file_label"))


def _positive(value):
    return np.isfinite(value) and value > 0


def build_global_peak_snr(summary, config):
    """Build IM ratios from window_results; inputs are never modified.

    Settings live in config['global_peak_snr']; references use the existing
    config['snr']['noise_files']['IM'] explicit IDs. Frequency tolerance only
    annotates the already selected global peak, never changes its selection.
    """
    settings = config.get("global_peak_snr", {})
    if not settings.get("enabled", True):
        return pd.DataFrame()
    tolerance = _number(settings.get("frequency_tolerance_hz", 1.0))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("global_peak_snr.frequency_tolerance_hz must be finite and nonnegative")
    names = config.get("snr", {}).get("noise_files", {}).get("IM", [])
    if (not isinstance(names, (tuple, list)) or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("snr.noise_files.IM must contain distinct explicit file IDs")
    if summary.empty:
        return pd.DataFrame()
    required = {"sensor", "role", "window_variant"}
    if not required.issubset(summary.columns):
        raise ValueError(f"Missing window metadata columns: {sorted(required - set(summary.columns))}")
    primary = summary.loc[summary["sensor"].eq("IM") & summary["window_variant"].eq("primary")]
    if "selected" in primary:
        primary = primary.loc[primary["selected"].map(_true)]
    sources = primary.loc[primary["role"].eq("sweep")]
    references = primary.loc[primary["role"].eq("noise")].to_dict("records")
    rows = []
    # Emit an unavailable row when references were omitted; do not invent a pairing.
    for source in sources.to_dict("records"):
        for name in names or [""]:
            row = {"excited_" + key: value for key, value in source.items()}
            row.update({"noise_" + key: np.nan for key in summary.columns})
            row.update(
                sensor="IM", excited_file=_file(source), file_label=_file(source),
                noise_file=name, noise_condition="", noise_path="",
                segment_index=source.get("segment_index", np.nan),
                target_hz=_number(source.get("target_hz")),
                amplitude_key=AMPLITUDE_KEY, frequency_key=FREQUENCY_KEY,
                global_peak_hz=_number(source.get(FREQUENCY_KEY)),
                global_peak_amplitude=_number(source.get(AMPLITUDE_KEY)),
                signal_rms=np.nan, noise_rms=np.nan, amplitude_ratio_linear=np.nan,
                snr_power_linear=np.nan, snr_db=np.nan, frequency_delta_hz=np.nan,
                frequency_tolerance_hz=tolerance, frequency_mismatch=False,
                excited_digital_support_high_hz=np.nan, noise_digital_support_high_hz=np.nan,
                digital_support_differs=False, availability_status="unavailable",
                snr_status="unavailable", quality_flags="", unit=_text(source.get("unit")),
                signal_estimator=METHODS["signal_estimator"],
                noise_estimator=METHODS["noise_estimator"],
                snr_definition=METHODS["linear_and_db"],
                bandwidth_definition=METHODS["bandwidth"],
                aggregation=METHODS["aggregation"],
                interpretation="conditional global-tone / full-record AC background ratio; not nominal-band SNR",
                matched_acquisition_conditions_confirmed=False,
                background_representativeness_confirmed=False,
            )
            flags = ["matched_acquisition_gain_filter_bandwidth_unverified",
                     "independent_background_representativeness_unverified",
                     "global_peak_sinusoidal_component_assumption"]
            problems = []
            matches = [ref for ref in references if _file(ref) == name]
            noise = matches[0] if len(matches) == 1 else None
            if not name:
                problems.append("noise_reference_not_configured")
            elif noise is None:
                problems.append("noise_reference_missing" if not matches else "noise_reference_ambiguous")
            if noise is not None:
                row.update({"noise_" + key: value for key, value in noise.items()})
                row.update(noise_condition=_text(noise.get("noise_condition")),
                           noise_path=_text(noise.get("path")), noise_rms=_number(noise.get("raw_rms")))
                if not _positive(row["noise_rms"]):
                    problems.append("noise_rms_invalid_or_nonpositive")
                if _text(source.get("unit")) != _text(noise.get("unit")):
                    problems.append("signal_noise_units_differ")
                elif _text(noise.get("unit")) != "Data28 counts":
                    problems.append("documented_im_count_units_required")
            if _text(source.get("unit")) != "Data28 counts":
                problems.append("documented_im_count_units_required")
            if not _true(source.get("fft_computed", False)):
                problems.append("excited_fft_unavailable")
            if _text(source.get("global_padded_interpolation_status")) != "ok":
                problems.append("global_padded_interpolation_unavailable")
            if not _positive(row["global_peak_amplitude"]):
                problems.append("global_peak_amplitude_invalid_or_nonpositive")
            if not _positive(row["global_peak_hz"]):
                problems.append("global_peak_frequency_invalid")
            if not _positive(row["target_hz"]):
                problems.append("nominal_frequency_invalid")
            else:
                row["frequency_delta_hz"] = row["global_peak_hz"] - row["target_hz"]
                row["frequency_mismatch"] = bool(np.isfinite(row["frequency_delta_hz"])
                                                  and abs(row["frequency_delta_hz"]) > tolerance)
                if row["frequency_mismatch"]:
                    flags.append("global_peak_differs_from_nominal")
            for prefix, original in (("excited", source), ("noise", noise)):
                if original is None:
                    continue
                fs = _number(original.get("fs_hz"))
                if _positive(fs):
                    row[prefix + "_digital_support_high_hz"] = fs / 2
                else:
                    flags.append(prefix + "_digital_sampling_support_unavailable")
                    if prefix == "excited":
                        problems.append("excited_sampling_support_unavailable")
                if _text(original.get("quality_flags")):
                    flags.append(prefix + "_quality_flags_present")
                if not _true(original.get("strict_gap_check_passed", original.get("fft_valid", False))):
                    flags.append(prefix + "_strict_timestamp_check_not_passed")
            sf, nf = row["excited_digital_support_high_hz"], row["noise_digital_support_high_hz"]
            if _positive(sf) and row["global_peak_hz"] >= sf:
                problems.append("global_peak_at_or_above_nyquist")
            if _positive(sf) and _positive(nf):
                row["digital_support_differs"] = not bool(np.isclose(sf, nf, rtol=1e-9, atol=1e-9))
                if row["digital_support_differs"]:
                    flags.append("signal_noise_digital_sampling_support_differs")
            if _positive(nf) and row["global_peak_hz"] >= nf:
                flags.append("global_peak_outside_noise_digital_support")
            if not problems:
                signal_rms = row["global_peak_amplitude"] / np.sqrt(2.0)
                with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                    ratio = np.float64(signal_rms) / row["noise_rms"]
                    power = ratio**2
                    db = 20 * np.log10(ratio)
                if not all(_positive(v) for v in (signal_rms, ratio, power)) or not np.isfinite(db):
                    problems.append("ratio_not_representable")
                else:
                    row.update(signal_rms=float(signal_rms), amplitude_ratio_linear=float(ratio),
                               snr_power_linear=float(power), snr_db=float(db),
                               availability_status="computed", snr_status="computed_conditional")
            if problems:
                row["snr_status"] = problems[0]
                flags.extend(problems)
            row["quality_flags"] = ";".join(dict.fromkeys(flags))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_global_peak_snr(table):
    """Show separate-control SNR curves and actual selected global frequencies."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    axes[0].set_title("IM global FFT peak / full-record AC background SNR\nConditional on independent noise controls")
    axes[0].set_ylabel("20 log10[(Aglobal / sqrt(2)) / noise RMS] [dB]")
    axes[0].axhline(0, color="0.6", lw=0.8)
    axes[1].set_title("Actual global FFT peak frequency used")
    axes[1].set_ylabel("Global peak frequency [Hz]")
    axes[1].set_xlabel("Nominal excitation frequency [Hz]")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.25)
    axes[1].set_yscale("log")
    if table.empty:
        axes[0].text(0.5, 0.5, "No primary IM sweep / noise comparisons", transform=axes[0].transAxes,
                     ha="center", va="center")
        fig.tight_layout()
        return fig
    for (excited_file, noise_file), group in table.groupby(["excited_file", "noise_file"], sort=False):
        group = group.sort_values("target_hz")
        x = pd.to_numeric(group["target_hz"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(group["snr_db"], errors="coerce").to_numpy(float)
        condition = _text(group["noise_condition"].iloc[0])
        label = f"{noise_file}: {condition}" if condition else str(noise_file)
        if table["excited_file"].nunique() > 1:
            label = f"{excited_file} vs {label}"
        valid_x = np.isfinite(x) & (x > 0)
        axes[0].plot(x[valid_x], np.where(np.isfinite(y[valid_x]), y[valid_x], np.nan), "o-", ms=4, label=label)
        mismatch = group["frequency_mismatch"].fillna(False).to_numpy(bool) & valid_x & np.isfinite(y)
        if mismatch.any():
            axes[0].scatter(x[mismatch], y[mismatch], marker="x", s=65, color="crimson", zorder=5)
    # Controls share the identical signal-frequency estimate; show it only once.
    peaks = table.drop_duplicates(["excited_file", "segment_index", "target_hz"])
    for name, group in peaks.groupby("excited_file", sort=False):
        group = group.sort_values("target_hz")
        x = pd.to_numeric(group["target_hz"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(group["global_peak_hz"], errors="coerce").to_numpy(float)
        keep = np.isfinite(x) & (x > 0) & np.isfinite(y) & (y > 0)
        axes[1].plot(x[keep], y[keep], "o-", ms=4, label=str(name) + " global peak")
        mismatch = group["frequency_mismatch"].fillna(False).to_numpy(bool) & keep
        if mismatch.any():
            axes[1].scatter(x[mismatch], y[mismatch], marker="x", s=65, color="crimson", zorder=5,
                            label="Global / nominal mismatch")
    targets = pd.to_numeric(peaks["target_hz"], errors="coerce")
    targets = targets.loc[np.isfinite(targets) & targets.gt(0)]
    if len(targets):
        bounds = [targets.min(), targets.max()]
        axes[1].plot(bounds, bounds, "--", color="0.4", label="Nominal identity")
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=8)
    unavailable = int((~np.isfinite(pd.to_numeric(table["snr_db"], errors="coerce"))).sum())
    differing = int(table["digital_support_differs"].fillna(False).sum())
    axes[0].text(0.01, 0.01, f"Unavailable comparisons: {unavailable}; differing digital supports: {differing}\n"
                 "Red crosses: global / nominal frequency mismatch. Windows and quality flags in CSV.",
                 transform=axes[0].transAxes, fontsize=8, va="bottom")
    fig.tight_layout()
    return fig
