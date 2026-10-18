"""Descriptive frequency discrimination summaries; no frequency calibration.

Each input row describes one file, one analysis window, and one estimator.
Baseline means are calculated separately for each sensor, base frequency,
window variant, and method. The relative representation subtracts only that
mean; the estimates and their differences from nominal settings stay intact.
Error bars describe sample scatter (ddof=1), not uncertainty in a difference
of means. In particular, baseline-reference uncertainty is not removed by
displaying its mean at zero, and overlapping windows are never pooled as
independent repeats.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASELINE_ATOL_HZ = 1e-9
REFERENCE_KEYS = ["sensor", "base_hz", "window_variant", "method"]
GROUP_KEYS = REFERENCE_KEYS + ["target_hz"]
IDENTITY_KEYS = ["sensor", "path", "window_variant", "method"]
REQUIRED_COLUMNS = set(GROUP_KEYS + ["path", "file_label", "estimate_hz"])

RELATIVE_SUMMARY_METHODS = {
    "reference": "For each sensor/base_hz/window_variant/method, use the mean of finite estimates at target_hz equal to base_hz (absolute tolerance 1e-9 Hz, zero relative tolerance). Missing finite baseline leaves relative estimates unavailable; no reference fallback.",
    "relative_record": "1000*(estimate_hz-baseline_mean_hz), in mHz. All original estimates, statuses and timestamp quality flags are retained. Reference subtraction is a display/summary convention, not a frequency or timebase calibration.",
    "group_scatter": "Means and sample SD (ddof=1) use finite independent-file estimates for one target/method/window. SD is unavailable for fewer than two estimates. SD bars describe record scatter, not a confidence interval or propagated uncertainty in the baseline-referenced mean. Baseline SD and SEM remain explicit in CSV outputs.",
    "nominal_bias": "nominal_error_ppm=1e6*(estimate_hz-target_hz)/target_hz; target_hz is a nominal setting, not calibrated truth. Group ppm mean and sample SD are computed without baseline subtraction. Zero nominal target has unavailable ppm.",
    "replicates": "One row per sensor/file/window/method, with duplicate identities rejected. Window variants are summarized separately and never treated as independent records. No exclusions are introduced by this summary; original estimator/quality statuses remain available.",
}


def _finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    mean = float(np.mean(values)) if count else np.nan
    sd = float(np.std(values, ddof=1)) if count >= 2 else np.nan
    return count, mean, sd


def build_relative_summary(table):
    """Return (record table, group table) without modifying the supplied table.

    Input/output frequency estimates and reference statistics are in Hz.
    ``relative_record_mhz``, ``mean_relative_mhz``, ``sample_sd_mhz`` and
    ``nominal_shift_mhz`` are in mHz; nominal-error ppm is uncorrected.
    Finite estimates are summarized regardless of descriptive quality flags;
    estimator-rejected results must already be represented as unavailable.
    """
    missing = sorted(REQUIRED_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"Missing frequency-estimator columns: {missing}")
    records = table.copy(deep=True)
    if records[IDENTITY_KEYS].isna().any().any():
        raise ValueError("Record identity columns must not contain missing values")
    duplicate = records.duplicated(IDENTITY_KEYS, keep=False)
    if duplicate.any():
        identities = records.loc[duplicate, IDENTITY_KEYS].to_dict("records")
        raise ValueError(f"Duplicate file/window/method identities: {identities}")
    for column in ("base_hz", "target_hz", "estimate_hz"):
        records[column] = pd.to_numeric(records[column], errors="raise")
    if not np.isfinite(records[["base_hz", "target_hz"]].to_numpy(dtype=float)).all():
        raise ValueError("Base and nominal target frequencies must be finite")
    # A fresh index avoids ambiguous assignments with caller-supplied indices.
    records = records.reset_index(drop=True)
    if "method_label" not in records:
        records["method_label"] = records["method"]
    if "status" not in records:
        records["status"] = np.where(np.isfinite(records["estimate_hz"]), "available", "unavailable")
    records["is_baseline_setting"] = np.isclose(
        records["target_hz"], records["base_hz"], rtol=0, atol=BASELINE_ATOL_HZ)
    records["nominal_shift_mhz"] = (records["target_hz"] - records["base_hz"]) * 1000
    records["nominal_error_mhz"] = (records["estimate_hz"] - records["target_hz"]) * 1000
    with np.errstate(divide="ignore", invalid="ignore"):
        records["nominal_error_ppm"] = np.where(
            records["target_hz"] != 0,
            1e6 * (records["estimate_hz"] - records["target_hz"]) / records["target_hz"], np.nan)
    records["baseline_count"] = 0
    for column in ("baseline_mean_hz", "baseline_sd_hz", "baseline_sem_hz", "relative_record_mhz"):
        records[column] = np.nan
    records["relative_status"] = "missing_baseline"

    for _, group in records.groupby(REFERENCE_KEYS, sort=False, dropna=False):
        baseline = group.loc[group["is_baseline_setting"], "estimate_hz"]
        count, mean, sd = _finite_stats(baseline)
        records.loc[group.index, "baseline_count"] = count
        records.loc[group.index, "baseline_mean_hz"] = mean
        records.loc[group.index, "baseline_sd_hz"] = sd
        records.loc[group.index, "baseline_sem_hz"] = sd / np.sqrt(count) if count >= 2 else np.nan
        if count:
            valid = np.isfinite(group["estimate_hz"])
            records.loc[group.index, "relative_status"] = np.where(valid, "ok", "estimate_unavailable")
            records.loc[group.index[valid], "relative_record_mhz"] = (
                group.loc[valid, "estimate_hz"] - mean) * 1000

    summaries = []
    for keys, group in records.groupby(GROUP_KEYS, sort=True, dropna=False):
        row = dict(zip(GROUP_KEYS, keys))
        first = group.iloc[0]
        count, mean, sd = _finite_stats(group["estimate_hz"])
        relative_count, relative_mean, relative_sd = _finite_stats(group["relative_record_mhz"])
        _, ppm_mean, ppm_sd = _finite_stats(group["nominal_error_ppm"])
        is_baseline = bool(group["is_baseline_setting"].all())
        # The full reference group's mean relative to itself is exactly zero.
        if is_baseline and relative_count and count == first["baseline_count"]:
            relative_mean = 0.0
        status = "ok" if relative_count else (
            "missing_baseline" if not first["baseline_count"] else "estimate_unavailable")
        row.update({
            "method_label": first["method_label"],
            "is_baseline_setting": is_baseline,
            "nominal_shift_mhz": first["nominal_shift_mhz"],
            "record_count": len(group), "valid_count": count,
            "relative_valid_count": relative_count,
            "baseline_count": int(first["baseline_count"]),
            "baseline_mean_hz": first["baseline_mean_hz"],
            "baseline_sd_hz": first["baseline_sd_hz"],
            "baseline_sem_hz": first["baseline_sem_hz"],
            "mean_estimate_hz": mean, "sample_sd_hz": sd,
            "mean_relative_mhz": relative_mean, "sample_sd_mhz": relative_sd,
            "mean_nominal_error_ppm": ppm_mean,
            "sample_sd_nominal_error_ppm": ppm_sd,
            "group_status": status,
        })
        summaries.append(row)
    group_columns = GROUP_KEYS + [
        "method_label", "is_baseline_setting", "nominal_shift_mhz", "record_count",
        "valid_count", "relative_valid_count", "baseline_count", "baseline_mean_hz",
        "baseline_sd_hz", "baseline_sem_hz", "mean_estimate_hz", "sample_sd_hz",
        "mean_relative_mhz", "sample_sd_mhz", "mean_nominal_error_ppm",
        "sample_sd_nominal_error_ppm", "group_status",
    ]
    return records, pd.DataFrame(summaries, columns=group_columns)


def _plot_summary(records, groups, variant, method, *, relative):
    selected = records.loc[(records["window_variant"] == variant) & (records["method"] == method)]
    summarized = groups.loc[(groups["window_variant"] == variant) & (groups["method"] == method)]
    if selected.empty:
        raise ValueError(f"No records for window {variant!r} and method {method!r}")
    panels = sorted(selected[["sensor", "base_hz"]].drop_duplicates().itertuples(index=False, name=None),
                    key=lambda item: (str(item[0]), item[1]))
    columns = min(3, len(panels))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(5.4 * columns, 4.8 * rows + 1.35),
                             squeeze=False)
    point_column = "relative_record_mhz" if relative else "nominal_error_ppm"
    mean_column = "mean_relative_mhz" if relative else "mean_nominal_error_ppm"
    sd_column = "sample_sd_mhz" if relative else "sample_sd_nominal_error_ppm"
    count_column = "relative_valid_count" if relative else "valid_count"
    palette = plt.get_cmap("tab10")

    for ax, (sensor, base) in zip(axes.flat, panels):
        part = selected.loc[(selected["sensor"] == sensor) & (selected["base_hz"] == base)]
        stats = summarized.loc[(summarized["sensor"] == sensor) & (summarized["base_hz"] == base)]
        stats = stats.sort_values("target_hz")
        shifts = stats["nominal_shift_mhz"].to_numpy(dtype=float)
        positive_gaps = np.diff(np.unique(shifts))
        spacing = float(np.min(positive_gaps)) if len(positive_gaps) else 5.0
        margin = spacing * .28
        if relative:
            ax.plot(shifts, shifts, color="0.55", linestyle="--", linewidth=1.2,
                    label="Nominal increment", zorder=1)
        else:
            ax.axhline(0, color="0.55", linestyle="--", linewidth=1.2, zorder=1)
        tick_labels = []
        for index, (_, group) in enumerate(stats.iterrows()):
            at_target = part.loc[part["target_hz"] == group["target_hz"]]
            at_target = at_target.sort_values(["file_label", "path"], kind="stable")
            count = len(at_target)
            jitter = np.linspace(-.075, .075, count) * spacing if count > 1 else np.zeros(count)
            x = float(group["nominal_shift_mhz"])
            y = at_target[point_column].to_numpy(dtype=float)
            valid = np.isfinite(y)
            ax.scatter((x + jitter)[valid], y[valid], color=palette(index % 10), s=44,
                       alpha=.85, edgecolors="white", linewidths=.5, zorder=3)
            mean = group[mean_column]
            sd = group[sd_column]
            if np.isfinite(mean):
                ax.errorbar(x, mean, yerr=sd if np.isfinite(sd) else None, fmt="D",
                            color="black", markersize=5.5, capsize=6, elinewidth=1.4, zorder=4)
            else:
                ax.text(x, .035, "unavailable", ha="center", fontsize=8, color="0.4",
                        transform=ax.get_xaxis_transform())
            usable = int(group[count_column])
            # Zero nominal frequency cannot have a ppm estimate.
            if not relative and group["target_hz"] == 0:
                usable = 0
            total = int(group["record_count"])
            n_label = f"n={usable}" if usable == total else f"n={usable}/{total}"
            tick_labels.append(f"{x:g}\n{n_label}")
        baseline_count = int(stats["baseline_count"].iloc[0])
        title = f"{sensor} | {base:g} Hz"
        if relative:
            title += f" | baseline n={baseline_count}"
        ax.set_title(title, fontsize=11)
        ax.set_xticks(shifts, tick_labels)
        ax.set_xlim(float(np.min(shifts)) - margin, float(np.max(shifts)) + margin)
        ax.set_xlabel("Nominal increment from base setting (mHz)")
        ax.set_ylabel("Frequency relative to baseline mean (mHz)" if relative else
                      "Estimate minus nominal / nominal (ppm)")
        ax.grid(alpha=.22)
        ax.margins(y=.14)
        if relative and baseline_count == 0:
            ax.text(.5, .94, "No finite baseline estimate", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9, color="0.35")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    method_labels = selected["method_label"].dropna().astype(str).unique()
    method_label = method_labels[0] if len(method_labels) == 1 else method
    title = "Single-tone frequency discrimination" if relative else "Uncorrected frequency difference from nominal"
    fig.suptitle(f"{title} | {variant}\n{method_label}", fontsize=14, y=.985)
    legend = [Line2D([], [], marker="o", linestyle="none", color="tab:blue", label="Individual recording"),
              Line2D([], [], marker="D", linestyle="none", color="black", label="Mean ± sample SD")]
    if relative:
        legend.append(Line2D([], [], color="0.55", linestyle="--", label="Nominal increment"))
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .115), ncol=len(legend), frameon=False)
    fig.tight_layout(rect=(0, .18, 1, .90))
    return fig


def plot_relative_frequency(records, groups, variant, method="global_padded_3point"):
    """Plot each file and group mean ± sample SD after baseline subtraction."""
    return _plot_summary(records, groups, variant, method, relative=True)


def plot_absolute_ppm(records, groups, variant, method="global_padded_3point"):
    """Plot uncorrected nominal-frequency differences without subtraction."""
    return _plot_summary(records, groups, variant, method, relative=False)
