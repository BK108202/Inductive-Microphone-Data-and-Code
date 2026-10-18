"""Resolution figures for the explicitly bounded global search; no target substitution."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from resolution_relative_summary import (
    build_relative_summary, plot_relative_frequency, plot_absolute_ppm,
    RELATIVE_SUMMARY_METHODS,
)

ABOVE_CUTOFF_METHODS = (
    ('global_above_cutoff_native_bin_hz', 'Largest FFT\nnative bin'),
    ('global_above_cutoff_native_fit_hz', 'Native\n3-point log fit'),
    ('global_above_cutoff_native_5point_hz', 'Native\n5-point log fit'),
    ('global_above_cutoff_padded_bin_hz', 'Largest FFT\npadded bin'),
    ('global_above_cutoff_padded_fit_hz', 'Padded\n3-point log fit'),
    ('global_above_cutoff_padded_5point_hz', 'Padded\n5-point log fit'),
    ('global_above_cutoff_sine_frequency_hz', 'Original timestamps\nsine fit'),
)
PRIMARY_METHOD = 'global_above_cutoff_padded_fit_hz'

PLOT_METHODS = {
    **RELATIVE_SUMMARY_METHODS,
    'selection': 'Every native/padded FFT bin strictly above config.global_search_min_hz is searched, up to the recorded-grid Nyquist. This is a peak-selection bound, not a high-pass filter. No upper target band, nominal-seeded search or fallback to target_* is used. Original f>0 global and common target-band estimators remain separate in resolution_records.csv.',
    'sine_search': 'The original-timestamp sine fit starts from the observed native maximum above the cutoff. Halfwidth=min(config.sine_search_halfwidth_bins*native_df, seed-nextafter(cutoff,+inf), Nyquist-seed); invalid/boundary fits are unavailable. This is a local refinement of a global FFT seed, not a global nonlinear optimization.',
    'spectral_overlay': 'Periodic-Hann, demeaned, linearly resampled one-sided peak spectra as in the main analysis. Each record is normalized to its own maximum WITHIN the displayed crop, for visualization only. All records are overlaid individually; no mean spectrum is implied. This crop does not select the estimator peak.',
    'estimator_plot': 'All seven estimators use the same retained file/window. Colors encode nominal settings; marker shapes and fixed small horizontal offsets distinguish repeat files. The plotted offset is estimate minus configured display base, not baseline-subtracted. Dashed lines are nominal settings, not calibrated truth. Unavailable estimates stay NaN and are counted in CSV.',
    'fit_example': 'The explicitly configured example file is used for both windows, never selected for best fit. Show three actual native FFT samples around the eligible maximum and a quadratic least-squares fit in log amplitude. Overlay padded FFT samples and native/padded three-point vertices. This illustrates interpolation, not extra physical frequency resolution.',
    'quality': 'Finite estimates are displayed without introducing any new quality gate. Gap flags and original estimator statuses are retained in the record table. SD bars are between-file sample SD, not a confidence interval or uncertainty propagated from the subtracted baseline. Window variants are not independent repeats.',
}

NOTEBOOK_EXPLANATION = r"""## Resolution supplements: global peak, f > 1 Hz

Search every FFT bin strictly above `CONFIG["global_search_min_hz"]` through Nyquist.
The default lower bound is **1 Hz**. This selects peaks; it is not a high-pass filter
or a 99-101 Hz target band. Stronger harmonics remain selected. The full positive-
frequency `global_*` and target-band `target_*` results remain separate in the CSVs.

1. **Relative frequency:** subtract the three-record 100 Hz baseline mean, separately
   for each window and estimator.
2. **Uncorrected ppm offset:** `1e6 * (estimate - nominal) / nominal`, without
   baseline subtraction or clock correction.
3. **Spectra and seven estimators:** normalize each spectrum within its display crop;
   retain every record's estimate in the adjacent panel.
4. **Three-point fit:** use IM_12 by default (first record at 100.005 Hz), showing
   native FFT samples, the quadratic log-amplitude fit, and zero-padded samples.

The four primary 50 s figures are displayed by default; corresponding trimmed 40 s
figures are also saved. Black error bars show between-record sample SD (ddof=1),
not confidence intervals. Baseline subtraction and spectral normalization affect
display only. Zero padding does not improve duration-limited physical resolution.
These are separate-record single-tone comparisons, not simultaneous two-tone tests.
"""


def notebook_cells():
    import nbformat
    return [
        nbformat.v4.new_markdown_cell(NOTEBOOK_EXPLANATION, metadata={'data11_resolution_plots': True}),
        nbformat.v4.new_code_cell('''# data11-cell: resolution_detail_figures
RESOLUTION_DISPLAY_WINDOW = "primary_50s"  # Or "trimmed_40s".
for figure_type in ("relative_frequency", "absolute_ppm", "spectra_estimators", "quadratic_example"):
    show_figures(f"resolution_{RESOLUTION_DISPLAY_WINDOW}_{figure_type}")
print("New figures and CSVs:", RESULTS["output_path"])
''', metadata={'data11_resolution_plots': True}),
    ]


def estimator_records(detail, raw_records, c):
    """Adapt current-run values to the existing relative-summary implementation."""
    table = []
    cutoff = c['global_search_min_hz']
    for row in detail.to_dict('records'):
        for method, label in ABOVE_CUTOFF_METHODS:
            value = float(row[method]) if pd.notna(row[method]) else np.nan
            prefix = method.rsplit('_', 1)[0]
            if method.endswith('_bin_hz'):
                status = 'available' if np.isfinite(value) else 'unavailable'
            elif method.endswith('_fit_hz'):
                status = row[prefix.removesuffix('_fit') + '_status']
            elif method.endswith('_frequency_hz'):
                status = row[prefix.removesuffix('_frequency') + '_status']
            else:
                status = row[prefix + '_status']
            table.append(dict(
                sensor='IM', base_hz=c['resolution_plot_base_hz'], target_hz=row['nominal_hz'],
                path=str(Path(raw_records[row['file']]['path']).resolve()), file_label=row['file'],
                window_variant=row['variant'], method=method,
                method_label=f'Global f > {cutoff:g} Hz | ' + label.replace('\n', ' '),
                estimate_hz=value, status=status, global_search_min_hz=cutoff,
                requested_start_s=row['requested_start_s'], requested_end_s=row['requested_end_s'],
                actual_start_s=row['actual_start_s'], actual_end_s=row['actual_end_s'],
                sample_count=row['samples'], native_df_hz=row['native_df_hz'],
                padded_df_hz=row['padded_df_hz'], enbw_hz=row['enbw_hz'],
                dt_cv=row['dt_cv'], max_gap_ratio=row['max_gap_ratio'],
                gap_interval_count=row['gap_interval_count'], gap_ok=row['gap_ok'],
            ))
    return pd.DataFrame(table)


def _summary_figure(records, groups, variant, *, relative, config=None):
    plot = plot_relative_frequency if relative else plot_absolute_ppm
    fig = plot(records, groups, variant, method=PRIMARY_METHOD)
    fig.set_size_inches(10.5, 7.0)
    first = ('A separate baseline mean is subtracted for each window/method; display only, not calibration.'
             if relative else 'ppm = 1e6 * (estimate - nominal) / nominal; no baseline subtraction or timebase correction.')
    fig.text(.025, .018, first + '\n'
             'Bars: between-record sample SD (ddof=1), not a confidence interval or baseline-reference uncertainty.\n'
             'n = finite estimates (valid/total if incomplete); window variants are not independent repeats.\n'
             'Every positive FFT bin above the stated cutoff is searched; no 99-101 Hz restriction.', fontsize=9)
    fig.tight_layout(rect=(0, .19, 1, .9))
    config = config or {}
    if relative and config.get('resolution_relative_square', False):
        side = float(config.get('resolution_relative_figure_size_in', 10.0))
        if not np.isfinite(side) or side < 7:
            plt.close(fig)
            raise ValueError('resolution_relative_figure_size_in must be finite and >= 7 inches')
        fig.set_size_inches(side, side)
        for ax in fig.axes:
            ax.set_box_aspect(1)
        # Stack legend entries to fit a square canvas at the paper font size.
        for legend in list(fig.legends):
            handles = legend.legend_handles
            labels = [text.get_text() for text in legend.get_texts()]
            legend.remove()
            fig.legend(handles, labels, loc='lower center', ncol=1, frameon=False)
        if fig._suptitle is not None:
            fig._suptitle.set_text(fig._suptitle.get_text().replace(' | ', '\n', 1))
        # save_fig applies the shared paper layout, reserves space for the
        # unchanged method notes, and exports square PNG/SVG without tight cropping.
    return fig


def _spectra_comparison(part, spectra, c):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.1), gridspec_kw={'width_ratios': [1, 1.5]})
    base = c['resolution_plot_base_hz']
    crop = c['resolution_spectrum_halfwidth_hz']
    variant = part.variant.iloc[0]
    spectrum_rows = []
    for group_index, (nominal, group) in enumerate(part.groupby('nominal_hz', sort=True)):
        color = plt.get_cmap('tab10')(group_index)
        group = group.sort_values('file', key=lambda s: s.str.extract(r'(\d+)', expand=False).astype(int))
        for repeat_index, row in enumerate(group.to_dict('records')):
            f, amp = spectra[row['file']]['padded']
            keep = (f >= base-crop) & (f <= base+crop)
            norm = float(np.max(amp[keep])) if keep.any() else np.nan
            normalized = amp[keep] / norm if norm > 0 else np.full(keep.sum(), np.nan)
            axes[0].plot(f[keep], normalized, color=color, lw=1.25, alpha=.8,
                         ls=('-', '--', ':')[repeat_index % 3],
                         label=f'{nominal:g} Hz ({len(group)} records)' if repeat_index == 0 else None)
            spectrum_rows.extend(dict(file=row['file'], variant=variant, nominal_hz=nominal,
                                      frequency_hz=ff, amplitude_peak_counts=aa,
                                      display_normalization_peak_counts=norm, normalized_amplitude=nn)
                                 for ff, aa, nn in zip(f[keep], amp[keep], normalized))
            offset = (repeat_index-(len(group)-1)/2) * .085
            values = [(row[key]-base)*1000 for key, _ in ABOVE_CUTOFF_METHODS]
            axes[1].scatter(np.arange(len(values)) + offset, values,
                            marker=('o', 's', '^')[repeat_index % 3], color=color, s=34, alpha=.85)
        axes[0].axvline(nominal, color=color, ls=':', alpha=.65)
        axes[1].axhline((nominal-base)*1000, color=color, ls=':', alpha=.65)
    axes[0].set(xlim=(base-crop, base+crop), xlabel='Frequency (Hz)',
                ylabel='Normalized Hann peak amplitude', title='Overlaid spectra from separate recordings')
    axes[0].legend(fontsize=9)
    axes[0].ticklabel_format(axis='x', style='plain', useOffset=False)
    labels = [label.replace('Padded', f"{c['fft_pad_factor']}x padded").replace('padded bin', f"{c['fft_pad_factor']}x padded bin")
              for _, label in ABOVE_CUTOFF_METHODS]
    axes[1].set(xticks=range(len(labels)), xticklabels=labels, ylabel=f'Estimated frequency - {base:g} Hz (mHz)',
                title='Every recording and frequency estimator')
    axes[1].tick_params(axis='x', labelsize=9)
    axes[1].legend(handles=[Line2D([], [], color='0.3', marker=m, ls='none', label=f'Repeat {i+1}')
                            for i, m in enumerate(('o', 's', '^'))], fontsize=9, loc='best')
    for ax in axes:
        ax.grid(alpha=.22)
    df_min, df_max = 1000*part.native_df_hz.min(), 1000*part.native_df_hz.max()
    fig.suptitle(f'{base:g} Hz frequency estimation | {variant}\nGlobal peak, f > {c["global_search_min_hz"]:g} Hz', fontsize=15)
    fig.text(.025, .022,
             'Left: each recording normalized within the displayed crop; the crop does not constrain peak selection.\n'
             f'Native bin spacing: {df_min:.3f}-{df_max:.3f} mHz; padding refines the grid only. Dotted lines: nominal settings.\n'
             'Sine fit: local refinement around the observed native global peak using original timestamps; no clock correction.\n'
             'Independent single-tone recordings, not a two-tone test. Quality flags and unavailable estimates remain in CSVs.', fontsize=9)
    fig.tight_layout(rect=(0, .14, 1, .9))
    return fig, spectrum_rows


def _quadratic_example(row, specs, c):
    f, amp = specs['native']
    fp, ap = specs['padded']
    cutoff = c['global_search_min_hz']
    selected = row['global_above_cutoff_native_bin_hz']
    fig, ax = plt.subplots(figsize=(12.5, 6))
    native_fit = row['global_above_cutoff_native_fit_hz']
    padded_fit = row[PRIMARY_METHOD]
    diagnostic_rows = []
    if np.isfinite(selected):
        i = int(np.argmin(abs(f-selected)))
        if i > 0 and i+1 < len(f) and row['global_above_cutoff_native_status'] == 'ok':
            ids = np.arange(i-1, i+2)
            df = f[1]-f[0]
            coefficients = np.polyfit(np.arange(-1, 2), np.log(np.maximum(amp[ids], np.finfo(float).tiny)), 2)
            fine = np.linspace(f[ids[0]], f[ids[-1]], 301)
            curve = np.exp(np.polyval(coefficients, (fine-selected)/df))
            keep = (fp >= fine[0]-df*1e-7) & (fp <= fine[-1]+df*1e-7)
            ax.plot(fp[keep], ap[keep], '.-', ms=5, label=f'{c["fft_pad_factor"]}x padded FFT samples')
            ax.scatter(f[ids], amp[ids], s=75, color='tab:green', zorder=4, label='3 actual native FFT bins')
            ax.plot(fine, curve, '--', color='tab:orange', label='Quadratic least-squares fit in log amplitude (3 native bins)')
            ax.axvline(native_fit, color='tab:orange', ls=':', label=f'Native 3-point peak: {native_fit:.8f} Hz')
            if np.isfinite(padded_fit):
                ax.axvline(padded_fit, color='tab:blue', ls=':', label=f'Padded 3-point peak: {padded_fit:.8f} Hz')
            for kind, ff, aa in [('native_bins', f[ids], amp[ids]), ('padded_bins', fp[keep], ap[keep]), ('log_quadratic_curve', fine, curve)]:
                diagnostic_rows.extend(dict(file=row['file'], variant=row['variant'], series=kind, frequency_hz=x,
                                            amplitude_peak_counts=y) for x, y in zip(ff, aa))
            ax.legend(fontsize=9)
        else:
            ax.text(.5, .5, 'Three-point fit unavailable at the selected peak; no target substitution.',
                    transform=ax.transAxes, ha='center', wrap=True)
    else:
        ax.text(.5, .5, 'No eligible FFT peak above the configured cutoff.', transform=ax.transAxes, ha='center')
    ax.set(xlabel='Frequency (Hz)', ylabel='Sinusoidal peak amplitude (recorded counts)',
           title=f'{Path(row["file"]).stem}: 3-point quadratic fit on the spectral curve | {row["variant"]}')
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.grid(alpha=.22)
    fig.text(.025, .022,
             f'Nominal setting: {row["nominal_hz"]:g} Hz. Native maximum above {cutoff:g} Hz: {selected:.8f} Hz; no target band.\n'
             'Example file chosen explicitly in config. Padding and interpolation refine estimates, not physical resolving power.', fontsize=10)
    fig.tight_layout(rect=(0, .11, 1, 1))
    return fig, diagnostic_rows


def create_resolution_figures(run, detail, raw_records, c, figures):
    from data11_analysis import save_fig, spectrum, window
    table = estimator_records(detail, raw_records, c)
    records, groups = build_relative_summary(table)
    records.to_csv(run.path/'resolution_relative_records.csv', index=False)
    groups.to_csv(run.path/'resolution_relative_summary.csv', index=False)
    run.update_parameters({'resolution_plot_methods': PLOT_METHODS})
    methods_file = run.path/'methods.txt'
    run.write_text('methods.txt', methods_file.read_text(encoding='utf-8') + '\n\n' +
                   '\n\n'.join('resolution_plots.' + k + ': ' + v for k, v in PLOT_METHODS.items()))
    plotted_spectra, examples, issues = [], [], []
    for variant, part in detail.groupby('variant', sort=False):
        prefix = 'resolution_' + variant + '_'
        for name, relative in [('relative_frequency', True), ('absolute_ppm', False)]:
            save_fig(run, _summary_figure(records, groups, variant, relative=relative, config=c), prefix+name, figures)
        specs = {}
        for row in part.to_dict('records'):
            t, y = window(raw_records[row['file']], row['requested_start_s'], row['requested_end_s'])
            specs[row['file']] = {name: spectrum(t, y, pad)[:2]
                                  for name, pad in [('native', 1), ('padded', c['fft_pad_factor'])]}
        fig, rows = _spectra_comparison(part, specs, c)
        plotted_spectra.extend(rows)
        save_fig(run, fig, prefix+'spectra_estimators', figures)
        example = part[part.file.eq(c['resolution_fit_example'])]
        if len(example) == 1:
            row = example.iloc[0].to_dict()
            fig, rows = _quadratic_example(row, specs[row['file']], c)
            examples.extend(rows)
            save_fig(run, fig, prefix+'quadratic_example', figures)
        else:
            issues.append(dict(variant=variant, file=c['resolution_fit_example'], reason='Configured example window unavailable; no substitute selected'))
    pd.DataFrame(plotted_spectra).to_csv(run.path/'resolution_displayed_spectra.csv', index=False)
    pd.DataFrame(examples).to_csv(run.path/'resolution_quadratic_example.csv', index=False)
    run.write_json('resolution_plot_issues.json', issues)
    return records, groups
