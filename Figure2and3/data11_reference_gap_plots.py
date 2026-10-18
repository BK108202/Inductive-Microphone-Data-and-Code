"""Additional reference-gap views; preserve strict results and use existing diagnostic estimates."""
from pathlib import Path
from hashlib import sha256
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from characterization_run import AnalysisRun
from data11_analysis import save_fig
from data11_harmonic_tracking import diagnostic_reference
from portable_runtime import ensure_environment, load_config, notebook_path as publication_notebook_path, project_root, source_paths

PROJECT = project_root(__file__)
METHOD = (
    'Additional diagnostic presentation of archived harmonic tracking. Strict source results are unchanged. '
    'Only accelerometer signal windows rejected by the timestamp-gap gate alone may supply diagnostic '
    'reference amplitudes. Keep original timestamps and archived quadrature-filter estimates; do not '
    'interpolate frequencies or invent samples. Retain trajectory support, sample/cycle, Nyquist, noise '
    'window, bandwidth, SNR threshold, minimum-record-count and reference-condition requirements. IM '
    'quality exclusions remain unchanged. Fill only strict-unavailable counts/g points that become '
    'eligible under this explicitly diagnostic reference rule. Existing strict-eligible ratios remain '
    'identical. Counts/g is the ratio of independent condition-mean fundamental peak amplitudes, not '
    'a paired/synchronized transfer function. Do not use diagnostic points to extend strict detectable '
    'intervals. Marked and unmarked views use exactly the same values and axes; marked views add hollow '
    'circles at recovered reference-gap points, unmarked views connect them with ordinary line segments. '
    'Other unavailable points remain NaN and break the line. Top amplitude and SNR panels reuse strict '
    'archived summaries. No signal/SNR estimator is rerun; parent estimator, noise, window, preprocessing, '
    'bandwidth, aggregation and dB definitions are included in parameters.json. Arial 17 pt PNG/SVG.'
)


def plot_views(run, summary, reference, config, figures):
    for group, members in config['comparison_groups'].items():
        for combined in (True, False):
            for marked in (True, False):
                if combined:
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
                    target = axes[2]
                else:
                    fig, target = plt.subplots(figsize=(12, 6))
                    axes = [target]
                any_marked = False
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                for material_index, material in enumerate(members):
                    p = summary[summary.material.eq(material)]
                    rr = reference[reference.material.eq(material)].sort_values('grid_index')
                    if not np.isfinite(rr.display_counts_per_g).any():
                        continue
                    color = colors[material_index % len(colors)]
                    target.plot(rr.frequency_hz, rr.display_counts_per_g, color=color, label=material)
                    special = rr[rr.reference_gap_diagnostic]
                    if marked and len(special):
                        target.plot(special.frequency_hz, special.display_counts_per_g,
                            linestyle='none', marker='o', markersize=9, markerfacecolor='white',
                            markeredgecolor=color, markeredgewidth=1.8,
                            label='_nolegend_')
                        any_marked = True
                    if combined:
                        axes[0].plot(p.frequency_hz, p.amplitude_peak_mean, color=color, label=material)
                        axes[0].fill_between(p.frequency_hz, p.amplitude_peak_mean-p.amplitude_peak_sd,
                            p.amplitude_peak_mean+p.amplitude_peak_sd, color=color, alpha=.13)
                        axes[1].plot(p.frequency_hz, p.snr_db_mean, color=color, label=material)
                if combined:
                    axes[0].set(ylabel='Fundamental peak (counts)', title=group + ': harmonic tracking')
                    axes[1].set(ylabel='Estimator-equivalent SNR (dB)', title='Amplifier-on noise')
                    axes[1].axhline(config['snr_threshold_db'], color='k', linestyle='--')
                target.set(xlabel='Data-derived model frequency (Hz)',
                    ylabel='Nominal responsivity (counts/g)', yscale='log',
                    title=group + ': reference-gap diagnostic')
                for ax in axes:
                    ax.set_xscale('log')
                    ax.legend()
                if any_marked:
                    handles, labels = target.get_legend_handles_labels()
                    handles.append(Line2D([], [], color='black', marker='o', linestyle='none',
                        markerfacecolor='white', markersize=9))
                    target.legend(handles, labels + ['Reference-gap points'])
                name = ('tracking_comparison_' if combined else 'tracking_responsivity_') + group
                name += '_reference_gap_' + ('marked' if marked else 'connected')
                save_fig(run, fig, name, figures)


def export_reference_gap_views(parent_run, notebook_path=None):
    ensure_environment()
    parent = Path(parent_run).resolve()
    parent_params = json.loads((parent/'parameters.json').read_text(encoding='utf-8'))
    config = parent_params['config'].copy()
    config['figure_font_size_pt'] = 17
    root = Path(load_config()['results_root'])
    names = ['run_status.json', 'parameters.json', 'inputs.json', 'input_hashes.json',
        'harmonic_tracking_records.csv', 'harmonic_tracking_summary.csv',
        'acceleration_normalized_responsivity.csv', 'detectable_frequency_intervals.csv']
    paths = [parent/name for name in names]
    inputs = [dict(path=str(p), role='archived_tracking_evidence', sha256=sha256(p.read_bytes()).hexdigest())
              for p in paths]
    scripts = source_paths(notebook_path)
    params = dict(config=config, parent_run=str(parent), methods=METHOD,
                  inherited_parent_methods=parent_params.get('methods', {}))
    with AnalysisRun(root, 'data11_reference_gap_views', params, inputs, scripts,
                     repository_path=PROJECT) as run:
        print('Reference-gap figures:', run.path, flush=True)
        run.write_text('methods.txt', METHOD)
        if json.loads((parent/'run_status.json').read_text())['status'] != 'completed':
            raise ValueError('Parent tracking run must be completed')
        detail = pd.read_csv(parent/'harmonic_tracking_records.csv')
        summary = pd.read_csv(parent/'harmonic_tracking_summary.csv')
        strict = pd.read_csv(parent/'acceleration_normalized_responsivity.csv')
        reference, candidates = diagnostic_reference(detail, config, strict)
        reference.to_csv(run.path/'responsivity_strict_and_diagnostic.csv', index=False)
        candidates.to_csv(run.path/'reference_gap_window_estimates.csv', index=False)
        # Preserve exact numerical context for the unchanged upper panels and strict range.
        for name in ['harmonic_tracking_summary.csv', 'detectable_frequency_intervals.csv']:
            (run.path/name).write_bytes((parent/name).read_bytes())
        figures = []
        plot_views(run, summary, reference, config, figures)
        run.finish(dict(figure_count=len(figures), diagnostic_reference_windows=len(candidates),
                        recovered_ratio_points=int(reference.reference_gap_diagnostic.sum())))
        return dict(output_path=str(run.path), figures=figures, reference=reference)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python data11_reference_gap_plots.py <completed tracking run>')
    export_reference_gap_views(sys.argv[1], publication_notebook_path('data11_frequency_range_analysis'))
