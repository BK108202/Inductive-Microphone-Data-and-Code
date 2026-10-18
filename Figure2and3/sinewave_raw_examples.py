"""Unprocessed Data28 waveform examples from an explicitly configured schedule."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

METHOD = (
    "Original IM Data28 counts including DC, plotted at recorded timestamps. "
    "No detrending, mean subtraction, filtering, smoothing, resampling, decimation "
    "or amplitude scaling. Marker-free straight lines connect recorded samples. "
    "Time is milliseconds relative to requested window start. Half-open windows "
    "start at configured trimmed sweep start plus display offset. Frequency labels "
    "are nominal schedule settings; switching times are not verified. Timestamp "
    "and samples-per-nominal-cycle diagnostics do not establish sensor bandwidth."
)


def save_raw_examples(run, figures, record, plan, effective, save_figure, audit_timestamps):
    spec = effective['plots']['raw_examples']
    if not spec['enabled'] or record['sensor'] != 'IM':
        return []
    path = Path(spec['file']).expanduser()
    if not path.is_absolute():
        path = Path(effective['data_root']) / path
    if Path(record['path']).resolve() != path.resolve():
        return []
    targets = spec['frequencies_hz']
    if not targets or len(set(targets)) != len(targets):
        raise ValueError('Raw examples need nonempty unique frequencies_hz')
    schedule = plan[(plan.sensor == 'IM') & (plan.role == 'sweep') & (plan.path == record['path'])]
    offset = float(spec['offset_from_analysis_start_s'])
    if not np.isfinite(offset) or offset < 0:
        raise ValueError('Raw example offset must be finite and nonnegative')
    summary, paths = [], []
    t, y = record['t'], record['raw_counts']
    for order, target in enumerate(targets, 1):
        target = float(target)
        matches = schedule[np.isclose(schedule.target_hz, target, rtol=0, atol=1e-10)]
        if len(matches) != 1 or not np.isfinite(target) or target <= 0:
            raise ValueError(f'Raw example {target:g} Hz needs one configured IM segment')
        segment = matches.iloc[0]
        overrides = spec.get('duration_ms_by_frequency', {})
        duration = float(overrides.get(target, overrides.get(str(target), spec['duration_ms'])))
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('Raw duration_ms must be positive and finite')
        start = float(segment.analysis_start_s) + offset
        end = start + duration / 1000
        if end > float(segment.analysis_end_s) + 1e-10 or start < t[0] or end > t[-1]:
            raise ValueError(f'Raw {target:g} Hz window exceeds configured/recorded support')
        indices = np.flatnonzero((t >= start) & (t < end))
        ts, values = t[indices], y[indices]
        audit = audit_timestamps(ts, values)
        fs = 1.0 / audit['dt_mean_s']
        flags = []
        if audit['max_gap_over_median'] > effective['fft']['max_gap_over_median']:
            flags.append('timestamp_gap')
        if audit['dt_cv'] > effective['fft']['dt_cv_warning']:
            flags.append('timestamp_jitter')
        if fs / target < 10:
            flags.append('fewer_than_10_samples_per_nominal_cycle')
        if target >= fs / 2:
            flags.append('nominal_at_or_above_estimated_nyquist')
        time_ms = (ts - start) * 1000
        name = f"raw_IM_{Path(record['path']).stem}_{order:02d}_{target:g}Hz"
        pd.DataFrame({'source_row_zero_based': indices,
            'Time_us': ts * 1e6 + record['metadata']['timestamp_origin_us'],
            'record_time_s': ts, 'window_time_ms': time_ms, 'Data28': values
        }).to_csv(run.path / (name + '.csv'), index=False)
        summary.append(dict(path=record['path'], nominal_frequency_hz=target,
            timing_source=segment.timing_source, requested_start_s=start, requested_end_s=end,
            duration_ms=duration, sample_count=len(indices), actual_start_s=float(ts[0]),
            actual_end_s=float(ts[-1]), estimated_fs_hz=fs, samples_per_nominal_cycle=fs/target,
            quality_flags=';'.join(flags), preprocessing='none; DC retained',
            timestamp_audit=audit, figure_name=name))
        pd.DataFrame([{k:v for k,v in row.items() if k != 'timestamp_audit'} for row in summary]
                     ).to_csv(run.path / 'raw_examples_summary.csv', index=False)
        run.update_parameters({'raw_examples_actual_windows': summary, 'raw_examples_method': METHOD})
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_box_aspect(1)
        ax.plot(time_ms, values, color=spec['color'], linewidth=spec['linewidth'],
                marker=None, solid_capstyle='round')
        ax.set(xlabel='Time (ms)', ylabel='Raw LDC output (counts)',
               title=f"{Path(record['path']).stem} | {target:g} Hz (nominal)", xlim=(0, duration))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=False, useMathText=False)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(direction='out', length=5, width=1)
        ax.margins(y=0.08)
        save_figure(run, figures, fig, name)
        paths.append(figures[-1])
        if flags:
            print(f"Raw {target:g} Hz quality flags: {'; '.join(flags)}", flush=True)
    return paths
