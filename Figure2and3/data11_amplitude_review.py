"""Preview all configured amplitude candidates before selecting independent records."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from characterization_run import AnalysisRun
from data11_config import PROJECT, AMPLITUDE_CANDIDATES, default_config, amplitude_bounds, amplitude_analysis_bounds, amplitude_subwindow_bounds
from data11_analysis import load_record, window, tone_metrics, save_fig
from portable_runtime import ensure_environment, load_config, source_paths

METHOD = (
    'All-candidate review only; no triplet is selected and no cross-record response is fitted. '
    'Use the configured per-file volume schedule, not recording length: IM_15-17 20s/level; '
    'IM_53-56 30s/level. First-level start=0 remains an assumption unless overridden. '
    'Use configurable central windows (notebook default: 26s for 30s IM blocks). Split into complete nonoverlapping subwindows, 4s by default; report the trailing remainder omitted from stability only. The main amplitude estimator uses the entire central window. '
    'Plot the existing target-conditioned peak estimator near 100Hz (target_hz +/- '
    'target_halfwidth_hz): linear interpolation at timestamp-derived rate, mean removal, '
    'periodic Hann, padded FFT and log-amplitude three-point peak interpolation. '
    'Peak amplitude is in counts; it is distinct from raw RMS and peak-to-peak. '
    'No smoothing of amplitudes; gaps and unsupported windows remain flagged/exported. '
    'Time axis uses actual recording-relative window centers. Different dwell groups '
    'do not share volume timing; additional per-dwell panels show correct boundaries. '
    'Subwindows describe within-record stability, not independent experimental repeats. '
    'Source timestamps, ADC values and files are unchanged. PNG and editable SVG exports.'
)


def preview_config():
    c = default_config('amplitude')
    c['files'] = {'IM': [53,54,55,56], 'accel': []}
    c['central_window_s'] = 26.0
    c['max_files'] = {'IM': None, 'accel': None}
    return c


def run_preview(config, notebook_path):
    ensure_environment()
    c = deepcopy(config)
    from data11_analysis import manifest
    inputs = manifest(c)
    if any(item['sensor'] != 'IM' for item in inputs):
        raise ValueError('Candidate preview expects IM amplitude files only')
    root = Path(load_config()['results_root'])
    scripts = source_paths(notebook_path)
    rows, audits, failures, hashes, schedules, figures = [], [], [], [], [], []
    with AnalysisRun(root, 'data11_amplitude_candidate_preview',
                     dict(config=c, methods=METHOD), inputs, scripts, repository_path=PROJECT) as run:
        print('AMPLITUDE_PREVIEW_OUTPUT:', run.path, flush=True)
        run.write_text('methods.txt', METHOD+'\n')
        for item in inputs:
            name = item['file']
            print('Amplitude candidate:', name, flush=True)
            try:
                path = Path(item['path'])
                hashes.append(dict(path=str(path), sha256=sha256(path.read_bytes()).hexdigest()))
                record = load_record(item,c)
                audits.append(record['audit'])
                nominal = amplitude_bounds(c,name)
                retained = amplitude_analysis_bounds(c,name)
                for level,(a,b),(lo,hi) in zip(c['levels'],nominal,retained):
                    sub_bounds = amplitude_subwindow_bounds(lo,hi,c['subwindow_s'])
                    covered = len(sub_bounds)*c['subwindow_s']
                    schedules.append(dict(subwindow_count=len(sub_bounds),
                        stability_covered_duration_s=covered, stability_unused_tail_s=hi-lo-covered, file=name, level=level, nominal_start_s=a,
                        nominal_end_s=b, analysis_start_s=lo, analysis_end_s=hi,
                        analysis_duration_s=hi-lo, trim_start_s=lo-a, trim_end_s=b-hi))
                    for sa,sb in sub_bounds:
                        row = dict(file=name, sensor='IM', level=level, start_s=sa, end_s=sb,
                                   center_s=(sa+sb)/2, status='unavailable', target_peak=np.nan)
                        try:
                            t,y = window(record,sa,sb)
                            metrics,_ = tone_metrics(t,y,c,c['target_hz'])
                            row.update(metrics, target_peak=metrics['target_padded_fit_amplitude'],
                                       status=metrics['target_padded_status'])
                        except ValueError as exc:
                            row['reason']=str(exc)
                            failures.append(dict(file=name, level=level, start_s=sa, end_s=sb, reason=str(exc)))
                        rows.append(row)
            except (ValueError, OSError, KeyError, pd.errors.ParserError) as exc:
                failures.append(dict(file=name, reason=str(exc)))
            finally:
                pd.DataFrame(rows).to_csv(run.path/'amplitude_candidate_subwindows.csv',index=False)
                pd.DataFrame(audits).to_csv(run.path/'timestamp_audit.csv',index=False)
                pd.DataFrame(schedules).to_csv(run.path/'amplitude_candidate_windows.csv',index=False)
                run.write_json('input_hashes.json',hashes)
                run.write_json('unavailable_windows.json',failures)
        table = pd.DataFrame(rows)
        if table.empty or not np.isfinite(table.target_peak).any():
            raise ValueError('No available candidate peaks; see archived failures')
        names = [item['file'] for item in inputs]
        colors = dict(zip(names, plt.get_cmap('tab10').colors))
        def draw(ax, selected, boundaries=False):
            edges=set()
            for name in selected:
                part=table[table.file.eq(name)]
                ax.plot(part.center_s,part.target_peak,'o-',ms=3,lw=1.2,
                        color=colors[name],label=name)
                if boundaries:
                    edges.update(edge for pair in amplitude_bounds(c,name) for edge in pair)
            for edge in sorted(edges): ax.axvline(edge,color='0.6',lw=.8,alpha=.5)
            ax.set(xlabel='Recorded time (s)',ylabel='100 Hz peak amplitude (counts)')
            ax.spines[['top','right']].set_visible(False)
            ax.legend(frameon=False)
        fig,ax=plt.subplots(figsize=(13,5))
        draw(ax,names)
        ax.set_title(f"All candidates | {c['subwindow_s']:g} s subwindows in central {c['central_window_s']:g} s\n"
                     'Complete subwindows; trailing remainder recorded separately')
        save_fig(run,fig,'amplitude_all_candidates_stability',figures)
        # Group by the full configured boundary sequence, including explicit overrides.
        groups={}
        for name in names:
            groups.setdefault(tuple(tuple(x) for x in amplitude_bounds(c,name)),[]).append(name)
        fig,axes=plt.subplots(len(groups),1,figsize=(12,4.5*len(groups)),squeeze=False)
        for ax,(bounds,group) in zip(axes.flat,groups.items()):
            draw(ax,group,boundaries=True)
            durations=sorted({b-a for a,b in bounds})
            ax.set_title(f"Level duration {durations} s | within-record stability, not repeats")
        save_fig(run,fig,'amplitude_candidates_stability_by_schedule',figures)
        summary=table.groupby('file',sort=False).agg(
            requested_subwindows=('status','size'), finite_peaks=('target_peak',lambda x:int(np.isfinite(x).sum())))
        summary.to_csv(run.path/'candidate_summary.csv')
        run.update_parameters(dict(actual_windows=schedules, selection='pending user; no automatic selection'))
        run.finish(dict(candidate_count=len(inputs), subwindows=len(table), failures=len(failures),
                        selection='pending user', figures=len(figures)))
        return dict(output_path=str(run.path), figures=figures, summary=summary,
                    subwindows=table, windows=pd.DataFrame(schedules), failures=failures)
