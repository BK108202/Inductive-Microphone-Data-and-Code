"""Read-only candidate review and editable time-domain examples for Data11."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from characterization_run import AnalysisRun
from data11_config import PROJECT, AMPLITUDE_CANDIDATES
from data11_analysis import load_record, local_quality, window, save_fig
from data11_harmonic_tracking import tracking_config, independent_track, tracking_filter, _sampling_ok
from portable_runtime import ensure_environment, load_config, notebook_path as publication_notebook_path, source_paths

METHOD = '''Candidate review, not a new SNR or sensitivity analysis. Original CSVs are read-only.
Time_us is converted to seconds relative to the first sample; Data28 remains counts.
All listed amplitude candidates are displayed over their entire duration. Overview figures
use contiguous sample-bucket minima/maxima, with no sample excursion discarded. One-second
RMS diagnostics subtract each bin mean, use original samples, and apply no filtering.
These RMS values are not fundamental amplitudes, event RMS, or volume-level estimates.
No amplitude triplet is selected automatically. User confirmed 20 s per level for IM_15-17
and 30 s per level for IM_53-56 on 2026-10-13. Full records are shown; t=0 alignment of
the first volume remains an analysis assumption for subsequent segmentation.
Sweep checks retain the existing endpoint, timestamp, trajectory and sampling gates.
The additional max adjacent jump / median absolute adjacent jump >100 flag is a conservative,
unvalidated review heuristic, not proof of clipping or a calibrated quality criterion.
Flagged records remain in diagnostics but require review before triplet recommendation.
Compatibility review also flags a fitted 30-to-300 Hz duration differing by >20% from
the documented 30 s duration. This is a conservative, unvalidated combination screen,
not a change to the existing tracker or independent proof of the excitation schedule.
Compare linear-time and log-time residuals on the already estimated local frequencies
as diagnostics only; partial track support can bias this comparison. Do not refit an
incompatible sweep into the old log model simply to fill a group.
Recommend up to three passing sweeps in ascending file-number order; never rank by signal
amplitude or SNR, never duplicate a file or silently fill a deficient group.
Frequency coverage is evaluated on the existing 81-point grid using the existing 0.8 s
tracking windows. A candidate must support at least two grid points. Detailed support is
exported; a passing record does not imply complete 30-300 Hz coverage or calibration.
Raw examples use the independently fitted log-frequency trajectory to locate requested
center frequencies. Each 200 ms window contains original timestamps and all recorded samples.
Both untouched Data28 and ordinary least-squares intercept+linear-time detrended versions
are exported; no smoothing, bandpass, resampling, or fitted sinusoid replaces the samples.
Sweep frequency changes across each example; requested frequency describes its center.
Timing, analog bandwidth, anti-aliasing and independent frequency verification remain unresolved.
PNG previews and SVGs use Arial 15 pt, editable text, no grids, <=5 major y tick values.
'''


def review_config():
    c = tracking_config()
    return dict(tracking=c, amplitude_ids=list(AMPLITUDE_CANDIDATES),
                raw_file='IM_3.csv', raw_frequencies_hz=[50., 100., 200.],
                raw_window_s=.2, jump_ratio_review_threshold=100., include_raw_examples=True,
                duration_relative_review_tolerance=.2,
                requested_sweep_count=3, amplitude_selected_ids=[],
                documented_sweep_window_s=[5., 35.],
                documented_sweep_spacing='User uncertain on 2026-10-13 whether recollected sweeps were Linear; keep data-model evidence separate from instrument settings.',
                documented_amplitude=dict(frequency_hz=100, volumes=[25,24,23,22,21,20],
                    dwell_s_by_file={f'IM_{i}.csv': (20 if i<53 else 30) for i in AMPLITUDE_CANDIDATES},
                    start_s=None, note='Dwell confirmed 2026-10-13; first-level start unresolved'))


def _envelope(ax, r):
    t, y = r['t'], r['y']
    edges = np.linspace(0, len(t), min(1800, len(t))+1, dtype=int)
    center = np.array([np.mean(t[a:b]) for a,b in zip(edges[:-1],edges[1:])])
    low = np.array([y[a:b].min() for a,b in zip(edges[:-1],edges[1:])])
    high = np.array([y[a:b].max() for a,b in zip(edges[:-1],edges[1:])])
    ax.vlines(center, low, high, lw=.6)
    ax.set(xlabel='Recorded time (s)', ylabel='Raw LDC output (counts)')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset=False, useMathText=True)


def _inputs(c):
    items = []
    for material, (_, ids) in c['tracking']['materials'].items():
        for i in ids:
            items.append(dict(file=f'IM_{i}.csv', sensor='IM', role='sweep_candidate', material=material))
    for i in c['amplitude_ids']:
        items.append(dict(file=f'IM_{i}.csv', sensor='IM', role='amplitude_candidate', material='OO-50'))
    for item in items:
        item['path'] = str((Path(c['tracking']['data_root'])/item['file']).resolve())
        p = Path(item['path'])
        item['sha256'] = sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    if len({x['path'] for x in items}) != len(items):
        raise ValueError('A file cannot be assigned to multiple tests')
    return items


def _raw_examples(run, record, meta, c, figures):
    rows, samples = [], []
    run.update_parameters({'raw_plot_style': dict(line_color='#1f77b4', line_width_pt=1.2,
        markers=False, scientific_notation='1eN', additive_offset=False,
        title='file stem | tracked center frequency',
        saved_versions=['raw with DC offset', 'linear detrended'])})
    half = c['raw_window_s']/2
    lower = max(meta['support_start_s'], record['t'][0])+half
    upper = min(meta['support_end_s'], record['t'][-1])-half
    supported = np.exp(meta['log_intercept']+meta['rate_per_s']*np.array([lower, upper]))
    for freq in c['raw_frequencies_hz']:
        if not supported[0] <= freq <= supported[1]:
            raise ValueError(f'{record["file"]}: {freq:g} Hz is outside the supported center-frequency '
                f'range {supported[0]:.2f}–{supported[1]:.2f} Hz for a {2*half*1000:g} ms window. '
                'Choose frequencies inside this range or shorten RAW_WINDOW_MS.')
    for freq in c['raw_frequencies_hz']:
        center = (np.log(freq)-meta['log_intercept'])/meta['rate_per_s']
        a, b = center-c['raw_window_s']/2, center+c['raw_window_s']/2
        if not meta['support_start_s'] <= a < b <= meta['support_end_s']:
            raise ValueError(f'{freq:g} Hz raw window is outside observed trajectory support')
        t,y = window(record, a, b)
        q = local_quality(t,c['tracking'])
        f_low,f_high = np.exp(meta['log_intercept']+meta['rate_per_s']*np.array([a,b]))
        if not q['gap_ok'] or f_high >= q['nominal_nyquist_hz']*c['tracking']['nyquist_fraction']:
            raise ValueError(f'Unsupported sampling in {freq:g} Hz window')
        x = np.column_stack([np.ones(len(t)), t-center])
        coefficients = np.linalg.lstsq(x,y,rcond=None)[0]
        detrended = y-x@coefficients
        elapsed = (t-a)*1000
        rows.append(dict(file=record['file'], target_center_hz=freq, center_s=center,
            window_start_s=a,window_end_s=b,model_start_hz=f_low,model_end_hz=f_high,
            detrend_intercept_counts=coefficients[0],detrend_slope_counts_per_s=coefficients[1],
            samples_per_cycle_min=q['fs_hz']/f_high,**q))
        samples.append(pd.DataFrame(dict(file=record['file'],target_center_hz=freq,
            recorded_time_s=t,window_elapsed_ms=elapsed,raw_counts=y,detrended_counts=detrended)))
        for name, values, label in [('raw',y,'Raw LDC output (counts)'),
                                    ('detrended',detrended,'Detrended LDC output (counts)')]:
            fig,ax=plt.subplots(figsize=(7,7))
            ax.set_box_aspect(1)
            ax.plot(elapsed, values, '-', color='#1f77b4', lw=1.2)
            ax.set(xlim=(0,c['raw_window_s']*1000),xlabel='Time (ms)',ylabel=label,
                   title=f'{Path(record["file"]).stem} | {freq:g} Hz')
            ax.spines[['top','right']].set_visible(False)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False,useMathText=False)
            token = np.format_float_positional(float(freq), trim='-')
            save_fig(run,fig,f'OO50_thin_{Path(record["file"]).stem}_{token}Hz_{name}',figures)
    pd.DataFrame(rows).to_csv(run.path/'raw_window_summary.csv',index=False)
    pd.concat(samples).to_csv(run.path/'raw_window_samples.csv',index=False)


def export_raw_examples(frequencies_hz, raw_file='IM_3.csv', window_ms=200.,
                        notebook_path=None, tracking=None):
    """Regenerate three frequencies directly from one raw file, without rerunning all groups."""
    ensure_environment()
    freq=np.asarray(frequencies_hz,dtype=float)
    if freq.shape!=(3,) or not np.isfinite(freq).all() or np.any(freq<=0) or len(set(freq))!=3:
        raise ValueError('RAW_FREQUENCIES_HZ must contain three distinct positive finite frequencies.')
    if not np.isfinite(window_ms) or window_ms<=0:
        raise ValueError('RAW_WINDOW_MS must be positive and finite.')
    if raw_file not in ['IM_3.csv','IM_4.csv','IM_5.csv']:
        raise ValueError('Thin OO-50 examples support RAW_FILE = IM_3.csv, IM_4.csv, or IM_5.csv.')
    c=review_config()
    c.update(raw_file=raw_file,raw_frequencies_hz=freq.tolist(),raw_window_s=float(window_ms)/1000)
    if tracking is not None:c['tracking']=deepcopy(tracking)
    path=(Path(c['tracking']['data_root'])/raw_file).resolve()
    item=dict(path=str(path),file=raw_file,sensor='IM',role='thin_OO50_raw_example',
              material='OO-50 0.23 mm',sha256=sha256(path.read_bytes()).hexdigest())
    root=Path(load_config()['results_root'])
    scripts=source_paths(notebook_path)
    method=('Single-file raw waveform export only. No candidate ranking, group response, '
            'amplitude-volume analysis, or SNR computation is performed.\n'
            'Use original timestamps and all samples in each requested window; save untouched '
            'Data28 and OLS intercept+linear-time detrended values separately. No filtering, '
            'smoothing, resampling or fitted sinusoid replaces the plotted samples.\n'
            'Center times use the existing independent log-frequency tracker (STFT discovery '
            'then original-time local chirp refinement). Frequency varies across each window; '
            'raw_window_summary.csv reports model start/end frequency, sampling and detrend '
            'coefficients. Time_us relative to first sample is converted to seconds; counts '
            'are not converted to volts. Physical timebase and analog bandwidth unverified.\n')
    with AnalysisRun(root,'data11_raw_examples',dict(config=c,methods=method),[item],scripts,
                     repository_path=PROJECT) as run:
        print('RAW_OUTPUT:',run.path,flush=True)
        run.write_text('methods.txt',method)
        record=load_record(item,c['tracking'])
        pd.DataFrame([record['audit']]).to_csv(run.path/'timestamp_audit.csv',index=False)
        if record['audit']['endpoint_code_count']:
            raise ValueError('Raw example record has numeric endpoints; review before plotting.')
        meta,track,_=independent_track(record,c['tracking'])
        pd.DataFrame([meta]).to_csv(run.path/'trajectory_summary.csv',index=False)
        track.to_csv(run.path/'frequency_track.csv',index=False)
        if not meta['trajectory_ok']:
            raise ValueError('Raw example needs an accepted frequency trajectory.')
        figures=[]
        _raw_examples(run,record,meta,c,figures)
        summary=pd.read_csv(run.path/'raw_window_summary.csv')
        run.finish(dict(file=raw_file,frequencies_hz=freq.tolist(),figures=len(figures)))
        return dict(output_path=str(run.path),figures=figures,windows=summary)


def run_review(config=None, notebook_path=None):
    ensure_environment()
    c=deepcopy(config if config is not None else review_config())
    freqs=np.asarray(c['raw_frequencies_hz'],float)
    if len(freqs)!=3 or len(set(freqs))!=3 or not np.isfinite(freqs).all() or np.any(freqs<=0):
        raise ValueError('Choose three distinct positive frequencies')
    if c['raw_window_s']<=0:
        raise ValueError('raw_window_s must be positive')
    items=_inputs(c)
    root=Path(load_config()['results_root'])
    scripts=source_paths(notebook_path)
    figures=[]; audits=[]; tracks={}; records={}; rms_rows=[]; coverage=[]
    with AnalysisRun(root,'data11_candidate_review',dict(config=c,methods=METHOD),items,scripts,
                     repository_path=PROJECT) as run:
        print('REVIEW_OUTPUT:',run.path,flush=True)
        run.write_text('methods.txt',METHOD)
        for item in items:
            print('Review:',item['file'],flush=True)
            row={k:item[k] for k in ('file','material','role')}
            row.update(recommended=False,quality_candidate=False)
            try:
                r=load_record(item,c['tracking']); records[item['file']]=r
                row.update(r['audit'])
                jumps=np.abs(np.diff(r['y'])); j=int(np.argmax(jumps))
                ratio=jumps[j]/max(float(np.median(jumps)),1.)
                row.update(max_adjacent_jump_counts=jumps[j],max_jump_time_s=r['t'][j],
                    max_to_median_jump_ratio=ratio,jump_review_flag=ratio>c['jump_ratio_review_threshold'])
                fig,ax=plt.subplots(figsize=(11,4.8)); _envelope(ax,r)
                ax.set_title(f'{item["file"]} | {item["material"]} | full-record min/max envelope')
                save_fig(run,fig,Path(item['file']).stem+'_raw_overview',figures)
                if row['jump_review_flag']:
                    mask=np.abs(r['t']-r['t'][j])<=.05
                    fig,ax=plt.subplots(figsize=(11,4.8))
                    ax.plot(r['t'][mask],r['y'][mask],'o-',lw=.8,ms=3)
                    ax.set(xlabel='Recorded time (s)',ylabel='Raw LDC output (counts)',
                        title=f'{item["file"]} | largest adjacent jump (diagnostic)')
                    save_fig(run,fig,Path(item['file']).stem+'_jump_diagnostic',figures)
                if item['role']=='amplitude_candidate':
                    for start in range(int(np.floor(r['t'][-1]))):
                        mask=(r['t']>=start)&(r['t']<start+1)
                        yy=r['y'][mask]
                        rms_rows.append(dict(file=item['file'],start_s=start,end_s=start+1,
                            samples=len(yy),demeaned_rms_counts=float(np.std(yy)),
                            raw_peak_to_peak_counts=float(np.ptp(yy))))
                    row['dwell_s']=c['documented_amplitude']['dwell_s_by_file'][item['file']]
                    row['review_reason']='Selection pending user; first-level start not confirmed'
                elif row['endpoint_code_count']:
                    row['review_reason']='Excluded by existing record-level numeric-endpoint policy'
                else:
                    meta,track,_=independent_track(r,c['tracking'])
                    tracks[item['file']]=meta; row.update(meta)
                    duration_error=abs(meta['inferred_f0_to_f1_duration_s']/c['tracking']['nominal_duration_s']-1)
                    row['duration_relative_error']=duration_error
                    row['schedule_review_flag']=duration_error>c['duration_relative_review_tolerance']
                    finite=np.isfinite(track.local_hz)&np.isfinite(track.center_s)
                    tx=track.loc[finite,'center_s'].to_numpy()
                    fy=track.loc[finite,'local_hz'].to_numpy()
                    if len(tx)>=3:
                        slope,intercept=np.polyfit(tx,fy,1)
                        row['linear_local_slope_hz_per_s']=slope
                        row['linear_local_rmse_hz']=float(np.sqrt(np.mean((fy-(intercept+slope*tx))**2)))
                        row['log_model_local_rmse_hz']=float(np.sqrt(np.mean((fy-np.exp(meta['log_intercept']+meta['rate_per_s']*tx))**2)))
                        if row['schedule_review_flag']:
                            fig,ax=plt.subplots(figsize=(11,4.8))
                            ax.plot(tx,fy,'o',ms=3,label='Local frequency estimates')
                            ax.plot(tx,np.exp(meta['log_intercept']+meta['rate_per_s']*tx),label='Existing log model')
                            ax.plot(tx,intercept+slope*tx,label='Linear fit (diagnostic)')
                            ax.set(xlabel='Recorded time (s)',ylabel='Frequency (Hz)',
                                title=f'{item["file"]} | sweep-model compatibility review')
                            ax.legend()
                            save_fig(run,fig,Path(item['file']).stem+'_schedule_diagnostic',figures)
                    track.to_csv(run.path/(Path(item['file']).stem+'_frequency_track.csv'),index=False)
                    count=0
                    for freq in np.geomspace(c['tracking']['f0_hz'],c['tracking']['f1_hz'],c['tracking']['grid_points']):
                        center=(np.log(freq)-meta['log_intercept'])/meta['rate_per_s']
                        if not meta['support_start_s']<=center<=meta['support_end_s']: continue
                        try:
                            duration=c['tracking']['window_s']
                            t,y=window(r,center-duration/2,center+duration/2)
                            fit=tracking_filter(t,y,center,freq,meta['rate_per_s'],duration)
                            q=local_quality(t,c['tracking'])
                            ok=meta['trajectory_ok'] and _sampling_ok(q,fit,c['tracking'])
                            coverage.append(dict(file=item['file'],frequency_hz=freq,sampling_ok=ok,**q))
                            count+=int(ok)
                        except ValueError: pass
                    row['supported_grid_points']=count
                    row['quality_candidate']=bool(meta['trajectory_ok'] and count>=2
                        and not row['jump_review_flag'] and not row['schedule_review_flag'])
                    row['review_reason']=('Pass existing gates and jump review screen' if row['quality_candidate']
                        else 'Needs review: jump flag, sweep-schedule compatibility, trajectory, or sampling support')
            except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as exc:
                row['review_reason']=str(exc)
            audits.append(row)
            pd.DataFrame(audits).to_csv(run.path/'candidate_audit.csv',index=False)
        audit=pd.DataFrame(audits)
        selected={}; selection=[]
        for material,(noise,ids) in c['tracking']['materials'].items():
            good=audit.loc[(audit.material==material)&audit.quality_candidate,'file'].tolist()
            chosen=[i for i in sorted(ids) if f'IM_{i}.csv' in good][:c['requested_sweep_count']]
            selected[material]=[noise,chosen]
            audit.loc[(audit.material==material)&audit.file.isin([f'IM_{i}.csv' for i in chosen]),'recommended']=True
            selection.append(dict(material=material,selected_ids=chosen,count=len(chosen),
                requested=3,status='three available' if len(chosen)==3 else 'insufficient passing records'))
        audit.to_csv(run.path/'candidate_audit.csv',index=False)
        pd.DataFrame(selection).to_csv(run.path/'sweep_selection.csv',index=False)
        pd.DataFrame(coverage).to_csv(run.path/'sweep_sampling_support.csv',index=False)
        pd.DataFrame(tracks.values()).to_csv(run.path/'trajectory_summary.csv',index=False)
        rms=pd.DataFrame(rms_rows); rms.to_csv(run.path/'amplitude_time_diagnostics.csv',index=False)
        if c['amplitude_ids']:
            fig,axes=plt.subplots(len(c['amplitude_ids']),1,figsize=(12,3*len(c['amplitude_ids'])),sharex=True)
            for ax,i in zip(np.atleast_1d(axes),c['amplitude_ids']):
                p=rms[rms.file.eq(f'IM_{i}.csv')]
                ax.plot((p.start_s+p.end_s)/2,p.demeaned_rms_counts)
                ax.set(ylabel='RMS (counts)',title=f'IM_{i} | 1 s bins, mean removed')
            np.atleast_1d(axes)[-1].set_xlabel('Recorded time (s)')
            save_fig(run,fig,'amplitude_all_candidates_time_RMS',figures)
        chosen_config=deepcopy(c['tracking']); chosen_config['materials']=selected
        run.write_json('recommended_tracking_config.json',chosen_config)
        run.write_json('amplitude_selection.json',dict(candidate_ids=c['amplitude_ids'],
            selected_ids=c['amplitude_selected_ids'],status='pending user selection; no automatic triplet'))
        raw_file=c['raw_file']
        if c['include_raw_examples']:
            if raw_file not in tracks or not tracks[raw_file]['trajectory_ok']:
                raise ValueError(f'Raw example requires accepted trajectory: {raw_file}')
            if records[raw_file]['material']!='OO-50 0.23 mm':
                raise ValueError('This example section is restricted to thin OO-50')
            _raw_examples(run,records[raw_file],tracks[raw_file],c,figures)
        run.finish(dict(candidate_count=len(audit),selection=selection,amplitude_selection='pending user',
                        raw_file=raw_file,raw_frequencies_hz=list(freqs)))
        return dict(output_path=str(run.path),audit=audit,selection=pd.DataFrame(selection),
            recommended_config=chosen_config,figures=figures)


if __name__=='__main__':
    result=run_review(notebook_path=publication_notebook_path('data11_frequency_range_analysis'))
    print(result['selection'].to_string(index=False),flush=True)
    print('COMPLETED:',result['output_path'],flush=True)
