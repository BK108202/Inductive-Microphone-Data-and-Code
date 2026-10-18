"""Separate whole-sweep FFT comparison; never replaces harmonic tracking."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from characterization_run import AnalysisRun
from data11_config import PROJECT
from data11_analysis import manifest, load_record, local_quality, save_fig
from portable_runtime import config_source, ensure_environment, load_config, source_paths

METHODS = {
    'scope': 'Additional whole-sweep FFT analysis of the same Data11 inputs. No harmonic tracking, dechirping, peak search, Welch averaging or transfer-function estimation is used for the amplitudes. Existing data-derived log trajectories are used ONLY to select start/end times and audit fundamental coverage. Parent run, trajectory CSV, parameters and original input hashes are recorded and validated.',
    'timing': 'Per file t(f)=(log(f)-log_intercept)/rate_per_s from the parent independently fitted trajectory. Default FFT window spans inferred 30 Hz to inferred 300 Hz crossings. These are model-inferred boundaries, not measured switch/trigger times; extrapolated boundary distances to coherent track support are exported. Explicit per-file time-window overrides are accepted and recorded as user choices, not experimental evidence. No clipping to record boundaries, padding of missing data, or repair of timestamps. Incomplete windows are unavailable.',
    'fft': 'Read Time_us as seconds relative to first sample; IM Data28 counts; accel Raw_Z times inherited nominal g/count conversion. Interpolate original samples linearly onto N=floor(T*record_mean_fs) samples at start+arange(N)*T/N, fs=N/T, with no extrapolation. Remove arithmetic mean. Default rectangular (boxcar) window; optional periodic Hann. One rFFT per whole sweep, default no zero padding. Peak-equivalent bin magnitude A=2*abs(X)/sum(w) (DC and even-length Nyquist not doubled). A is a spectrum magnitude, NOT instantaneous fundamental peak of a swept sine. Full native/padded frequency grid saved. PSD uses one-sided |X|^2/(fs*sum(w^2)) with correct endpoint factors.',
    'bandwidth': 'Save native df=1/T, displayed df=fs/nfft, ENBW=fs*sum(w^2)/sum(w)^2. Boxcar ENBW=1/T; Hann ENBW=1.5/T. Common reporting frequencies are a configurable linear grid; FFT POWER (A^2 and PSD) is linearly interpolated without extrapolation, then square-rooted for amplitude. This is interpolation of adjacent bin powers, not a new narrow filter or improved resolution. Signal and noise use identical T, N, window and nfft, including identical reporting interpolation. Reference cross-sensor ENBW and sweep durations must agree within configured tolerances.',
    'noise': 'Separate nothing_on and amplifier_on baseline records; windows start after configured noise_start_s and are nonoverlapping, each exactly as long as that signal sweep. No re-use/wrapping to make a short baseline longer. At each frequency N_eq=sqrt(mean(A_noise^2)) over available same-length windows. Default minimum=1 because many baselines only contain one complete 30 s window; counts exported and single-periodogram uncertainty explicitly retained. Shared baseline is not an independent sweep repeat. All noise spectra are FFTs, no frequency search or smoothing.',
    'snr': 'Per-record FFT SNR_linear=A_sweep/N_eq, SNR_dB=20*log10(SNR_linear). Numerator includes signal plus noise, harmonics and leakage; no noise-power subtraction. This is whole-sweep spectral-bin SNR, distinct from tracking-estimator SNR and broadband SNR. Group SNR is mean of per-record dB ratios; require configured independent sweep count. Threshold strictly >10 dB by default, not a calibrated false-alarm threshold.',
    'quality': 'Strict: accepted parent trajectory, complete selected window, no IM record endpoint codes, local timestamp gap ratio<=inherited max_gap_ratio, adequate Nyquist/samples per cycle, and frequency inside observed parent coherent-track support. Noise must independently meet the same gap and sampling checks. Diagnostic_with_gap_interpolation keeps original gap flags but permits linear interpolation across gaps for BOTH signal and noise; it still rejects IM endpoint records and frequencies outside coherent fundamental support. Diagnostic amplitudes for every computable frequency are retained separately. All raw data remain unchanged. Jitter/interpolation can bias high-frequency amplitudes.',
    'responsivity': 'Ratio of independently averaged IM FFT magnitudes to independently averaged accelerometer FFT magnitudes at common frequencies, never file pairing, complex division, phase or coherence. Require both sensors SNR above threshold, repeats, inherited reference conditions, and configured duration/ENBW agreement. counts/g (and counts/(m/s^2)=counts/g/9.80665) is nominal spectral-amplitude responsivity, not calibrated sensitivity. Whole-sweep spectral ratios also depend on sweep speed, window, harmonics and between-batch repeatability. Strict and gap-interpolated diagnostic ratios are separately exported and plotted.',
    'comparison': 'Overlay FFT and tracking separately with their own amplitude/SNR definitions and bandwidths. Interpolate parent aggregate curves only between adjacent available parent grid points, never across missing points or outside measured coverage. Difference CSV includes actual record populations/counts; differing populations are not silently presented as a pure estimator effect. A difference in dB does not establish a better method.',
    'range': 'Every consecutive common-grid interval above threshold is exported, no bridging of failed/missing points or interpolated cutoff. Fourier energy beyond the fundamental track may be harmonics/leakage and is not proof of fundamental detectability. FFT spectral support is not intrinsic or acoustic bandwidth.',
    'reference_source': 'FFT/PSD scaling and ENBW relationship: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html .',
}


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fft_config(parent_run):
    parent=Path(parent_run).resolve()
    c=deepcopy(_json(parent/'parameters.json')['config'])
    c.update(fft_parent_run=str(parent),fft_window='boxcar',fft_pad_factor=1,
             fft_frequency_step_hz=0.1,fft_noise_min_windows=1,
             fft_time_overrides_s={},fft_duration_relative_tolerance=0.01,
             fft_enbw_relative_tolerance=0.03)
    return c


def resampled_window(r,start,end,c,n=None):
    """Uniform FFT grid; surrounding interpolation samples included in gap audit."""
    t,y=r['t'],r['y']; duration=end-start
    if not np.isfinite([start,end]).all() or duration<=0 or start<t[0] or end>t[-1]:
        raise ValueError('Requested whole-sweep window is outside raw record support')
    n=int(np.floor(duration*r['audit']['fs_hz'])) if n is None else int(n)
    if n<16: raise ValueError('Too few FFT samples')
    u=start+np.arange(n)*duration/n
    i=max(0,np.searchsorted(t,start,side='right')-1)
    j=min(len(t),np.searchsorted(t,end,side='left')+1)
    tt=t[i:j]
    if u[0]<tt[0] or u[-1]>tt[-1]: raise ValueError('FFT interpolation would extrapolate')
    q=local_quality(tt,c)
    q.update(window_start_s=float(start),window_end_s=float(end),duration_s=float(duration),
             interpolated_samples=n,uniform_fs_hz=n/duration)
    return np.interp(u,tt,y[i:j]),q


def fft_spectrum(y,duration,window_name='boxcar',pad=1):
    y=np.asarray(y,float)
    if pad<1 or int(pad)!=pad or duration<=0 or not np.isfinite(y).all():
        raise ValueError('Invalid FFT inputs')
    n=len(y); fs=n/duration
    if window_name not in ('boxcar','hann'): raise ValueError('Use boxcar or hann')
    w=signal.get_window(window_name,n,fftbins=True)
    nfft=n*int(pad)
    x=np.fft.rfft((y-y.mean())*w,n=nfft)
    factor=np.full(len(x),2.); factor[0]=1.
    if nfft%2==0: factor[-1]=1.
    amplitude=factor*np.abs(x)/w.sum()
    psd=factor*np.abs(x)**2/(fs*np.sum(w*w))
    return dict(frequency_hz=np.fft.rfftfreq(nfft,1/fs),amplitude_peak_equivalent=amplitude,psd=psd),dict(
        fft_samples=n,nfft=nfft,fft_fs_hz=fs,native_df_hz=1/duration,
        displayed_df_hz=fs/nfft,enbw_hz=fs*np.sum(w*w)/w.sum()**2)


def common_amplitude(spectrum,frequencies):
    return np.sqrt(np.interp(frequencies,spectrum['frequency_hz'],
                            spectrum['amplitude_peak_equivalent']**2,left=np.nan,right=np.nan))


def supported_interp(x,y,new_x):
    """No interpolation across NaNs: preserve unavailable parent-grid intervals."""
    x,y,new_x=np.asarray(x),np.asarray(y,float),np.asarray(new_x)
    out=np.full(new_x.shape,np.nan)
    ids=np.flatnonzero(np.isfinite(y))
    for group in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
        if len(group)==1:
            out[np.isclose(new_x,x[group[0]],rtol=0,atol=1e-10)]=y[group[0]]
        elif len(group)>1:
            keep=(new_x>=x[group[0]])&(new_x<=x[group[-1]])
            out[keep]=np.interp(new_x[keep],x[group],y[group])
    return out


def _snr(a,noise):
    linear=np.divide(a,noise,out=np.full_like(a,np.nan),where=noise>0)
    with np.errstate(divide='ignore',invalid='ignore'):
        db=np.where(linear>0,20*np.log10(linear),np.nan)
    return linear,db


def aggregate(detail,c):
    summaries=[]
    for population in ('strict','diagnostic_with_gap_interpolation'):
        for (material,idx),allp in detail.groupby(['material','grid_index'],sort=False):
            p=allp[allp[population+'_ok']]
            a=p.amplitude_peak_equivalent
            row=dict(population=population,material=material,grid_index=idx,
                     frequency_hz=allp.frequency_hz.iloc[0],n_amplitude=len(p),
                     eligible_files=';'.join(sorted(p.file.unique())),
                     amplitude_mean=a.mean() if len(p)>=c['minimum_repeat_count'] else np.nan,
                     amplitude_sd=a.std() if len(p)>=c['minimum_repeat_count'] else np.nan,
                     duration_mean_s=p.duration_s.mean(),enbw_mean_hz=p.enbw_hz.mean(),
                     gap_records=int((~p.gap_ok).sum()))
            for condition in ('nothing_on','amplifier_on'):
                values=p[population+'_'+condition+'_snr_db'].dropna()
                noise=p[population+'_'+condition+'_noise_peak'].dropna()
                row[condition+'_n_snr']=len(values)
                row[condition+'_snr_db_mean']=values.mean() if len(values)>=c['minimum_repeat_count'] else np.nan
                row[condition+'_snr_db_sd']=values.std() if len(values)>=c['minimum_repeat_count'] else np.nan
                row[condition+'_noise_peak_rms']=np.sqrt(np.mean(noise**2)) if len(noise)>=c['minimum_repeat_count'] else np.nan
            row['detectable']=bool(row['amplifier_on_snr_db_mean']>c['snr_threshold_db'])
            summaries.append(row)
    summary=pd.DataFrame(summaries)
    ratios=[]; intervals=[]
    for population,sub in summary.groupby('population',sort=False):
        reference=sub[sub.material.eq('accelerometer')].set_index('grid_index')
        for row in sub[~sub.material.eq('accelerometer')].itertuples():
            if row.grid_index not in reference.index: continue
            acc=reference.loc[row.grid_index]
            duration_mismatch=abs(row.duration_mean_s/acc.duration_mean_s-1)
            bandwidth_mismatch=abs(row.enbw_mean_hz/acc.enbw_mean_hz-1)
            ok=bool(c['reference_conditions_confirmed'] and row.detectable and acc.detectable and acc.amplitude_mean>0
                    and duration_mismatch<=c['fft_duration_relative_tolerance']
                    and bandwidth_mismatch<=c['fft_enbw_relative_tolerance'])
            value=row.amplitude_mean/acc.amplitude_mean if ok else np.nan
            ratios.append(dict(population=population,material=row.material,grid_index=row.grid_index,frequency_hz=row.frequency_hz,
                               n_im=row.n_amplitude,n_accel=int(acc.n_amplitude),
                               im_files=row.eligible_files,accel_files=acc.eligible_files,
                               duration_relative_mismatch=duration_mismatch,enbw_relative_mismatch=bandwidth_mismatch,
                               eligible=ok,counts_per_g=value,counts_per_m_s2=value/c['standard_gravity_m_s2']))
        for material,p in sub.groupby('material',sort=False):
            p=p.sort_values('grid_index').set_index('grid_index')
            indices=p.index[p.detectable].to_numpy(int)
            for segment in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
                if len(segment):
                    pp=p.loc[segment]
                    intervals.append(dict(population=population,material=material,start_hz=pp.frequency_hz.iloc[0],
                                          end_hz=pp.frequency_hz.iloc[-1],grid_points=len(pp),threshold_db=c['snr_threshold_db']))
    return summary,pd.DataFrame(ratios),pd.DataFrame(intervals,columns=['population','material','start_hz','end_hz','grid_points','threshold_db'])


def compare_tracking(summary,ratios,parent,c):
    tracking=pd.read_csv(parent/'harmonic_tracking_summary.csv')
    reference=pd.read_csv(parent/'acceleration_normalized_responsivity.csv')
    records=pd.read_csv(parent/'harmonic_tracking_records.csv')
    rows=[]
    for (population,material),p in summary.groupby(['population','material'],sort=False):
        p=p.sort_values('frequency_hz').copy()
        old=tracking[tracking.material.eq(material)].sort_values('frequency_hz')
        p['tracking_peak_mean']=supported_interp(old.frequency_hz,old.amplitude_peak_mean,p.frequency_hz)
        p['tracking_snr_db_mean']=supported_interp(old.frequency_hz,old.snr_db_mean,p.frequency_hz)
        p['tracking_enbw_nominal_hz']=1.5/c['window_s']
        p['fft_minus_tracking_snr_db']=p.amplifier_on_snr_db_mean-p.tracking_snr_db_mean
        rr=ratios[(ratios.population==population)&(ratios.material==material)].sort_values('frequency_hz') if len(ratios) else pd.DataFrame()
        p['fft_counts_per_g']=rr.counts_per_g.to_numpy() if len(rr)==len(p) else np.nan
        oldr=reference[reference.material.eq(material)].sort_values('frequency_hz')
        p['tracking_counts_per_g']=supported_interp(oldr.frequency_hz,oldr.counts_per_g,p.frequency_hz) if len(oldr) else np.nan
        # Explicit bracketing populations: interpolation is not another repeat.
        oldfiles={idx:';'.join(sorted(part.loc[part.primary_ok,'file'].unique()))
                  for idx,part in records[records.material.eq(material)].groupby('grid_index')}
        hi=np.searchsorted(old.frequency_hz.to_numpy(),p.frequency_hz.to_numpy(),side='left')
        lo=np.maximum(0,hi-1); hi=np.minimum(len(old)-1,hi)
        p['tracking_files_lower']=[oldfiles.get(int(old.iloc[i].grid_index),'') for i in lo]
        p['tracking_files_upper']=[oldfiles.get(int(old.iloc[i].grid_index),'') for i in hi]
        p['same_im_population_as_bracketing_tracking']=(p.eligible_files==p.tracking_files_lower)&(p.eligible_files==p.tracking_files_upper)
        rows.append(p)
    return pd.concat(rows,ignore_index=True)


def _plots(run,summary,reference,comparison,c,figures):
    diag=summary[summary.population.eq('diagnostic_with_gap_interpolation')]
    for group,members in c['comparison_groups'].items():
        fig,axes=plt.subplots(3,1,figsize=(12,11),sharex=True)
        for material in members:
            p=diag[diag.material.eq(material)]
            rr=reference[reference.population.eq('diagnostic_with_gap_interpolation')&reference.material.eq(material)]
            line,=axes[0].plot(p.frequency_hz,p.amplitude_mean,label=material)
            color=line.get_color()
            axes[0].plot(p.frequency_hz,p.amplifier_on_noise_peak_rms,':',color=color,alpha=.6)
            axes[1].plot(p.frequency_hz,p.amplifier_on_snr_db_mean,color=color,label=material)
            axes[2].plot(rr.frequency_hz,rr.counts_per_g,color=color,label=material)
        axes[0].set(ylabel='FFT peak-equivalent magnitude',yscale='log',title='Whole sweep FFT: solid = IM; dotted = amplifier-on noise FFT')
        axes[1].set(ylabel='Whole-sweep FFT SNR (dB)')
        axes[1].axhline(c['snr_threshold_db'],color='k',ls='--')
        axes[2].set(ylabel='Spectral ratio (counts/g)',xlabel='FFT frequency (Hz)',yscale='log')
        for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=7,loc='upper right')
        fig.suptitle(group+' | Gap-interpolated diagnostic; not calibrated sensitivity',fontsize=12)
        fig.tight_layout(); save_fig(run,fig,'fft_diagnostic_'+group,figures)
    for material,p in comparison[comparison.population.eq('diagnostic_with_gap_interpolation')].groupby('material',sort=False):
        if material=='accelerometer': continue
        fig,axes=plt.subplots(3,1,figsize=(11,10),sharex=True)
        axes[0].plot(p.frequency_hz,p.amplitude_mean,label='Whole FFT spectral magnitude')
        axes[0].plot(p.frequency_hz,p.tracking_peak_mean,label='Tracking fundamental peak')
        axes[0].set(ylabel='Counts (different estimators)',yscale='log')
        axes[1].plot(p.frequency_hz,p.amplifier_on_snr_db_mean,label='Whole FFT SNR')
        axes[1].plot(p.frequency_hz,p.tracking_snr_db_mean,label='Tracking SNR')
        axes[1].set(ylabel='SNR (dB; different bandwidths)')
        axes[1].axhline(c['snr_threshold_db'],ls='--',color='k')
        axes[2].plot(p.frequency_hz,p.fft_counts_per_g,label='Whole FFT diagnostic ratio')
        axes[2].plot(p.frequency_hz,p.tracking_counts_per_g,label='Tracking primary ratio')
        axes[2].set(ylabel='Nominal ratio (counts/g)',xlabel='Frequency (Hz)',yscale='log')
        for ax in axes: ax.legend(loc='upper right',fontsize=8); ax.grid(alpha=.2)
        fig.suptitle(material+' | FFT retains interpolated gaps; record populations may differ')
        fig.tight_layout(); safe=''.join(ch if ch.isalnum() else '_' for ch in material)
        save_fig(run,fig,'fft_vs_tracking_'+safe,figures)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for population,style in [('strict','-'),('diagnostic_with_gap_interpolation','--')]:
        p=summary[summary.material.eq('accelerometer')&summary.population.eq(population)]
        axes[0].plot(p.frequency_hz,p.amplitude_mean,style,label=population)
        axes[0].plot(p.frequency_hz,p.amplifier_on_noise_peak_rms,':',label=population+' noise')
        axes[1].plot(p.frequency_hz,p.amplifier_on_snr_db_mean,style,label=population)
    axes[0].set(ylabel='Accelerometer FFT (g)',yscale='log',title='Reference whole-sweep FFT; absent strict curves mean unavailable')
    axes[1].set(xlabel='Frequency (Hz)',ylabel='FFT SNR (dB)')
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); save_fig(run,fig,'fft_accelerometer_reference',figures)


def run_whole_fft(config,notebook_path,executed_cells=None):
    c=deepcopy(config); parent=Path(c['fft_parent_run']).resolve()
    ensure_environment()
    if not (0<c['f0_hz']<c['f1_hz'] and c['fft_frequency_step_hz']>0 and c['fft_noise_min_windows']>=1):
        raise ValueError('Invalid FFT configuration')
    if _json(parent/'run_status.json')['status']!='completed': raise ValueError('Parent tracking run is not complete')
    inherited=_json(parent/'parameters.json')['config']
    for key in ('accel_g_per_count','reference_conditions_confirmed','im_endpoint_codes'):
        if c[key]!=inherited[key]: raise ValueError('Reference/raw conversion changed from parent: '+key)
    if c['im_endpoint_scope']!='record': raise ValueError('Retain the existing record-level endpoint safeguard')
    inputs=manifest(c)
    if len({x['path'] for x in inputs})!=len(inputs): raise ValueError('Duplicate input paths')
    parent_roles={Path(x['path']).name:{**x,'file':Path(x['path']).name} for x in _json(parent/'inputs.json')}
    for item in inputs:
        if item!=parent_roles.get(item['file']): raise ValueError('Input roles differ from parent run: '+item['file'])
    auxiliary=['parameters.json','inputs.json','input_hashes.json','provenance.json','run_status.json',
               'trajectory_summary.csv','harmonic_tracking_summary.csv','harmonic_tracking_records.csv','acceleration_normalized_responsivity.csv']
    all_inputs=inputs+[dict(path=str(parent/name),role='parent_tracking_'+name) for name in auxiliary]
    root=Path(load_config()['results_root'])
    scripts=source_paths(notebook_path)
    figures=[]; issues=[]; audits=[]; hashes=[]; windows=[]; noise_audit=[]; details=[]; records={}
    trajectories=pd.read_csv(parent/'trajectory_summary.csv').set_index('file')
    expected={str(Path(x['path']).resolve()):x['sha256'] for x in _json(parent/'input_hashes.json')}
    frequencies=np.arange(c['f0_hz'],c['f1_hz']+c['fft_frequency_step_hz']/2,c['fft_frequency_step_hz'])
    with AnalysisRun(root,'data11_whole_sweep_fft',dict(config=c,methods=METHODS,
                     results_root_source=config_source()),all_inputs,scripts,repository_path=PROJECT) as run:
        print('FFT output directory:',run.path,flush=True)
        run.write_text('methods.txt','\n\n'.join(k+': '+v for k,v in METHODS.items()))
        if executed_cells: run.write_text('scripts/executed_notebook_cells.py','\n\n'.join(executed_cells))
        for item in all_inputs:
            try:
                path=Path(item['path']); digest=sha256(path.read_bytes()).hexdigest()
                hashes.append(dict(path=str(path),role=item['role'],sha256=digest))
                if item in inputs:
                    if digest!=expected.get(str(path.resolve())): raise RuntimeError('Input differs from parent hash: '+str(path))
                    r=load_record(item,c); records[item['file']]=r; audits.append(r['audit'])
            except FileNotFoundError as exc:
                issues.append(dict(**item,error=str(exc)))
                if item not in inputs or c['missing_policy']=='fail': raise
            finally:
                run.write_json('inputs_hashes.json',hashes); run.write_json('input_issues.json',issues)
                pd.DataFrame(audits).to_csv(run.path/'timestamp_audit.csv',index=False)
        (run.path/'spectra').mkdir()
        baseline={(r['material'],r['condition']):r for r in records.values() if r['role']=='noise'}
        for r in records.values():
            if r['role']!='sweep': continue
            try:
                if r['file'] not in trajectories.index: raise ValueError('No parent trajectory')
                meta=trajectories.loc[r['file']]
                if not meta.trajectory_ok: raise ValueError('Parent trajectory failed')
                a=(np.log(c['f0_hz'])-meta.log_intercept)/meta.rate_per_s
                b=(np.log(c['f1_hz'])-meta.log_intercept)/meta.rate_per_s
                inferred_a,inferred_b=a,b
                override=c['fft_time_overrides_s'].get(r['file'])
                if override is not None: a,b=map(float,override)
                y,q=resampled_window(r,a,b,c)
                spectrum,scalars=fft_spectrum(y,b-a,c['fft_window'],c['fft_pad_factor'])
            except (ValueError,KeyError) as exc:
                issues.append(dict(file=r['file'],role='sweep_fft',error=str(exc)))
                run.write_json('input_issues.json',issues)
                continue
            n=len(y); amp=common_amplitude(spectrum,frequencies)
            lower=np.exp(meta.log_intercept+meta.rate_per_s*meta.support_start_s)
            upper=np.exp(meta.log_intercept+meta.rate_per_s*meta.support_end_s)
            endpoint_ok=r['sensor']!='IM' or r['audit']['endpoint_code_count']==0
            sampling_ok=(frequencies<q['nominal_nyquist_hz']*c['nyquist_fraction'])&(q['fs_hz']/frequencies>=c['min_samples_per_cycle'])
            support=(frequencies>=lower)&(frequencies<=upper)
            selected_f0=np.exp(meta.log_intercept+meta.rate_per_s*a)
            selected_f1=np.exp(meta.log_intercept+meta.rate_per_s*b)
            # Reject time overrides that do not represent the requested same sweep span.
            matching_span=abs(selected_f0/c['f0_hz']-1)<=c['fft_duration_relative_tolerance'] and abs(selected_f1/c['f1_hz']-1)<=c['fft_duration_relative_tolerance']
            diag_ok=sampling_ok&support&endpoint_ok&matching_span
            common=dict(file=r['file'],sensor=r['sensor'],material=r['material'],record_endpoint_count=r['audit']['endpoint_code_count'],
                        inferred_start_s=inferred_a,inferred_end_s=inferred_b,start_basis='explicit_override' if override else 'parent_log_trajectory_crossings',
                        start_extrapolation_s=max(0.,meta.support_start_s-a),end_extrapolation_s=max(0.,b-meta.support_end_s),
                        track_support_min_hz=lower,track_support_max_hz=upper,selected_f0_hz=selected_f0,selected_f1_hz=selected_f1,
                        matching_sweep_span=matching_span,**q,**scalars)
            windows.append(common)
            row=pd.DataFrame(dict(grid_index=np.arange(len(frequencies)),frequency_hz=frequencies,
                                  amplitude_peak_equivalent=amp,fundamental_support=support,
                                  strict_ok=diag_ok&q['gap_ok'],diagnostic_with_gap_interpolation_ok=diag_ok))
            for key,value in common.items(): row[key]=value
            native=pd.DataFrame(spectrum)
            for condition in ('nothing_on','amplifier_on'):
                noise=baseline.get((r['material'],condition)); estimates=[]; strict_estimates=[]
                if noise is not None:
                    T=b-a
                    starts=np.arange(c['noise_start_s'],noise['t'][-1]-T+1e-9,T)
                    for idx,start in enumerate(starts):
                        yn,nq=resampled_window(noise,start,start+T,c,n=n)
                        ns,nsc=fft_spectrum(yn,T,c['fft_window'],c['fft_pad_factor'])
                        na=common_amplitude(ns,frequencies)
                        no_endpoints=noise['sensor']!='IM' or noise['audit']['endpoint_code_count']==0
                        nsampling=(frequencies<nq['nominal_nyquist_hz']*c['nyquist_fraction'])&(nq['fs_hz']/frequencies>=c['min_samples_per_cycle'])
                        na=np.where(nsampling&no_endpoints,na,np.nan)
                        estimates.append(na)
                        strict_estimates.append(na if nq['gap_ok'] else np.full_like(na,np.nan))
                        noise_audit.append(dict(signal_file=r['file'],noise_file=noise['file'],condition=condition,window_index=idx,
                                                endpoint_ok=no_endpoints,**nq,**nsc))
                        pd.DataFrame(ns).to_csv(run.path/'spectra'/f"{Path(r['file']).stem}_{condition}_noise_{idx:02d}.csv",index=False)
                for population,values in [('strict',strict_estimates),('diagnostic_with_gap_interpolation',estimates)]:
                    stack=np.array(values) if values else np.empty((0,len(frequencies)))
                    count=np.sum(np.isfinite(stack),axis=0)
                    power=np.divide(np.nansum(stack**2,axis=0),count,out=np.full(len(frequencies),np.nan),where=count>0)
                    noiseamp=np.where(count>=c['fft_noise_min_windows'],np.sqrt(power),np.nan)
                    linear,db=_snr(amp,noiseamp)
                    prefix=population+'_'+condition
                    row[prefix+'_noise_count']=count; row[prefix+'_noise_peak']=noiseamp
                    row[prefix+'_snr_linear']=linear; row[prefix+'_snr_db']=db
            native.to_csv(run.path/'spectra'/f"{Path(r['file']).stem}_sweep.csv",index=False)
            details.append(row)
            # Overview ties the inferred crop to raw evidence without changing it.
            fig,axes=plt.subplots(2,1,figsize=(10,6))
            step=max(1,len(r['t'])//15000)
            axes[0].plot(r['t'][::step],r['y'][::step],lw=.5)
            axes[0].axvspan(a,b,color='orange',alpha=.15,label='FFT time window')
            axes[0].axvline(a,color='r',ls='--'); axes[0].axvline(b,color='r',ls='--')
            axes[0].set(xlabel='Recorded time (s)',ylabel=r['audit']['unit'],title=f"{r['file']}: inferred [{a:.3f}, {b:.3f}] s | gap_ok={q['gap_ok']}")
            axes[0].legend(loc='upper right')
            axes[1].semilogy(spectrum['frequency_hz'],spectrum['amplitude_peak_equivalent'],lw=.7)
            axes[1].set(xlim=(0,min(500,q['nominal_nyquist_hz'])),xlabel='FFT frequency (Hz)',ylabel='Peak-equivalent FFT magnitude',title='Whole-window diagnostic: includes harmonics/leakage; not instantaneous fundamental')
            fig.tight_layout(); save_fig(run,fig,Path(r['file']).stem+'_fft_window',figures)
            pd.DataFrame(windows).to_csv(run.path/'sweep_window_audit.csv',index=False)
            pd.DataFrame(noise_audit).to_csv(run.path/'noise_window_audit.csv',index=False)
            row.to_csv(run.path/'spectra'/f"{Path(r['file']).stem}_common_grid_snr.csv",index=False)
            print('Whole-sweep FFT:',r['file'],flush=True)
        if not details: raise ValueError('No complete whole-sweep windows')
        detail=pd.concat(details,ignore_index=True)
        detail.to_csv(run.path/'fft_records.csv',index=False)
        summary,reference,intervals=aggregate(detail,c)
        comparison=compare_tracking(summary,reference,parent,c)
        summary.to_csv(run.path/'fft_summary.csv',index=False)
        reference.to_csv(run.path/'fft_responsivity.csv',index=False)
        intervals.to_csv(run.path/'fft_detectable_intervals.csv',index=False)
        comparison.to_csv(run.path/'fft_vs_tracking.csv',index=False)
        _plots(run,summary,reference,comparison,c,figures)
        run.write_json('input_issues.json',issues)
        run.finish(dict(sweep_windows=len(windows),input_issues=len(issues),figures=len(figures),
                        strict_reference_points=int(reference.loc[reference.population.eq('strict'),'eligible'].sum()),
                        diagnostic_reference_points=int(reference.loc[reference.population.eq('diagnostic_with_gap_interpolation'),'eligible'].sum())))
        return dict(output_path=str(run.path),summary=summary,reference=reference,intervals=intervals,comparison=comparison,
                    windows=pd.DataFrame(windows),noise_windows=pd.DataFrame(noise_audit),input_issues=issues,figures=figures)
