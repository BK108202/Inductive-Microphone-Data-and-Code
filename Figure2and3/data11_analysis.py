"""Read-only Data11 analyses. Each call archives an exclusive reproducible run."""
from __future__ import annotations
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
from pathlib import Path
import json
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize
from characterization_run import AnalysisRun
from resolution_sine_fit import fit_sine_frequency
from data11_config import PROJECT, amplitude_files, amplitude_bounds, amplitude_analysis_bounds, amplitude_subwindow_bounds
from portable_runtime import config_source, ensure_environment, load_config, source_paths

METHODS = {
 'amplitude_global_peak': 'Additional amplitude analysis selects the largest padded FFT BIN amplitude strictly above config.global_search_min_hz, through the recorded-grid Nyquist, in each full trimmed level window. No target-band restriction or fallback; the cutoff is peak selection, not filtering. Same linear interpolation, demean, periodic Hann and one-sided peak normalization as existing FFT metrics. Native/padded bin and 3-point interpolated estimates are retained separately; new response uses padded BIN, not interpolation. Peak check compares selected bin frequency with target_hz +/- amplitude_peak_check_halfwidth_hz only AFTER selection. All peaks, including off-target peaks, are plotted/exported. Response means/sample SD are across independent records, never subwindows. Global-peak ratios/OLS require all selected records from both sensors, finite peaks, accelerometer extrema gate, confirmed reference conditions, and all selected peaks within the check band; strict results additionally require gap_ok. Frequencies outside that band are not treated as a common 100Hz response. Diagnostic ratios retain gap warnings. Check-band membership is not a synchronization or calibration claim. Original target-conditioned response remains separately available.',
 'amplitude_selection': 'Amplitude files are explicitly listed per sensor; max_files takes the first N in that list (None=all, 0=disabled). No automatic replacement for missing/invalid files. inputs.json records selected paths. Summary n_requested is selected count and n is available count per level. Independent sensor counts may differ; ratios require every selected record in both sensors to pass the stated gates. A single record has undefined sample SD (NaN), and no SD bar. Timing and metrics are unchanged by file selection.',
 'scope': 'This shared dictionary documents all three notebooks. Only the analysis named in parameters.config.analysis is executed. Literal numeric values describe defaults; parameters.config records effective editable settings.',
 'aggregation': 'Sweep groups: arithmetic mean and sample SD (ddof=1) across independent sweep records at each grid frequency, including flagged estimates in descriptive summaries. SNR group values are the mean of per-record dB ratios (not dB of a ratio of group means); each per-record noise estimate is RMS of quadrature magnitudes across its recorded baseline windows. Ratio curves instead use independent condition-mean signal peak amplitudes. No overlapping window or subwindow is counted as an independent experimental repeat.',
 'input': 'CSV Time_us minus first timestamp, multiplied by 1e-6. IM Data28 stays counts; accel Raw_Z multiplied by user-confirmed 0.000488 g/count. No raw writes, sorting, outlier removal or timestamp repair.',
 'timing': 'Reject nonfinite or nonincreasing timestamps. Audit fs=(N-1)/span, median dt, CV, largest dt/median dt. Local windows with gaps >3 median intervals are ineligible for strict summaries; estimates remain diagnostic. Timestamp accuracy is uncalibrated. Analog bandwidth unknown.',
 'fft': 'Linear interpolation of retained samples to N equally spaced timestamps including endpoints, demean, periodic Hann. One-sided peak amplitude 2*abs(rFFT)/sum(window), DC and Nyquist not doubled. Native, native 3-point log-parabola, 8x padded bin and padded 3-point log-parabola are separate. Resolution also retains 5-point fits. Global search excludes DC; target search is separately labeled. No extrapolation. FFT interpolation may attenuate high-frequency amplitudes on irregular timestamps.',
 'chirp': 'Fit original timestamps with Hann-in-time weighted least squares: intercept + local linear trend + cos(phi) + sin(phi), phi=2*pi*f_center*expm1(rate*tau)/rate. Peak amplitude=hypot(quadratures). Fit fraction is weighted SSE reduction relative to intercept+trend; not SNR. No bandpass filter or resampling for amplitudes.',
 'trajectory': 'Diagnostic spectrogram: interpolated samples, 0.5s periodic Hann, 0.1s hop, 4x FFT. Search log chirp start and duration on configured grid against spectral energy, then refine local raw-timestamp chirp frequencies. Robust log-frequency regression using coherent windows. Review fitted trajectory over spectrogram; start and end are model-inferred crossings, not manually confirmed switch times. Partial sweeps allowed; never extrapolate response.',
 'snr_matched': 'Signal=chirp peak amplitude in 0.8s window. Noise=sqrt(mean(A_noise**2)) of same chirp template applied to nonoverlapping 0.8s independent baseline windows after 5s, up to 40 evenly spaced windows. No frequency optimization on noise. Linear=A_signal/A_noise_equiv; dB=20*log10(linear). Both nothing-on and amplifier-on controls retained. This is estimator-equivalent SNR, with positive amplitude-estimator bias, not acoustic or conventional broadband SNR. Overlapping signal windows are not independent repeats.',
 'bandwidth': 'Hann-in-time weighting: nominal ENBW=1.5/T (1.875 Hz for T=0.8s) in dechirped coordinates; save sampled approximation per window. Actual regression includes nuisance intercept/trend and irregular sample times. Save physical frequency excursion f_high-f_low in each chirp window. No fixed +/-1 Hz stationary band is claimed.',
 'snr_broadband': 'Separate metric: fitted chirp peak/sqrt(2) divided by sqrt(mean(noise-window demeaned RMS**2)). Same baseline windows; original-sample, full recorded-band RMS, no filtering. Save linear and 20log10 ratio for each baseline. Noise bandwidth limited by recorded sampling and unknown analog chain; nominal Nyquist saved. This metric must not replace matched SNR.',
 'reference': 'Compare independent condition means at common frequency/volume; never pair IM_i and accel_i. IM counts / acceleration peak g yields nominal reference-normalized counts/g when conditions confirmed. SD is across independent records, not windows. Reference must pass fit, coverage, local-gap and (sweep) amplifier-on SNR gates. Calibration, clock error, mounting repeatability and between-batch drift not included. Not pressure sensitivity, phase, coherence or a synchronized transfer function.',
 'range': 'Primary per-frequency summaries use records that pass local fit/gap and numeric-endpoint checks, with at least 2 eligible independent records; n_total, n and n_excluded are exported. SNR definition unchanged. All-record diagnostic summaries are retained separately. Two populations are exported: Q-screened (original filenames) and Q-retained (_q_retained suffix). Only the local response Q gate is disabled in the latter; trajectory identification, gap, endpoint and repeat gates are identical. n_low_q and hollow plot markers identify means containing low-Q records (either sensor for counts/g). Threshold-supported intervals require mean amplifier-on matched SNR strictly > configured 10 dB for each population. Low-Q retained intervals are descriptive threshold support, not validated response. Adjacent passing grid points form intervals; holes stay holes. Threshold is an analysis setting, not statistical detection or -3 dB acoustic bandwidth. Counts/g relative-to-100-Hz shape is separate from signal SNR.',
 'numeric_endpoints': 'IM Data28 endpoint codes 0 and 2^28-1=268435455 are flagged by exact numeric equality. Default conservative primary population excludes entire endpoint-bearing IM records, because these records also show huge excursions in windows without exact endpoint codes. response_window_ok preserves the less strict local window criterion separately. im_endpoint_scope=window is available only as an explicitly different configured population. No correction, clipping or sample deletion occurs. The underlying hardware/firmware cause is unverified. All computed estimates including endpoint-contaminated amplitude/SNR remain in sweep_records and all-record diagnostic summaries, never evidence of better sensitivity. Audit full-record and window endpoint counts; insufficient remaining repeats means unavailable, not zero response.',
 'resolution': 'Primary new figures use global_above_cutoff_*: search every native/padded FFT bin strictly above config.global_search_min_hz (user selected 1Hz), up to the recorded-grid Nyquist, without a nominal target band or fallback. This is peak selection, not high-pass filtering. Preserve original global_* (all f>0) and separate target_* (common 99-101Hz band) estimators. Preserve native bins, native 3/5-point log fits, padded bins, padded 3/5-point log fits, and original-timestamp sine fits for each family. Sine fit is local refinement seeded by the observed native maximum of its family; above-cutoff halfwidth is +/-2 native bins capped to remain above cutoff and below Nyquist. No nominal-frequency seed or narrow mHz prior. 50s primary and 40s trimmed windows are not independent repeats. Report mean/sample SD and exploratory unpaired Welch intervals. Relative-frequency display subtracts a per-window/per-method baseline mean only; absolute estimates and ppm stay uncorrected. Single-tone repeatability is distinct from two-tone resolution; padding does not establish 0.005Hz resolvability. Timebase/generator accuracy remain unresolved.',
 'amplitude': 'Six volume blocks 25->20; per-file dwell_overrides_s takes precedence over global dwell_s, and explicit bounds_overrides_s takes precedence over both. User confirmed IM_15-17: 20s each, IM_53-56: 30s each. Default central_window_s=16 centers a 16s primary window in every block (20s: 2..18s; 30s: 7..23s). Per-file central_window_overrides_s takes precedence; notebook controls may choose 26s for 30s IM blocks while keeping 16s for 20s accelerometer blocks. Complete equal-duration stability subwindows omit and report any trailing remainder; the primary estimate uses the whole central window. Explicit central_window_s=None retains legacy trim_s behavior; start=0 remains an assumption unless overridden. Preserve raw peak-to-peak, demeaned RMS, four global FFT estimates and target-conditioned 100+/-1Hz estimate. Save 4s subwindows for within-record stability, not independent replication. Explicit boundaries and overview figures allow review. Volume is a control setting, not calibrated input amplitude. Strict ratios require all records to pass max-gap gate. Additional explicitly diagnostic ratios retain finite target estimates with gap warnings; eligible_diagnostic and gap duration report this distinction. Free-intercept OLS of condition-mean IM vs accel is descriptive, separately for strict/diagnostic populations; slope counts/g distinct from pointwise ratios. No SNR or detection limit inferred for amplitude/resolution.',
}

def manifest(c):
    rows=[]
    def add(sensor, i, role, **extra):
        name = i if isinstance(i,str) else f'{sensor}_{i}.csv'
        rows.append(dict(path=str((Path(c['data_root'])/name).resolve()), file=name, sensor=sensor, role=role, **extra))
    if c['analysis']=='frequency_range':
        for material,(noise,sweeps) in c['materials'].items():
            for cond,i in zip(['nothing_on','amplifier_on'],noise): add('IM',i,'noise',material=material,condition=cond)
            for i in sweeps: add('IM',i,'sweep',material=material)
        for cond,i in zip(['nothing_on','amplifier_on'],c['accel_noise']): add('accel',i,'noise',material='accelerometer',condition=cond)
        for i in c['accel_sweeps']: add('accel',i,'sweep',material='accelerometer')
    elif c['analysis']=='resolution':
        for name,f in c['files'].items(): add('IM',name,'resolution',nominal_hz=f)
    else:
        for sensor,ids in amplitude_files(c).items():
            for i in ids: add(sensor,i,'amplitude')
    return rows

def load_record(item,c):
    d=pd.read_csv(item['path'])
    col='Data28' if item['sensor']=='IM' else 'Raw_Z'
    t=d['Time_us'].to_numpy(float); y=d[col].to_numpy(float)
    if len(t)<8 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError('nonfinite or insufficient samples')
    t=(t-t[0])*1e-6
    if np.any(np.diff(t)<=0): raise ValueError('timestamps not strictly increasing; no repair performed')
    scale=1. if item['sensor']=='IM' else c['accel_g_per_count']
    if scale is None or scale<=0: raise ValueError('explicit positive accelerometer conversion required')
    dt=np.diff(t)
    audit=dict(file=item['file'],samples=len(t),duration_s=t[-1],fs_hz=(len(t)-1)/t[-1],
               dt_median_s=np.median(dt),dt_cv=np.std(dt)/np.mean(dt),max_gap_ratio=dt.max()/np.median(dt),
               gap_count=int(np.sum(dt>c['max_gap_ratio']*np.median(dt))),
               unit='counts' if item['sensor']=='IM' else 'g',scale=scale,
               timestamp_valid=True)
    audit.update(raw_min=y.min(),raw_max=y.max(),endpoint_code_count=int(np.isin(y,c['im_endpoint_codes']).sum()) if item['sensor']=='IM' else 0)
    return dict(t=t,y=y*scale,**item,audit=audit)

def window(r,a,b):
    t=r['t']; dt=np.median(np.diff(t))
    if a<0 or b>t[-1]+dt or b<=a: raise ValueError(f'window [{a},{b}) outside recorded support [0,{t[-1]}]')
    lo,hi=np.searchsorted(t,[a,b]); tt=t[lo:hi]; yy=r['y'][lo:hi]
    if len(tt)<8 or tt[0]-a>2*dt or b-tt[-1]>2*dt: raise ValueError('insufficient boundary coverage')
    return tt,yy

def local_quality(t,c):
    dt=np.diff(t); fs=(len(t)-1)/(t[-1]-t[0])
    gap=dt.max()/np.median(dt)
    gaps=dt>c['max_gap_ratio']*np.median(dt)
    return dict(samples=len(t),fs_hz=fs,dt_cv=np.std(dt)/np.mean(dt),max_gap_ratio=gap,
                max_interval_s=dt.max(),gap_interval_count=int(gaps.sum()),
                gap_duration_fraction=float(dt[gaps].sum()/(t[-1]-t[0])),
                gap_ok=bool(gap<=c['max_gap_ratio']),nominal_nyquist_hz=fs/2)

def chirp_fit(t,y,center,f,rate,duration):
    tau=t-center; w=np.maximum(0,0.5+0.5*np.cos(2*np.pi*tau/duration))
    phase=2*np.pi*f*(np.expm1(rate*tau)/rate if abs(rate)>1e-12 else tau)
    m=np.column_stack([np.ones(len(t)),tau,np.cos(phase),np.sin(phase)])
    root=np.sqrt(w); mw=m*root[:,None]; yw=(y-np.mean(y))*root
    coef=np.linalg.lstsq(mw,yw,rcond=None)[0]
    err=yw-mw@coef; base=np.linalg.lstsq(mw[:,:2],yw,rcond=None)[0]
    total=np.sum((yw-mw[:,:2]@base)**2)
    frac=1-np.sum(err**2)/total if total>0 else 0.
    fs=(len(t)-1)/(t[-1]-t[0])
    return dict(amplitude_peak=float(np.hypot(*coef[2:])),fit_fraction=float(frac),
                enbw_hz=float(fs*np.sum(w*w)/np.sum(w)**2))

def spectrum(t,y,pad=1):
    u=np.linspace(t[0],t[-1],len(t)); fs=(len(t)-1)/(t[-1]-t[0])
    v=np.interp(u,t,y); v-=v.mean(); w=signal.windows.hann(len(v),sym=False)
    n=len(v)*pad; f=np.fft.rfftfreq(n,1/fs); a=2*np.abs(np.fft.rfft(v*w,n=n))/w.sum()
    a[0]/=2
    if n%2==0: a[-1]/=2
    return f,a,fs*np.sum(w*w)/w.sum()**2

def peak(f,a,band=None,points=3,min_frequency_hz=0.):
    valid=f>min_frequency_hz
    if band is not None: valid&=(f>=band[0])&(f<=band[1])
    ids=np.flatnonzero(valid)
    if not len(ids): return dict(bin_hz=np.nan,bin_amplitude=np.nan,fit_hz=np.nan,fit_amplitude=np.nan,status='empty_band')
    i=ids[np.argmax(a[ids])]; out=dict(bin_hz=f[i],bin_amplitude=a[i],fit_hz=np.nan,fit_amplitude=np.nan,status='edge')
    h=points//2
    if i-h<1 or i+h>=len(a) or i==ids[0] or i==ids[-1]: return out
    x=np.arange(-h,h+1); v=np.log(np.maximum(a[i-h:i+h+1],np.finfo(float).tiny))
    q=np.polyfit(x,v,2)
    delta=-q[1]/(2*q[0]) if q[0]<0 else np.inf
    ff=f[i]+delta*(f[1]-f[0])
    if abs(delta)>1 or ff<=min_frequency_hz or (band and not band[0]<=ff<=band[1]): return out
    out.update(fit_hz=ff,fit_amplitude=np.exp(np.polyval(q,delta)),status='ok')
    return out

def tone_metrics(t,y,c,target=None):
    out=local_quality(t,c)
    out.update(raw_peak_to_peak=np.ptp(y),raw_demeaned_rms=np.std(y),
               extrema_fraction=max(np.mean(y==y.min()),np.mean(y==y.max())),
               actual_start_s=t[0],actual_end_s=t[-1],native_df_hz=out['fs_hz']/len(t))
    specs={}
    for name,pad in [('native',1),('padded',c['fft_pad_factor'])]:
        f,a,enbw=spectrum(t,y,pad); specs[name]=(f,a)
        out['enbw_hz']=enbw
        p=peak(f,a)
        for k,v in p.items(): out[f'global_{name}_{k}']=v
        if c['analysis']=='amplitude':
            cutoff=c.get('global_search_min_hz',1.0)
            if not np.isfinite(cutoff) or cutoff<0: raise ValueError('global_search_min_hz must be finite and nonnegative')
            selected=peak(f,a,min_frequency_hz=cutoff)
            for k,v in selected.items(): out[f'global_above_cutoff_{name}_{k}']=v
        p5=peak(f,a,points=5)
        out[f'global_{name}_5point_hz']=p5['fit_hz']
        if target is not None:
            p=peak(f,a,[target-c['target_halfwidth_hz'],target+c['target_halfwidth_hz']])
            for k,v in p.items(): out[f'target_{name}_{k}']=v
            out[f'target_{name}_5point_hz']=peak(f,a,[target-c['target_halfwidth_hz'],target+c['target_halfwidth_hz']],points=5)['fit_hz']
    out['padded_df_hz']=out['native_df_hz']/c['fft_pad_factor']
    return out,specs

def save_fig(run,fig,name,figures):
    from paper_figure_style import save_paper_figure
    config = run.parameters.get('config', {})
    # Review/raw exports nest the effective tracking configuration.
    font_size = config.get('figure_font_size_pt',
                           config.get('tracking', {}).get('figure_font_size_pt', 15))
    save_paper_figure(run,fig,name,figures,dpi=145,font_size=font_size)

def overview(run,r,figures,bounds=None):
    # Min/max in contiguous sample buckets preserves excursions for a compact view.
    t,y=r['t'],r['y']; edges=np.linspace(0,len(t),min(1800,len(t))+1,dtype=int)
    tx=[]; low=[]; high=[]
    for a,b in zip(edges[:-1],edges[1:]):
        tx.append(np.mean(t[a:b])); low.append(y[a:b].min()); high.append(y[a:b].max())
    fig,ax=plt.subplots(figsize=(11,3.2)); ax.fill_between(tx,low,high,alpha=.65)
    if bounds:
        for level,a,b in bounds:
            ax.axvline(a,color='k',alpha=.3); ax.text((a+b)/2,1,str(level),transform=ax.get_xaxis_transform(),ha='center')
    ax.set(xlabel='Recorded time (s)',ylabel=r['audit']['unit'],title=r['file']+' | raw min/max envelope; inspect transitions')
    save_fig(run,fig,Path(r['file']).stem+'_overview',figures)

def fit_trajectory(r,c):
    t,y=r['t'],r['y']; fs=r['audit']['fs_hz']; u=np.linspace(0,t[-1],len(t))
    n=round(.5*fs); hop=round(.1*fs)
    f,tt,p=signal.spectrogram(np.interp(u,t,y),fs=fs,window='hann',nperseg=n,noverlap=n-hop,nfft=4*n,scaling='density')
    use=(f>=25)&(f<=330); f=f[use]; p=p[use]
    pnorm=p/np.maximum(p.sum(axis=0),np.finfo(float).tiny)
    best=(-np.inf,None,None)
    for duration in np.arange(c['duration_search_s'][0],c['duration_search_s'][1]+.01,.5):
        rate=np.log(c['f1_hz']/c['f0_hz'])/duration
        for start in np.arange(c['start_search_s'][0],c['start_search_s'][1]+.01,.1):
            model=c['f0_hz']*np.exp(rate*(tt-start)); keep=(tt>=start+.4)&(tt<=start+duration-.4)&(model>=32)&(model<=280)
            ids=np.flatnonzero(keep)
            if len(ids)<50: continue
            vals=np.array([np.interp(model[j],f,pnorm[:,j]) for j in ids])
            score=np.mean(vals)
            if score>best[0]: best=(score,start,duration)
    _,start,duration=best
    if start is None: raise ValueError('no supported log-sweep trajectory')
    start=c['onset_overrides_s'].get(r['file'],start)
    duration=c['duration_overrides_s'].get(r['file'],duration)
    rate=np.log(c['f1_hz']/c['f0_hz'])/duration
    points=[]; half=c['window_s']/2
    for center in np.arange(max(half,start+half),min(t[-1]-half,start+duration-half),c['track_hop_s']):
        try: ts,ys=window(r,center-half,center+half)
        except ValueError as exc:
            points.append(dict(center_s=center,local_hz=np.nan,amplitude_peak=np.nan,fit_fraction=0.,window_status=str(exc)))
            continue
        seed=c['f0_hz']*np.exp(rate*(center-start))
        width=max(2.5,seed*.015); grid=np.linspace(seed-width,seed+width,17)
        scores=[chirp_fit(ts,ys,center,x,rate,c['window_s'])['fit_fraction'] for x in grid]
        k=int(np.argmax(scores))
        if k in (0,len(grid)-1): continue
        opt=optimize.minimize_scalar(lambda x:-chirp_fit(ts,ys,center,x,rate,c['window_s'])['fit_fraction'],bounds=(grid[k-1],grid[k+1]),method='bounded',options={'xatol':1e-6})
        fit=chirp_fit(ts,ys,center,opt.x,rate,c['window_s'])
        endpoints=int(np.isin(ys,c['im_endpoint_codes']).sum()) if r.get('sensor')=='IM' else 0
        points.append(dict(center_s=center,local_hz=opt.x,**fit,endpoint_code_count=endpoints,window_status='endpoint_code_present' if endpoints else 'ok'))
    track=pd.DataFrame(points)
    if track.empty: raise ValueError('empty coherent sweep track')
    good=(track.fit_fraction>=c['track_min_fraction']) & track.window_status.eq('ok')
    if good.sum()<c['track_min_points'] or np.ptp(track.loc[good,'center_s'])<c['track_min_span_s']:
        raise ValueError('insufficient coherent log-sweep support')
    xx=track.loc[good,'center_s'].to_numpy(); yy=np.log(track.loc[good,'local_hz'].to_numpy())
    fit=optimize.least_squares(lambda b:b[0]+b[1]*xx-yy,[np.log(c['f0_hz'])-rate*start,rate],loss='soft_l1',f_scale=.002)
    intercept,rate=fit.x
    duration=np.log(c['f1_hz']/c['f0_hz'])/rate; start=(np.log(c['f0_hz'])-intercept)/rate
    if r['file'] in c['duration_overrides_s'] or r['file'] in c['onset_overrides_s']:
        start=c['onset_overrides_s'].get(r['file'],start); duration=c['duration_overrides_s'].get(r['file'],duration)
        rate=np.log(c['f1_hz']/c['f0_hz'])/duration; intercept=np.log(c['f0_hz'])-rate*start
    residual=np.sqrt(np.mean((intercept+rate*xx-yy)**2))
    ok=bool(c['duration_search_s'][0]<=duration<=c['duration_search_s'][1] and c['start_search_s'][0]<=start<=c['start_search_s'][1] and residual<=c['track_max_log_residual'])
    track['model_hz']=np.exp(intercept+rate*track.center_s)
    meta=dict(file=r['file'],start_s=start,duration_s=duration,end_s=start+duration,rate=rate,
              log_frequency_rmse=residual,trajectory_ok=ok,coherent_points=int(good.sum()),
              record_duration_s=t[-1],end_truncated=bool(start+duration>t[-1]),start_grid_score=best[0])
    return meta,track,(f,tt,p)

def noise_windows(r,c):
    dur=c['window_s']; centers=np.arange(c['noise_start_s']+dur/2,r['t'][-1]-dur/2,dur)
    if len(centers)>c['max_noise_windows']:
        centers=centers[np.unique(np.round(np.linspace(0,len(centers)-1,c['max_noise_windows'])).astype(int))]
    windows=[]
    for center in centers:
        try: t,y=window(r,center-dur/2,center+dur/2)
        except ValueError: continue
        q=local_quality(t,c)
        if q['gap_ok']: windows.append((t-center,y,center,q))
    if len(windows)<2: raise ValueError('fewer than two gap-qualified noise windows')
    return windows

def frequency_analysis(run,records,c,figures):
    rows=[]; trajectories=[]; noise_audit=[]; failed=[]
    baseline={}
    for r in records.values():
        if r['role']=='noise':
            try: wins=noise_windows(r,c)
            except ValueError as exc:
                failed.append(dict(file=r['file'],reason=str(exc))); continue
            baseline[(r['material'],r['condition'])]=wins
            for _,y,center,q in wins:
                noise_audit.append(dict(file=r['file'],condition=r['condition'],center_s=center,window_s=c['window_s'],raw_demeaned_rms=np.std(y),**q))
    pd.DataFrame(noise_audit).to_csv(run.path/'noise_windows.csv',index=False)
    grid=np.geomspace(c['f0_hz'],c['f1_hz'],c['grid_points']); half=c['window_s']/2
    for r in records.values():
        if r['role']!='sweep': continue
        print('Sweep',r['file'],flush=True)
        try: meta,track,spec=fit_trajectory(r,c)
        except ValueError as exc:
            failed.append(dict(file=r['file'],reason=str(exc))); continue
        trajectories.append(meta); track.to_csv(run.path/(Path(r['file']).stem+'_track.csv'),index=False)
        fig,ax=plt.subplots(figsize=(10,4)); f,tt,p=spec
        ax.pcolormesh(tt,f,10*np.log10(np.maximum(p,1e-30)),shading='auto',cmap='viridis')
        ax.plot(track.center_s,track.model_hz,'w-',lw=1.5,label='fitted log trajectory'); ax.scatter(track.center_s,track.local_hz,s=5,c='red',label='local fits')
        ax.set(xlabel='Recorded time (s)',ylabel='Hz',ylim=(25,320),title=f"{r['file']} | inferred start {meta['start_s']:.3f}s, duration {meta['duration_s']:.3f}s; accepted={meta['trajectory_ok']}")
        ax.legend(); save_fig(run,fig,Path(r['file']).stem+'_trajectory',figures)
        if not meta['trajectory_ok']:
            failed.append(dict(file=r['file'],reason='trajectory quality gate failed')); continue
        rate=meta['rate']
        for j,freq in enumerate(grid):
            center=meta['start_s']+np.log(freq/c['f0_hz'])/rate
            if center-half<meta['start_s'] or center+half>min(meta['end_s'],r['t'][-1]): continue
            try: t,y=window(r,center-half,center+half)
            except ValueError: continue
            fit=chirp_fit(t,y,center,freq,rate,c['window_s']); q=local_quality(t,c)
            endpoints=int(np.isin(y,c['im_endpoint_codes']).sum()) if r['sensor']=='IM' else 0
            endpoint_scope_ok=(r['audit']['endpoint_code_count']==0 if c['im_endpoint_scope']=='record' else endpoints==0)
            local_ok=bool(q['gap_ok'] and fit['fit_fraction']>=c['response_min_fraction'] and endpoints==0)
            row=dict(file=r['file'],sensor=r['sensor'],material=r['material'],grid_index=j,frequency_hz=freq,center_s=center,
                     window_start_s=center-half,window_end_s=center+half,**fit,**q,
                     unit=r['audit']['unit'],frequency_excursion_hz=2*freq*np.sinh(rate*half),
                     endpoint_code_count=endpoints,
                     record_endpoint_code_count=r['audit']['endpoint_code_count'],response_window_ok=local_ok,
                     fit_quality_ok=bool(fit['fit_fraction']>=c['response_min_fraction']),
                     response_without_q_ok=bool(q['gap_ok'] and endpoints==0 and endpoint_scope_ok),
                     response_ok=bool(local_ok and endpoint_scope_ok),samples_per_cycle=q['fs_hz']/freq)
            for condition in ['nothing_on','amplifier_on']:
                wins=baseline.get((r['material'],condition),[])
                amps=[chirp_fit(tn,yn,0,freq,rate,c['window_s'])['amplitude_peak'] for tn,yn,_,_ in wins]
                noise=np.sqrt(np.mean(np.square(amps))) if amps else np.nan
                broad=np.sqrt(np.mean([np.var(yn) for _,yn,_,_ in wins])) if amps else np.nan
                linear=fit['amplitude_peak']/noise if noise>0 else np.nan
                blin=fit['amplitude_peak']/np.sqrt(2)/broad if broad>0 else np.nan
                row.update({condition+'_noise_equivalent_peak':noise,condition+'_noise_rms':broad,
                            condition+'_snr_linear':linear,condition+'_snr_db':20*np.log10(linear),
                            condition+'_broadband_snr_linear':blin,condition+'_broadband_snr_db':20*np.log10(blin),condition+'_noise_window_count':len(wins)})
            rows.append(row)
        pd.DataFrame(rows).to_csv(run.path/'sweep_records.csv',index=False)
        pd.DataFrame(trajectories).to_csv(run.path/'trajectory_summary.csv',index=False)
    run.write_json('unavailable_records.json',failed)
    if not rows: raise ValueError('no sweep windows qualified for computation')
    detail=pd.DataFrame(rows)
    result=frequency_outputs(run,detail,c,figures,retain_low_q=False)
    result['q_retained']=frequency_outputs(run,detail,c,figures,retain_low_q=True)
    result['trajectories']=pd.DataFrame(trajectories)
    return result


def frequency_outputs(run,detail,c,figures,retain_low_q=False):
    """Compare populations from identical estimates; only the local Q gate differs."""
    grid=np.geomspace(c['f0_hz'],c['f1_hz'],c['grid_points'])
    suffix='_q_retained' if retain_low_q else ''
    population='q_retained' if retain_low_q else 'q_screened'
    q_label=f"Q < {100*c['response_min_fraction']:g}% retained" if retain_low_q else f"Q >= {100*c['response_min_fraction']:g}%"
    eligibility='response_without_q_ok' if retain_low_q else 'response_ok'
    def csv(frame,name):
        frame.to_csv(run.path/(name+suffix+'.csv'),index=False)
    groups=[]; qualified=[]
    for (material,j),part in detail.groupby(['material','grid_index']):
        row=dict(material=material,grid_index=j,frequency_hz=grid[j],n=len(part),n_quality=int(part.response_ok.sum()),
                 all_quality=bool(part.response_ok.all()))
        for key in ['amplitude_peak','nothing_on_snr_db','amplifier_on_snr_db','nothing_on_broadband_snr_db','amplifier_on_broadband_snr_db']:
            row[key+'_mean']=part[key].mean(); row[key+'_sd']=part[key].std(ddof=1)
        row['supported']=bool(row['all_quality'] and row['n']>=c['minimum_repeat_count'] and row['amplifier_on_snr_db_mean']>c['snr_threshold_db'])
        groups.append(row)
        accepted=part[part[eligibility]]
        primary=dict(material=material,grid_index=j,frequency_hz=grid[j],n_total=len(part),n=len(accepted),n_excluded=len(part)-len(accepted),
                     n_endpoint_flagged=int(part.endpoint_code_count.gt(0).sum()),all_quality=bool(len(accepted)>=c['minimum_repeat_count']),
                     population=population,n_low_q=int((~accepted.fit_quality_ok).sum()))
        for key in ['amplitude_peak','nothing_on_snr_db','amplifier_on_snr_db','nothing_on_broadband_snr_db','amplifier_on_broadband_snr_db']:
            primary[key+'_mean']=accepted[key].mean() if len(accepted)>=c['minimum_repeat_count'] else np.nan
            primary[key+'_sd']=accepted[key].std(ddof=1) if len(accepted)>=c['minimum_repeat_count'] else np.nan
        primary['supported']=bool(primary['all_quality'] and primary['amplifier_on_snr_db_mean']>c['snr_threshold_db'])
        qualified.append(primary)
    diagnostic_summary=pd.DataFrame(groups)
    if not retain_low_q: csv(diagnostic_summary,'sweep_all_record_diagnostic_summary')
    summary=pd.DataFrame(qualified); csv(summary,'sweep_condition_summary')
    reference=summary[summary.material.eq('accelerometer')].set_index('grid_index'); ratios=[]
    for row in summary[~summary.material.eq('accelerometer')].to_dict('records'):
        j=row['grid_index']; rr=dict(material=row['material'],grid_index=j,frequency_hz=row['frequency_hz'],counts_per_g=np.nan,status='reference_unavailable',population=population,includes_low_q=bool(row['n_low_q']))
        if j in reference.index:
            ref=reference.loc[j]
            rr['includes_low_q']=bool(row['n_low_q'] or ref.n_low_q)
            ok=row['all_quality'] and row['n']>=c['minimum_repeat_count'] and bool(ref.supported) and c['reference_conditions_confirmed']
            rr.update(IM_n=row['n'],accel_n=int(ref.n),IM_mean_counts=row['amplitude_peak_mean'],accel_mean_g=ref.amplitude_peak_mean,
                      status='eligible_nominal_reference_response' if ok else 'quality_or_reference_gate_failed')
            if ok and ref.amplitude_peak_mean>0: rr['counts_per_g']=row['amplitude_peak_mean']/ref.amplitude_peak_mean
        ratios.append(rr)
    ratios=pd.DataFrame(ratios)
    for _,ix in ratios.groupby('material').groups.items():
        part=ratios.loc[ix].sort_values('frequency_hz'); valid=part.counts_per_g.notna(); good=part[valid]
        # Normalize only with immediate valid grid neighbors bracketing 100 Hz.
        below=good[good.frequency_hz<=100]; above=good[good.frequency_hz>=100]
        if len(below) and len(above) and int(above.iloc[0].grid_index-below.iloc[-1].grid_index)<=1:
            anchor=np.interp(100,good.frequency_hz,good.counts_per_g)
            ratios.loc[ix,'relative_to_100Hz_db']=20*np.log10(ratios.loc[ix,'counts_per_g']/anchor)
    csv(ratios,'reference_normalized_response')
    intervals=[]
    for material,part in summary.groupby('material'):
        ids=part.loc[part.supported,'grid_index'].to_numpy(int)
        for chunk in np.split(ids,np.flatnonzero(np.diff(ids)>1)+1):
            if len(chunk): intervals.append(dict(material=material,start_hz=grid[chunk[0]],end_hz=grid[chunk[-1]],grid_points=len(chunk),threshold_db=c['snr_threshold_db'],threshold_operator='>',population=population))
    interval_table=pd.DataFrame(intervals,columns=['material','start_hz','end_hz','grid_points','threshold_db','threshold_operator','population'])
    csv(interval_table,'supported_frequency_intervals')
    for name,materials in c['comparison_groups'].items():
        fig,axes=plt.subplots(3,1,figsize=(11,11),sharex=True)
        low_q_markers=[]
        for color_index,material in enumerate(materials):
            color=plt.get_cmap('tab10')(color_index%10)
            p=summary[summary.material.eq(material)].sort_values('frequency_hz')
            if p.empty: continue
            label=material if p.amplitude_peak_mean.notna().any() else material+' (unavailable)'
            axes[0].plot(p.frequency_hz,p.amplitude_peak_mean,label=label,color=color)
            axes[0].fill_between(p.frequency_hz,p.amplitude_peak_mean-p.amplitude_peak_sd,p.amplitude_peak_mean+p.amplitude_peak_sd,alpha=.13,color=color)
            axes[1].plot(p.frequency_hz,p.amplifier_on_snr_db_mean,label=label,color=color)
            axes[1].plot(p.frequency_hz,p.nothing_on_snr_db_mean,ls=':',alpha=.6,color=color)
            rr=ratios[ratios.material.eq(material)]; axes[2].plot(rr.frequency_hz,rr.counts_per_g,label=label,color=color)
            if retain_low_q:
                low=p[p.n_low_q.gt(0)]
                for ax,key in [(axes[0],'amplitude_peak_mean'),(axes[1],'amplifier_on_snr_db_mean'),(axes[1],'nothing_on_snr_db_mean')]:
                    low_q_markers.append(ax.scatter(low.frequency_hz,low[key],facecolors='none',edgecolors=color,s=28,zorder=3))
                low_ref=rr[rr.includes_low_q]
                low_q_markers.append(axes[2].scatter(low_ref.frequency_hz,low_ref.counts_per_g,facecolors='none',edgecolors=color,s=28,zorder=3))
        axes[0].set(ylabel='Chirp peak (IM counts)',title=f'{name} | {q_label}; n >= {c["minimum_repeat_count"]}; shading = record SD')
        axes[1].set(ylabel='Matched SNR (dB)',title=f'Solid: amplifier on; dotted: nothing on | criterion: SNR > {c["snr_threshold_db"]:g} dB'); axes[1].axhline(c['snr_threshold_db'],color='k',ls='--',alpha=.5)
        axes[2].set(ylabel='Reference response (counts/g)',xlabel='Measured-model frequency (Hz)',yscale='log')
        for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.2); ax.set_xscale('log')
        if retain_low_q: marker_note=fig.text(.5,.003,'Hollow markers: mean includes low-Q records (either sensor for counts/g); gap and endpoint rules retained',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.02,1,1)); save_fig(run,fig,'comparison_'+name+suffix,figures)
        if retain_low_q:
            # Save the same Figure without its circle annotations. Closing its
            # pyplot manager above does not discard the Figure or change data.
            for artist in low_q_markers: artist.remove()
            marker_note.set_text('Low-Q estimates retained; markers hidden; gap and endpoint rules retained')
            save_fig(run,fig,'comparison_'+name+suffix+'_no_markers',figures)
    fig,ax=plt.subplots(figsize=(11,5))
    for material,p in summary[~summary.material.eq('accelerometer')].groupby('material'):
        ax.plot(p.frequency_hz,p.amplifier_on_broadband_snr_db_mean,label=material)
    ax.set(xlabel='Hz',ylabel='Fitted signal RMS / broadband baseline RMS (dB)',title='Separate broadband metric; amplifier-on baseline | '+q_label); ax.legend(fontsize=8); ax.grid(alpha=.2)
    ax.axhline(c['snr_threshold_db'],color='k',ls='--',alpha=.5,label=f'{c["snr_threshold_db"]:g} dB reference; range uses matched SNR')
    ax.legend(fontsize=8)
    save_fig(run,fig,'broadband_snr_comparison'+suffix,figures)
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for material,p in diagnostic_summary[~diagnostic_summary.material.eq('accelerometer')].groupby('material'):
        axes[0].plot(p.frequency_hz,p.amplitude_peak_mean,label=material)
        axes[1].plot(p.frequency_hz,p.amplifier_on_snr_db_mean,label=material)
    axes[0].set(ylabel='Peak counts (log)',yscale='log',title='All-record diagnostic means: includes endpoint codes / low fit / gaps')
    axes[1].set(xlabel='Hz',ylabel='Matched amplifier-on SNR (dB)',title='Flagged high ratios are not evidence of superior sensitivity')
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
    axes[1].axhline(c['snr_threshold_db'],color='k',ls='--',alpha=.5)
    fig.tight_layout()
    if not retain_low_q: save_fig(run,fig,'all_record_diagnostic_comparison',figures)
    else: plt.close(fig)
    if 'relative_to_100Hz_db' in ratios:
        fig,ax=plt.subplots(figsize=(11,5))
        for material,p in ratios.groupby('material'): ax.plot(p.frequency_hz,p.relative_to_100Hz_db,label=material)
        ax.set(xlabel='Hz',ylabel='Reference response relative to 100 Hz (dB)',title='Relative shape only; adjacent points required for 100Hz anchor | '+q_label); ax.legend(fontsize=7); ax.grid(alpha=.2)
        save_fig(run,fig,'relative_response'+suffix,figures)
    return dict(summary=summary,diagnostic_summary=diagnostic_summary,reference=ratios,intervals=interval_table)

def resolution_analysis(run,records,c,figures):
    from scipy.stats import t as student_t
    from data11_resolution_plots import ABOVE_CUTOFF_METHODS, create_resolution_figures
    cutoff=float(c['global_search_min_hz'])
    if not np.isfinite(cutoff) or cutoff<0:
        raise ValueError('global_search_min_hz must be finite and nonnegative')
    rows=[]; failures=[]
    for r in records.values():
        print('Resolution',r['file'],flush=True); overview(run,r,figures)
        for variant,(a,b) in c['windows_s'].items():
            try: t,y=window(r,a,b)
            except ValueError as exc:
                failures.append(dict(file=r['file'],variant=variant,reason=str(exc))); continue
            metrics,specs=tone_metrics(t,y,c,c['target_center_hz'])
            fit=fit_sine_frequency(t,y,metrics['global_native_bin_hz'],c['sine_search_halfwidth_bins']*metrics['native_df_hz'],coarse_points=c['sine_coarse_points'],frequency_tolerance_hz=c['sine_tolerance_hz'])
            row=dict(file=r['file'],nominal_hz=r['nominal_hz'],variant=variant,requested_start_s=a,requested_end_s=b,**metrics)
            row.update({'sine_'+k:v for k,v in fit.items() if not isinstance(v,(list,dict))})
            target_fit=fit_sine_frequency(t,y,metrics['target_native_bin_hz'],c['sine_search_halfwidth_bins']*metrics['native_df_hz'],coarse_points=c['sine_coarse_points'],frequency_tolerance_hz=c['sine_tolerance_hz'])
            row.update({'target_sine_'+k:v for k,v in target_fit.items() if not isinstance(v,(list,dict))})
            # A separate estimator family: maximum across every FFT bin above
            # the user-selected lower bound, with no target/nominal upper band.
            row['global_search_min_hz']=cutoff
            for name,(f,amp) in specs.items():
                for k,v in peak(f,amp,min_frequency_hz=cutoff).items():
                    row[f'global_above_cutoff_{name}_{k}']=v
                p5=peak(f,amp,points=5,min_frequency_hz=cutoff)
                row[f'global_above_cutoff_{name}_5point_hz']=p5['fit_hz']
                row[f'global_above_cutoff_{name}_5point_status']=p5['status']
            seed=row['global_above_cutoff_native_bin_hz']
            halfwidth=min(c['sine_search_halfwidth_bins']*metrics['native_df_hz'],
                          seed-np.nextafter(cutoff,np.inf),metrics['nominal_nyquist_hz']-seed)
            above_fit=fit_sine_frequency(t,y,seed,halfwidth,coarse_points=c['sine_coarse_points'],frequency_tolerance_hz=c['sine_tolerance_hz'])
            row.update({'global_above_cutoff_sine_'+k:v for k,v in above_fit.items() if not isinstance(v,(list,dict))})
            row['global_peak_matches_excitation']=bool(abs(metrics['global_padded_fit_hz']-c['target_center_hz'])<=c['target_halfwidth_hz'])
            rows.append(row)
            fig,ax=plt.subplots(figsize=(8,3.4))
            for name,(f,amp) in specs.items():
                keep=abs(f-100)<.12; ax.plot(f[keep],amp[keep],'.-',label=name)
            ax.axvline(r['nominal_hz'],color='k',ls=':'); ax.set(xlabel='Hz',ylabel='FFT peak counts',title=f"{r['file']} / {variant}: global f>{cutoff:g} Hz = {row['global_above_cutoff_padded_fit_hz']:.6f} Hz\nglobal f>0 = {metrics['global_padded_fit_hz']:.6f} Hz; 100 Hz zoom is display only")
            ax.legend(); save_fig(run,fig,Path(r['file']).stem+'_'+variant+'_spectrum',figures)
        pd.DataFrame(rows).to_csv(run.path/'resolution_records.csv',index=False)
    run.write_json('unavailable_windows.json',failures)
    if not rows: raise ValueError('no supported resolution windows')
    detail=pd.DataFrame(rows); methods=['global_native_bin_hz','global_native_fit_hz','global_native_5point_hz','global_padded_bin_hz','global_padded_fit_hz','global_padded_5point_hz','sine_frequency_hz',
                                     'target_native_bin_hz','target_native_fit_hz','target_native_5point_hz','target_padded_bin_hz','target_padded_fit_hz','target_padded_5point_hz','target_sine_frequency_hz']
    methods += [name for name,_ in ABOVE_CUTOFF_METHODS]
    summaries=[]; contrasts=[]
    for (variant,freq),p in detail.groupby(['variant','nominal_hz']):
        for method in methods:
            values=pd.to_numeric(p[method],errors='coerce').dropna()
            summaries.append(dict(variant=variant,nominal_hz=freq,method=method,n=len(values),mean_hz=values.mean(),sd_hz=values.std(),mean_minus_nominal_hz=values.mean()-freq,all_gap_ok=bool(p.gap_ok.all())))
    summary=pd.DataFrame(summaries)
    for (variant,method),part in summary.groupby(['variant','method']):
        for left,right in combinations(part.sort_values('nominal_hz').to_dict('records'),2):
            n1,n2=left['n'],right['n']; v1,v2=left['sd_hz']**2/n1 if n1 else np.nan,right['sd_hz']**2/n2 if n2 else np.nan
            se=np.sqrt(v1+v2); df=(v1+v2)**2/(v1*v1/(n1-1)+v2*v2/(n2-1)) if n1>1 and n2>1 and v1+v2>0 else np.nan
            delta=right['mean_hz']-left['mean_hz']; margin=student_t.ppf(.975,df)*se
            contrasts.append(dict(variant=variant,method=method,lower_setting_hz=left['nominal_hz'],upper_setting_hz=right['nominal_hz'],nominal_difference_hz=right['nominal_hz']-left['nominal_hz'],measured_mean_difference_hz=delta,welch_se_hz=se,welch_df=df,ci95_low_hz=delta-margin,ci95_high_hz=delta+margin,
                                  note='Exploratory unpaired mean contrast; n=3, independence/normality assumed; no clock/generator uncertainty; not two-tone resolution'))
    summary.to_csv(run.path/'resolution_summary.csv',index=False); pd.DataFrame(contrasts).to_csv(run.path/'resolution_unpaired_contrasts.csv',index=False)
    for variant,p in detail.groupby('variant'):
        fig,axes=plt.subplots(2,2,figsize=(13,9))
        for ax,method in zip(axes.flat,['global_above_cutoff_padded_fit_hz','global_above_cutoff_sine_frequency_hz','global_padded_fit_hz','sine_frequency_hz']):
            for i,(freq,part) in enumerate(p.groupby('nominal_hz')):
                yy=(pd.to_numeric(part[method])-100)*1000
                ax.scatter(np.repeat(i,len(yy))+np.linspace(-.08,.08,len(yy)),yy)
                ax.errorbar(i,yy.mean(),yerr=yy.std(),fmt='ks',capsize=5)
                ax.scatter(i,(freq-100)*1000,marker='_',s=350,color='red')
            label=(f'Global f > {cutoff:g} Hz | ' if 'above_cutoff' in method else 'Global f > 0 Hz | ')+('Original-timestamp sine fit' if 'sine' in method else 'Padded 3-point fit')
            ax.set(xticks=range(3),xticklabels=['100','100.005','100.010'],xlabel='Generator setting (Hz)',ylabel='Estimate minus 100 Hz (mHz)',title=label); ax.grid(alpha=.2)
        fig.suptitle(variant+' | dots=records; bars=SD; red=nominal (not measured truth)'); fig.tight_layout()
        save_fig(run,fig,'resolution_'+variant+'_comparison',figures)
    relative_records,relative_summary=create_resolution_figures(run,detail,records,c,figures)
    return dict(summary=summary,contrasts=pd.DataFrame(contrasts),records=detail,
                relative_records=relative_records,relative_summary=relative_summary)

def amplitude_analysis(run,records,c,figures):
    selected_files = amplitude_files(c)
    requested_counts = {sensor: len(names) for sensor,names in selected_files.items()}
    rows=[]; sub=[]; failures=[]
    for r in records.values():
        print('Amplitude',r['file'],flush=True)
        bounds=amplitude_bounds(c,r['file'])
        if len(bounds)!=len(c['levels']) or any(b<=a for a,b in bounds) or any(bounds[i][0]<bounds[i-1][1] for i in range(1,len(bounds))):
            raise ValueError('amplitude bounds must be ordered, nonoverlapping, one per level')
        overview(run,r,figures,[(lev,a,b) for lev,(a,b) in zip(c['levels'],bounds)])
        analysis_bounds=amplitude_analysis_bounds(c,r['file'])
        for level,(lo,hi) in zip(c['levels'],analysis_bounds):
            try: t,y=window(r,lo,hi)
            except ValueError as exc:
                failures.append(dict(file=r['file'],level=level,reason=str(exc))); continue
            metrics,_=tone_metrics(t,y,c,c['target_hz'])
            row=dict(file=r['file'],sensor=r['sensor'],unit=r['audit']['unit'],level=level,requested_start_s=lo,requested_end_s=hi,**metrics)
            row['reference_ok']=bool(metrics['gap_ok'] and metrics['target_padded_status']=='ok' and (r['sensor']=='IM' or metrics['extrema_fraction']<c['extrema_fraction_warning']))
            row['reference_diagnostic_ok']=bool(metrics['target_padded_status']=='ok' and (r['sensor']=='IM' or metrics['extrema_fraction']<c['extrema_fraction_warning']))
            sub_bounds=amplitude_subwindow_bounds(lo,hi,c['subwindow_s'])
            row.update(analysis_duration_s=hi-lo, stability_subwindow_count=len(sub_bounds),
                       stability_unused_tail_s=hi-lo-len(sub_bounds)*c['subwindow_s'])
            rows.append(row)
            for sa,sb in sub_bounds:
                try: ts,ys=window(r,sa,sa+c['subwindow_s'])
                except ValueError: continue
                mm,_=tone_metrics(ts,ys,c,c['target_hz'])
                sub.append(dict(file=r['file'],sensor=r['sensor'],level=level,start_s=sa,end_s=sa+c['subwindow_s'],target_peak=mm['target_padded_fit_amplitude'],target_hz=mm['target_padded_fit_hz'],gap_ok=mm['gap_ok']))
        pd.DataFrame(rows).to_csv(run.path/'amplitude_records.csv',index=False)
    run.write_json('unavailable_windows.json',failures)
    if not rows: raise ValueError('no supported amplitude windows')
    detail=pd.DataFrame(rows); sub=pd.DataFrame(sub,columns=['file','sensor','level','start_s','end_s','target_peak','target_hz','gap_ok']); sub.to_csv(run.path/'amplitude_subwindows.csv',index=False)
    keys=['target_padded_fit_amplitude','raw_demeaned_rms','raw_peak_to_peak','global_native_bin_amplitude','global_native_fit_amplitude','global_padded_bin_amplitude','global_padded_fit_amplitude']
    groups=[]
    for (sensor,level),p in detail.groupby(['sensor','level']):
        row=dict(sensor=sensor,level=level,n=len(p),n_requested=requested_counts[sensor],all_reference_ok=bool(p.reference_ok.all()),all_diagnostic_ok=bool(p.reference_diagnostic_ok.all()),
                 max_gap_duration_fraction=p.gap_duration_fraction.max(),gap_warning_records=int((~p.gap_ok).sum()),unit=p.unit.iloc[0])
        for key in keys: row[key+'_mean']=p[key].mean(); row[key+'_sd']=p[key].std()
        groups.append(row)
    summary=pd.DataFrame(groups); summary.to_csv(run.path/'amplitude_summary.csv',index=False)
    ratio=[]; amp='target_padded_fit_amplitude'
    for level in sorted(c['levels']):
        p=summary[summary.level.eq(level)].set_index('sensor')
        if set(p.index)!={'IM','accel'}: continue
        im,acc=p.loc['IM'],p.loc['accel']; ok=bool(im.all_reference_ok and acc.all_reference_ok and im.n==requested_counts['IM'] and acc.n==requested_counts['accel'] and c['reference_conditions_confirmed'])
        diagnostic_ok=bool(im.all_diagnostic_ok and acc.all_diagnostic_ok and im.n==requested_counts['IM'] and acc.n==requested_counts['accel'] and c['reference_conditions_confirmed'])
        ratio.append(dict(level=level,IM_peak_counts=im[amp+'_mean'],IM_sd_counts=im[amp+'_sd'],accel_peak_g=acc[amp+'_mean'],accel_sd_g=acc[amp+'_sd'],eligible=ok,counts_per_g=im[amp+'_mean']/acc[amp+'_mean'] if ok and acc[amp+'_mean']>0 else np.nan,
                          eligible_diagnostic=diagnostic_ok,diagnostic_counts_per_g=im[amp+'_mean']/acc[amp+'_mean'] if diagnostic_ok and acc[amp+'_mean']>0 else np.nan,
                          gap_warning_records=int(im.gap_warning_records+acc.gap_warning_records),max_gap_duration_fraction=max(im.max_gap_duration_fraction,acc.max_gap_duration_fraction)))
    ratios=pd.DataFrame(ratio); ratios.to_csv(run.path/'amplitude_reference_response.csv',index=False)
    fits=[]
    if len(ratios):
        for population,mask in [('strict_gap',ratios.eligible),('diagnostic_with_gap_warnings',ratios.eligible_diagnostic)]:
            p=ratios[mask]
            if len(p)>=3 and np.ptp(p.accel_peak_g)>0:
                coeff=np.polyfit(p.accel_peak_g,p.IM_peak_counts,1); residual=p.IM_peak_counts-np.polyval(coeff,p.accel_peak_g)
                fits.append(dict(method='unweighted free-intercept OLS, descriptive condition means',population=population,n_levels=len(p),slope_counts_per_g=coeff[0],intercept_counts=coeff[1],r_squared=1-np.sum(residual**2)/np.sum((p.IM_peak_counts-p.IM_peak_counts.mean())**2),limitations='Errors in both axes and within-record level correlations not modeled; diagnostic population retains interpolation across gaps; not calibrated sensitivity'))
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        # A single independent record has no sample SD; omit its error bar.
        axes[0].errorbar(ratios.accel_peak_g,ratios.IM_peak_counts,xerr=ratios.accel_sd_g if ratios.accel_sd_g.notna().any() else None,yerr=ratios.IM_sd_counts if ratios.IM_sd_counts.notna().any() else None,fmt='o',capsize=3)
        for row in ratios.itertuples(): axes[0].annotate(str(row.level),(row.accel_peak_g,row.IM_peak_counts))
        if fits and len(p):
            x=np.linspace(p.accel_peak_g.min(),p.accel_peak_g.max(),100); axes[0].plot(x,np.polyval(coeff,x),'--')
        axes[0].set(xlabel='Accelerometer 100Hz peak (g)',ylabel='IM 100Hz peak (counts)',title='Diagnostic means with gap warnings; bars=record SD')
        axes[1].plot(ratios.level,ratios.diagnostic_counts_per_g,'o--',label='Diagnostic (gap warnings)')
        axes[1].plot(ratios.level,ratios.counts_per_g,'s',label='Strict gap-qualified')
        axes[1].set(xlabel='Volume setting',ylabel='counts/g',title='Pointwise ratio of independent means'); axes[1].legend(fontsize=8)
        fig.tight_layout(); save_fig(run,fig,'amplitude_reference_response',figures)
    pd.DataFrame(fits,columns=['method','population','n_levels','slope_counts_per_g','intercept_counts','r_squared','limitations']).to_csv(run.path/'amplitude_response_fit.csv',index=False)
    for sensor,p in summary.groupby('sensor'):
        fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
        for ax,key in zip(axes.flat,keys[:3]+['global_native_bin_amplitude','global_native_fit_amplitude','global_padded_fit_amplitude']):
            ax.errorbar(p.level,p[key+'_mean'],yerr=p[key+'_sd'] if p[key+'_sd'].notna().any() else None,fmt='o-',capsize=4); ax.set(xlabel='Volume',ylabel=p.unit.iloc[0],title=key); ax.grid(alpha=.2)
            for row in p.itertuples():
                ax.annotate(f'n={row.n}',(row.level,getattr(row,key+'_mean')),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
        used = ', '.join(detail.loc[detail.sensor.eq(sensor),'file'].unique())
        fig.suptitle(sensor+' | '+used+'\nMean +/- independent-record SD (undefined for n=1)'); fig.tight_layout(); save_fig(run,fig,sensor+'_amplitude_metrics',figures)
        fig,ax=plt.subplots(figsize=(10,4))
        for name,part in sub[sub.sensor.eq(sensor)].groupby('file'):
            ax.plot((part.start_s+part.end_s)/2,part.target_peak,'.-',label=name)
        for name in detail.loc[detail.sensor.eq(sensor),'file'].unique():
            bounds=amplitude_bounds(c,name)
            for b in sorted({edge for pair in bounds for edge in pair}): ax.axvline(b,color='k',alpha=.15)
        ax.set(xlabel='Time (s)',ylabel=p.unit.iloc[0],title=f"{c['subwindow_s']:g}s subwindow {c['target_hz']:g}Hz peak: within-record stability, not repeats")
        if ax.get_legend_handles_labels()[0]: ax.legend()
        else: ax.text(.5,.5,'No complete subwindows for selected data',transform=ax.transAxes,ha='center')
        save_fig(run,fig,sensor+'_amplitude_stability',figures)
    from data11_amplitude_plots import global_peak_outputs
    extra=global_peak_outputs(run,detail,c,figures,save_fig)
    return dict(summary=summary,reference=ratios,fit=pd.DataFrame(fits),records=detail,**extra)

def run_analysis(config,notebook_path,executed_cells=None,extra_script_paths=None):
    c=deepcopy(config)
    if c['analysis']=='amplitude':
        c.setdefault('global_search_min_hz',1.0)
        c.setdefault('amplitude_peak_check_halfwidth_hz',1.0)
    if c['im_endpoint_scope'] not in {'record','window'}: raise ValueError('im_endpoint_scope must be record or window')
    ensure_environment()
    root=Path(load_config()['results_root'])
    inputs=manifest(c); scripts=source_paths(notebook_path, extras=extra_script_paths or ())
    params=dict(config=c,methods=METHODS,results_root_source=config_source(),
                evidence_notes=['Photos show logarithmic sweep selection, 30s, start30Hz/stop300Hz and 5Vpp screen setting; not proof of delivered mechanical amplitude or per-file settings.',
                'OO-50 thickness comparison 0.23 vs4.05mm confirmed by user. On 2026-10-13 user confirmed six volume levels: IM_15-17 each20s; IM_53-56 each30s.',
                'Updated 2026-10-13: IM_53 belongs to amplitude. Sweep candidates updated for 945, 18 and large coil; IM_48, IM_3_1, IM_3_2 remain unassigned and excluded.'],
                numerical_defaults={'spectrogram_window_s':.5,'spectrogram_hop_s':.1,'spectrogram_pad':4,'trajectory_start_step_s':.1,'trajectory_duration_step_s':.5,'track_frequency_grid_points':17,'track_frequency_halfwidth':'max(2.5 Hz, 0.015*f)','trajectory_robust_loss':'soft_l1','trajectory_f_scale_log':.002,'global_fft_minimum':'strictly positive frequencies','peak_interpolation':'log amplitude quadratic, vertex within +/-1 bin','confidence_level':.95})
    figures=[]; records={}; audits=[]; issues=[]; hashes=[]
    with AnalysisRun(root,'data11_'+c['analysis'],params,inputs,scripts,repository_path=PROJECT) as run:
        print('Output directory:',run.path,flush=True)
        run.write_text('methods.txt','\n\n'.join(k+': '+v for k,v in METHODS.items()))
        if executed_cells: run.write_text('scripts/executed_notebook_cells.py','\n\n'.join(executed_cells))
        run.write_json('notebook_execution_source.json',dict(live_cells_available=bool(executed_cells),disk_notebook_archived=True,effective_config_in_parameters=True))
        for item in inputs:
            try:
                path=Path(item['path'])
                if not path.is_file(): raise FileNotFoundError(str(path))
                hashes.append(dict(path=str(path),sha256=sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size))
                r=load_record(item,c); records[item['file']]=r; audits.append(r['audit'])
            except (ValueError,FileNotFoundError,KeyError,pd.errors.ParserError) as exc:
                issues.append(dict(**item,error=str(exc),error_type=type(exc).__name__))
                if c['missing_policy']=='fail': raise
            finally:
                pd.DataFrame(audits).to_csv(run.path/'timestamp_audit.csv',index=False)
                run.write_json('input_issues.json',issues); run.write_json('input_hashes.json',hashes)
        analysis={'frequency_range':frequency_analysis,'resolution':resolution_analysis,'amplitude':amplitude_analysis}[c['analysis']]
        result=analysis(run,records,c,figures)
        run.finish(dict(available_records=len(records),input_issues=len(issues),figures=len(figures),completion_scope='available mapped inputs; inspect input_issues and unavailable windows/records for omissions'))
        result.update(output_path=str(run.path),figures=figures,audit=pd.DataFrame(audits),input_issues=issues)
    return result
