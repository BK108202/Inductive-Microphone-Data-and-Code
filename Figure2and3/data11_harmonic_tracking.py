"""Swept-sine quadrature tracking, separately defined from existing Data11 metrics.

Original CSVs are read only. The notebook is the editable entry point.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, signal

from characterization_run import AnalysisRun
from data11_config import PROJECT, default_config
from data11_analysis import load_record, manifest, window, local_quality, chirp_fit, save_fig
from portable_runtime import config_source, ensure_environment, load_config, source_paths


REFERENCES = {
    'Siemens Sine Data Reduction': 'https://community.sw.siemens.com/articles/en_US/Knowledge/Sine-Data-Reduction',
    'Siemens Sine Control: Amplitude Estimation Methods': 'https://community.sw.siemens.com/articles/en_US/Knowledge/sine-control-amplitude-estimation-methods',
    'ISO 16063-21:2003 official scope': 'https://www.iso.org/standard/27053.html',
    'ISO 16063-21 Amendment 2:2024': 'https://www.iso.org/standard/85624.html',
}

METHODS = {
    'scope': 'Data11 continuous logarithmic sweeps only. Offline, centered, noncausal quadrature tracking implementation inspired by the Siemens harmonic tracking principle; not an implementation of proprietary Siemens software. ISO 16063-21 reference comparison is design guidance only, not a compliance or calibration claim. Official public scope was consulted; full standard text was not reviewed.',
    'evidence': 'Input roles and nominal acceleration conversion inherited from data11_config.py. That file records prior user confirmation of identical drive/mounting/load, separate sensor batches, and Raw_Z*0.000488 g/count. Existing data11_analysis.py records photos of a logarithmic 30-300 Hz/30 s setting; this is not verified per-file timing or delivered amplitude. Configured search bands, gates and thresholds are analysis choices.',
    'tracking': 'Strongest STFT ridge in the configured broad frequency search band (not a nominal trajectory). Deterministic robust positive log-frequency line consensus, followed by local original-timestamp chirp frequency refinement and robust log-line fitting. Nominal f0/f1/duration/start are never used to estimate the trajectory. Logarithmic form is an assumed sweep model. Fundamental identity is not guaranteed if a harmonic dominates; inspect spectrograms. Frequency is data-derived but NOT an independent instrument measurement and is estimated from the same output used for amplitude (selection bias at low SNR).',
    'stft': 'For frequency discovery only: linear interpolation to the record mean-rate uniform grid, constant detrend, Hann STFT, configured window/hop/padding. Zero padding is not frequency resolution. Final amplitudes use original timestamps, no resampling.',
    'filter': 'For tau=t-center, phi=2*pi*f_center*expm1(rate*tau)/rate. Form trapezoidal-time quadrature weights times centered Hann, normalize sum to one, subtract a weighted intercept+linear trend, and compute z=2*sum(w*x_detrended*exp(-i*phi)). A_peak=abs(z), A_rms=A_peak/sqrt(2). No frequency search at amplitude grid points or on baseline windows. This is a finite-support baseband low-pass tracking filter; negative-frequency image leakage and rapid within-window amplitude/phase changes can bias amplitudes. No causal delay correction is needed for a centered offline kernel.',
    'bandwidth': 'Nominal full two-sided baseband Hann ENBW=1.5/T (equivalently the positive-frequency real bandpass lobe bandwidth); half of that is the one-sided baseband integral. Export fs*sum(w^2) as sampled pre-detrend approximation. Also export fs*sum(abs(h)^2)/4 for the actual complex analysis kernel h including weighted detrending: exact white discrete-noise gain expressed as equivalent bandwidth with one-sided PSD=2*sigma^2/fs. Irregular sampling and unknown analog response preclude an exact physical continuous-noise bandwidth. Noise windows must match the signal effective bandwidth within configured relative tolerance. Frequency excursion during T, grid spacing, ENBW and physical response bandwidth remain distinct.',
    'snr': 'Signal numerator=observed tracking fundamental peak. Noise equivalent peak=sqrt(mean(abs(z_noise)^2)) over nonoverlapping independent-baseline windows, same f/rate/T/kernel/detrending as signal, with no frequency optimization. SNR_linear=A_peak/N_peak; SNR_dB=20*log10(SNR_linear). Both nothing_on and amplifier_on retained; amplifier_on defines range. No noise-power subtraction; numerator has positive low-SNR amplitude bias. This is estimator-equivalent SNR, not conventional broadband or acoustic SNR. A fixed 10 dB default is a chosen reporting criterion, not a false-alarm-calibrated detection threshold.',
    'broadband': 'Separate metric: tracking peak/sqrt(2) divided by sqrt(mean(var(original baseline samples, ddof=0))) using the same eligible baseline windows. Demean only for this full recorded-band noise metric. No analog bandwidth claim and no replacement of the existing notebooks or their metrics.',
    'quality': 'Reject nonfinite/nonincreasing timestamps without repair. Save gap/jitter/sample-rate/Nyquist audit. Strict response requires an accepted trajectory, observed track support, full data coverage, gap ratio gate, all-window frequency below configured Nyquist margin, minimum samples per cycle and minimum cycles. Endpoint-bearing IM records excluded from primary population, also when their local window has no endpoint. All computable amplitudes retained diagnostically. Local chirp fit fraction is saved as a diagnostic, not a local amplitude/SNR selection gate. Missing or invalid baseline produces unavailable SNR, never zero noise.',
    'aggregation': 'At each common frequency: arithmetic mean and sample SD across eligible independent records; never count overlapping frequency windows as repeats. Group SNR is arithmetic mean of per-record dB SNR. Require configured minimum record count separately for amplitude and SNR. A minimum of one is supported: the mean is the single eligible record, sample SD remains undefined (NaN), and no between-record repeatability can be inferred. Shared baseline uncertainty is not reduced by the number of sweep repeats. No bootstrap/CI or calibrated uncertainty claim.',
    'range': 'Contiguous adjacent tested grid points with primary amplifier_on mean SNR strictly greater than threshold and sufficient repeats. Report every segment; do not bridge missing/failed points or interpolate cutoff crossings. These are demonstrated grid intervals within measured support under this excitation/filter, not a complete intrinsic sensor bandwidth or acoustic frequency response.',
    'reference': 'At identical grid frequencies ratio of independent mean IM peak counts to independent mean accelerometer peak g, requiring confirmed comparison conditions, both sensors primary SNR above threshold and sufficient repeats. No file pairing, synchronized phase, coherence or transfer function. counts/g is nominal acceleration-normalized responsivity; counts/(m/s^2)=counts/g / 9.80665. No ADC-to-voltage calibration or individual accelerometer calibration is available. Between-batch drift, clock, reference frequency response and mounting uncertainty are unresolved.',
    'provenance': 'Every run uses the portable results_root configuration, exclusive New York date/run number, full input paths/roles/hashes, effective config, exact source snapshots, Git state and completed/failed status. Partial outputs preserved; no raw writes.',
}


def tracking_config():
    """Explicit settings for the new method; old notebook configuration is untouched."""
    c = default_config('frequency_range')
    # Delete settings belonging exclusively to the old nominal-seeded tracker.
    for key in ('start_search_s', 'duration_search_s', 'onset_overrides_s',
                'duration_overrides_s', 'response_min_fraction'):
        c.pop(key, None)
    c.update(
        method='complex_demodulation_hann_tracking', window_s=0.8,
        track_search_hz=[15., 420.], stft_window_s=0.3, stft_hop_s=0.15,
        stft_padding=8, ridge_log_tolerance=0.045, ridge_trials=2000,
        ridge_random_seed=1606321, log_rate_bounds_per_s=[0.01, 0.4],
        local_search_min_halfwidth_hz=4., local_search_step_hz=0.8,
        local_search_excursion_factor=0.6, local_frequency_tolerance_hz=1e-5,
        robust_log_scale=0.003, track_residual_gate_log=0.025,
        min_samples_per_cycle=3., min_cycles_per_window=12., nyquist_fraction=0.8,
        noise_enbw_relative_tolerance=0.03, max_noise_windows=30,
        minimum_noise_windows=4, nominal_start_overrides_s={},
        nominal_start_source='Unknown unless a per-file measured/recorded start is entered; otherwise comparison is onset-aligned for shape only.',
        grid_points=81, snr_threshold_db=10., minimum_repeat_count=2,
        standard_gravity_m_s2=9.80665,
    )
    return c


def _phase(tau, freq, rate):
    return 2*np.pi*freq*(np.expm1(rate*tau)/rate if abs(rate)>1e-12 else tau)


def tracking_filter(t, y, center, freq, rate, duration):
    """Original-timestamp quadrature filter and its noise gain; no peak search."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    if len(t)<8 or len(t)!=len(y) or not np.isfinite(t).all() or not np.isfinite(y).all():
        raise ValueError('Insufficient/nonfinite filter samples')
    if np.any(np.diff(t)<=0) or duration<=0 or freq<=0:
        raise ValueError('Invalid timestamps or filter settings')
    tau = t-center
    if np.any(np.abs(tau)>duration/2+1e-9):
        raise ValueError('Samples outside filter support')
    dt = np.diff(t)
    q = np.r_[dt[0]/2, (dt[:-1]+dt[1:])/2, dt[-1]/2]
    w = q*np.maximum(0., 0.5+0.5*np.cos(2*np.pi*tau/duration))
    w /= w.sum()
    nuisance = np.column_stack([np.ones(len(t)), tau/duration])
    gram = nuisance.T@(w[:, None]*nuisance)
    h0 = 2*w*np.exp(-1j*_phase(tau, freq, rate))
    # h @ y is identical to filtering after weighted linear detrending.
    h = h0 - ((h0@nuisance)@np.linalg.solve(gram, nuisance.T*w))
    z = h@(y-y.mean())
    fs = (len(t)-1)/(t[-1]-t[0])
    return dict(amplitude_peak=float(abs(z)), amplitude_rms=float(abs(z)/np.sqrt(2)),
                enbw_nominal_hz=1.5/duration, enbw_sampled_hz=float(fs*np.sum(w*w)),
                enbw_effective_hz=float(fs*np.vdot(h,h).real/4),
                white_noise_peak_gain=float(np.vdot(h,h).real),
                frequency_low_hz=float(freq*np.exp(-rate*duration/2)),
                frequency_high_hz=float(freq*np.exp(rate*duration/2)))


def _robust_line(t, logf, c, initial=None):
    if initial is None:
        b, a = np.polyfit(t, logf, 1)
        initial = [a, b]
    opt = optimize.least_squares(lambda x: x[0]+x[1]*t-logf, initial,
                                 loss='soft_l1', f_scale=c['robust_log_scale'])
    return opt.x


def independent_track(r, c):
    """No nominal frequency/time trajectory enters discovery or refinement."""
    t, y = r['t'], r['y']
    fs = r['audit']['fs_hz']
    n = round(c['stft_window_s']*fs)
    hop = round(c['stft_hop_s']*fs)
    if n<16 or hop<1 or n>=len(t):
        raise ValueError('Record too short for configured STFT')
    u = np.linspace(t[0], t[-1], len(t))
    f, tt, p = signal.spectrogram(np.interp(u,t,y), fs=fs, window='hann',
                                nperseg=n, noverlap=n-hop, nfft=n*c['stft_padding'],
                                detrend='constant', scaling='density')
    band = (f>=c['track_search_hz'][0]) & (f<=min(c['track_search_hz'][1],fs/2*c['nyquist_fraction']))
    f, p = f[band], p[band]
    if len(f)<3:
        raise ValueError('No search band below Nyquist margin')
    ridge = f[np.argmax(p,axis=0)]
    logf = np.log(ridge)
    rng = np.random.default_rng(c['ridge_random_seed'])
    best = None
    for _ in range(c['ridge_trials']):
        i, j = np.sort(rng.choice(len(tt),2,replace=False))
        if tt[j]-tt[i]<c['track_min_span_s']/2:
            continue
        b = (logf[j]-logf[i])/(tt[j]-tt[i])
        if not c['log_rate_bounds_per_s'][0]<=b<=c['log_rate_bounds_per_s'][1]:
            continue
        a = logf[i]-b*tt[i]
        keep = np.abs(a+b*tt-logf)<c['ridge_log_tolerance']
        score = int(keep.sum())
        if best is None or score>best[0]:
            best = (score,a,b,keep)
    if best is None or best[0]<c['track_min_points']:
        raise ValueError('No independently discovered rising logarithmic ridge')
    _, a, b, keep = best
    a, b = _robust_line(tt[keep],logf[keep],c,[a,b])
    keep = np.abs(a+b*tt-logf)<c['ridge_log_tolerance']
    half = c['window_s']/2
    lo, hi = max(half,tt[keep].min()), min(t[-1]-half,tt[keep].max())
    points = []
    for center in np.arange(lo,hi+1e-9,c['track_hop_s']):
        try:
            ts, ys = window(r,center-half,center+half)
        except ValueError:
            continue
        q = local_quality(ts,c)
        seed = np.exp(a+b*center)
        width = max(c['local_search_min_halfwidth_hz'],seed*b*c['window_s']*c['local_search_excursion_factor'])
        left = max(c['track_search_hz'][0],seed-width)
        right = min(c['track_search_hz'][1],seed+width, q['nominal_nyquist_hz']*c['nyquist_fraction']/np.exp(b*half))
        if right<=left:
            continue
        grid = np.linspace(left,right,max(5,int(np.ceil((right-left)/c['local_search_step_hz']))+1))
        scores = [chirp_fit(ts,ys,center,ff,b,c['window_s'])['fit_fraction'] for ff in grid]
        k = int(np.argmax(scores))
        if k in (0,len(grid)-1):
            continue
        opt = optimize.minimize_scalar(lambda ff: -chirp_fit(ts,ys,center,ff,b,c['window_s'])['fit_fraction'],
                                       bounds=(grid[k-1],grid[k+1]), method='bounded',
                                       options={'xatol':c['local_frequency_tolerance_hz']})
        fit = chirp_fit(ts,ys,center,opt.x,b,c['window_s'])
        endpoints = int(np.isin(ys,c['im_endpoint_codes']).sum()) if r['sensor']=='IM' else 0
        points.append(dict(center_s=center,local_hz=opt.x,fit_fraction=fit['fit_fraction'],
                           gap_ok=q['gap_ok'],endpoint_code_count=endpoints))
    track = pd.DataFrame(points)
    if track.empty:
        raise ValueError('No refined frequency points')
    good = track.gap_ok & track.endpoint_code_count.eq(0) & track.fit_fraction.ge(c['track_min_fraction'])
    if good.sum()<c['track_min_points']:
        raise ValueError('Insufficient coherent frequency support')
    a, b = _robust_line(track.loc[good,'center_s'].to_numpy(),np.log(track.loc[good,'local_hz']),c,[a,b])
    residual = np.log(track.local_hz)-(a+b*track.center_s)
    good &= residual.abs().le(c['track_residual_gate_log'])
    if good.sum()<c['track_min_points'] or np.ptp(track.loc[good,'center_s'])<c['track_min_span_s']:
        raise ValueError('Insufficient frequency support after residual gate')
    a,b = _robust_line(track.loc[good,'center_s'].to_numpy(),np.log(track.loc[good,'local_hz']),c,[a,b])
    residual = np.log(track.local_hz)-(a+b*track.center_s)
    rmse = float(np.sqrt(np.mean(residual[good]**2)))
    ok = bool(c['log_rate_bounds_per_s'][0]<=b<=c['log_rate_bounds_per_s'][1] and rmse<=c['track_max_log_residual'])
    track['used_for_model'] = good
    track['model_hz'] = np.exp(a+b*track.center_s)
    track['local_minus_model_hz'] = track.local_hz-track.model_hz
    # Nominal comparison is downstream of, and never feeds, frequency fitting.
    nominal_start = c['nominal_start_overrides_s'].get(r['file'])
    inferred_crossing = (np.log(c['f0_hz'])-a)/b
    compare_start = inferred_crossing if nominal_start is None else nominal_start
    track['nominal_comparison_hz'] = c['f0_hz']*np.exp(np.log(c['f1_hz']/c['f0_hz'])/c['nominal_duration_s']*(track.center_s-compare_start))
    track['local_minus_nominal_hz'] = track.local_hz-track.nominal_comparison_hz
    track['nominal_comparison_basis'] = 'onset_aligned_shape_only' if nominal_start is None else 'configured_recorded_start'
    meta = dict(file=r['file'],log_intercept=a,rate_per_s=b,trajectory_ok=ok,
                coherent_points=int(good.sum()),log_frequency_rmse=rmse,
                support_start_s=float(track.loc[good,'center_s'].min()),
                support_end_s=float(track.loc[good,'center_s'].max()),
                inferred_f0_crossing_s=inferred_crossing,
                inferred_f0_to_f1_duration_s=float(np.log(c['f1_hz']/c['f0_hz'])/b),
                nominal_duration_s=c['nominal_duration_s'],
                nominal_comparison_basis=track.nominal_comparison_basis.iloc[0])
    return meta,track,(f,tt,p)


def _sampling_ok(q, fit, c):
    return bool(q['gap_ok'] and fit['frequency_high_hz']<q['nominal_nyquist_hz']*c['nyquist_fraction']
                and q['fs_hz']/fit['frequency_high_hz']>=c['min_samples_per_cycle']
                and fit['frequency_low_hz']*c['window_s']>=c['min_cycles_per_window'])


def baseline_windows(r,c):
    half = c['window_s']/2
    centers = np.arange(c['noise_start_s']+half,r['t'][-1]-half,c['window_s'])
    if len(centers)>c['max_noise_windows']:
        centers=centers[np.unique(np.round(np.linspace(0,len(centers)-1,c['max_noise_windows'])).astype(int))]
    out=[]
    # Same conservative endpoint rule for baseline and signal records.
    if r['sensor']=='IM' and r['audit']['endpoint_code_count']:
        return out
    for center in centers:
        try:
            t,y=window(r,center-half,center+half)
        except ValueError:
            continue
        q=local_quality(t,c)
        if q['gap_ok']:
            out.append((t-center,y,center,q))
    return out


def contiguous_intervals(summary,c):
    rows=[]
    for material,part in summary.groupby('material',sort=False):
        part=part.sort_values('grid_index')
        indices=part.loc[part.detectable,'grid_index'].to_numpy(int)
        for group in np.split(indices,np.flatnonzero(np.diff(indices)>1)+1):
            if not len(group):
                continue
            pp=part.set_index('grid_index').loc[group]
            rows.append(dict(material=material,start_hz=pp.frequency_hz.iloc[0],end_hz=pp.frequency_hz.iloc[-1],
                             grid_points=len(group),minimum_n=int(pp.n_snr.min()),threshold_db=c['snr_threshold_db'],
                             minimum_group_snr_db=pp.snr_db_mean.min(),
                             interpretation='tested adjacent grid points; no interpolated cutoff'))
    return pd.DataFrame(rows,columns=['material','start_hz','end_hz','grid_points','minimum_n','threshold_db','minimum_group_snr_db','interpretation'])


def summarize(detail,c):
    rows=[]
    grid=np.geomspace(c['f0_hz'],c['f1_hz'],c['grid_points'])
    for material in [*c['materials'],'accelerometer']:
        requested=len(c['accel_sweeps']) if material=='accelerometer' else len(c['materials'][material][1])
        for j,f in enumerate(grid):
            allp=detail[detail.material.eq(material)&detail.grid_index.eq(j)]
            p=allp[allp.primary_ok]
            snr=p.amplifier_on_snr_db.dropna()
            row=dict(material=material,grid_index=j,frequency_hz=f,n_requested=requested,n_computed=len(allp),n_amplitude=len(p),n_snr=len(snr),
                     amplitude_peak_mean=p.amplitude_peak.mean() if len(p)>=c['minimum_repeat_count'] else np.nan,
                     amplitude_peak_sd=p.amplitude_peak.std() if len(p)>=c['minimum_repeat_count'] else np.nan,
                     snr_db_mean=snr.mean() if len(snr)>=c['minimum_repeat_count'] else np.nan,
                     snr_db_sd=snr.std() if len(snr)>=c['minimum_repeat_count'] else np.nan,
                     diagnostic_all_record_peak_mean=allp.amplitude_peak.mean())
            for condition in ('nothing_on','amplifier_on'):
                for metric in ('snr_db','broadband_snr_db'):
                    vals=p[condition+'_'+metric].dropna()
                    row[condition+'_'+metric+'_mean']=vals.mean() if len(vals)>=c['minimum_repeat_count'] else np.nan
            row['detectable']=bool(row['snr_db_mean']>c['snr_threshold_db'])
            rows.append(row)
    summary=pd.DataFrame(rows)
    ref=summary[summary.material.eq('accelerometer')].set_index('grid_index')
    ratios=[]
    for row in summary[~summary.material.eq('accelerometer')].itertuples():
        acc=ref.loc[row.grid_index]
        ok=bool(c['reference_conditions_confirmed'] and row.detectable and acc.detectable and acc.amplitude_peak_mean>0)
        ratio=row.amplitude_peak_mean/acc.amplitude_peak_mean if ok else np.nan
        ratios.append(dict(material=row.material,grid_index=row.grid_index,frequency_hz=row.frequency_hz,
                           n_im=row.n_amplitude,n_accel=int(acc.n_amplitude),
                           im_peak_counts=row.amplitude_peak_mean,im_sd_counts=row.amplitude_peak_sd,
                           accel_peak_g=acc.amplitude_peak_mean,accel_sd_g=acc.amplitude_peak_sd,
                           eligible=ok,counts_per_g=ratio,counts_per_m_s2=ratio/c['standard_gravity_m_s2']))
    return summary,pd.DataFrame(ratios),contiguous_intervals(summary,c)


def _reference_gap_results(detail, config):
    """Reaggregate existing estimates after relaxing only the reference gap gate."""
    diagnostic = detail.copy(deep=True)
    gap_only = []
    for idx, row in detail.iterrows():
        if row.sensor != 'accel' or row.primary_ok or row.gap_ok:
            continue
        q = row.to_dict()
        q['gap_ok'] = True
        if _sampling_ok(q, q, config):
            diagnostic.loc[idx, 'primary_ok'] = True
            gap_only.append(idx)
    summary, relaxed, _ = summarize(diagnostic, config)
    return summary, relaxed, detail.loc[gap_only].copy()


def diagnostic_reference(detail, config, strict_reference):
    """Restore only missing ratios supported by otherwise-valid reference-gap windows."""
    _, relaxed, candidates = _reference_gap_results(detail, config)
    keys = ['material', 'grid_index']
    out = strict_reference.copy().set_index(keys)
    relaxed = relaxed.set_index(keys).reindex(out.index)
    fill = ~out.eligible & relaxed.eligible
    out['strict_counts_per_g'] = out.counts_per_g
    out['display_counts_per_g'] = out.counts_per_g
    out.loc[fill, 'display_counts_per_g'] = relaxed.loc[fill, 'counts_per_g']
    out['reference_gap_diagnostic'] = fill
    out['diagnostic_accel_peak_g'] = relaxed.accel_peak_g
    out['diagnostic_n_accel'] = relaxed.n_accel
    return out.reset_index(), candidates


def accelerometer_display(detail, config, strict_summary):
    """Keep strict reference values alongside the connected amplitude/SNR display."""
    relaxed, _, _ = _reference_gap_results(detail, config)
    strict = strict_summary[strict_summary.material.eq('accelerometer')].set_index('grid_index')
    relaxed = relaxed[relaxed.material.eq('accelerometer')].set_index('grid_index').reindex(strict.index)
    out = strict[['frequency_hz', 'n_requested', 'n_amplitude', 'n_snr']].copy()
    out['reference_gap_diagnostic'] = False
    for metric in ['amplitude_peak', 'snr_db']:
        original = strict[metric + '_mean']
        candidate = relaxed[metric + '_mean']
        fill = original.isna() & candidate.notna()
        out['strict_' + metric] = original
        out['display_' + metric] = original.where(~fill, candidate)
        out[metric + '_gap_diagnostic'] = fill
        out['reference_gap_diagnostic'] |= fill
    out['display_n_amplitude'] = relaxed.n_amplitude
    out['display_n_snr'] = relaxed.n_snr
    return out.reset_index()


PLOT_PRESENTATION = {
    'main_reference': 'connected_existing_estimates',
    'main_accelerometer_reference': 'connected_existing_estimates',
    'method': 'Main comparison plots use existing reference estimates from accelerometer windows rejected only by the timestamp-gap gate, retaining all other sampling, noise, SNR and reference checks. No interpolation, fabricated points or new signal estimation. Existing strict-eligible ratios remain identical. Other unsupported points stay unavailable. Main figures have no gap annotations or point markers; strict comparison plots are saved with _strict suffix for the supplement.',
    'accelerometer_method': 'The standalone main accelerometer amplitude and SNR curves reuse the same gap-only reference estimates, preserving every existing strict value. This display does not alter threshold eligibility. The original curves are exported as tracking_accelerometer_reference_strict for the supplement; both sets of values and flags are saved under presentation/.',
    'strict_results': 'Original strict CSVs, eligibility flags, amplitudes, SNR and detectable intervals are unchanged. Display values and per-point diagnostic flags are saved separately under presentation/.',
    'legend_layout': 'Comparison y-axis upper limits can be set independently for amplitude, SNR and responsivity. Numeric limits override automatic padding after layout; None retains automatic legend clearance (12% minimum transformed padding for materials, 30% for coil design). Lower limits and all plotted values are preserved.',
    'comparison_ymax': {'materials': [40000., 120., 80000.],
                        'coil_design': [150000., 85., 40000.]},
}


def _plot_presentation(config, plot_ymax=None):
    """Resolve display-only upper limits without changing scientific settings."""
    policy = deepcopy(PLOT_PRESENTATION)
    limits = policy['comparison_ymax']
    groups = config['comparison_groups']
    if plot_ymax is not None:
        if not isinstance(plot_ymax, dict):
            raise ValueError('plot_ymax must map comparison groups to three upper limits.')
        unknown = set(plot_ymax) - set(groups)
        if unknown:
            raise ValueError(f'Unknown plot_ymax comparison groups: {sorted(unknown)}')
        limits.update(deepcopy(plot_ymax))
    policy['comparison_ymax'] = {group: value for group, value in limits.items() if group in groups}
    for group, values in policy['comparison_ymax'].items():
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f'plot_ymax[{group!r}] requires [amplitude, SNR, responsivity]; use None for automatic.')
        normalized = []
        for value in values:
            if value is None:
                normalized.append(None)
            else:
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError(f'plot_ymax[{group!r}] limits must be numbers or None, not booleans.')
                number = float(value)
                if not np.isfinite(number) or number <= 0:
                    raise ValueError(f'plot_ymax[{group!r}] limits must be positive finite numbers or None.')
                normalized.append(number)
        policy['comparison_ymax'][group] = normalized
    return policy


def _save_display_reference(run, detail, config, strict_reference):
    reference, candidates = diagnostic_reference(detail, config, strict_reference)
    folder = run.path / 'presentation'
    folder.mkdir(exist_ok=True)
    reference.to_csv(folder / 'responsivity_strict_and_diagnostic.csv', index=False)
    candidates.to_csv(folder / 'reference_gap_window_estimates.csv', index=False)
    return reference


def _save_accelerometer_display(run, detail, config, strict_summary):
    reference = accelerometer_display(detail, config, strict_summary)
    folder = run.path / 'presentation'
    folder.mkdir(exist_ok=True)
    reference.to_csv(folder / 'accelerometer_reference_strict_and_display.csv', index=False)
    return reference


def _figure_lists(figures):
    supplemental = [path for path in figures if Path(path).stem.startswith(('tracking_comparison_', 'tracking_accelerometer_reference'))
                    and Path(path).stem.endswith('_strict')]
    return [path for path in figures if path not in supplemental], supplemental


def _plots(run,summary,ratios,c,figures,display_reference=None,display_accelerometer=None,plot_ymax=None):
    upper_limits = _plot_presentation(c, plot_ymax)['comparison_ymax']
    for name,members in c['comparison_groups'].items():
        # Comparison groups may still name materials removed in notebook controls.
        members = [material for material in members if material in c['materials']]
        if not members:
            continue
        single_record = all(len(c['materials'][material][1]) <= 1 for material in members)
        variants = [('', ratios, 'counts_per_g')]
        if display_reference is not None:
            variants = [('', display_reference, 'display_counts_per_g'), ('_strict', ratios, 'counts_per_g')]
        for suffix, reference, reference_column in variants:
            fig,axes=plt.subplots(3,1,figsize=(11,11),sharex=True)
            fig._tracking_legend_headroom = name in {'materials', 'coil_design'} or name in upper_limits
            fig._tracking_ymax = upper_limits.get(name, [None, None, None])
            if name == 'coil_design':
                fig._tracking_legend_headroom_fraction = .30
            for material in members:
                p=summary[summary.material.eq(material)]
                rr=reference[reference.material.eq(material)]
                line,=axes[0].plot(p.frequency_hz,p.amplitude_peak_mean,label=material)
                color=line.get_color()
                if not single_record:
                    axes[0].fill_between(p.frequency_hz,p.amplitude_peak_mean-p.amplitude_peak_sd,p.amplitude_peak_mean+p.amplitude_peak_sd,color=color,alpha=.13)
                axes[1].plot(p.frequency_hz,p.snr_db_mean,color=color,label=material)
                axes[1].plot(p.frequency_hz,p.nothing_on_snr_db_mean,color=color,ls=':',alpha=.6)
                axes[2].plot(rr.frequency_hz,rr[reference_column],color=color,label=material)
            title = (f'{name}: harmonic tracking of selected sweeps' if single_record else
                     f'{name}: harmonic tracking; shading = independent-record SD')
            axes[0].set(ylabel='Fundamental peak (counts)',title=title)
            axes[1].set(ylabel='Estimator-equivalent SNR (dB)',title='Solid: amplifier on; dotted: nothing on')
            axes[1].axhline(c['snr_threshold_db'],color='k',ls='--')
            axes[2].set(xlabel='Data-derived model frequency (Hz)',ylabel='Nominal responsivity (counts/g)',yscale='log')
            for ax in axes:
                ax.legend(fontsize=7, loc='upper right'); ax.grid(alpha=.2); ax.set_xscale('log')
            fig.tight_layout(); save_fig(run,fig,'tracking_comparison_'+name+suffix,figures)
        # A single selected sweep has no between-record SD; export only its curve.
        # Multiple-record exports share identical data and axis limits.
        limits = None
        for shaded in ((False,) if single_record else (True, False)):
            peak_fig, ax = plt.subplots(figsize=(11, 5))
            for material in members:
                p = summary[summary.material.eq(material)]
                line, = ax.plot(p.frequency_hz, p.amplitude_peak_mean, label=material)
                if shaded:
                    ax.fill_between(p.frequency_hz,
                        p.amplitude_peak_mean-p.amplitude_peak_sd,
                        p.amplitude_peak_mean+p.amplitude_peak_sd,
                        color=line.get_color(), alpha=.13)
            title = (name + ': selected sweep fundamental' if single_record else
                     name + (': mean and independent-record SD' if shaded else ': mean'))
            ax.set(xscale='log', xlabel='Data-derived model frequency (Hz)',
                   ylabel='Fundamental peak (counts)', title=title)
            if shaded:
                limits = ax.get_ylim()
            elif limits is not None:
                ax.set_ylim(limits)
            ax.legend()
            save_fig(run, peak_fig, 'tracking_fundamental_'+name+
                     ('_with_region' if shaded else '_no_region'), figures)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for material,p in summary[~summary.material.eq('accelerometer')].groupby('material'):
        axes[0].plot(p.frequency_hz,p.amplifier_on_broadband_snr_db_mean,label=material)
        axes[1].plot(p.frequency_hz,p.n_snr,label=material)
    axes[0].set(ylabel='Broadband SNR (dB)',title='Separate broadband noise metric; not used for range')
    axes[1].set(xlabel='Frequency (Hz)',ylabel='Eligible sweep record count')
    for ax in axes: ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); save_fig(run,fig,'tracking_broadband_and_repeat_counts',figures)
    p=summary[summary.material.eq('accelerometer')]
    variants = [('', p, 'amplitude_peak_mean', 'snr_db_mean')]
    if display_accelerometer is not None:
        variants = [('', display_accelerometer, 'display_amplitude_peak', 'display_snr_db'),
                    ('_strict', p, 'amplitude_peak_mean', 'snr_db_mean')]
    for suffix, reference, amplitude_column, snr_column in variants:
        fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
        axes[0].plot(reference.frequency_hz,reference[amplitude_column])
        axes[0].set(ylabel='Fundamental peak (g)',title='Separately acquired reference: nominal conversion')
        axes[1].plot(reference.frequency_hz,reference[snr_column])
        axes[1].axhline(c['snr_threshold_db'],ls='--',color='k')
        axes[1].set(xlabel='Frequency (Hz)',ylabel='Tracking SNR (dB)')
        fig.tight_layout(); save_fig(run,fig,'tracking_accelerometer_reference'+suffix,figures)


def run_tracking(config,notebook_path,executed_cells=None,plot_ymax=None):
    c=deepcopy(config)
    presentation = _plot_presentation(c, plot_ymax)
    ensure_environment()
    if c['im_endpoint_scope']!='record':
        raise ValueError('This notebook primary population requires record endpoint exclusion')
    if not (0<c['f0_hz']<c['f1_hz'] and c['window_s']>0 and c['grid_points']>=2
            and c['minimum_repeat_count']>=1 and c['minimum_noise_windows']>=2
            and c['max_noise_windows']>=c['minimum_noise_windows']):
        raise ValueError('Invalid frequency/window/repeat configuration')
    root=Path(load_config()['results_root'])
    inputs=manifest(c)
    paths=[x['path'] for x in inputs]
    if len(paths)!=len(set(paths)):
        raise ValueError('Duplicate input path: do not count one file as independent repeats')
    scripts=source_paths(notebook_path)
    params=dict(config=c,methods=METHODS,references=REFERENCES,results_root_source=config_source(),
                plot_presentation=presentation)
    records={}; audits=[]; issues=[]; hashes=[]; figures=[]
    details=[]; tracks=[]; noise_rows=[]; failures=[]
    with AnalysisRun(root,'data11_harmonic_tracking',params,inputs,scripts,repository_path=PROJECT) as run:
        print('Output directory:',run.path,flush=True)
        run.write_text('methods.txt','\n\n'.join(k+': '+v for k,v in METHODS.items())+
                       '\n\nPlot presentation: '+PLOT_PRESENTATION['method']+' '+PLOT_PRESENTATION['accelerometer_method']+' '+PLOT_PRESENTATION['strict_results'])
        if executed_cells:
            run.write_text('scripts/executed_notebook_cells.py','\n\n'.join(executed_cells))
        run.write_json('notebook_execution_source.json',dict(live_cells_available=bool(executed_cells),disk_notebook_archived=True,effective_config_in_parameters=True))
        for item in inputs:
            try:
                path=Path(item['path'])
                hashes.append(dict(path=str(path),sha256=sha256(path.read_bytes()).hexdigest(),size_bytes=path.stat().st_size))
                r=load_record(item,c)
                records[item['file']]=r
                audits.append(dict(**r['audit'],role=item['role'],material=item['material']))
            except (FileNotFoundError,ValueError,KeyError,pd.errors.ParserError) as exc:
                issues.append(dict(**item,error=str(exc)))
                if c['missing_policy']=='fail': raise
            finally:
                pd.DataFrame(audits).to_csv(run.path/'timestamp_audit.csv',index=False)
                run.write_json('input_issues.json',issues)
                run.write_json('input_hashes.json',hashes)
        baselines={}
        for r in records.values():
            if r['role']=='noise':
                baselines[(r['material'],r['condition'])]=(r,baseline_windows(r,c))
        grid=np.geomspace(c['f0_hz'],c['f1_hz'],c['grid_points'])
        half=c['window_s']/2
        for r in records.values():
            if r['role']!='sweep': continue
            print('Harmonic tracking:',r['file'],flush=True)
            try:
                meta,track,spec=independent_track(r,c)
            except ValueError as exc:
                failures.append(dict(file=r['file'],reason=str(exc)))
                run.write_json('unavailable_records.json',failures)
                continue
            tracks.append(meta)
            track.to_csv(run.path/(Path(r['file']).stem+'_frequency_track.csv'),index=False)
            pd.DataFrame(tracks).to_csv(run.path/'trajectory_summary.csv',index=False)
            f,tt,p=spec
            fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
            axes[0].pcolormesh(tt,f,10*np.log10(np.maximum(p,1e-30)),shading='auto')
            axes[0].plot(track.center_s,track.model_hz,'w-',label='Data-fitted log trajectory')
            axes[0].scatter(track.center_s,track.local_hz,s=6,c='red',label='Local frequency estimates')
            axes[0].plot(track.center_s,track.nominal_comparison_hz,'c--',label='Nominal: '+meta['nominal_comparison_basis'])
            axes[0].set(ylabel='Hz',title=f"{r['file']} | trajectory accepted={meta['trajectory_ok']}")
            axes[0].legend(fontsize=7)
            axes[1].plot(track.center_s,track.local_minus_nominal_hz,'.-',label='Local minus nominal')
            axes[1].plot(track.center_s,track.local_minus_model_hz,'.-',label='Local minus model')
            axes[1].set(xlabel='Recorded time (s)',ylabel='Frequency difference (Hz)'); axes[1].legend()
            fig.tight_layout(); save_fig(run,fig,Path(r['file']).stem+'_harmonic_tracking',figures)
            if not meta['trajectory_ok']:
                failures.append(dict(file=r['file'],reason='Independent trajectory quality failed'))
                continue
            rate=meta['rate_per_s']
            for j,freq in enumerate(grid):
                center=(np.log(freq)-meta['log_intercept'])/rate
                if not meta['support_start_s']<=center<=meta['support_end_s']:
                    continue
                try:
                    t,y=window(r,center-half,center+half)
                except ValueError:
                    continue
                fit=tracking_filter(t,y,center,freq,rate,c['window_s'])
                quality=local_quality(t,c)
                sample_ok=_sampling_ok(quality,fit,c)
                endpoint_ok=r['sensor']!='IM' or r['audit']['endpoint_code_count']==0
                row=dict(file=r['file'],sensor=r['sensor'],material=r['material'],grid_index=j,frequency_hz=freq,
                         center_s=center,window_start_s=center-half,window_end_s=center+half,
                         record_endpoint_count=r['audit']['endpoint_code_count'],
                         primary_ok=bool(sample_ok and endpoint_ok),sampling_ok=sample_ok,
                         fit_fraction_diagnostic=chirp_fit(t,y,center,freq,rate,c['window_s'])['fit_fraction'],
                         samples_per_cycle_min=quality['fs_hz']/fit['frequency_high_hz'],
                         frequency_excursion_hz=fit['frequency_high_hz']-fit['frequency_low_hz'],**fit,**quality)
                for condition in ('nothing_on','amplifier_on'):
                    pair=baselines.get((r['material'],condition))
                    amps=[]; broad=[]
                    for tn,yn,ncenter,nq in ([] if pair is None else pair[1]):
                        nf=tracking_filter(tn,yn,0.,freq,rate,c['window_s'])
                        mismatch=abs(nf['enbw_effective_hz']/fit['enbw_effective_hz']-1)
                        valid=_sampling_ok(nq,nf,c) and mismatch<=c['noise_enbw_relative_tolerance']
                        noise_rows.append(dict(signal_file=r['file'],noise_file=pair[0]['file'],condition=condition,
                                               grid_index=j,frequency_hz=freq,noise_center_s=ncenter,noise_start_s=ncenter-half,noise_end_s=ncenter+half,
                                               accepted=valid,enbw_relative_mismatch=mismatch,raw_demeaned_rms=float(np.std(yn)),**nf,**nq))
                        if valid:
                            amps.append(nf['amplitude_peak']); broad.append(np.var(yn))
                    enough=len(amps)>=c['minimum_noise_windows']
                    noise=float(np.sqrt(np.mean(np.square(amps)))) if enough else np.nan
                    rms=float(np.sqrt(np.mean(broad))) if enough else np.nan
                    linear=fit['amplitude_peak']/noise if noise>0 else np.nan
                    blin=fit['amplitude_rms']/rms if rms>0 else np.nan
                    row.update({condition+'_noise_equivalent_peak':noise,condition+'_noise_windows':len(amps),
                                condition+'_snr_linear':linear,condition+'_snr_db':20*np.log10(linear) if linear>0 else np.nan,
                                condition+'_broadband_noise_rms':rms,condition+'_broadband_snr_linear':blin,
                                condition+'_broadband_snr_db':20*np.log10(blin) if blin>0 else np.nan})
                details.append(row)
            pd.DataFrame(details).to_csv(run.path/'harmonic_tracking_records.csv',index=False)
            pd.DataFrame(noise_rows).to_csv(run.path/'noise_estimator_windows.csv',index=False)
            run.write_json('unavailable_records.json',failures)
        if not details:
            raise ValueError('No supported tracking windows; inspect unavailable_records.json')
        detail=pd.DataFrame(details)
        summary,ratios,intervals=summarize(detail,c)
        summary.to_csv(run.path/'harmonic_tracking_summary.csv',index=False)
        ratios.to_csv(run.path/'acceleration_normalized_responsivity.csv',index=False)
        intervals.to_csv(run.path/'detectable_frequency_intervals.csv',index=False)
        display_reference = _save_display_reference(run, detail, c, ratios)
        display_accelerometer = _save_accelerometer_display(run, detail, c, summary)
        _plots(run,summary,ratios,c,figures,display_reference=display_reference,
               display_accelerometer=display_accelerometer,plot_ymax=presentation['comparison_ymax'])
        run.write_json('unavailable_records.json',failures)
        run.finish(dict(available_records=len(records),input_issues=len(issues),unavailable_sweeps=len(failures),
                        figure_count=len(figures),completion_scope='All configured available inputs attempted; scientific gaps remain unavailable.'))
        main_figures, supplement_figures = _figure_lists(figures)
        return dict(output_path=str(run.path),summary=summary,reference=ratios,intervals=intervals,
                    trajectories=pd.DataFrame(tracks),records=detail,audit=pd.DataFrame(audits),
                    input_issues=issues,unavailable_records=failures,figures=main_figures,
                    supplement_figures=supplement_figures,display_reference=display_reference,
                    display_accelerometer=display_accelerometer)


def export_tracking_figures(parent_run, notebook_path=None, plot_ymax=None):
    """Redraw archived tracking results in a new run without processing raw data."""
    from characterization_run import _git_provenance
    from portable_runtime import validate_tracking_run
    parent = validate_tracking_run(parent_run)
    ensure_environment()
    parent_params = json.loads((parent / 'parameters.json').read_text(encoding='utf-8'))
    c = deepcopy(parent_params['config'])
    if plot_ymax is None:
        plot_ymax = parent_params.get('plot_presentation', {}).get('comparison_ymax')
    presentation = _plot_presentation(c, plot_ymax)
    names = ['run_status.json', 'parameters.json', 'inputs.json', 'input_hashes.json',
             'harmonic_tracking_records.csv', 'harmonic_tracking_summary.csv',
             'acceleration_normalized_responsivity.csv', 'detectable_frequency_intervals.csv']
    inputs = [dict(path=str(parent / name), role='archived_tracking_evidence',
                   sha256=sha256((parent / name).read_bytes()).hexdigest()) for name in names]
    params = dict(config=c, parent_run=str(parent), plot_presentation=presentation,
                  inherited_parent_methods=parent_params.get('methods', {}), raw_data_reprocessed=False)
    scripts = source_paths(notebook_path)
    # In the author's workspace the release is an untracked repository subfolder.
    # Record the actual enclosing commit/state and still snapshot every exact file.
    git_context = _git_provenance(PROJECT)
    repository = Path(git_context.get('enclosing_repository_path') or PROJECT)
    params['source_repository_path'] = str(repository)
    with AnalysisRun(load_config()['results_root'], 'data11_tracking_presentation', params,
                     inputs, scripts, repository_path=repository) as run:
        print('Tracking figure output:', run.path, flush=True)
        run.write_text('methods.txt', PLOT_PRESENTATION['method']+'\n\n'+PLOT_PRESENTATION['accelerometer_method']+'\n\n'+PLOT_PRESENTATION['strict_results']+
                       '\n\nInherited estimator definitions:\n'+json.dumps(parent_params.get('methods', {}), indent=2))
        detail = pd.read_csv(parent / 'harmonic_tracking_records.csv')
        summary = pd.read_csv(parent / 'harmonic_tracking_summary.csv')
        ratios = pd.read_csv(parent / 'acceleration_normalized_responsivity.csv')
        for name in ['harmonic_tracking_summary.csv', 'acceleration_normalized_responsivity.csv',
                     'detectable_frequency_intervals.csv']:
            (run.path / name).write_bytes((parent / name).read_bytes())
        display_reference = _save_display_reference(run, detail, c, ratios)
        display_accelerometer = _save_accelerometer_display(run, detail, c, summary)
        figures = []
        _plots(run, summary, ratios, c, figures, display_reference=display_reference,
               display_accelerometer=display_accelerometer,plot_ymax=presentation['comparison_ymax'])
        main_figures, supplement_figures = _figure_lists(figures)
        run.finish(dict(figure_count=len(figures), main_figures=len(main_figures),
                        supplement_figures=len(supplement_figures), raw_data_reprocessed=False,
                        recovered_reference_points=int(display_reference.reference_gap_diagnostic.sum()),
                        connected_accelerometer_grid_points=int(display_accelerometer.reference_gap_diagnostic.sum())))
        return dict(output_path=str(run.path), figures=main_figures, supplement_figures=supplement_figures,
                    reference=ratios, display_reference=display_reference,
                    display_accelerometer=display_accelerometer)
