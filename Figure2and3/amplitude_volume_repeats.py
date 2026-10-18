"""Independent-record volume summaries; no cross-sensor record pairing."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import odr
from pathlib import Path


METHODS = {
    'aggregation': 'One primary window per file and volume; arithmetic mean and sample SD (ddof=1) across independent records. Subwindows are not repeats. No cross-sensor file pairing.',
    'amplitude': 'Target-conditioned padded FFT local peak with three-point interpolation within configured 100 Hz +/- 1 Hz search band; existing resampling, demean, periodic Hann and coherent-gain normalization. No fallback to global peak. Search band is not noise bandwidth. Original global FFT, RMS and peak-to-peak retained.',
    'quality': 'All finite record estimates retained in descriptive means and plots. Fit eligibility requires the expected record count, finite positive amplitudes, computed FFTs, matching targets, no accel repeated-extrema flags, and target estimates inside search band for every record. Timing/other quality warnings remain explicit; this is a diagnostic response, not calibrated sensitivity.',
    'errorbars': 'Both plot axes show across-record sample SD, not SEM or total calibration uncertainty.',
    'fit': 'Free-intercept OLS descriptive fit plus ODR using independent-axis SEM=SD/sqrt(n). ODR only when all included SEMs are finite and positive; never invent uncertainty floors. No confidence intervals from three records; level correlations within a record and systematic reference/drift errors are not modeled.',
    'ratio': 'Ratio of IM condition mean to accel condition mean; not mean of paired ratios, not response slope.',
    'limits': 'Three files per sensor; mounting repeatability and between-batch drive stability unresolved unless documented. Fixed 100 Hz only; no phase/coherence/synchronized transfer function, absolute calibration, SNR or detection limit inferred.'
}


def summarize_volume_repeats(summary, config):
    c = config['volume_reference']
    key = c['amplitude_key']
    fkey = key.removesuffix('_amplitude') + '_hz'
    table = summary.loc[summary.role.eq('volume')].copy()
    if 'window_variant' in table:
        table = table.loc[table.window_variant.eq('primary')].copy()
    if table.duplicated(['sensor', 'path', 'volume_setting']).any():
        raise ValueError('Duplicate primary file/volume rows are not independent repeats.')
    records = table.copy()
    groups = []
    for (sensor, level), part in table.groupby(['sensor', 'volume_setting'], sort=True):
        expected = len(config['volume'][sensor])
        root = Path(config['data_root'])
        def identity(path):
            path = Path(path)
            return str((path if path.is_absolute() else root / path).resolve()).casefold()
        configured_paths = [identity(e['path']) for e in config['volume'][sensor]]
        if len(set(configured_paths)) != expected:
            raise ValueError('Repeated configured file is not another replicate.')
        if not set(part.path.map(identity)).issubset(configured_paths):
            raise ValueError('Summary contains a record outside the configured volume inputs.')
        values = pd.to_numeric(part[key], errors='coerce')
        finite = np.isfinite(values)
        frequency = pd.to_numeric(part[fkey], errors='coerce')
        valid = finite & values.gt(0) & part.fft_computed.astype(bool)
        valid &= part.target_hz.eq(c['target_hz'])
        valid &= (frequency - c['target_hz']).abs().lt(c['search_halfwidth_hz'])
        if sensor == 'accel':
            occupancy = part.raw_extrema_occupancy_fraction
            valid &= occupancy.ge(0) & occupancy.lt(c['plateau_fraction_threshold'])
        if part.unit.nunique() != 1:
            raise ValueError('Mixed units within a sensor/volume group.')
        records.loc[part.index, 'reference_record_eligible'] = valid
        for index, row in part.iterrows():
            reasons = []
            if not np.isfinite(values.loc[index]) or values.loc[index] <= 0:
                reasons.append('amplitude_missing_or_nonpositive')
            if not row.fft_computed:
                reasons.append('fft_unavailable')
            if row.target_hz != c['target_hz']:
                reasons.append('nominal_frequency_mismatch')
            if not np.isfinite(frequency.loc[index]) or abs(frequency.loc[index]-c['target_hz']) >= c['search_halfwidth_hz']:
                reasons.append('target_peak_missing_or_outside_band')
            if sensor == 'accel' and not 0 <= row.raw_extrema_occupancy_fraction < c['plateau_fraction_threshold']:
                reasons.append('accel_extrema_flag_or_unknown')
            records.loc[index, 'reference_record_reasons'] = ';'.join(reasons)
        vals = values[finite]
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
        groups.append(dict(sensor=sensor, volume_setting=level, unit=part.unit.iloc[0],
            amplitude_key=key, n_expected=expected, n_records=len(part), n_finite=len(vals),
            n_eligible=int(valid.sum()), mean=float(vals.mean()) if len(vals) else np.nan,
            sd=sd, sem=sd / np.sqrt(len(vals)) if len(vals) else np.nan,
            timing_warning_count=int((~part.fft_valid.astype(bool)).sum()),
            quality_warning_count=int(part.quality_flags.fillna('').ne('').sum()),
            group_eligible=bool(len(part) == expected and valid.all()),
            paths=';'.join(part.path.astype(str))))
    grouped = pd.DataFrame(groups)
    left = grouped.loc[grouped.sensor.eq('IM')].drop(columns='sensor').add_prefix('im_')
    right = grouped.loc[grouped.sensor.eq('accel')].drop(columns='sensor').add_prefix('accel_')
    reference = left.merge(right, left_on='im_volume_setting', right_on='accel_volume_setting', how='outer')
    reference['volume_setting'] = reference.im_volume_setting.combine_first(reference.accel_volume_setting)
    reference['ratio_of_means'] = reference.im_mean / reference.accel_mean.where(reference.accel_mean.gt(0))
    reference['fit_eligible'] = (reference.im_group_eligible.eq(True) & reference.accel_group_eligible.eq(True)
                                 & (c.get('conditions_confirmed') is True))
    reference['comparison_status'] = np.where(reference.fit_eligible, 'diagnostic_condition_comparison', 'flagged_or_incomplete')
    reference = reference.sort_values('volume_setting').reset_index(drop=True)
    return records, grouped, reference


def fit_volume_response(reference):
    result = dict(status='insufficient_eligible_levels', n_levels=0, intercept_forced_zero=False,
                  slope_unit='Data28 counts/g', uncertainty_note=METHODS['fit'])
    selected = reference.loc[reference.fit_eligible].copy()
    x, y = selected.accel_mean.to_numpy(float), selected.im_mean.to_numpy(float)
    result['n_levels'] = len(x)
    result['volume_levels'] = selected.volume_setting.tolist()
    if len(x) < 3 or not np.all(np.isfinite(x) & np.isfinite(y)) or np.ptp(x) <= 0:
        return result
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (intercept + slope * x)
    sst = np.sum((y - y.mean()) ** 2)
    result.update(status='ols_available', ols_slope=float(slope), ols_intercept=float(intercept),
                  ols_r_squared=float(1 - np.sum(residual**2) / sst) if sst > 0 else None,
                  ols_residuals=residual.tolist(), odr_status='unavailable_nonpositive_or_missing_sem')
    sx, sy = selected.accel_sem.to_numpy(float), selected.im_sem.to_numpy(float)
    if np.all(np.isfinite(sx) & np.isfinite(sy) & (sx > 0) & (sy > 0)):
        model = odr.Model(lambda beta, z: beta[0] * z + beta[1])
        fitted = odr.ODR(odr.RealData(x, y, sx=sx, sy=sy), model, beta0=[slope, intercept]).run()
        result['odr_stop_reason'] = fitted.stopreason
        if fitted.info in (1, 2, 3) and np.all(np.isfinite(fitted.beta)):
            result.update(status='ols_and_odr_available', odr_status='converged',
                          odr_slope=float(fitted.beta[0]), odr_intercept=float(fitted.beta[1]),
                          odr_vertical_residuals=(y - model.fcn(fitted.beta, x)).tolist())
        else:
            result['odr_status'] = 'not_converged'
    return result


def plot_volume_repeats(records, groups, reference, fit, amplitude_key):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, sensor in zip(axes, ('IM', 'accel')):
        for path, part in records.loc[records.sensor.eq(sensor)].groupby('path'):
            part = part.sort_values('volume_setting')
            ax.plot(part.volume_setting, part[amplitude_key], 'o-', alpha=.45, label=part.file_label.iloc[0])
        part = groups.loc[groups.sensor.eq(sensor)]
        ax.errorbar(part.volume_setting, part['mean'], yerr=part.sd.fillna(0), fmt='ko-', capsize=4, label='Mean +/- record SD')
        unit = part.unit.iloc[0] if len(part) else 'unavailable'
        ax.set(xlabel='Amplifier volume setting', ylabel=f'Target FFT peak amplitude [{unit}]', title=sensor)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    response, (ax, residual_ax) = plt.subplots(2, 1, figsize=(7, 8), constrained_layout=True)
    for _, row in reference.iterrows():
        if not np.isfinite(row.accel_mean) or not np.isfinite(row.im_mean):
            continue
        ax.errorbar(row.accel_mean, row.im_mean,
                    xerr=row.accel_sd if np.isfinite(row.accel_sd) else None,
                    yerr=row.im_sd if np.isfinite(row.im_sd) else None,
                    fmt='o' if row.fit_eligible else 'x', color='C0' if row.fit_eligible else 'C3', capsize=4)
        ax.annotate(str(int(row.volume_setting)), (row.accel_mean, row.im_mean), xytext=(5, 5), textcoords='offset points')
    selected = reference.loc[reference.fit_eligible]
    for method, style in [('ols', '--'), ('odr', '-')]:
        if method + '_slope' in fit:
            x = np.array([selected.accel_mean.min(), selected.accel_mean.max()])
            ax.plot(x, fit[method + '_intercept'] + fit[method + '_slope'] * x, style, label=method.upper() + ' free intercept')
            residual = selected.im_mean - (fit[method + '_intercept'] + fit[method + '_slope'] * selected.accel_mean)
            residual_ax.plot(selected.accel_mean, residual, 'o' + style, label=method.upper())
    ax.set(xlabel='Reference acceleration peak [g]', ylabel='IM target peak [Data28 counts]',
           title='100 Hz condition means; error bars: record SD\nRed crosses: flagged/incomplete; diagnostic fits')
    residual_ax.axhline(0, color='gray', lw=1)
    residual_ax.set(xlabel='Reference acceleration peak [g]', ylabel='Vertical residual [Data28 counts]')
    for a in (ax, residual_ax):
        a.grid(alpha=.2)
        if a.get_legend_handles_labels()[0]:
            a.legend()
    ratio, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for eligible, part in reference.groupby('fit_eligible'):
        ax.plot(part.accel_mean, part.ratio_of_means, 'o' if eligible else 'x',
                label='Eligible' if eligible else 'Flagged/incomplete')
        for _, row in part.iterrows():
            if np.isfinite(row.accel_mean) and np.isfinite(row.ratio_of_means):
                ax.annotate(str(int(row.volume_setting)), (row.accel_mean, row.ratio_of_means))
    ax.set(xlabel='Reference acceleration peak [g]', ylabel='Ratio of condition means [Data28 counts/g]',
           title='Amplitude ratio; distinct from response slope')
    ax.grid(alpha=.2)
    ax.legend()
    return [('volume_repeat_summary', fig), ('volume_reference_response', response), ('volume_reference_ratio', ratio)]
