"""Global FFT maxima and independent-record response; no raw-data writes."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data11_config import amplitude_files


def global_peak_outputs(run, detail, c, figures, save_fig):
    cutoff = c.get('global_search_min_hz', 1.)
    tolerance = c.get('amplitude_peak_check_halfwidth_hz', 1.)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('amplitude_peak_check_halfwidth_hz must be finite and nonnegative')
    prefix = 'global_above_cutoff_padded_bin_'
    d = detail.copy()
    d['selected_peak_hz'] = d[prefix+'hz']
    d['selected_peak_amplitude'] = d[prefix+'amplitude']
    d['frequency_offset_hz'] = d.selected_peak_hz-c['target_hz']
    d['near_target'] = d.frequency_offset_hz.abs().le(tolerance)
    d['finite_peak'] = np.isfinite(d.selected_peak_hz) & np.isfinite(d.selected_peak_amplitude)
    d['global_reference_diagnostic_ok'] = (d.finite_peak & d.near_target &
        (d.sensor.eq('IM') | d.extrema_fraction.lt(c['extrema_fraction_warning'])))
    d['global_reference_ok'] = d.global_reference_diagnostic_ok & d.gap_ok
    columns = ['file','sensor','level','requested_start_s','requested_end_s',
               'selected_peak_hz','selected_peak_amplitude','unit','frequency_offset_hz',
               'near_target','finite_peak','gap_ok','gap_duration_fraction',
               'global_reference_ok','global_reference_diagnostic_ok']
    check = d[columns]
    check.to_csv(run.path/'amplitude_global_peak_check.csv',index=False)
    requested = {s:len(v) for s,v in amplitude_files(c).items()}
    summary = []
    for (sensor,level), p in d.groupby(['sensor','level']):
        values = p.selected_peak_amplitude.where(p.finite_peak)
        summary.append(dict(sensor=sensor,level=level,n=len(p),n_requested=requested[sensor],
            n_finite=int(p.finite_peak.sum()),peak_mean=values.mean(),peak_sd=values.std(),
            min_peak_hz=p.selected_peak_hz.min(),max_peak_hz=p.selected_peak_hz.max(),
            all_near_target=bool(p.near_target.all()),
            all_strict_ok=bool(p.global_reference_ok.all()),
            all_diagnostic_ok=bool(p.global_reference_diagnostic_ok.all()),
            gap_warning_records=int((~p.gap_ok).sum())))
    summary = pd.DataFrame(summary)
    summary.to_csv(run.path/'amplitude_global_peak_summary.csv',index=False)

    # Full-range frequency panel prevents harmonics/off-target maxima being hidden.
    # Offset panel zooms automatically only when all peaks lie near the target.
    sensors = list(d.sensor.unique())
    fig, axes = plt.subplots(len(sensors),2,figsize=(12,4*len(sensors)),squeeze=False)
    for row,sensor in zip(axes,sensors):
        p=d[d.sensor.eq(sensor)]
        for name,part in p.groupby('file'):
            part=part.sort_values('level')
            row[0].plot(part.level,part.selected_peak_hz,'o-',label=name)
            row[1].plot(part.level,part.frequency_offset_hz,'o-',label=name)
            bad=part[~part.near_target & part.finite_peak]
            row[0].scatter(bad.level,bad.selected_peak_hz,color='red',marker='x',s=90,zorder=5)
        row[0].axhspan(c['target_hz']-tolerance,c['target_hz']+tolerance,color='green',alpha=.12)
        row[0].axhline(c['target_hz'],color='gray',ls='--',lw=1)
        row[0].set_ylim(bottom=0)
        row[0].set(title=f'{sensor}: selected maximum (red x = outside check band)',ylabel='Selected peak frequency (Hz)')
        row[1].axhline(0,color='gray',ls='--',lw=1)
        if not p.near_target.all():
            for edge in [-tolerance,tolerance]: row[1].axhline(edge,color='red',ls=':',lw=1)
        missing=int((~p.finite_peak).sum())
        row[1].set(title=f'Offset from {c["target_hz"]:g} Hz | near: {int(p.near_target.sum())}/{len(p)}, missing: {missing}',ylabel='Frequency offset (Hz)')
        for ax in row:
            ax.set_xlabel('Volume setting'); ax.grid(alpha=.2); ax.legend(fontsize=8)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.suptitle(f'Global FFT maximum, f > {cutoff:g} Hz | full trimmed windows | check band +/-{tolerance:g} Hz')
    fig.tight_layout()
    save_fig(run,fig,'amplitude_global_peak_frequency_check',figures)

    ratios=[]
    for level,p in summary.groupby('level'):
        p=p.set_index('sensor')
        if not {'IM','accel'} <= set(p.index): continue
        im,acc=p.loc['IM'],p.loc['accel']
        complete=bool(im.n==requested['IM'] and acc.n==requested['accel'])
        common=complete and c['reference_conditions_confirmed'] and acc.peak_mean>0
        strict=bool(common and im.all_strict_ok and acc.all_strict_ok)
        diagnostic=bool(common and im.all_diagnostic_ok and acc.all_diagnostic_ok)
        ratios.append(dict(level=level,IM_peak_counts=im.peak_mean,IM_sd_counts=im.peak_sd,
            accel_peak_g=acc.peak_mean,accel_sd_g=acc.peak_sd,IM_n=int(im.n),accel_n=int(acc.n),
            all_peaks_near_target=bool(im.all_near_target and acc.all_near_target),
            eligible=strict,eligible_diagnostic=diagnostic,
            counts_per_g=im.peak_mean/acc.peak_mean if strict else np.nan,
            diagnostic_counts_per_g=im.peak_mean/acc.peak_mean if diagnostic else np.nan,
            gap_warning_records=int(im.gap_warning_records+acc.gap_warning_records)))
    reference=pd.DataFrame(ratios,columns=['level','IM_peak_counts','IM_sd_counts','accel_peak_g',
        'accel_sd_g','IM_n','accel_n','all_peaks_near_target','eligible','eligible_diagnostic',
        'counts_per_g','diagnostic_counts_per_g','gap_warning_records'])
    reference.to_csv(run.path/'amplitude_global_peak_reference_response.csv',index=False)
    fits=[]
    if len(reference):
        square_figures = c.get('amplitude_reference_square_figures', False)
        if square_figures:
            side = float(c.get('amplitude_reference_figure_size_in', 7.0))
            if not np.isfinite(side) or side < 7:
                raise ValueError('amplitude_reference_figure_size_in must be finite and >= 7 inches')
            panels = [plt.subplots(figsize=(side,side)) for _ in range(2)]
            axes = [ax for _,ax in panels]
            for ax in axes:
                ax.set_box_aspect(1)
        else:
            fig,axes=plt.subplots(1,2,figsize=(12,4))
        def errors(column):
            return reference[column] if reference[column].notna().any() else None
        axes[0].errorbar(reference.accel_peak_g,reference.IM_peak_counts,
            xerr=errors('accel_sd_g'),yerr=errors('IM_sd_counts'),fmt='o',capsize=3)
        for r in reference.itertuples():
            axes[0].annotate(str(r.level),(r.accel_peak_g,r.IM_peak_counts))
        for population,column in [('strict_gap','eligible'),('diagnostic_with_gap_warnings','eligible_diagnostic')]:
            p=reference[reference[column]]
            if len(p)<3 or np.ptp(p.accel_peak_g)<=0: continue
            coef=np.polyfit(p.accel_peak_g,p.IM_peak_counts,1)
            residual=p.IM_peak_counts-np.polyval(coef,p.accel_peak_g)
            total=np.sum((p.IM_peak_counts-p.IM_peak_counts.mean())**2)
            fits.append(dict(population=population,n_levels=len(p),slope_counts_per_g=coef[0],
                intercept_counts=coef[1],r_squared=1-np.sum(residual**2)/total if total>0 else np.nan,
                method='Unweighted free-intercept OLS of independent condition means; both-axis errors not modeled'))
            if population=='diagnostic_with_gap_warnings':
                x=np.linspace(p.accel_peak_g.min(),p.accel_peak_g.max(),100)
                label = 'Diagnostic OLS\n(check-band qualified)' if square_figures else 'Diagnostic OLS (check-band qualified)'
                axes[0].plot(x,np.polyval(coef,x),'--',label=label)
                axes[0].legend(fontsize=7)
        axes[0].set(xlabel='Accelerometer global FFT peak (g)',ylabel='IM global FFT peak (counts)',
                    title='All selected maxima: mean +/- record SD')
        axes[1].plot(reference.level,reference.diagnostic_counts_per_g,'o--',label='Diagnostic (gap warnings)')
        axes[1].plot(reference.level,reference.counts_per_g,'s',label='Strict gap-qualified')
        axes[1].set(xlabel='Volume setting',ylabel='counts/g',title='Ratios require all peaks in check band')
        axes[1].legend(fontsize=8)
        for ax in axes: ax.grid(alpha=.2)
        if square_figures:
            # Figure legends stay below the axes in the shared paper style.
            # This preserves the full OLS qualification without a wide side margin.
            for (fig,ax),suffix in zip(panels,('response','ratio')):
                handles,labels = ax.get_legend_handles_labels()
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                if handles:
                    fig.legend(handles,labels,loc='lower center')
                fig.suptitle(f'Global FFT maximum, f > {cutoff:g} Hz\nPadded bin amplitude | SD undefined for n=1')
                save_fig(run,fig,'amplitude_global_peak_reference_'+suffix+'_square',figures)
        else:
            fig.suptitle(f'Global FFT maximum, f > {cutoff:g} Hz | padded bin amplitude | SD undefined for n=1')
            fig.tight_layout()
            save_fig(run,fig,'amplitude_global_peak_reference_response',figures)
    fit=pd.DataFrame(fits,columns=['population','n_levels','slope_counts_per_g','intercept_counts','r_squared','method'])
    fit.to_csv(run.path/'amplitude_global_peak_response_fit.csv',index=False)
    return dict(global_peak_check=check,global_peak_summary=summary,
                global_peak_reference=reference,global_peak_fit=fit)
