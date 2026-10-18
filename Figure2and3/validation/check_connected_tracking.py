"""Check connected/strict plot values and legend clearance using saved tables."""
import argparse
import ast
from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nbformat
import data11_harmonic_tracking as tracking
import data11_reference_gap_plots as gap_views
import paper_figure_style as style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', type=Path, default=ROOT/'reference_results/data11_harmonic_tracking')
    args = parser.parse_args()
    parent = args.parent_run.resolve()
    config = json.loads((parent/'parameters.json').read_text(encoding='utf-8'))['config']
    summary = pd.read_csv(parent/'harmonic_tracking_summary.csv')
    strict = pd.read_csv(parent/'acceleration_normalized_responsivity.csv')
    detail = pd.read_csv(parent/'harmonic_tracking_records.csv')
    before = [table.copy(deep=True) for table in [summary, strict, detail]]
    before_config = deepcopy(config)
    display, candidates = tracking.diagnostic_reference(detail, config, strict)
    accel_display = tracking.accelerometer_display(detail, config, summary)
    accel_strict = summary[summary.material.eq('accelerometer')]
    accel_added = accel_display.reference_gap_diagnostic
    assert accel_added.any()
    assert set(accel_display.loc[accel_added, 'grid_index']) <= set(candidates.grid_index)
    for metric in ['amplitude_peak', 'snr_db']:
        original = accel_strict[metric + '_mean'].to_numpy()
        connected = accel_display['display_' + metric].to_numpy()
        np.testing.assert_array_equal(accel_display['strict_' + metric], original)
        valid = np.isfinite(original)
        np.testing.assert_array_equal(connected[valid], original[valid])
        # This single-reference fixture must reuse exact archived estimates.
        if len(config['accel_sweeps']) == 1:
            field = 'amplitude_peak' if metric == 'amplitude_peak' else 'amplifier_on_snr_db'
            flagged = accel_display[metric + '_gap_diagnostic']
            expected = candidates.set_index('grid_index').loc[accel_display.loc[flagged, 'grid_index'], field]
            np.testing.assert_array_equal(connected[flagged], expected)
    assert gap_views.diagnostic_reference is tracking.diagnostic_reference
    same = display.eligible
    np.testing.assert_array_equal(display.loc[same, 'display_counts_per_g'], strict.loc[same, 'counts_per_g'])
    added = display.reference_gap_diagnostic
    assert added.any() and display.loc[added, 'display_counts_per_g'].notna().all()
    assert strict.loc[added, 'counts_per_g'].isna().all()
    assert candidates.gap_ok.eq(False).all() and candidates.sensor.eq('accel').all()
    audit = {}
    values = {}
    all_values = {}
    figures = []
    expected_ymax = {
        'tracking_comparison_materials': [40000., 120., 80000.],
        'tracking_comparison_coil_design': [150000., 85., 40000.],
    }

    def capture(run, fig, name, all_figures):
        # Match the actual exporter, including final font sizes and layout.
        report = style.apply_style(fig, name, font_size=config.get('figure_font_size_pt', 15))
        values[name] = [[(line.get_label(), np.asarray(line.get_ydata()).copy())
                         for line in ax.lines if not line.get_label().startswith('_')]
                        for ax in fig.axes]
        all_values[name] = [[np.asarray(line.get_ydata()).copy() for line in ax.lines] for ax in fig.axes]
        clearance = []
        comparison_name = name.removesuffix('_strict')
        if comparison_name in expected_ymax:
            assert len(report['tracking_legend_headroom']) == 3
            for ax, entry, requested in zip(fig.axes, report['tracking_legend_headroom'], expected_ymax[comparison_name]):
                assert entry['expanded_ylim'][0] == entry['original_ylim'][0]
                assert entry['expanded_ylim'][1] > entry['original_ylim'][1]
                assert ax.get_ylim()[1] == requested, (name, ax.get_ylabel(), ax.get_ylim(), requested)
                assert entry['expanded_ylim'][1] == entry['requested_ymax'] == requested
                assert entry['automatic_ylim'][0] == entry['original_ylim'][0]
                assert np.isfinite(entry['automatic_ylim'][1])
                assert entry['automatic_ylim'][1] > entry['original_ylim'][1]
                if comparison_name == 'tracking_comparison_coil_design':
                    assert entry['minimum_extra_display_range_fraction'] == .30
                legend = ax.get_legend()
                if entry['legend_inside']:
                    box = legend.get_window_extent(fig.canvas.get_renderer())
                    top = style._legend_band_data_top(ax, box, 6 * fig.dpi / 72)
                    gap = box.y0 - top if top is not None else None
                    assert gap is None or gap >= 6 * fig.dpi / 72 - 1., (name, ax.get_ylabel(), gap)
                    clearance.append(gap)
                else:
                    clearance.append('outside')
            if name == comparison_name:
                text = ' '.join([ax.get_title() for ax in fig.axes] +
                                [t.get_text() for ax in fig.axes for t in ax.get_legend().get_texts()])
                assert all(word not in text.lower() for word in ['gap', 'diagnostic', 'strict'])
                assert all(line.get_marker() in ['None', '', ' '] for line in fig.axes[2].lines)
        if name == 'tracking_accelerometer_reference':
            assert all(word not in ' '.join(ax.get_title() for ax in fig.axes).lower()
                       for word in ['gap', 'diagnostic', 'strict'])
            assert all(ax.lines[0].get_marker() in ['None', '', ' '] for ax in fig.axes)
        audit[name] = {'headroom': report['tracking_legend_headroom'], 'legend_clearance_pixels': clearance}
        all_figures.append(str(Path(name + '.png')))
        plt.close(fig)

    with patch.object(tracking, 'save_fig', capture):
        tracking._plots(None, summary, strict, config, figures, display_reference=display,
                        display_accelerometer=accel_display)
    for group, members in config['comparison_groups'].items():
        members = [m for m in members if m in config['materials']]
        if not members:
            continue
        main_key = 'tracking_comparison_' + group
        strict_key = main_key + '_strict'
        for axis, metric in enumerate(['amplitude_peak_mean', 'snr_db_mean']):
            for (_, current), (_, previous) in zip(values[main_key][axis], values[strict_key][axis]):
                np.testing.assert_array_equal(current, previous)
            for material, current in values[main_key][axis]:
                np.testing.assert_array_equal(current, summary.loc[summary.material.eq(material), metric])
        for material, y in values[main_key][2]:
            np.testing.assert_array_equal(y, display.loc[display.material.eq(material), 'display_counts_per_g'])
        for material, y in values[strict_key][2]:
            np.testing.assert_array_equal(y, strict.loc[strict.material.eq(material), 'counts_per_g'])
        # Standalone fundamental exports repeat the first comparison panel.
        for (_, single), (_, combined) in zip(values['tracking_fundamental_' + group + '_no_region'][0],
                                               values[main_key][0]):
            np.testing.assert_array_equal(single, combined)
    for axis, metric in enumerate(['amplitude_peak', 'snr_db']):
        np.testing.assert_array_equal(all_values['tracking_accelerometer_reference'][axis][0],
                                      accel_display['display_' + metric])
        np.testing.assert_array_equal(all_values['tracking_accelerometer_reference_strict'][axis][0],
                                      accel_strict[metric + '_mean'])
    main_figures, supplements = tracking._figure_lists(figures)
    assert all(not Path(p).stem.endswith('_strict') for p in main_figures)
    assert all(Path(p).stem.endswith('_strict') for p in supplements)
    assert 'tracking_accelerometer_reference_strict.png' in supplements
    for table, original in zip([summary, strict, detail], before):
        pd.testing.assert_frame_equal(table, original)
    assert config == before_config
    for name in ['data11_frequency_range_analysis', 'data11_frequency_range_supplement']:
        book = nbformat.read(ROOT/(name+'.ipynb'), as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type == 'code': ast.parse(cell.source)
    for name in ['data11_harmonic_tracking.py', 'data11_reference_gap_plots.py', 'paper_figure_style.py',
                 'verify_release.py', 'replot_data11_tracking.py']:
        ast.parse((ROOT/name).read_text(encoding='utf-8'))
    result = {'status':'passed', 'finished_at':datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'scope':'In-memory check of saved plot values, exact default y-axis upper limits, strict-data preservation, notebook syntax and rendered legend clearance. No raw-data analysis or saved scientific figures.',
        'parent_run':str(parent), 'recovered_reference_points':int(added.sum()),
        'reference_windows':len(candidates), 'main_figures':len(main_figures), 'supplement_figures':len(supplements),
        'connected_accelerometer_grid_points':int(accel_added.sum()),
        'input_tables_and_config_unchanged':True, 'default_ymax':expected_ymax, 'plot_checks':audit,
        'source_hashes':{name:sha256((ROOT/name).read_bytes()).hexdigest() for name in
                        ['data11_harmonic_tracking.py','data11_reference_gap_plots.py','paper_figure_style.py','verify_release.py']},
        'checker_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
        'limitations':['No full raw-data tracking rerun.']}
    (ROOT/'validation/connected_tracking.json').write_text(
        json.dumps(result,indent=2,default=lambda value: value.item())+'\n',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ['status','recovered_reference_points','reference_windows','main_figures','supplement_figures']},indent=2))


if __name__ == '__main__':
    main()
