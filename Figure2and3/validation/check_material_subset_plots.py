"""Regression checks for edited material selections; no raw records are processed."""
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
import pandas as pd
import data11_harmonic_tracking as tracking


def check_case(name, config, detail, expected_groups, multi_record=False):
    original_config = deepcopy(config)
    selected = detail[detail.material.isin([*config['materials'], 'accelerometer'])].copy()
    summary, ratios, _ = tracking.summarize(selected, config)
    before_summary, before_ratios = summary.copy(deep=True), ratios.copy(deep=True)
    captured = {}

    def capture(run, figure, filename, figures):
        figure.canvas.draw()
        captured[filename] = {
            'labels': [label for ax in figure.axes for label in ax.get_legend_handles_labels()[1]],
            'titles': [ax.get_title() for ax in figure.axes],
            'first_axis_collections': len(figure.axes[0].collections),
        }
        plt.close(figure)

    with patch.object(tracking, 'save_fig', capture):
        tracking._plots(None, summary, ratios, config, [])
    assert config == original_config, 'Plotting mutated the caller configuration'
    pd.testing.assert_frame_equal(summary, before_summary)
    pd.testing.assert_frame_equal(ratios, before_ratios)
    observed_groups = {key.removeprefix('tracking_comparison_') for key in captured
                       if key.startswith('tracking_comparison_')}
    assert observed_groups == set(expected_groups), (name, observed_groups)
    for key, item in captured.items():
        if key.startswith(('tracking_comparison_', 'tracking_fundamental_')):
            assert set(item['labels']) <= set(config['materials']), (key, item['labels'])
            if not multi_record:
                assert item['first_axis_collections'] == 0
                assert 'selected sweep' in item['titles'][0]
    shaded = [key for key in captured if key.endswith('_with_region')]
    assert len(shaded) == (len(expected_groups) if multi_record else 0)
    assert len(captured) == len(expected_groups) * (3 if multi_record else 2) + 2
    return {'case': name, 'status': 'passed', 'comparison_groups': sorted(observed_groups),
            'figures_drawn_in_memory': len(captured), 'caller_config_and_tables_unchanged': True}


def main():
    parent = ROOT / 'reference_results/data11_harmonic_tracking'
    config = json.loads((parent / 'parameters.json').read_text(encoding='utf-8'))['config']
    detail = pd.read_csv(parent / 'harmonic_tracking_records.csv')
    original = deepcopy(config)
    cases = [check_case('published_selection', config, detail, list(config['comparison_groups']))]
    del config['materials']['12']
    cases.append(check_case('material_12_deleted_with_stale_group', config, detail, list(config['comparison_groups'])))
    config['comparison_groups']['removed_only'] = ['12']
    cases.append(check_case('entire_group_removed', config, detail, list(original['comparison_groups'])))
    config = deepcopy(original)
    config['materials'] = {'OO-50 0.23 mm': config['materials']['OO-50 0.23 mm']}
    cases.append(check_case('only_one_material_selected', config, detail, ['materials', 'thickness', 'coil_design']))
    config = deepcopy(original)
    del config['materials']['12']
    for _, sweep_ids in config['materials'].values():
        sweep_ids.append(999)  # Plot-branch fixture only; this file is never opened.
    cases.append(check_case('multiple_sweeps_with_material_deleted', config, detail,
                            list(original['comparison_groups']), multi_record=True))
    source = ROOT / 'data11_harmonic_tracking.py'
    ast.parse(source.read_text(encoding='utf-8'))
    report = {
        'status': 'passed', 'finished_at': datetime.now(ZoneInfo('America/New_York')).isoformat(),
        'scope': 'Plotting regression using archived tables in memory; no raw-data analysis, new scientific run, or saved figures.',
        'module_sha256': sha256(source.read_bytes()).hexdigest(),
        'checker_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'cases': cases,
        'limitations': ['No complete notebook or raw-data analysis rerun. Existing failed runs are left unchanged.'],
    }
    (ROOT / 'validation/material_subset_plots.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
