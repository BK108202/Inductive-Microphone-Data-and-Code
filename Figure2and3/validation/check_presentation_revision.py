"""Check the release presentation split without processing raw recordings."""
import ast
import argparse
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from datetime import datetime
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile
from zoneinfo import ZoneInfo

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1] if SCRIPT.parent.name == 'validation' else SCRIPT.parents[1] / 'GitHub_Release'
WORKSPACE = ROOT.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-zip', type=Path, default=WORKSPACE/'GitHub_Release.zip',
                    help='Original combined release ZIP, before the presentation split')
args = parser.parse_args()
sys.path.insert(0, str(ROOT))
os.environ['MPLBACKEND'] = 'Agg'
os.chdir(ROOT)
import nbformat
import pandas as pd
import matplotlib.pyplot as plt
import data11_harmonic_tracking as ht
import data11_record_review as review
import data11_reference_gap_plots as gaps
import data11_whole_sweep_fft as fft
import portable_runtime as runtime
import verify_release

report = {'status': 'running', 'scope': 'Presentation/code checks; no raw-data analysis or new scientific run.',
          'python': sys.executable, 'checks': {}, 'limitations': [
              'No new end-to-end execution of the split notebooks.',
              'Notebook flow checks substitute existing tables for analysis calls; they do not validate new scientific outputs.',
              'Plot checks draw existing archived summary tables in memory; no PNG/SVG identity claim.']}
checks = report['checks']
archive = zipfile.ZipFile(args.baseline_zip)
old_text = lambda name: archive.read('GitHub_Release/' + name).decode('utf-8-sig')
old_book = json.loads(old_text('data11_frequency_range_analysis.ipynb'))
books = {name: nbformat.read(ROOT/(name+'.ipynb'), as_version=4) for name in runtime.ALL_NOTEBOOKS}

for book in books.values():
    nbformat.validate(book)
    for cell in book.cells:
        if cell.cell_type == 'code':
            ast.parse(cell.source)
            assert not cell.outputs and cell.execution_count is None
for path in ROOT.glob('*.py'):
    ast.parse(path.read_text(encoding='utf-8-sig'))
checks['syntax_and_notebook_schema'] = {'status': 'passed', 'notebooks': len(books)}

old_tree = ast.parse(old_text('data11_harmonic_tracking.py'))
new_tree = ast.parse((ROOT/'data11_harmonic_tracking.py').read_text(encoding='utf-8'))
def without_plots(tree):
    tree = deepcopy(tree)
    tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name != '_plots']
    return ast.dump(tree, include_attributes=False)
assert without_plots(old_tree) == without_plots(new_tree)
for name in ['data11_config.py', 'data11_analysis.py', 'data11_record_review.py',
             'data11_reference_gap_plots.py', 'data11_whole_sweep_fft.py']:
    assert ast.dump(ast.parse(old_text(name))) == ast.dump(ast.parse((ROOT/name).read_text(encoding='utf-8')))
checks['scientific_module_asts'] = {'status': 'passed', 'only_tracking_function_changed': '_plots',
                                  'related_scientific_modules_unchanged': 5}

main = books['data11_frequency_range_analysis']
supplement = books['data11_frequency_range_supplement']
core_map = {2:2, 4:4, 8:6, 10:8, 12:10, 13:11, 16:14, 18:16}
for old_index, new_index in core_map.items():
    assert ast.dump(ast.parse(''.join(old_book['cells'][old_index]['source']))) == ast.dump(ast.parse(main.cells[new_index].source))
move_map = {6:3, 22:5, 27:7, 30:10, 31:11, 33:13, 35:15}
for old_index, new_index in move_map.items():
    old = ''.join(old_book['cells'][old_index]['source'])
    old = old.replace('notebook_path("data11_frequency_range_analysis")', 'notebook_path("data11_frequency_range_supplement")')
    old = old.replace('Run the tracking analysis cell successfully first.', 'Select a completed tracking run in the first code cell.')
    assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(supplement.cells[new_index].source))
checks['notebook_scientific_cells'] = {'status': 'passed', 'unchanged_main_code_cells': len(core_map),
    'moved_code_cells_equal_except_notebook_identity_and_error_message': len(move_map)}

parent = ROOT/'reference_results/data11_harmonic_tracking'
config = json.loads((parent/'parameters.json').read_text())['config']
read = lambda folder, name: pd.read_csv(folder/name)
summary = read(parent, 'harmonic_tracking_summary.csv')
reference = read(parent, 'acceleration_normalized_responsivity.csv')
intervals = read(parent, 'detectable_frequency_intervals.csv')
detail = read(parent, 'harmonic_tracking_records.csv')
for actual, expected in zip(ht.summarize(detail, config), [summary, reference, intervals]):
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-12)
assert summary.n_amplitude.max() == summary.n_snr.max() == 1
assert summary.amplitude_peak_sd.isna().all()
checks['archived_table_aggregation'] = {'status': 'passed', 'tables': 3,
                                      'maximum_accepted_records': 1, 'single_record_sd_all_nan': True}

def draw(c):
    captured = {}
    def capture(run, fig, name, figures):
        fig.canvas.draw()
        captured[name] = {'titles': [ax.get_title() for ax in fig.axes],
                          'first_axis_collections': len(fig.axes[0].collections)}
        plt.close(fig)
    with patch.object(ht, 'save_fig', capture):
        ht._plots(None, summary, reference, c, [])
    return captured
single = draw(config)
assert len(single) == 10 and not any(name.endswith('_with_region') for name in single)
for name, item in single.items():
    if name.startswith(('tracking_comparison_', 'tracking_fundamental_')):
        assert 'selected sweep' in item['titles'][0]
        assert 'SD' not in item['titles'][0] and 'mean' not in item['titles'][0]
        assert item['first_axis_collections'] == 0
multi = deepcopy(config)
for name, ids in multi['materials'].items():
    if ids[1]: ids[1].append(999)  # Plot-branch fixture only; no record is read or analyzed.
repeated = draw(multi)
assert len(repeated) == 14 and sum(name.endswith('_with_region') for name in repeated) == 4
checks['plot_branches_in_memory'] = {'status': 'passed', 'single_record_figures': 10,
    'multiple_record_figures': 14, 'single_record_sd_shading': False}

tracking_result = dict(output_path=str(parent), summary=summary, reference=reference, intervals=intervals,
    trajectories=pd.DataFrame(), audit=pd.DataFrame(), input_issues=[], unavailable_records=[], figures=[])
fft_parent = ROOT/'reference_results/data11_whole_sweep_fft'
fft_result = dict(output_path=str(fft_parent), figures=[], input_issues=[])
for key, filename in [('windows','sweep_window_audit.csv'), ('noise_windows','noise_window_audit.csv'),
                      ('reference','fft_responsivity.csv'), ('intervals','fft_detectable_intervals.csv')]:
    fft_result[key] = read(fft_parent, filename)
raw_result = dict(output_path=str(ROOT/'reference_results/data11_raw_examples'), windows=pd.DataFrame(), figures=[])
review_result = dict(output_path=str(ROOT/'reference_results/data11_candidate_review'), audit=pd.DataFrame(), selection=pd.DataFrame())
gap_result = dict(output_path=str(ROOT/'reference_results/data11_reference_gap_views'), figures=[])
with ExitStack() as stack:
    stack.enter_context(patch('IPython.display.display', lambda *a, **k: None))
    stack.enter_context(patch('importlib.reload', lambda module: module))
    stack.enter_context(patch.dict(os.environ, {'IM_TRACKING_RUN':str(parent)}))
    for module, function, result in [(ht,'run_tracking',tracking_result), (review,'run_review',review_result),
                                    (review,'export_raw_examples',raw_result), (gaps,'export_reference_gap_views',gap_result),
                                    (fft,'run_whole_fft',fft_result)]:
        stack.enter_context(patch.object(module, function, return_value=result))
    states = {}
    with redirect_stdout(io.StringIO()):
        for name in ['data11_frequency_range_analysis','data11_frequency_range_supplement']:
            state = {'get_ipython': lambda: None}
            for cell in books[name].cells:
                if cell.cell_type == 'code': exec(compile(cell.source, name, 'exec'), state)
            states[name] = state
for key, value in config.items():
    if key != 'data_root': assert states['data11_frequency_range_analysis']['CONFIG'][key] == value, key
assert states['data11_frequency_range_supplement']['CONFIG'] == config
assert states['data11_frequency_range_supplement']['NOTEBOOK_PATH'].stem == 'data11_frequency_range_supplement'
checks['independent_namespace_notebook_flow'] = {'status': 'passed', 'notebooks': 2,
    'analysis_calls_mocked_with_archived_results': True, 'main_and_supplement_selections_preserved': True}
try:
    runtime.validate_tracking_run('')
except ValueError:
    pass
else:
    raise AssertionError('Empty tracking parent accepted')
checks['explicit_parent_required'] = {'status': 'passed'}

report.update(status='passed', finished_at=datetime.now(ZoneInfo('America/New_York')).isoformat(),
              baseline_manifest_sha256=hashlib.sha256(archive.read('GitHub_Release/release_manifest.json')).hexdigest(),
              checker_original_path=str(Path(__file__).resolve()),
              checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
archive.close()
(ROOT/'validation/presentation_revision.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
