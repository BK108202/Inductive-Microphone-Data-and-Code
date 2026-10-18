"""Explicit Data11 experiment manifest and analysis settings (not Data9/10 defaults)."""
from copy import deepcopy
from pathlib import Path
from numbers import Integral
import math
from portable_runtime import data_root, project_root

PROJECT = project_root(__file__)
DATA_ROOT = str(data_root("Data11"))

# Shared candidate catalog for reusable analysis functions. The published frequency
# notebook explicitly overrides this with one selected sweep per enabled material.
MATERIALS = {
    'OO-50 0.23 mm': ([1, 2], [3, 4, 5]),
    '10 NV': ([18, 19], [20, 21, 22]),
    '945': ([23, 24], [25, 26, 27, 62, 63, 64]),
    '12': ([28, 29], [30, 31, 32]),
    '18': ([33, 34], [35, 36, 37, 57, 58]),
    'OO-50 4.05 mm': ([38, 39], [40, 41, 42]),
    'Foam 4.17 mm': ([43, 44], [45, 46, 47]),
    'OO-50 large coil, silicone bonded, 3.5 uH': ([49, 50], [51, 52, 60, 61]),
}

# User manifest update, 2026-10-13: IM_53 is amplitude, never a sweep.
AMPLITUDE_CANDIDATES = [15, 16, 17, 53, 54, 55, 56]

COMMON = {
    'data_root': DATA_ROOT,
    'accel_g_per_count': 0.000488,
    'accel_conversion_source': 'User confirmed +/-16 g and Raw_Z * 0.000488 on 2026-10-10; nominal conversion, not individual calibration.',
    'reference_conditions_confirmed': True,
    'reference_conditions_source': 'User confirmed identical drive settings, mounting position and load on 2026-10-10; independently recorded batches, no file pairing.',
    'timestamp_basis': 'Time_us converted to seconds; whether timestamps equal physical sample instants and clock accuracy remain unverified.',
    'max_gap_ratio': 3.0,
    'dt_cv_warning': 0.01,
    'fft_pad_factor': 8,
    'im_endpoint_codes': [0, 268435455],
    'im_endpoint_scope': 'record',
    'im_endpoint_policy': 'Conservative primary comparison uses endpoint-free IM records. Some endpoint-bearing records also contain huge excursions between endpoints; retain all local-window estimates as diagnostics. No raw correction; cause unverified.',
    'missing_policy': 'record_and_skip',
}

def default_config(kind):
    c = deepcopy(COMMON)
    c['analysis'] = kind
    if kind == 'frequency_range':
        c.update(materials=deepcopy(MATERIALS), accel_sweeps=[3, 4, 5, 6],
                 accel_noise=[1, 2], f0_hz=30., f1_hz=300., nominal_duration_s=30.,
                 start_search_s=[0., 12.], duration_search_s=[27., 33.],
                 onset_overrides_s={}, duration_overrides_s={},
                 window_s=0.8, track_hop_s=0.5, grid_points=81,
                 track_min_fraction=0.15, track_min_points=16, track_min_span_s=10.,
                 # 0.15 is an unvalidated heuristic, not a calibrated detection/cutoff threshold.
                 track_max_log_residual=0.015, response_min_fraction=0.15,
                 noise_start_s=5., max_noise_windows=40,
                 snr_threshold_db=10., minimum_repeat_count=2,
                 comparison_groups={
                     'materials': ['OO-50 0.23 mm', '10 NV', '945', '12', '18'],
                     'thickness': ['OO-50 0.23 mm', 'OO-50 4.05 mm'],
                     'thick_materials': ['OO-50 4.05 mm', 'Foam 4.17 mm'],
                     'coil_design': ['OO-50 0.23 mm', 'OO-50 large coil, silicone bonded, 3.5 uH'],
                 })
    elif kind == 'resolution':
        c.update(files={f'IM_{i}.csv': f for f, ids in [(100., [6,7,8]), (100.01,[9,10,11]), (100.005,[12,13,14])] for i in ids},
                 windows_s={'primary_50s': [0., 50.], 'trimmed_40s': [5.,45.]},
                 global_search_min_hz=1.0,
                 resolution_plot_base_hz=100.0, resolution_spectrum_halfwidth_hz=0.08,
                 resolution_fit_example='IM_12.csv',
                 target_center_hz=100., target_halfwidth_hz=1.,
                 sine_search_halfwidth_bins=2., sine_coarse_points=17, sine_tolerance_hz=1e-9)
    elif kind == 'amplitude':
        c.update(files={'IM': [15,16,17], 'accel': [7,8,9]}, levels=[25,24,23,22,21,20],
                 max_files={'IM': None, 'accel': None},
                 global_search_min_hz=1.0, amplitude_peak_check_halfwidth_hz=1.0,
                 dwell_s=20., dwell_overrides_s={
                     **{f'IM_{i}.csv':30. for i in [53,54,55,56]},
                     **{f'accel_{i}.csv':30. for i in [10,11,12]},
                 },
                 start_s=0., start_overrides_s={}, bounds_overrides_s={},
                 trim_s=2., central_window_s=16., target_hz=100., target_halfwidth_hz=1.,
                 subwindow_s=4., extrema_fraction_warning=0.01,
                 schedule_source='User confirmed IM_15-17: 20s/level; IM_53-56: 30s/level; six descending levels at 100Hz. User confirmed accel_10-12: 30s/level on 2026-10-15 and requested central 26s in the amplitude notebook; accel_7-9 retain 20s/level. Shared central-window default remains 16s; notebook controls override it. start_s=0 remains an analysis assumption; inspect boundaries or override.',
                 noise_files={'IM':[1,2], 'accel':[1,2]})
    else:
        raise ValueError(kind)
    return c


def amplitude_bounds(config, filename):
    """Explicit bounds take precedence over per-file dwell and global defaults."""
    start = config['start_overrides_s'].get(filename, config['start_s'])
    dwell = config.get('dwell_overrides_s', {}).get(filename, config['dwell_s'])
    if dwell <= 0:
        raise ValueError('Amplitude dwell must be positive')
    return config['bounds_overrides_s'].get(filename,
        [[start+i*dwell, start+(i+1)*dwell] for i in range(len(config['levels']))])


def amplitude_analysis_bounds(config, filename):
    """Center a fixed-duration window inside each actual configured level block.

    Explicit central_window_s=None retains legacy symmetric trim_s behavior.
    Never infer dwell from recording length or shorten an unsupported window.
    """
    bounds = amplitude_bounds(config, filename)
    if (len(bounds) != len(config['levels']) or
            any(not math.isfinite(a) or not math.isfinite(b) or a < 0 or b <= a for a,b in bounds) or
            any(bounds[i][0] < bounds[i-1][1] for i in range(1, len(bounds)))):
        raise ValueError('amplitude bounds must be finite, ordered, nonoverlapping, one per level')
    duration = config.get('central_window_overrides_s', {}).get(filename, config.get('central_window_s'))
    if duration is not None and (isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0):
        raise ValueError('central_window_s must be positive and finite, or None for legacy trims')
    result = []
    for a,b in bounds:
        if duration is None:
            trim = config['trim_s']
            if not math.isfinite(trim) or trim < 0 or 2*trim >= b-a:
                raise ValueError('trim_s leaves no supported amplitude window')
            lo,hi = a+trim,b-trim
        else:
            if b-a < duration:
                raise ValueError(f'{filename}: block [{a}, {b}) is shorter than {duration:g} s')
            center = (a+b)/2
            lo,hi = center-duration/2,center+duration/2
        result.append([lo,hi])
    return result


def amplitude_subwindow_bounds(start, end, duration):
    """Complete equal-duration subwindows; any trailing remainder is not shortened."""
    if not all(math.isfinite(v) for v in (start, end, duration)) or end <= start or duration <= 0:
        raise ValueError('Subwindow bounds and duration must be finite and positive')
    count = int(math.floor((end-start)/duration + 1e-10))
    return [[start+i*duration, start+(i+1)*duration] for i in range(count)]


def amplitude_files(config):
    """Resolve ordered file selections without changing the requested configuration.

    None uses the entire list; 0 disables that sensor. Limits are exact requested
    counts, not a search for that many valid files or a pairing between sensors.
    """
    files = config['files']
    limits = config.get('max_files', {})
    if set(files) - {'IM', 'accel'} or set(limits) - {'IM', 'accel'}:
        raise ValueError('amplitude files/max_files only support IM and accel')
    selected = {}
    seen = set()
    for sensor in ('IM', 'accel'):
        items = files.get(sensor, [])
        if not isinstance(items, (list, tuple)):
            raise ValueError(f'files[{sensor}] must be a list of IDs or CSV filenames')
        names = []
        for item in items:
            if isinstance(item, Integral) and not isinstance(item, bool) and item > 0:
                name = f'{sensor}_{item}.csv'
            elif isinstance(item, str) and item.lower().endswith('.csv'):
                name = item
            else:
                raise ValueError(f'Invalid {sensor} file: {item!r}; use positive IDs or CSV filenames')
            names.append(name)
        count = limits.get(sensor)
        if count is not None and (isinstance(count, bool) or not isinstance(count, Integral)
                                  or not 0 <= count <= len(names)):
            raise ValueError(f'max_files[{sensor}] must be None or an integer from 0 to {len(names)}')
        selected[sensor] = names if count is None else names[:count]
        for name in selected[sensor]:
            path = (Path(config['data_root']) / name).resolve()
            if path in seen:
                raise ValueError(f'Duplicate selected input: {name}; one file is not multiple repeats')
            seen.add(path)
    if not any(selected.values()):
        raise ValueError('Select at least one amplitude file')
    return selected
