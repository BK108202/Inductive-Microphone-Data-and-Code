"""Presentation only: Arial (DejaVu Sans fallback), 15 pt, editable SVG."""
from pathlib import Path
import json
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.collections import QuadMesh
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter, NullLocator
import numpy as np
from portable_runtime import figure_font

FONT = figure_font()
SETTINGS = {'svg.fonttype': 'none', 'font.family': FONT, 'font.size': 15,
            'axes.labelsize': 15, 'axes.titlesize': 15, 'xtick.labelsize': 15,
            'ytick.labelsize': 15, 'legend.fontsize': 15, 'figure.titlesize': 15,
            'figure.labelsize': 15, 'axes.grid': False, 'text.usetex': False,
            'mathtext.fontset': 'custom', 'mathtext.rm': FONT,
            'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold',
            'mathtext.sf': FONT, 'mathtext.fallback': None}
METHOD = ('Presentation only: all text uses Arial with 15 pt base size (math scripts keep '
          'normal relative sizing), grids are disabled, each y axis has at most five major '
          'tick values and no minor tick labels. Plotted data are preserved. Selected '
          'tracking comparisons use automatic upper y limits with legend headroom '
          'unless an exact per-axis upper limit is configured; lower limits and other '
          'axis limits are preserved. '
          'Frequency-response SNR plots show amplifier-on noise only; all original conditions '
          'remain in numerical CSVs. Dotted noise overlays are omitted from tracking/FFT '
          'response figures. Dashed threshold, timing, fit and diagnostic-population lines '
          'have different meanings and are retained. SVG text is not outlined. Layout is '
          'expanded/reflowed for legibility; analysis definitions and estimates are unchanged.')


def amplifier_on_display(table):
    """Select the display population, leaving original analysis tables untouched."""
    if table.empty:
        return table.copy()
    if 'noise_condition' not in table.columns:
        raise ValueError('Noise condition is required for amplifier-on display selection')
    condition = table['noise_condition'].astype(str).str.lower().str.replace(' ', '_')
    return table.loc[condition.isin(['amplifier_on', 'power_amplifier_on'])].copy()


def _response_overlays(fig, name):
    removed = 0
    target = None
    if name.startswith('tracking_comparison_'):
        target = fig.axes[1]
        target.set_title('Estimator-equivalent SNR: amplifier on')
    elif name.startswith('fft_diagnostic_'):
        target = fig.axes[0]
        target.set_title('Whole sweep FFT: IM spectral magnitude')
    elif name == 'fft_accelerometer_reference':
        target = fig.axes[0]
    if target is not None:
        for line in list(target.lines):
            if line.get_linestyle() == ':':
                line.remove()
                removed += 1
        if target.get_legend() is not None:
            target.legend(loc='upper right')
    return removed


def _font(fig, font_family="Arial", font_size=15):
    if font_family == "Arial":
        font_family = figure_font()
    for text in fig.findobj(match=Text):
        text.set_fontfamily(font_family)
        text.set_fontsize(font_size)
        text.set_usetex(False)
        text.set_math_fontfamily('custom')


def _legend_band_data_top(ax, legend_box, gap_pixels):
    """Find the highest drawn segment across the legend's horizontal footprint."""
    left, right = legend_box.x0 - gap_pixels, legend_box.x1 + gap_pixels
    paths = [(line.get_path(), line.get_transform()) for line in ax.lines if line.get_visible()]
    paths += [(path, collection.get_transform()) for collection in ax.collections
              if collection.get_visible() for path in collection.get_paths()]
    tops = []
    for path, transform in paths:
        vertices = transform.transform_path(path).vertices
        finite = np.all(np.isfinite(vertices), axis=1)
        inside = finite & (vertices[:, 0] >= left) & (vertices[:, 0] <= right)
        if inside.any():
            tops.append(float(vertices[inside, 1].max()))
        # Include segments that cross into the band even when neither endpoint
        # is inside it. Interpolate in display coordinates, as rendered.
        start, end = vertices[:-1], vertices[1:]
        overlap = (finite[:-1] & finite[1:] & (np.maximum(start[:, 0], end[:, 0]) >= left)
                   & (np.minimum(start[:, 0], end[:, 0]) <= right))
        start, end = start[overlap], end[overlap]
        if len(start):
            delta = end - start
            t0, t1 = np.zeros(len(start)), np.ones(len(start))
            moving = delta[:, 0] != 0
            edges = (np.asarray([left, right]) - start[moving, 0, None]) / delta[moving, 0, None]
            t0[moving] = np.maximum(0., edges.min(axis=1))
            t1[moving] = np.minimum(1., edges.max(axis=1))
            tops.append(float(np.maximum(start[:, 1] + t0 * delta[:, 1],
                                         start[:, 1] + t1 * delta[:, 1]).max()))
    return max(tops) if tops else None


def _tracking_legend_headroom(fig, original_locators):
    """Reserve rendered legend space on opted-in tracking figures, including log axes.

    Keep the lower limit and use at least 12% extra space in the displayed y scale.
    A figure can override that fraction with ``_tracking_legend_headroom_fraction``
    and the six-point legend clearance with ``_tracking_legend_gap_points``.
    ``_tracking_ymax`` can supply one exact upper limit per axis; a None entry keeps
    automatic headroom. Explicit upper limits take precedence over automatic space.
    An inside legend gets a clear band above curves within its horizontal footprint;
    an outside legend needs only the modest minimum padding. Call after fonts,
    figure size and layout are final.
    """
    extra_fraction = float(getattr(fig, '_tracking_legend_headroom_fraction', .12))
    gap_points = float(getattr(fig, '_tracking_legend_gap_points', 6.))
    if not np.isfinite(extra_fraction) or extra_fraction < 0:
        raise ValueError('Tracking legend headroom fraction must be finite and nonnegative')
    if not np.isfinite(gap_points) or gap_points < 0:
        raise ValueError('Tracking legend clearance must be finite and nonnegative')
    requested_ymax = getattr(fig, '_tracking_ymax', None)
    if requested_ymax is None:
        requested_ymax = [None] * len(fig.axes)
    else:
        try:
            requested_ymax = list(requested_ymax)
        except TypeError as exc:
            raise ValueError('Tracking y maxima must contain one number or None per axis') from exc
        if len(requested_ymax) != len(fig.axes):
            raise ValueError('Tracking y maxima must contain one number or None per axis')
    for index, (ax, requested) in enumerate(zip(fig.axes, requested_ymax)):
        if requested is None:
            continue
        try:
            upper = float(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Tracking y maximum for axis {index} must be finite') from exc
        lo, hi = ax.get_ylim()
        if (isinstance(requested, (bool, np.bool_)) or not np.isfinite(upper)
                or upper <= lo or (ax.get_yscale() == 'log' and upper <= 0)):
            raise ValueError(f'Tracking y maximum for axis {index} must be finite and above its lower limit')
        if lo >= hi or ax.get_yscale() not in ('linear', 'log'):
            raise ValueError(f'Tracking y maximum requires an increasing linear or log axis (axis {index})')
        requested_ymax[index] = upper
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    adjustments = []
    for index, (ax, locator) in enumerate(zip(fig.axes, original_locators)):
        lo, hi = ax.get_ylim()
        if lo >= hi or ax.get_yscale() not in ('linear', 'log'):
            continue
        transform = ax.yaxis.get_transform()
        scaled_lo, scaled_hi = transform.transform(np.asarray([lo, hi], dtype=float))
        if not np.all(np.isfinite([scaled_lo, scaled_hi])):
            continue
        scaled_top = scaled_hi + extra_fraction * (scaled_hi - scaled_lo)
        axis_box = ax.get_window_extent(renderer)
        legend = ax.get_legend()
        legend_inside = False
        if legend is not None and legend.get_visible():
            legend_box = legend.get_window_extent(renderer)
            legend_inside = (legend_box.x0 < axis_box.x1 and legend_box.x1 > axis_box.x0
                             and legend_box.y0 < axis_box.y1 and legend_box.y1 > axis_box.y0)
            if legend_inside:
                gap_pixels = gap_points * fig.dpi / 72.
                available_fraction = (legend_box.y0 - gap_pixels - axis_box.y0) / axis_box.height
                data_top = _legend_band_data_top(ax, legend_box, gap_pixels)
                if available_fraction > 0 and data_top is not None:
                    data_fraction = min(1., (data_top - axis_box.y0) / axis_box.height)
                    scaled_data_hi = scaled_lo + data_fraction * (scaled_hi - scaled_lo)
                    scaled_top = max(scaled_top,
                                     scaled_lo + (scaled_data_hi - scaled_lo) / available_fraction)
        automatic_upper = float(transform.inverted().transform(np.asarray([scaled_top]))[0])
        upper = requested_ymax[index] if requested_ymax[index] is not None else automatic_upper
        ax.set_ylim(lo, upper)
        # The earlier style pass fixes the tick locations. Recompute them for the
        # enlarged limits from the original locator, not from that fixed subset.
        if not isinstance(locator, FixedLocator):
            candidates = (MaxNLocator(nbins=4).tick_values(lo, upper)
                          if ax.get_yscale() == 'linear' else locator.tick_values(lo, upper))
            candidates = candidates[np.isfinite(candidates) & (candidates >= lo) & (candidates <= upper)]
            if len(candidates) > 5:
                indices = np.unique(np.round(np.linspace(0, len(candidates)-1, 5)).astype(int))
                candidates = candidates[indices]
            ax.yaxis.set_major_locator(FixedLocator(candidates))
        adjustments.append(dict(axis=index, original_ylim=[lo, hi],
                                requested_ymax=requested_ymax[index],
                                automatic_ylim=[lo, automatic_upper],
                                expanded_ylim=[lo, upper], legend_inside=legend_inside,
                                minimum_extra_display_range_fraction=extra_fraction,
                                legend_gap_points=gap_points))
    return adjustments


def apply_style(fig, name, font_family="Arial", font_size=15):
    if font_family == "Arial":
        font_family = figure_font()
    removed = _response_overlays(fig, name)
    # Dense spectrogram cells remain an image layer; text and curves remain vectors.
    for mesh in fig.findobj(match=QuadMesh):
        if np.size(mesh.get_array()) > 20000:
            mesh.set_rasterized(True)
    # Materialize labels before changing locators; preserve categorical tick labels.
    fig.canvas.draw()
    original_y_locators = [ax.yaxis.get_major_locator() for ax in fig.axes]
    for ax in fig.axes:
        ax.grid(False, which='both', axis='both')
        ax.tick_params(axis='both', which='both', labelsize=font_size)
        lo, hi = sorted(ax.get_ylim())
        original = np.asarray(ax.get_yticks(), dtype=float)
        categorical = isinstance(ax.yaxis.get_major_locator(), FixedLocator)
        if ax.get_yscale() == 'linear' and not categorical:
            candidates = MaxNLocator(nbins=4).tick_values(lo, hi)
        else:
            candidates = original
        candidates = candidates[np.isfinite(candidates) & (candidates >= lo) & (candidates <= hi)]
        if len(candidates) > 5:
            candidates = candidates[np.unique(np.round(np.linspace(0, len(candidates)-1, 5)).astype(int))]
        # Keep all category strings mapped to their original values, even when thinned.
        labels = {float(tick): label.get_text() for tick, label in zip(original, ax.get_yticklabels())}
        ax.yaxis.set_major_locator(FixedLocator(candidates))
        if categorical:
            ax.set_yticklabels([labels.get(float(tick), str(tick)) for tick in candidates])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
    _font(fig, font_family, font_size)
    width, height = fig.get_size_inches()
    square = len(fig.axes) == 1 and fig.axes[0].get_box_aspect() == 1
    width = max(width, 7 if square else 11)
    # Put long legends beside the data instead of increasing their overlap at 15 pt.
    outside = False
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            longest = max((len(t.get_text()) for t in legend.get_texts()), default=0)
            if longest > 32 or len(legend.get_texts()) > 5:
                legend.set_loc('upper left')
                legend.set_bbox_to_anchor((1.02, 1))
                outside = True
    if outside:
        width += 5
    footers = [t for t in fig.texts if t.get_visible() and t.get_text() and t.get_position()[1] < .16]
    max_chars = max(50, int((width-.8)*72/7.7))
    bottom_inches = .25
    for text in footers:
        wrapped = '\n'.join(textwrap.fill(line, width=max_chars, break_long_words=False,
            break_on_hyphens=False) for line in text.get_text().splitlines())
        text.set_text(wrapped)
        bottom_inches += (wrapped.count('\n')+1)*.25 + .15
    legend_inches = 0.
    renderer = fig.canvas.get_renderer()
    legend_heights = []
    for legend in fig.legends:
        legend.set_in_layout(False)
        extent = legend.get_window_extent(renderer)
        legend_heights.append(max(.4, extent.height/fig.dpi)+.15)
        legend_inches += legend_heights[-1]
    height = max(height, 4.8) + max(0, bottom_inches-.5) + legend_inches
    if square:
        width = height = max(width, height)
    fig.set_size_inches(width, height)
    baseline = .12/height
    for text in reversed(footers):
        text.set_position((.02, baseline))
        text.set_ha('left'); text.set_va('bottom')
        baseline += ((text.get_text().count('\n')+1)*.25+.15)/height
    legend_bottom = bottom_inches/height
    for legend, band in zip(fig.legends, legend_heights):
        legend.set_loc('lower center')
        legend.set_bbox_to_anchor((.5, legend_bottom), transform=fig.transFigure)
        legend_bottom += band/height
    # Wrap long titles at the subplot width without reducing their type size.
    for ax in fig.axes:
        chars = max(35, int(ax.get_position().width*width*72/7.7))
        title = ax.get_title()
        if title:
            ax.set_title('\n'.join(textwrap.fill(line, width=chars, break_long_words=False,
                break_on_hyphens=False) for line in title.splitlines()), fontsize=font_size)
    bottom = min(.55, (bottom_inches+legend_inches)/height) if footers or fig.legends else 0
    engine = fig.get_layout_engine()
    if isinstance(engine, ConstrainedLayoutEngine):
        # A colorbar fixes layout-engine compatibility; keep constrained layout.
        engine.set(rect=(0, bottom, 1, 1-bottom), w_pad=.12, h_pad=.12)
    else:
        if engine is not None:
            fig.set_layout_engine('none')
        fig.tight_layout(rect=(0, bottom, 1, 1), pad=1.3)
    _font(fig, font_family, font_size)
    fig.canvas.draw()
    headroom = []
    if getattr(fig, '_tracking_legend_headroom', False):
        headroom = _tracking_legend_headroom(fig, original_y_locators)
        _font(fig, font_family, font_size)
        fig.canvas.draw()
    audits = []
    for i, ax in enumerate(fig.axes):
        lo, hi = sorted(ax.get_ylim())
        ticks = [float(v) for v in ax.get_yticks() if lo <= v <= hi]
        assert len(ticks) <= 5, (name, i, ticks)
        assert not any(line.get_visible() for line in [*ax.get_xgridlines(), *ax.get_ygridlines()])
        audits.append(dict(axis=i, ylabel=ax.get_ylabel(), yscale=ax.get_yscale(),
                           ylim=list(ax.get_ylim()), y_tick_values=ticks, y_tick_count=len(ticks)))
    assert all(t.get_fontsize()==font_size for t in fig.findobj(match=Text))
    return dict(figure=name, font=font_family, base_fontsize_pt=font_size, grids=False,
                square_canvas=square, figure_size_inches=list(fig.get_size_inches()),
                removed_dotted_noise_curves=removed, tracking_legend_headroom=headroom, axes=audits)


def save_paper_figure(run, fig, name, figures, dpi=150, font_family="Arial", font_size=15):
    if font_family == "Arial":
        font_family = figure_font()
    png = run.path/(name+'.png')
    svg = run.path/(name+'.svg')
    try:
        if png.exists() or svg.exists():
            raise FileExistsError(f'Refusing to overwrite {name}')
        if not np.isfinite(font_size) or font_size <= 0:
            raise ValueError("font_size must be positive and finite")
        font = findfont(FontProperties(family=font_family), fallback_to_default=False)
        settings = dict(SETTINGS)
        settings.update({key: font_size for key in ('font.size', 'axes.labelsize', 'axes.titlesize',
            'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize', 'figure.titlesize', 'figure.labelsize')})
        settings.update({'font.family': font_family, 'mathtext.rm': font_family,
            'mathtext.it': font_family + ':italic', 'mathtext.bf': font_family + ':bold',
            'mathtext.sf': font_family})
        with mpl.rc_context(settings):
            audit = apply_style(fig, name, font_family=font_family, font_size=font_size)
            # Tight cropping changes the aspect of an otherwise square canvas.
            bounds = None if audit['square_canvas'] else 'tight'
            with mpl.rc_context({'savefig.bbox': None}):
                fig.savefig(png, dpi=dpi, bbox_inches=bounds)
                fig.savefig(svg, format='svg', dpi=300, bbox_inches=bounds)
    finally:
        plt.close(fig)
    run.update_parameters({'figure_export': dict(formats=['png','svg'], font_family=font_family,
        font_file=font, font_size_pt=font_size, y_tick_maximum=5, grid=False,
        svg_fonttype='none', editable_text=True, png_dpi=dpi, svg_raster_layer_dpi=300,
        dense_heatmap_cells_rasterized=True, method=METHOD.replace("Arial with 15 pt", f"{font_family} with {font_size:g} pt"),
        matplotlib_settings=settings)})
    figures.append(str(png))
    run.write_json('figures.json', figures)
    run.write_json('svg_figures.json', [str(Path(p).with_suffix('.svg')) for p in figures])
    audit_path = run.path/'figure_style_audit.json'
    audits = json.loads(audit_path.read_text(encoding='utf-8')) if audit_path.exists() else []
    audits.append(audit)
    run.write_json('figure_style_audit.json', audits)
