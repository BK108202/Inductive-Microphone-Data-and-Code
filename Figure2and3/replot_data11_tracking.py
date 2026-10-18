"""Redraw a completed Data11 tracking run without reprocessing raw recordings."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-run', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, help='Optional output-root override for this redraw')
    args = parser.parse_args()
    if args.results_root is not None:
        os.environ['IM_RESULTS_ROOT'] = str(args.results_root.resolve())
    os.environ.setdefault('MPLBACKEND', 'Agg')
    from data11_harmonic_tracking import export_tracking_figures
    from portable_runtime import notebook_path
    result = export_tracking_figures(args.parent_run, notebook_path('data11_frequency_range_analysis'))
    print(json.dumps({key: result[key] for key in ['output_path', 'figures', 'supplement_figures']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
