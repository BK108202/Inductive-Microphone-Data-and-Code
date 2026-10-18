"""Supplement existing amplitude results with global-peak figures; no raw-data writes."""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from characterization_run import AnalysisRun
from amplitude_volume_repeats import summarize_volume_repeats, fit_volume_response, plot_volume_repeats
from portable_runtime import load_config, notebook_path, project_root, source_paths

BASE = project_root(__file__)
KEY = "global_padded_parabolic_amplitude"
FREQ = "global_padded_parabolic_hz"
METHOD = "Global maximum across all strictly positive zero-padded FFT bins (DC excluded), followed by valid three-point parabolic interpolation using the source interpolation scale. No 99--101 Hz restriction in peak selection. Source timestamp resampling, demeaning, periodic Hann and coherent-gain normalization retained. Missing interpolation is not replaced with a target peak."
QUALITY = "All finite global amplitudes enter descriptive means and sample SD (ddof=1). Preserve source fit eligibility: complete repeats, positive computed amplitudes, nominal 100 Hz, selected frequencies within 1 Hz of nominal, and accelerometer extrema checks. This frequency check flags fit eligibility only; it never changes the global peak or removes descriptive points. OLS/ODR and ratio-of-condition-means follow the source method. Sequential sensor records are not paired. Timing warnings remain; this is not calibrated sensitivity."


def create_global_figures(source_run):
    source = Path(source_run).resolve()
    if json.loads((source / "run_status.json").read_text())["status"] != "completed":
        raise ValueError("Source analysis must be completed")
    paths = [source / name for name in ("volume_repeat_records.csv", "parameters.json", "inputs.json", "provenance.json", "run_status.json")]
    prior = json.loads(paths[1].read_text(encoding="utf-8"))
    rows = pd.read_csv(paths[0]).drop(columns=["reference_record_reasons", "reference_record_eligible"], errors="ignore")
    config = deepcopy(prior["config"])
    config["volume_reference"]["amplitude_key"] = KEY
    root = Path(load_config()["results_root"])
    parameters = dict(source_run=str(source), source_parameters=prior, effective_config=config,
                      signal_estimator=METHOD, aggregation_and_quality=QUALITY,
                      frequency_table="Interpolated global peak frequency per independent record and volume, Hz; displayed to 6 decimals. Full precision in CSV.",
                      raw_data_reprocessed=False)
    inputs = [dict(path=str(p), role="source_analysis_" + p.stem, sha256=sha256(p.read_bytes()).hexdigest()) for p in paths]
    inputs.append(dict(path=str(paths[2]), role="original_input_manifest", original_inputs=json.loads(paths[2].read_text())))
    scripts = source_paths(notebook_path("data10_amplitude_analysis"))
    with AnalysisRun(root, "amplitude_global_figures", parameters, inputs, scripts) as run:
        run.write_text("methods.txt", METHOD + "\n\n" + QUALITY + "\n\nFFT values reused from the source run; its archived scripts and provenance describe the original estimation.\n")
        records, groups, reference = summarize_volume_repeats(rows, config)
        fit = fit_volume_response(reference)
        records.to_csv(run.path / "global_records.csv", index=False)
        groups.to_csv(run.path / "global_groups.csv", index=False)
        reference.to_csv(run.path / "global_reference.csv", index=False)
        run.write_json("global_fit.json", fit)
        frequencies = records.pivot(index="volume_setting", columns="file_label", values=FREQ).sort_index()
        frequencies.to_csv(run.path / "global_peak_frequencies_hz.csv")
        comparison = records[["sensor", "file_label", "volume_setting", FREQ, KEY, "target_padded_parabolic_hz", "target_padded_parabolic_amplitude"]].copy()
        comparison["global_minus_target_amplitude"] = comparison[KEY] - comparison.target_padded_parabolic_amplitude
        comparison.to_csv(run.path / "global_target_comparison.csv", index=False)
        figures = []
        for name, fig in plot_volume_repeats(records, groups, reference, fit, KEY):
            for ax in fig.axes:
                ax.set_ylabel(ax.get_ylabel().replace("Target FFT", "Global FFT").replace("IM target", "IM global"))
                ax.set_xlabel(ax.get_xlabel().replace("Reference acceleration peak", "Reference global acceleration peak"))
            fig.suptitle("Global FFT peak: all positive frequencies; three-point interpolation", fontsize=12)
            fig.set_layout_engine(None)
            w, h = fig.get_size_inches()
            fig.set_size_inches(max(w, 10), h + 3.0)
            fig.subplots_adjust(bottom=0.34, top=0.91, hspace=0.42, wspace=0.25)
            table_ax = fig.add_axes([0.09, 0.045, 0.86, 0.20])
            table_ax.axis("off")
            table_ax.set_title("Selected global peak frequency [Hz] by volume and record", fontsize=10, pad=8)
            table = table_ax.table(cellText=[[f"{v:.6f}" for v in row] for row in frequencies.to_numpy()],
                                  rowLabels=[f"Volume {v:g}" for v in frequencies.index],
                                  colLabels=frequencies.columns.tolist(), cellLoc="center", loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.15)
            path = run.path / ("global_" + name + ".png")
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            figures.append(str(path))
        summary = dict(figure_count=len(figures), records=len(records),
                       frequency_min_hz=float(records[FREQ].min()), frequency_max_hz=float(records[FREQ].max()),
                       max_absolute_amplitude_difference_from_target=float(comparison.global_minus_target_amplitude.abs().max()),
                       timing_warning_records=int((~records.fft_valid.astype(bool)).sum()),
                       eligible_levels=int(reference.fit_eligible.sum()), limitations=prior.get("limitations", []))
        run.write_json("figures.json", figures)
        run.write_json("summary.json", summary)
        run.finish(summary)
    return dict(output_path=str(run.path), figures=figures, summary=summary)


if __name__ == "__main__":
    import sys
    result = create_global_figures(sys.argv[1])
    print(json.dumps(result, indent=2))
