# Inductive microphone characterization and application

Code and data for the frequency, amplitude and single-tone frequency-estimation analyses associated with Figures 2 and 3, with supplementary diagnostics. Mechanical images in panels 2(a)/3(a), final figure assembly, and Figures 4/5 code and figure.

**The original combined release was validated on Windows with a fresh Python 3.11 environment:** six notebooks completed after directory relocation; all 210 scientific CSVs and 10 scientific JSON files matched the original-code reference results. The Data11 frequency notebook now separates its core analysis from optional diagnostics and labels single-record plots explicitly. See [VALIDATION.md](VALIDATION.md) for the checks on this revision and the limits of the earlier execution record. This folder has not been published and does not yet have a public URL.

## Notebooks

| Notebook | Analysis |
|---|---|
| `data10_frequency_range_analysis.ipynb` | Stepped-frequency response, three separate SNR definitions, reference ratios and raw examples |
| `data10_amplitude_analysis.ipynb` | Repeated amplitude levels, target-conditioned/global FFT estimates and reference fits |
| `data10_resolution_analysis.ipynb` | Three recordings per setting at 50/100/200 Hz bases; 100 s primary and 90 s comparison windows |
| `data11_frequency_range_analysis.ipynb` | One selected sweep per enabled material/structure: raw examples, quality review, harmonic tracking, SNR, connected nominal-responsivity display and strict detectable intervals |
| `data11_amplitude_analysis.ipynb` | Candidate preview and selected three-record responses; central 26 s for both sensors |
| `data11_resolution_analysis.ipynb` | Three recordings per setting around 100 Hz; 50 s primary and 40 s comparison windows |

An optional seventh notebook, `data11_frequency_range_supplement.ipynb`, holds strict comparison plots with missing reference points, historical previews, reference timestamp-gap diagnostics and the separate whole-sweep FFT comparison. It uses an explicitly selected completed tracking run. The published main frequency analysis uses IM_4/20/27/36/42/46/52 and the independent accel_3 reference; material 12 remains excluded. Single-record plots have no between-record SD shading. Existing CSV `_mean`/`_sd` fields remain compatible: with one accepted record, the mean equals that record and sample SD is undefined.

Main comparison plots and the standalone accelerometer amplitude/SNR plot use the existing connected reference estimates, without gap markers or annotations. Their per-point values and flags are saved in `presentation/`; strict numerical results and detectable intervals remain unchanged. The supplement displays the `_strict` comparison and accelerometer figures. Materials and coil-design plots have editable upper y-axis limits in the main notebook's `PLOT_YMAX` cell. Each standalone fundamental plot repeats the first panel of its corresponding three-panel comparison, using identical values. See [METHODS.md](METHODS.md) for the display policy.

To change these limits, edit and run the `PLOT_YMAX` cell. Its three values are the amplitude (counts), SNR (dB) and responsivity (counts/g) upper limits, from top to bottom. `None` restores automatic limits for that panel. At the end of the notebook, set `REDRAW_ONLY = True` and run the optional redraw cell to create updated figures from a completed tracking run without repeating the raw-data analysis. It keeps the original `RESULTS` and tracking `OUTPUT_PATH` and reports a separate `FIGURE_OUTPUT_PATH`.

Read [METHODS.md](METHODS.md) for estimator definitions and retained quality limitations, and [PUBLICATION_NOTES.md](PUBLICATION_NOTES.md) for manuscript/figure differences. Single-tone repeatability does not establish simultaneous two-tone resolution. Diagnostic reference curves are not strict calibrated responses.

## Setup

Open a terminal in this folder. Use Python 3.11 and create a local environment.

Windows Command Prompt:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

macOS or Linux:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The commands below assume that environment is active. In PowerShell, activation is optional: use `.\.venv\Scripts\python.exe` in place of `python`.

## Verify and run

Verify the packaged sources and raw input hashes before analysis:

```sh
python verify_release.py --check-inputs
```

Run all six main notebooks in separate kernels, with at most two running concurrently:

```sh
python run_notebooks.py --all --jobs 2
```

Or select one exact notebook stem, without `.ipynb`:

```sh
python run_notebooks.py --notebook data11_amplitude_analysis
```

The other five main stems are the names in the table above without their extension. These six notebooks produce 10 child analyses, compared against 162 scientific CSVs and 9 scientific JSON files. Runs create new dated directories under `results/`, preserve raw data and source notebooks, and save executed notebook copies, figures, CSVs, effective settings, input/source hashes and completion status. Previous runs are not overwritten. Scientific comparisons and quality flags must be inspected even when execution succeeds.

To run the optional frequency diagnostics, pass the exact completed `data11_harmonic_tracking` output directory printed by the main frequency notebook:

```sh
python run_notebooks.py --notebook data11_frequency_range_supplement --tracking-run "<exact tracking output path>"
```

This adds two diagnostic analyses, compared against the remaining 48 scientific CSVs and 1 scientific JSON file. The runner does not choose a latest run automatically. In JupyterLab, set `TRACKING_RUN` in the supplement's first code cell instead; the command-line argument is supplied through `IM_TRACKING_RUN` and takes precedence if that environment variable is set.

To redraw an existing completed tracking run with the current figure layout, without reprocessing raw recordings:

```sh
python replot_data11_tracking.py --parent-run "<exact tracking output path>"
```

This creates a separate `data11_tracking_presentation` run with main and strict figures, display-value audits, copied strict summaries and provenance. An optional `--results-root` overrides the output destination. Numerical baseline comparisons require the baseline selection; intentional edits to material lists can change table rows and parameters and therefore fail those comparisons.

**Results are saved on the reader's own computer.** The default is this repository's local `results/` folder. No Google Drive account, author-owned drive or private `config.local.json` is required or included. Local CSV outputs are retained because later plotting cells use them.

For an optional browser interface:

```sh
python -m pip install -r requirements-interactive.txt
python -m jupyterlab
```

Select the environment's Python kernel, open a notebook and run its cells in order. Complete the main Data11 frequency notebook before opening its optional supplement.

## Files and paths

| Location | Contents |
|---|---|
| Repository root | Six main notebooks, one optional supplement, Python modules, runtime/verification tools, requirements, configuration example and documentation |
| `data/Data10/`, `data/Data11/` | 81 raw CSV files: 80 current primary inputs plus the additional IM_54 candidate preview |
| `reference_results/` | Verified baseline outputs: 210 scientific CSVs, 10 scientific JSON files and 67 metadata files |
| `historical_display/` | Nine archived preview files, displayed as historical evidence rather than recalculated current results |
| `release_manifest.json` | Package inventory and integrity information |
| `results/` | Newly generated outputs; excluded from Git |

Defaults in `config.example.json` resolve `data` and `results` relative to this folder. Optional `config.local.json` overrides them locally. Environment variables `IM_DATA_ROOT` and `IM_RESULTS_ROOT` take precedence; the data root must contain `Data10/` and `Data11/`. Results must be outside the read-only data directory. The supplied `.gitignore` excludes local environments, private configuration, caches and generated `results/`.