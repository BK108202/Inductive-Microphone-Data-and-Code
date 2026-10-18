# Validation record

The 2026-10-18 display revision uses existing connected reference estimates in main comparison plots and the standalone accelerometer amplitude/SNR figure, and saves their strict versions for the supplement. Main figures contain no reference-gap annotations or markers. The numerical estimator and strict CSVs/intervals are unchanged; display values and flags are saved in three `presentation/` CSVs. The verifier compares the original two presentation tables against existing reference-gap baselines and checks the accelerometer sidecar against values derived from archived tracking estimates, without changing historical scientific comparison counts. Materials and coil-design figures now support notebook-editable exact upper limits and optional redraws from saved tables; `None` retains automatic legend spacing. [Connected-display checks](validation/connected_tracking.json) cover unchanged strict values and upper-panel data, requested upper limits, connected versus strict reference values, exact reuse of accelerometer window estimates, standalone/composite fundamental identity, figure routing, notebook syntax and rendered legend clearance using saved tables. The exact checker is [check_connected_tracking.py](validation/check_connected_tracking.py). This is not a new end-to-end raw-data validation.

The material-selection follow-up fixes a plotting `KeyError` when a notebook removes a material that remains named in `comparison_groups`. Tracking plots now ignore removed members and skip empty comparison groups. Five plotting regression cases passed: the published selection, deleted material 12, a fully removed group, a single remaining material, and multiple sweeps with a deleted material. The checks used archived tables in memory, preserved the caller's configuration and numerical tables, and did not rerun raw-data analysis. See [material_subset_plots.json](validation/material_subset_plots.json) and [the regression checker](validation/check_material_subset_plots.py).

The 2026-10-17 presentation revision splits `data11_frequency_range_analysis.ipynb` into a main notebook and `data11_frequency_range_supplement.ipynb`. The main notebook retains raw examples, selected-record review and harmonic tracking; the supplement contains historical previews, reference timestamp-gap views and whole-sweep FFT comparisons. Single-record plots use selected-sweep labels, omit between-record SD shading and produce one peak-curve variant. Multi-record plotting remains supported. Numerical estimators, selections, quality gates and CSV schemas are retained.

The six main notebooks now cover 10 child analyses, 162 scientific CSVs and 9 scientific JSON files. The optional supplement covers the remaining 2 child analyses, 48 scientific CSVs and 1 scientific JSON file, using an explicitly selected completed tracking run. The comparison tool retains support for the exact prior combined-notebook source hash.

This revision is checked separately from the historical full execution below. Its check record is [presentation_revision.json](validation/presentation_revision.json). The split notebooks have not undergone a new end-to-end scientific execution; the earlier 275-PNG identity result applies to the prior combined release. New plot titles and the removal of duplicate single-record figures intentionally change figure outputs. Original reference results and historical validation files remain intact.

Revision checks cover notebook schema and Python syntax, unchanged scientific function syntax trees and selected parameters, moved code cells, reaggregation of three archived tracking tables, and in-memory plotting for single- and multiple-record configurations. Both notebooks' code cells also passed independent-namespace flow checks with analysis calls replaced by archived-result fixtures. These checks process no raw recordings. The exact checker is [check_presentation_revision.py](validation/check_presentation_revision.py); rerunning it requires the original combined-release ZIP via `--baseline-zip`. Package integrity is checked separately with `python verify_release.py --check-inputs`.

## Original combined release: full execution

The publication candidate passed an independent Windows / Python 3.11 execution on 2026-10-16T02:34:54.552859-04:00.
The dependencies in `requirements.txt` were installed into a new environment; `pip check` passed.
The full publication folder was copied to a different directory containing spaces. Six notebooks ran in separate fresh kernels from that copy.

- Six notebooks and all 12 child analyses completed.
- All 210 scientific CSVs, 10 scientific JSON files and 12 effective-parameter records matched their corresponding latest-original-code reference results under the declared comparison rules.
- CSV outcomes: {"numerically_equivalent_path_changes": 37, "byte_identical": 173}.
- Maximum absolute numeric CSV difference: 0.0. Tables with numeric changes: 0.
- Numeric tolerance: relative 1e-10, absolute 1e-12. NaN masks, boolean quality flags and non-path labels were checked separately. Relocated paths retain recording and analysis identity.
- All 81 original raw recordings, all 58 original source files and their preserved copies still matched their hashes. The executed package source and raw inputs also remained unchanged.
- Module-origin checks confirmed that project modules were loaded from the relocated publication folder.
- All 275 PNGs were byte-identical to their corresponding original-code outputs in the tested environment.

The first wrapper check reported one parameter-description difference: the old text said 18 recordings/two per setting, although both executions selected 27 recordings/three per setting. The verifier was corrected to accept only that exact documented text pair, while still strictly comparing the actual input lists and numerical settings. A separate comparison-only run then passed. The original failed wrapper report and all six successfully executed notebooks were preserved; no scientific code or analysis result changed, and the scientific analyses were not rerun for this verifier correction.

The distributed configuration saves new outputs to the reader's own local `results/`. For this verification only, an environment override directed outputs to the authors' project-required results directory; that private configuration is not distributed.

See [validation/summary.json](validation/summary.json), [CSV comparisons](validation/csv_comparisons.csv), [parameter/JSON checks](validation/parameter_and_json_checks.json) and [the recorded environment](validation/environment-windows-py311.txt). The environment listing records the tested Windows installation; install the platform-neutral `requirements.txt` rather than the Windows environment listing on another operating system.

This verifies the declared analyses and selections. It does not resolve manuscript differences, establish additional experimental validity, prove the final assembled artwork's provenance, or establish pixel-identical rendering on other computers. macOS/Linux and the optional JupyterLab browser interface were not tested. Read [PUBLICATION_NOTES.md](PUBLICATION_NOTES.md) and [METHODS.md](METHODS.md).

Scientific execution: `2026-10-16_publication_verification_run001`. Final comparison: `2026-10-16_publication_comparison_run001`. No public repository, tag or license has been created.
