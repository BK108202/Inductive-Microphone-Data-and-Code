# Publication scope and version notes

This release covers the numerical analyses underlying Figure 2(b-f), Figure 3(b-e), and related diagnostics/supplements. Mechanical photographs/schematics in panels 2(a)/3(a), final manual figure assembly, and Figures 4/5 are outside this collection. Figure correspondence is based on available results, numerical comparisons and visual inspection; it does not prove the exact provenance of the final assembled artwork.

The release follows the latest saved analysis settings. Scientific estimators, data selections and numerical windows are preserved. Source files are not byte-identical to the original project: declared changes adapt paths, environment checks, provenance source collection, notebook names, English documentation/UI, and font selection (Arial when available, otherwise DejaVu Sans). The original snapshots and detailed change audit remain separately preserved. Consult the release validation report for the tests actually completed; original-machine execution alone is not evidence of portability.

## Manuscript and supplied-figure differences

| Item | Current release and remaining publication issue |
|---|---|
| Figure 2(f) repeats | Three recordings per setting, 27 total. The supplied PDF's caption/acquisition description says three, but Methods still says two. Older descriptive fields also say two; actual input lists and result counts govern. |
| Figure 3(b) raw frequencies | 50/100/150 Hz, matching the paper. The earlier 200 Hz discrepancy came from an older saved version and is resolved. |
| Figure 2(b) raw display | Current order is 100/50/600 Hz, all 200 ms. The supplied panel orders 50/100/600 Hz and shows 600 Hz over 50 ms. Reordering/cropping is a presentation difference; exact final assembly is unproven. |
| Figure 2(e) estimator | The supplied panel corresponds to the separate global padded/interpolated peak result: OLS slope 2814.179842 counts/g, R² 0.998937484. The 100±1 Hz target-conditioned result remains distinct. |
| Figure 3(d) amplitude | Current IM_53/55/56 and accel_10/11/12 use central 26 s of 30 s levels. Global padded-bin OLS: slope **488.752271 counts/g**, intercept 81.307472 counts, R² **0.991796865**; reference means 1.117701-1.892366 g. |
| Older amplitude text | The PDF's approximately 508 counts/g, R² 0.9913 and 1.02-1.76 g correspond to an older reference batch: accel_7/8/9, central 16 s of 20 s levels, with the same IM selection at 26 s. Its slope is 507.910627 counts/g. Do not mix the two reference batches or their acquisition descriptions. |
| Figure 3(e) window | Current primary display is 50 s; the 40 s comparison is retained. The 50 s adjacent mean changes, 4.876976 and 5.450970 mHz, match the PDF's 4.88/5.45 mHz text. The supplied panel appears consistent with the historical 40 s export (4.849590/5.499559 mHz adjacent changes). This is an inference, not proof of the final assembled file's source. Choose one window consistently in the paper. |
| Figure 3(c) reference gaps | The connected comparison includes diagnostic reference-gap points. Retain their identification and the strict/marked variants; a continuous line does not certify strict timestamp eligibility. |

The supplied PDF also conflicts on amplitude acquisition duration: its acquisition section describes 30 s for both sensors, while the older Methods description uses a 20 s accelerometer block. The release's actual settings are 30 s blocks and central 26 s for both. Old notebook prose saying central IM 16 s was corrected to the unchanged code value of 26 s.

These differences are disclosed without adjusting estimators or selecting another result merely to match a manuscript number. The paper itself was not edited. The quality flags and separate SNR definitions in [METHODS.md](METHODS.md) remain material to publication, including diagnostic-only amplitude/reference results and the absence of a simultaneous two-tone resolution experiment.

## Distribution and citation

No license has been selected and no LICENSE file is supplied; this release makes no additional reuse-license grant. Repository ownership, author list, citation metadata, permanent archive identifier and final public URLs must come from the authors. Do not substitute a local folder path for a public data/code availability URL. A finalized versioned release or archive should be cited after those publication details are assigned.
