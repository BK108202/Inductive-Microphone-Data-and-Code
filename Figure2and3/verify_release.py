"""Check release input hashes and compare completed scientific results."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from portable_runtime import ROOT, ensure_environment

RTOL = 1e-10
ATOL = 1e-12
EXPECTED_ANALYSES = {
    "data10_frequency_range_analysis": {"frequency_range_analysis_sinewave"},
    "data10_amplitude_analysis": {"amplitude_analysis_sinewave", "amplitude_global_figures"},
    "data10_resolution_analysis": {"resolution_analysis_sinewave"},
    "data11_frequency_range_analysis": {"data11_candidate_review", "data11_harmonic_tracking",
        "data11_raw_examples"},
    "data11_frequency_range_supplement": {"data11_reference_gap_views", "data11_whole_sweep_fft"},
    "data11_amplitude_analysis": {"data11_amplitude", "data11_amplitude_candidate_preview"},
    "data11_resolution_analysis": {"data11_resolution"},
}
# The previously validated main notebook included both optional child analyses.
# Accept that coverage only for its exact recorded source, retaining every numerical check.
LEGACY_DATA11_FREQUENCY_SHA256 = "8bb54c5668f36398ef1cacdc7b07b56c8c6f6972f23512bc7dae05e57f09ba69"


def expected_analyses(notebook):
    expected = EXPECTED_ANALYSES.get(notebook.get("name"), set())
    if (notebook.get("name") == "data11_frequency_range_analysis"
            and notebook.get("source_sha256") == LEGACY_DATA11_FREQUENCY_SHA256):
        return expected | EXPECTED_ANALYSES["data11_frequency_range_supplement"]
    return expected


EXPECTED_COUNTS = {
    "frequency_range_analysis_sinewave": (26, 0),
    "amplitude_analysis_sinewave": (10, 1), "amplitude_global_figures": (5, 1),
    "resolution_analysis_sinewave": (67, 0),
    "data11_candidate_review": (12, 0), "data11_harmonic_tracking": (15, 2),
    "data11_raw_examples": (5, 0), "data11_reference_gap_views": (4, 0),
    "data11_whole_sweep_fft": (44, 1), "data11_amplitude": (10, 2),
    "data11_amplitude_candidate_preview": (4, 1), "data11_resolution": (8, 2),
}
SCIENTIFIC_PARAMETER_FIELDS = {
    "config", "effective_config", "schedule_presets_hz", "complete_schedule",
    "selected_schedule", "selected_files", "conversions", "fixed_quality_defaults",
    "numerical_defaults", "actual_windows", "alternative_windows", "volume_subwindows",
    "raw_examples_actual_windows", "timestamp_audits", "snr_comparisons",
    "global_peak_snr_comparisons", "same_frequency_background_comparisons",
    "same_frequency_noise_window_powers", "same_frequency_signal_projection_check",
    "frequency_accuracy_metrics", "reference_response", "snr_noise_reference_metadata",
    "documented_generator_amplitude_vpp", "inherited_generator_setting_vpp",
    "raw_data_reprocessed",
}

def _sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))

def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)) and not np.isfinite(value):
        return str(value)
    if isinstance(value, np.floating):
        return float(value)
    return value

def _pathlike(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.search(r"[A-Za-z]:[\\/]|(?:^|[\s\[\"'])/(?:[^\s/]+/)", value))

def _path_column(column: str, old: pd.Series, new: pd.Series) -> bool:
    name = column.lower()
    candidates = list(old.dropna().head(10)) + list(new.dropna().head(10))
    return ("path" in name or name in {"file", "files", "source", "source_run", "parent_run", "repository"}) and any(_pathlike(x) for x in candidates)


def _path_identity(value: str):
    """Retain recording/analysis identity while allowing a relocated absolute root."""
    normalized = value.replace("\\", "/")
    if not re.match(r"^(?:[A-Za-z]:/|/)", normalized):
        return None
    parts = tuple(part for part in normalized.split("/") if part)
    for index, part in enumerate(parts):
        if part.lower() in {"data10", "data11"}:
            return ("data", part.lower(), parts[index + 1:])
    for index, part in enumerate(parts):
        match = re.fullmatch(r"\d{4}-\d{2}-\d{2}_(.+)_run\d+(?:_use)?", part)
        if match:
            return ("analysis", match.group(1), parts[index + 1:])
    if parts and parts[-1].endswith((".py", ".ipynb")):
        return ("publication_source",)
    return ("other_path", parts[-1:])


def _relocated_path(old: Any, new: Any) -> bool:
    if not isinstance(old, str) or not isinstance(new, str):
        return False
    left, right = old.split(";"), new.split(";")
    if len(left) != len(right):
        return False
    identities = [(_path_identity(a.strip()), _path_identity(b.strip()))
                  for a, b in zip(left, right)]
    return all(a is not None and a == b for a, b in identities)


def _scientific_parameters(value: dict[str, Any]) -> dict[str, Any]:
    selected = {key: item for key, item in value.items() if key in SCIENTIFIC_PARAMETER_FIELDS}
    if "source_parameters" in value:
        source = value["source_parameters"]
        selected["source_parameters"] = _scientific_parameters(source) if isinstance(source, dict) else source
    return selected

def _samples(old: pd.Series, new: pd.Series, mask: np.ndarray) -> list[dict[str, Any]]:
    positions = np.flatnonzero(mask)[:8]
    return [{"row_zero_based": int(i), "baseline": _plain(old.iloc[i]), "actual": _plain(new.iloc[i])} for i in positions]

def compare_frames(old: pd.DataFrame, new: pd.DataFrame) -> dict[str, Any]:
    """Compare ordered existing tables, distinguishing paths from measurements."""
    report: dict[str, Any] = {
        "baseline_rows": len(old), "actual_rows": len(new),
        "baseline_columns": list(old.columns), "actual_columns": list(new.columns),
        "scientific_values_match": True, "exact_values_match": True,
        "numeric_differences": [], "text_differences": [], "path_changes": [],
        "nan_mask_differences": [], "maximum_absolute_numeric_difference": 0.0,
    }
    if list(old.columns) != list(new.columns) or old.shape != new.shape:
        report.update(scientific_values_match=False, exact_values_match=False, structure_matches=False)
        return report
    report["structure_matches"] = True
    for column in old.columns:
        left, right = old[column], new[column]
        lm, rm = left.isna().to_numpy(), right.isna().to_numpy()
        changed_mask = lm != rm
        if changed_mask.any():
            report["nan_mask_differences"].append({"column": column, "count": int(changed_mask.sum()), "examples": _samples(left, right, changed_mask)})
            report["scientific_values_match"] = report["exact_values_match"] = False
        keep = ~(lm | rm)
        if not keep.any():
            continue
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right) and not (pd.api.types.is_bool_dtype(left) or pd.api.types.is_bool_dtype(right)):
            a = left.loc[keep].to_numpy(dtype=float)
            b = right.loc[keep].to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b)
            absolute = np.abs(a[finite] - b[finite])
            maximum = float(absolute.max()) if len(absolute) else 0.0
            report["maximum_absolute_numeric_difference"] = max(report["maximum_absolute_numeric_difference"], maximum)
            equal = np.isclose(b, a, rtol=RTOL, atol=ATOL, equal_nan=False)
            exact = a == b
            if not exact.all():
                report["exact_values_match"] = False
                report["numeric_differences"].append({"column": column, "changed_values": int((~exact).sum()), "outside_tolerance": int((~equal).sum()), "maximum_absolute_difference": maximum})
            if not equal.all():
                report["scientific_values_match"] = False
        else:
            a = left.loc[keep].astype(object).to_numpy()
            b = right.loc[keep].astype(object).to_numpy()
            equal = np.asarray([bool(x == y) and
                (isinstance(x, (bool, np.bool_)) == isinstance(y, (bool, np.bool_)))
                for x, y in zip(a, b)], dtype=bool)
            if equal.all():
                continue
            mask = np.zeros(len(left), dtype=bool)
            mask[np.flatnonzero(keep)] = ~equal
            report["exact_values_match"] = False
            details = {"column": column, "changed_values": int(mask.sum()), "examples": _samples(left, right, mask)}
            if _path_column(str(column), left, right):
                report["path_changes"].append(details)
                valid = np.asarray([_relocated_path(x, y) for x, y in zip(a, b)], dtype=bool)
                if np.any(~equal & ~valid):
                    details["recording_or_analysis_identity_changed"] = True
                    report["text_differences"].append(details)
                    report["scientific_values_match"] = False
            else:
                report["text_differences"].append(details)
                report["scientific_values_match"] = False
    return report

def _compare_csv(baseline: Path, actual: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"baseline_path": str(baseline), "actual_path": str(actual), "baseline_sha256": _sha256(baseline), "baseline_size_bytes": baseline.stat().st_size}
    if not actual.is_file():
        return dict(result, status="missing_actual", scientific_values_match=False, byte_identical=False)
    result.update(actual_sha256=_sha256(actual), actual_size_bytes=actual.stat().st_size)
    result["byte_identical"] = result["baseline_sha256"] == result["actual_sha256"]
    if result["byte_identical"]:
        return dict(result, status="byte_identical", scientific_values_match=True, path_changes=[])
    try:
        result.update(compare_frames(pd.read_csv(baseline, float_precision="round_trip"), pd.read_csv(actual, float_precision="round_trip")))
    except Exception as exc:
        return dict(result, status="comparison_error", scientific_values_match=False, error=f"{type(exc).__name__}: {exc}")
    if not result["scientific_values_match"]:
        result["status"] = "scientific_difference"
    elif result["path_changes"]:
        result["status"] = "numerically_equivalent_path_changes"
    elif result["exact_values_match"]:
        result["status"] = "values_identical_csv_format_changed"
    else:
        result["status"] = "numerically_equivalent_within_tolerance"
    return result


def _compare_accelerometer_display(actual: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive the new display baseline from hash-verified archived estimates."""
    result: dict[str, Any] = {
        "actual_path": str(actual), "baseline_sources": [], "byte_identical": False,
        "baseline_method": "accelerometer_display from archived tracking records, summary and config",
    }
    try:
        baselines = {}
        for filename in ("harmonic_tracking_records.csv", "harmonic_tracking_summary.csv", "parameters.json"):
            relative = "reference_results/data11_harmonic_tracking/" + filename
            records = [item for item in manifest if item["path"] == relative
                       and item["analysis"] == "data11_harmonic_tracking"]
            baseline = ROOT / relative
            if len(records) != 1 or not baseline.is_file() or _sha256(baseline) != records[0]["sha256"]:
                return dict(result, status="invalid_baseline", scientific_values_match=False,
                            error="Missing, changed or ambiguous archived baseline: " + relative)
            baselines[filename] = baseline
            result["baseline_sources"].append({"path": str(baseline), "sha256": records[0]["sha256"]})
        if not actual.is_file():
            return dict(result, status="missing_actual", scientific_values_match=False)
        from data11_harmonic_tracking import accelerometer_display
        expected = accelerometer_display(
            pd.read_csv(baselines["harmonic_tracking_records.csv"], float_precision="round_trip"),
            _json(baselines["parameters.json"])["config"],
            pd.read_csv(baselines["harmonic_tracking_summary.csv"], float_precision="round_trip"),
        )
        result.update(actual_sha256=_sha256(actual), actual_size_bytes=actual.stat().st_size)
        result.update(compare_frames(expected, pd.read_csv(actual, float_precision="round_trip")))
        return dict(result, status="values_match_derived_baseline" if result["scientific_values_match"]
                    else "scientific_difference")
    except Exception as exc:
        return dict(result, status="comparison_error", scientific_values_match=False,
                    error=f"{type(exc).__name__}: {exc}")


def _json_differences(old: Any, new: Any, key: str = "") -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if isinstance(old, bool) or isinstance(new, bool):
        if type(old) is not type(new) or old != new:
            differences.append({"key": key, "baseline": old, "actual": new, "kind": "boolean_or_type_change"})
    elif isinstance(old, dict) and isinstance(new, dict):
        for name in sorted(old.keys() | new.keys()):
            if name not in old or name not in new:
                differences.append({"key": key + "." + name, "baseline": old.get(name), "actual": new.get(name), "kind": "key_added_or_removed"})
            else:
                differences.extend(_json_differences(old[name], new[name], key + "." + name))
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new)):
            differences.extend(_json_differences(a, b, f"{key}[{i}]"))
    elif old != new:
        is_number = isinstance(old, (float, int)) and isinstance(new, (float, int)) and not isinstance(old, bool) and not isinstance(new, bool)
        close = bool(np.isclose(old, new, rtol=RTOL, atol=ATOL, equal_nan=True)) if is_number else False
        differences.append({"key": key, "baseline": old, "actual": new, "kind": "path_change" if _relocated_path(old, new) else "within_numeric_tolerance" if close else "value_change"})
    return differences


def check_inputs():
    """Read and hash the packaged inputs and source; write nothing."""
    ensure_environment()
    import ast
    import nbformat
    from portable_runtime import ALL_NOTEBOOKS, data_root
    issues = []
    with (ROOT / "metadata/data_manifest.csv").open(encoding="utf-8-sig") as stream:
        inputs = list(csv.DictReader(stream))
    raw_hashes = {}
    for record in inputs:
        path = data_root(record["dataset"]) / record["filename"]
        actual = _sha256(path) if path.is_file() else None
        raw_hashes[str(path)] = actual
        if actual != record["sha256"]:
            issues.append("Missing or changed raw input: " + str(path))
    source_hashes = {}
    manifest = _json(ROOT / "release_manifest.json")
    if not manifest.get("source_files"):
        issues.append("Source manifest is empty")
    for record in manifest.get("source_files", []):
        path = ROOT / record["path"]
        actual = _sha256(path) if path.is_file() else None
        source_hashes[str(path)] = actual
        if actual != record["sha256"]:
            issues.append("Source differs from the packaged release: " + record["path"])
    for name in ALL_NOTEBOOKS:
        path = ROOT / (name + ".ipynb")
        if not path.is_file():
            issues.append("Missing notebook: " + name)
            continue
        book = nbformat.read(path, as_version=4)
        nbformat.validate(book)
        for cell in book.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    for filename in ("historical_display_manifest.json", "reference_manifest.json"):
        for record in _json(ROOT / "metadata" / filename)["files"]:
            path = ROOT / record["path"]
            if not path.is_file() or _sha256(path) != record["sha256"]:
                issues.append("Missing or changed packaged evidence: " + record["path"])
    return {"status": "passed" if not issues else "failed", "issues": issues,
            "raw_file_count": len(raw_hashes), "raw_hashes": raw_hashes,
            "source_hashes": source_hashes}


def compare_execution(run):
    """Compare every included baseline table for the analyses that were run."""
    run = Path(run).resolve()
    summary = _json(run / "execution_summary.json")
    requested = _json(run / "parameters.json")["notebooks"]
    coverage_issues = []
    if Counter(x.get("name") for x in summary) != Counter(requested):
        coverage_issues.append("Executed notebooks do not match the requested notebooks")
    unknown = set(requested) - set(EXPECTED_ANALYSES)
    if unknown:
        coverage_issues.append("Unknown requested notebooks: " + ", ".join(sorted(unknown)))
    expected = set().union(*(EXPECTED_ANALYSES.get(name, set()) for name in requested))
    for notebook in summary:
        if notebook.get("name") in requested:
            expected |= expected_analyses(notebook)
    children = {}
    for notebook in summary:
        observed = set()
        for path in notebook.get("result_paths", {}).values():
            folder = Path(path)
            match = re.fullmatch(r"\d{4}-\d{2}-\d{2}_(.+)_run\d+", folder.name)
            if match:
                name = match.group(1)
                observed.add(name)
                if name in children and children[name] != folder:
                    coverage_issues.append("Multiple output folders for " + name)
                children[name] = folder
            else:
                coverage_issues.append("Unrecognized analysis output path: " + str(folder))
        if observed != expected_analyses(notebook):
            coverage_issues.append("Unexpected or missing analysis outputs for " + str(notebook.get("name")))
    if set(children) != expected:
        coverage_issues.append("Analysis outputs do not match the requested notebook coverage")
    for name, folder in children.items():
        status_path = folder / "run_status.json"
        if not status_path.is_file() or _json(status_path).get("status") != "completed":
            coverage_issues.append("Child analysis is missing or incomplete: " + name)
    files = _json(ROOT / "metadata/reference_manifest.json")["files"]
    comparisons, json_comparisons, config_comparisons = [], [], []
    presentation_comparisons = []
    for item in files:
        if item["analysis"] not in children:
            continue
        baseline = ROOT / item["path"]
        if not baseline.is_file() or _sha256(baseline) != item["sha256"]:
            coverage_issues.append("Packaged baseline is missing or changed: " + item["path"])
            continue
        relative = Path(item["path"]).relative_to(Path("reference_results") / item["analysis"])
        actual = children[item["analysis"]] / relative
        if item["kind"] == "scientific_csv":
            comparison = _compare_csv(baseline, actual)
            comparison.update(analysis=item["analysis"], relative_file=relative.as_posix())
            comparisons.append(comparison)
        elif item["kind"] == "scientific_json":
            changes = _json_differences(_json(baseline), _json(actual)) if actual.is_file() else [
                {"key": "file", "kind": "missing_actual"}]
            json_comparisons.append({"analysis": item["analysis"], "file": relative.as_posix(),
                "differences": changes,
                "scientific_values_match": all(x["kind"] in {"path_change", "within_numeric_tolerance"} for x in changes)})
        elif relative.as_posix() == "parameters.json":
            if not actual.is_file():
                config_comparisons.append({"analysis": item["analysis"],
                    "differences": [{"kind": "missing_actual"}], "scientific_values_match": False})
                continue
            left, right = _json(baseline), _json(actual)
            changes = _json_differences(_scientific_parameters(left), _scientific_parameters(right), "parameters")
            for change in changes:
                if (item["analysis"] == "resolution_analysis_sinewave"
                    and change["key"] == "parameters.config.resolution.file_mapping_source"
                    and change.get("baseline") == "User-supplied nominal mapping; first two files per setting in original configuration order; 18 recordings"
                    and change.get("actual") == "User-supplied nominal mapping; explicit FILES list contains 27 recordings, three per setting"):
                    # Correct an obsolete description; actual file lists remain strictly compared.
                    change["kind"] = "documented_text_correction"
            if not any(key in right for key in ("config", "effective_config")):
                changes.append({"key": "parameters", "kind": "missing_effective_config"})
            config_comparisons.append({"analysis": item["analysis"], "differences": changes,
                "scientific_values_match": all(x["kind"] in {"path_change", "within_numeric_tolerance", "documented_text_correction"} for x in changes)})
    tracking = children.get("data11_harmonic_tracking")
    if tracking is not None and (tracking / "parameters.json").is_file():
        presentation = _json(tracking / "parameters.json").get("plot_presentation", {})
        if presentation.get("main_reference") == "connected_existing_estimates":
            # These display tables have their own archived diagnostic baseline;
            # retain the original strict scientific comparison counts unchanged.
            for filename in ("responsivity_strict_and_diagnostic.csv", "reference_gap_window_estimates.csv"):
                baseline_relative = "reference_results/data11_reference_gap_views/" + filename
                records = [item for item in files if item["path"] == baseline_relative
                           and item["analysis"] == "data11_reference_gap_views"
                           and item["kind"] == "scientific_csv"]
                if len(records) != 1:
                    coverage_issues.append("Missing or ambiguous presentation baseline: " + baseline_relative)
                    continue
                baseline = ROOT / baseline_relative
                if not baseline.is_file() or _sha256(baseline) != records[0]["sha256"]:
                    coverage_issues.append("Packaged presentation baseline is missing or changed: " + baseline_relative)
                    continue
                relative = Path("presentation") / filename
                comparison = _compare_csv(baseline, tracking / relative)
                comparison.update(analysis="data11_harmonic_tracking", relative_file=relative.as_posix(),
                                  baseline_analysis="data11_reference_gap_views")
                presentation_comparisons.append(comparison)
            if len(presentation_comparisons) != 2:
                coverage_issues.append("Connected tracking presentation requires two archived CSV comparisons")
        if presentation.get("main_accelerometer_reference") == "connected_existing_estimates":
            relative = Path("presentation/accelerometer_reference_strict_and_display.csv")
            comparison = _compare_accelerometer_display(tracking / relative, files)
            comparison.update(analysis="data11_harmonic_tracking", relative_file=relative.as_posix(),
                              baseline_analysis="data11_harmonic_tracking")
            presentation_comparisons.append(comparison)
    expected_counts = {}
    for record in files:
        if record["kind"] == "scientific_csv":
            expected_counts[record["analysis"]] = expected_counts.get(record["analysis"], 0) + 1
    missing_baselines = sorted(set(children) - set(expected_counts))
    for name in expected:
        csv_expected, json_expected = EXPECTED_COUNTS[name]
        if sum(x["analysis"] == name for x in comparisons) != csv_expected:
            coverage_issues.append(f"Expected {csv_expected} scientific CSV comparisons for {name}")
        if sum(x["analysis"] == name for x in json_comparisons) != json_expected:
            coverage_issues.append(f"Expected {json_expected} scientific JSON comparisons for {name}")
        if sum(x["analysis"] == name for x in config_comparisons) != 1:
            coverage_issues.append("Expected one effective-parameter comparison for " + name)
    expected_totals = (len(expected), sum(EXPECTED_COUNTS[name][0] for name in expected),
                       sum(EXPECTED_COUNTS[name][1] for name in expected))
    if (len(children), len(comparisons), len(json_comparisons)) != expected_totals:
        coverage_issues.append("Requested notebook coverage requires "
                               f"{expected_totals[0]} child runs, {expected_totals[1]} scientific CSVs "
                               f"and {expected_totals[2]} scientific JSONs")
    status_ok = all(x.get("status") == "completed" for x in summary)
    matched = (bool(comparisons) and status_ok and not missing_baselines and not coverage_issues
               and all(x["scientific_values_match"] for x in
                       comparisons + json_comparisons + config_comparisons + presentation_comparisons))
    report = {"status": "passed" if matched else "failed", "execution_run": str(run),
        "csv_count": len(comparisons), "scientific_json_count": len(json_comparisons),
        "presentation_csv_count": len(presentation_comparisons),
        "parameter_comparison_count": len(config_comparisons),
        "child_runs": {name: str(path) for name, path in children.items()},
        "csv_status_counts": dict(Counter(x["status"] for x in comparisons)),
        "all_compared_scientific_values_match": matched, "missing_baselines": missing_baselines,
        "coverage_issues": coverage_issues,
        "rtol": RTOL, "atol": ATOL, "csv_comparisons": comparisons,
        "scientific_json_comparisons": json_comparisons, "config_comparisons": config_comparisons,
        "presentation_csv_comparisons": presentation_comparisons,
        "interpretation": "Numeric values, NaN masks, booleans and non-path labels are checked. Absolute path changes are recorded separately. PNG/SVG pixel identity is not required; scientific quality flags remain unchanged."}
    (run / "comparison.json").write_text(json.dumps(_plain(report), indent=2) + "\n", encoding="utf-8")
    flat = [{"analysis": x["analysis"], "file": x["relative_file"], "status": x["status"],
             "scientific_values_match": x["scientific_values_match"],
             "byte_identical": x["byte_identical"]} for x in comparisons]
    pd.DataFrame(flat).to_csv(run / "comparison.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--run", type=Path, help="Compare a completed run_notebooks.py execution directory")
    args = parser.parse_args()
    if args.run:
        result = compare_execution(args.run)
        print(json.dumps({key: result[key] for key in ("status", "csv_count", "scientific_json_count", "csv_status_counts")}, indent=2))
    else:
        result = check_inputs()
        print(json.dumps({key: result[key] for key in ("status", "raw_file_count", "issues")}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
