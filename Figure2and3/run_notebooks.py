"""Run publication notebooks in fresh kernels and compare their scientific outputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback
from zoneinfo import ZoneInfo

from portable_runtime import (ROOT, NOTEBOOKS, ALL_NOTEBOOKS, OPTIONAL_NOTEBOOKS,
                              TRACKING_EVIDENCE_FILES, load_config, notebook_path,
                              source_paths, validate_tracking_run)
from verify_release import check_inputs, compare_execution, _sha256


def now():
    return datetime.now(ZoneInfo("America/New_York")).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def worker(name, parent, tracking_run=None):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client.kernelspec import KernelSpecManager

    if name in OPTIONAL_NOTEBOOKS:
        tracking_run = validate_tracking_run(tracking_run)
        os.environ["IM_TRACKING_RUN"] = str(tracking_run)
    elif tracking_run is not None:
        raise ValueError("--tracking-run is only used by data11_frequency_range_supplement")
    folder = Path(parent) / "executions" / name
    folder.mkdir(parents=True, exist_ok=False)
    source = notebook_path(name)
    book = nbformat.read(source, as_version=4)
    original_cells = len(book.cells)
    for cell in book.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    status = {"name": name, "status": "running", "started_at": now(),
              "source_sha256": _sha256(source), "source_notebook": str(source)}
    save(folder / "execution_status.json", status)
    capture = f'''# Execution metadata only; no additional scientific estimator.
import json as _audit_json
import sys as _audit_sys
from pathlib import Path as _AuditPath
from portable_runtime import ROOT as _audit_root
def _audit_value(value):
    if isinstance(value, dict):
        return {{str(k): _audit_value(v) for k, v in value.items()}}
    if isinstance(value, (list, tuple)):
        return [_audit_value(v) for v in value]
    if isinstance(value, _AuditPath):
        return str(value)
    if hasattr(value, 'item'):
        return value.item()
    return value
_audit_selected = {{k: globals()[k] for k in ('CONFIG', 'RAW_FREQUENCIES_HZ',
    'IM_CENTER_WINDOW_S', 'ACCEL_CENTER_WINDOW_S', 'RESOLUTION_DISPLAY_WINDOW') if k in globals()}}
_audit_results = {{k: str(v['output_path']) for k, v in list(globals().items())
    if not k.startswith('_') and isinstance(v, dict) and v.get('output_path')}}
_audit_local_modules = {{p.stem: str(_AuditPath(_audit_sys.modules[p.stem].__file__).resolve())
    for p in _audit_root.glob('*.py') if p.stem in _audit_sys.modules}}
if any(not _AuditPath(p).is_relative_to(_audit_root) for p in _audit_local_modules.values()):
    raise RuntimeError('A project module was imported from outside the release directory')
_AuditPath({str(folder / 'state.json')!r}).write_text(_audit_json.dumps({{
    'selected_state': _audit_value(_audit_selected), 'result_paths': _audit_results,
    'local_module_paths': _audit_local_modules
}}, indent=2, default=str), encoding='utf-8')
'''
    book.cells.append(nbformat.v4.new_code_cell(capture))
    if book.nbformat_minor < 5:
        book.cells[-1].pop("id", None)
    (folder / "runtime_capture_cell.py").write_text(capture, encoding="utf-8")

    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            status.update(current_cell_index=cell_index, last_progress_at=now())
            save(folder / "execution_status.json", status)
            print(f"{name}: cell {cell_index + 1}/{original_cells}", flush=True)

    error = None
    os.environ.setdefault("MPLBACKEND", "Agg")
    with tempfile.TemporaryDirectory(prefix="inductive_microphone_kernel_") as directory:
        spec = Path(directory) / "paper-release"
        spec.mkdir()
        save(spec / "kernel.json", {"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                                  "display_name": "Paper release", "language": "python"})
        client = NotebookClient(book, timeout=3600, startup_timeout=120, kernel_name="paper-release",
            resources={"metadata": {"path": str(ROOT)}}, on_cell_start=progress)
        client.create_kernel_manager()
        client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[directory])
        try:
            client.execute()
        except BaseException:
            error = traceback.format_exc()
        finally:
            nbformat.write(book, folder / (name + ".executed.ipynb"))
            stream = []
            for index, cell in enumerate(book.cells):
                for output in cell.get("outputs", []):
                    if output.output_type == "stream":
                        stream.append(f"CELL {index}\n{output.text}")
                    elif output.output_type == "error":
                        stream.append("\n".join(output.get("traceback", [])))
            (folder / "notebook_stream.txt").write_text("\n".join(stream), encoding="utf-8")
    state_path = folder / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {}
    status.update(status="failed" if error else "completed", finished_at=now(), error=error,
                  result_paths=state.get("result_paths", {}), source_sha256_after=_sha256(source))
    if status["source_sha256"] != status["source_sha256_after"]:
        status.update(status="failed", error="Notebook source changed during execution")
    save(folder / "execution_status.json", status)
    if error:
        print(error, flush=True)
    return 0 if status["status"] == "completed" else 1


def execute(names, jobs, tracking_run=None):
    from characterization_run import AnalysisRun
    import pandas as pd

    if any(name in OPTIONAL_NOTEBOOKS for name in names):
        tracking_run = validate_tracking_run(tracking_run)
    elif tracking_run is not None:
        raise ValueError("--tracking-run is only used by data11_frequency_range_supplement")
    before = check_inputs()
    if before["status"] != "passed":
        raise RuntimeError("Input verification failed: " + "; ".join(before["issues"]))
    inputs = [{"path": path, "role": "read_only_raw_recording", "sha256": value}
              for path, value in before["raw_hashes"].items()]
    config = load_config()
    parameters = {"notebooks": names, "parallel_kernels": jobs, "python": sys.executable,
                  "results_root": config["results_root"], "comparison_rtol": 1e-10, "comparison_atol": 1e-12}
    if tracking_run is not None:
        parameters["tracking_run"] = str(tracking_run)
        inputs += [{"path": str(tracking_run / name), "role": "parent_tracking_evidence",
                    "sha256": _sha256(tracking_run / name)} for name in TRACKING_EVIDENCE_FILES]
    scripts = source_paths(extras=[notebook_path(name) for name in names])
    with AnalysisRun(config["results_root"], "publication_verification", parameters, inputs,
                     scripts, repository_path=ROOT) as run:
        print("EXECUTION_RUN:", run.path, flush=True)
        run.write_json("input_verification_before.json", before)
        run.write_text("methods.txt", "Execute each requested notebook's cells in order in separate fresh kernels. The optional frequency supplement requires an explicitly selected completed tracking parent. Record effective settings, source/input hashes, partial failures, and comparisons with packaged reference results. No raw inputs or source notebooks are written. Scientific methods and limitations are retained in each analysis run.\n")
        (run.path / "logs").mkdir()

        def launch(name):
            command = [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                       "--worker", name, "--parent-run", str(run.path)]
            if name in OPTIONAL_NOTEBOOKS:
                command += ["--tracking-run", str(tracking_run)]
            with (run.path / "logs" / (name + ".log")).open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, check=False)
            path = run.path / "executions" / name / "execution_status.json"
            item = json.loads(path.read_text()) if path.is_file() else {"name": name, "status": "failed", "error": "Worker did not start"}
            item["worker_returncode"] = result.returncode
            if result.returncode:
                item["status"] = "failed"
            return item

        outcomes = []
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for future in as_completed([pool.submit(launch, name) for name in names]):
                item = future.result()
                outcomes.append(item)
                run.write_json("execution_summary.json", outcomes)
                print("NOTEBOOK_FINISHED:", item["name"], item["status"], f"{len(outcomes)}/{len(names)}", flush=True)
        after = check_inputs()
        run.write_json("input_verification_after.json", after)
        unchanged = before["raw_hashes"] == after["raw_hashes"] and before["source_hashes"] == after["source_hashes"]
        integrity = {"raw_inputs_unchanged": before["raw_hashes"] == after["raw_hashes"],
                     "source_files_unchanged": before["source_hashes"] == after["source_hashes"],
                     "final_input_checks_passed": after["status"] == "passed"}
        run.write_json("integrity_check.json", integrity)
        comparison = compare_execution(run.path)
        pd.DataFrame([{k: row.get(k) for k in ("name", "status", "started_at", "finished_at", "worker_returncode")}
                      for row in outcomes]).to_csv(run.path / "execution_summary.csv", index=False)
        pictures = run.path / "figures"
        pictures.mkdir()
        for row in outcomes:
            paths = row.get("result_paths", {})
            primary = paths.get("RESULTS") or paths.get("FFT_RESULTS")
            candidates = sorted(Path(primary).glob("*.png")) if primary else []
            if candidates:
                shutil.copyfile(candidates[0], pictures / (row["name"] + "__" + candidates[0].name))
        failed = [row["name"] for row in outcomes if row["status"] != "completed"]
        passed = not failed and unchanged and after["status"] == "passed" and comparison["status"] == "passed"
        summary = {"status": "completed" if passed else "failed", "notebooks_completed": len(outcomes) - len(failed),
                   "notebooks_failed": failed, "scientific_comparison_passed": comparison["status"] == "passed",
                   "csv_count": comparison["csv_count"], "source_and_raw_integrity_passed": unchanged,
                   "finished_at": now(), "run_path": str(run.path)}
        run.write_json("summary.json", summary)
        if not passed:
            raise RuntimeError("Verification failed; partial results retained at " + str(run.path))
        run.finish(summary)
        print("VERIFICATION_COMPLETED:", run.path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="Run the six main notebooks (default; supplement excluded)")
    selection.add_argument("--notebook", choices=ALL_NOTEBOOKS)
    parser.add_argument("--tracking-run", type=Path,
                        help="Existing completed tracking output; required for the optional frequency supplement")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    parser.add_argument("--worker", choices=ALL_NOTEBOOKS, help=argparse.SUPPRESS)
    parser.add_argument("--parent-run", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.parent_run or not (args.parent_run / "parameters.json").is_file():
            raise ValueError("Worker requires an existing parent run")
        return worker(args.worker, args.parent_run, args.tracking_run)
    selected = args.worker or args.notebook
    if selected in OPTIONAL_NOTEBOOKS and args.tracking_run is None:
        parser.error("data11_frequency_range_supplement requires --tracking-run PATH")
    if selected not in OPTIONAL_NOTEBOOKS and args.tracking_run is not None:
        parser.error("--tracking-run is only used by data11_frequency_range_supplement")
    execute([args.notebook] if args.notebook else list(NOTEBOOKS), args.jobs, args.tracking_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
