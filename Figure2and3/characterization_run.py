"""Create exclusive analysis runs with immutable source snapshots and provenance."""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo
from portable_runtime import project_root

BASE = project_root(__file__)
TIMEZONE = "America/New_York"


def _now():
    return datetime.now(ZoneInfo(TIMEZONE))


def _json_value(value):
    """Convert paths/numpy values and represent unavailable numeric results as null."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if hasattr(value, "item"):
        return _json_value(value.item())
    return value


def _inputs_with_paths(value, key=""):
    if isinstance(value, dict):
        return {name: _inputs_with_paths(item, name) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_inputs_with_paths(item, key[:-1] if key.endswith("_paths") else key)
                for item in value]
    if isinstance(value, Path) or (isinstance(value, str) and
            (key in {"path", "file"} or key.endswith("_path"))):
        return str(Path(value).expanduser().resolve())
    return value


def _git(repository, *arguments):
    return subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repository), *arguments], capture_output=True,
        check=False, timeout=15,
    )


def _git_provenance(repository):
    info = {"repository_path": str(repository), "git_available": bool(shutil.which("git")),
            "is_repository": False, "commit": None, "no_commit": None,
            "staged_changes": None, "unstaged_changes": None,
            "untracked_changes": None, "untracked_paths": [], "status_entries": []}
    if not info["git_available"]:
        info["commit_status"] = "unavailable: git executable not found"
        return info
    try:
        top = _git(repository, "rev-parse", "--show-toplevel")
        if top.returncode:
            info["commit_status"] = "unavailable: path is not a Git repository"
            return info
        actual_root = Path(top.stdout.decode().strip()).resolve()
        if actual_root != Path(repository).resolve():
            info["enclosing_repository_path"] = str(actual_root)
            info["commit_status"] = "unavailable: publication folder is not the Git repository root"
            return info
        info["is_repository"] = True
        info["repository_path"] = str(Path(top.stdout.decode().strip()).resolve())
        commit = _git(repository, "rev-parse", "--verify", "HEAD")
        info["commit"] = commit.stdout.decode().strip() if not commit.returncode else None
        info["no_commit"] = commit.returncode != 0
        info["commit_status"] = "recorded" if info["commit"] else "no commit exists"
        status = _git(repository, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        if status.returncode:
            raise RuntimeError(status.stderr.decode(errors="replace").strip())
        records = iter(status.stdout.decode(errors="replace").split("\0"))
        for record in records:
            if not record:
                continue
            code, name = record[:2], record[3:]
            entry = {"status": code, "path": name}
            if "R" in code or "C" in code:
                entry["original_path"] = next(records)
            info["status_entries"].append(entry)
            if code == "??":
                info["untracked_paths"].append(name)
        info["staged_changes"] = any(e["status"][0] not in " ?!" for e in info["status_entries"])
        info["unstaged_changes"] = any(e["status"][1] not in " ?!" for e in info["status_entries"])
        info["untracked_changes"] = bool(info["untracked_paths"])
    except (OSError, subprocess.SubprocessError, RuntimeError, StopIteration) as exc:
        info["git_error"] = f"{type(exc).__name__}: {exc}"
    return info


class AnalysisRun:
    """Context manager; callers supply every executed local script/helper path.

    Inputs should be records containing a ``path`` and ``role``; nested metadata
    is preserved. No input file is changed. Script bytes are snapshotted at entry,
    before the analysis executes, even when the repository is clean.
    """

    def __init__(self, results_root, analysis_name, parameters, inputs, script_paths,
                 repository_path=BASE):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", analysis_name):
            raise ValueError("analysis_name must contain only ASCII letters, digits, _ or -")
        self.results_root = Path(results_root).expanduser().resolve()
        self.analysis_name = analysis_name
        self.parameters = _json_value(parameters)
        self.inputs = _inputs_with_paths(inputs)
        self.script_paths = list(dict.fromkeys(Path(p).resolve() for p in script_paths))
        self.repository_path = Path(repository_path).resolve()
        self.path = None
        self.status = None

    def __enter__(self):
        if self.path is not None:
            raise RuntimeError("An AnalysisRun instance cannot be reused")
        self.results_root.mkdir(parents=True, exist_ok=True)
        started = _now()
        prefix = f"{started:%Y-%m-%d}_{self.analysis_name}_run"
        number = 1
        while True:
            candidate = self.results_root / f"{prefix}{number:03d}"
            try:
                candidate.mkdir()
                self.path = candidate
                break
            except FileExistsError:
                number += 1
        self.status = {"status": "running", "started_at": started.isoformat(),
                       "timezone": TIMEZONE, "output_path": str(self.path)}
        try:
            self.write_json("run_status.json", self.status)
            self.write_json("parameters.json", self.parameters)
            self.write_json("inputs.json", self.inputs)
            self._capture_provenance()
        except BaseException as exc:
            self._fail(exc)
            raise
        return self

    def _capture_provenance(self):
        git = _git_provenance(self.repository_path)
        repository = Path(git["repository_path"])
        scripts = []
        provenance = {"captured_at": _now().isoformat(), "timezone": TIMEZONE,
                      "git": git, "scripts": scripts,
                      "python": {"version": sys.version, "executable": sys.executable},
                      "platform": platform.platform(), "argv": list(sys.argv),
                      "package_versions": {}}
        for package in ("numpy", "scipy", "pandas", "matplotlib", "tzdata"):
            try:
                provenance["package_versions"][package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                provenance["package_versions"][package] = None
        self.write_json("provenance.json", provenance)
        for source in self.script_paths:
            content = source.read_bytes()
            try:
                relative = source.relative_to(repository)
                inside = True
            except ValueError:
                inside = False
                relative = Path("external") / (sha256(str(source).encode()).hexdigest()[:12] + "_" + source.name)
            destination = self._destination(Path("scripts") / relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            record = {"original_path": str(source), "sha256": sha256(content).hexdigest(),
                      "snapshot_path": destination.relative_to(self.path).as_posix(),
                      "inside_repository": inside, "tracked": None,
                      "matches_recorded_commit": False}
            if inside and git["is_repository"]:
                try:
                    record["tracked"] = _git(repository, "ls-files", "--error-unmatch", "--",
                                              relative.as_posix()).returncode == 0
                    if record["tracked"] and git["commit"]:
                        committed = _git(repository, "show", f"{git['commit']}:{relative.as_posix()}")
                        record["matches_recorded_commit"] = committed.returncode == 0 and committed.stdout == content
                except (OSError, subprocess.SubprocessError) as exc:
                    record["git_error"] = f"{type(exc).__name__}: {exc}"
            scripts.append(record)
            self.write_json("provenance.json", provenance)

    def _destination(self, name):
        if self.path is None:
            raise RuntimeError("Enter the AnalysisRun before writing outputs")
        destination = (self.path / name).resolve()
        if not destination.is_relative_to(self.path) or destination == self.path:
            raise ValueError("Output filenames must remain inside the new run directory")
        return destination

    def write_json(self, name, obj):
        return self.write_text(name, json.dumps(_json_value(obj), ensure_ascii=False,
                                               indent=2, allow_nan=False) + "\n")

    def write_text(self, name, content):
        destination = self._destination(name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        return destination

    def update_parameters(self, obj):
        self.parameters.update(_json_value(obj))
        self.write_json("parameters.json", self.parameters)

    def finish(self, summary=None):
        if self.status is None:
            raise RuntimeError("Enter the AnalysisRun before finishing")
        self.status.update(status="completed", finished_at=_now().isoformat())
        if summary is not None:
            self.status["summary"] = _json_value(summary)
        self.write_json("run_status.json", self.status)

    def _fail(self, exc):
        self.status.update(status="failed", finished_at=_now().isoformat(),
                           error=f"{type(exc).__name__}: {exc}",
                           traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        self.write_json("run_status.json", self.status)

    def __exit__(self, exception_type, exception, tb):
        if exception is not None:
            self._fail(exception)
        elif self.status["status"] != "completed":
            self.finish()
        return False
