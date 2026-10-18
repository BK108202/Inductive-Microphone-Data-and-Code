"""Portable paths, environment checks and source capture for the paper analyses."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = (
    "data10_frequency_range_analysis", "data10_amplitude_analysis", "data10_resolution_analysis",
    "data11_frequency_range_analysis", "data11_amplitude_analysis", "data11_resolution_analysis",
)
OPTIONAL_NOTEBOOKS = ("data11_frequency_range_supplement",)
ALL_NOTEBOOKS = NOTEBOOKS + OPTIONAL_NOTEBOOKS

TRACKING_EVIDENCE_FILES = (
    "run_status.json", "parameters.json", "inputs.json", "input_hashes.json", "provenance.json",
    "trajectory_summary.csv", "harmonic_tracking_records.csv", "harmonic_tracking_summary.csv",
    "acceleration_normalized_responsivity.csv", "detectable_frequency_intervals.csv",
)


def project_root(start=None):
    """Locate this release independently of the caller's working directory."""
    if start is not None:
        candidate = Path(start).resolve()
        if candidate.is_file():
            candidate = candidate.parent
        for folder in (candidate, *candidate.parents):
            if (folder / "release_manifest.json").is_file() and (folder / "portable_runtime.py").is_file():
                return folder
    return ROOT


def _resolve(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def load_config():
    """Read relative defaults, optional local settings, then environment overrides."""
    config = {"data_root": "data", "results_root": "results"}
    for name in ("config.example.json", "config.local.json"):
        path = ROOT / name
        if path.is_file():
            config.update(json.loads(path.read_text(encoding="utf-8-sig")))
    for key, variable in (("data_root", "IM_DATA_ROOT"), ("results_root", "IM_RESULTS_ROOT")):
        config[key] = str(_resolve(os.environ.get(variable) or config[key]))
    inputs, outputs = Path(config["data_root"]), Path(config["results_root"])
    if outputs == inputs or outputs.is_relative_to(inputs):
        raise ValueError("results_root must be outside the read-only data directory.")
    return config


def config_source():
    sources = [str(ROOT / name) for name in ("config.example.json", "config.local.json")
               if (ROOT / name).is_file()]
    sources += ["environment:" + name for name in ("IM_DATA_ROOT", "IM_RESULTS_ROOT") if os.environ.get(name)]
    return "; ".join(sources) or "portable_runtime defaults"


def data_root(dataset):
    if dataset not in {"Data10", "Data11"}:
        raise ValueError("dataset must be Data10 or Data11")
    return Path(load_config()["data_root"]) / dataset


def notebook_path(name):
    stem = Path(name).stem
    if stem not in ALL_NOTEBOOKS:
        raise ValueError(f"Unknown publication notebook: {name}")
    return ROOT / (stem + ".ipynb")


def validate_tracking_run(value):
    """Require an explicitly selected completed parent for optional diagnostics."""
    if value is None or not str(value).strip():
        raise ValueError("Set TRACKING_RUN to the completed main notebook's exact tracking output "
                         "directory, or provide --tracking-run PATH. No run is selected automatically.")
    path = Path(value).expanduser().resolve()
    missing = [name for name in TRACKING_EVIDENCE_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete tracking evidence at {path}: " + ", ".join(missing))
    status = json.loads((path / "run_status.json").read_text(encoding="utf-8-sig"))
    if status.get("status") != "completed":
        raise ValueError(f"Tracking parent must be completed: {path}")
    parameters = json.loads((path / "parameters.json").read_text(encoding="utf-8-sig"))
    config = parameters.get("config", {})
    if config.get("method") != "complex_demodulation_hann_tracking":
        raise ValueError(f"Parent is not a harmonic-tracking run: {path}")
    return path


def source_paths(notebook_path=None, extras=()):
    """Capture published source files, not obsolete notebook-building tools."""
    paths = sorted(ROOT.glob("*.py"))
    paths += [ROOT / name for name in ("config.example.json", "config.local.json", "release_manifest.json")
              if (ROOT / name).is_file()]
    if notebook_path is not None:
        paths.append(Path(notebook_path))
    paths.extend(Path(path) for path in extras)
    paths = list(dict.fromkeys(path.resolve() for path in paths))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing provenance source files: " + ", ".join(missing))
    return paths


@lru_cache(maxsize=1)
def ensure_environment():
    if sys.version_info < (3, 11):
        raise RuntimeError("Use Python 3.11 or later; Python 3.11 is the reference environment.")
    required = ("numpy", "scipy", "pandas", "matplotlib", "IPython", "nbformat", "nbclient", "ipykernel", "tzdata")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Install requirements.txt in this Python environment. Missing: " + ", ".join(missing))


@lru_cache(maxsize=1)
def figure_font():
    """Preserve Arial when available; otherwise use Matplotlib's bundled font."""
    from matplotlib.font_manager import FontProperties, findfont
    try:
        findfont(FontProperties(family="Arial"), fallback_to_default=False)
        return "Arial"
    except ValueError:
        return "DejaVu Sans"
