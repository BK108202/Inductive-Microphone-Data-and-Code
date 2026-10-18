"""Single-record sinusoidal frequency fit on the supplied native timestamps.

The model is y = c + a*cos(2*pi*f*(t-t_center)) + b*sin(...). For each
frequency the three linear coefficients are solved by unweighted least squares;
only frequency is optimized. No resampling, detrending, filter, taper, nominal
frequency label, or timebase correction is applied here. The caller supplies a
search center obtained from the observed FFT peak and an explicit search width.

Residual RMS describes mismatch to this single stationary sinusoid, including
noise, harmonics, modulation, and drift. Explained variance is not SNR, and no
absolute-frequency calibration or uncertainty interval is inferred.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def fit_sine_frequency(
    t,
    y,
    initial_hz,
    halfwidth_hz,
    *,
    coarse_points=17,
    frequency_tolerance_hz=1e-9,
):
    """Fit one real sinusoid using every supplied sample, without modifying it.

    ``t`` must contain finite, strictly increasing timestamps in seconds and
    ``y`` finite real amplitudes. All results are JSON-serializable Python
    values. Only ``status == 'ok'`` provides a usable ``frequency_hz``. Invalid
    input, failed optimization, or a minimum at the search boundary produces
    ``frequency_hz=None`` and explanatory flags. Boundary candidates and their
    diagnostics are retained separately for review.

    The requested coarse grid is automatically refined when necessary to have
    frequency spacing <= 1/(4*timestamp_span), before locally optimizing around
    its lowest residual. This limits missed narrow minima, but is not proof of
    an unambiguous physical frequency for arbitrary or aliased input. The final
    optimization uses frequency offsets to avoid an absolute-frequency relative
    stopping tolerance becoming larger than the requested frequency tolerance.
    """
    result = {
        "status": "unavailable",
        "flags": [],
        "frequency_hz": None,
        "candidate_frequency_hz": None,
        "amplitude_peak": None,
        "offset": None,
        "residual_rms": None,
        "fitted_ac_rms": None,
        "explained_variance_fraction": None,
        "sample_count": 0,
        "t_center_s": None,
        "timestamp_span_s": None,
        "initial_hz": None,
        "halfwidth_hz": None,
        "search_lower_hz": None,
        "search_upper_hz": None,
        "coarse_points_requested": None,
        "coarse_points_effective": None,
        "frequency_tolerance_hz": None,
        "optimizer_converged": False,
        "optimizer_iterations": 0,
        "function_evaluations": 0,
        "optimizer_message": None,
    }

    def unavailable(flag):
        result["flags"].append(flag)
        return result

    try:
        if np.iscomplexobj(t) or np.iscomplexobj(y):
            return unavailable("complex_input")
        times = np.asarray(t, dtype=float)
        values = np.asarray(y, dtype=float)
        initial_hz = float(initial_hz)
        halfwidth_hz = float(halfwidth_hz)
        frequency_tolerance_hz = float(frequency_tolerance_hz)
        points_float = float(coarse_points)
    except (TypeError, ValueError, OverflowError):
        return unavailable("invalid_input_type")
    if times.ndim != 1 or values.ndim != 1 or times.size != values.size:
        return unavailable("invalid_array_shape")
    result["sample_count"] = int(times.size)
    if times.size < 8:
        return unavailable("insufficient_samples")
    if not (np.all(np.isfinite(times)) and np.all(np.isfinite(values))):
        return unavailable("nonfinite_input")
    if not np.all(np.diff(times) > 0):
        return unavailable("timestamps_not_strictly_increasing")
    # A reduction mean can differ by roundoff from a repeated nonzero float.
    # Check the original samples so that a constant cannot become a tiny sine.
    if np.all(values == values[0]):
        return unavailable("constant_signal")
    if not (
        math.isfinite(initial_hz)
        and initial_hz > 0
        and math.isfinite(halfwidth_hz)
        and halfwidth_hz > 0
        and math.isfinite(frequency_tolerance_hz)
        and 0 < frequency_tolerance_hz < halfwidth_hz
        and math.isfinite(points_float)
        and points_float.is_integer()
        and points_float >= 5
    ):
        return unavailable("invalid_search_parameters")

    # Subtract the first time before taking a mean, preserving small intervals
    # when timestamps have a large origin. This creates new arrays only.
    relative_times = times - times[0]
    mean_relative_time = float(np.mean(relative_times))
    centered_times = relative_times - mean_relative_time
    span = float(relative_times[-1])
    lower = max(0.0, initial_hz - halfwidth_hz)
    upper = initial_hz + halfwidth_hz
    if not (math.isfinite(span) and span > 0 and math.isfinite(upper)):
        return unavailable("invalid_search_extent")
    result.update(
        t_center_s=float(times[0] + mean_relative_time),
        timestamp_span_s=span,
        initial_hz=initial_hz,
        halfwidth_hz=halfwidth_hz,
        search_lower_hz=lower,
        search_upper_hz=upper,
        coarse_points_requested=int(points_float),
        frequency_tolerance_hz=frequency_tolerance_hz,
    )
    grid_intervals = (upper - lower) * 4.0 * span
    if not math.isfinite(grid_intervals) or grid_intervals > 10000:
        return unavailable("search_grid_too_large")
    point_count = max(int(points_float), int(math.ceil(grid_intervals)) + 1)
    if point_count > 10001:
        return unavailable("search_grid_too_large")
    result["coarse_points_effective"] = point_count

    value_mean = float(np.mean(values))
    centered_values = values - value_mean
    value_scale = float(np.max(np.abs(centered_values)))
    if not math.isfinite(value_scale):
        return unavailable("amplitude_numeric_range")
    if value_scale == 0:
        return unavailable("constant_signal")
    scaled_values = centered_values / value_scale
    variance_scaled = float(np.mean(scaled_values * scaled_values))
    angular_times = (2.0 * np.pi) * centered_times

    def evaluate(delta_hz, *, full=False):
        result["function_evaluations"] += 1
        frequency = initial_hz + float(delta_hz)
        if frequency <= 0 or not math.isfinite(frequency):
            return (math.inf, None) if full else math.inf
        phase = frequency * angular_times
        cos_values = np.cos(phase)
        sin_values = np.sin(phase)
        cos_mean = float(np.mean(cos_values))
        sin_mean = float(np.mean(sin_values))
        cos_values -= cos_mean
        sin_values -= sin_mean
        # Solve the two-column normal equations after eliminating the intercept.
        cc = float(np.mean(cos_values * cos_values))
        ss = float(np.mean(sin_values * sin_values))
        cs = float(np.mean(cos_values * sin_values))
        cy = float(np.mean(cos_values * scaled_values))
        sy = float(np.mean(sin_values * scaled_values))
        determinant = cc * ss - cs * cs
        if determinant <= np.finfo(float).eps * 100 * max(cc * ss, 1e-300):
            return (math.inf, None) if full else math.inf
        a = (ss * cy - cs * sy) / determinant
        b = (cc * sy - cs * cy) / determinant
        fitted_ac = a * cos_values + b * sin_values
        residual = scaled_values - fitted_ac
        # Compute residuals directly: total-minus-explained energy loses useful
        # precision close to the optimum for a nearly exact synthetic sinusoid.
        mse = float(np.mean(residual * residual))
        if not full:
            return mse
        details = {
            "amplitude_peak": float(value_scale * math.hypot(a, b)),
            "offset": float(value_mean - value_scale * (a * cos_mean + b * sin_mean)),
            "residual_rms": float(value_scale * math.sqrt(mse)),
            "fitted_ac_rms": float(value_scale * math.sqrt(float(np.mean(fitted_ac * fitted_ac)))),
            "explained_variance_fraction": float(1.0 - mse / variance_scaled),
        }
        return mse, details

    grid = np.linspace(lower - initial_hz, upper - initial_hz, point_count)
    scores = np.array([evaluate(delta) for delta in grid])
    if not np.any(np.isfinite(scores)):
        return unavailable("no_identifiable_sinusoid_in_search")
    best_index = int(np.argmin(scores))
    local_lower = float(grid[max(0, best_index - 1)])
    local_upper = float(grid[min(point_count - 1, best_index + 1)])
    optimized = minimize_scalar(
        evaluate,
        bounds=(local_lower, local_upper),
        method="bounded",
        options={"xatol": frequency_tolerance_hz, "maxiter": 100},
    )
    result.update(
        optimizer_converged=bool(optimized.success),
        optimizer_iterations=int(optimized.nit),
        optimizer_message=str(optimized.message),
    )
    if not optimized.success or not math.isfinite(float(optimized.fun)):
        return unavailable("optimizer_failed")
    # Include the observed grid minimum, so a true optimum on a search endpoint
    # cannot be mistaken for an interior point returned by bounded optimization.
    best_delta = float(optimized.x)
    if scores[best_index] < float(optimized.fun):
        best_delta = float(grid[best_index])
    candidate = initial_hz + best_delta
    mse, details = evaluate(best_delta, full=True)
    if details is None or not all(math.isfinite(v) for v in details.values()):
        return unavailable("nonfinite_fit_result")
    result.update(details)
    result["candidate_frequency_hz"] = float(candidate)
    boundary_tolerance = max(10 * frequency_tolerance_hz, (upper - lower) * 1e-6)
    if candidate - lower <= boundary_tolerance or upper - candidate <= boundary_tolerance:
        result["status"] = "search_boundary"
        result["flags"].append("minimum_at_search_boundary")
        return result
    result["status"] = "ok"
    result["frequency_hz"] = float(candidate)
    return result
