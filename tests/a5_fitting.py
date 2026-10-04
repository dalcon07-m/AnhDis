#!/usr/bin/env python3
"""a5: extract Gaussian/ORCA energies and fit V(s) in the constant-mass coordinate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import Bounds, LinearConstraint, lsq_linear, minimize

import config
from workflow_utils import (
    extract_energy,
    iter_scan_points,
    normal_termination,
    output_contains_job_tag,
    output_path_for_record,
    read_input_manifest,
    read_scan_manifest,
    resolve_energy_source,
    run_timed_stage,
    selected_modes,
    single_point_software,
    validate_input_record,
)


def derivative_design(x, powers, scale):
    """Columns z^n/n!; fitted coefficients convert to d^nV/ds^n at zero."""
    z = np.asarray(x, float) / scale
    return np.column_stack([z**power / math.factorial(power) for power in powers])


def evaluate_derivative_polynomial(x, derivatives):
    result = np.zeros_like(np.asarray(x, float))
    for power, value in derivatives.items():
        result += value * np.asarray(x, float)**power / math.factorial(power)
    return result


def evaluate_polynomial_slope(x, derivatives):
    """Evaluate dV/ds for coefficients stored as derivatives at s=0."""
    x = np.asarray(x, float)
    result = np.zeros_like(x)
    for power, value in derivatives.items():
        if power >= 1:
            result += value * x ** (power - 1) / math.factorial(power - 1)
    return result


def cubic_derivative_bound(s, relative_energy, max_cubic_factor):
    """Translate the original |a3| limit into the derivative kc=6*a3."""
    if max_cubic_factor is None:
        return np.inf
    if not np.isfinite(max_cubic_factor) or max_cubic_factor <= 0:
        raise ValueError("FIT_MAX_CUBIC_FACTOR must be positive and finite.")
    coordinate_span = float(np.max(s) - np.min(s))
    energy_span = float(np.max(relative_energy) - np.min(relative_energy))
    a2_guess = max(energy_span / (coordinate_span**2 + 1e-12), 1.0e-3)
    max_raw_a3 = max_cubic_factor * a2_guess
    return 6.0 * max_raw_a3


def cubic_factor_for_mode(mode):
    """Return the effective cubic factor and whether it is global or per-mode."""
    overrides = getattr(config, "FIT_MAX_CUBIC_FACTOR_BY_MODE", {})
    if mode in overrides:
        setting = overrides[mode]
        source = "mode override"
    else:
        setting = getattr(config, "FIT_MAX_CUBIC_FACTOR", None)
        source = "global default"
    if setting is None:
        return None, source
    factor = float(setting)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            f"Mode {mode}: the effective cubic-limit factor must be positive, "
            "finite, or None."
        )
    return factor, source


def fit_potential(
    s, relative_energy, degree, stationary, positive_highest_even, weight_scale,
    max_cubic_factor=None, enforce_outward_monotonic=False,
    stability_half_range=None, constraint_grid_points=1001,
    point_weight_multipliers=None,
):
    if not isinstance(degree, (int, np.integer)) or not 2 <= int(degree) <= 6:
        raise ValueError("Polynomial degree must be an integer from 2 through 6.")
    degree = int(degree)
    if not np.isfinite(weight_scale) or weight_scale <= 0:
        raise ValueError("FIT_WEIGHT_SCALE_ANG must be positive and finite.")
    if not np.isfinite(s).all() or not np.isfinite(relative_energy).all():
        raise ValueError("Potential-fit data must be finite.")
    powers = list(range(2 if stationary else 1, degree + 1))
    scale = max(float(np.max(np.abs(s))), 1e-8)
    design = derivative_design(s, powers, scale)
    weights = 1.0 / (1.0 + (np.abs(s) / weight_scale)**2)
    if point_weight_multipliers is not None:
        multipliers = np.asarray(point_weight_multipliers, float)
        if (
            multipliers.shape != weights.shape
            or not np.isfinite(multipliers).all()
            or np.any(multipliers <= 0.0)
        ):
            raise ValueError("Potential-fit weight multipliers must be positive and finite.")
        weights *= multipliers
    weighted_design = np.sqrt(weights)[:, None] * design
    weighted_energy = np.sqrt(weights) * relative_energy
    lower = np.full(len(powers), -np.inf)
    upper = np.full(len(powers), np.inf)
    max_abs_kc = cubic_derivative_bound(s, relative_energy, max_cubic_factor)
    if 3 in powers and np.isfinite(max_abs_kc):
        cubic_index = powers.index(3)
        # result.x contains derivative_n * scale**n because the design matrix
        # is written in the dimensionless coordinate z=s/scale.
        lower[cubic_index] = -max_abs_kc * scale**3
        upper[cubic_index] = max_abs_kc * scale**3
    leading_even_order = degree if degree % 2 == 0 else None
    if positive_highest_even and leading_even_order in powers:
        lower[powers.index(leading_even_order)] = 0.0

    initial = lsq_linear(
        weighted_design, weighted_energy, bounds=(lower, upper), lsmr_tol="auto",
    )
    if not initial.success:
        raise RuntimeError(f"Potential fit failed: {initial.message}")

    coefficients = initial.x
    if enforce_outward_monotonic:
        if stability_half_range is None or not np.isfinite(stability_half_range) or stability_half_range <= 0:
            raise ValueError("A positive stability_half_range is required for shape constraints.")
        constraint_grid_points = int(constraint_grid_points)
        if constraint_grid_points < 101:
            raise ValueError("FIT_SHAPE_CONSTRAINT_GRID_POINTS must be at least 101.")
        if constraint_grid_points % 2 == 0:
            constraint_grid_points += 1

        constraint_s = np.linspace(
            -float(stability_half_range), float(stability_half_range), constraint_grid_points,
        )
        z_constraint = constraint_s / scale
        potential_constraint = np.column_stack([
            z_constraint**power / math.factorial(power) for power in powers
        ])
        nonzero = np.abs(constraint_s) > 1.0e-14
        outward_slope_constraint = np.column_stack([
            np.sign(constraint_s[nonzero]) / scale
            * z_constraint[nonzero] ** (power - 1) / math.factorial(power - 1)
            for power in powers
        ])
        constraint_matrix = np.vstack((potential_constraint, outward_slope_constraint))

        def objective(values):
            residual = weighted_design @ values - weighted_energy
            return 0.5 * float(residual @ residual)

        def objective_gradient(values):
            return weighted_design.T @ (weighted_design @ values - weighted_energy)

        constrained = minimize(
            objective, coefficients, jac=objective_gradient, method="SLSQP",
            bounds=Bounds(lower, upper),
            constraints=[LinearConstraint(constraint_matrix, 0.0, np.inf)],
            options={"ftol": 1.0e-15, "maxiter": 5000},
        )
        if not constrained.success:
            raise RuntimeError(f"Shape-constrained potential fit failed: {constrained.message}")
        coefficients = constrained.x
        violation = float(np.min(constraint_matrix @ coefficients))
        if violation < -1.0e-9:
            raise RuntimeError(
                f"Shape-constrained fit violates an inequality by {violation:.3e}."
            )

    active_mask = np.zeros(len(powers), dtype=int)
    for index, value in enumerate(coefficients):
        tolerance = 1.0e-8 * max(1.0, abs(value))
        if np.isfinite(lower[index]) and abs(value - lower[index]) <= tolerance:
            active_mask[index] = -1
        elif np.isfinite(upper[index]) and abs(value - upper[index]) <= tolerance:
            active_mask[index] = 1

    derivatives = {power: coefficient / scale**power for power, coefficient in zip(powers, coefficients)}
    if stationary:
        derivatives[1] = 0.0
    fitted = evaluate_derivative_polynomial(s, derivatives)
    condition = float(np.linalg.cond(weighted_design))
    return derivatives, fitted, weights, condition, active_mask, max_abs_kc


def fit_model_for_mode(mode):
    """Return the configured potential representation for one mode."""
    overrides = getattr(config, "FIT_MODEL_BY_MODE", {})
    model = str(overrides.get(mode, getattr(config, "FIT_MODEL", "polynomial"))).strip().lower()
    if model == "poly":
        model = "polynomial"
    if model not in {"polynomial", "pchip"}:
        raise ValueError(
            f"Mode {mode}: FIT_MODEL must be 'polynomial' or 'pchip', found {model!r}."
        )
    return model


def point_weight_multipliers_for_mode(mode, q):
    """Build optional mode/Q-window multipliers for polynomial least squares."""
    q = np.asarray(q, float)
    multipliers = np.ones_like(q)
    windows = getattr(config, "FIT_EXTRA_WEIGHT_Q_BY_MODE", {}).get(mode, [])
    for window in windows:
        if not isinstance(window, (tuple, list)) or len(window) != 3:
            raise ValueError(
                f"Mode {mode}: every FIT_EXTRA_WEIGHT_Q_BY_MODE entry must be "
                "(Q_min, Q_max, multiplier)."
            )
        q_min, q_max, factor = map(float, window)
        if (
            not np.isfinite([q_min, q_max, factor]).all()
            or q_min > q_max or factor <= 0.0
        ):
            raise ValueError(f"Mode {mode}: invalid extra-weight window {window!r}.")
        multipliers[(q >= q_min - 1.0e-12) & (q <= q_max + 1.0e-12)] *= factor
    return multipliers


def pchip_tail_curvature_factor_for_mode(mode):
    overrides = getattr(config, "PCHIP_TAIL_CURVATURE_FACTOR_BY_MODE", {})
    factor = float(overrides.get(
        mode, getattr(config, "PCHIP_TAIL_CURVATURE_FACTOR", 1.0),
    ))
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"Mode {mode}: PCHIP tail-curvature factor must be positive.")
    return factor


def build_pchip_model(s, relative_energy, harmonic_k, tail_curvature_factor):
    """Create a shape-preserving interpolant plus confining quadratic tails.

    The endpoint derivative is retained when it points outwards.  An inward
    endpoint derivative is clipped to zero before attaching the quadratic
    tail, which prevents an artificial decrease beyond the calculated scan.
    """
    s = np.asarray(s, float)
    relative_energy = np.asarray(relative_energy, float)
    if (
        s.ndim != 1 or relative_energy.shape != s.shape or s.size < 3
        or not np.isfinite(s).all() or not np.isfinite(relative_energy).all()
        or np.any(np.diff(s) <= 0.0)
    ):
        raise ValueError("PCHIP requires at least three finite, strictly ordered points.")
    if not np.isfinite(harmonic_k) or harmonic_k <= 0.0:
        raise ValueError("PCHIP requires a positive harmonic force constant.")
    if not np.isfinite(tail_curvature_factor) or tail_curvature_factor <= 0.0:
        raise ValueError("PCHIP tail-curvature factor must be positive.")
    interpolator = PchipInterpolator(s, relative_energy, extrapolate=False)
    derivative = interpolator.derivative()
    raw_left_slope = float(derivative(s[0]))
    raw_right_slope = float(derivative(s[-1]))
    left_slope = min(raw_left_slope, 0.0)
    right_slope = max(raw_right_slope, 0.0)
    return dict(
        knots_s_angstrom=s,
        knots_v_hartree=relative_energy,
        left_slope_hartree_per_angstrom=left_slope,
        right_slope_hartree_per_angstrom=right_slope,
        raw_left_slope_hartree_per_angstrom=raw_left_slope,
        raw_right_slope_hartree_per_angstrom=raw_right_slope,
        equilibrium_slope_hartree_per_angstrom=float(derivative(0.0)),
        tail_curvature_hartree_per_angstrom2=float(
            tail_curvature_factor * harmonic_k
        ),
        tail_curvature_factor=float(tail_curvature_factor),
    )


def evaluate_pchip_with_tails(x, model):
    """Evaluate a saved PCHIP model in angstrom/hartree units."""
    scalar = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, float))
    knots = np.asarray(model["knots_s_angstrom"], float)
    values = np.asarray(model["knots_v_hartree"], float)
    result = np.empty_like(x)
    inside = (x >= knots[0]) & (x <= knots[-1])
    result[inside] = PchipInterpolator(knots, values, extrapolate=False)(x[inside])
    left = x < knots[0]
    delta = x[left] - knots[0]
    result[left] = (
        values[0]
        + float(model["left_slope_hartree_per_angstrom"]) * delta
        + 0.5 * float(model["tail_curvature_hartree_per_angstrom2"]) * delta**2
    )
    right = x > knots[-1]
    delta = x[right] - knots[-1]
    result[right] = (
        values[-1]
        + float(model["right_slope_hartree_per_angstrom"]) * delta
        + 0.5 * float(model["tail_curvature_hartree_per_angstrom2"]) * delta**2
    )
    return float(result[0]) if scalar else result


def evaluate_pchip_slope(x, model):
    """Evaluate dV/ds for a PCHIP model with quadratic tails."""
    scalar = np.ndim(x) == 0
    x = np.atleast_1d(np.asarray(x, float))
    knots = np.asarray(model["knots_s_angstrom"], float)
    values = np.asarray(model["knots_v_hartree"], float)
    result = np.empty_like(x)
    inside = (x >= knots[0]) & (x <= knots[-1])
    result[inside] = PchipInterpolator(
        knots, values, extrapolate=False,
    ).derivative()(x[inside])
    left = x < knots[0]
    result[left] = float(model["left_slope_hartree_per_angstrom"]) + float(
        model["tail_curvature_hartree_per_angstrom2"]
    ) * (x[left] - knots[0])
    right = x > knots[-1]
    result[right] = float(model["right_slope_hartree_per_angstrom"]) + float(
        model["tail_curvature_hartree_per_angstrom2"]
    ) * (x[right] - knots[-1])
    return float(result[0]) if scalar else result


def fit_degree_for_mode(mode):
    """Return and validate the configured polynomial degree for one mode."""
    overrides = getattr(config, "FIT_DEGREE_BY_MODE", {})
    degree = int(overrides.get(mode, config.FIT_DEGREE))
    if degree not in (2, 3, 4, 5, 6):
        raise ValueError(f"Mode {mode}: polynomial degree must be between 2 and 6.")
    return degree


def harmonic_force_constant(frequency_cm, mass_amu):
    omega_au = frequency_cm * config.CM_TO_HARTREE
    return omega_au**2 * mass_amu * config.AMU_TO_EMASS / config.BOHR_TO_ANG**2


def select_fit_points(mode, available_points):
    """Select the calculated points used by a5 for one mode.

    FIT_Q_RANGE_ANG_BY_MODE is deliberately a fitting-only option.  The
    Electronic-structure outputs outside the selected interval remain untouched.
    """
    ranges = getattr(config, "FIT_Q_RANGE_ANG_BY_MODE", {})
    if mode not in ranges:
        return list(available_points), None

    limits = ranges[mode]
    if not isinstance(limits, (tuple, list)) or len(limits) != 2:
        raise ValueError(
            f"Mode {mode}: FIT_Q_RANGE_ANG_BY_MODE must contain (Q_min, Q_max)."
        )
    q_min, q_max = map(float, limits)
    if not np.isfinite([q_min, q_max]).all() or q_min >= q_max:
        raise ValueError(
            f"Mode {mode}: invalid fitting interval ({q_min}, {q_max}) angstrom."
        )
    if not (q_min <= 0.0 <= q_max):
        raise ValueError(f"Mode {mode}: the fitting interval must contain Q=0.")

    tolerance = 1.0e-12
    selected = [
        point for point in available_points
        if q_min - tolerance <= point["Q"] <= q_max + tolerance
    ]
    return selected, (q_min, q_max)


def polynomial_label(degree):
    return {2: "quadratic", 3: "cubic", 4: "quartic", 5: "quintic", 6: "sextic"}.get(
        degree, f"degree-{degree} polynomial"
    )


def coefficient_annotation(derivatives, harmonic_k, reference_energy):
    """Text shared by the local-fit and full-domain figures."""
    aliases = {2: "k", 3: "k_c", 4: "k_q", 5: "k_5", 6: "k_6"}
    lines = [rf"$E_0 = {reference_energy:.12f}$ Hartree"]
    for order in sorted(order for order in derivatives if order >= 2):
        label = aliases.get(order, f"k_{order}")
        lines.append(
            rf"${label} = {derivatives[order]:.6e}$ Hartree/$\AA^{order}$"
        )
    lines.append(rf"$k_{{\mathrm{{harm}}}} = {harmonic_k:.6e}$ Hartree/$\AA^2$")
    return "\n".join(lines)


def pchip_annotation(model, harmonic_k, reference_energy):
    """Compact unit-explicit annotation for PCHIP figures."""
    return "\n".join((
        rf"$E_0 = {reference_energy:.12f}$ Hartree",
        "PCHIP interpolation inside scan",
        rf"$k_{{\mathrm{{tail}}}} = {model['tail_curvature_hartree_per_angstrom2']:.6e}$ Hartree/$\AA^2$",
        rf"$k_{{\mathrm{{harm}}}} = {harmonic_k:.6e}$ Hartree/$\AA^2$",
    ))


def main():
    parser = argparse.ArgumentParser(description="Fit one-dimensional path potentials as V(s).")
    parser.add_argument("--scan-root", type=Path, default=config.SCAN_ROOT)
    parser.add_argument("--modes", nargs="+", help="Subset of completed a1 modes")
    parser.add_argument("--allow-partial", action="store_true", help="Fit incomplete scans if enough points exist")
    args = parser.parse_args()

    root = args.scan_root.resolve()
    scan = read_scan_manifest(root)
    coordinate_system = str(scan.get("coordinate_system", "curvilinear")).lower()
    inputs = read_input_manifest(root)
    software = single_point_software(inputs)
    energy_source = resolve_energy_source(config.ENERGY_SOURCE, inputs)
    chosen = selected_modes(args.modes, scan["n_modes"]) if args.modes else inputs["selected_modes"]
    if not set(chosen) <= set(inputs["selected_modes"]):
        raise ValueError("Requested modes are absent from the a2 inputs.")
    input_by_xyz = {point["xyz"]: point for point in inputs["points"]}
    reference = np.load(root / "reference_data.npz", allow_pickle=False)
    frequencies = np.asarray(reference["frequencies_cm"], float)
    if frequencies.shape != (scan["n_modes"],):
        raise ValueError("Frequency array disagrees with scan manifest.")

    full_half_range_bohr = float(getattr(config, "FULL_POTENTIAL_HALF_RANGE_BOHR", 5.0))
    quantum_half_range_bohr = float(getattr(config, "L", full_half_range_bohr))
    if (
        not np.isfinite(full_half_range_bohr) or full_half_range_bohr <= 0
        or not np.isfinite(quantum_half_range_bohr) or quantum_half_range_bohr <= 0
    ):
        raise ValueError("The full-potential and a6 half ranges must be positive and finite.")
    stability_half_range_bohr = max(full_half_range_bohr, quantum_half_range_bohr)
    stability_half_range_ang = stability_half_range_bohr * config.BOHR_TO_ANG
    constraint_grid_points = int(getattr(config, "FIT_SHAPE_CONSTRAINT_GRID_POINTS", 1001))
    positive_highest_even = bool(getattr(
        config, "FIT_REQUIRE_POSITIVE_HIGHEST_EVEN",
        getattr(config, "FIT_REQUIRE_POSITIVE_QUARTIC", True),
    ))
    monotonic_modes = getattr(config, "FIT_MONOTONIC_OUTWARD_BY_MODE", {})

    points_by_mode = {mode: [] for mode in chosen}
    problems = []
    for point in iter_scan_points(root, chosen):
        input_record = input_by_xyz.get(point["xyz"])
        if input_record is None:
            problems.append(f"no a2 input record for {point['xyz']}")
            continue
        try:
            _, input_path = validate_input_record(root, input_record)
        except (FileNotFoundError, ValueError) as exc:
            problems.append(str(exc))
            continue
        log_path = output_path_for_record(root, input_record)
        if not log_path.is_file():
            problems.append(f"missing {log_path.relative_to(root)}")
            continue
        text = log_path.read_text(errors="replace")
        if not normal_termination(text, software):
            problems.append(f"not normally terminated: {log_path.relative_to(root)}")
            continue
        energy, source = extract_energy(text, energy_source, software)
        if energy is None:
            problems.append(f"no {energy_source} energy: {log_path.relative_to(root)}")
            continue
        if not output_contains_job_tag(text, input_record["job_tag"]):
            problems.append(f"wrong/missing a2 identity: {log_path.relative_to(root)}")
            continue
        points_by_mode[int(point["mode"])].append(dict(
            Q=float(point["Q_angstrom"]), s=float(point["s_angstrom"]),
            mu=float(point["mu_amu"]), energy=float(energy), source=source,
            log=str(log_path.relative_to(root)),
        ))
    if problems and not args.allow_partial:
        preview = "\n".join(problems[:12])
        suffix = "" if len(problems) <= 12 else f"\n... and {len(problems)-12} more"
        raise RuntimeError(f"a5 requires complete successful scans. Run a4 first:\n{preview}{suffix}")

    reference_energies = []
    for mode in chosen:
        zeros = [point["energy"] for point in points_by_mode[mode] if abs(point["Q"]) < 1e-12]
        if len(zeros) == 1:
            reference_energies.append((mode, zeros[0]))
    if len(reference_energies) > 1:
        spread = max(value for _, value in reference_energies) - min(value for _, value in reference_energies)
        if spread > config.REFERENCE_ENERGY_TOL_HARTREE:
            raise ValueError(
                "The identical Q=0 geometries give inconsistent energies across modes: "
                f"spread={spread:.3e} hartree. Check methods, inputs, and outputs."
            )

    coefficient_rows, summary_rows, mode_results = [], [], []
    for mode in chosen:
        fit_model = fit_model_for_mode(mode)
        degree = fit_degree_for_mode(mode) if fit_model == "polynomial" else 0
        enforce_outward_monotonic = (
            fit_model == "polynomial" and bool(monotonic_modes.get(mode, False))
        )
        available_points = sorted(points_by_mode[mode], key=lambda item: item["Q"])
        points, requested_fit_range = select_fit_points(mode, available_points)
        minimum_points = (
            degree + (1 if config.FIT_ENFORCE_STATIONARY_EQUILIBRIUM else 2)
            if fit_model == "polynomial" else 3
        )
        if len(points) < minimum_points:
            raise ValueError(
                f"Mode {mode}: only {len(points)} points remain inside the fitting "
                f"interval; need at least {minimum_points}."
            )
        q = np.asarray([point["Q"] for point in points])
        s = np.asarray([point["s"] for point in points])
        mu = np.asarray([point["mu"] for point in points])
        energy = np.asarray([point["energy"] for point in points])
        if np.any(np.diff(q) <= 0) or np.any(np.diff(s) <= 0):
            raise ValueError(f"Mode {mode}: Q and s must both be strictly monotonic.")
        zero = np.flatnonzero(np.isclose(q, 0.0, atol=1e-12))
        if len(zero) != 1:
            raise ValueError(f"Mode {mode}: exactly one Q=0 energy is required.")
        zero = int(zero[0])
        relative = energy - energy[zero]
        mu0 = float(mu[zero])

        path_table = np.loadtxt(root / f"vib{mode}" / "path_metric.dat")
        path_q, path_s, path_mu = path_table[:, 0], path_table[:, 1], path_table[:, 2]
        if np.any(np.diff(path_q) <= 0) or np.any(np.diff(path_s) <= 0):
            raise ValueError(f"Mode {mode}: invalid dense Q-to-s mapping.")
        if not (
            np.allclose(np.interp(q, path_q, path_s), s, atol=1e-10, rtol=0)
            and np.allclose(np.interp(q, path_q, path_mu), mu, atol=1e-8, rtol=0)
        ):
            raise ValueError(f"Mode {mode}: energy-point mapping disagrees with path_metric.dat.")

        path_fit_mask = (path_q >= q[0] - 1.0e-12) & (path_q <= q[-1] + 1.0e-12)
        if not np.any(path_fit_mask):
            raise ValueError(f"Mode {mode}: no path-metric points lie inside the fitting interval.")

        cubic_factor, cubic_factor_source = cubic_factor_for_mode(mode)
        k_harmonic = harmonic_force_constant(frequencies[mode - 1], mu0)
        pchip_model = None
        if fit_model == "polynomial":
            point_multipliers = point_weight_multipliers_for_mode(mode, q)
            derivatives, fitted, weights, condition, active, max_abs_kc = fit_potential(
                s, relative, degree,
                config.FIT_ENFORCE_STATIONARY_EQUILIBRIUM,
                positive_highest_even,
                config.FIT_WEIGHT_SCALE_ANG,
                cubic_factor,
                enforce_outward_monotonic=enforce_outward_monotonic,
                stability_half_range=stability_half_range_ang,
                constraint_grid_points=constraint_grid_points,
                point_weight_multipliers=point_multipliers,
            )
        else:
            pchip_model = build_pchip_model(
                s, relative, k_harmonic,
                pchip_tail_curvature_factor_for_mode(mode),
            )
            derivatives = {}
            fitted = evaluate_pchip_with_tails(s, pchip_model)
            weights = np.ones_like(s)
            condition = np.nan
            active = np.empty(0, dtype=int)
            max_abs_kc = np.nan
        residual_cm = (fitted - relative) * config.HARTREE_TO_CM
        rmse_cm = float(np.sqrt(np.mean(residual_cm**2)))
        max_error_cm = float(np.max(np.abs(residual_cm)))
        dense_s = np.linspace(float(s[0]), float(s[-1]), config.FIT_GRID_POINTS)
        fit_dense = (
            evaluate_derivative_polynomial(dense_s, derivatives)
            if fit_model == "polynomial"
            else evaluate_pchip_with_tails(dense_s, pchip_model)
        )
        harmonic_dense = 0.5 * k_harmonic * dense_s**2
        minimum_index = int(np.argmin(fit_dense))
        fitted_minimum_s = float(dense_s[minimum_index])
        fitted_minimum_cm = float(fit_dense[minimum_index] * config.HARTREE_TO_CM)
        stability_s = np.linspace(
            -stability_half_range_ang, stability_half_range_ang, constraint_grid_points,
        )
        if fit_model == "polynomial":
            stability_fit = evaluate_derivative_polynomial(stability_s, derivatives)
            stability_slope = evaluate_polynomial_slope(stability_s, derivatives)
        else:
            stability_fit = evaluate_pchip_with_tails(stability_s, pchip_model)
            stability_slope = evaluate_pchip_slope(stability_s, pchip_model)
        nonzero_stability = np.abs(stability_s) > 1.0e-12
        minimum_stability_potential_cm = float(np.min(stability_fit) * config.HARTREE_TO_CM)
        minimum_outward_slope = float(np.min(
            np.sign(stability_s[nonzero_stability]) * stability_slope[nonzero_stability]
        ))

        left_outward = relative[:zero + 1][::-1]
        right_outward = relative[zero:]
        outward_steps = np.concatenate((np.diff(left_outward), np.diff(right_outward)))
        minimum_data_outward_step_cm = float(
            np.min(outward_steps) * config.HARTREE_TO_CM
        ) if outward_steps.size else 0.0

        warnings = []
        powers = (
            list(range(2 if config.FIT_ENFORCE_STATIONARY_EQUILIBRIUM else 1, degree + 1))
            if fit_model == "polynomial" else []
        )
        if fit_model == "polynomial" and derivatives.get(2, 0.0) <= 0:
            warnings.append("non-positive fitted curvature at equilibrium")
        if fit_model == "pchip":
            if not np.isclose(
                pchip_model["left_slope_hartree_per_angstrom"],
                pchip_model["raw_left_slope_hartree_per_angstrom"],
            ):
                warnings.append("left PCHIP endpoint slope was clipped before adding its tail")
            if not np.isclose(
                pchip_model["right_slope_hartree_per_angstrom"],
                pchip_model["raw_right_slope_hartree_per_angstrom"],
            ):
                warnings.append("right PCHIP endpoint slope was clipped before adding its tail")
            if abs(pchip_model["equilibrium_slope_hartree_per_angstrom"]) > 1.0e-8:
                warnings.append("PCHIP is not stationary at equilibrium")
        if np.min(relative) < -10.0 / config.HARTREE_TO_CM:
            warnings.append("a scanned point lies more than 10 cm-1 below Q=0")
        leading_even_order = degree if degree % 2 == 0 else None
        if positive_highest_even and leading_even_order in powers:
            if active[powers.index(leading_even_order)] != 0:
                warnings.append(f"positive-k{leading_even_order} bound is active")
        cubic_bound_active = (
            cubic_factor is not None
            and 3 in powers
            and active[powers.index(3)] != 0
        )
        if cubic_bound_active:
            warnings.append("cubic bound is active")
        if fit_model == "polynomial" and condition > 1e6:
            warnings.append("ill-conditioned polynomial design")
        if fitted_minimum_cm < -10:
            warnings.append("fitted minimum displaced below equilibrium")
        if minimum_stability_potential_cm < -1.0e-3:
            warnings.append("potential becomes negative on the full a5/a6 domain")
        if enforce_outward_monotonic and minimum_outward_slope < -1.0e-9:
            warnings.append("outward-monotonic constraint is numerically violated")
        if enforce_outward_monotonic and minimum_data_outward_step_cm < -1.0:
            warnings.append("ab initio points are not outward-monotonic but the fit is constrained")

        energy_table = np.column_stack((q, s, mu, energy, relative, relative * config.HARTREE_TO_CM, fitted, residual_cm))
        np.savetxt(
            root / f"vib{mode}" / "energies_qs.dat", energy_table, fmt="%.14e",
            header="Q_angstrom s_angstrom mu_amu E_total_hartree V_relative_hartree V_relative_cm-1 V_fit_hartree residual_fit_minus_data_cm-1",
        )
        fit_table = np.column_stack((
            dense_s, fit_dense, fit_dense * config.HARTREE_TO_CM,
            harmonic_dense, harmonic_dense * config.HARTREE_TO_CM,
            np.interp(dense_s, path_s, path_q),
        ))
        np.savetxt(
            root / f"vib{mode}" / "fit_curve.dat", fit_table, fmt="%.14e",
            header="s_angstrom V_fit_hartree V_fit_cm-1 V_harmonic_hartree V_harmonic_cm-1 Q_of_s_angstrom",
        )
        np.savez_compressed(
            root / f"vib{mode}" / "potential_for_quantum_step.npz",
            mode=mode, coordinate=np.asarray("s"), fit_model=np.asarray(fit_model),
            mass_amu=mu0,
            frequency_cm=frequencies[mode - 1],
            derivative_orders=np.asarray(sorted(derivatives)),
            derivatives_hartree_per_angstrom_power=np.asarray([derivatives[x] for x in sorted(derivatives)]),
            energy_Q_angstrom=q, energy_s_angstrom=s, energy_hartree=energy,
            relative_energy_hartree=relative,
            path_Q_angstrom=path_q, path_s_angstrom=path_s,
            path_mu_amu=path_mu,
            polynomial_degree=degree,
            cubic_limit_factor=np.nan if cubic_factor is None else cubic_factor,
            max_abs_d3V_ds3_hartree_per_angstrom3=max_abs_kc,
            outward_monotonic_constraint=enforce_outward_monotonic,
            stability_half_range_bohr=stability_half_range_bohr,
            pchip_s_angstrom=(
                pchip_model["knots_s_angstrom"] if pchip_model is not None
                else np.empty(0)
            ),
            pchip_v_hartree=(
                pchip_model["knots_v_hartree"] if pchip_model is not None
                else np.empty(0)
            ),
            pchip_left_slope_hartree_per_angstrom=(
                pchip_model["left_slope_hartree_per_angstrom"]
                if pchip_model is not None else np.nan
            ),
            pchip_right_slope_hartree_per_angstrom=(
                pchip_model["right_slope_hartree_per_angstrom"]
                if pchip_model is not None else np.nan
            ),
            pchip_tail_curvature_hartree_per_angstrom2=(
                pchip_model["tail_curvature_hartree_per_angstrom2"]
                if pchip_model is not None else np.nan
            ),
        )
        coefficient_rows.append(dict(
            mode=mode, fit_model=fit_model, polynomial_degree=degree,
            coordinate="s", mu0_amu=f"{mu0:.12e}",
            frequency_cm=f"{frequencies[mode-1]:.12e}",
            dV_ds=f"{derivatives.get(1, np.nan):.12e}",
            d2V_ds2=f"{derivatives.get(2, np.nan):.12e}",
            d3V_ds3=f"{derivatives.get(3, np.nan):.12e}",
            d4V_ds4=f"{derivatives.get(4, np.nan):.12e}",
            d5V_ds5=f"{derivatives.get(5, np.nan):.12e}",
            d6V_ds6=f"{derivatives.get(6, np.nan):.12e}",
            pchip_left_slope=f"{pchip_model['left_slope_hartree_per_angstrom']:.12e}" if pchip_model else "nan",
            pchip_right_slope=f"{pchip_model['right_slope_hartree_per_angstrom']:.12e}" if pchip_model else "nan",
            pchip_tail_curvature=f"{pchip_model['tail_curvature_hartree_per_angstrom2']:.12e}" if pchip_model else "nan",
            harmonic_d2V_ds2=f"{k_harmonic:.12e}", energy_source=points[0]["source"],
        ))
        summary_rows.append(dict(
            mode=mode, fit_model=fit_model, polynomial_degree=degree,
            available_points=len(available_points), fitted_points=len(points),
            fit_Q_min_angstrom=f"{q[0]:.8f}", fit_Q_max_angstrom=f"{q[-1]:.8f}",
            rmse_cm=f"{rmse_cm:.6f}", max_error_cm=f"{max_error_cm:.6f}",
            condition_number=f"{condition:.6e}", fitted_minimum_s_angstrom=f"{fitted_minimum_s:.8f}",
            fitted_minimum_cm=f"{fitted_minimum_cm:.6f}",
            curvature_ratio_fit_over_harmonic=(
                f"{derivatives.get(2, np.nan)/k_harmonic:.8f}"
                if fit_model == "polynomial" else "nan"
            ),
            max_abs_mass_change_percent=f"{np.max(np.abs(path_mu[path_fit_mask]/mu0-1))*100:.6f}",
            cubic_limit_factor="none" if cubic_factor is None else f"{cubic_factor:.6f}",
            cubic_limit_source=cubic_factor_source,
            max_abs_d3V_ds3=f"{max_abs_kc:.12e}",
            cubic_bound_active="yes" if cubic_bound_active else "no",
            outward_monotonic_constraint="yes" if enforce_outward_monotonic else "no",
            minimum_full_domain_potential_cm=f"{minimum_stability_potential_cm:.6f}",
            minimum_outward_slope_hartree_per_angstrom=f"{minimum_outward_slope:.12e}",
            minimum_ab_initio_outward_step_cm=f"{minimum_data_outward_step_cm:.6f}",
            warnings="; ".join(warnings) if warnings else "none",
        ))
        mode_results.append(dict(
            mode=mode, fit_model=fit_model, q=q, s=s, mu=mu,
            relative=relative, fitted=fitted,
            residual_cm=residual_cm, dense_s=dense_s, fit_dense=fit_dense,
            harmonic_dense=harmonic_dense, path_q=path_q, path_s=path_s, path_mu=path_mu,
            reference_energy=float(energy[zero]), derivatives=derivatives,
            harmonic_k=k_harmonic, degree=degree, pchip_model=pchip_model,
            available_points=len(available_points),
            requested_fit_range=requested_fit_range,
            enforce_outward_monotonic=enforce_outward_monotonic,
        ))

    coeff_path = root / config.FIT_COEFF_FILE
    with coeff_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=coefficient_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(coefficient_rows)
    summary_path = root / config.FIT_SUMMARY_FILE
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    with PdfPages(root / config.FIT_PDF) as pdf:
        for result in mode_results:
            fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.5), constrained_layout=True)
            e0 = result["reference_energy"]
            fit_name = (
                polynomial_label(result["degree"])
                if result["fit_model"] == "polynomial" else "PCHIP"
            )
            axes[0].plot(result["dense_s"], e0 + result["fit_dense"], label=f"{fit_name} fit")
            axes[0].plot(result["dense_s"], e0 + result["harmonic_dense"], "--", label="harmonic")
            axes[0].plot(result["s"], e0 + result["relative"], "o", label="single points")
            axes[0].axhline(e0, color="0.6", linewidth=0.7)
            axes[0].set(
                xlabel=r"constant-mass coordinate $s$ ($\AA$)",
                ylabel="Electronic energy (Hartree)",
                title=f"Mode {result['mode']}",
            )
            axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
            axes[0].legend(loc="upper right")
            axes[0].text(
                0.025, 0.97,
                (
                    coefficient_annotation(
                        result["derivatives"], result["harmonic_k"], e0,
                    )
                    if result["fit_model"] == "polynomial"
                    else pchip_annotation(result["pchip_model"], result["harmonic_k"], e0)
                ),
                transform=axes[0].transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.88),
            )
            axes[1].axhline(0, color="0.5", linewidth=0.7)
            axes[1].plot(result["s"], result["fitted"] - result["relative"], "o-")
            axes[1].set(xlabel=r"$s$ ($\AA$)", ylabel="Fit residual (Hartree)")
            axes[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
            pdf.savefig(fig)
            plt.close(fig)

    full_pdf_name = getattr(config, "FULL_POTENTIAL_PDF", "full_potentials.pdf")
    full_half_range_ang = full_half_range_bohr * config.BOHR_TO_ANG
    with PdfPages(root / full_pdf_name) as pdf:
        for result in mode_results:
            full_s = np.linspace(-full_half_range_ang, full_half_range_ang, config.FIT_GRID_POINTS)
            full_fit = (
                evaluate_derivative_polynomial(full_s, result["derivatives"])
                if result["fit_model"] == "polynomial"
                else evaluate_pchip_with_tails(full_s, result["pchip_model"])
            )
            full_harmonic = 0.5 * result["harmonic_k"] * full_s**2
            if not np.isfinite(full_fit).all() or not np.isfinite(full_harmonic).all():
                raise ValueError(f"Mode {result['mode']}: non-finite full-domain potential.")
            e0 = result["reference_energy"]
            fig, ax = plt.subplots(figsize=(7.2, 8.0), constrained_layout=True)
            ax.axvspan(
                float(result["s"][0]), float(result["s"][-1]),
                color="0.92", label="ab initio scan interval", zorder=0,
            )
            fit_name = (
                polynomial_label(result["degree"])
                if result["fit_model"] == "polynomial" else "PCHIP + tails"
            )
            ax.plot(full_s, e0 + full_fit, label=fit_name)
            ax.plot(full_s, e0 + full_harmonic, "--", label="harmonic")
            ax.plot(result["s"], e0 + result["relative"], "o", label="single points")
            ax.axhline(e0, color="0.55", linewidth=0.7)
            ax.set(
                xlim=(-full_half_range_ang, full_half_range_ang),
                xlabel=r"constant-mass coordinate $s$ ($\AA$)",
                ylabel="Electronic energy (Hartree)",
                title=(f"Mode {result['mode']}: full potential diagnostic "
                       f"($-{full_half_range_bohr:g}$ to $+{full_half_range_bohr:g}$ bohr)"),
            )
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.legend(loc="upper right")
            ax.text(
                0.025, 0.97,
                (
                    coefficient_annotation(
                        result["derivatives"], result["harmonic_k"], e0,
                    )
                    if result["fit_model"] == "polynomial"
                    else pchip_annotation(result["pchip_model"], result["harmonic_k"], e0)
                ),
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(facecolor="white", edgecolor="0.75", alpha=0.88),
            )
            ax.text(
                0.5, 0.015,
                "Electronic-structure data exist only in the shaded interval; outside it PCHIP modes use confining tails.",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="0.3",
            )
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(root / config.METRIC_PDF) as pdf:
        for result in mode_results:
            fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.5), constrained_layout=True)
            axes[0].plot(result["path_q"], result["path_s"], "-")
            axes[0].plot(result["q"], result["s"], "o")
            axes[0].plot(result["path_q"], result["path_q"], "--", label="s = Q")
            axes[0].set(xlabel=r"$Q$ ($\AA$)", ylabel=r"$s(Q)$ ($\AA$)", title=f"Mode {result['mode']}: reparameterization")
            axes[0].legend()
            mu0 = float(np.interp(0.0, result["path_q"], result["path_mu"]))
            axes[1].plot(result["path_q"], result["path_mu"] / mu0)
            axes[1].axhline(1, color="0.5", linestyle="--")
            axes[1].set(xlabel=r"$Q$ ($\AA$)", ylabel=r"$\mu(Q)/\mu(0)$")
            pdf.savefig(fig)
            plt.close(fig)

    metadata = dict(
        stage="curvilinear_potential_fit_in_constant_mass_coordinate",
        coordinate_system=coordinate_system,
        coordinate="s",
        coordinate_definition=scan.get(
            "coordinate_definition", "ds/dQ=sqrt(mu(Q)/mu(0))",
        ),
        constant_hamiltonian_mass="mu(0)",
        potential_definition="V(s(Q))=E(R(Q))-E(R(0))",
        single_point_software=software,
        requested_energy_source=config.ENERGY_SOURCE,
        resolved_energy_source=energy_source,
        selected_modes=chosen,
        default_fit_model=getattr(config, "FIT_MODEL", "polynomial"),
        fit_model_by_mode=getattr(config, "FIT_MODEL_BY_MODE", {}),
        default_fit_degree=config.FIT_DEGREE,
        fit_degree_by_mode=getattr(config, "FIT_DEGREE_BY_MODE", {}),
        stationary_equilibrium=config.FIT_ENFORCE_STATIONARY_EQUILIBRIUM,
        positive_highest_even_coefficient=positive_highest_even,
        outward_monotonic_by_mode=getattr(config, "FIT_MONOTONIC_OUTWARD_BY_MODE", {}),
        shape_constraint_grid_points=constraint_grid_points,
        stability_half_range_bohr=stability_half_range_bohr,
        cubic_limit_convention="|a3| <= factor*a2_guess; kc=d3V/ds3=6*a3",
        default_cubic_limit_factor=config.FIT_MAX_CUBIC_FACTOR,
        cubic_limit_factor_by_mode=getattr(config, "FIT_MAX_CUBIC_FACTOR_BY_MODE", {}),
        fit_Q_range_angstrom_by_mode=getattr(config, "FIT_Q_RANGE_ANG_BY_MODE", {}),
        coefficient_convention="V(s)=sum_n [d^nV/ds^n at 0] s^n/n!",
        coefficient_aliases=dict(
            k="d2V/ds2", kc="kcub=d3V/ds3", kq="d4V/ds4",
            k5="d5V/ds5", k6="d6V/ds6",
        ),
        full_potential_half_range_bohr=full_half_range_bohr,
        full_potential_half_range_angstrom=full_half_range_ang,
        pchip_tail_curvature_factor=getattr(config, "PCHIP_TAIL_CURVATURE_FACTOR", 1.0),
        pchip_tail_curvature_factor_by_mode=getattr(
            config, "PCHIP_TAIL_CURVATURE_FACTOR_BY_MODE", {},
        ),
        full_potential_warning=(
            "Polynomial values outside the ab initio scan are extrapolations; "
            "PCHIP modes use endpoint-matched confining quadratic tails."
        ),
        units=dict(s="angstrom", Q="angstrom", mass="amu", energy="hartree", frequency="cm-1"),
        future_sampling=(
            "Solve in s; sample s; invert saved monotonic s(Q); recover the selected path geometry."
        ),
    )
    (root / "fit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"a5: fitted {len(mode_results)} {coordinate_system} potentials "
        "in the constant-mass coordinate s."
    )
    print(f"Coefficients: {coeff_path}")
    print(
        f"Diagnostics: {summary_path}, {root / config.FIT_PDF}, "
        f"{root / full_pdf_name}, {root / config.METRIC_PDF}"
    )
    for row in summary_rows:
        if row["fit_model"] == "pchip":
            cubic_status = "ignored by PCHIP"
        elif row["cubic_limit_factor"] == "none":
            cubic_status = f"disabled ({row['cubic_limit_source']})"
        else:
            activation = "active" if row["cubic_bound_active"] == "yes" else "not active"
            cubic_status = (
                f"factor {row['cubic_limit_factor']} "
                f"({row['cubic_limit_source']}; {activation})"
            )
        print(
            f"Mode {row['mode']}: model={row['fit_model']}"
            + (
                f", degree={row['polynomial_degree']}"
                if row["fit_model"] == "polynomial" else ""
            )
            + "; "
            f"fitted Q=[{row['fit_Q_min_angstrom']}, "
            f"{row['fit_Q_max_angstrom']}] A using {row['fitted_points']}/"
            f"{row['available_points']} points; RMSE={row['rmse_cm']} cm-1; "
            f"cubic limit={cubic_status}; "
            f"warnings={row['warnings']}"
        )


if __name__ == "__main__":
    run_timed_stage("a5", main, config.EXECUTION_TIMING_FILE)
