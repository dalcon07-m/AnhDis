#!/usr/bin/env python3
"""a7: sample independent position/momentum marginals and build ensembles.

For the anharmonic ensemble, a uniform random number is transformed through
the numerical inverse CDF into the constant-mass coordinate s (bohr), and the
monotonic a1 relation s(Q) is inverted.  For the harmonic ensemble, s and p_s
are sampled directly from their exact, unbounded thermal Gaussian marginals;
Q=s after converting bohr to angstrom.  Thus harmonic ensemble sampling does
not use the a1/a5/a6 support limits or a discretized CDF.

A curvilinear sample is reconstructed from the simultaneous independent-
internal target

    xi_target = sum_i Q_i l_i,

where l_i is the equilibrium normal-mode tangent in the fixed a1 delocalized
internal-coordinate basis. A single iterative internal-to-Cartesian
back-transformation, with continuation from equilibrium, produces the final
geometry.  For a Cartesian a1 scan, the same sampled modal coordinates are
instead mapped analytically as R=R0+sum_i Q_i*d_i.  No fitted-coordinate shift
is required in either representation. The numerical harmonic calculation in
a6 remains an independent energetic/grid check and is not used to sample the
harmonic ensemble.

--coordinates overrides only the output geometry representation. The default
scan inherits a1. Source densities, s(Q), wavefunctions and harmonic Q=s are
unchanged; a different reconstruction is not a recalculation of the source PES.

The statistical model remains the product of independent one-dimensional
mode marginals,

    F_i(s_i, p_i; T) proportional to rho_i(s_i,T) rho_i(p_i,T).

Distinct uniform variates are used for position and momentum.  This is the
positive separable AnhDis approximation defined in the manuscript, not an
anharmonic Wigner quasiprobability.  The nonlinear back-transformation changes
only the molecular geometry/vector reconstruction; it does not add energetic
coupling between modes.

The first member of every generated ensemble is the optimized equilibrium
geometry.  Therefore, NSAMPLES is the total ensemble size: one equilibrium
geometry followed by NSAMPLES-1 statistically sampled geometries.
Its momentum is still independently sampled, because fixing the first member's
position at equilibrium does not impose zero velocity.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import ndtr, ndtri
from scipy.stats import kstest

import config
from internal_coordinates import (
    Internal,
    continue_backtransform,
    internal_difference,
    internal_values,
    mass_metric_step,
    remove_rigid_velocity,
    vibrational_dof,
    wilson_b,
)
from workflow_utils import (
    ensure_new_directory,
    read_scan_manifest,
    run_timed_stage,
    selected_modes,
    sha256_file,
    write_xyz,
)


ANHARMONIC = "thermal_anharmonic"
HARMONIC = "thermal_harmonic"
HARMONIC_CARTESIAN = "thermal_harmonic_cartesian"
DENSITY_KEYS = {
    ANHARMONIC: "p_thermal_anh_per_bohr",
    HARMONIC: "p_thermal_harm_per_bohr",
}
OUTPUT_LAYOUT = {
    HARMONIC_CARTESIAN: ("ensemble_har_cart", "histograms_har_cart", "all_geometries_har_cart.xyz"),
    ANHARMONIC: ("ensemble_anh_therm", "histograms_anh", "all_geometries_anh.xyz"),
    HARMONIC: ("ensemble_har_therm", "histograms_har", "all_geometries_har.xyz"),
}
VECTOR_OUTPUT_NAMES = {
    HARMONIC_CARTESIAN: ("all_velocities_har_cart.xyz", "all_momenta_har_cart.xyz"),
    ANHARMONIC: ("all_velocities_anh.xyz", "all_momenta_anh.xyz"),
    HARMONIC: ("all_velocities_har.xyz", "all_momenta_har.xyz"),
}


_PARALLEL_REFERENCE = None
_PARALLEL_MODE_NUMBERS = None


def _initialize_reconstruction_worker(reference, mode_numbers):
    """Install read-only reconstruction data once in each worker process."""
    global _PARALLEL_REFERENCE, _PARALLEL_MODE_NUMBERS
    _PARALLEL_REFERENCE = reference
    _PARALLEL_MODE_NUMBERS = mode_numbers


def _reconstruct_candidate_in_worker(task):
    """Return one indexed reconstruction without allowing expected failures to abort the pool."""
    candidate_index, q_ang = task
    try:
        geometry, info = reconstruct_geometry(
            q_ang, _PARALLEL_MODE_NUMBERS, _PARALLEL_REFERENCE,
        )
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        return candidate_index, None, None, reason
    return candidate_index, geometry, info, None


def harmonic_thermal_sigmas(frequency_cm, mass_amu, temperature_K):
    """Return exact thermal harmonic widths in s (bohr) and p_s (a.u.)."""
    frequency_cm = float(frequency_cm)
    mass_emass = float(mass_amu) * config.AMU_TO_EMASS
    temperature_K = float(temperature_K)
    if frequency_cm <= 0.0 or mass_emass <= 0.0 or temperature_K < 0.0:
        raise ValueError("Harmonic frequency and mass must be positive and T cannot be negative.")
    omega_au = frequency_cm * config.CM_TO_HARTREE
    coth = (
        1.0
        if temperature_K == 0.0
        else 1.0 / math.tanh(omega_au / (2.0 * config.KB_AU * temperature_K))
    )
    sigma_s_bohr = math.sqrt(coth / (2.0 * mass_emass * omega_au))
    sigma_p_au = math.sqrt(0.5 * mass_emass * omega_au * coth)
    return sigma_s_bohr, sigma_p_au


def normal_density(grid, sigma):
    """Centered normalized Gaussian density evaluated on any grid."""
    grid = np.asarray(grid, float)
    sigma = float(sigma)
    return np.exp(-0.5 * (grid / sigma)**2) / (math.sqrt(2.0 * math.pi) * sigma)


def normal_quantile(uniform, sigma):
    """Centered Gaussian inverse CDF, protected against exact endpoints."""
    probability = float(np.clip(uniform, np.finfo(float).tiny, 1.0 - np.finfo(float).eps))
    return float(sigma * ndtri(probability))


def pchip_potential_on_angstrom_grid(model, s_angstrom):
    """Evaluate the a5 PCHIP potential and its confining quadratic tails."""
    scalar = np.ndim(s_angstrom) == 0
    x = np.atleast_1d(np.asarray(s_angstrom, float))
    knots = np.asarray(model["knots_s_angstrom"], float)
    values = np.asarray(model["knots_v_hartree"], float)
    if (
        knots.ndim != 1 or values.shape != knots.shape or knots.size < 3
        or not np.isfinite(knots).all() or not np.isfinite(values).all()
        or np.any(np.diff(knots) <= 0.0)
    ):
        raise ValueError("Invalid PCHIP knots in the a5 output.")
    left_slope = float(model["left_slope_hartree_per_angstrom"])
    right_slope = float(model["right_slope_hartree_per_angstrom"])
    tail_curvature = float(model["tail_curvature_hartree_per_angstrom2"])
    if (
        not np.isfinite([left_slope, right_slope, tail_curvature]).all()
        or left_slope > 1.0e-14 or right_slope < -1.0e-14
        or tail_curvature <= 0.0
    ):
        raise ValueError("Invalid PCHIP confining-tail parameters.")
    potential = np.empty_like(x)
    inside = (x >= knots[0]) & (x <= knots[-1])
    potential[inside] = PchipInterpolator(
        knots, values, extrapolate=False,
    )(x[inside])
    left = x < knots[0]
    delta = x[left] - knots[0]
    potential[left] = values[0] + left_slope * delta + 0.5 * tail_curvature * delta**2
    right = x > knots[-1]
    delta = x[right] - knots[-1]
    potential[right] = values[-1] + right_slope * delta + 0.5 * tail_curvature * delta**2
    return float(potential[0]) if scalar else potential


def cumulative_trapezoid_cdf(grid, density):
    """Normalize a density and return its piecewise-linear trapezoidal CDF."""
    grid = np.asarray(grid, float)
    density = np.asarray(density, float)
    if grid.ndim != 1 or density.shape != grid.shape or grid.size < 3:
        raise ValueError("A probability grid and density must be matching 1D arrays.")
    if not np.isfinite(grid).all() or not np.isfinite(density).all():
        raise ValueError("Probability data contain non-finite values.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("The probability grid must be strictly increasing.")
    scale = max(1.0, float(np.max(np.abs(density))))
    if float(np.min(density)) < -1.0e-12 * scale:
        raise ValueError("A probability density contains significant negative values.")
    density = np.maximum(density, 0.0)
    norm = float(np.trapezoid(density, grid))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("A probability density has zero or invalid normalization.")
    density = density / norm
    increments = 0.5 * (density[:-1] + density[1:]) * np.diff(grid)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf /= cdf[-1]
    cdf[0], cdf[-1] = 0.0, 1.0
    return density, cdf


def evaluate_piecewise_linear_cdf(grid, density, cdf, values):
    """Evaluate the trapezoidal CDF with exact integration inside each cell."""
    grid = np.asarray(grid, float)
    density = np.asarray(density, float)
    cdf = np.asarray(cdf, float)
    values_array = np.asarray(values, float)
    flat = values_array.ravel()
    result = np.empty_like(flat)
    below = flat <= grid[0]
    above = flat >= grid[-1]
    result[below] = 0.0
    result[above] = 1.0
    middle = ~(below | above)
    if np.any(middle):
        x = flat[middle]
        index = np.searchsorted(grid, x, side="right") - 1
        width = grid[index + 1] - grid[index]
        offset = x - grid[index]
        slope = (density[index + 1] - density[index]) / width
        result[middle] = cdf[index] + density[index] * offset + 0.5 * slope * offset**2
    result = np.clip(result, 0.0, 1.0)
    return float(result[0]) if values_array.ndim == 0 else result.reshape(values_array.shape)


def invert_piecewise_linear_cdf(grid, density, cdf, probabilities):
    """Invert a piecewise-linear density by solving the cell area exactly."""
    grid = np.asarray(grid, float)
    density = np.asarray(density, float)
    cdf = np.asarray(cdf, float)
    probabilities_array = np.asarray(probabilities, float)
    if np.any((probabilities_array < 0) | (probabilities_array > 1)):
        raise ValueError("CDF probabilities must lie between zero and one.")
    flat = probabilities_array.ravel()
    result = np.empty_like(flat)
    at_start = flat <= 0.0
    at_end = flat >= 1.0
    result[at_start] = grid[0]
    result[at_end] = grid[-1]
    middle = ~(at_start | at_end)
    for output_index, probability in zip(np.flatnonzero(middle), flat[middle]):
        cell = int(np.searchsorted(cdf, probability, side="right") - 1)
        cell = min(max(cell, 0), grid.size - 2)
        width = float(grid[cell + 1] - grid[cell])
        p0 = float(density[cell])
        slope = float((density[cell + 1] - density[cell]) / width)
        area = float(probability - cdf[cell])
        if abs(slope) < 1.0e-14 * max(1.0, p0 / width):
            if p0 <= 0:
                offset = 0.0
            else:
                offset = area / p0
        else:
            discriminant = max(0.0, p0 * p0 + 2.0 * slope * area)
            denominator = p0 + math.sqrt(discriminant)
            if abs(denominator) > 1.0e-300:
                offset = 2.0 * area / denominator
            else:
                offset = (-p0 + math.sqrt(discriminant)) / slope
        result[output_index] = grid[cell] + np.clip(offset, 0.0, width)
    return float(result[0]) if probabilities_array.ndim == 0 else result.reshape(probabilities_array.shape)


def internal_definitions_from_json(path):
    records = json.loads(Path(path).read_text())
    definitions = []
    for record in records:
        kind = record.get("kind")
        atoms = tuple(int(value) - 1 for value in record.get("atoms_1based", []))
        axis_reference = record.get("axis_reference")
        expected = (
            3 if kind == "L" and axis_reference in {-1, -2}
            else {"R": 2, "A": 3, "D": 4, "L": 4}.get(kind)
        )
        if expected is None or len(atoms) != expected or len(set(atoms)) != expected:
            raise ValueError(f"Invalid internal-coordinate record: {record}")
        component = record.get("component")
        frame_axis = record.get("frame_axis")
        frame_direction_1 = record.get("frame_direction_1")
        if kind == "L":
            if component not in {-1, -2}:
                raise ValueError(f"Invalid linear-bend component: {record}")
            if axis_reference is None:
                if frame_axis is not None or frame_direction_1 is not None:
                    raise ValueError(f"Unexpected frame for a real-reference linear bend: {record}")
            elif axis_reference in {-1, -2}:
                if (
                    np.asarray(frame_axis).shape != (3,)
                    or np.asarray(frame_direction_1).shape != (3,)
                ):
                    raise ValueError(f"Invalid automatic linear-bend frame: {record}")
            else:
                raise ValueError(f"Invalid automatic linear-bend reference: {record}")
        elif any(value is not None for value in (
            component, axis_reference, frame_axis, frame_direction_1,
        )):
            raise ValueError(f"Unexpected linear-bend metadata: {record}")
        label = str(record.get("label", ""))
        name = label.split(":", 1)[0]
        if not name:
            raise ValueError(f"Missing internal-coordinate name: {record}")
        equilibrium = float(record["equilibrium"])
        printed = equilibrium if kind == "R" else float(np.rad2deg(equilibrium))
        definitions.append(Internal(
            name=name, kind=kind, atoms=atoms, printed_value=printed,
            component=component, axis_reference=axis_reference,
            frame_axis=None if frame_axis is None else tuple(float(x) for x in frame_axis),
            frame_direction_1=(
                None if frame_direction_1 is None
                else tuple(float(x) for x in frame_direction_1)
            ),
        ))
    if not definitions:
        raise ValueError(f"No internal coordinates found in {path}.")
    return definitions


def load_reference_data(scan_root):
    path = Path(scan_root) / "reference_data.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing a1 reference data: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "symbols", "equilibrium_angstrom", "masses_amu", "normal_modes",
            "projected_tangents", "internal_basis", "internal_weights",
            "equilibrium_internals", "equilibrium_jacobian",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Incomplete a1 reference_data.npz; missing {missing}.")
        reference = {key: np.array(data[key], copy=True) for key in required}
    reference["symbols"] = [str(value) for value in reference["symbols"]]
    r0 = reference["equilibrium_angstrom"]
    masses = reference["masses_amu"]
    modes = reference["normal_modes"]
    projected = reference["projected_tangents"]
    basis = reference["internal_basis"]
    weights = reference["internal_weights"]
    s0 = reference["equilibrium_internals"]
    j0 = reference["equilibrium_jacobian"]
    natoms = len(reference["symbols"])
    nmodes = int(modes.shape[0])
    expected_nmodes = vibrational_dof(r0)
    if nmodes != expected_nmodes:
        raise ValueError(
            f"Reference geometry implies {expected_nmodes} vibrational modes, "
            f"but reference_data.npz contains {nmodes}."
        )
    rank = nmodes
    expected_shapes = {
        "equilibrium_angstrom": (natoms, 3), "masses_amu": (natoms,),
        "normal_modes": (nmodes, natoms, 3),
        "projected_tangents": (nmodes, natoms, 3),
        "internal_basis": (s0.size, rank), "internal_weights": (s0.size,),
        "equilibrium_jacobian": (rank, 3 * natoms),
    }
    for key, shape in expected_shapes.items():
        if reference[key].shape != shape or not np.isfinite(reference[key]).all():
            raise ValueError(f"Invalid {key} shape or values in {path}; expected {shape}.")
    if np.any(masses <= 0) or np.any(weights <= 0):
        raise ValueError("Masses and internal-coordinate weights must be positive.")
    singular = np.linalg.svd(j0, compute_uv=False)
    reference["equilibrium_rank_ratio"] = float(singular[-1] / singular[0])

    definitions_path = Path(scan_root) / "internal_coordinates.json"
    if not definitions_path.is_file():
        raise FileNotFoundError(f"Missing a1 internal definitions: {definitions_path}")
    definitions = internal_definitions_from_json(definitions_path)
    if len(definitions) != s0.size:
        raise ValueError("Internal-coordinate JSON and reference_data.npz disagree.")
    equilibrium_check = internal_difference(internal_values(r0, definitions), s0, definitions)
    if float(np.max(np.abs(equilibrium_check))) > 1.0e-10:
        raise ValueError("Saved equilibrium internals do not reproduce the saved geometry.")

    independent_tangents = j0 @ modes.reshape(nmodes, -1).T
    reconstructed_projected = np.asarray([
        mass_metric_step(j0, independent_tangents[:, index], masses).reshape(natoms, 3)
        for index in range(nmodes)
    ])
    denominator = max(1.0, float(np.max(np.abs(projected))))
    if float(np.max(np.abs(reconstructed_projected - projected))) / denominator > 2.0e-9:
        raise ValueError("Saved a1 projected tangents are inconsistent with J0 and the normal modes.")
    reference.update(
        definitions=definitions,
        independent_tangents=independent_tangents,
        reference_path=path,
        definitions_path=definitions_path,
        n_atoms=natoms,
        n_modes=nmodes,
        reference_is_linear=(expected_nmodes == 3 * natoms - 5),
    )
    return reference


def validated_a6_temperature(scan_root, requested_temperature):
    """Reject unknown or stale thermal data instead of silently changing T."""
    requested = float(requested_temperature)
    if not np.isfinite(requested) or requested <= 0:
        raise ValueError("Requested temperature must be finite and positive, as in a6.")
    path = Path(scan_root) / "a6_metadata.json"
    remedy = (
        f"Rerun a6_probdis_therm.py with --temperature {requested:.12g} "
        "and the same --scan-root, then rerun a7. Do not edit metadata to relabel old densities."
    )
    if not path.is_file():
        raise ValueError(f"Cannot verify a6 temperature: missing {path}. {remedy}")
    metadata = json.loads(path.read_text())
    try:
        stored = float(metadata["temperature_K"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid temperature_K in {path}. {remedy}") from exc
    if not np.isfinite(stored) or stored <= 0:
        raise ValueError(f"Invalid a6 temperature {stored!r} in {path}. {remedy}")
    if not np.isclose(stored, requested, atol=1.0e-8, rtol=0):
        raise ValueError(
            f"Temperature mismatch: a7 requested {requested:.12g} K "
            f"(config.TEMP or --temperature), but a6 data in {path} "
            f"were generated at {stored:.12g} K. {remedy}"
        )
    return stored


def load_mode_data(scan_root, mode, reference, requested_temperature=None):
    """Load the actual text outputs of the supplied a5/a6 scripts.

    This production reader deliberately has no dependency on either
    potential_for_quantum_step.npz or probability_for_ensemble.npz.
    """
    root = Path(scan_root)
    temperature = validated_a6_temperature(
        root, config.TEMP if requested_temperature is None else requested_temperature,
    )
    mode_dir = root / f"vib{mode}"
    path_file = mode_dir / "path_geometries.npz"
    energy_file = mode_dir / "energies_qs.dat"
    probability_file = root / config.PROBABILITY_DIR / f"vib{mode}.dat"
    momentum_file = root / config.MOMENTUM_DIR / f"vib{mode}.dat"
    coefficient_file = root / config.FIT_COEFF_FILE
    metadata_file = root / "a6_metadata.json"
    for path in (
        path_file, energy_file, probability_file, momentum_file, coefficient_file,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing prerequisite for mode {mode}: {path}")

    with np.load(path_file, allow_pickle=False) as data:
        required = {"Q_angstrom", "s_angstrom", "mu_amu", "geometries_angstrom"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Mode {mode}: incomplete a1 path; missing {missing}.")
        path_q = np.asarray(data["Q_angstrom"], float)
        path_s = np.asarray(data["s_angstrom"], float)
        path_mu = np.asarray(data["mu_amu"], float)
        path_geometries = np.asarray(data["geometries_angstrom"], float)
    expected_geometry_shape = (path_q.size, reference["n_atoms"], 3)
    if (
        path_q.ndim != 1 or path_q.size < 3 or path_s.shape != path_q.shape
        or path_mu.shape != path_q.shape or path_geometries.shape != expected_geometry_shape
        or not all(np.isfinite(value).all() for value in (
            path_q, path_s, path_mu, path_geometries,
        ))
        or np.any(np.diff(path_q) <= 0) or np.any(np.diff(path_s) <= 0)
        or np.any(path_mu <= 0)
    ):
        raise ValueError(f"Mode {mode}: invalid or non-monotonic a1 path data.")
    zero_index = int(np.argmin(np.abs(path_q)))
    if abs(path_q[zero_index]) > 1.0e-12 or abs(path_s[zero_index]) > 1.0e-12:
        raise ValueError(f"Mode {mode}: a1 path lacks the Q=s=0 equilibrium point.")
    if not np.allclose(
        path_geometries[zero_index], reference["equilibrium_angstrom"],
        atol=5.0e-10, rtol=0,
    ):
        raise ValueError(f"Mode {mode}: a1 path equilibrium geometry is inconsistent.")

    with coefficient_file.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        coefficient_rows = [row for row in reader if int(row["mode"]) == mode]
    if len(coefficient_rows) != 1:
        raise ValueError(
            f"Mode {mode}: expected one record in {coefficient_file}, "
            f"found {len(coefficient_rows)}."
        )
    coefficient = coefficient_rows[0]
    required_columns = {
        "polynomial_degree", "coordinate", "mu0_amu", "dV_ds", "d2V_ds2",
        "d3V_ds3", "d4V_ds4", "d5V_ds5", "d6V_ds6",
        "harmonic_d2V_ds2",
    }
    missing_columns = sorted(required_columns - set(coefficient))
    if missing_columns:
        raise ValueError(
            f"Mode {mode}: {coefficient_file} lacks columns {missing_columns}."
        )
    if coefficient["coordinate"] != "s":
        raise ValueError(f"Mode {mode}: fit coordinate is not s.")
    fit_model = coefficient.get("fit_model", "polynomial").strip().lower()
    if fit_model not in {"polynomial", "pchip"}:
        raise ValueError(f"Mode {mode}: unknown a5 fit model {fit_model!r}.")
    polynomial_degree = int(coefficient["polynomial_degree"])
    mass_amu = float(coefficient["mu0_amu"])
    derivative_columns = {
        1: "dV_ds", 2: "d2V_ds2", 3: "d3V_ds3", 4: "d4V_ds4",
        5: "d5V_ds5", 6: "d6V_ds6",
    }
    derivatives_ang = (
        {
            order: float(coefficient[column])
            for order, column in derivative_columns.items()
            if order <= polynomial_degree
        }
        if fit_model == "polynomial" else {}
    )
    harmonic_k_ang = float(coefficient["harmonic_d2V_ds2"])
    if (
        (fit_model == "polynomial" and polynomial_degree not in (2, 3, 4, 5, 6))
        or (fit_model == "pchip" and polynomial_degree != 0)
        or not np.isfinite(mass_amu) or mass_amu <= 0.0
        or not np.isfinite(list(derivatives_ang.values())).all()
        or not np.isfinite(harmonic_k_ang) or harmonic_k_ang <= 0.0
        or not np.isclose(mass_amu, path_mu[zero_index], atol=2.0e-9, rtol=2.0e-9)
    ):
        raise ValueError(f"Mode {mode}: invalid a5 coefficient record or inconsistent mass.")
    mass_emass = mass_amu * config.AMU_TO_EMASS
    frequency_cm = (
        math.sqrt(harmonic_k_ang * config.BOHR_TO_ANG**2 / mass_emass)
        / config.CM_TO_HARTREE
    )

    energy_table = np.loadtxt(energy_file)
    if (
        energy_table.ndim != 2 or energy_table.shape[0] < 3 or energy_table.shape[1] < 5
        or not np.isfinite(energy_table).all()
        or np.any(np.diff(energy_table[:, 0]) <= 0)
        or np.any(np.diff(energy_table[:, 1]) <= 0)
    ):
        raise ValueError(f"Mode {mode}: invalid {energy_file}.")
    fit_s_min_ang = float(energy_table[0, 1])
    fit_s_max_ang = float(energy_table[-1, 1])

    probability_table = np.loadtxt(probability_file)
    if (
        probability_table.ndim != 2 or probability_table.shape[0] < 3
        or probability_table.shape[1] < 8 or not np.isfinite(probability_table).all()
    ):
        raise ValueError(
            f"Mode {mode}: {probability_file} must contain the eight columns written by a6."
        )
    grid_bohr = np.asarray(probability_table[:, 0], float)
    grid_ang = np.asarray(probability_table[:, 4], float)
    if (
        np.any(np.diff(grid_bohr) <= 0)
        or not np.allclose(
            grid_ang, grid_bohr * config.BOHR_TO_ANG, atol=2.0e-12, rtol=2.0e-12,
        )
    ):
        raise ValueError(f"Mode {mode}: invalid a6 grid or bohr/angstrom conversion.")

    # The last two a6 text columns provide an independent consistency check
    # against the current a5 representation.
    if fit_model == "polynomial":
        expected_anh = np.zeros_like(grid_bohr)
        for order, derivative in derivatives_ang.items():
            expected_anh += (
                derivative * config.BOHR_TO_ANG**order * grid_bohr**order
                / math.factorial(order)
            )
    else:
        pchip_columns = {
            "pchip_left_slope", "pchip_right_slope", "pchip_tail_curvature",
        }
        missing_pchip_columns = sorted(pchip_columns - set(coefficient))
        if missing_pchip_columns:
            raise ValueError(
                f"Mode {mode}: {coefficient_file} lacks PCHIP columns "
                f"{missing_pchip_columns}."
            )
        pchip_model = dict(
            knots_s_angstrom=np.asarray(energy_table[:, 1], float),
            knots_v_hartree=np.asarray(energy_table[:, 4], float),
            left_slope_hartree_per_angstrom=float(coefficient["pchip_left_slope"]),
            right_slope_hartree_per_angstrom=float(coefficient["pchip_right_slope"]),
            tail_curvature_hartree_per_angstrom2=float(
                coefficient["pchip_tail_curvature"]
            ),
        )
        expected_anh = pchip_potential_on_angstrom_grid(pchip_model, grid_ang)
    expected_harm = 0.5 * harmonic_k_ang * config.BOHR_TO_ANG**2 * grid_bohr**2
    if not np.allclose(
        probability_table[:, 6], expected_anh, atol=2.0e-9, rtol=2.0e-7,
    ):
        raise ValueError(
            f"Mode {mode}: {probability_file} does not match the current "
            f"{fit_model} potential in {coefficient_file}."
        )
    if not np.allclose(
        probability_table[:, 7], expected_harm, atol=2.0e-9, rtol=2.0e-7,
    ):
        raise ValueError(
            f"Mode {mode}: harmonic potential in {probability_file} is inconsistent."
        )

    source_files = [path_file, energy_file, probability_file, coefficient_file]
    if metadata_file.is_file():
        metadata = json.loads(metadata_file.read_text())
        metadata_modes = [int(value) for value in metadata.get("selected_modes", [])]
        if metadata_modes and mode not in metadata_modes:
            raise ValueError(f"Mode {mode}: absent from the current a6 metadata.")
        metadata_points = int(metadata.get("grid_points", grid_bohr.size))
        metadata_half_range = float(metadata.get(
            "grid_half_length_bohr", max(abs(grid_bohr[0]), abs(grid_bohr[-1])),
        ))
        if (
            metadata_points != grid_bohr.size
            or not np.isclose(grid_bohr[0], -metadata_half_range, atol=2.0e-10, rtol=0)
            or not np.isclose(grid_bohr[-1], metadata_half_range, atol=2.0e-10, rtol=0)
        ):
            raise ValueError(f"Mode {mode}: a6 metadata and probability grid disagree.")
        source_files.append(metadata_file)

    harmonic_sigma_s_bohr, harmonic_sigma_p_au = harmonic_thermal_sigmas(
        frequency_cm, mass_amu, temperature,
    )

    raw_densities = {
        ANHARMONIC: np.asarray(probability_table[:, 3], float),
        HARMONIC: np.asarray(probability_table[:, 5], float),
    }
    densities, cdfs = {}, {}
    for ensemble in DENSITY_KEYS:
        density, cdf = cumulative_trapezoid_cdf(grid_bohr, raw_densities[ensemble])
        densities[ensemble], cdfs[ensemble] = density, cdf

    momentum_table = np.loadtxt(momentum_file)
    if (
        momentum_table.ndim != 2 or momentum_table.shape[0] < 3
        or momentum_table.shape[1] < 8 or not np.isfinite(momentum_table).all()
    ):
        raise ValueError(
            f"Mode {mode}: {momentum_file} must contain the eight columns written by a6."
        )
    momentum_grid_au = np.asarray(momentum_table[:, 0], float)
    mass_emass = mass_amu * config.AMU_TO_EMASS
    if (
        np.any(np.diff(momentum_grid_au) <= 0.0)
        or not np.allclose(
            momentum_table[:, 1],
            momentum_grid_au * config.MOMENTUM_AU_TO_AMU_ANG_PER_FS,
            atol=2.0e-12, rtol=2.0e-12,
        )
        or not np.allclose(
            momentum_table[:, 2], momentum_grid_au / mass_emass,
            atol=2.0e-12, rtol=2.0e-12,
        )
        or not np.allclose(
            momentum_table[:, 3],
            momentum_grid_au / mass_emass
            * config.BOHR_PER_ATOMIC_TIME_TO_ANG_PER_FS,
            atol=2.0e-12, rtol=2.0e-12,
        )
    ):
        raise ValueError(f"Mode {mode}: invalid momentum grid or unit conversion in {momentum_file}.")
    raw_momentum_densities = {
        ANHARMONIC: np.asarray(momentum_table[:, 6], float),
        HARMONIC: np.asarray(momentum_table[:, 7], float),
    }
    momentum_densities, momentum_cdfs = {}, {}
    for ensemble in DENSITY_KEYS:
        density, cdf = cumulative_trapezoid_cdf(
            momentum_grid_au, raw_momentum_densities[ensemble],
        )
        momentum_densities[ensemble] = density
        momentum_cdfs[ensemble] = cdf
    source_files.append(momentum_file)
    return dict(
        mode=mode, path_q_ang=path_q, path_s_ang=path_s, path_mu_amu=path_mu,
        path_geometries_ang=path_geometries,
        grid_bohr=grid_bohr, densities=densities, cdfs=cdfs,
        mass_amu=mass_amu, frequency_cm=frequency_cm,
        harmonic_sigma_s_bohr=harmonic_sigma_s_bohr,
        harmonic_sigma_p_au=harmonic_sigma_p_au,
        momentum_grid_au=momentum_grid_au,
        momentum_densities=momentum_densities,
        momentum_cdfs=momentum_cdfs,
        fit_s_min_ang=fit_s_min_ang, fit_s_max_ang=fit_s_max_ang,
        temperature_K=temperature, fit_model=fit_model,
        polynomial_degree=polynomial_degree,
        probability_source=str(probability_file), momentum_source=str(momentum_file),
        source_files=tuple(source_files),
    )


def prepare_conditional_distribution(mode_data, ensemble):
    grid = mode_data["grid_bohr"]
    density = mode_data["densities"][ensemble]
    cdf = mode_data["cdfs"][ensemble]
    path_low = float(mode_data["path_s_ang"][0] / config.BOHR_TO_ANG)
    path_high = float(mode_data["path_s_ang"][-1] / config.BOHR_TO_ANG)
    fit_low = float(mode_data["fit_s_min_ang"] / config.BOHR_TO_ANG)
    fit_high = float(mode_data["fit_s_max_ang"] / config.BOHR_TO_ANG)
    if ensemble == HARMONIC:
        # The harmonic marginal is known on the full real line. It is not
        # conditioned on the a1 path, the a5 fit interval, or the a6 grid.
        # These finite grids exist only for plotting the analytic Gaussians.
        sigma_s = float(mode_data["harmonic_sigma_s_bohr"])
        sigma_p = float(mode_data["harmonic_sigma_p_au"])
        plot_points = max(1001, int(mode_data["grid_bohr"].size))
        position_plot_grid = np.linspace(-6.0 * sigma_s, 6.0 * sigma_s, plot_points)
        momentum_plot_grid = np.linspace(-6.0 * sigma_p, 6.0 * sigma_p, plot_points)
        position_density = normal_density(position_plot_grid, sigma_s)
        momentum_density = normal_density(momentum_plot_grid, sigma_p)
        position_density_map = dict(mode_data["densities"])
        momentum_density_map = dict(mode_data["momentum_densities"])
        position_density_map[HARMONIC] = position_density
        momentum_density_map[HARMONIC] = momentum_density
        position_cdf_map = dict(mode_data["cdfs"])
        momentum_cdf_map = dict(mode_data["momentum_cdfs"])
        position_cdf_map[HARMONIC] = ndtr(position_plot_grid / sigma_s)
        momentum_cdf_map[HARMONIC] = ndtr(momentum_plot_grid / sigma_p)
        prepared = dict(mode_data)
        prepared.update(
            ensemble=ensemble, grid_bohr=position_plot_grid,
            densities=position_density_map, cdfs=position_cdf_map,
            momentum_grid_au=momentum_plot_grid,
            momentum_densities=momentum_density_map,
            momentum_cdfs=momentum_cdf_map,
            support_low_bohr=-np.inf, support_high_bohr=np.inf,
            cdf_low=0.0, cdf_high=1.0,
            geometry_path_low_bohr=path_low, geometry_path_high_bohr=path_high,
            fit_low_bohr=fit_low, fit_high_bohr=fit_high,
            retained_probability=1.0, tail_probability=0.0,
            conditional_grid_bohr=position_plot_grid,
            conditional_density_per_bohr=position_density,
        )
        return prepared
    if ensemble == ANHARMONIC:
        # The anharmonic density is trusted only where both a curvilinear
        # geometry and ab initio points used by a5 are available.  The harmonic
        # reference is analytic, so it is limited only by the geometric path.
        support_low = max(path_low, fit_low, float(grid[0]))
        support_high = min(path_high, fit_high, float(grid[-1]))
    else:
        support_low = max(path_low, float(grid[0]))
        support_high = min(path_high, float(grid[-1]))
    if support_low >= support_high:
        raise ValueError(f"Mode {mode_data['mode']}: the validated sampling intervals do not overlap.")
    cdf_low = evaluate_piecewise_linear_cdf(grid, density, cdf, support_low)
    cdf_high = evaluate_piecewise_linear_cdf(grid, density, cdf, support_high)
    retained = float(cdf_high - cdf_low)
    if retained <= 0:
        raise ValueError(f"Mode {mode_data['mode']}: no probability lies on the validated support.")
    tail = float(np.clip(1.0 - retained, 0.0, 1.0))

    inside = (grid > support_low) & (grid < support_high)
    conditional_grid = np.concatenate(([support_low], grid[inside], [support_high]))
    conditional_density = np.interp(conditional_grid, grid, density) / retained
    conditional_density /= np.trapezoid(conditional_density, conditional_grid)
    return dict(
        **mode_data, ensemble=ensemble, support_low_bohr=support_low,
        support_high_bohr=support_high, cdf_low=cdf_low, cdf_high=cdf_high,
        geometry_path_low_bohr=path_low, geometry_path_high_bohr=path_high,
        fit_low_bohr=fit_low, fit_high_bohr=fit_high,
        retained_probability=retained, tail_probability=tail,
        conditional_grid_bohr=conditional_grid,
        conditional_density_per_bohr=conditional_density,
    )


def sample_mode(distribution, uniform):
    if distribution["ensemble"] == HARMONIC:
        s_bohr = normal_quantile(uniform, distribution["harmonic_sigma_s_bohr"])
        s_ang = float(s_bohr * config.BOHR_TO_ANG)
        # For the purely harmonic reference, s itself is the linear modal
        # amplitude. No finite numerical s(Q) path is required.
        return s_bohr, s_ang, s_ang
    target_probability = distribution["cdf_low"] + uniform * distribution["retained_probability"]
    s_bohr = invert_piecewise_linear_cdf(
        distribution["grid_bohr"], distribution["densities"][distribution["ensemble"]],
        distribution["cdfs"][distribution["ensemble"]], target_probability,
    )
    s_ang = float(s_bohr * config.BOHR_TO_ANG)
    q_ang = float(np.interp(s_ang, distribution["path_s_ang"], distribution["path_q_ang"]))
    return float(s_bohr), s_ang, q_ang


def sample_momentum(distribution, uniform):
    """Sample the independent momentum marginal in atomic units."""
    if distribution["ensemble"] == HARMONIC:
        return normal_quantile(uniform, distribution["harmonic_sigma_p_au"])
    p_s_au = invert_piecewise_linear_cdf(
        distribution["momentum_grid_au"],
        distribution["momentum_densities"][distribution["ensemble"]],
        distribution["momentum_cdfs"][distribution["ensemble"]],
        uniform,
    )
    return float(p_s_au)


def minimum_pair_distance(coords):
    coords = np.asarray(coords, float)
    delta = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(delta, axis=2)
    upper = distances[np.triu_indices(len(coords), 1)]
    return float(np.min(upper))


def bond_length_diagnostics(coords, reference):
    ratios = []
    for index, definition in enumerate(reference["definitions"]):
        if definition.kind != "R":
            continue
        atom1, atom2 = definition.atoms
        current = float(np.linalg.norm(coords[atom2] - coords[atom1]))
        equilibrium = float(reference["equilibrium_internals"][index])
        ratios.append(current / equilibrium)
    if not ratios:
        return 1.0, 1.0, 0.0
    ratios = np.asarray(ratios)
    maximum_change = float(np.max(np.abs(ratios - 1.0)) * 100.0)
    return float(np.min(ratios)), float(np.max(ratios)), maximum_change


def reconstruct_geometry(q_ang, mode_numbers, reference):
    q_ang = np.asarray(q_ang, float)
    mode_indices = np.asarray(mode_numbers, int) - 1
    if reference.get("coordinate_system", "curvilinear") == "cartesian":
        # Exact simultaneous rectilinear displacement.  This branch avoids
        # Wilson-B evaluation and nonlinear back-transformation completely.
        current = (
            reference["equilibrium_angstrom"]
            + np.tensordot(
                q_ang, reference["normal_modes"][mode_indices], axes=(0, 0),
            )
        )
        min_distance = minimum_pair_distance(current)
        if min_distance < config.A7_MIN_INTERATOMIC_DISTANCE_ANG:
            raise RuntimeError(
                f"Minimum interatomic distance {min_distance:.3f} Ang is below "
                "the safety limit."
            )
        minimum_bond_ratio, maximum_bond_ratio, maximum_bond_change = (
            bond_length_diagnostics(current, reference)
        )
        return current, dict(
            residual=0.0,
            rank_ratio=reference["equilibrium_rank_ratio"],
            rank_ratio_is_local=False,
            minimum_pair_distance_ang=min_distance,
            minimum_bond_length_ratio=minimum_bond_ratio,
            maximum_bond_length_ratio=maximum_bond_ratio,
            maximum_bond_length_change_percent=maximum_bond_change,
            maximum_torsion_displacement_rad=0.0,
            continuation_steps=1,
            backtransform_iterations=0,
            largest_history=0,
        )

    target = reference["independent_tangents"][:, mode_indices] @ q_ang

    projected = reference["projected_tangents"][mode_indices]
    individual_extent = 0.0
    for amplitude, tangent in zip(q_ang, projected):
        individual_extent = max(
            individual_extent,
            float(abs(amplitude) * np.max(np.linalg.norm(tangent, axis=1))),
        )
    combined_linear = np.tensordot(q_ang, projected, axes=(0, 0))
    combined_extent = float(np.max(np.linalg.norm(combined_linear, axis=1)))
    estimated_extent = max(individual_extent, combined_extent)
    n_steps = max(1, int(math.ceil(estimated_extent / config.A7_CONTINUATION_STEP_ANG)))
    if n_steps > int(config.A7_MAX_CONTINUATION_STEPS):
        raise RuntimeError(
            f"Continuation requires {n_steps} steps, above A7_MAX_CONTINUATION_STEPS."
        )

    current = reference["equilibrium_angstrom"].copy()
    primitive_difference = np.zeros_like(reference["equilibrium_internals"])
    total_iterations = 0
    largest_history = 0
    accepted_steps = 0
    for step_index in range(1, n_steps + 1):
        fraction = step_index / n_steps
        current, primitive_difference, iterations, history, steps = continue_backtransform(
            fraction * target,
            reference["equilibrium_angstrom"], reference["definitions"],
            reference["equilibrium_internals"], reference["internal_weights"],
            reference["internal_basis"], reference["masses_amu"], initial=current,
            initial_difference=primitive_difference, rank_rtol=config.INTERNAL_RANK_RTOL,
            step=config.INTERNAL_FD_STEP_ANG, tolerance=config.BACKTRANS_TOL,
            max_iterations=config.BACKTRANS_MAX_ITER,
            cart_step_limit=config.BACKTRANS_CART_STEP_LIMIT_ANG,
        )
        total_iterations += iterations
        accepted_steps += steps
        largest_history = max(largest_history, len(history))

    achieved = reference["internal_basis"].T @ (
        reference["internal_weights"] * primitive_difference
    )
    residual = float(np.linalg.norm(achieved - target, np.inf))
    if residual > config.BACKTRANS_TOL:
        raise RuntimeError(f"Final internal residual {residual:.3e} exceeds tolerance.")

    jacobian = reference["internal_basis"].T @ (
        reference["internal_weights"][:, None]
        * wilson_b(current, reference["definitions"], config.INTERNAL_FD_STEP_ANG / 2)
    )
    singular = np.linalg.svd(jacobian, compute_uv=False)
    rank_ratio = float(singular[-1] / singular[0])
    if rank_ratio <= config.INTERNAL_RANK_RTOL:
        raise RuntimeError(f"Final internal Jacobian is singular: ratio={rank_ratio:.3e}.")

    torsion_indices = [
        index for index, definition in enumerate(reference["definitions"])
        if definition.kind == "D"
    ]
    if torsion_indices:
        max_torsion = float(np.max(np.abs(primitive_difference[torsion_indices])))
    else:
        max_torsion = 0.0
    min_distance = minimum_pair_distance(current)
    if min_distance < config.A7_MIN_INTERATOMIC_DISTANCE_ANG:
        raise RuntimeError(
            f"Minimum interatomic distance {min_distance:.3f} Ang is below the safety limit."
        )
    minimum_bond_ratio, maximum_bond_ratio, maximum_bond_change = bond_length_diagnostics(
        current, reference,
    )
    return current, dict(
        residual=residual, rank_ratio=rank_ratio, minimum_pair_distance_ang=min_distance,
        rank_ratio_is_local=True,
        minimum_bond_length_ratio=minimum_bond_ratio,
        maximum_bond_length_ratio=maximum_bond_ratio,
        maximum_bond_length_change_percent=maximum_bond_change,
        maximum_torsion_displacement_rad=max_torsion, continuation_steps=accepted_steps,
        backtransform_iterations=total_iterations, largest_history=largest_history,
    )


def project_modal_momenta(
    geometry, q_ang, p_s_au, mode_numbers, distributions, reference,
):
    """Project independent modal momenta to Cartesian velocities.

    Each sampled ``p_s`` is conjugate to the constant-mass coordinate ``s``.
    The independent-mode relation ``ds/dt = p_s/mu(0)`` is retained. At the
    sampled geometry the local Wilson-B pseudoinverse provides ``dR/dQ``. The
    anharmonic ensemble uses the stored path mass through
    ``dQ/ds = sqrt(mu(0)/mu(Q))``; the analytic harmonic ensemble has ``Q=s``
    and therefore ``dQ/ds=1``. This preserves the uncoupled-mode approximation
    while using the local internal-coordinate tangent.
    """
    geometry = np.asarray(geometry, float)
    q_ang = np.asarray(q_ang, float)
    p_s_au = np.asarray(p_s_au, float)
    mode_indices = np.asarray(mode_numbers, int) - 1
    if (
        q_ang.shape != p_s_au.shape
        or q_ang.shape != (len(distributions),)
        or len(mode_numbers) != len(distributions)
    ):
        raise ValueError("Modal coordinates and momenta have inconsistent dimensions.")

    tangent_columns = []
    dQ_ds = []
    mu0_amu = []
    coordinate_system = reference.get("coordinate_system", "curvilinear")
    if coordinate_system == "cartesian":
        jacobian = None
        for mode_index, q_value, distribution in zip(mode_indices, q_ang, distributions):
            factor = 1.0
            if (distribution["ensemble"] == ANHARMONIC
                    and reference.get("density_coordinate_system", "cartesian") == "curvilinear"):
                if not distribution["path_q_ang"][0] <= q_value <= distribution["path_q_ang"][-1]:
                    raise ValueError("A sampled Q lies outside its saved source path.")
                local_mu = float(np.interp(q_value, distribution["path_q_ang"],
                                           distribution["path_mu_amu"]))
                if local_mu <= 0:
                    raise ValueError("A path mass is not positive.")
                factor = math.sqrt(float(distribution["mass_amu"]) / local_mu)
            tangent_columns.append(
                reference["normal_modes"][mode_index].reshape(-1) * factor
            )
            dQ_ds.append(factor)
            mu0_amu.append(float(distribution["mass_amu"]))
    else:
        jacobian = reference["internal_basis"].T @ (
            reference["internal_weights"][:, None]
            * wilson_b(
                geometry, reference["definitions"],
                config.INTERNAL_FD_STEP_ANG / 2,
            )
        )
        for mode_index, q_value, distribution in zip(
            mode_indices, q_ang, distributions,
        ):
            reference_mu = float(distribution["mass_amu"])
            if distribution["ensemble"] == HARMONIC:
                # The analytic harmonic coordinate is linear: Q=s and dQ/ds=1.
                # Consequently no interpolation or extrapolation of mu(Q) is used.
                local_dQ_ds = 1.0
            else:
                if (
                    q_value < distribution["path_q_ang"][0] - 1.0e-10
                    or q_value > distribution["path_q_ang"][-1] + 1.0e-10
                ):
                    raise ValueError("A sampled Q lies outside its saved curvilinear path.")
                local_mu = float(np.interp(
                    q_value, distribution["path_q_ang"], distribution["path_mu_amu"],
                ))
                if local_mu <= 0.0 or reference_mu <= 0.0:
                    raise ValueError("A path mass is not positive.")
                local_dQ_ds = math.sqrt(reference_mu / local_mu)
            dR_dQ = mass_metric_step(
                jacobian,
                reference["independent_tangents"][:, mode_index],
                reference["masses_amu"],
            )
            tangent_columns.append(dR_dQ * local_dQ_ds)
            dQ_ds.append(local_dQ_ds)
            mu0_amu.append(reference_mu)

    tangent_matrix = np.column_stack(tangent_columns)
    dQ_ds = np.asarray(dQ_ds)
    mu0_amu = np.asarray(mu0_amu)
    modal_velocity_bohr_per_atomic_time = (
        p_s_au / (mu0_amu * config.AMU_TO_EMASS)
    )
    velocity_flat = tangent_matrix @ modal_velocity_bohr_per_atomic_time
    velocity_raw = velocity_flat.reshape(reference["n_atoms"], 3)
    if reference["reference_is_linear"]:
        # A linear reference has only two finite rigid rotations.  At a bent
        # geometry, subtracting all three instantaneous rotations would also
        # erase the vibrational angular momentum conjugate to the azimuth of
        # the two-component degenerate bend.  The Wilson-metric solution is
        # already orthogonal to the five-dimensional rigid nullspace of the
        # linear chart; remove only any round-off translation here.
        velocity_bohr_per_atomic_time = velocity_raw - np.average(
            velocity_raw, axis=0, weights=reference["masses_amu"],
        )
    else:
        velocity_bohr_per_atomic_time = remove_rigid_velocity(
            geometry, velocity_raw, reference["masses_amu"],
        )
    velocity_angstrom_per_fs = (
        velocity_bohr_per_atomic_time
        * config.BOHR_PER_ATOMIC_TIME_TO_ANG_PER_FS
    )
    cartesian_momentum_au = (
        reference["masses_amu"][:, None] * config.AMU_TO_EMASS
        * velocity_bohr_per_atomic_time
    )

    modal_velocity_angstrom_per_fs = (
        modal_velocity_bohr_per_atomic_time
        * config.BOHR_PER_ATOMIC_TIME_TO_ANG_PER_FS
    )
    if coordinate_system == "cartesian":
        # The velocity is the exact derivative of the selected rectilinear
        # geometry model; an internal-coordinate rate residual is inapplicable.
        absolute_rate_residual = 0.0
        relative_rate_residual = 0.0
    else:
        q_velocity_angstrom_per_fs = dQ_ds * modal_velocity_angstrom_per_fs
        target_internal_rate = (
            reference["independent_tangents"][:, mode_indices]
            @ q_velocity_angstrom_per_fs
        )
        achieved_internal_rate = jacobian @ velocity_angstrom_per_fs.reshape(-1)
        absolute_rate_residual = float(np.linalg.norm(
            achieved_internal_rate - target_internal_rate, np.inf,
        ))
        rate_scale = max(1.0e-14, float(np.linalg.norm(target_internal_rate, np.inf)))
        relative_rate_residual = absolute_rate_residual / rate_scale
        if relative_rate_residual > config.A7_VELOCITY_INTERNAL_RTOL:
            raise RuntimeError(
                "Cartesian velocity does not reproduce the sampled internal rates: "
                f"relative residual={relative_rate_residual:.3e}."
            )

    masses = reference["masses_amu"]
    center_of_mass_velocity = np.average(
        velocity_angstrom_per_fs, axis=0, weights=masses,
    )
    centered_bohr = (
        geometry - np.average(geometry, axis=0, weights=masses)
    ) / config.BOHR_TO_ANG
    angular_momentum_au = np.sum(
        np.cross(centered_bohr, cartesian_momentum_au), axis=0,
    )
    kinetic_independent_hartree = float(np.sum(
        p_s_au**2 / (2.0 * mu0_amu * config.AMU_TO_EMASS)
    ))
    kinetic_cartesian_hartree = float(0.5 * np.sum(
        reference["masses_amu"][:, None] * config.AMU_TO_EMASS
        * velocity_bohr_per_atomic_time**2
    ))

    mass_xyz = np.repeat(masses, 3)
    local_metric_amu = tangent_matrix.T @ (mass_xyz[:, None] * tangent_matrix)
    metric_scale = np.sqrt(np.outer(np.diag(local_metric_amu), np.diag(local_metric_amu)))
    normalized_metric = local_metric_amu / metric_scale
    np.fill_diagonal(normalized_metric, 0.0)
    maximum_metric_coupling = (
        float(np.max(np.abs(normalized_metric))) if normalized_metric.size else 0.0
    )
    return dict(
        modal_momentum_au=p_s_au,
        modal_momentum_amu_angstrom_per_fs=(
            p_s_au * config.MOMENTUM_AU_TO_AMU_ANG_PER_FS
        ),
        modal_velocity_bohr_per_atomic_time=modal_velocity_bohr_per_atomic_time,
        modal_velocity_angstrom_per_fs=modal_velocity_angstrom_per_fs,
        dQ_ds=dQ_ds,
        cartesian_velocity_bohr_per_atomic_time=velocity_bohr_per_atomic_time,
        cartesian_velocity_angstrom_per_fs=velocity_angstrom_per_fs,
        cartesian_momentum_au=cartesian_momentum_au,
        internal_rate_absolute_residual=absolute_rate_residual,
        internal_rate_relative_residual=relative_rate_residual,
        center_of_mass_velocity_angstrom_per_fs=center_of_mass_velocity,
        angular_momentum_au=angular_momentum_au,
        independent_modal_kinetic_energy_hartree=kinetic_independent_hartree,
        cartesian_kinetic_energy_hartree=kinetic_cartesian_hartree,
        maximum_local_metric_coupling=maximum_metric_coupling,
    )


def draw_candidate(distributions, rng):
    uniforms = rng.random(len(distributions))
    s_bohr, s_ang, q_ang = [], [], []
    for distribution, uniform in zip(distributions, uniforms):
        sampled_s_bohr, sampled_s_ang, sampled_q_ang = sample_mode(distribution, uniform)
        s_bohr.append(sampled_s_bohr)
        s_ang.append(sampled_s_ang)
        q_ang.append(sampled_q_ang)
    return uniforms, np.asarray(s_bohr), np.asarray(s_ang), np.asarray(q_ang)


def draw_momenta(distributions, rng):
    uniforms = rng.random(len(distributions))
    momenta = np.asarray([
        sample_momentum(distribution, uniform)
        for distribution, uniform in zip(distributions, uniforms)
    ])
    return uniforms, momenta


def independent_random_generators(seed):
    """Return reproducible, distinct position and momentum random streams."""
    if int(seed) < 0:
        raise ValueError("The random seed must be a non-negative integer.")
    coordinate_rng = np.random.default_rng(seed)
    seed_value = int(seed)
    momentum_seed = np.random.SeedSequence([
        seed_value & 0xFFFFFFFF,
        (seed_value >> 32) & 0xFFFFFFFF,
        0x414E4844,  # ASCII-like fixed AnhDis stream tag.
        0x4953504D,
    ])
    return coordinate_rng, np.random.default_rng(momentum_seed)


def validate_saved_single_mode_paths(mode_data, reference):
    """Reconstruct both finite endpoints and compare them with the a1 paths."""
    rows = []
    for item in mode_data:
        for index in (0, -1):
            q_ang = float(item["path_q_ang"][index])
            reconstructed, info = reconstruct_geometry(
                np.asarray([q_ang]), [item["mode"]], reference,
            )
            difference = float(np.max(np.abs(
                reconstructed - item["path_geometries_ang"][index]
            )))
            rows.append(dict(
                mode=item["mode"], Q_angstrom=f"{q_ang:.12e}",
                maximum_cartesian_difference_angstrom=f"{difference:.12e}",
                internal_residual=f"{info['residual']:.12e}",
                jacobian_singular_ratio=f"{info['rank_ratio']:.12e}",
            ))
            if difference > config.A7_PATH_RECONSTRUCTION_TOL_ANG:
                raise RuntimeError(
                    f"Mode {item['mode']}, Q={q_ang:g}: a7 reconstruction differs from "
                    f"the saved a1 path by {difference:.3e} Ang."
                )
    return rows


def generate_ensemble(distributions, reference, n_samples, seed, workers=1):
    # Preserve the exact historical position stream.  Momentum uses a distinct
    # deterministic stream, so adding phase-space sampling cannot change any
    # geometry previously produced with the same seed.
    coordinate_rng, momentum_rng = independent_random_generators(seed)
    mode_numbers = [item["mode"] for item in distributions]
    geometries, uniforms, sampled_s_bohr, sampled_s_ang, sampled_q_ang = [], [], [], [], []
    momentum_uniforms, sampled_p_s_au, velocity_info = [], [], []
    geometry_info = []
    rejection_reasons = Counter()
    maximum_attempts = int(n_samples) * int(config.A7_MAX_ATTEMPTS_FACTOR)
    workers = int(workers)
    if workers < 1:
        raise ValueError("a7 workers must be a positive integer.")

    # g1 is always the optimized geometry.  Its curvilinear coordinates are
    # exactly Q=s=0 for every mode.  Store the corresponding conditional-CDF
    # quantiles as metadata so every array in the NPZ keeps the same length.
    zero_q = np.zeros(len(distributions), dtype=float)
    equilibrium_geometry, equilibrium_info = reconstruct_geometry(
        zero_q, mode_numbers, reference,
    )
    zero_uniforms = np.asarray([
        float(conditional_cdf_function(distribution)(0.0))
        for distribution in distributions
    ])
    geometries.append(equilibrium_geometry)
    uniforms.append(zero_uniforms)
    sampled_s_bohr.append(np.zeros(len(distributions), dtype=float))
    sampled_s_ang.append(np.zeros(len(distributions), dtype=float))
    sampled_q_ang.append(zero_q)
    geometry_info.append(equilibrium_info)
    p_uniform, p_s_au = draw_momenta(distributions, momentum_rng)
    equilibrium_velocity = project_modal_momenta(
        equilibrium_geometry, zero_q, p_s_au, mode_numbers, distributions, reference,
    )
    momentum_uniforms.append(p_uniform)
    sampled_p_s_au.append(p_s_au)
    velocity_info.append(equilibrium_velocity)

    # Count the deterministic equilibrium member as one accepted attempt so
    # rejected_attempts remains attempts-accepted_samples in the diagnostics.
    attempts = 1
    if workers == 1:
        while len(geometries) < n_samples and attempts < maximum_attempts:
            attempts += 1
            u, s_bohr, s_ang, q_ang = draw_candidate(distributions, coordinate_rng)
            try:
                geometry, info = reconstruct_geometry(q_ang, mode_numbers, reference)
                p_uniform, p_s_au = draw_momenta(distributions, momentum_rng)
                projected_momentum = project_modal_momenta(
                    geometry, q_ang, p_s_au, mode_numbers, distributions, reference,
                )
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                rejection_reasons[reason] += 1
                continue
            geometries.append(geometry)
            uniforms.append(u)
            sampled_s_bohr.append(s_bohr)
            sampled_s_ang.append(s_ang)
            sampled_q_ang.append(q_ang)
            geometry_info.append(info)
            momentum_uniforms.append(p_uniform)
            sampled_p_s_au.append(p_s_au)
            velocity_info.append(projected_momentum)
    else:
        # Candidate quantiles are drawn by the parent in the historical order.
        # executor.map also returns reconstructions in that order.  Momentum is
        # drawn only after a geometry succeeds, exactly as in the sequential
        # algorithm.  Consequently workers=1 and workers>1 give identical
        # accepted samples for a fixed seed.
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_reconstruction_worker,
            initargs=(reference, mode_numbers),
        ) as executor:
            while len(geometries) < n_samples and attempts < maximum_attempts:
                remaining_attempts = maximum_attempts - attempts
                batch_size = min(max(2 * workers, 1), remaining_attempts)
                candidates = []
                tasks = []
                for offset in range(batch_size):
                    u, s_bohr, s_ang, q_ang = draw_candidate(
                        distributions, coordinate_rng,
                    )
                    candidate_index = attempts + offset + 1
                    candidates.append((u, s_bohr, s_ang, q_ang))
                    tasks.append((candidate_index, q_ang))
                reconstructed = executor.map(
                    _reconstruct_candidate_in_worker, tasks, chunksize=1,
                )
                for candidate, worker_result in zip(candidates, reconstructed):
                    candidate_index, geometry, info, reason = worker_result
                    attempts = candidate_index
                    if reason is not None:
                        rejection_reasons[reason] += 1
                        continue
                    u, s_bohr, s_ang, q_ang = candidate
                    try:
                        p_uniform, p_s_au = draw_momenta(distributions, momentum_rng)
                        projected_momentum = project_modal_momenta(
                            geometry, q_ang, p_s_au, mode_numbers,
                            distributions, reference,
                        )
                    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                        reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
                        rejection_reasons[reason] += 1
                        continue
                    geometries.append(geometry)
                    uniforms.append(u)
                    sampled_s_bohr.append(s_bohr)
                    sampled_s_ang.append(s_ang)
                    sampled_q_ang.append(q_ang)
                    geometry_info.append(info)
                    momentum_uniforms.append(p_uniform)
                    sampled_p_s_au.append(p_s_au)
                    velocity_info.append(projected_momentum)
                    if len(geometries) >= n_samples:
                        break
    if len(geometries) != n_samples:
        detail = "; ".join(f"{count} x {reason}" for reason, count in rejection_reasons.items())
        raise RuntimeError(
            f"Only {len(geometries)}/{n_samples} valid geometries after {attempts} attempts. "
            f"Rejections: {detail or 'none'}"
        )
    return dict(
        geometries_angstrom=np.asarray(geometries), uniforms=np.asarray(uniforms),
        s_bohr=np.asarray(sampled_s_bohr), s_angstrom=np.asarray(sampled_s_ang),
        Q_angstrom=np.asarray(sampled_q_ang), geometry_info=geometry_info,
        momentum_uniforms=np.asarray(momentum_uniforms),
        p_s_au=np.asarray(sampled_p_s_au), velocity_info=velocity_info,
        attempts=attempts, rejection_reasons=rejection_reasons,
    )


def conditional_cdf_function(distribution):
    def function(values):
        if distribution["ensemble"] == HARMONIC:
            return ndtr(np.asarray(values, float) / distribution["harmonic_sigma_s_bohr"])
        full = evaluate_piecewise_linear_cdf(
            distribution["grid_bohr"],
            distribution["densities"][distribution["ensemble"]],
            distribution["cdfs"][distribution["ensemble"]], values,
        )
        return np.clip(
            (np.asarray(full) - distribution["cdf_low"])
            / distribution["retained_probability"], 0.0, 1.0,
        )
    return function


def momentum_cdf_function(distribution):
    def function(values):
        if distribution["ensemble"] == HARMONIC:
            return ndtr(np.asarray(values, float) / distribution["harmonic_sigma_p_au"])
        return evaluate_piecewise_linear_cdf(
            distribution["momentum_grid_au"],
            distribution["momentum_densities"][distribution["ensemble"]],
            distribution["momentum_cdfs"][distribution["ensemble"]], values,
        )
    return function


def write_multiframe_xyz(
    path, symbols, geometries, ensemble, temperature, coordinate_system,
):
    with Path(path).open("w") as stream:
        for sample, geometry in enumerate(geometries, 1):
            stream.write(f"{len(symbols)}\n")
            if sample == 1:
                description = "optimized equilibrium geometry"
            elif coordinate_system == "cartesian":
                description = "simultaneous rectilinear Cartesian displacement"
            else:
                description = "simultaneous curvilinear internal back-transformation"
            stream.write(
                f"Sample {sample}; {ensemble}; T={temperature:.6f} K; "
                f"{description}\n"
            )
            for symbol, xyz in zip(symbols, geometry):
                stream.write(
                    f"{symbol:2s} {xyz[0]: .12f} {xyz[1]: .12f} {xyz[2]: .12f}\n"
                )


def write_multiframe_vectors(path, symbols, vectors, ensemble, temperature, quantity, unit):
    """Write atom-resolved Cartesian vectors in an XYZ-shaped text file."""
    with Path(path).open("w") as stream:
        for sample, values in enumerate(vectors, 1):
            stream.write(f"{len(symbols)}\n")
            stream.write(
                f"Sample {sample}; {ensemble}; T={temperature:.6f} K; "
                f"Cartesian {quantity}; unit={unit}\n"
            )
            for symbol, vector in zip(symbols, values):
                stream.write(
                    f"{symbol:2s} {vector[0]: .12e} {vector[1]: .12e} "
                    f"{vector[2]: .12e}\n"
                )


def save_ensemble(output_root, ensemble, distributions, result, reference, source_log_name):
    xyz_dir_name, histogram_dir_name, all_xyz_name = OUTPUT_LAYOUT[ensemble]
    xyz_dir = output_root / xyz_dir_name
    histogram_dir = output_root / histogram_dir_name
    xyz_dir.mkdir()
    histogram_dir.mkdir()
    temperature = distributions[0]["temperature_K"]
    geometries = result["geometries_angstrom"]
    cartesian_velocities_bohr_au_time = np.asarray([
        item["cartesian_velocity_bohr_per_atomic_time"]
        for item in result["velocity_info"]
    ])
    cartesian_velocities_ang_fs = np.asarray([
        item["cartesian_velocity_angstrom_per_fs"]
        for item in result["velocity_info"]
    ])
    cartesian_momenta_au = np.asarray([
        item["cartesian_momentum_au"] for item in result["velocity_info"]
    ])
    if config.A7_SAVE_INDIVIDUAL_XYZ:
        for sample, geometry in enumerate(geometries, 1):
            description = (
                "optimized equilibrium geometry" if sample == 1
                else (
                    "rectilinear Cartesian modes"
                    if reference.get("coordinate_system", "curvilinear") == "cartesian"
                    else "curvilinear internals"
                )
            )
            write_xyz(
                xyz_dir / f"g{sample}_{source_log_name}.xyz",
                reference["symbols"], geometry,
                f"Sample {sample}; {ensemble}; T={temperature:.6f} K; {description}",
            )
            write_xyz(
                xyz_dir / f"g{sample}_{source_log_name}.vel",
                reference["symbols"], cartesian_velocities_ang_fs[sample - 1],
                f"Sample {sample}; {ensemble}; Cartesian velocity; unit=angstrom/fs",
            )
            write_xyz(
                xyz_dir / f"g{sample}_{source_log_name}.mom",
                reference["symbols"], cartesian_momenta_au[sample - 1],
                (
                    f"Sample {sample}; {ensemble}; Cartesian momentum; "
                    "unit=atomic momentum (electron_mass*bohr/atomic_time)"
                ),
            )
    write_multiframe_xyz(
        output_root / all_xyz_name, reference["symbols"], geometries, ensemble,
        temperature, reference.get("coordinate_system", "curvilinear"),
    )
    velocity_name, momentum_name = VECTOR_OUTPUT_NAMES[ensemble]
    write_multiframe_vectors(
        output_root / velocity_name, reference["symbols"],
        cartesian_velocities_ang_fs, ensemble, temperature,
        "velocity", "angstrom/fs",
    )
    write_multiframe_vectors(
        output_root / momentum_name, reference["symbols"],
        cartesian_momenta_au, ensemble, temperature,
        "momentum", "electron_mass*bohr/atomic_time",
    )
    np.savez_compressed(
        output_root / f"sampled_coordinates_{ensemble}.npz",
        ensemble=np.asarray(ensemble), temperature_K=temperature,
        is_equilibrium_geometry=np.arange(len(geometries)) == 0,
        mode_numbers=np.asarray([item["mode"] for item in distributions], int),
        uniform_quantiles=result["uniforms"],
        position_uniform_quantiles=result["uniforms"],
        momentum_uniform_quantiles=result["momentum_uniforms"],
        s_bohr=result["s_bohr"],
        s_angstrom=result["s_angstrom"], Q_angstrom=result["Q_angstrom"],
        p_s_au=result["p_s_au"],
        p_s_amu_angstrom_per_fs=(
            result["p_s_au"] * config.MOMENTUM_AU_TO_AMU_ANG_PER_FS
        ),
        modal_velocity_bohr_per_atomic_time=np.asarray([
            item["modal_velocity_bohr_per_atomic_time"]
            for item in result["velocity_info"]
        ]),
        modal_velocity_angstrom_per_fs=np.asarray([
            item["modal_velocity_angstrom_per_fs"]
            for item in result["velocity_info"]
        ]),
        dQ_ds=np.asarray([item["dQ_ds"] for item in result["velocity_info"]]),
        geometries_angstrom=geometries,
        cartesian_velocity_bohr_per_atomic_time=cartesian_velocities_bohr_au_time,
        cartesian_velocity_angstrom_per_fs=cartesian_velocities_ang_fs,
        cartesian_momentum_au=cartesian_momenta_au,
        internal_residual=np.asarray([item["residual"] for item in result["geometry_info"]]),
        jacobian_singular_ratio=np.asarray([item["rank_ratio"] for item in result["geometry_info"]]),
        minimum_pair_distance_angstrom=np.asarray([
            item["minimum_pair_distance_ang"] for item in result["geometry_info"]
        ]),
        minimum_bond_length_ratio=np.asarray([
            item["minimum_bond_length_ratio"] for item in result["geometry_info"]
        ]),
        maximum_bond_length_ratio=np.asarray([
            item["maximum_bond_length_ratio"] for item in result["geometry_info"]
        ]),
        maximum_bond_length_change_percent=np.asarray([
            item["maximum_bond_length_change_percent"] for item in result["geometry_info"]
        ]),
        continuation_steps=np.asarray([
            item["continuation_steps"] for item in result["geometry_info"]
        ], int),
        velocity_internal_rate_absolute_residual=np.asarray([
            item["internal_rate_absolute_residual"] for item in result["velocity_info"]
        ]),
        velocity_internal_rate_relative_residual=np.asarray([
            item["internal_rate_relative_residual"] for item in result["velocity_info"]
        ]),
        center_of_mass_velocity_angstrom_per_fs=np.asarray([
            item["center_of_mass_velocity_angstrom_per_fs"]
            for item in result["velocity_info"]
        ]),
        angular_momentum_au=np.asarray([
            item["angular_momentum_au"] for item in result["velocity_info"]
        ]),
        independent_modal_kinetic_energy_hartree=np.asarray([
            item["independent_modal_kinetic_energy_hartree"]
            for item in result["velocity_info"]
        ]),
        cartesian_kinetic_energy_hartree=np.asarray([
            item["cartesian_kinetic_energy_hartree"]
            for item in result["velocity_info"]
        ]),
        maximum_local_metric_coupling=np.asarray([
            item["maximum_local_metric_coupling"] for item in result["velocity_info"]
        ]),
    )

    rows = []
    for mode_index, distribution in enumerate(distributions):
        # g1 is deliberately fixed at equilibrium and is not an independent
        # CDF draw.  Exclude it from the sampling histogram and KS statistic.
        # With NSAMPLES=1, retain it only to keep the diagnostic code defined.
        all_samples = result["s_bohr"][:, mode_index]
        samples = all_samples[1:] if all_samples.size > 1 else all_samples
        theoretical_grid = distribution["conditional_grid_bohr"]
        theoretical_density = distribution["conditional_density_per_bohr"]
        if ensemble in (HARMONIC, HARMONIC_CARTESIAN):
            theoretical_mean = 0.0
            theoretical_std = float(distribution["harmonic_sigma_s_bohr"])
        else:
            theoretical_mean = float(np.trapezoid(theoretical_grid * theoretical_density, theoretical_grid))
            theoretical_second = float(np.trapezoid(theoretical_grid**2 * theoretical_density, theoretical_grid))
            theoretical_std = math.sqrt(max(0.0, theoretical_second - theoretical_mean**2))
        ks_result = kstest(samples, conditional_cdf_function(distribution))
        momentum_samples = result["p_s_au"][:, mode_index]
        momentum_grid = distribution["momentum_grid_au"]
        momentum_density = distribution["momentum_densities"][ensemble]
        if ensemble in (HARMONIC, HARMONIC_CARTESIAN):
            momentum_mean = 0.0
            momentum_std = float(distribution["harmonic_sigma_p_au"])
        else:
            momentum_mean = float(np.trapezoid(
                momentum_grid * momentum_density, momentum_grid,
            ))
            momentum_second = float(np.trapezoid(
                momentum_grid**2 * momentum_density, momentum_grid,
            ))
            momentum_std = math.sqrt(max(0.0, momentum_second - momentum_mean**2))
        momentum_ks = kstest(momentum_samples, momentum_cdf_function(distribution))
        rows.append(dict(
            ensemble=ensemble, mode=distribution["mode"],
            total_geometries=len(all_samples), random_samples=len(samples),
            probability_source=distribution["probability_source"],
            sampling_s_min_bohr=f"{distribution['support_low_bohr']:.12e}",
            sampling_s_max_bohr=f"{distribution['support_high_bohr']:.12e}",
            geometry_path_s_min_bohr=f"{distribution['geometry_path_low_bohr']:.12e}",
            geometry_path_s_max_bohr=f"{distribution['geometry_path_high_bohr']:.12e}",
            fit_data_s_min_bohr=f"{distribution['fit_low_bohr']:.12e}",
            fit_data_s_max_bohr=f"{distribution['fit_high_bohr']:.12e}",
            omitted_tail_probability=f"{distribution['tail_probability']:.12e}",
            sample_mean_s_bohr=f"{np.mean(samples):.12e}",
            theoretical_mean_s_bohr=f"{theoretical_mean:.12e}",
            sample_std_s_bohr=f"{np.std(samples, ddof=1 if samples.size > 1 else 0):.12e}",
            theoretical_std_s_bohr=f"{theoretical_std:.12e}",
            sampled_s_min_bohr=f"{np.min(samples):.12e}",
            sampled_s_max_bohr=f"{np.max(samples):.12e}",
            sampled_Q_min_angstrom=f"{np.min(result['Q_angstrom'][:, mode_index]):.12e}",
            sampled_Q_max_angstrom=f"{np.max(result['Q_angstrom'][:, mode_index]):.12e}",
            ks_D=f"{ks_result.statistic:.12e}", ks_pvalue=f"{ks_result.pvalue:.12e}",
            momentum_source=distribution["momentum_source"],
            momentum_random_samples=len(momentum_samples),
            sample_mean_p_s_au=f"{np.mean(momentum_samples):.12e}",
            theoretical_mean_p_s_au=f"{momentum_mean:.12e}",
            sample_std_p_s_au=f"{np.std(momentum_samples, ddof=1 if momentum_samples.size > 1 else 0):.12e}",
            theoretical_std_p_s_au=f"{momentum_std:.12e}",
            sampled_p_s_min_au=f"{np.min(momentum_samples):.12e}",
            sampled_p_s_max_au=f"{np.max(momentum_samples):.12e}",
            momentum_ks_D=f"{momentum_ks.statistic:.12e}",
            momentum_ks_pvalue=f"{momentum_ks.pvalue:.12e}",
        ))

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(
            samples, bins=int(config.A7_HISTOGRAM_BINS), density=True,
            alpha=0.60, label=r"randomly sampled $s$",
        )
        ax.plot(theoretical_grid, theoretical_density, "r-", linewidth=2.0, label="conditional thermal density")
        if np.isfinite(distribution["support_low_bohr"]):
            ax.axvline(distribution["support_low_bohr"], color="0.5", linestyle=":", linewidth=0.9)
        if np.isfinite(distribution["support_high_bohr"]):
            ax.axvline(
                distribution["support_high_bohr"], color="0.5", linestyle=":", linewidth=0.9,
                label="sampling limits",
            )
        ax.text(
            0.04, 0.96,
            f"KS D={ks_result.statistic:.3e}\np={ks_result.pvalue:.3f}\n"
            f"omitted tail={distribution['tail_probability']:.3e}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.set(
            xlabel=r"constant-mass coordinate $s$ (bohr)", ylabel="Probability density",
            title=f"Mode {distribution['mode']} {ensemble} sampling",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(histogram_dir / f"mode{distribution['mode']}_distribution.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.hist(
            momentum_samples, bins=int(config.A7_HISTOGRAM_BINS), density=True,
            alpha=0.60, label=r"independently sampled $p_s$",
        )
        ax.plot(
            momentum_grid, momentum_density, "r-", linewidth=2.0,
            label=r"thermal momentum marginal $\rho(p_s,T)$",
        )
        visible = momentum_density > max(
            float(np.max(momentum_density)) * 1.0e-7, 1.0e-15,
        )
        if np.any(visible):
            visible_p = momentum_grid[visible]
            margin = max(0.1, 0.08 * (visible_p[-1] - visible_p[0]))
            ax.set_xlim(visible_p[0] - margin, visible_p[-1] + margin)
        ax.text(
            0.04, 0.96,
            f"KS D={momentum_ks.statistic:.3e}\np={momentum_ks.pvalue:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )
        ax.set(
            xlabel=r"constant-mass momentum $p_s$ (a.u.; $\hbar/a_0$)",
            ylabel="Probability density per a.u.",
            title=f"Mode {distribution['mode']} {ensemble} momentum sampling",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            histogram_dir / f"mode{distribution['mode']}_momentum_distribution.png",
            dpi=180,
        )
        plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate ensembles using the Cartesian or curvilinear a1 path model."
    )
    parser.add_argument("--scan-root", type=Path, default=config.SCAN_ROOT)
    parser.add_argument("--output", type=Path, default=config.ENSEMBLE_ROOT)
    parser.add_argument("--samples", type=int, default=config.NSAMPLES)
    parser.add_argument("--seed", type=int, default=config.A7_RANDOM_SEED)
    parser.add_argument(
        "--temperature", type=float, default=config.TEMP,
        help="Requested temperature in K (default config.TEMP); must match saved a6 data",
    )
    parser.add_argument(
        "--workers", type=int, default=getattr(config, "A7_WORKERS", 1),
        help="Independent geometry-reconstruction processes; default A7_WORKERS/1",
    )
    parser.add_argument("--modes", nargs="+", help="Explicit subset for diagnostic ensembles")
    parser.add_argument(
        "--ensemble", choices=("both", ANHARMONIC, HARMONIC), default="both",
        help="Choose distributions; curvilinear harmonic output also includes a Cartesian harmonic reference",
    )
    parser.add_argument(
        "--coordinates", choices=("scan", "cartesian", "curvilinear"), default="scan",
        help="Geometry reconstruction only; scan preserves the a1 model. Densities are unchanged.",
    )
    parser.add_argument(
        "--allow-truncation", action="store_true",
        help="Renormalize on the validated support even when configured tail limits are exceeded",
    )
    args = parser.parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be a positive integer.")
    if args.workers < 1:
        raise ValueError("--workers must be a positive integer.")
    if config.A7_CONTINUATION_STEP_ANG <= 0 or config.A7_MAX_ATTEMPTS_FACTOR < 1:
        raise ValueError("Invalid a7 continuation/attempt settings in config.py.")

    requested_output = args.output.resolve()
    if requested_output.exists() and (
        not requested_output.is_dir() or any(requested_output.iterdir())
    ):
        raise FileExistsError(f"Use a new or empty a7 output directory: {requested_output}")

    scan_root = args.scan_root.resolve()
    verified_temperature = validated_a6_temperature(scan_root, args.temperature)
    print(
        f"a7: requested temperature={args.temperature:.12g} K; "
        f"verified a6 temperature={verified_temperature:.12g} K", flush=True,
    )
    manifest = read_scan_manifest(scan_root)
    reference = load_reference_data(scan_root)
    coordinate_system = str(
        manifest.get("coordinate_system", "curvilinear")
    ).strip().lower()
    if coordinate_system not in {"curvilinear", "cartesian"}:
        raise ValueError(f"Unknown a1 coordinate system: {coordinate_system!r}.")
    scan_coordinate_system = coordinate_system
    coordinate_system = scan_coordinate_system if args.coordinates == "scan" else args.coordinates
    reference["coordinate_system"] = coordinate_system
    reference["density_coordinate_system"] = scan_coordinate_system
    if args.ensemble in ("both", ANHARMONIC) and coordinate_system != scan_coordinate_system:
        raise ValueError("Anharmonic reconstruction must match the a1 PES coordinates. "
                         "Use the matching a1--a6 scan tree; harmonic-only runs may choose either.")
    if manifest["n_atoms"] != reference["n_atoms"] or manifest["n_modes"] != reference["n_modes"]:
        raise ValueError("The a1 manifest and reference_data.npz dimensions disagree.")
    available = [int(value) for value in manifest["selected_modes"]]
    if args.modes:
        chosen = selected_modes(args.modes, reference["n_modes"])
    else:
        chosen = available
        expected = list(range(1, reference["n_modes"] + 1))
        if chosen != expected:
            raise ValueError(
                "A production ensemble requires every vibrational mode.  Complete a1--a6 "
                "for all modes, or pass --modes explicitly for a diagnostic partial ensemble."
            )
    if not set(chosen) <= set(available):
        raise ValueError("Requested modes are absent from the a1 paths.")
    mode_data = [load_mode_data(scan_root, mode, reference, args.temperature) for mode in chosen]
    temperatures = np.asarray([item["temperature_K"] for item in mode_data])
    if not np.allclose(temperatures, temperatures[0], atol=1.0e-10, rtol=0):
        raise ValueError("The a6 mode distributions were generated at different temperatures.")

    path_validation_rows = []
    if config.A7_VALIDATE_SINGLE_MODE_PATHS and coordinate_system != "cartesian":
        print("a7: validating endpoint reconstruction against the saved a1 paths ...", flush=True)
        source_reference = dict(reference, coordinate_system=scan_coordinate_system)
        path_validation_rows = validate_saved_single_mode_paths(mode_data, source_reference)

    ensembles = [ANHARMONIC, HARMONIC] if args.ensemble == "both" else [args.ensemble]
    print(f"a7: distributions={','.join(ensembles)}; reconstruction={coordinate_system}; "
          f"source PES coordinates={scan_coordinate_system}", flush=True)
    prepared = {
        ensemble: [prepare_conditional_distribution(item, ensemble) for item in mode_data]
        for ensemble in ensembles
    }
    references = {ensemble: reference for ensemble in ensembles}
    if coordinate_system == "curvilinear" and HARMONIC in ensembles:
        ensembles.append(HARMONIC_CARTESIAN)
        # Reuse the exact analytic distributions, but not the accepted curved
        # geometries: doing so would inherit curvilinear rejection bias.
        prepared[HARMONIC_CARTESIAN] = []
        for distribution in prepared[HARMONIC]:
            copied = dict(distribution)
            copied["momentum_densities"] = dict(distribution["momentum_densities"])
            copied["momentum_densities"][HARMONIC_CARTESIAN] = distribution["momentum_densities"][HARMONIC]
            prepared[HARMONIC_CARTESIAN].append(copied)
        references[HARMONIC_CARTESIAN] = dict(reference, coordinate_system="cartesian")
    for ensemble, distributions in prepared.items():
        if ensemble in (HARMONIC, HARMONIC_CARTESIAN):
            continue
        per_mode_max = max(item["tail_probability"] for item in distributions)
        joint_tail = 1.0 - float(np.prod([item["retained_probability"] for item in distributions]))
        if not args.allow_truncation and (
            per_mode_max > config.A7_MAX_TAIL_PROBABILITY_PER_MODE
            or joint_tail > config.A7_MAX_JOINT_TAIL_PROBABILITY
        ):
            worst = max(distributions, key=lambda item: item["tail_probability"])
            raise RuntimeError(
                f"{ensemble}: probability outside the validated sampling support is too large "
                f"(mode {worst['mode']} tail={worst['tail_probability']:.3e}; "
                f"joint tail={joint_tail:.3e}). Extend the affected a1 path and/or a5 "
                "data interval, then rerun a5/a6. Use --allow-truncation only for a "
                "deliberate approximation."
            )

    results = {}
    for ensemble in ensembles:
        workers = 1 if references[ensemble]["coordinate_system"] == "cartesian" else args.workers
        print(
            f"a7: reconstructing {args.samples} {ensemble} geometries "
            f"with {workers} worker(s) ...", flush=True,
        )
        results[ensemble] = generate_ensemble(
            prepared[ensemble], references[ensemble], args.samples, args.seed, workers,
        )

    output_root = ensure_new_directory(requested_output)
    source_log_name = Path(manifest.get("source_log", "gaussian.log")).stem
    mode_rows = []
    for ensemble in ensembles:
        mode_rows.extend(save_ensemble(
            output_root, ensemble, prepared[ensemble], results[ensemble], references[ensemble],
            source_log_name,
        ))
    with (output_root / config.A7_MODE_DIAGNOSTICS_FILE).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=mode_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(mode_rows)

    geometry_rows = []
    for ensemble in ensembles:
        result = results[ensemble]
        info = result["geometry_info"]
        joint_tail = 1.0 - float(np.prod([
            item["retained_probability"] for item in prepared[ensemble]
        ]))
        geometry_rows.append(dict(
            ensemble=ensemble, requested_samples=args.samples,
            coordinate_system=references[ensemble]["coordinate_system"],
            jacobian_rank_ratio_scope=(
                "local_sampled_geometry"
                if references[ensemble]["coordinate_system"] == "curvilinear"
                else "equilibrium_reference_only"
            ),
            equilibrium_geometry_as_g1=True,
            randomly_sampled_geometries=max(0, args.samples - 1),
            accepted_samples=len(result["geometries_angstrom"]), attempts=result["attempts"],
            rejected_attempts=result["attempts"] - len(result["geometries_angstrom"]),
            maximum_internal_residual=f"{max(item['residual'] for item in info):.12e}",
            minimum_jacobian_singular_ratio=f"{min(item['rank_ratio'] for item in info):.12e}",
            minimum_interatomic_distance_angstrom=f"{min(item['minimum_pair_distance_ang'] for item in info):.12e}",
            minimum_bond_length_ratio=f"{min(item['minimum_bond_length_ratio'] for item in info):.12e}",
            maximum_bond_length_ratio=f"{max(item['maximum_bond_length_ratio'] for item in info):.12e}",
            maximum_bond_length_change_percent=f"{max(item['maximum_bond_length_change_percent'] for item in info):.8f}",
            maximum_continuation_steps=max(item["continuation_steps"] for item in info),
            maximum_backtransform_iterations=max(item["backtransform_iterations"] for item in info),
            maximum_velocity_internal_rate_relative_residual=(
                f"{max(item['internal_rate_relative_residual'] for item in result['velocity_info']):.12e}"
            ),
            maximum_center_of_mass_speed_angstrom_per_fs=(
                f"{max(np.linalg.norm(item['center_of_mass_velocity_angstrom_per_fs']) for item in result['velocity_info']):.12e}"
            ),
            maximum_angular_momentum_norm_au=(
                f"{max(np.linalg.norm(item['angular_momentum_au']) for item in result['velocity_info']):.12e}"
            ),
            mean_independent_modal_kinetic_energy_hartree=(
                f"{np.mean([item['independent_modal_kinetic_energy_hartree'] for item in result['velocity_info']]):.12e}"
            ),
            mean_cartesian_kinetic_energy_hartree=(
                f"{np.mean([item['cartesian_kinetic_energy_hartree'] for item in result['velocity_info']]):.12e}"
            ),
            maximum_local_metric_coupling=(
                f"{max(item['maximum_local_metric_coupling'] for item in result['velocity_info']):.12e}"
            ),
            joint_probability_conditioned_away=f"{joint_tail:.12e}",
            rejection_reasons="; ".join(
                f"{count} x {reason}" for reason, count in result["rejection_reasons"].items()
            ) or "none",
        ))
    with (output_root / config.A7_ENSEMBLE_DIAGNOSTICS_FILE).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=geometry_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(geometry_rows)
    if path_validation_rows:
        with (output_root / config.A7_PATH_VALIDATION_FILE).open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=path_validation_rows[0].keys(), delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(path_validation_rows)

    source_files = [reference["reference_path"], reference["definitions_path"]]
    for item in mode_data:
        source_files.extend(item["source_files"])
    source_files = sorted(set(Path(path) for path in source_files))
    manifest_out = dict(
        stage="curvilinear_internal_coordinate_ensemble",
        scan_root=str(scan_root), selected_modes=chosen,
        coordinate_system=coordinate_system,
        density_coordinate_system=scan_coordinate_system,
        reconstruction_selection=args.coordinates,
        reference_is_linear=bool(reference["reference_is_linear"]),
        full_dimensional_ensemble=(chosen == list(range(1, reference["n_modes"] + 1))),
        n_samples=args.samples, random_seed=args.seed, temperature_K=float(temperatures[0]),
        requested_temperature_K=float(args.temperature),
        reconstruction_workers=args.workers,
        equilibrium_geometry_as_g1=True,
        randomly_sampled_geometries=max(0, args.samples - 1),
        ensembles=ensembles,
        ensemble_coordinate_systems={key: value["coordinate_system"] for key, value in references.items()},
        ensemble_workers={key: (1 if value["coordinate_system"] == "cartesian" else args.workers)
                          for key, value in references.items()},
        probability_model=(
            "positive separable product over modes and, within every mode, "
            "F_i(s_i,p_i;T)=rho_i(s_i,T)*rho_i(p_i,T); no anharmonic Wigner "
            "quasiprobability"
        ),
        sampling=(
            "anharmonic: independent inverse numerical CDFs with s -> inverse s(Q); "
            "harmonic: independent exact Gaussian inverse CDFs on the full real line "
            "with Q=s; momenta sampled independently in both ensembles"
        ),
        independent_position_momentum_sampling=True,
        coordinate_random_stream_preserved_from_position_only_a7=True,
        equilibrium_geometry_g1_has_independently_sampled_momentum=True,
        geometry_reconstruction=(
            (
                "xi_target=sum_i Q_i*l_i in the fixed equilibrium independent-internal "
                "basis; one simultaneous iterative internal-to-Cartesian back-transformation"
            )
            if coordinate_system == "curvilinear"
            else "analytic simultaneous rectilinear displacement R=R0+sum_i Q_i*d_i"
        ),
        energetic_mode_coupling_included=False,
        velocity_reconstruction=(
            (
                "ds_i/dt=p_s_i/mu_i(0); anharmonic "
                "dQ_i/ds_i=sqrt(mu_i(0)/mu_i(Q_i)); harmonic dQ_i/ds_i=1; "
                "local dR/dQ_i from the Wilson-B mass-metric pseudoinverse; summed "
                "Cartesian velocity"
            )
            if coordinate_system == "curvilinear"
            else (
                "ds_i/dt=p_s_i/mu_i(0); harmonic Q=s; anharmonic inverse source s(Q) "
                "with source dQ/ds=sqrt(mu0/mu(Q)); constant dR/dQ_i=d_i; "
                "summed Cartesian velocity with existing rigid-motion projection"
            )
        ),
        coordinate_shift_applied=False,
        tail_handling=(
            "anharmonic only: conditional renormalization on the a6 grid intersected with "
            "the a1 path and a5 fit-data interval; harmonic position and momentum use "
            "unbounded analytic Gaussian marginals and have no scan/grid tail truncation"
        ),
        allow_large_truncation=args.allow_truncation,
        single_mode_path_endpoint_validation=(
            dict(
                enabled=True, tolerance_angstrom=config.A7_PATH_RECONSTRUCTION_TOL_ANG,
                maximum_difference_angstrom=max(
                    float(row["maximum_cartesian_difference_angstrom"])
                    for row in path_validation_rows
                ),
            )
            if path_validation_rows else dict(enabled=False)
        ),
        units=dict(
            s_sampling="bohr", Q_and_Cartesian="angstrom", angles="radian",
            modal_momentum="atomic momentum = hbar/bohr = electron_mass*bohr/atomic_time",
            cartesian_momentum="atomic momentum = electron_mass*bohr/atomic_time",
            cartesian_velocity="angstrom/fs",
            internally_saved_cartesian_velocity="bohr/atomic_time",
        ),
        source_sha256={str(path.relative_to(scan_root)): sha256_file(path) for path in source_files},
        outputs={
            ensemble: dict(
                coordinate_system=references[ensemble]["coordinate_system"],
                xyz_directory=OUTPUT_LAYOUT[ensemble][0],
                all_geometries=OUTPUT_LAYOUT[ensemble][2],
                all_velocities=VECTOR_OUTPUT_NAMES[ensemble][0],
                all_momenta=VECTOR_OUTPUT_NAMES[ensemble][1],
                sampled_coordinates=f"sampled_coordinates_{ensemble}.npz",
                histogram_directory=OUTPUT_LAYOUT[ensemble][1],
            )
            for ensemble in ensembles
        },
    )
    (output_root / config.A7_MANIFEST_FILE).write_text(json.dumps(manifest_out, indent=2) + "\n")

    print(f"a7: PASS. Output: {output_root}")
    print(
        "a7: position and momentum marginals were sampled independently; "
        "Cartesian velocities are in angstrom/fs and momenta are in atomic units."
    )
    for row in geometry_rows:
        rank_label = (
            "min local rank ratio"
            if row["coordinate_system"] == "curvilinear"
            else "equilibrium rank ratio"
        )
        print(
            f"{row['ensemble']}: {row['accepted_samples']} geometries; "
            f"rejected={row['rejected_attempts']}; max residual={row['maximum_internal_residual']}; "
            f"{rank_label}={row['minimum_jacobian_singular_ratio']}."
        )


if __name__ == "__main__":
    run_timed_stage("a7", main, config.EXECUTION_TIMING_FILE)
