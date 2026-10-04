#!/usr/bin/env python3
"""a1: build Cartesian or curvilinear mode paths and the mass coordinate s."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import re
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_simpson

import config
from internal_coordinates import (
    backtransform,
    continue_backtransform,
    internal_difference,
    internal_values,
    mass_metric_step,
    read_internals,
    read_masses,
    remove_rigid_velocity,
    rigid_directions,
    vibrational_dof,
    wilson_b,
)
from workflow_utils import (
    ELEMENT_SYMBOLS,
    ensure_new_directory,
    load_xyz,
    q_label,
    run_timed_stage,
    selected_modes,
    sha256_file,
    write_xyz,
)


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?"


def normalize_coordinate_system(value):
    """Return the supported lower-case coordinate-system name."""
    name = str(value).strip().lower()
    if name not in {"curvilinear", "cartesian"}:
        raise ValueError("The coordinate system must be 'curvilinear' or 'cartesian'.")
    return name


def normalize_opt_freq_software(value):
    """Return a supported optimization/frequency output format."""
    name = str(value).strip().lower()
    if name not in {"auto", "gaussian", "orca"}:
        raise ValueError(
            "Optimization/frequency software must be 'auto', 'gaussian', or 'orca'."
        )
    return name


def detect_opt_freq_software(text):
    """Detect Gaussian or ORCA from stable section/termination markers."""
    is_gaussian = "Harmonic frequencies (cm" in text and "Gaussian" in text
    is_orca = (
        "VIBRATIONAL FREQUENCIES" in text
        and "NORMAL MODES" in text
        and "Program Version" in text
    )
    if is_gaussian == is_orca:
        raise ValueError(
            "Could not unambiguously detect a Gaussian or ORCA optimization/frequency output. "
            "Set OPT_FREQ_SOFTWARE explicitly."
        )
    return "gaussian" if is_gaussian else "orca"


def _parse_gaussian_harmonic_output(path: Path):
    """Read one linear or nonlinear Gaussian frequency job without reordering modes."""
    path = Path(path)
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    if text.count("Harmonic frequencies (cm") != 1:
        raise ValueError("a1 requires exactly one Gaussian harmonic analysis.")
    harmonic_line = next(i for i, line in enumerate(lines) if "Harmonic frequencies (cm" in line)
    if "Error termination" in text or "Normal termination of Gaussian" not in text:
        raise ValueError("The Gaussian optimization/frequency calculation did not terminate normally.")

    orientation_indices = [
        i for i, line in enumerate(lines[:harmonic_line])
        if "Standard orientation:" in line
    ]
    if not orientation_indices:
        orientation_indices = [
            i for i, line in enumerate(lines[:harmonic_line])
            if "Input orientation:" in line
        ]
    if not orientation_indices:
        raise ValueError("No Cartesian orientation found before the harmonic analysis.")
    start = orientation_indices[-1] + 5
    atoms, coords = [], []
    for line in lines[start:]:
        if "---" in line:
            break
        fields = line.split()
        if len(fields) < 6:
            raise ValueError("Incomplete Gaussian orientation table.")
        atoms.append(int(fields[1]))
        coords.append([float(fields[3]), float(fields[4]), float(fields[5])])
    coords = np.asarray(coords, float)
    if not len(atoms) or coords.shape != (len(atoms), 3):
        raise ValueError("Invalid optimized Cartesian geometry.")

    frequencies, gaussian_masses, modes = [], [], []
    i = harmonic_line
    while i < len(lines):
        line = lines[i]
        if "Frequencies --" not in line:
            i += 1
            continue
        block_freq = [float(x.replace("D", "E")) for x in re.findall(FLOAT_RE, line.split("--", 1)[1])]
        nblock = len(block_freq)
        frequencies.extend(block_freq)
        j = i
        while j < len(lines) and "Red. masses --" not in lines[j]:
            j += 1
        if j == len(lines):
            raise ValueError("Missing reduced masses in a Gaussian frequency block.")
        block_mass = [float(x.replace("D", "E")) for x in re.findall(FLOAT_RE, lines[j].split("--", 1)[1])]
        if len(block_mass) != nblock:
            raise ValueError("Frequency/reduced-mass block size mismatch.")
        gaussian_masses.extend(block_mass)
        while j < len(lines) and "Atom  AN" not in lines[j]:
            j += 1
        if j == len(lines):
            raise ValueError("Missing normal-vector table.")
        block = []
        for atom_line in lines[j + 1:j + 1 + len(atoms)]:
            fields = atom_line.split()
            needed = 2 + 3 * nblock
            if len(fields) < needed:
                raise ValueError("Incomplete normal-vector table.")
            block.append([float(x.replace("D", "E")) for x in fields[2:needed]])
        block = np.asarray(block)
        for column in range(nblock):
            modes.append(block[:, 3 * column:3 * (column + 1)])
        i = j + 1 + len(atoms)

    modes = np.asarray(modes, float)
    frequencies = np.asarray(frequencies, float)
    gaussian_masses = np.asarray(gaussian_masses, float)
    expected = vibrational_dof(coords)
    if modes.shape != (expected, len(atoms), 3):
        geometry_type = "linear" if expected == 3 * len(atoms) - 5 else "nonlinear"
        raise ValueError(
            f"Expected {expected} modes for the detected {geometry_type} geometry, "
            f"read {modes.shape[0]}."
        )
    if frequencies.shape != (expected,) or gaussian_masses.shape != (expected,):
        raise ValueError("Incomplete frequencies or Gaussian reduced masses.")
    masses = read_masses(text, len(atoms))
    if any(z < 1 or z >= len(ELEMENT_SYMBOLS) for z in atoms):
        raise ValueError("Unsupported atomic number.")
    symbols = [ELEMENT_SYMBOLS[z] for z in atoms]
    return text, np.asarray(atoms, int), symbols, coords, masses, frequencies, gaussian_masses, modes


def _orca_geometry_and_masses(lines, frequency_line):
    """Read the final ORCA angstrom geometry and matching atomic-mass table."""
    geometry_headers = [
        index for index, line in enumerate(lines[:frequency_line])
        if line.strip() == "CARTESIAN COORDINATES (ANGSTROEM)"
    ]
    if not geometry_headers:
        raise ValueError("No ORCA Cartesian geometry was found before the frequencies.")
    atoms, symbols, coords = [], [], []
    symbol_lookup = {symbol.lower(): (number, symbol) for number, symbol in enumerate(ELEMENT_SYMBOLS)}
    for line in lines[geometry_headers[-1] + 2:]:
        fields = line.split()
        if not fields:
            break
        if len(fields) < 4:
            raise ValueError("Incomplete ORCA Cartesian-coordinate table.")
        key = fields[0].rstrip(":").lower()
        if key not in symbol_lookup or key == "x":
            raise ValueError(f"Unsupported ORCA atom label {fields[0]!r}.")
        atomic_number, symbol = symbol_lookup[key]
        try:
            xyz = [float(value.replace("D", "E")) for value in fields[-3:]]
        except ValueError as exc:
            raise ValueError("Invalid number in the ORCA Cartesian-coordinate table.") from exc
        atoms.append(atomic_number)
        symbols.append(symbol)
        coords.append(xyz)
    coords = np.asarray(coords, float)
    if not atoms or coords.shape != (len(atoms), 3) or not np.isfinite(coords).all():
        raise ValueError("Invalid optimized ORCA Cartesian geometry.")

    mass_headers = [
        index for index, line in enumerate(lines[:frequency_line])
        if line.strip() == "CARTESIAN COORDINATES (A.U.)"
    ]
    if not mass_headers:
        raise ValueError("No ORCA atomic-mass table was found before the frequencies.")
    found = {}
    for line in lines[mass_headers[-1] + 1:]:
        fields = line.split()
        if not fields:
            if found:
                break
            continue
        if len(fields) < 8 or not fields[0].isdigit():
            continue
        atom_index = int(fields[0])
        try:
            mass = float(fields[4].replace("D", "E"))
        except ValueError as exc:
            raise ValueError("Invalid mass in the ORCA Cartesian-coordinate table.") from exc
        found[atom_index] = (fields[1].rstrip(":"), mass)
        if len(found) == len(atoms):
            break
    if sorted(found) != list(range(len(atoms))):
        raise ValueError("Could not read one ORCA mass for every atom.")
    masses = np.asarray([found[index][1] for index in range(len(atoms))], float)
    if not np.isfinite(masses).all() or np.any(masses <= 0.0):
        raise ValueError("ORCA atomic masses must be positive and finite.")
    for index, symbol in enumerate(symbols):
        if found[index][0].lower() != symbol.lower():
            raise ValueError("ORCA geometry and mass tables use different atom orders.")
    return np.asarray(atoms, int), symbols, coords, masses


def _orca_normal_mode_matrix(lines, normal_line, ncart):
    """Read ORCA's block-column normal-mode matrix, including rigid modes."""
    end = next(
        (index for index in range(normal_line + 1, len(lines))
         if lines[index].strip() in {"IR SPECTRUM", "RAMAN SPECTRUM"}),
        len(lines),
    )
    matrix = np.full((ncart, ncart), np.nan, float)
    index = normal_line + 1
    while index < end:
        header = lines[index].split()
        if header and all(re.fullmatch(r"\d+", value) for value in header):
            columns = [int(value) for value in header]
            if columns and all(0 <= column < ncart for column in columns):
                rows = []
                cursor = index + 1
                for expected_row in range(ncart):
                    if cursor >= end:
                        rows = []
                        break
                    fields = lines[cursor].split()
                    if (
                        len(fields) != len(columns) + 1
                        or not fields[0].isdigit()
                        or int(fields[0]) != expected_row
                    ):
                        rows = []
                        break
                    try:
                        rows.append([
                            float(value.replace("D", "E")) for value in fields[1:]
                        ])
                    except ValueError:
                        rows = []
                        break
                    cursor += 1
                if rows:
                    matrix[:, columns] = np.asarray(rows, float)
                    index = cursor
                    continue
        index += 1
    if not np.isfinite(matrix).all():
        missing = np.flatnonzero(~np.isfinite(matrix).all(axis=0)).tolist()
        raise ValueError(f"Incomplete ORCA normal-mode table; missing columns {missing}.")
    return matrix


def _parse_orca_harmonic_output(path: Path):
    """Read a standard ORCA optimization/frequency text output."""
    path = Path(path)
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    frequency_headers = [
        index for index, line in enumerate(lines)
        if line.strip() == "VIBRATIONAL FREQUENCIES"
    ]
    normal_headers = [
        index for index, line in enumerate(lines)
        if line.strip() == "NORMAL MODES"
    ]
    if len(frequency_headers) != 1 or len(normal_headers) != 1:
        raise ValueError("a1 requires exactly one ORCA vibrational analysis.")
    if "ORCA TERMINATED NORMALLY" not in text:
        raise ValueError("The ORCA optimization/frequency calculation did not terminate normally.")
    frequency_line = frequency_headers[0]
    normal_line = normal_headers[0]
    if normal_line <= frequency_line:
        raise ValueError("The ORCA normal modes precede the frequency table.")
    atoms, symbols, coords, masses = _orca_geometry_and_masses(lines, frequency_line)
    ncart = 3 * len(atoms)
    expected = vibrational_dof(coords)
    rigid_count = ncart - expected

    frequency_pattern = re.compile(
        r"^\s*(\d+):\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?)\s+cm\*\*-1"
    )
    indexed_frequencies = {}
    for line in lines[frequency_line + 1:normal_line]:
        match = frequency_pattern.match(line)
        if match:
            indexed_frequencies[int(match.group(1))] = float(
                match.group(2).replace("D", "E").replace("d", "e")
            )
    if sorted(indexed_frequencies) != list(range(ncart)):
        raise ValueError(
            f"Expected ORCA frequency indices 0 through {ncart - 1}; "
            f"read {len(indexed_frequencies)} entries."
        )
    full_modes = _orca_normal_mode_matrix(lines, normal_line, ncart)
    vibrational_columns = list(range(rigid_count, ncart))
    frequencies = np.asarray(
        [indexed_frequencies[index] for index in vibrational_columns], float,
    )
    modes = full_modes[:, vibrational_columns].T.reshape(expected, len(atoms), 3)
    norms = np.linalg.norm(modes.reshape(expected, -1), axis=1)
    if np.max(np.abs(norms - 1.0)) > 5.0e-3:
        raise ValueError("ORCA printed vibrational vectors are not normalized as expected.")
    # ORCA prints Cartesian displacement vectors normalized to unit Euclidean
    # norm after the 1/sqrt(m) transformation. For R=R0+Q*d, the matching
    # modal mass is therefore sum_a m_a |d_a|^2.
    reduced_masses = np.sum(masses[None, :, None] * modes**2, axis=(1, 2))
    return (
        text, atoms, symbols, coords, masses, frequencies,
        reduced_masses, modes,
    )


def parse_harmonic_log(path: Path, software="auto", return_software=False):
    """Read Gaussian or ORCA equilibrium geometry, masses, frequencies and modes."""
    path = Path(path)
    requested = normalize_opt_freq_software(software)
    text = path.read_text(errors="replace")
    resolved = detect_opt_freq_software(text) if requested == "auto" else requested
    if requested != "auto" and detect_opt_freq_software(text) != requested:
        raise ValueError(
            f"The configured optimization/frequency software is {requested}, "
            f"but {path.name} has the other supported output format."
        )
    result = (
        _parse_gaussian_harmonic_output(path)
        if resolved == "gaussian" else _parse_orca_harmonic_output(path)
    )
    return (*result, resolved) if return_software else result


def build_dense_q_grid(energy_q, maximum_step):
    energy_q = np.unique(np.asarray(energy_q, float))
    if not np.isfinite(energy_q).all() or not np.any(np.isclose(energy_q, 0.0, atol=1e-12)):
        raise ValueError("Q_ENERGY_ANG must contain finite, unique points including zero.")
    if maximum_step <= 0 or not np.isfinite(maximum_step):
        raise ValueError("PATH_Q_STEP_ANG must be positive and finite.")
    knots = sorted(set(np.round(np.append(energy_q, 0.0), 12)))
    dense = [knots[0]]
    for left, right in zip(knots[:-1], knots[1:]):
        count = max(1, int(np.ceil((right - left) / maximum_step - 1e-12)))
        dense.extend(np.linspace(left, right, count + 1)[1:])
    dense = np.asarray(dense, float)
    if np.max(np.diff(dense)) > maximum_step * (1 + 1e-10):
        raise RuntimeError("Internal error while constructing the dense Q grid.")
    return energy_q, dense


def energy_grid_for_mode(mode_number, maximum_step):
    """Return the electronic and dense Q grids configured for one mode."""
    overrides = getattr(config, "Q_ENERGY_ANG_BY_MODE", {})
    if not isinstance(overrides, dict):
        raise ValueError("Q_ENERGY_ANG_BY_MODE must be a dictionary keyed by mode number.")
    raw = overrides.get(mode_number, config.Q_ENERGY_ANG)
    return build_dense_q_grid(raw, maximum_step)


def mass_coordinate(q, mu, mu0):
    """Integrate ds/dQ=sqrt(mu(Q)/mu(0)) by cumulative Simpson quadrature."""
    q, mu = np.asarray(q, float), np.asarray(mu, float)
    if np.any(np.diff(q) <= 0) or np.any(mu <= 0) or mu0 <= 0:
        raise ValueError("A monotonic Q grid and positive masses are required.")
    integrand = np.sqrt(mu / mu0)
    s = cumulative_simpson(integrand, x=q, initial=0.0)
    zero = np.flatnonzero(np.isclose(q, 0.0, atol=1e-12))
    if len(zero) != 1:
        raise ValueError("The dense Q grid must contain zero exactly once.")
    s = s - s[zero[0]]
    if not np.all(np.diff(s) > 0):
        raise ValueError("Mass-coordinate quadrature is not monotonic; reduce PATH_Q_STEP_ANG.")
    return s


def validate_printed_internals(definitions, equilibrium_values, software="gaussian"):
    """Check a printed internal table without assuming its finite-L frame convention.

    R, A and D values are directly comparable.  A paired linear bend L has a
    convention-dependent orientation in its two-dimensional transverse plane,
    especially when the source program selects that plane with a real fourth
    atom. The
    invariant comparison is therefore the norm of the two deviations from 180
    degrees, not either signed component separately.
    """
    equilibrium_values = np.asarray(equilibrium_values, float)
    printed = np.asarray([
        item.printed_value if item.kind == "R" else np.deg2rad(item.printed_value)
        for item in definitions
    ])
    ordinary_indices = [
        index for index, item in enumerate(definitions) if item.kind != "L"
    ]
    if ordinary_indices:
        ordinary_error = internal_difference(
            equilibrium_values, printed, definitions,
        )[ordinary_indices]
        display_error = np.asarray([
            value if definitions[index].kind == "R" else np.rad2deg(value)
            for value, index in zip(ordinary_error, ordinary_indices)
        ])
        tolerances = np.asarray([
            5.0e-4
            if definitions[index].kind == "R" or software == "gaussian"
            else 1.1e-2
            for index in ordinary_indices
        ])
        if np.any(np.abs(display_error) > tolerances):
            raise ValueError(
                f"{str(software).upper()} internal table and equilibrium geometry disagree."
            )

    linear_groups = {}
    for index, item in enumerate(definitions):
        if item.kind == "L":
            linear_groups.setdefault(item.atoms, {})[item.component] = index
    maximum_linear_deviation = np.deg2rad(30.0)
    absolute_tolerance = np.deg2rad(0.5)
    for atoms, indices in linear_groups.items():
        if set(indices) != {-1, -2}:
            raise ValueError("A linear bend is missing one orthogonal component.")
        ordered = [indices[-1], indices[-2]]
        calculated_vector = equilibrium_values[ordered] - np.pi
        printed_vector = printed[ordered] - np.pi
        calculated_magnitude = float(np.linalg.norm(calculated_vector))
        printed_magnitude = float(np.linalg.norm(printed_vector))
        if calculated_magnitude > maximum_linear_deviation:
            atom_label = ",".join(str(index + 1) for index in atoms[:3])
            raise ValueError(
                f"L({atom_label},...) does not describe a near-linear angle."
            )
        tolerance = max(
            absolute_tolerance,
            0.10 * max(calculated_magnitude, printed_magnitude),
        )
        if abs(calculated_magnitude - printed_magnitude) > tolerance:
            atom_label = ",".join(str(index + 1) for index in atoms)
            raise ValueError(
                "Linear-bend pair and equilibrium geometry disagree for "
                f"atoms ({atom_label})."
            )


def reconstruct_mode_path(task):
    """Construct and validate one mode without writing shared output files.

    This top-level worker is intentionally side-effect free so independent
    modes can be evaluated by ``ProcessPoolExecutor``.  All filesystem writes
    remain in the parent process after every requested mode has succeeded.
    """
    (
        mode_number, tangent, cartesian_tangent, r0, definitions, s0, weights,
        basis, masses, back_options, fd, validate_mass, coordinate_system,
        equilibrium_rank_ratio,
    ) = task
    energy_q, dense_q = energy_grid_for_mode(
        mode_number, config.PATH_Q_STEP_ANG,
    )
    if coordinate_system == "cartesian":
        # Conventional rectilinear normal-mode displacement.  The tangent and
        # its effective mass are constant, so no Wilson-B recalculation,
        # nonlinear back-transformation, or numerical quadrature is required.
        cartesian_tangent = np.asarray(cartesian_tangent, float)
        ordered_geometry = (
            np.asarray(r0, float)[None, :, :]
            + dense_q[:, None, None] * cartesian_tangent[None, :, :]
        )
        mu0 = float(np.sum(np.asarray(masses)[:, None] * cartesian_tangent**2))
        if not np.isfinite(mu0) or mu0 <= 0.0:
            raise ValueError(f"Mode {mode_number}: non-positive Cartesian path mass.")
        return dict(
            mode=mode_number,
            coordinate_system=coordinate_system,
            tangent=tangent,
            energy_q=energy_q.copy(),
            q=dense_q.copy(),
            s=dense_q.copy(),
            mu=np.full(dense_q.shape, mu0),
            geometries=ordered_geometry,
            cartesian_tangents=np.repeat(
                cartesian_tangent[None, :, :], dense_q.size, axis=0,
            ),
            # The Cartesian construction is exact and does not impose an
            # internal-coordinate closure condition.  The equilibrium rank is
            # retained only as a reference diagnostic.
            closure=np.zeros(dense_q.shape),
            rank_ratio=np.full(dense_q.shape, equilibrium_rank_ratio),
            iterations=np.zeros(dense_q.shape, dtype=int),
            mass_fd_errors=np.empty((0, 2), dtype=float),
        )

    geometries = {0.0: r0.copy()}
    path_differences = {0.0: np.zeros_like(s0)}
    iteration_count = {0.0: 0}
    for sign in (-1, 1):
        previous = r0.copy()
        previous_difference = np.zeros_like(s0)
        for q in sorted((x for x in dense_q if sign * x > 1e-14), key=abs):
            try:
                geometry, difference, iterations, _, _ = continue_backtransform(
                    q * tangent, r0, definitions, s0, weights, basis, masses,
                    initial=previous, initial_difference=previous_difference,
                    rank_rtol=config.INTERNAL_RANK_RTOL, **back_options,
                )
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(f"Mode {mode_number}, Q={q:g}: {exc}") from exc
            key = round(float(q), 12)
            geometries[key] = geometry
            path_differences[key] = difference
            iteration_count[key] = iterations
            previous = geometry
            previous_difference = difference

    ordered_geometry = np.asarray([
        geometries[round(float(q), 12)] for q in dense_q
    ])
    mus, closures, rank_ratios, tangents = [], [], [], []
    for q, geometry in zip(dense_q, ordered_geometry):
        difference = path_differences[round(float(q), 12)]
        closure = float(np.linalg.norm(
            basis.T @ (weights * difference) - q * tangent, np.inf,
        ))
        j = basis.T @ (
            weights[:, None] * wilson_b(geometry, definitions, fd / 2)
        )
        current_singular = np.linalg.svd(j, compute_uv=False)
        ratio = float(current_singular[-1] / current_singular[0])
        velocity = mass_metric_step(j, tangent, masses).reshape(r0.shape)
        mu = float(np.sum(masses[:, None] * velocity**2))
        if (
            closure > config.BACKTRANS_TOL
            or ratio <= config.INTERNAL_RANK_RTOL
            or mu <= 0
        ):
            raise ValueError(
                f"Invalid path point: mode {mode_number}, Q={q:g}."
            )
        closures.append(closure)
        rank_ratios.append(ratio)
        tangents.append(velocity)
        mus.append(mu)

    mus = np.asarray(mus)
    zero_index = int(np.flatnonzero(
        np.isclose(dense_q, 0.0, atol=1e-12),
    )[0])
    mu0 = float(mus[zero_index])
    scoord = mass_coordinate(dense_q, mus, mu0)
    if not np.all(np.diff(scoord) > 0):
        raise ValueError(
            f"Mode {mode_number}: mass-reparameterized coordinate is not monotonic."
        )

    mass_fd_errors = []
    if validate_mass:
        hmass = config.PATH_MASS_FD_STEP_ANG
        for q in energy_q:
            idx = int(np.argmin(np.abs(dense_q - q)))
            geometry = ordered_geometry[idx]
            plus, *_ = continue_backtransform(
                (q + hmass) * tangent, r0, definitions, s0, weights, basis,
                masses, initial=geometry,
                initial_difference=path_differences[round(float(dense_q[idx]), 12)],
                rank_rtol=config.INTERNAL_RANK_RTOL, **back_options,
            )
            minus, *_ = continue_backtransform(
                (q - hmass) * tangent, r0, definitions, s0, weights, basis,
                masses, initial=geometry,
                initial_difference=path_differences[round(float(dense_q[idx]), 12)],
                rank_rtol=config.INTERNAL_RANK_RTOL, **back_options,
            )
            velocity_fd = remove_rigid_velocity(
                geometry, (plus - minus) / (2 * hmass), masses,
            )
            error = float(np.sqrt(
                np.sum(
                    masses[:, None] * (velocity_fd - tangents[idx])**2,
                ) / mus[idx]
            ))
            mass_fd_errors.append((float(q), error))
            if error > config.PATH_MASS_TANGENT_RTOL:
                raise ValueError(
                    f"Path-mass validation failed for mode {mode_number}, "
                    f"Q={q:g}: {error:.3e}."
                )

    return dict(
        mode=mode_number,
        coordinate_system=coordinate_system,
        tangent=tangent,
        energy_q=energy_q.copy(),
        q=dense_q.copy(),
        s=scoord,
        mu=mus,
        geometries=ordered_geometry,
        cartesian_tangents=np.asarray(tangents),
        closure=np.asarray(closures),
        rank_ratio=np.asarray(rank_ratios),
        iterations=np.asarray([
            iteration_count[round(float(q), 12)] for q in dense_q
        ]),
        mass_fd_errors=np.asarray(mass_fd_errors, float).reshape(-1, 2),
    )


def multiframe_xyz(symbols, frames):
    chunks = []
    for comment, geometry in frames:
        chunks.extend([str(len(symbols)), comment])
        chunks.extend(
            f"{symbol:2s} {row[0]: .12f} {row[1]: .12f} {row[2]: .12f}"
            for symbol, row in zip(symbols, geometry)
        )
    return "\n".join(chunks) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Cartesian or curvilinear AnhDis scans and the coordinate s."
    )
    configured_log = getattr(
        config, "OPT_FREQ_OUTPUT", getattr(config, "GAUSSIAN_LOG", None),
    )
    if configured_log is None:
        raise ValueError("Define OPT_FREQ_OUTPUT (or legacy GAUSSIAN_LOG) in config.py.")
    parser.add_argument("--log", type=Path, default=configured_log)
    parser.add_argument(
        "--software", choices=("auto", "gaussian", "orca"),
        default=getattr(config, "OPT_FREQ_SOFTWARE", "auto"),
        help="Format of the optimization/frequency output; default OPT_FREQ_SOFTWARE/auto",
    )
    parser.add_argument("--output", type=Path, default=config.SCAN_ROOT)
    parser.add_argument("--modes", nargs="+", help="Mode numbers; default SELECTED_MODES/all")
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Independent mode processes; 1 preserves the sequential calculation",
    )
    parser.add_argument(
        "--coordinate-system", choices=("curvilinear", "cartesian"),
        default=None,
        help=(
            "Path used for electronic-energy geometries; default A1_COORDINATE_SYSTEM. "
            "Cartesian paths are analytic and do not use iterative back-transformation"
        ),
    )
    parser.add_argument("--skip-mass-validation", action="store_true")
    parser.add_argument(
        "--update", action="store_true",
        help="Safely extend selected modes in an existing a1 scan without deleting calculations",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be a positive integer.")
    coordinate_system = normalize_coordinate_system(
        args.coordinate_system
        if args.coordinate_system is not None
        else getattr(config, "A1_COORDINATE_SYSTEM", "curvilinear")
    )

    opt_freq_software = normalize_opt_freq_software(args.software)
    log_path = args.log.resolve()
    if not log_path.is_file():
        raise FileNotFoundError(f"Optimization/frequency output not found: {log_path}")
    (
        text, atoms, symbols, r0, masses, frequencies, reference_rm, modes,
        opt_freq_software,
    ) = parse_harmonic_log(log_path, opt_freq_software, return_software=True)
    chosen = selected_modes(args.modes if args.modes else config.SELECTED_MODES, len(modes))
    nonpositive = [mode for mode in chosen if frequencies[mode - 1] <= 0]
    if nonpositive:
        raise ValueError(
            f"Selected modes with non-positive {opt_freq_software.upper()} frequencies: "
            f"{nonpositive}. "
            "This workflow requires a stable reference minimum."
        )
    overrides = getattr(config, "Q_ENERGY_ANG_BY_MODE", {})
    if not isinstance(overrides, dict):
        raise ValueError("Q_ENERGY_ANG_BY_MODE must be a dictionary keyed by mode number.")
    invalid_override_modes = sorted(
        key for key in overrides
        if not isinstance(key, int) or key < 1 or key > len(modes)
    )
    if invalid_override_modes:
        raise ValueError(f"Invalid Q_ENERGY_ANG_BY_MODE keys: {invalid_override_modes}")

    definitions = read_internals(
        text, len(atoms), reference_coords=r0, software=opt_freq_software,
    )
    s0 = internal_values(r0, definitions)
    validate_printed_internals(definitions, s0, opt_freq_software)

    fd = config.INTERNAL_FD_STEP_ANG
    b_coarse = wilson_b(r0, definitions, fd)
    b0 = wilson_b(r0, definitions, fd / 2)
    weights = np.asarray([
        1.0 / config.INTERNAL_REFERENCE_LENGTH_ANG if item.kind == "R" else 1.0
        for item in definitions
    ])
    scaled_b0 = weights[:, None] * b0
    u, singular, _ = np.linalg.svd(scaled_b0, full_matrices=False)
    rank = int(np.sum(singular > config.INTERNAL_RANK_RTOL * singular[0]))
    expected = vibrational_dof(r0)
    if rank != expected:
        raise ValueError(f"Wilson B rank is {rank}; expected {expected}.")
    basis = u[:, :rank]
    j0 = basis.T @ scaled_b0
    equilibrium_singular = np.linalg.svd(j0, compute_uv=False)
    equilibrium_rank_ratio = float(
        equilibrium_singular[-1] / equilibrium_singular[0]
    )
    flat_modes = modes.reshape(len(modes), -1)
    primitive_tangents = b0 @ flat_modes.T
    direct_tangents = np.column_stack([
        internal_difference(
            internal_values(r0 + fd * mode, definitions),
            internal_values(r0 - fd * mode, definitions),
            definitions,
        ) / (2 * fd)
        for mode in modes
    ])
    b_step_error = float(np.max(np.abs(weights[:, None] * (b0 - b_coarse))))
    tangent_error = float(np.max(np.abs(weights[:, None] * (primitive_tangents - direct_tangents))))
    rigid_error = float(np.max(np.abs(scaled_b0 @ rigid_directions(r0))))
    span_error = float(np.max(np.abs(scaled_b0 - basis @ j0)))
    if max(b_step_error, tangent_error, rigid_error, span_error) > 1e-7:
        raise ValueError("Wilson-B/SVD numerical validation failed.")

    mass_xyz = np.repeat(masses, 3)
    printed_mode_norms = np.linalg.norm(flat_modes, axis=1)
    independent_tangents = j0 @ flat_modes.T
    projected = np.asarray([
        mass_metric_step(j0, independent_tangents[:, i], masses).reshape(r0.shape)
        for i in range(len(modes))
    ])
    original_mu = np.sum(mass_xyz[None, :] * flat_modes**2, axis=1)
    projected_mu = np.sum(mass_xyz[None, :] * projected.reshape(len(modes), -1)**2, axis=1)
    mass_cosines = np.sum(
        mass_xyz[None, :] * flat_modes * projected.reshape(len(modes), -1), axis=1
    ) / np.sqrt(original_mu * projected_mu)
    reference_mass_difference_percent = 100.0 * np.abs(original_mu / reference_rm - 1.0)

    local_rows = []
    hlocal = config.BACKTRANS_TEST_AMPLITUDE_ANG
    back_options = dict(
        step=fd,
        tolerance=config.BACKTRANS_TOL,
        max_iterations=config.BACKTRANS_MAX_ITER,
        cart_step_limit=config.BACKTRANS_CART_STEP_LIMIT_ANG,
    )
    if coordinate_system == "curvilinear":
        for mode_index in range(len(modes)):
            tangent = independent_tangents[:, mode_index]
            plus, _, plus_error, _ = backtransform(
                hlocal * tangent, r0, definitions, s0, weights, basis, masses,
                initial=r0 + hlocal * projected[mode_index], **back_options,
            )
            minus, _, minus_error, _ = backtransform(
                -hlocal * tangent, r0, definitions, s0, weights, basis, masses,
                initial=r0 - hlocal * projected[mode_index], **back_options,
            )
            numerical = (plus - minus) / (2 * hlocal)
            relative = np.sqrt(
                np.sum(masses[:, None] * (numerical - projected[mode_index])**2)
                / projected_mu[mode_index]
            )
            local_rows.append((
                mode_index + 1, mass_cosines[mode_index], relative,
                plus_error, minus_error,
            ))
        if max(row[2] for row in local_rows) > 2e-5:
            raise ValueError("Local internal-to-Cartesian tangent validation failed.")
        maximum_local_tangent_error = float(max(row[2] for row in local_rows))
    else:
        # Keep one row per mode for a stable diagnostic file while explicitly
        # marking curvilinear-only validation quantities as not applicable.
        local_rows = [
            (mode_index + 1, mass_cosines[mode_index], np.nan, np.nan, np.nan)
            for mode_index in range(len(modes))
        ]
        maximum_local_tangent_error = None

    validate_mass = (
        coordinate_system == "curvilinear"
        and config.VALIDATE_PATH_MASS
        and not args.skip_mass_validation
    )
    tasks = [
        (
            mode_number,
            independent_tangents[:, mode_number - 1],
            modes[mode_number - 1],
            r0,
            definitions,
            s0,
            weights,
            basis,
            masses,
            back_options,
            fd,
            validate_mass,
            coordinate_system,
            equilibrium_rank_ratio,
        )
        for mode_number in chosen
    ]
    worker_count = min(args.workers, len(tasks))
    path_results = []
    if worker_count == 1:
        for completed, task in enumerate(tasks, 1):
            result = reconstruct_mode_path(task)
            path_results.append(result)
            print(
                f"a1: mode {result['mode']} completed "
                f"({completed}/{len(tasks)}; workers=1)",
                flush=True,
            )
    else:
        print(
            f"a1: reconstructing {len(tasks)} modes with {worker_count} workers ...",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(reconstruct_mode_path, task): task[0]
                for task in tasks
            }
            for completed, future in enumerate(as_completed(futures), 1):
                mode_number = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Parallel reconstruction failed for mode {mode_number}."
                    ) from exc
                path_results.append(result)
                print(
                    f"a1: mode {mode_number} completed "
                    f"({completed}/{len(tasks)}; workers={worker_count})",
                    flush=True,
                )
    path_results.sort(key=lambda item: int(item["mode"]))

    global_max_closure = max(
        float(np.max(result["closure"])) for result in path_results
    )
    global_min_rank_ratio = min(
        float(np.min(result["rank_ratio"])) for result in path_results
    )
    mass_error_arrays = [
        result["mass_fd_errors"][:, 1]
        for result in path_results if result["mass_fd_errors"].size
    ]
    global_max_mass_fd_error = (
        max(float(np.max(values)) for values in mass_error_arrays)
        if mass_error_arrays else 0.0
    )

    requested_output = args.output.resolve()
    old_manifest = None
    if args.update:
        manifest_path = requested_output / "scan_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"--update requires an existing a1 manifest: {manifest_path}")
        old_manifest = json.loads(manifest_path.read_text())
        if old_manifest.get("stage") != "curvilinear_scan_geometries_with_mass_coordinate":
            raise ValueError("The existing output is not an a1 curvilinear scan.")
        old_coordinate_system = normalize_coordinate_system(
            old_manifest.get("coordinate_system", "curvilinear")
        )
        if old_coordinate_system != coordinate_system:
            raise ValueError(
                "Refusing to mix Cartesian and curvilinear paths in one scan directory."
            )
        if old_manifest.get("source_log_sha256") != sha256_file(log_path):
            raise ValueError(
                "Refusing to update scans generated from a different "
                "optimization/frequency output."
            )
        old_software = old_manifest.get("opt_freq_software")
        if old_software is not None and old_software != opt_freq_software:
            raise ValueError("The existing scan uses a different optimization/frequency format.")
        if old_manifest.get("n_atoms") != len(atoms) or old_manifest.get("n_modes") != len(modes):
            raise ValueError("The existing scan dimensions disagree with the source output.")
        old_points_by_mode = {}
        for point in old_manifest.get("energy_points", []):
            old_points_by_mode.setdefault(int(point["mode"]), set()).add(
                round(float(point["Q_angstrom"]), 12)
            )
        for result in path_results:
            mode_number = result["mode"]
            old_q = old_points_by_mode.get(mode_number, set())
            new_q = {round(float(q), 12) for q in result["energy_q"]}
            if not old_q <= new_q:
                removed = sorted(old_q - new_q)
                raise ValueError(
                    f"Mode {mode_number}: --update only permits extending a scan; "
                    f"the new grid would remove Q values {removed}."
                )
        out = requested_output
    else:
        out = ensure_new_directory(requested_output)

    equilibrium_path = out / "equilibrium.xyz"
    if args.update and equilibrium_path.is_file():
        old_symbols, old_equilibrium = load_xyz(equilibrium_path)
        if old_symbols != symbols or not np.allclose(old_equilibrium, r0, atol=5e-11, rtol=0):
            raise ValueError("The existing equilibrium.xyz disagrees with the source output.")
    else:
        write_xyz(equilibrium_path, symbols, r0, "Equilibrium geometry; Cartesian angstrom")
    np.savez_compressed(
        out / "reference_data.npz",
        atomic_numbers=atoms,
        symbols=np.asarray(symbols),
        equilibrium_angstrom=r0,
        masses_amu=masses,
        frequencies_cm=frequencies,
        source_reduced_masses_amu=reference_rm,
        # Retained for backward compatibility with a6/a7 data readers and old
        # analysis notebooks; for ORCA it contains the derived modal masses.
        gaussian_reduced_masses_amu=reference_rm,
        opt_freq_software=np.asarray(opt_freq_software),
        normal_modes=modes,
        projected_tangents=projected,
        internal_basis=basis,
        internal_weights=weights,
        equilibrium_internals=s0,
        equilibrium_jacobian=j0,
    )
    np.savetxt(
        out / "vibrational_data.dat",
        np.column_stack((np.arange(1, len(modes) + 1), frequencies, reference_rm, original_mu, projected_mu, mass_cosines)),
        fmt=["%d"] + ["%.12e"] * 5,
        header="mode frequency_cm-1 source_reduced_mass_amu printed_vector_mass_amu projected_path_mass_amu mass_cosine",
    )
    np.savetxt(
        out / "local_validation.dat", np.asarray(local_rows),
        fmt=["%d"] + ["%.12e"] * 4,
        header="mode original_projected_mass_cosine relative_tangent_error residual_plus residual_minus",
    )
    (out / "internal_coordinates.json").write_text(json.dumps([
        dict(
            label=item.label,
            kind=item.kind,
            atoms_1based=[index + 1 for index in item.atoms],
            component=item.component,
            axis_reference=item.axis_reference,
            frame_axis=item.frame_axis,
            frame_direction_1=item.frame_direction_1,
            equilibrium=float(value),
            unit="angstrom" if item.kind == "R" else "radian",
        )
        for item, value in zip(definitions, s0)
    ], indent=2) + "\n")

    energy_records = []
    mode_summaries = []
    for result in path_results:
        mode_number = result["mode"]
        mode_dir = out / f"vib{mode_number}"
        mode_dir.mkdir(exist_ok=args.update)
        table = np.column_stack((
            result["q"], result["s"], result["mu"],
            100 * (result["mu"] / result["mu"][np.argmin(np.abs(result["q"]))] - 1),
            result["closure"], result["rank_ratio"], result["iterations"],
        ))
        np.savetxt(
            mode_dir / "path_metric.dat", table, fmt="%.12e",
            header=(
                "Q_angstrom s_angstrom mu_amu relative_mu_percent "
                "coordinate_constraint_residual sigma_min_over_max_J "
                "path_construction_iterations"
            ),
        )
        np.savez_compressed(
            mode_dir / "path_geometries.npz",
            coordinate_system=np.asarray(coordinate_system),
            Q_angstrom=result["q"], s_angstrom=result["s"], mu_amu=result["mu"],
            geometries_angstrom=result["geometries"],
            cartesian_tangents_dR_dQ=result["cartesian_tangents"],
        )
        if result["mass_fd_errors"].size:
            np.savetxt(
                mode_dir / "mass_validation.dat", result["mass_fd_errors"], fmt="%.12e",
                header="Q_angstrom relative_mass_weighted_tangent_error",
            )
        target_frames, original_frames, projected_frames = [], [], []
        for q in sorted(result["energy_q"]):
            idx = int(np.argmin(np.abs(result["q"] - q)))
            if abs(result["q"][idx] - q) > 1e-10:
                raise RuntimeError("Energy Q point missing from dense path.")
            geometry = result["geometries"][idx]
            label = q_label(q)
            point_dir = mode_dir / label
            point_dir.mkdir(exist_ok=args.update)
            xyz_path = point_dir / f"vib{mode_number}_{label}.xyz"
            comment = (
                f"{coordinate_system}; mode={mode_number}; Q={q:.12g} Ang; "
                f"s={result['s'][idx]:.12g} Ang; mu={result['mu'][idx]:.12g} amu"
            )
            if args.update and xyz_path.is_file():
                old_symbols, old_geometry = load_xyz(xyz_path)
                if old_symbols != symbols or not np.allclose(old_geometry, geometry, atol=5e-11, rtol=0):
                    calculation_files = [
                        xyz_path.with_suffix(suffix)
                        for suffix in (".com", ".log", ".out", ".chk")
                    ]
                    if any(path.exists() for path in calculation_files):
                        raise ValueError(
                            f"Refusing to change a geometry that already has Gaussian files: {xyz_path}"
                        )
                    write_xyz(xyz_path, symbols, geometry, comment)
                # If the Cartesian geometry agrees, preserve the original XYZ
                # byte-for-byte so its a2 identity and completed output remain valid.
            else:
                write_xyz(xyz_path, symbols, geometry, comment)
            target_frames.append((comment, geometry))
            original = r0 + q * modes[mode_number - 1]
            projected_geometry = r0 + q * projected[mode_number - 1]
            original_frames.append((f"Cartesian original; mode={mode_number}; Q={q:.12g} Ang", original))
            projected_frames.append((f"Cartesian projected; mode={mode_number}; Q={q:.12g} Ang", projected_geometry))
            energy_records.append(dict(
                mode=mode_number,
                coordinate_system=coordinate_system,
                Q_angstrom=float(q),
                s_angstrom=float(result["s"][idx]),
                mu_amu=float(result["mu"][idx]),
                xyz=str(xyz_path.relative_to(out)),
            ))
        (mode_dir / f"vib{mode_number}_{coordinate_system}.xyz").write_text(
            multiframe_xyz(symbols, target_frames)
        )
        (mode_dir / f"vib{mode_number}_cartesian_original.xyz").write_text(multiframe_xyz(symbols, original_frames))
        (mode_dir / f"vib{mode_number}_cartesian_projected.xyz").write_text(multiframe_xyz(symbols, projected_frames))
        mu0 = float(result["mu"][np.argmin(np.abs(result["q"]))])
        mode_summaries.append(dict(
            mode=mode_number,
            coordinate_system=coordinate_system,
            q_min_angstrom=float(result["energy_q"][0]),
            q_max_angstrom=float(result["energy_q"][-1]),
            energy_point_count=len(result["energy_q"]),
            mu0_amu=mu0,
            min_mu_amu=float(np.min(result["mu"])),
            max_mu_amu=float(np.max(result["mu"])),
            max_abs_relative_mu_percent=float(np.max(np.abs(result["mu"] / mu0 - 1)) * 100),
            s_min_angstrom=float(result["s"][0]),
            s_max_angstrom=float(result["s"][-1]),
        ))

    if old_manifest is not None:
        chosen_set = set(chosen)
        energy_records = [
            point for point in old_manifest.get("energy_points", [])
            if int(point["mode"]) not in chosen_set
        ] + energy_records
        mode_summaries = [
            item for item in old_manifest.get("modes", [])
            if int(item["mode"]) not in chosen_set
        ] + mode_summaries
        selected_for_manifest = sorted(set(old_manifest.get("selected_modes", [])) | chosen_set)
        old_diagnostics = old_manifest.get("diagnostics", {})
        global_max_closure = max(
            global_max_closure,
            float(old_diagnostics.get("maximum_finite_internal_residual", 0.0)),
        )
        global_min_rank_ratio = min(
            global_min_rank_ratio,
            float(old_diagnostics.get("minimum_finite_jacobian_singular_ratio", np.inf)),
        )
        if validate_mass:
            global_max_mass_fd_error = max(
                global_max_mass_fd_error,
                float(old_diagnostics.get("maximum_path_mass_tangent_error") or 0.0),
            )
    else:
        selected_for_manifest = list(chosen)

    energy_records.sort(key=lambda item: (int(item["mode"]), float(item["Q_angstrom"])))
    mode_summaries.sort(key=lambda item: int(item["mode"]))
    q_by_mode = {
        str(mode): [
            float(point["Q_angstrom"]) for point in energy_records
            if int(point["mode"]) == mode
        ]
        for mode in selected_for_manifest
    }
    for item in mode_summaries:
        values = q_by_mode[str(int(item["mode"]))]
        item.setdefault("q_min_angstrom", float(min(values)))
        item.setdefault("q_max_angstrom", float(max(values)))
        item.setdefault("energy_point_count", len(values))

    manifest = dict(
        stage="curvilinear_scan_geometries_with_mass_coordinate",
        coordinate_system=coordinate_system,
        cartesian_path_definition=(
            f"R(Q)=R0+Q*d_i using the {opt_freq_software.upper()} printed normal-mode vector"
            if coordinate_system == "cartesian" else None
        ),
        source_log=str(log_path),
        source_log_sha256=sha256_file(log_path),
        opt_freq_software=opt_freq_software,
        normal_mode_numbering=(
            "vibrational modes reindexed from 1 in ascending ORCA column order"
            if opt_freq_software == "orca"
            else "Gaussian vibrational mode order"
        ),
        source_reduced_mass_definition=(
            "sum_a m_a |d_ia|^2 derived from the normalized printed ORCA vector"
            if opt_freq_software == "orca"
            else "Gaussian Red. masses table"
        ),
        n_atoms=len(atoms),
        n_modes=len(modes),
        reference_is_linear=(rigid_directions(r0).shape[1] == 5),
        n_primitive_internals=len(definitions),
        independent_rank=rank,
        selected_modes=selected_for_manifest,
        Q_energy_angstrom=sorted(float(x) for x in config.Q_ENERGY_ANG),
        Q_energy_angstrom_by_mode=q_by_mode,
        path_Q_step_max_angstrom=float(config.PATH_Q_STEP_ANG),
        coordinate_definition=(
            "xi(R)=U0.T W [S(R)-S0]=Q l_i; ds/dQ=sqrt(mu(Q)/mu(0))"
            if coordinate_system == "curvilinear"
            else "R(Q)=R0+Q*d_i; mu(Q)=mu(0); s=Q"
        ),
        mass_coordinate_quadrature=(
            "cumulative Simpson on the dense Q grid"
            if coordinate_system == "curvilinear" else "analytic identity s=Q"
        ),
        units=dict(cartesian="angstrom", Q="angstrom", s="angstrom", mass="amu", angles="radian"),
        energy_points=energy_records,
        modes=mode_summaries,
        diagnostics=dict(
            B_step_difference=b_step_error,
            primitive_mode_tangent_error=tangent_error,
            rigid_motion_error=rigid_error,
            independent_span_error=span_error,
            minimum_original_projected_mass_cosine=float(np.min(mass_cosines)),
            maximum_local_tangent_error=maximum_local_tangent_error,
            maximum_finite_internal_residual=global_max_closure,
            minimum_finite_jacobian_singular_ratio=global_min_rank_ratio,
            path_internal_diagnostics_evaluated=(coordinate_system == "curvilinear"),
            maximum_path_mass_tangent_error=global_max_mass_fd_error if validate_mass else None,
            maximum_printed_mode_norm_deviation=float(np.max(np.abs(printed_mode_norms - 1.0))),
            maximum_printed_vector_vs_source_mass_difference_percent=float(
                np.max(reference_mass_difference_percent)
            ),
            # Deprecated name retained so older diagnostics remain readable.
            maximum_printed_vector_vs_gaussian_mass_difference_percent=float(
                np.max(reference_mass_difference_percent)
            ),
        ),
        gaussian_inputs_generated=False,
        gaussian_jobs_submitted=False,
        future_sampling=(
            "Sample s, invert monotonic s(Q), and reconstruct through the internal-coordinate path."
            if coordinate_system == "curvilinear"
            else "Sample s=Q and apply the analytic Cartesian normal-mode displacement."
        ),
    )
    (out / "scan_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        f"AnhDis {coordinate_system} a1: PASS",
        f"Optimization/frequency source: {opt_freq_software.upper()} ({log_path.name})",
        f"Atoms: {len(atoms)}; vibrational modes: {len(modes)}; primitive internals: {len(definitions)}; rank: {rank}",
        f"Modes reconstructed in this run: {chosen}",
        f"Modes present in the scan: {selected_for_manifest}",
        f"Electronic-energy geometries present: {len(energy_records)}",
        f"Minimum original/projected mass cosine: {np.min(mass_cosines):.10f}",
        (
            f"Maximum local tangent error: {maximum_local_tangent_error:.3e}"
            if maximum_local_tangent_error is not None
            else "Maximum local tangent error: not applicable to analytic Cartesian paths"
        ),
        f"Maximum norm deviation of rounded printed modes: {np.max(np.abs(printed_mode_norms - 1.0)):.3e}",
        f"Maximum printed-vector/source reduced-mass difference: {np.max(reference_mass_difference_percent):.3f}%",
        (
            f"Maximum finite internal residual: {global_max_closure:.3e}"
            if coordinate_system == "curvilinear"
            else "Maximum finite internal residual: not applicable to analytic Cartesian paths"
        ),
        (
            f"Minimum finite Jacobian singular ratio: {global_min_rank_ratio:.3e}"
            if coordinate_system == "curvilinear"
            else f"Equilibrium Jacobian singular ratio: {equilibrium_rank_ratio:.3e}"
        ),
    ]
    for item in mode_summaries:
        report.append(
            f"Mode {item['mode']}: mu(0)={item['mu0_amu']:.8f} amu; "
            f"max |Delta mu/mu0|={item['max_abs_relative_mu_percent']:.3f}%; "
            f"Q=[{item.get('q_min_angstrom', np.nan):.2f},{item.get('q_max_angstrom', np.nan):.2f}] Ang; "
            f"s=[{item['s_min_angstrom']:.6f},{item['s_max_angstrom']:.6f}] Ang"
        )
    (out / "a1_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nSaved validated {coordinate_system} scans to {out}")


if __name__ == "__main__":
    run_timed_stage("a1", main, config.EXECUTION_TIMING_FILE)
