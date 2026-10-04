#!/usr/bin/env python3
"""a6: solve the curvilinear 1D Hamiltonians and build thermal densities.

The electronic potential is V(s), where s is the constant-mass coordinate
created by a1 and fitted by a5.  Internally s is converted from angstrom to
bohr, and the equilibrium path mass mu(0) is converted from amu to electron
masses, so the finite-difference Hamiltonian is entirely in atomic units.

After the coordinate-space calculation, every thermally retained eigenstate
is Fourier transformed using

    phi(p_s) = (2*pi)^(-1/2) integral psi(s) exp(-i*p_s*s) ds.

The resulting positive momentum marginal is the Boltzmann-weighted incoherent
sum sum_v w_v |phi_v(p_s)|^2.  This does not construct an anharmonic Wigner
quasiprobability: position and momentum marginals are saved separately for the
independent sampling prescribed by the AnhDis model.
"""

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
from scipy.fft import fft, fftfreq, fftshift, ifftshift, next_fast_len
from scipy.interpolate import PchipInterpolator
from scipy.linalg import eigh_tridiagonal

import config
from workflow_utils import read_scan_manifest, run_timed_stage, selected_modes


def potential_on_grid(derivatives, xgrid):
    """Evaluate V(s)=sum_n k_n*s^n/n! for arbitrary saved orders."""
    xgrid = np.asarray(xgrid, float)
    potential = np.zeros_like(xgrid)
    for order, value in derivatives.items():
        order = int(order)
        if order >= 1:
            potential += float(value) * xgrid**order / math.factorial(order)
    return potential


def pchip_potential_on_angstrom_grid(model, s_angstrom):
    """Evaluate the a5 PCHIP interpolant and its confining endpoint tails."""
    scalar = np.ndim(s_angstrom) == 0
    x = np.atleast_1d(np.asarray(s_angstrom, float))
    knots = np.asarray(model["knots_s_angstrom"], float)
    values = np.asarray(model["knots_v_hartree"], float)
    if (
        knots.ndim != 1 or values.shape != knots.shape or knots.size < 3
        or not np.isfinite(knots).all() or not np.isfinite(values).all()
        or np.any(np.diff(knots) <= 0.0)
    ):
        raise ValueError("Invalid PCHIP knots in the a5 potential file.")
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


def build_hamiltonian(derivatives, mr_emass, xgrid, potential_values=None):
    """Return the tridiagonal finite-difference Hamiltonian and V(x)."""
    xgrid = np.asarray(xgrid, float)
    if xgrid.ndim != 1 or xgrid.size < 5:
        raise ValueError("The a6 grid must be one-dimensional with at least five points.")
    dx_values = np.diff(xgrid)
    if not np.allclose(dx_values, dx_values[0], atol=1e-14, rtol=1e-12):
        raise ValueError("The a6 finite-difference grid must be uniform.")
    if not np.isfinite(mr_emass) or mr_emass <= 0:
        raise ValueError("The effective mass must be positive and finite.")
    dx = float(dx_values[0])
    if potential_values is None:
        potential = potential_on_grid(derivatives, xgrid)
    else:
        potential = np.asarray(potential_values, float)
        if potential.shape != xgrid.shape or not np.isfinite(potential).all():
            raise ValueError("Explicit potential values must be finite and match the a6 grid.")
    kinetic_diagonal = 1.0 / (mr_emass * dx**2)
    kinetic_offdiagonal = -0.5 / (mr_emass * dx**2)
    diagonal = potential + kinetic_diagonal
    offdiagonal = np.full(xgrid.size - 1, kinetic_offdiagonal)
    return diagonal, offdiagonal, potential


def normalize_wave(psi, xgrid):
    norm = float(np.trapezoid(np.abs(psi)**2, xgrid))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize a non-finite or zero wavefunction.")
    return psi / np.sqrt(norm)


def solve_states(
    derivatives, mr_emass, temperature, xgrid, max_states, weight_cutoff,
    potential_values=None,
):
    diagonal, offdiagonal, potential = build_hamiltonian(
        derivatives, mr_emass, xgrid, potential_values=potential_values,
    )
    n_states = min(int(max_states), xgrid.size)
    if n_states < 2:
        raise ValueError("A6_MAX_EIGENSTATES must be at least two.")
    eigenvalues, eigenvectors = eigh_tridiagonal(
        diagonal, offdiagonal, select="i", select_range=(0, n_states - 1),
        check_finite=True,
    )
    waves = np.empty_like(eigenvectors)
    for state in range(eigenvectors.shape[1]):
        wave = normalize_wave(eigenvectors[:, state], xgrid)
        pivot = int(np.argmax(np.abs(wave)))
        if wave[pivot] < 0:
            wave = -wave
        waves[:, state] = wave

    beta = 1.0 / (config.KB_AU * temperature)
    relative_boltzmann = np.exp(-beta * (eigenvalues - eigenvalues[0]))
    included = np.flatnonzero(relative_boltzmann > weight_cutoff)
    if included.size == 0 or included[0] != 0:
        raise RuntimeError("The ground state was unexpectedly excluded from the Boltzmann sum.")
    if included[-1] == n_states - 1:
        raise RuntimeError(
            "The highest computed state still exceeds THERMAL_WEIGHT_CUTOFF; "
            "increase A6_MAX_EIGENSTATES."
        )
    raw_weights = relative_boltzmann[included]
    weights = raw_weights / np.sum(raw_weights)
    thermal_density = np.sum(np.abs(waves[:, included])**2 * weights[None, :], axis=1)
    thermal_density /= np.trapezoid(thermal_density, xgrid)
    computed_partition = float(np.sum(relative_boltzmann))
    retained_partition = float(np.sum(raw_weights))
    omitted_fraction = max(0.0, 1.0 - retained_partition / computed_partition)
    return dict(
        xgrid=xgrid, potential=potential, eigenvalues=eigenvalues,
        waves=waves, included=included, weights=weights,
        thermal_density=thermal_density,
        omitted_weight_fraction=omitted_fraction,
    )


def momentum_distribution(states, mr_emass, pad_factor=8):
    """Fourier-transform retained states and build their thermal p marginal.

    The coordinate grid is in bohr, therefore its conjugate momentum ``p_s``
    is in atomic units (hbar/bohr = electron_mass*bohr/atomic_time).  Symmetric
    zero-padding refines the sampled momentum grid without changing the
    coordinate-space wavefunctions.
    """
    xgrid = np.asarray(states["xgrid"], float)
    included = np.asarray(states["included"], int)
    weights = np.asarray(states["weights"], float)
    selected_waves = np.asarray(states["waves"][:, included], float)
    if selected_waves.ndim != 2 or selected_waves.shape[0] != xgrid.size:
        raise ValueError("Invalid coordinate-space waves for Fourier transformation.")
    if included.size != weights.size or included.size == 0:
        raise ValueError("Momentum transformation requires retained states and weights.")
    if not isinstance(pad_factor, (int, np.integer)) or int(pad_factor) < 1:
        raise ValueError("A6_MOMENTUM_FFT_PAD_FACTOR must be a positive integer.")
    dx = float(xgrid[1] - xgrid[0])
    if not np.allclose(np.diff(xgrid), dx, atol=1e-14, rtol=1e-12):
        raise ValueError("Momentum transformation requires a uniform coordinate grid.")

    n_grid = xgrid.size
    n_fft = next_fast_len(int(pad_factor) * n_grid)
    padded = np.zeros((n_fft, included.size), dtype=float)
    first = (n_fft - n_grid) // 2
    padded[first:first + n_grid] = selected_waves
    transformed = fftshift(
        fft(ifftshift(padded, axes=0), axis=0), axes=0,
    ) * dx / np.sqrt(2.0 * np.pi)
    p_grid = 2.0 * np.pi * fftshift(fftfreq(n_fft, d=dx))
    state_densities = np.abs(transformed)**2
    norms = np.trapezoid(state_densities, p_grid, axis=0)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise RuntimeError("A Fourier-transformed eigenstate has invalid normalization.")
    state_densities /= norms[None, :]
    thermal_density = state_densities @ weights
    thermal_density /= np.trapezoid(thermal_density, p_grid)

    kinetic_from_momentum = np.trapezoid(
        p_grid[:, None]**2 * state_densities, p_grid, axis=0,
    ) / (2.0 * mr_emass)
    kinetic_from_coordinate = np.asarray([
        states["eigenvalues"][state]
        - np.trapezoid(
            states["potential"] * np.abs(states["waves"][:, state])**2,
            xgrid,
        )
        for state in included
    ])
    scale = np.maximum(np.abs(kinetic_from_coordinate), 1.0e-14)
    kinetic_relative_errors = np.abs(
        kinetic_from_momentum - kinetic_from_coordinate
    ) / scale
    return dict(
        grid_p_au=p_grid,
        p0_density_per_au=state_densities[:, 0],
        thermal_density_per_au=thermal_density,
        kinetic_from_momentum_hartree=kinetic_from_momentum,
        kinetic_from_coordinate_hartree=kinetic_from_coordinate,
        maximum_kinetic_relative_error=float(np.max(kinetic_relative_errors)),
        fft_points=n_fft,
    )


def probability_outside_interval(xgrid, density, lower, upper):
    """Integrate probability outside [lower, upper], interpolating the CDF."""
    increments = 0.5 * (density[:-1] + density[1:]) * np.diff(xgrid)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf /= cdf[-1]
    inside = float(np.interp(upper, xgrid, cdf) - np.interp(lower, xgrid, cdf))
    return float(np.clip(1.0 - inside, 0.0, 1.0))


def edge_density_ratio(density, edge_points):
    edge_points = max(1, min(int(edge_points), density.size // 2))
    edge_max = max(float(np.max(density[:edge_points])), float(np.max(density[-edge_points:])))
    density_max = float(np.max(density))
    return edge_max / density_max if density_max > 0 else np.inf


def harmonic_force_constant_bohr(frequency_cm, mass_emass):
    omega_au = frequency_cm * config.CM_TO_HARTREE
    return omega_au**2 * mass_emass


def derivatives_angstrom_to_bohr(derivatives):
    """Convert d^nV/ds^n from hartree/angstrom^n to hartree/bohr^n."""
    converted = {}
    for order, value in derivatives.items():
        order = int(order)
        converted[order] = float(value) * config.BOHR_TO_ANG**order
    return converted


def load_mode_input(scan_root, mode):
    path = scan_root / f"vib{mode}" / "potential_for_quantum_step.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing a5 potential: {path}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "coordinate", "mass_amu", "frequency_cm", "energy_s_angstrom",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"Mode {mode}: incomplete a5 potential file; missing {missing}.")
        coordinate = str(np.asarray(data["coordinate"]).item())
        if coordinate != "s":
            raise ValueError(f"Mode {mode}: expected coordinate s, found {coordinate!r}.")
        fit_model = (
            str(np.asarray(data["fit_model"]).item()).strip().lower()
            if "fit_model" in data.files else "polynomial"
        )
        if fit_model not in {"polynomial", "pchip"}:
            raise ValueError(f"Mode {mode}: unknown a5 fit model {fit_model!r}.")
        derivatives = {}
        pchip_model = None
        if fit_model == "polynomial":
            polynomial_required = {
                "derivative_orders", "derivatives_hartree_per_angstrom_power",
            }
            missing_polynomial = sorted(polynomial_required - set(data.files))
            if missing_polynomial:
                raise ValueError(
                    f"Mode {mode}: incomplete polynomial potential; missing {missing_polynomial}."
                )
            orders = np.asarray(data["derivative_orders"], int)
            values = np.asarray(data["derivatives_hartree_per_angstrom_power"], float)
            if (
                orders.ndim != 1 or values.ndim != 1 or orders.size != values.size
                or orders.size == 0 or len(np.unique(orders)) != orders.size
                or np.any(orders < 1) or not np.isfinite(values).all()
            ):
                raise ValueError(f"Mode {mode}: invalid polynomial derivative arrays.")
            derivatives = {
                int(order): float(value) for order, value in zip(orders, values)
            }
        else:
            pchip_required = {
                "pchip_s_angstrom", "pchip_v_hartree",
                "pchip_left_slope_hartree_per_angstrom",
                "pchip_right_slope_hartree_per_angstrom",
                "pchip_tail_curvature_hartree_per_angstrom2",
            }
            missing_pchip = sorted(pchip_required - set(data.files))
            if missing_pchip:
                raise ValueError(
                    f"Mode {mode}: incomplete PCHIP potential; missing {missing_pchip}."
                )
            pchip_model = dict(
                knots_s_angstrom=np.asarray(data["pchip_s_angstrom"], float),
                knots_v_hartree=np.asarray(data["pchip_v_hartree"], float),
                left_slope_hartree_per_angstrom=float(np.asarray(
                    data["pchip_left_slope_hartree_per_angstrom"]
                ).item()),
                right_slope_hartree_per_angstrom=float(np.asarray(
                    data["pchip_right_slope_hartree_per_angstrom"]
                ).item()),
                tail_curvature_hartree_per_angstrom2=float(np.asarray(
                    data["pchip_tail_curvature_hartree_per_angstrom2"]
                ).item()),
            )
            # Evaluate once here so malformed knots or tail parameters fail
            # before the Hamiltonian is constructed.
            pchip_potential_on_angstrom_grid(pchip_model, pchip_model["knots_s_angstrom"])
        mass_amu = float(np.asarray(data["mass_amu"]).item())
        frequency_cm = float(np.asarray(data["frequency_cm"]).item())
        scan_s_ang = np.asarray(data["energy_s_angstrom"], float)
    if not np.isfinite(mass_amu) or mass_amu <= 0:
        raise ValueError(f"Mode {mode}: invalid equilibrium mass.")
    if not np.isfinite(frequency_cm) or frequency_cm <= 0:
        raise ValueError(f"Mode {mode}: invalid Gaussian frequency.")
    if scan_s_ang.ndim != 1 or scan_s_ang.size < 2 or np.any(np.diff(scan_s_ang) <= 0):
        raise ValueError(f"Mode {mode}: invalid a5 energy-coordinate interval.")
    if fit_model == "polynomial":
        if abs(derivatives.get(1, 0.0)) > 1e-12:
            raise ValueError(f"Mode {mode}: the fitted potential is not stationary at s=0.")
        if derivatives.get(2, 0.0) <= 0:
            raise ValueError(f"Mode {mode}: the fitted curvature at s=0 is not positive.")
    return dict(
        path=path, fit_model=fit_model, derivatives=derivatives,
        pchip_model=pchip_model, mass_amu=mass_amu,
        frequency_cm=frequency_cm, scan_s_ang=scan_s_ang,
        polynomial_degree=max(derivatives) if derivatives else 0,
    )


def main():
    parser = argparse.ArgumentParser(description="Solve AnhDis one-dimensional path Hamiltonians.")
    parser.add_argument("--scan-root", type=Path, default=config.SCAN_ROOT)
    parser.add_argument("--modes", nargs="+", help="Subset of a5 modes")
    parser.add_argument("--temperature", type=float, default=config.TEMP)
    parser.add_argument("--strict", action="store_true", help="Fail if a grid/support diagnostic warns")
    args = parser.parse_args()

    root = args.scan_root.resolve()
    manifest = read_scan_manifest(root)
    coordinate_system = str(manifest.get("coordinate_system", "curvilinear")).lower()
    available = [int(mode) for mode in manifest["selected_modes"]]
    chosen = selected_modes(args.modes, manifest["n_modes"]) if args.modes else available
    if not set(chosen) <= set(available):
        raise ValueError("Requested modes were not generated by a1/a5.")
    if not np.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("Temperature must be positive and finite.")
    if not np.isfinite(config.L) or config.L <= 0:
        raise ValueError("L must be positive and finite (bohr).")
    if int(config.NP) < 101:
        raise ValueError("NP must contain at least 101 grid points.")
    if not 0 < config.THERMAL_WEIGHT_CUTOFF < 1:
        raise ValueError("THERMAL_WEIGHT_CUTOFF must lie between zero and one.")

    xgrid = np.linspace(-float(config.L), float(config.L), int(config.NP))
    probability_dir = root / config.PROBABILITY_DIR
    momentum_dir = root / config.MOMENTUM_DIR
    probability_dir.mkdir(parents=True, exist_ok=True)
    momentum_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for mode in chosen:
        item = load_mode_input(root, mode)
        derivatives_ang = item["derivatives"]
        derivatives_bohr = derivatives_angstrom_to_bohr(derivatives_ang)
        mass_amu = item["mass_amu"]
        mass_emass = mass_amu * config.AMU_TO_EMASS
        k_harm_bohr = harmonic_force_constant_bohr(item["frequency_cm"], mass_emass)
        anh_potential_values = (
            None
            if item["fit_model"] == "polynomial"
            else pchip_potential_on_angstrom_grid(
                item["pchip_model"], xgrid * config.BOHR_TO_ANG,
            )
        )

        anh = solve_states(
            derivatives_bohr, mass_emass, args.temperature, xgrid,
            config.A6_MAX_EIGENSTATES, config.THERMAL_WEIGHT_CUTOFF,
            potential_values=anh_potential_values,
        )
        harm = solve_states(
            {2: k_harm_bohr}, mass_emass, args.temperature, xgrid,
            config.A6_MAX_EIGENSTATES, config.THERMAL_WEIGHT_CUTOFF,
        )
        anh_momentum = momentum_distribution(
            anh, mass_emass, config.A6_MOMENTUM_FFT_PAD_FACTOR,
        )
        harm_momentum = momentum_distribution(
            harm, mass_emass, config.A6_MOMENTUM_FFT_PAD_FACTOR,
        )
        if not np.array_equal(
            anh_momentum["grid_p_au"], harm_momentum["grid_p_au"],
        ):
            raise RuntimeError(f"Mode {mode}: anharmonic and harmonic momentum grids differ.")
        psi0_anh = anh["waves"][:, 0]
        psi0_harm = harm["waves"][:, 0]
        p0_anh = np.abs(psi0_anh)**2
        p0_harm = np.abs(psi0_harm)**2

        harmonic_e0_theory = 0.5 * np.sqrt(k_harm_bohr / mass_emass)
        harmonic_e0_relative_error = abs(harm["eigenvalues"][0] - harmonic_e0_theory) / harmonic_e0_theory
        if harmonic_e0_relative_error > config.A6_HARMONIC_E0_RTOL:
            raise RuntimeError(
                f"Mode {mode}: harmonic E0 error is {harmonic_e0_relative_error:.3%}; "
                "increase L and/or NP before using the distribution."
            )

        scan_lower_bohr = float(item["scan_s_ang"][0] / config.BOHR_TO_ANG)
        scan_upper_bohr = float(item["scan_s_ang"][-1] / config.BOHR_TO_ANG)
        outside_anh = probability_outside_interval(
            xgrid, anh["thermal_density"], scan_lower_bohr, scan_upper_bohr,
        )
        outside_harm = probability_outside_interval(
            xgrid, harm["thermal_density"], scan_lower_bohr, scan_upper_bohr,
        )
        edge_ratio_anh = edge_density_ratio(anh["thermal_density"], config.A6_EDGE_POINTS)
        edge_ratio_harm = edge_density_ratio(harm["thermal_density"], config.A6_EDGE_POINTS)
        momentum_edge_ratio_anh = edge_density_ratio(
            anh_momentum["thermal_density_per_au"], config.A6_EDGE_POINTS,
        )
        momentum_edge_ratio_harm = edge_density_ratio(
            harm_momentum["thermal_density_per_au"], config.A6_EDGE_POINTS,
        )
        omega_au = item["frequency_cm"] * config.CM_TO_HARTREE
        thermal_argument = omega_au / (2.0 * config.KB_AU * args.temperature)
        harmonic_p2_theory = 0.5 * mass_emass * omega_au / np.tanh(thermal_argument)
        harmonic_p2_numeric = float(np.trapezoid(
            harm_momentum["grid_p_au"]**2
            * harm_momentum["thermal_density_per_au"],
            harm_momentum["grid_p_au"],
        ))
        harmonic_p2_relative_error = abs(
            harmonic_p2_numeric / harmonic_p2_theory - 1.0
        )
        highest_thermal_state = int(anh["included"][-1])
        highest_thermal_energy = float(anh["eigenvalues"][highest_thermal_state])
        minimum_potential = float(np.min(anh["potential"]))
        boundary_potential = float(min(anh["potential"][0], anh["potential"][-1]))

        warnings = []
        if outside_anh > config.A6_SCAN_TAIL_PROBABILITY_TOL:
            warnings.append("anharmonic probability extends beyond the calculated curvilinear path")
        if edge_ratio_anh > config.A6_EDGE_DENSITY_RATIO_TOL:
            warnings.append("anharmonic density is non-negligible at the numerical-grid edge")
        if edge_ratio_harm > config.A6_EDGE_DENSITY_RATIO_TOL:
            warnings.append("harmonic density is non-negligible at the numerical-grid edge")
        if minimum_potential < -config.A6_NEGATIVE_MINIMUM_TOL_CM / config.HARTREE_TO_CM:
            warnings.append("anharmonic fitted potential has a lower minimum on the quantum grid")
        if boundary_potential <= highest_thermal_energy:
            warnings.append("anharmonic potential boundary lies below a thermally retained state")
        if momentum_edge_ratio_anh > config.A6_MOMENTUM_EDGE_DENSITY_RATIO_TOL:
            warnings.append("anharmonic momentum density is non-negligible at the FFT-grid edge")
        if momentum_edge_ratio_harm > config.A6_MOMENTUM_EDGE_DENSITY_RATIO_TOL:
            warnings.append("harmonic momentum density is non-negligible at the FFT-grid edge")
        if (
            anh_momentum["maximum_kinetic_relative_error"]
            > config.A6_MOMENTUM_KINETIC_RTOL
        ):
            warnings.append("anharmonic Fourier momentum kinetic energy failed its tolerance")
        if (
            harm_momentum["maximum_kinetic_relative_error"]
            > config.A6_MOMENTUM_KINETIC_RTOL
        ):
            warnings.append("harmonic Fourier momentum kinetic energy failed its tolerance")
        if harmonic_p2_relative_error > config.A6_MOMENTUM_HARMONIC_VARIANCE_RTOL:
            warnings.append("harmonic thermal momentum variance failed its analytic tolerance")

        table = np.column_stack((
            xgrid, p0_anh, p0_harm, anh["thermal_density"],
            xgrid * config.BOHR_TO_ANG, harm["thermal_density"],
            anh["potential"], harm["potential"],
        ))
        np.savetxt(
            probability_dir / f"vib{mode}.dat", table, fmt="%.14e",
            header=(
                "x_s_bohr psi0_squared_anh psi0_squared_harm p_thermal_anh_per_bohr "
                "s_angstrom p_thermal_harm_per_bohr V_anh_hartree V_harm_hartree"
            ),
        )
        p_grid = anh_momentum["grid_p_au"]
        modal_velocity_bohr_per_au_time = p_grid / mass_emass
        momentum_table = np.column_stack((
            p_grid,
            p_grid * config.MOMENTUM_AU_TO_AMU_ANG_PER_FS,
            modal_velocity_bohr_per_au_time,
            modal_velocity_bohr_per_au_time
            * config.BOHR_PER_ATOMIC_TIME_TO_ANG_PER_FS,
            anh_momentum["p0_density_per_au"],
            harm_momentum["p0_density_per_au"],
            anh_momentum["thermal_density_per_au"],
            harm_momentum["thermal_density_per_au"],
        ))
        np.savetxt(
            momentum_dir / f"vib{mode}.dat", momentum_table, fmt="%.14e",
            header=(
                "p_s_au p_s_amu_angstrom_per_fs "
                "v_s_bohr_per_atomic_time v_s_angstrom_per_fs "
                "phi0_squared_anh_per_au phi0_squared_harm_per_au "
                "p_thermal_anh_per_au p_thermal_harm_per_au"
            ),
        )
        np.savez_compressed(
            root / f"vib{mode}" / "probability_for_ensemble.npz",
            mode=mode, coordinate=np.asarray("s"), temperature_K=args.temperature,
            mass_amu=mass_amu, grid_s_bohr=xgrid,
            grid_s_angstrom=xgrid * config.BOHR_TO_ANG,
            psi0_anh=psi0_anh, psi0_harm=psi0_harm,
            p0_anh_per_bohr=p0_anh, p0_harm_per_bohr=p0_harm,
            p_thermal_anh_per_bohr=anh["thermal_density"],
            p_thermal_harm_per_bohr=harm["thermal_density"],
            anh_eigenvalues_hartree=anh["eigenvalues"],
            anh_included_states=anh["included"], anh_boltzmann_weights=anh["weights"],
            harmonic_eigenvalues_hartree=harm["eigenvalues"],
            harmonic_included_states=harm["included"], harmonic_boltzmann_weights=harm["weights"],
            grid_p_s_au=p_grid,
            grid_p_s_amu_angstrom_per_fs=(
                p_grid * config.MOMENTUM_AU_TO_AMU_ANG_PER_FS
            ),
            p0_momentum_anh_per_au=anh_momentum["p0_density_per_au"],
            p0_momentum_harm_per_au=harm_momentum["p0_density_per_au"],
            p_thermal_momentum_anh_per_au=anh_momentum["thermal_density_per_au"],
            p_thermal_momentum_harm_per_au=harm_momentum["thermal_density_per_au"],
            fit_model=np.asarray(item["fit_model"]),
            polynomial_degree=item["polynomial_degree"],
            derivative_orders=np.asarray(sorted(derivatives_ang)),
            derivatives_hartree_per_angstrom_power=np.asarray([
                derivatives_ang[order] for order in sorted(derivatives_ang)
            ]),
            scan_s_min_angstrom=item["scan_s_ang"][0],
            scan_s_max_angstrom=item["scan_s_ang"][-1],
        )
        results.append(dict(
            mode=mode, mass_amu=mass_amu, frequency_cm=item["frequency_cm"],
            fit_model=item["fit_model"],
            polynomial_degree=item["polynomial_degree"], derivatives_ang=derivatives_ang,
            k_harm_ang=k_harm_bohr / config.BOHR_TO_ANG**2,
            anh=anh, harm=harm, psi0_anh=psi0_anh, psi0_harm=psi0_harm,
            p0_anh=p0_anh, p0_harm=p0_harm,
            scan_lower_bohr=scan_lower_bohr, scan_upper_bohr=scan_upper_bohr,
            outside_anh=outside_anh, outside_harm=outside_harm,
            edge_ratio_anh=edge_ratio_anh, edge_ratio_harm=edge_ratio_harm,
            harmonic_e0_theory=harmonic_e0_theory,
            harmonic_e0_relative_error=harmonic_e0_relative_error,
            minimum_potential=minimum_potential, boundary_potential=boundary_potential,
            highest_thermal_energy=highest_thermal_energy,
            anh_momentum=anh_momentum, harm_momentum=harm_momentum,
            momentum_edge_ratio_anh=momentum_edge_ratio_anh,
            momentum_edge_ratio_harm=momentum_edge_ratio_harm,
            harmonic_p2_theory=harmonic_p2_theory,
            harmonic_p2_numeric=harmonic_p2_numeric,
            harmonic_p2_relative_error=harmonic_p2_relative_error,
            warnings=warnings,
        ))

    diagnostics_path = root / config.A6_DIAGNOSTICS_FILE
    diagnostic_rows = []
    for result in results:
        diagnostic_rows.append(dict(
            mode=result["mode"], fit_model=result["fit_model"],
            polynomial_degree=result["polynomial_degree"],
            mass_amu=f"{result['mass_amu']:.12e}",
            frequency_cm=f"{result['frequency_cm']:.8f}",
            k_hartree_per_angstrom2=f"{result['derivatives_ang'].get(2, np.nan):.12e}",
            kc_hartree_per_angstrom3=f"{result['derivatives_ang'].get(3, np.nan):.12e}",
            kq_hartree_per_angstrom4=f"{result['derivatives_ang'].get(4, np.nan):.12e}",
            k5_hartree_per_angstrom5=f"{result['derivatives_ang'].get(5, np.nan):.12e}",
            k6_hartree_per_angstrom6=f"{result['derivatives_ang'].get(6, np.nan):.12e}",
            anharmonic_E0_hartree=f"{result['anh']['eigenvalues'][0]:.12e}",
            harmonic_E0_numeric_hartree=f"{result['harm']['eigenvalues'][0]:.12e}",
            harmonic_E0_theory_hartree=f"{result['harmonic_e0_theory']:.12e}",
            harmonic_E0_relative_error=f"{result['harmonic_e0_relative_error']:.6e}",
            anharmonic_thermal_states=len(result["anh"]["included"]),
            harmonic_thermal_states=len(result["harm"]["included"]),
            probability_outside_scan_anh=f"{result['outside_anh']:.12e}",
            probability_outside_scan_harm=f"{result['outside_harm']:.12e}",
            grid_edge_density_ratio_anh=f"{result['edge_ratio_anh']:.12e}",
            grid_edge_density_ratio_harm=f"{result['edge_ratio_harm']:.12e}",
            momentum_fft_points=result["anh_momentum"]["fft_points"],
            momentum_edge_density_ratio_anh=(
                f"{result['momentum_edge_ratio_anh']:.12e}"
            ),
            momentum_edge_density_ratio_harm=(
                f"{result['momentum_edge_ratio_harm']:.12e}"
            ),
            momentum_max_kinetic_relative_error_anh=(
                f"{result['anh_momentum']['maximum_kinetic_relative_error']:.12e}"
            ),
            momentum_max_kinetic_relative_error_harm=(
                f"{result['harm_momentum']['maximum_kinetic_relative_error']:.12e}"
            ),
            harmonic_thermal_p2_numeric_au=f"{result['harmonic_p2_numeric']:.12e}",
            harmonic_thermal_p2_theory_au=f"{result['harmonic_p2_theory']:.12e}",
            harmonic_thermal_p2_relative_error=(
                f"{result['harmonic_p2_relative_error']:.12e}"
            ),
            warnings="; ".join(result["warnings"]) if result["warnings"] else "none",
        ))
    with diagnostics_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostic_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(diagnostic_rows)

    thermal_summary_path = root / config.THERMAL_SUMMARY_FILE
    with thermal_summary_path.open("w") as stream:
        stream.write(f"# Thermal Analysis at T = {args.temperature:.6f} K\n")
        stream.write("# Mode | N_states | Weights (v0, v1, ...)\n")
        for result in results:
            weights = ", ".join(f"{weight:.6f}" for weight in result["anh"]["weights"])
            stream.write(f"vib{result['mode']:<4d} | {len(result['anh']['weights']):3d} | {weights}\n")

    with PdfPages(root / config.WAVE_PDF) as pdf:
        for page_start in range(0, len(results), 3):
            page = results[page_start:page_start + 3]
            fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), squeeze=False)
            for row in range(3):
                if row >= len(page):
                    axes[row, 0].axis("off")
                    axes[row, 1].axis("off")
                    continue
                result = page[row]
                anh, harm = result["anh"], result["harm"]
                ax_wave, ax_prob = axes[row]
                ax_wave.plot(xgrid, np.abs(result["psi0_anh"]), "o", markersize=2.2, label="Anharmonic")
                ax_wave.plot(xgrid, np.abs(result["psi0_harm"]), "-", linewidth=1.2, label="Harmonic")
                ax_wave.plot(xgrid, np.sqrt(anh["thermal_density"]), "o", markersize=1.2, label="Therm_Anhar_eff")
                ax_wave.set(xlabel=r"$s$ (bohr)", ylabel=r"$|\Psi(s)|$", title=f"vib{result['mode']} Wavefunction")
                ax_wave.legend(fontsize=7)

                ax_prob.plot(xgrid, result["p0_anh"], "-", label=r"$|\Psi_0(s)|^2$")
                ax_prob.plot(xgrid, result["p0_harm"], "-", label=r"$|\Psi_0(s)|^2$ Harmonic")
                ax_prob.plot(xgrid, anh["thermal_density"], "-", label=r"$|\Psi(s,T)|^2$")
                ax_prob.plot(xgrid, harm["thermal_density"], "--", label="Harmonic thermal")
                ax_prob.axvline(result["scan_lower_bohr"], color="0.55", linestyle=":", linewidth=0.8)
                ax_prob.axvline(result["scan_upper_bohr"], color="0.55", linestyle=":", linewidth=0.8, label="a5 scan limits")
                ax_prob.set(xlabel=r"$s$ (bohr)", ylabel="Probability density", title=f"vib{result['mode']} Probability Distribution")
                ax_prob.legend(fontsize=6.5)
            fig.suptitle(f"Curvilinear AnhDis distributions at T = {args.temperature:.2f} K", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(root / config.EXCITED_STATES_PDF) as pdf:
        for result in results:
            anh = result["anh"]
            nplot = min(len(anh["included"]), int(config.A6_MAX_PLOTTED_STATES))
            states = anh["included"][:nplot]
            fig, axes = plt.subplots(nplot, 1, figsize=(8.0, 10.5), sharex=True, squeeze=False)
            for row, state in enumerate(states):
                ax = axes[row, 0]
                weight = anh["weights"][row]
                ax.plot(xgrid, anh["waves"][:, state], "b-", linewidth=1.0)
                ax.axhline(0, color="0.5", linestyle="--", linewidth=0.6)
                ax.axvline(result["scan_lower_bohr"], color="0.7", linestyle=":", linewidth=0.6)
                ax.axvline(result["scan_upper_bohr"], color="0.7", linestyle=":", linewidth=0.6)
                ax.set_ylabel(rf"$\Psi_{{{state}}}$", fontsize=8)
                ax.set_title(f"v={state}, Boltzmann weight={weight:.5f}", fontsize=8)
                ax.tick_params(labelsize=7)
            axes[-1, 0].set_xlabel(r"$s$ (bohr)")
            suffix = "" if nplot == len(anh["included"]) else f"; first {nplot} shown"
            fig.suptitle(
                f"Anharmonic Excited States - vib{result['mode']} "
                f"({len(anh['included'])} thermally retained{suffix})",
                fontsize=12,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

    with PdfPages(root / config.MOMENTUM_PDF) as pdf:
        for page_start in range(0, len(results), 3):
            page = results[page_start:page_start + 3]
            fig, axes = plt.subplots(3, 1, figsize=(8.5, 11.0), squeeze=False)
            for row in range(3):
                ax = axes[row, 0]
                if row >= len(page):
                    ax.axis("off")
                    continue
                result = page[row]
                anh_momentum = result["anh_momentum"]
                harm_momentum = result["harm_momentum"]
                p_grid = anh_momentum["grid_p_au"]
                ax.plot(
                    p_grid, anh_momentum["p0_density_per_au"], "-",
                    linewidth=1.0, label=r"$|\widetilde{\Psi}_0(p_s)|^2$",
                )
                ax.plot(
                    p_grid, harm_momentum["p0_density_per_au"], "-",
                    linewidth=1.0,
                    label=r"$|\widetilde{\Psi}_0(p_s)|^2$ harmonic",
                )
                ax.plot(
                    p_grid, anh_momentum["thermal_density_per_au"], "-",
                    linewidth=1.5, label=r"$\rho(p_s,T)$",
                )
                ax.plot(
                    p_grid, harm_momentum["thermal_density_per_au"], "--",
                    linewidth=1.2, label=r"$\rho(p_s,T)$ harmonic",
                )
                combined = np.maximum(
                    anh_momentum["thermal_density_per_au"],
                    harm_momentum["thermal_density_per_au"],
                )
                visible = combined > max(float(np.max(combined)) * 1.0e-7, 1.0e-15)
                if np.any(visible):
                    visible_p = p_grid[visible]
                    margin = max(0.1, 0.08 * (visible_p[-1] - visible_p[0]))
                    ax.set_xlim(visible_p[0] - margin, visible_p[-1] + margin)
                ax.set(
                    xlabel=r"constant-mass momentum $p_s$ (a.u.; $\hbar/a_0$)",
                    ylabel="Probability density per a.u.",
                    title=f"vib{result['mode']} Momentum Distribution",
                )
                ax.legend(fontsize=7)
            fig.suptitle(
                f"Independent AnhDis momentum marginals at T = {args.temperature:.2f} K",
                fontsize=13,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)

    metadata = dict(
        stage="curvilinear_quantum_thermal_distribution",
        coordinate_system=coordinate_system,
        coordinate="s", grid_unit="bohr", saved_coordinate_unit="angstrom_and_bohr",
        mass="mu(0)", mass_unit="amu_converted_to_electron_masses",
        hamiltonian="H=-(1/(2*mu0))*d2/ds2 + V(s)",
        potential_representation=(
            "Mode-selectable derivative polynomial or PCHIP interpolation inside the "
            "ab initio interval with confining quadratic endpoint tails"
        ),
        temperature_K=args.temperature, thermal_weight_cutoff=config.THERMAL_WEIGHT_CUTOFF,
        grid_half_length_bohr=config.L, grid_points=config.NP,
        selected_modes=chosen,
        potential_models={str(result["mode"]): result["fit_model"] for result in results},
        probability_normalization="integral P(s) ds = 1 with ds in bohr",
        momentum_definition=(
            "phi(p_s)=(2*pi)^(-1/2)*integral psi(s)*exp(-i*p_s*s) ds; "
            "rho(p_s,T)=sum_v w_v*|phi_v(p_s)|^2"
        ),
        momentum_unit="atomic momentum hbar/bohr = electron_mass*bohr/atomic_time",
        momentum_probability_normalization="integral rho(p_s,T) dp_s = 1",
        phase_space_model=(
            "positive separable product of independently sampled coordinate and momentum "
            "marginals; no anharmonic Wigner quasiprobability"
        ),
        momentum_fft_padding_factor=config.A6_MOMENTUM_FFT_PAD_FACTOR,
        future_sampling=(
            "sample s and p_s independently; convert s to angstrom; invert s(Q); "
            "recover the selected path geometry and project p_s to Cartesian velocities"
        ),
    )
    (root / "a6_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(
        f"a6: solved {len(results)} {coordinate_system} path Hamiltonians "
        f"at T={args.temperature:.2f} K."
    )
    print(f"PDF generated: {root / config.WAVE_PDF}")
    print(f"Excited states: {root / config.EXCITED_STATES_PDF}")
    print(f"Probability distributions: {probability_dir}/vib*.dat")
    print(f"Momentum distributions: {momentum_dir}/vib*.dat")
    print(f"Momentum PDF: {root / config.MOMENTUM_PDF}")
    print(f"Diagnostics: {diagnostics_path}")
    warning_count = 0
    for result in results:
        warning_text = "; ".join(result["warnings"]) if result["warnings"] else "none"
        warning_count += len(result["warnings"])
        print(
            f"Mode {result['mode']}: states={len(result['anh']['included'])}; "
            f"P_outside_scan={result['outside_anh']:.3e}; warnings={warning_text}"
        )
    if args.strict and warning_count:
        raise RuntimeError(f"a6 strict validation failed with {warning_count} warning(s).")


if __name__ == "__main__":
    run_timed_stage("a6", main, config.EXECUTION_TIMING_FILE)
