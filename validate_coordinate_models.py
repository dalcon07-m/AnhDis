#!/usr/bin/env python3
"""Compare curvilinear and rectilinear Cartesian AnhDis mode paths.

With only ``--curvilinear-root``, this script performs a calculation-free
geometry screen against R(Q)=R0+Q*d_i.  With ``--cartesian-root``, it also
compares relative one-dimensional potentials without repeating electronic-
structure jobs.  The Cartesian root may be either a new a1--a4 scan tree or
an original AnhDis tree containing ``vibN/energies.dat`` tables.

Rigid translation and rotation are removed by a proper mass-weighted Kabsch
alignment before Cartesian geometry differences are reported.  The automatic
labels are screening decisions under explicit, configurable tolerances; the
pointwise TSV and PDF remain the authoritative diagnostics.
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

import config
from workflow_utils import (
    extract_energy,
    load_xyz,
    normal_termination,
    output_contains_job_tag,
    read_input_manifest,
    read_scan_manifest,
    selected_modes,
    validate_input_record,
)


def coordinate_system(manifest):
    """Read the explicit model, treating legacy manifests as curvilinear."""
    value = str(manifest.get("coordinate_system", "curvilinear")).strip().lower()
    if value not in {"curvilinear", "cartesian"}:
        raise ValueError(f"Unknown coordinate system {value!r} in scan manifest.")
    return value


def harmonic_sigma_angstrom(frequency_cm, mass_amu, temperature_K):
    """Exact thermal harmonic position standard deviation in angstrom."""
    frequency_cm = float(frequency_cm)
    mass_emass = float(mass_amu) * config.AMU_TO_EMASS
    temperature_K = float(temperature_K)
    if frequency_cm <= 0.0 or mass_emass <= 0.0 or temperature_K < 0.0:
        raise ValueError("Frequency/mass must be positive and temperature non-negative.")
    omega = frequency_cm * config.CM_TO_HARTREE
    coth = (
        1.0
        if temperature_K == 0.0
        else 1.0 / math.tanh(omega / (2.0 * config.KB_AU * temperature_K))
    )
    return math.sqrt(coth / (2.0 * mass_emass * omega)) * config.BOHR_TO_ANG


def mass_weighted_alignment(target, moving, masses):
    """Align ``moving`` to ``target`` by proper mass-weighted rotation."""
    target = np.asarray(target, float)
    moving = np.asarray(moving, float)
    masses = np.asarray(masses, float)
    if target.shape != moving.shape or target.shape != (masses.size, 3):
        raise ValueError("Geometry/mass shapes disagree during alignment.")
    weights = masses / np.sum(masses)
    target_center = np.sum(weights[:, None] * target, axis=0)
    moving_center = np.sum(weights[:, None] * moving, axis=0)
    target0 = target - target_center
    moving0 = moving - moving_center
    covariance = (moving0 * weights[:, None]).T @ target0
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    aligned = moving0 @ rotation + target_center
    return aligned


def geometry_metrics(curvilinear, cartesian, equilibrium, masses):
    aligned_cartesian = mass_weighted_alignment(curvilinear, cartesian, masses)
    delta = curvilinear - aligned_cartesian
    per_atom = np.linalg.norm(delta, axis=1)
    mass_weighted_rmsd = math.sqrt(
        float(np.sum(masses * per_atom**2) / np.sum(masses))
    )
    rmsd = math.sqrt(float(np.mean(per_atom**2)))

    aligned_equilibrium = mass_weighted_alignment(curvilinear, equilibrium, masses)
    displacement = curvilinear - aligned_equilibrium
    displacement_norm = np.linalg.norm(displacement, axis=1)
    displacement_scale = math.sqrt(
        float(np.sum(masses * displacement_norm**2) / np.sum(masses))
    )
    relative_percent = (
        100.0 * mass_weighted_rmsd / displacement_scale
        if displacement_scale > 1.0e-12 else 0.0
    )
    return rmsd, mass_weighted_rmsd, float(np.max(per_atom)), relative_percent


def record_map(manifest):
    return {
        (int(item["mode"]), round(float(item["Q_angstrom"]), 10)): item
        for item in manifest.get("energy_points", [])
    }


def mode_mass_at_zero(root, mode):
    table = np.loadtxt(Path(root) / f"vib{mode}" / "path_metric.dat")
    table = np.atleast_2d(table)
    zero = int(np.argmin(np.abs(table[:, 0])))
    if abs(table[zero, 0]) > 1.0e-10 or table[zero, 2] <= 0.0:
        raise ValueError(f"Mode {mode}: invalid Q=0 mass in path_metric.dat.")
    return float(table[zero, 2])


def legacy_energy_directory(root, mode):
    """Return the original AnhDis mode directory, accepting both old names."""
    root = Path(root)
    for candidate in (root / f"vib{mode}", root / str(mode)):
        if candidate.is_dir():
            return candidate
    return root / f"vib{mode}"


def read_legacy_energy_tables(root, modes, filename="energies.dat"):
    """Read original AnhDis two-column ``Q  E_hartree`` tables.

    The original Cartesian a5 writes one table per mode with displacement Q
    in angstrom in column 1 and total electronic energy in hartree in column 2.
    These values are deliberately treated as pre-existing results: their
    electronic method cannot be identity-checked from the table alone.
    """
    root = Path(root)
    energies = {}
    files = 0
    for mode in modes:
        table_path = legacy_energy_directory(root, mode) / filename
        if not table_path.is_file():
            continue
        try:
            table = np.loadtxt(table_path, comments="#")
        except ValueError as exc:
            raise ValueError(f"Cannot parse legacy energy table {table_path}: {exc}") from exc
        table = np.asarray(table, float)
        if table.ndim == 1:
            table = table[None, :]
        if table.ndim != 2 or table.shape[1] != 2 or table.shape[0] < 1:
            raise ValueError(
                f"Legacy energy table {table_path} must contain exactly two columns: "
                "Q_angstrom and E_total_hartree."
            )
        if not np.isfinite(table).all():
            raise ValueError(f"Legacy energy table {table_path} contains non-finite values.")
        files += 1
        for q, energy in table:
            key = (int(mode), round(float(q), 10))
            if key in energies:
                raise ValueError(f"Duplicate Q={q:g} in {table_path}.")
            energies[key] = (
                float(energy), "legacy_energies.dat", str(table_path.relative_to(root)),
            )
    status = (
        f"{len(energies)} energies read from {files} legacy {filename} tables "
        "(method identity must be checked by the user)"
    )
    return energies, status


def available_energies(root, modes, energy_source, allow_legacy_tables=False):
    """Return energies already present; missing calculations remain missing.

    A modern scan uses normally terminated, a2-tagged Gaussian outputs.  For
    an original Cartesian tree, ``energies.dat`` is accepted explicitly so
    old expensive calculations can be reused.
    """
    root = Path(root)
    try:
        inputs = read_input_manifest(root)
    except (FileNotFoundError, ValueError):
        if allow_legacy_tables:
            return read_legacy_energy_tables(root, modes)
        return {}, "a2 manifest unavailable or inconsistent"
    chosen = set(int(mode) for mode in modes)
    energies = {}
    invalid = 0
    for record in inputs.get("points", []):
        mode = int(record["mode"])
        if mode not in chosen:
            continue
        try:
            _, input_path = validate_input_record(root, record)
        except (FileNotFoundError, ValueError):
            invalid += 1
            continue
        log_path = input_path.with_suffix(".log")
        if not log_path.is_file():
            continue
        text = log_path.read_text(errors="replace")
        if not normal_termination(text) or not output_contains_job_tag(
            text, record["job_tag"],
        ):
            invalid += 1
            continue
        energy, used = extract_energy(text, energy_source)
        if energy is None:
            invalid += 1
            continue
        key = (mode, round(float(record["Q_angstrom"]), 10))
        energies[key] = (float(energy), used, str(log_path.relative_to(root)))
    table_status = None
    if allow_legacy_tables:
        table_energies, table_status = read_legacy_energy_tables(root, modes)
        for key, value in table_energies.items():
            energies.setdefault(key, value)
    status = f"{len(energies)} energies available; {invalid} invalid completed records"
    if table_status is not None and table_energies:
        status += f"; fallback: {table_status}"
    return energies, status


def cell_widths(x):
    x = np.asarray(x, float)
    if x.size == 1:
        return np.ones(1)
    edges = np.empty(x.size + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return np.diff(edges)


def write_tsv(path, rows):
    if not rows:
        raise ValueError(f"No rows available for {path}.")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Screen geometry differences and compare existing Cartesian/curvilinear potentials."
    )
    parser.add_argument("--curvilinear-root", type=Path, required=True)
    parser.add_argument(
        "--cartesian-root", type=Path,
        help=(
            "Optional Cartesian a1--a4 tree or original AnhDis root containing "
            "vibN/energies.dat; enables electronic-potential comparison"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("coordinate_model_validation"))
    parser.add_argument("--modes", nargs="+", help="Subset; default common generated modes")
    parser.add_argument("--temperature", type=float, default=config.TEMP)
    parser.add_argument("--energy-source", default=config.ENERGY_SOURCE)
    parser.add_argument("--force", action="store_true", help="Replace this validator's output files")
    args = parser.parse_args()
    if args.temperature < 0.0:
        raise ValueError("Temperature cannot be negative.")

    curv_root = args.curvilinear_root.resolve()
    curv_manifest = read_scan_manifest(curv_root)
    if coordinate_system(curv_manifest) != "curvilinear":
        raise ValueError("--curvilinear-root is not a curvilinear a1 scan.")

    cart_root = args.cartesian_root.resolve() if args.cartesian_root else None
    cart_manifest = None
    if cart_root is not None:
        if (cart_root / "scan_manifest.json").is_file():
            cart_manifest = read_scan_manifest(cart_root)
            if coordinate_system(cart_manifest) != "cartesian":
                raise ValueError("--cartesian-root is not a Cartesian a1 scan.")
            identity_fields = ("source_log_sha256", "n_atoms", "n_modes")
            disagreement = [
                field for field in identity_fields
                if curv_manifest.get(field) != cart_manifest.get(field)
            ]
            if disagreement:
                raise ValueError(
                    "The two scans do not describe the same Gaussian reference: "
                    + ", ".join(disagreement)
                )
        elif not any(
            (legacy_energy_directory(cart_root, mode) / "energies.dat").is_file()
            for mode in curv_manifest["selected_modes"]
        ):
            raise ValueError(
                "--cartesian-root has neither a Cartesian scan_manifest.json nor "
                "original vibN/energies.dat tables."
            )

    with np.load(curv_root / "reference_data.npz", allow_pickle=False) as data:
        equilibrium = np.asarray(data["equilibrium_angstrom"], float)
        masses = np.asarray(data["masses_amu"], float)
        frequencies = np.asarray(data["frequencies_cm"], float)
        normal_modes = np.asarray(data["normal_modes"], float)
        symbols = [str(item) for item in data["symbols"]]
    if equilibrium.shape != (len(masses), 3) or normal_modes.shape[1:] != equilibrium.shape:
        raise ValueError("Invalid reference_data.npz geometry dimensions.")

    available = set(int(mode) for mode in curv_manifest["selected_modes"])
    if cart_manifest is not None:
        available &= set(int(mode) for mode in cart_manifest["selected_modes"])
    chosen = selected_modes(args.modes, curv_manifest["n_modes"]) if args.modes else sorted(available)
    if not chosen or not set(chosen) <= available:
        raise ValueError("Requested modes are not present in both required scans.")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output / config.COORD_COMPARE_POINT_FILE,
        output / config.COORD_COMPARE_SUMMARY_FILE,
        output / config.COORD_COMPARE_PDF,
        output / config.COORD_COMPARE_REPORT,
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Coordinate-validation outputs already exist; use --force to replace only "
            "these files: " + ", ".join(str(path) for path in existing)
        )

    curv_records = record_map(curv_manifest)
    cart_records = record_map(cart_manifest) if cart_manifest is not None else {}
    curv_energies, curv_energy_status = available_energies(
        curv_root, chosen, args.energy_source,
    )
    if cart_root is not None:
        cart_energies, cart_energy_status = available_energies(
            cart_root, chosen, args.energy_source, allow_legacy_tables=True,
        )
    else:
        cart_energies = {}
        cart_energy_status = "no Cartesian scan root supplied"

    sigma_limit = float(config.COORD_COMPARE_THERMAL_SIGMA_LIMIT)
    geom_tol = float(config.COORD_COMPARE_GEOMETRY_RMSD_TOL_ANG)
    relative_tol = float(config.COORD_COMPARE_RELATIVE_PATH_TOL_PERCENT)
    energy_rmse_tol = float(config.COORD_COMPARE_ENERGY_WEIGHTED_RMSE_TOL_CM)
    energy_max_tol = float(config.COORD_COMPARE_ENERGY_MAX_TOL_CM)
    point_rows = []
    summary_rows = []
    plot_data = []

    for mode in chosen:
        curv_keys = sorted(
            (key for key in curv_records if key[0] == mode), key=lambda key: key[1],
        )
        if cart_manifest is not None:
            keys = [key for key in curv_keys if key in cart_records]
        else:
            keys = curv_keys
        if not keys:
            raise ValueError(f"Mode {mode}: no common Q points.")

        mu_curv = mode_mass_at_zero(curv_root, mode)
        sigma_curv = harmonic_sigma_angstrom(
            frequencies[mode - 1], mu_curv, args.temperature,
        )
        if cart_manifest is not None:
            mu_cart = mode_mass_at_zero(cart_root, mode)
        else:
            mu_cart = float(np.sum(masses[:, None] * normal_modes[mode - 1]**2))
        sigma_cart = harmonic_sigma_angstrom(
            frequencies[mode - 1], mu_cart, args.temperature,
        )

        mode_points = []
        for key in keys:
            q = float(key[1])
            curv_record = curv_records[key]
            curv_symbols, curv_geometry = load_xyz(curv_root / curv_record["xyz"])
            if curv_symbols != symbols:
                raise ValueError(f"Mode {mode}, Q={q:g}: curvilinear atom order changed.")
            if cart_manifest is None:
                cart_geometry = equilibrium + q * normal_modes[mode - 1]
                cart_s = q
                cart_geometry_source = "analytic_R0_plus_Qd"
            else:
                cart_record = cart_records[key]
                cart_symbols, cart_geometry = load_xyz(cart_root / cart_record["xyz"])
                if cart_symbols != symbols:
                    raise ValueError(f"Mode {mode}, Q={q:g}: Cartesian atom order changed.")
                cart_s = float(cart_record["s_angstrom"])
                cart_geometry_source = str(Path(cart_record["xyz"]))
            curv_s = float(curv_record["s_angstrom"])
            z_curv = abs(curv_s) / sigma_curv
            z_cart = abs(cart_s) / sigma_cart
            thermal_relevant = max(z_curv, z_cart) <= sigma_limit + 1.0e-12
            rmsd, mw_rmsd, max_atom, relative_percent = geometry_metrics(
                curv_geometry, cart_geometry, equilibrium, masses,
            )

            curv_energy = curv_energies.get(key)
            cart_energy = cart_energies.get(key)
            mode_points.append(dict(
                key=key, Q=q, curv_s=curv_s, cart_s=cart_s,
                z=max(z_curv, z_cart), thermal_relevant=thermal_relevant,
                rmsd=rmsd, mw_rmsd=mw_rmsd, max_atom=max_atom,
                relative_percent=relative_percent,
                curv_energy=curv_energy, cart_energy=cart_energy,
                cart_geometry_source=cart_geometry_source,
            ))

        zero_points = [item for item in mode_points if abs(item["Q"]) < 1.0e-10]
        curv_e0 = cart_e0 = None
        if len(zero_points) == 1:
            if zero_points[0]["curv_energy"] is not None:
                curv_e0 = zero_points[0]["curv_energy"][0]
            if zero_points[0]["cart_energy"] is not None:
                cart_e0 = zero_points[0]["cart_energy"][0]

        energy_differences = []
        for item in mode_points:
            delta_v_cm = np.nan
            if (
                curv_e0 is not None and cart_e0 is not None
                and item["curv_energy"] is not None
                and item["cart_energy"] is not None
            ):
                curv_v = item["curv_energy"][0] - curv_e0
                cart_v = item["cart_energy"][0] - cart_e0
                delta_v_cm = (cart_v - curv_v) * config.HARTREE_TO_CM
                energy_differences.append((item, float(delta_v_cm)))
            point_rows.append(dict(
                mode=mode,
                Q_angstrom=f"{item['Q']:.10f}",
                s_curvilinear_angstrom=f"{item['curv_s']:.12e}",
                s_cartesian_angstrom=f"{item['cart_s']:.12e}",
                thermal_sigma_curvilinear_angstrom=f"{sigma_curv:.12e}",
                thermal_sigma_cartesian_angstrom=f"{sigma_cart:.12e}",
                inside_thermal_sigma_window="yes" if item["thermal_relevant"] else "no",
                aligned_rmsd_angstrom=f"{item['rmsd']:.12e}",
                aligned_mass_weighted_rmsd_angstrom=f"{item['mw_rmsd']:.12e}",
                maximum_aligned_atom_difference_angstrom=f"{item['max_atom']:.12e}",
                relative_path_difference_percent=f"{item['relative_percent']:.8f}",
                deltaV_cartesian_minus_curvilinear_cm=f"{delta_v_cm:.8f}",
                cartesian_geometry_source=item["cart_geometry_source"],
            ))

        relevant = [item for item in mode_points if item["thermal_relevant"]]
        if not relevant:
            relevant = mode_points
        outer_relevant = [item for item in relevant if item["z"] >= 0.5]
        relative_sample = outer_relevant or [
            item for item in relevant if abs(item["Q"]) > 1.0e-10
        ]
        max_mw_rmsd = max(item["mw_rmsd"] for item in relevant)
        max_relative = max(
            (item["relative_percent"] for item in relative_sample), default=0.0,
        )
        geometry_equivalent = max_mw_rmsd <= geom_tol and max_relative <= relative_tol

        energy_status = "not_available"
        weighted_rmse = max_thermal_energy = full_rmse = np.nan
        potential_equivalent = None
        if len(energy_differences) >= 3:
            e_items = [item for item, _ in energy_differences]
            differences = np.asarray([value for _, value in energy_differences])
            q_values = np.asarray([item["Q"] for item in e_items])
            z_values = np.asarray([item["z"] for item in e_items])
            weights = np.exp(-0.5 * z_values**2) * cell_widths(q_values)
            weighted_rmse = math.sqrt(float(np.sum(weights * differences**2) / np.sum(weights)))
            full_rmse = math.sqrt(float(np.mean(differences**2)))
            thermal_mask = z_values <= sigma_limit + 1.0e-12
            max_thermal_energy = (
                float(np.max(np.abs(differences[thermal_mask])))
                if np.any(thermal_mask) else float(np.max(np.abs(differences)))
            )
            potential_equivalent = (
                weighted_rmse <= energy_rmse_tol
                and max_thermal_energy <= energy_max_tol
            )
            energy_status = "complete_enough_for_screen"

        if potential_equivalent is True:
            recommendation = "cartesian_supported_by_energy"
        elif potential_equivalent is False:
            recommendation = "curvilinear_recommended_by_energy"
        elif geometry_equivalent:
            recommendation = "cartesian_likely_geometry_screen_only"
        else:
            recommendation = "curvilinear_recommended_by_geometry"

        summary_rows.append(dict(
            mode=mode,
            frequency_cm=f"{frequencies[mode-1]:.8f}",
            temperature_K=f"{args.temperature:.6f}",
            common_geometry_points=len(mode_points),
            thermally_relevant_geometry_points=len(relevant),
            sigma_limit=sigma_limit,
            maximum_thermal_mass_weighted_rmsd_angstrom=f"{max_mw_rmsd:.8e}",
            maximum_thermal_relative_path_difference_percent=f"{max_relative:.6f}",
            geometry_screen_equivalent="yes" if geometry_equivalent else "no",
            common_energy_points=len(energy_differences),
            energy_status=energy_status,
            thermal_weighted_energy_rmse_cm=f"{weighted_rmse:.6f}",
            maximum_thermal_abs_deltaV_cm=f"{max_thermal_energy:.6f}",
            full_common_range_energy_rmse_cm=f"{full_rmse:.6f}",
            potential_screen_equivalent=(
                "yes" if potential_equivalent is True
                else "no" if potential_equivalent is False else "not_tested"
            ),
            recommendation=recommendation,
        ))
        plot_data.append((mode, mode_points, energy_differences, sigma_curv, sigma_cart))

    write_tsv(output_paths[0], point_rows)
    write_tsv(output_paths[1], summary_rows)
    with PdfPages(output_paths[2]) as pdf:
        for mode, mode_points, energy_differences, sigma_curv, sigma_cart in plot_data:
            q = np.asarray([item["Q"] for item in mode_points])
            rmsd = np.asarray([item["mw_rmsd"] for item in mode_points])
            relative = np.asarray([item["relative_percent"] for item in mode_points])
            relevant = np.asarray([item["thermal_relevant"] for item in mode_points])
            fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.5), sharex=True)
            axes[0].plot(q, rmsd, "o-", label="mass-weighted aligned RMSD")
            axes[0].axhline(geom_tol, color="tab:red", linestyle="--", label="screen tolerance")
            axes[0].set_ylabel("RMSD (angstrom)")
            axes[0].grid(alpha=0.25)
            axes[0].legend()
            second = axes[0].twinx()
            second.plot(q, relative, "s:", color="tab:orange", label="relative path difference")
            second.set_ylabel("relative difference (%)", color="tab:orange")
            if np.any(relevant):
                axes[0].scatter(q[relevant], rmsd[relevant], s=65, facecolors="none", edgecolors="k")

            if energy_differences:
                energy_q = np.asarray([item["Q"] for item, _ in energy_differences])
                delta_v = np.asarray([value for _, value in energy_differences])
                axes[1].plot(energy_q, delta_v, "o-", color="tab:purple")
                axes[1].axhline(energy_max_tol, color="tab:red", linestyle="--")
                axes[1].axhline(-energy_max_tol, color="tab:red", linestyle="--")
                axes[1].set_ylabel(r"$V_{cart}-V_{curv}$ (cm$^{-1}$)")
            else:
                axes[1].text(
                    0.5, 0.5, "No paired validated energies available",
                    ha="center", va="center", transform=axes[1].transAxes,
                )
                axes[1].set_ylabel("energy comparison")
            axes[1].set_xlabel("Q (angstrom)")
            axes[1].grid(alpha=0.25)
            fig.suptitle(
                f"Mode {mode}: Cartesian vs curvilinear\n"
                f"thermal sigma: curv={sigma_curv:.4g} A, cart={sigma_cart:.4g} A"
            )
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    counts = {}
    for row in summary_rows:
        counts[row["recommendation"]] = counts.get(row["recommendation"], 0) + 1
    report = [
        "AnhDis Cartesian-versus-curvilinear validation",
        "",
        f"Curvilinear scan: {curv_root}",
        f"Cartesian source: {cart_root if cart_root is not None else 'analytic geometry only'}",
        f"Modes: {chosen}",
        f"Temperature: {args.temperature:.6f} K; thermal window: +/-{sigma_limit:g} sigma",
        f"Geometry tolerances: MW-RMSD <= {geom_tol:g} A and relative path difference <= {relative_tol:g}%",
        f"Energy tolerances: weighted RMSE <= {energy_rmse_tol:g} cm-1 and thermal max <= {energy_max_tol:g} cm-1",
        f"Curvilinear energy status: {curv_energy_status}",
        f"Cartesian energy status: {cart_energy_status}",
        "",
        "Recommendations:",
    ]
    report.extend(f"  {name}: {count}" for name, count in sorted(counts.items()))
    report.extend((
        "",
        "Interpretation:",
        "  geometry_screen_only predicts whether path curvature is small in the populated region;",
        "  it does not prove energetic equivalence. A paired energy comparison is the stronger test.",
        "  Automatic labels apply only under the explicit tolerances printed above.",
    ))
    output_paths[3].write_text("\n".join(report) + "\n")

    print("\n".join(report))
    print(f"Pointwise diagnostics: {output_paths[0]}")
    print(f"Mode summary: {output_paths[1]}")
    print(f"Plots: {output_paths[2]}")


if __name__ == "__main__":
    main()
