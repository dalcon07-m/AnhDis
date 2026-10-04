#!/usr/bin/env python3
"""a2: create one independent Gaussian or ORCA input per a1 geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import config
from single_point_templates import generate_single_point_header
from workflow_utils import (
    infer_energy_source,
    iter_scan_points,
    load_xyz,
    q_label,
    read_scan_manifest,
    run_timed_stage,
    selected_modes,
    sha256_file,
    single_point_software,
)


def validate_gaussian_header(path: Path):
    text = Path(path).read_text().rstrip()
    lines = text.splitlines()
    route_start = next((i for i, line in enumerate(lines) if line.lstrip().startswith("#")), None)
    if route_start is None:
        raise ValueError("Gaussian header has no route section.")
    route_lines = []
    for line in lines[route_start:]:
        if not line.strip():
            break
        route_lines.append(line.strip())
    route = " ".join(route_lines)
    if not re.fullmatch(r"[+-]?\d+\s+[1-9]\d*", lines[-1].strip()):
        raise ValueError("Gaussian header must end with charge and multiplicity, without geometry.")
    forbidden = r"\b(opt|freq|irc|scan|geom|units|guess\s*=\s*read|gen|genecp)\b"
    if re.search(forbidden, route, re.IGNORECASE):
        raise ValueError("a2 requires an independent Cartesian single point in default angstrom units.")
    if "--link1--" in text.lower():
        raise ValueError("Use a single Gaussian job in gaussian_head.com.")
    chk_lines = [line for line in lines if re.match(r"\s*%chk\s*=", line, re.IGNORECASE)]
    if len(chk_lines) != 1 or "{checkpoint}" not in chk_lines[0]:
        raise ValueError("Use exactly one %chk={checkpoint}.chk line.")
    if text.count("{job_tag}") != 1:
        raise ValueError("Gaussian header must contain {job_tag} exactly once in its title section.")
    tag_line = next(i for i, line in enumerate(lines) if "{job_tag}" in line)
    if tag_line <= route_start or tag_line == len(lines) - 1:
        raise ValueError("Place {job_tag} in the Gaussian title, between the route and charge/multiplicity.")
    if any(re.match(r"\s*%(oldchk|rwf)\s*=", line, re.IGNORECASE) for line in lines):
        raise ValueError("External checkpoint or RWF dependencies are not supported.")
    return text, route


def validate_orca_header(path: Path):
    """Validate a geometry-free ORCA single-point template."""
    text = Path(path).read_text().rstrip()
    lines = text.splitlines()
    simple_lines = [line.strip() for line in lines if line.lstrip().startswith("!")]
    if not simple_lines:
        raise ValueError("ORCA header has no simple input line beginning with '!'.")
    simple_input = " ".join(simple_lines)
    forbidden = (
        r"\b(opt|copt|tightopt|verytightopt|looseopt|freq|anfreq|numfreq|"
        r"numericalfreq|irc|scan|engrad|numgrad|hess)\b"
    )
    if re.search(forbidden, simple_input, re.IGNORECASE):
        raise ValueError("a2 requires an ORCA single point: remove Opt/Freq/IRC/Scan keywords.")
    if text.count("{job_tag}") != 1:
        raise ValueError("ORCA header must contain {job_tag} exactly once in a comment.")
    if text.count("{nprocs}") != 1:
        raise ValueError("ORCA header must contain {nprocs} exactly once in its %pal block.")
    if not re.search(
        r"(?ims)^\s*%pal\b.*?\bnprocs\s+\{nprocs\}.*?^\s*end\s*$", text,
    ):
        raise ValueError("Place 'nprocs {nprocs}' inside an ORCA %pal ... end block.")
    if not re.fullmatch(
        r"\s*\*\s*xyz\s+[+-]?\d+\s+[1-9]\d*\s*", lines[-1], re.IGNORECASE,
    ):
        raise ValueError(
            "ORCA header must end with '* xyz charge multiplicity', without geometry."
        )
    if re.search(r"\bxyzfile\b", text, re.IGNORECASE):
        raise ValueError("ORCA XYZFILE inputs are not supported; a2 inserts Cartesian coordinates.")
    return text, simple_input


def normalize_single_point_software(value):
    software = str(value).strip().lower()
    if software not in {"auto", "gaussian", "orca"}:
        raise ValueError("Single-point software must be auto, gaussian, or orca.")
    return software


def resolve_single_point_software(requested, scan):
    """Resolve ``auto`` to the program that produced the a1 normal modes."""
    software = normalize_single_point_software(requested)
    if software == "auto":
        software = str(scan.get("opt_freq_software", "gaussian")).strip().lower()
    if software not in {"gaussian", "orca"}:
        raise ValueError("Cannot infer the a2 software from the a1 manifest; select it explicitly.")
    return software


def configured_header(software):
    if software == "gaussian":
        return Path(config.GAUSSIAN_HEADER)
    value = getattr(config, "ORCA_HEADER", Path(config.PROJECT_ROOT) / "orca_head.inp")
    return Path(value)


def main():
    parser = argparse.ArgumentParser(description="Generate Gaussian or ORCA inputs for a1 geometries.")
    parser.add_argument("--scan-root", type=Path, default=config.SCAN_ROOT)
    parser.add_argument(
        "--software", choices=("auto", "gaussian", "orca"),
        default=getattr(config, "SINGLE_POINT_SOFTWARE", "gaussian"),
    )
    parser.add_argument("--header", type=Path, default=None, help="Override the active program template")
    parser.add_argument("--modes", nargs="+", help="Subset of modes already generated by a1")
    parser.add_argument("--force", action="store_true", help="Replace existing input files and a2 manifest")
    parser.add_argument(
        "--update", action="store_true",
        help="Add inputs for extended modes while preserving existing input/output identities",
    )
    args = parser.parse_args()

    if args.force and args.update:
        raise ValueError("Choose either --force or --update, not both.")

    root = args.scan_root.resolve()
    scan = read_scan_manifest(root)
    chosen = selected_modes(args.modes, scan["n_modes"]) if args.modes else scan["selected_modes"]
    missing = sorted(set(chosen) - set(scan["selected_modes"]))
    if missing:
        raise ValueError(f"Modes were not generated by a1: {missing}")
    software = resolve_single_point_software(args.software, scan)
    header_path = (args.header or configured_header(software)).resolve()
    automatic_header = bool(getattr(config, "AUTO_GENERATE_SINGLE_POINT_HEADER", False))
    if automatic_header:
        # Manifests written before ORCA support did not record this field and
        # necessarily came from Gaussian.
        source_software = str(scan.get("opt_freq_software", "gaussian")).strip().lower()
        if source_software != software:
            raise ValueError(
                "Automatic header generation requires the a1 source and a2 program to match. "
                "Set AUTO_GENERATE_SINGLE_POINT_HEADER=False and provide a manual template "
                "for a mixed Gaussian/ORCA workflow."
            )
        source_output = Path(scan.get("source_log", ""))
        if not source_output.is_file():
            raise FileNotFoundError(
                f"Cannot generate the {software.upper()} header; a1 source output is missing: "
                f"{source_output}"
            )
        generate_single_point_header(source_output, header_path, software)
        print(f"a2: regenerated {software.upper()} header from {source_output.name}: {header_path}")
    if not header_path.is_file():
        raise FileNotFoundError(f"Missing {software.upper()} single-point header: {header_path}")
    if software == "gaussian":
        header, method_description = validate_gaussian_header(header_path)
        input_suffix = ".com"
    else:
        header, method_description = validate_orca_header(header_path)
        input_suffix = ".inp"
    inferred_energy_source = infer_energy_source(method_description)

    manifest_path = root / "input_manifest.json"
    old_manifest = None
    if args.update:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"--update requires an existing a2 manifest: {manifest_path}")
        old_manifest = json.loads(manifest_path.read_text())
        if old_manifest.get("stage") not in {
            "curvilinear_gaussian_single_point_inputs", "anhdis_single_point_inputs",
        }:
            raise ValueError("The existing input manifest is not an a2 manifest.")
        if single_point_software(old_manifest) != software:
            raise ValueError(
                "The existing a2 manifest uses a different single-point program. "
                "Use a new SCAN_ROOT instead of mixing Gaussian and ORCA outputs."
            )
        old_header_hash = old_manifest.get("header_sha256")
        if old_header_hash != sha256_file(header_path) and set(chosen) != set(scan["selected_modes"]):
            raise ValueError(
                f"The {software.upper()} header changed. Update every mode together or retain the original header."
            )

    outputs = []
    records = []
    for point in iter_scan_points(root, chosen):
        xyz_path = root / point["xyz"]
        symbols, coords = load_xyz(xyz_path)
        if len(symbols) != scan["n_atoms"]:
            raise ValueError(f"Atom count differs from a1 manifest: {xyz_path}")
        label = f"vib{point['mode']}_{float(point['Q_angstrom']):.2f}"
        output = xyz_path.with_suffix(input_suffix)
        atom_lines = [
            f"{symbol:2s} {row[0]: .12f} {row[1]: .12f} {row[2]: .12f}"
            for symbol, row in zip(symbols, coords)
        ]
        xyz_digest = sha256_file(xyz_path)
        qtag = q_label(float(point["Q_angstrom"])).replace("-", "m").replace(".", "p")
        job_tag = f"ANHDIS_M{int(point['mode']):03d}_Q{qtag}_{xyz_digest[:16]}"
        if software == "gaussian":
            content = (
                header.replace("{checkpoint}", label)
                .replace("{job_tag}", job_tag)
                .replace("{nprocs}", str(config.PBS_PPN))
                + "\n" + "\n".join(atom_lines) + "\n\n"
            )
            result_suffixes = (".log", ".out", ".chk")
        else:
            content = (
                header.replace("{job_tag}", job_tag).replace("{nprocs}", str(config.PBS_PPN))
                + "\n" + "\n".join(atom_lines) + "\n*\n"
            )
            result_suffixes = (".log", ".out", ".gbw", ".property.txt")
        result_files = [output.with_suffix(suffix) for suffix in result_suffixes]
        if args.force and any(path.exists() for path in result_files):
            existing = ", ".join(str(path) for path in result_files if path.exists())
            raise FileExistsError(
                f"Refusing to replace a {software.upper()} input that already has results. "
                f"Use a new scan directory after inspecting: {existing}"
            )
        if output.exists() and not args.force:
            if output.read_text() == content:
                action = "unchanged"
            else:
                raise FileExistsError(f"Existing input differs; use --force only before calculations: {output}")
        else:
            action = "write"
        outputs.append((output, content, action))
        records.append(dict(
            mode=int(point["mode"]),
            Q_angstrom=float(point["Q_angstrom"]),
            s_angstrom=float(point["s_angstrom"]),
            mu_amu=float(point["mu_amu"]),
            xyz=str(xyz_path.relative_to(root)),
            input=str(output.relative_to(root)),
            output=str(output.with_suffix(".log").relative_to(root)),
            xyz_sha256=xyz_digest,
            input_sha256=hashlib.sha256(content.encode()).hexdigest(),
            job_tag=job_tag,
        ))

    if old_manifest is not None:
        chosen_set = set(chosen)
        records = [
            point for point in old_manifest.get("points", [])
            if int(point["mode"]) not in chosen_set
        ] + records
        records.sort(key=lambda item: (int(item["mode"]), float(item["Q_angstrom"])))
        selected_for_manifest = sorted(set(old_manifest.get("selected_modes", [])) | chosen_set)
    else:
        selected_for_manifest = list(chosen)

    if manifest_path.exists() and not args.force and not args.update:
        old = json.loads(manifest_path.read_text())
        if old.get("points") != records or old.get("header_sha256") != sha256_file(header_path):
            raise FileExistsError("Existing input_manifest.json differs; use --force deliberately.")
    for output, content, action in outputs:
        if action == "write":
            output.write_text(content)
    manifest = dict(
        stage="anhdis_single_point_inputs",
        single_point_software=software,
        scan_manifest_sha256=sha256_file(root / "scan_manifest.json"),
        header_file=str(header_path),
        header_sha256=sha256_file(header_path),
        method=method_description,
        energy_source=inferred_energy_source,
        selected_modes=selected_for_manifest,
        input_count=len(records),
        points=records,
        jobs_submitted=False,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"a2: {len(records)} {software.upper()} inputs ready in {root}")
    print(f"Electronic energy selected from method: {inferred_energy_source}")
    print("No calculations were submitted.")


if __name__ == "__main__":
    run_timed_stage("a2", main, config.EXECUTION_TIMING_FILE)
