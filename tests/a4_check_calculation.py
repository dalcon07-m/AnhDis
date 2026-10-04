#!/usr/bin/env python3
"""a4: verify Gaussian/ORCA termination and the requested electronic energy."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description="Check all Gaussian/ORCA single points.")
    parser.add_argument("--scan-root", type=Path, default=config.SCAN_ROOT)
    parser.add_argument("--modes", nargs="+", help="Subset of a1 modes")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any point is not OK")
    args = parser.parse_args()

    root = args.scan_root.resolve()
    scan = read_scan_manifest(root)
    inputs = read_input_manifest(root)
    software = single_point_software(inputs)
    energy_source = resolve_energy_source(config.ENERGY_SOURCE, inputs)
    chosen = selected_modes(args.modes, scan["n_modes"]) if args.modes else inputs["selected_modes"]
    if not set(chosen) <= set(inputs["selected_modes"]):
        raise ValueError("Requested modes are absent from the a2 inputs.")
    input_by_xyz = {point["xyz"]: point for point in inputs["points"]}
    rows = []
    for point in iter_scan_points(root, chosen):
        xyz_path = root / point["xyz"]
        input_record = input_by_xyz.get(point["xyz"])
        input_path = (
            xyz_path.with_suffix(".com" if software == "gaussian" else ".inp")
            if input_record is None else root / input_record["input"]
        )
        log_path = (
            input_path.with_suffix(".log")
            if input_record is None else output_path_for_record(root, input_record)
        )
        energy, used = None, None
        status, detail = None, ""
        if input_record is None:
            status, detail = "NO_INPUT", "point absent from input_manifest.json"
        else:
            try:
                validate_input_record(root, input_record)
            except (FileNotFoundError, ValueError) as exc:
                status, detail = "INPUT_MISMATCH", str(exc)
        if status is not None:
            pass
        elif not log_path.exists():
            status, detail = "MISSING", "no .log file"
        else:
            text = log_path.read_text(errors="replace")
            energy, used = extract_energy(text, energy_source, software)
            error_marker = (
                "Error termination" if software == "gaussian"
                else "ORCA finished by error termination"
            )
            if error_marker in text:
                status, detail = "ERROR", f"{software.upper()} error termination"
            elif not normal_termination(text, software):
                status, detail = "INCOMPLETE", "normal termination not found"
            elif energy is None:
                status, detail = "NO_ENERGY", f"no {energy_source} energy found"
            elif not output_contains_job_tag(text, input_record["job_tag"]):
                status, detail = (
                    "OUTPUT_MISMATCH",
                    f"{software.upper()} output has the wrong/missing a2 identity",
                )
                energy, used = None, None
            else:
                status, detail = "OK", ""
        rows.append(dict(
            mode=int(point["mode"]), Q_angstrom=float(point["Q_angstrom"]),
            s_angstrom=float(point["s_angstrom"]), status=status,
            energy_source=used or "", energy_hartree="" if energy is None else f"{energy:.15f}",
            software=software, input=str(input_path.relative_to(root)),
            log=str(log_path.relative_to(root)), detail=detail,
        ))
        print(f"[{status}] vib{point['mode']} Q={float(point['Q_angstrom']):+.2f} {detail}")

    output = root / config.CALCULATION_STATUS_FILE
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    counts = {status: sum(row["status"] == status for row in rows) for status in sorted({r["status"] for r in rows})}
    print("Summary: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Energy selection: requested={config.ENERGY_SOURCE}; resolved={energy_source}")
    print(f"Saved {output}")
    if args.strict and counts.get("OK", 0) != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    run_timed_stage("a4", main, config.EXECUTION_TIMING_FILE)
