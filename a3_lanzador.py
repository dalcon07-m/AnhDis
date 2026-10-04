#!/usr/bin/env python3
"""a3: create PBS scripts and optionally submit Gaussian or ORCA jobs."""

from __future__ import annotations

import argparse
import getpass
import os
import random
import subprocess
from pathlib import Path

import config
from workflow_utils import (
    normal_termination,
    output_contains_job_tag,
    output_path_for_record,
    read_input_manifest,
    read_scan_manifest,
    run_timed_stage,
    selected_modes,
    single_point_software,
    validate_input_record,
)

PBS_TEMPLATE_VERSION = "SIN_SET_20260908"


def pbs_script(point_dir: Path, stem: str, node: str, user: str, software: str):
    workdir = str(point_dir.resolve())
    if software == "gaussian":
        execution = f""". {config.GAUSSIAN_PROFILE}
cd $ScrDir
{config.GAUSSIAN_COMMAND} < $Wdir/$FILE.com > $ScrDir/$FILE.log
cp *.log $Wdir
cp *.o $Wdir
cp *.e $Wdir"""
    elif software == "orca":
        profile = str(getattr(config, "ORCA_PROFILE", "")).strip()
        profile_line = f". {profile}\n" if profile else ""
        command = str(getattr(config, "ORCA_COMMAND", '"$ORCA_BIN"')).strip()
        if not command:
            raise ValueError("ORCA_COMMAND cannot be empty.")
        if command == "orca":
            command = '"$ORCA_BIN"'
        return f"""#!/bin/bash
#PBS -S /bin/bash
#PBS -m be
#PBS -u {user}
#PBS -N {stem}
#PBS -l nodes={node}:ppn={config.PBS_PPN}
# AnhDis ORCA PBS template: ORCA610_20260916

export MODULEPATH=/opt/ohpc/pub/moduledeps/gnu15:/opt/ohpc/pub/modulefiles:/soft/modulefiles
module() {{ eval `/usr/bin/modulecmd bash $*`; }}
export -f module
module pure
module load orca/6.1.0

export ORCA_DIR="/soft/Orca/orca_6_1_0_linux_x86-64_shared_openmpi418"
export MPI_DIR="/soft/Openmpi/openmpi_4.1.8_gnu12"
export LD_LIBRARY_PATH=$MPI_DIR/lib64:$MPI_DIR/lib:$ORCA_DIR:$LD_LIBRARY_PATH
export PATH=$MPI_DIR/bin:$ORCA_DIR:$PATH
export ORCA_BIN="$ORCA_DIR/orca"
export OMPI_MCA_plm_rsh_args="-x LD_LIBRARY_PATH"
{profile_line}
FILE="{stem}"
: "${{PBS_JOBID:?PBS_JOBID is required}}"
ScrDir="{config.SCRATCH_ROOT.rstrip('/')}/{user}/$FILE-$PBS_JOBID"
Wdir="{workdir}"

mkdir "$ScrDir" || exit 1
cd "$ScrDir" || exit 1
cp "$Wdir/$FILE.inp" "$ScrDir/$FILE.inp" || exit 1
for xyz in "$Wdir"/*.xyz; do
    [ ! -f "$xyz" ] || cp "$xyz" "$ScrDir/" || exit 1
done

{command} "$FILE.inp" > "$FILE.log" 2>&1
orca_status=$?

# Keep scratch if copying fails or ORCA exits with an error.
cp -r "$ScrDir"/* "$Wdir/" || exit 1
if [ "$orca_status" -eq 0 ]; then
    cd "$Wdir" || exit 1
    rm -r -- "$ScrDir" || exit 1
fi
exit "$orca_status"
"""
    else:
        raise ValueError(f"Unsupported single-point software: {software!r}")
    return f"""#!/bin/bash
#PBS -u {user}
#PBS -N {stem}
#PBS -l nodes={node}:ppn={config.PBS_PPN}
#PBS -S /bin/bash
# AnhDis PBS template: {PBS_TEMPLATE_VERSION}
FILE="{stem}"
export ScrDir={config.SCRATCH_ROOT.rstrip('/')}/{user}/${{PBS_JOBID}}_$FILE
mkdir -p $ScrDir
Wdir="{workdir}"
{execution}
exit
"""


def main():
    parser = argparse.ArgumentParser(description="Prepare or submit PBS Gaussian/ORCA jobs.")
    parser.add_argument("--scan-root", type=Path, default=config.SCAN_ROOT)
    parser.add_argument("--modes", nargs="+", help="Subset of a1 modes")
    parser.add_argument("--submit", action="store_true", help="Actually call qsub; default is dry-run")
    parser.add_argument("--resubmit", action="store_true", help="Submit even when a .log already exists")
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()

    root = args.scan_root.resolve()
    scan = read_scan_manifest(root)
    inputs = read_input_manifest(root)
    software = single_point_software(inputs)
    chosen = selected_modes(args.modes, scan["n_modes"]) if args.modes else inputs["selected_modes"]
    if not set(chosen) <= set(inputs["selected_modes"]):
        raise ValueError("Requested modes are absent from the a2 inputs.")

    available = {node: config.NODE_CPUS[node] for node in config.ALLOWED_NODES if node in config.NODE_CPUS}
    if not available:
        raise ValueError("No configured PBS nodes are available.")
    nodes, weights = list(available), list(available.values())
    user = config.PBS_USER or os.environ.get("USER") or getpass.getuser()
    if not user:
        raise ValueError("Set PBS_USER or the USER environment variable.")
    rng = random.Random(args.seed)

    prepared, skipped, submitted = [], [], []
    for point in inputs["points"]:
        if int(point["mode"]) not in chosen:
            continue
        _, input_path = validate_input_record(root, point)
        stem = input_path.stem
        log_path = output_path_for_record(root, point)
        node = rng.choices(nodes, weights=weights, k=1)[0]
        script_path = input_path.parent / f"run_{stem}.pbs"
        script_path.write_text(pbs_script(input_path.parent, stem, node, user, software))
        script_path.chmod(0o750)
        prepared.append((stem, node, script_path))
        if args.submit:
            if log_path.exists() and not args.resubmit:
                text = log_path.read_text(errors="replace")
                if normal_termination(text, software):
                    if output_contains_job_tag(text, point["job_tag"]):
                        reason = "normal output already exists; use --resubmit to run it again"
                    else:
                        reason = "normal output has a wrong/missing identity; use --resubmit to replace it"
                else:
                    reason = "unfinished/error output exists; use --resubmit after inspection"
                skipped.append((stem, reason))
                continue
            try:
                result = subprocess.run(
                    [config.SUBMIT_COMMAND, str(script_path)], check=True,
                    capture_output=True, text=True,
                )
            except subprocess.CalledProcessError as exc:
                print(f"[QSUB ERROR] {stem}: command returned {exc.returncode}")
                if exc.stdout.strip():
                    print(f"stdout: {exc.stdout.strip()}")
                if exc.stderr.strip():
                    print(f"stderr: {exc.stderr.strip()}")
                raise SystemExit(exc.returncode) from exc
            submitted.append((stem, result.stdout.strip()))

    print(
        f"a3: prepared {len(prepared)} {software.upper()} PBS scripts; "
        f"skipped {len(skipped)} jobs."
    )
    for stem, reason in skipped:
        print(f"[SKIP] {stem}: {reason}")
    if args.submit:
        print(f"Submitted {len(submitted)} jobs with {config.SUBMIT_COMMAND}.")
        for stem, job_id in submitted:
            print(f"[SUBMITTED] {stem}: {job_id}")
    else:
        print("Dry run only. Inspect the .pbs files, then rerun with --submit on the cluster.")


if __name__ == "__main__":
    run_timed_stage("a3", main, config.EXECUTION_TIMING_FILE)

