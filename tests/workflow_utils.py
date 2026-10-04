"""Shared numerical, manifest, and execution helpers for AnhDis stages."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import os
import re
import shlex
import socket
import sys
import time
from pathlib import Path

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - AnhDis production clusters are POSIX.
    fcntl = None


TIMING_FIELDS = (
    "started_at", "finished_at", "stage", "status", "elapsed_seconds",
    "elapsed_hms", "hostname", "pid", "working_directory", "command",
    "error",
)


def format_elapsed_time(seconds: float) -> str:
    """Format a duration without wrapping after 24 hours."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remainder = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remainder:06.3f}"


def _append_timing_record(path, record):
    """Append one TSV row atomically enough for independent PBS jobs."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="") as stream:
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0, os.SEEK_END)
        empty = stream.tell() == 0
        writer = csv.DictWriter(stream, fieldnames=TIMING_FIELDS, delimiter="\t")
        if empty:
            writer.writeheader()
        writer.writerow({field: record.get(field, "") for field in TIMING_FIELDS})
        stream.flush()
        os.fsync(stream.fileno())
        if fcntl is not None:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return path


def run_timed_stage(stage, main_function, timing_file):
    """Run one stage, report its wall time, and append it to the timing TSV."""
    if any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        return main_function()
    started_wall = datetime.now().astimezone()
    started_clock = time.perf_counter()
    status = "OK"
    error = ""
    try:
        return main_function()
    except BaseException as exc:
        if isinstance(exc, SystemExit) and exc.code in {None, 0}:
            status = "OK"
        else:
            status = "FAILED"
            error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        finished_wall = datetime.now().astimezone()
        elapsed = time.perf_counter() - started_clock
        formatted = format_elapsed_time(elapsed)
        record = dict(
            started_at=started_wall.isoformat(timespec="seconds"),
            finished_at=finished_wall.isoformat(timespec="seconds"),
            stage=str(stage), status=status,
            elapsed_seconds=f"{elapsed:.6f}", elapsed_hms=formatted,
            hostname=socket.gethostname(), pid=os.getpid(),
            working_directory=str(Path.cwd().resolve()),
            command=shlex.join(sys.argv), error=error,
        )
        try:
            saved = _append_timing_record(timing_file, record)
        except Exception as timing_error:  # Timing must never hide stage output.
            print(f"[TIMING WARNING] Could not save timing: {timing_error}", file=sys.stderr)
        else:
            print(f"[TIMING] {stage}: {status} in {formatted}")
            print(f"[TIMING] History: {saved}")


ELEMENT_SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn "
    "Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
    "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am "
    "Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_modes(selection, nmodes):
    if not selection:
        return list(range(1, nmodes + 1))
    result = []
    for value in selection:
        text = str(value).lower().removeprefix("vib")
        try:
            mode = int(text)
        except ValueError as exc:
            raise ValueError(f"Invalid mode selector: {value!r}") from exc
        result.append(mode)
    if len(set(result)) != len(result) or any(x < 1 or x > nmodes for x in result):
        raise ValueError(f"Select distinct modes from 1 to {nmodes}.")
    return result


def q_label(q: float) -> str:
    value = 0.0 if abs(float(q)) < 5.0e-13 else float(q)
    return f"{value:.2f}"


def read_scan_manifest(scan_root: Path):
    path = Path(scan_root) / "scan_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing a1 manifest: {path}")
    data = json.loads(path.read_text())
    if data.get("stage") != "curvilinear_scan_geometries_with_mass_coordinate":
        raise ValueError(f"Unexpected scan manifest stage in {path}")
    return data


INPUT_MANIFEST_STAGES = {
    "anhdis_single_point_inputs",
    # Backward compatibility with input manifests written before ORCA support.
    "curvilinear_gaussian_single_point_inputs",
}


def single_point_software(manifest) -> str:
    """Return the a2 electronic-structure backend recorded in a manifest."""
    software = str(manifest.get("single_point_software", "gaussian")).strip().lower()
    if software not in {"gaussian", "orca"}:
        raise ValueError(f"Unsupported single-point software in a2 manifest: {software!r}")
    return software


def read_input_manifest(scan_root: Path):
    """Read a2 metadata and prove that it belongs to the current a1 scan."""
    root = Path(scan_root)
    path = root / "input_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing a2 manifest: {path}")
    data = json.loads(path.read_text())
    if data.get("stage") not in INPUT_MANIFEST_STAGES:
        raise ValueError(f"Unexpected input manifest stage in {path}")
    single_point_software(data)
    scan_path = root / "scan_manifest.json"
    if data.get("scan_manifest_sha256") != sha256_file(scan_path):
        raise ValueError("The a2 inputs do not belong to the current a1 scan manifest.")
    points = data.get("points")
    if not isinstance(points, list) or data.get("input_count") != len(points):
        raise ValueError("Invalid a2 input count.")
    required = {
        "mode", "Q_angstrom", "xyz", "input", "xyz_sha256",
        "input_sha256", "job_tag",
    }
    if any(not isinstance(point, dict) or not required <= set(point) for point in points):
        raise ValueError("Incomplete point record in the a2 manifest.")
    xyz_names = [point["xyz"] for point in points]
    input_names = [point["input"] for point in points]
    tags = [point["job_tag"] for point in points]
    if len(set(xyz_names)) != len(points) or len(set(input_names)) != len(points) or len(set(tags)) != len(points):
        raise ValueError("Duplicate geometry, input, or job identity in the a2 manifest.")
    scan = read_scan_manifest(root)
    chosen = set(int(mode) for mode in data.get("selected_modes", []))
    if not chosen or not chosen <= set(scan["selected_modes"]):
        raise ValueError("Invalid a2 mode selection.")
    expected = {
        point["xyz"] for point in scan["energy_points"]
        if int(point["mode"]) in chosen
    }
    if set(xyz_names) != expected:
        raise ValueError("The a2 points do not exactly cover their selected a1 modes.")
    return data


def validate_input_record(scan_root: Path, record):
    """Validate the a1 geometry and a2 single-point input hashes for one point."""
    root = Path(scan_root)
    xyz_path = root / record["xyz"]
    input_path = root / record["input"]
    if not xyz_path.is_file():
        raise FileNotFoundError(f"Missing a1 geometry: {xyz_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Missing single-point input: {input_path}")
    if sha256_file(xyz_path) != record["xyz_sha256"]:
        raise ValueError(f"a1 geometry changed after a2: {xyz_path}")
    if sha256_file(input_path) != record["input_sha256"]:
        raise ValueError(f"Single-point input changed after a2: {input_path}")
    return xyz_path, input_path


def output_path_for_record(scan_root: Path, record) -> Path:
    """Return the output path recorded by a2, with legacy-manifest support."""
    root = Path(scan_root)
    if record.get("output"):
        return root / record["output"]
    return (root / record["input"]).with_suffix(".log")


def iter_scan_points(scan_root: Path, modes=None):
    root = Path(scan_root)
    manifest = read_scan_manifest(root)
    chosen = set(manifest["selected_modes"] if modes is None else modes)
    for record in manifest["energy_points"]:
        if int(record["mode"]) in chosen:
            yield record


def normal_termination(text: str, software: str = "gaussian") -> bool:
    software = str(software).strip().lower()
    if software == "gaussian":
        return "Normal termination of Gaussian" in text and "Error termination" not in text
    if software == "orca":
        return (
            "ORCA TERMINATED NORMALLY" in text
            and "ORCA finished by error termination" not in text
        )
    raise ValueError(f"Unsupported single-point software: {software!r}")


def output_contains_job_tag(log_text: str, job_tag: str) -> bool:
    """Match an a2 tag even when an electronic-structure program wraps it."""
    if not job_tag or any(character.isspace() for character in job_tag):
        raise ValueError("An a2 job tag must be nonempty and contain no whitespace.")
    return job_tag in "".join(log_text.split())


_GAUSSIAN_ENERGY_PATTERNS = {
    "SCF": re.compile(r"SCF Done:\s+E\(.+?\)\s*=\s*([-+0-9.DEde]+)"),
    "MP2": re.compile(r"\bE(?:U|R|RO)?MP2\s*=\s*([-+0-9.DEde]+)", re.IGNORECASE),
    "CCSD": re.compile(r"\bE\(CORR\)\s*=\s*([-+0-9.DEde]+)", re.IGNORECASE),
    "CCSD(T)": re.compile(r"\bCCSD\(T\)\s*=\s*([-+0-9.DEde]+)", re.IGNORECASE),
}

_ORCA_ENERGY_PATTERNS = {
    # For HF and DFT single points ORCA reports the electronic energy here.
    "SCF": re.compile(r"FINAL SINGLE POINT ENERGY\s+([-+0-9.DEde]+)"),
    "MP2": re.compile(
        r"(?:RI[- ]?)?MP2\s+TOTAL\s+ENERGY\s*[:=]\s*([-+0-9.DEde]+)",
        re.IGNORECASE,
    ),
    "CCSD": re.compile(r"\bE\(CCSD\)\s*(?:\.\.\.|=|:)\s*([-+0-9.DEde]+)", re.IGNORECASE),
    "CCSD(T)": re.compile(
        r"\bE\(CCSD\(T\)\)\s*(?:\.\.\.|=|:)\s*([-+0-9.DEde]+)",
        re.IGNORECASE,
    ),
}


def _floating(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def extract_energy(log_text: str, source: str, software: str = "gaussian"):
    """Return ``(energy_hartree, source_used)`` from Gaussian or ORCA output.

    AUTO deliberately chooses the highest-level recognized energy in the order
    CCSD(T), CCSD, MP2, SCF.  An explicit source is safer for production.
    """
    software = str(software).strip().lower()
    if software == "gaussian":
        patterns = _GAUSSIAN_ENERGY_PATTERNS
    elif software == "orca":
        patterns = _ORCA_ENERGY_PATTERNS
    else:
        raise ValueError(f"Unsupported single-point software: {software!r}")
    source = source.upper().strip()
    if source == "AUTO":
        candidates = ("CCSD(T)", "CCSD", "MP2", "SCF")
    elif source in patterns:
        candidates = (source,)
    else:
        raise ValueError("ENERGY_SOURCE must be SCF, MP2, CCSD, CCSD(T), or AUTO.")
    for name in candidates:
        matches = patterns[name].findall(log_text)
        if matches:
            value = _floating(matches[-1])
            if not np.isfinite(value):
                raise ValueError(f"Non-finite {name} energy found.")
            return value, name
    return None, None


def infer_energy_source(method_description: str) -> str:
    """Infer the supported energy marker from a Gaussian/ORCA method line.

    The ordering is intentional because CCSD(T) contains CCSD.  MP2 catches
    conventional, RI-, SCS-, SOS-, U-, RO-, and DLPNO spellings containing
    the common MP2 token.  HF, DFT, double hybrids, CASSCF, and other methods
    whose final energy is reported through the standard final-energy marker
    use the SCF reader.
    """
    method = str(method_description).upper()
    if re.search(r"CCSD\s*\(\s*T\s*\)", method):
        return "CCSD(T)"
    if "CCSD" in method:
        return "CCSD"
    if "MP2" in method:
        return "MP2"
    return "SCF"


def resolve_energy_source(requested: str, input_manifest=None) -> str:
    """Resolve AUTO from the a2 method instead of guessing from each output."""
    requested = str(requested).upper().strip()
    allowed = {"SCF", "MP2", "CCSD", "CCSD(T)", "AUTO"}
    if requested not in allowed:
        raise ValueError("ENERGY_SOURCE must be SCF, MP2, CCSD, CCSD(T), or AUTO.")
    if requested != "AUTO":
        return requested
    manifest = input_manifest or {}
    recorded = str(manifest.get("energy_source", "")).upper().strip()
    if recorded in allowed - {"AUTO"}:
        return recorded
    return infer_energy_source(manifest.get("method", ""))


def load_xyz(path: Path):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"Incomplete XYZ: {path}")
    try:
        natoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid atom count in {path}") from exc
    rows = [line.split() for line in lines[2:] if line.strip()]
    if len(rows) != natoms or any(len(row) != 4 for row in rows):
        raise ValueError(f"XYZ atom count/columns disagree in {path}")
    symbols = [row[0] for row in rows]
    try:
        coords = np.array([[float(x) for x in row[1:]] for row in rows])
    except ValueError as exc:
        raise ValueError(f"Invalid Cartesian coordinate in {path}") from exc
    if not np.isfinite(coords).all():
        raise ValueError(f"Non-finite Cartesian coordinate in {path}")
    return symbols, coords


def write_xyz(path: Path, symbols, coords, comment: str):
    coords = np.asarray(coords, float)
    if coords.shape != (len(symbols), 3) or not np.isfinite(coords).all():
        raise ValueError("Invalid XYZ data.")
    body = [str(len(symbols)), comment]
    body.extend(
        f"{symbol:2s} {row[0]: .12f} {row[1]: .12f} {row[2]: .12f}"
        for symbol, row in zip(symbols, coords)
    )
    Path(path).write_text("\n".join(body) + "\n")


def ensure_new_directory(path: Path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Use a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path
