"""Create geometry-free Gaussian/ORCA single-point templates from outputs."""

from __future__ import annotations

import re
from pathlib import Path


_TASK_PREFIXES = (
    "opt", "copt", "tightopt", "verytightopt", "looseopt",
    "freq", "anfreq", "numfreq", "numericalfreq", "irc", "scan",
    "engrad", "numgrad", "hess",
)

_ORCA_TASK_BLOCKS = {
    "coords", "freq", "geom", "goat", "irc", "md", "mep", "neb", "plots",
}

_ORCA_EXTERNAL_DEPENDENCIES = ("%moinp", "%pointcharges")


def _without_task_keywords(simple_line: str) -> str:
    tokens = simple_line.split()
    retained = []
    for token in tokens:
        normalized = token.lower().lstrip("#!")
        if any(
            normalized == prefix
            or normalized.startswith(prefix + "=")
            or normalized.startswith(prefix + "(")
            for prefix in _TASK_PREFIXES
        ):
            continue
        retained.append(token)
    return " ".join(retained)


def create_orca_header_from_output(source_output, destination):
    """Create an ORCA SP template from the input echoed in an ORCA output."""
    source_output = Path(source_output)
    destination = Path(destination)
    text = source_output.read_text(errors="replace")
    echoed = []
    for line in text.splitlines():
        match = re.match(r"^\s*\|\s*\d+>\s?(.*)$", line)
        if match:
            echoed.append(match.group(1).rstrip())
        elif echoed:
            break
    simple_lines = [line.strip() for line in echoed if line.lstrip().startswith("!")]
    if not simple_lines:
        raise ValueError(
            "Could not derive an ORCA header: the output does not echo a '! ...' line."
        )
    simple = "! " + " ".join(line.lstrip()[1:].strip() for line in simple_lines)
    simple = _without_task_keywords(simple)
    if not any(token.lower() == "sp" for token in simple.split()):
        simple += " SP"

    coordinate = next((line.strip() for line in echoed if re.match(
        r"\s*\*\s*xyz(?:file)?\s+[+-]?\d+\s+[1-9]\d*", line, re.I,
    )), None)
    if coordinate:
        match = re.match(
            r"\s*\*\s*xyz(?:file)?\s+([+-]?\d+)\s+([1-9]\d*)",
            coordinate, re.I,
        )
        charge, multiplicity = match.groups()
    else:
        charge_matches = re.findall(
            r"Total\s+Charge.*?\.\.\.\.\s*([+-]?\d+)", text, re.I,
        )
        multiplicity_matches = re.findall(
            r"Multiplicity(?:\s+Mult)?.*?\.\.\.\.\s*([1-9]\d*)", text, re.I,
        )
        if not charge_matches or not multiplicity_matches:
            raise ValueError(
                "Could not derive the ORCA charge and multiplicity from the echoed input "
                "or output summary."
            )
        charge, multiplicity = charge_matches[-1], multiplicity_matches[-1]

    echoed_lower = "\n".join(echoed).lower()
    dependency = next(
        (name for name in _ORCA_EXTERNAL_DEPENDENCIES if name in echoed_lower), None,
    )
    if dependency:
        raise ValueError(
            f"Automatic ORCA-header generation found the external dependency {dependency}. "
            "Set AUTO_GENERATE_SINGLE_POINT_HEADER=False and provide a self-contained "
            "manual template."
        )

    # Preserve electronic-structure blocks (for example %scf, %basis, %cpcm,
    # %mdci, or %casscf), but replace %pal and remove geometry/frequency task
    # blocks. This keeps the single-point method faithful without copying the
    # optimized molecule or an optimization request into every scan input.
    retained_blocks = []
    index = 0
    while index < len(echoed):
        stripped = echoed[index].strip()
        lower = stripped.lower()
        if not stripped or stripped.startswith("!") or "****end of input****" in lower:
            index += 1
            continue
        if re.match(r"^\*\s*xyz(?:file)?\b", stripped, re.I):
            if not re.match(r"^\*\s*xyzfile\b", stripped, re.I):
                index += 1
                while index < len(echoed) and echoed[index].strip() != "*":
                    index += 1
                if index < len(echoed):
                    index += 1
            else:
                index += 1
            continue
        block_match = re.match(r"^%(\w+)", stripped)
        if block_match:
            block_name = block_match.group(1).lower()
            if block_name == "maxcore":
                retained_blocks.append(stripped)
                index += 1
                continue
            if block_name == "base":
                # A fixed basename would make independent scan jobs overwrite
                # one another; a3 already supplies a unique input basename.
                index += 1
                continue
            block = [stripped]
            index += 1
            if not re.search(r"\bend\s*$", stripped, re.I):
                nested_basis_sections = 0
                while index < len(echoed):
                    block.append(echoed[index].rstrip())
                    index += 1
                    if block_name == "basis" and re.match(
                        r"^\s*new(?:aux|ecp|gto|auxj|auxc|cab)", block[-1], re.I,
                    ):
                        nested_basis_sections += 1
                    if re.match(r"^\s*end\s*$", block[-1], re.I):
                        if nested_basis_sections:
                            nested_basis_sections -= 1
                        else:
                            break
            if block_name not in _ORCA_TASK_BLOCKS | {"pal"}:
                retained_blocks.append("\n".join(block))
            continue
        # Preserve standalone ORCA directives/comments but not atom rows.
        if stripped.startswith(("#", "//")):
            index += 1
            continue
        index += 1

    body = [simple, "", "%pal", "  nprocs {nprocs}", "end"]
    for block in retained_blocks:
        body.extend(("", block))
    body.extend(("", "# {job_tag}", f"* xyz {charge} {multiplicity}"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(body) + "\n")
    return destination


def _gaussian_route(text: str) -> str:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("#"):
            continue
        route = [line.strip()]
        for following in lines[index + 1:]:
            stripped = following.strip()
            if not stripped or re.fullmatch(r"-+", stripped):
                break
            route.append(stripped)
        return " ".join(route)
    raise ValueError("Could not find the Gaussian route section in the output.")


def create_gaussian_header_from_output(source_output, destination):
    """Create a Gaussian SP template from an optimization/frequency log."""
    source_output = Path(source_output)
    destination = Path(destination)
    text = source_output.read_text(errors="replace")
    route = _without_task_keywords(_gaussian_route(text))
    if re.search(r"\b(gen|genecp)\b", route, re.IGNORECASE):
        raise ValueError(
            "Automatic Gaussian-header generation cannot reconstruct a Gen/GenECP "
            "basis section. Set AUTO_GENERATE_SINGLE_POINT_HEADER=False and provide "
            "a complete manual template."
        )
    if re.search(r"\b(geom|units)\s*=", route, re.IGNORECASE):
        raise ValueError(
            "Automatic Gaussian-header generation found Geom/Units controls that are "
            "unsafe for independent Cartesian single points. Use a manual template."
        )
    if not re.search(r"(?:^|\s)sp(?:\s|$)", route, re.IGNORECASE):
        tokens = route.split()
        tokens.insert(1, "sp")
        route = " ".join(tokens)
    charge_multiplicity = re.findall(
        r"Charge\s*=\s*([+-]?\d+)\s+Multiplicity\s*=\s*([1-9]\d*)",
        text,
        re.IGNORECASE,
    )
    if not charge_multiplicity:
        raise ValueError("Could not find Gaussian charge and multiplicity in the output.")
    charge, multiplicity = charge_multiplicity[-1]
    memory = re.findall(r"(?im)^\s*%mem\s*=\s*(\S+)", text)
    body = ["%chk={checkpoint}.chk", "%nprocshared={nprocs}"]
    if memory:
        body.append(f"%mem={memory[0]}")
    body.extend((
        route,
        "",
        "AnhDis single-point calculation",
        "{job_tag}",
        "",
        f"{charge} {multiplicity}",
    ))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(body) + "\n")
    return destination


def generate_single_point_header(source_output, destination, software):
    """Dispatch automatic template generation for one supported program."""
    software = str(software).strip().lower()
    if software == "gaussian":
        return create_gaussian_header_from_output(source_output, destination)
    if software == "orca":
        return create_orca_header_from_output(source_output, destination)
    raise ValueError(f"Unsupported single-point software: {software!r}")
