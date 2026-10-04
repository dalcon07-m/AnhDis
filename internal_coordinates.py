"""AnhDis internal-coordinate diagnostics, finite geometries and path masses.

No electronic-energy calculation or quantum Hamiltonian here.
Positions: angstrom; angles: radians internally (degrees only in readable reports).
The normal vectors received from a1 are neither normalized nor modified.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import numpy as np


def resolve_mode_selection(selection, nmodes):
    """Select vibrational mode indices, with molecule-independent ``all``."""
    if list(selection) == ["all"]:
        return list(range(1,nmodes+1))
    try:
        modes = [int(x) for x in selection]
    except (ValueError,TypeError) as exc:
        raise ValueError("Use integer mode numbers or the single selector 'all'.") from exc
    if not modes or len(set(modes)) != len(modes) or any(not 1 <= x <= nmodes for x in modes):
        raise ValueError(f"Select distinct mode numbers from 1 to {nmodes}, or 'all'.")
    return modes


@dataclass(frozen=True)
class Internal:
    name: str
    kind: str
    atoms: tuple  # zero-based internally
    printed_value: float  # angstrom or degrees in the source program's table
    component: int | None = None  # Gaussian L component: -1 or -2
    axis_reference: int | None = None  # Gaussian automatic L reference: -1 or -2
    frame_axis: tuple | None = None  # equilibrium molecular axis for automatic L
    frame_direction_1: tuple | None = None  # first perpendicular frame direction

    @property
    def label(self):
        arguments = [str(i + 1) for i in self.atoms]
        if self.kind == "L":
            if self.axis_reference is not None:
                arguments.append(str(self.axis_reference))
            arguments.append(str(self.component))
        return f"{self.name}:{self.kind}({','.join(arguments)})"


def _read_gaussian_internals(log_text, natoms, reference_coords=None):
    """Read the last Gaussian Initial Parameters table before the frequencies.

    Supports R/A/D primitives and both Gaussian forms of the two-component
    linear bend: L(i,j,k,l,M), with a real reference atom l, and
    L(i,j,k,-1/-2,M), with Gaussian's automatic transverse axes. Dummy atoms,
    generalized constraints and incomplete linear-bend pairs are rejected.
    """
    frequency_start = log_text.find("Harmonic frequencies (cm")
    if frequency_start < 0:
        raise ValueError("No harmonic analysis found.")
    prefix = log_text[:frequency_start]
    start = prefix.rfind("Initial Parameters")
    if start < 0:
        raise ValueError("No Gaussian Initial Parameters table; explicit internals will be needed.")
    end = prefix.find("Trust Radius", start)
    if end < 0:
        raise ValueError("Incomplete Initial Parameters table.")
    pattern = re.compile(r"!\s+(\w+)\s+([A-Za-z]+)\(([^)]+)\)\s+([-+\d.EeDd]+)")
    result = []
    for name, kind, atom_text, value in pattern.findall(prefix[start:end]):
        if kind not in {"R", "A", "D", "L"}:
            raise ValueError(f"Unsupported internal coordinate {name}: {kind}")
        arguments = tuple(int(x.strip()) for x in atom_text.split(","))
        component = None
        axis_reference = None
        frame_axis = None
        frame_direction_1 = None
        if kind == "L":
            if len(arguments) != 5 or arguments[-1] not in {-1, -2}:
                raise ValueError(f"Invalid Gaussian linear bend {name}")
            component = arguments[-1]
            if arguments[3] > 0:
                atom_arguments = arguments[:4]
            elif arguments[3] in {-1, -2}:
                atom_arguments = arguments[:3]
                axis_reference = arguments[3]
                if reference_coords is None:
                    raise ValueError(
                        f"Automatic-axis Gaussian linear bend {name} requires "
                        "the equilibrium Cartesian geometry."
                    )
                reference_coords = np.asarray(reference_coords, float)
                if reference_coords.shape != (natoms, 3):
                    raise ValueError("Invalid equilibrium geometry for a linear bend.")
                i_atom, j_atom, _ = (index - 1 for index in atom_arguments)
                axis = unit(reference_coords[i_atom] - reference_coords[j_atom])
                seeds = np.eye(3)
                seed = seeds[int(np.argmin(np.abs(seeds @ axis)))]
                direction_1 = unit(seed - np.dot(seed, axis) * axis)
                frame_axis = tuple(float(value) for value in axis)
                frame_direction_1 = tuple(float(value) for value in direction_1)
            else:
                raise ValueError(
                    f"Unsupported Gaussian linear-bend reference in {name}: {arguments[3]}"
                )
        else:
            expected = {"R": 2, "A": 3, "D": 4}[kind]
            if len(arguments) != expected:
                raise ValueError(f"Invalid internal coordinate {name}")
            atom_arguments = arguments
        atoms = tuple(index - 1 for index in atom_arguments)
        if len(set(atoms)) != len(atoms):
            raise ValueError(f"Invalid internal coordinate {name}")
        if min(atoms) < 0 or max(atoms) >= natoms:
            raise ValueError(f"Atom index out of range in {name}")
        result.append(Internal(
            name, kind, atoms, float(value.replace("D", "E")),
            component=component, axis_reference=axis_reference,
            frame_axis=frame_axis, frame_direction_1=frame_direction_1,
        ))
    if not result or len({x.name for x in result}) != len(result):
        raise ValueError("Empty or duplicate internal-coordinate definitions.")
    linear_groups = {}
    for item in result:
        if item.kind == "L":
            key = item.atoms
            linear_groups.setdefault(key, set()).add(item.component)
    incomplete = [key for key, components in linear_groups.items() if components != {-1, -2}]
    if incomplete:
        raise ValueError("A Gaussian linear bend is missing one orthogonal component.")
    return result


def _orca_linear_bend_pair(name, atoms, printed_angle, reference_coords):
    """Replace one near-180-degree ORCA angle by two regular components."""
    if reference_coords is None:
        raise ValueError(
            f"Near-linear ORCA angle {name} requires the equilibrium Cartesian geometry."
        )
    reference_coords = np.asarray(reference_coords, float)
    i_atom, j_atom, _ = atoms
    axis = unit(reference_coords[i_atom] - reference_coords[j_atom])
    seeds = np.eye(3)
    seed = seeds[int(np.argmin(np.abs(seeds @ axis)))]
    direction_1 = unit(seed - np.dot(seed, axis) * axis)
    frame_axis = tuple(float(value) for value in axis)
    frame_direction_1 = tuple(float(value) for value in direction_1)
    # Only the norm of the two deviations from 180 degrees is invariant.
    # Put the printed scalar deviation in the first component; the existing
    # validation compares the two-component norm and is frame independent.
    deviation_degrees = 180.0 - float(printed_angle)
    return [
        Internal(
            f"{name}x", "L", atoms, 180.0 + deviation_degrees,
            component=-1, axis_reference=-1, frame_axis=frame_axis,
            frame_direction_1=frame_direction_1,
        ),
        Internal(
            f"{name}y", "L", atoms, 180.0,
            component=-2, axis_reference=-1, frame_axis=frame_axis,
            frame_direction_1=frame_direction_1,
        ),
    ]


def _read_orca_internals(log_text, natoms, reference_coords=None):
    """Read the final ORCA redundant-internal table before the Hessian analysis.

    ORCA atom indices are zero based. Bond definitions ``B`` are translated to
    the internal ``R`` representation used by AnhDis. Near-linear angles are
    expanded into the two orthogonal components required for a nonsingular
    Wilson matrix.
    """
    frequency_start = log_text.find("VIBRATIONAL FREQUENCIES")
    if frequency_start < 0:
        raise ValueError("No ORCA vibrational analysis found.")
    prefix = log_text[:frequency_start]
    headings = [
        match.start()
        for match in re.finditer(r"Redundant Internal Coordinates", prefix)
    ]
    table_pattern = re.compile(
        r"^\s*\d+\.\s+([A-Za-z]+)\(([^)]*)\)\s+(.+?)\s*$",
        re.MULTILINE,
    )
    selected_text = None
    selected_matches = None
    for start in reversed(headings):
        section = prefix[start:]
        matches = list(table_pattern.finditer(section))
        if matches:
            selected_text = section
            selected_matches = matches
            break
    if not selected_matches:
        raise ValueError(
            "No ORCA redundant-internal table was found before the frequency "
            "analysis. Curvilinear paths require an ORCA redundant-internal "
            "optimization output."
        )

    optimized_table = "Optimized Parameters" in selected_text[:1000]
    coords = None if reference_coords is None else np.asarray(reference_coords, float)
    if coords is not None and coords.shape != (natoms, 3):
        raise ValueError("Invalid equilibrium geometry for ORCA internals.")
    result = []
    for ordinal, match in enumerate(selected_matches, 1):
        source_kind, atom_text, numeric_text = match.groups()
        source_kind = source_kind.upper()
        if source_kind not in {"B", "A", "D"}:
            raise ValueError(
                f"Unsupported ORCA internal coordinate {ordinal}: {source_kind}"
            )
        expected = {"B": 2, "A": 3, "D": 4}[source_kind]
        atom_parts = [part.strip() for part in atom_text.split(",")]
        if len(atom_parts) != expected:
            raise ValueError(f"Invalid ORCA internal coordinate {ordinal}: {match.group(0).strip()}")
        atoms = []
        for part in atom_parts:
            atom_match = re.search(r"(-?\d+)\s*$", part)
            if atom_match is None:
                raise ValueError(f"Cannot read an ORCA atom index from {part!r}.")
            atoms.append(int(atom_match.group(1)))
        atoms = tuple(atoms)
        if len(set(atoms)) != len(atoms) or min(atoms) < 0 or max(atoms) >= natoms:
            raise ValueError(f"Atom index out of range in ORCA internal {ordinal}.")
        numeric_values = [
            float(value.replace("D", "E").replace("d", "e"))
            for value in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][-+]?\d+)?", numeric_text)
        ]
        if not numeric_values:
            raise ValueError(f"Missing value for ORCA internal {ordinal}.")
        printed_value = numeric_values[-1] if optimized_table else numeric_values[0]
        name = f"O{ordinal}"
        if source_kind == "A" and coords is not None:
            first = unit(coords[atoms[0]] - coords[atoms[1]])
            second = unit(coords[atoms[2]] - coords[atoms[1]])
            sine = float(np.linalg.norm(np.cross(first, second)))
            cosine = float(np.dot(first, second))
            if sine < 1.0e-5:
                if cosine > 0.0:
                    raise ValueError(
                        f"ORCA angle {name} is near zero degrees and is not a supported bend."
                    )
                result.extend(
                    _orca_linear_bend_pair(name, atoms, printed_value, coords)
                )
                continue
        result.append(Internal(
            name, "R" if source_kind == "B" else source_kind,
            atoms, printed_value,
        ))
    if not result or len({item.name for item in result}) != len(result):
        raise ValueError("Empty or duplicate ORCA internal-coordinate definitions.")
    return result


def read_internals(log_text, natoms, reference_coords=None, software="auto"):
    """Read Gaussian or ORCA internal coordinates without molecule-specific data."""
    requested = str(software).strip().lower()
    if requested == "auto":
        if "VIBRATIONAL FREQUENCIES" in log_text and "ORCA TERMINATED NORMALLY" in log_text:
            requested = "orca"
        elif "Harmonic frequencies (cm" in log_text:
            requested = "gaussian"
        else:
            raise ValueError("Could not detect Gaussian or ORCA output format.")
    if requested == "gaussian":
        return _read_gaussian_internals(log_text, natoms, reference_coords)
    if requested == "orca":
        return _read_orca_internals(log_text, natoms, reference_coords)
    raise ValueError("Optimization/frequency software must be 'auto', 'gaussian', or 'orca'.")


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-12:
        raise ValueError("Coincident atoms or undefined internal coordinate.")
    return vector / length


def rotation_align_vector(source, target):
    """Return the minimum rotation carrying one unit vector onto another."""
    source, target = unit(source), unit(target)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1.0e-14:
        if cosine > 0.0:
            return np.eye(3)
        seeds = np.eye(3)
        axis = seeds[int(np.argmin(np.abs(seeds @ source)))]
        axis = unit(axis - np.dot(axis, source) * source)
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    x, y, z = cross
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def linear_bend_components(points, item):
    """Return Gaussian-compatible two-component linear-bend coordinates.

    The vector norm is ``pi-angle(i,j,k)``. A real fourth atom supplies the
    transverse frame intrinsically. For Gaussian automatic-axis L coordinates,
    the current molecular axis is minimally rotated onto the equilibrium axis
    saved in ``item`` before resolving the two transverse components.
    """
    points = np.asarray(points, float)
    point_i, point_j, point_k = points[:3]
    axis = unit(point_i - point_j)
    other = unit(point_k - point_j)
    if item.axis_reference is None:
        if points.shape != (4, 3):
            raise ValueError(f"Linear bend {item.name} requires a real fourth atom.")
        reference = point_i - points[3]
        direction_1 = unit(reference - np.dot(reference, axis) * axis)
        direction_2 = unit(np.cross(direction_1, axis))
    else:
        if item.frame_axis is None or item.frame_direction_1 is None:
            raise ValueError(f"Linear bend {item.name} lacks its equilibrium frame.")
        frame_axis = unit(np.asarray(item.frame_axis, float))
        direction_1 = unit(np.asarray(item.frame_direction_1, float))
        rotation = rotation_align_vector(axis, frame_axis)
        other = rotation @ other
        axis = frame_axis
        direction_2 = unit(np.cross(direction_1, axis))
    x = float(np.dot(other, direction_1))
    y = float(np.dot(other, direction_2))
    transverse = float(np.hypot(x, y))
    bend = float(np.arctan2(transverse, -np.dot(other, axis)))
    if transverse < 1.0e-12:
        return np.array([x, y])
    return (bend / transverse) * np.array([x, y])


def internal_values(coords, definitions):
    """Evaluate distances, angles, linear bends and signed dihedrals."""
    values = []
    for item in definitions:
        p = coords[list(item.atoms)]
        if item.kind == "R":
            value = np.linalg.norm(p[1] - p[0])
            if value < 1e-12:
                raise ValueError("Zero bond length.")
        elif item.kind == "A":
            v, w = unit(p[0]-p[1]), unit(p[2]-p[1])
            cross = np.linalg.norm(np.cross(v, w))
            if cross < 1e-8:
                raise ValueError("Linear angle: this step requires special linear-bend coordinates.")
            value = np.arctan2(cross, np.dot(v, w))
        elif item.kind == "D":
            axis = unit(p[2]-p[1])
            v, w = p[0]-p[1], p[3]-p[2]
            v = v - np.dot(v, axis)*axis
            w = w - np.dot(w, axis)*axis
            v, w = unit(v), unit(w)
            value = np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w))
        elif item.kind == "L":
            if item.component not in {-1, -2}:
                raise ValueError(f"Invalid linear-bend component in {item.name}")
            value = np.pi + linear_bend_components(p, item)[-item.component - 1]
        else:
            raise ValueError(f"Unsupported primitive {item.kind}")
        values.append(value)
    return np.array(values)


def internal_difference(a, b, definitions):
    """Use a local wrapped difference across the +/-pi dihedral boundary."""
    delta = np.asarray(a) - np.asarray(b)
    delta = delta.copy()
    for i, item in enumerate(definitions):
        if item.kind == "D":
            delta[i] = np.arctan2(np.sin(delta[i]), np.cos(delta[i]))
    return delta


def continuous_internal_difference(a, b, definitions, branch_reference):
    """Lift each periodic dihedral to the branch nearest a known path point.

    The reference is a vector of continuous differences from b, not a vector
    of absolute angles. Local Wilson-B differences must still use the wrapped
    internal_difference; only finite path displacements need this lift.
    """
    delta = internal_difference(a, b, definitions)
    anchor = np.asarray(branch_reference, float)
    if anchor.shape != delta.shape or not np.isfinite(anchor).all():
        raise ValueError("Invalid continuous angular branch reference.")
    for index, item in enumerate(definitions):
        if item.kind == "D":
            delta[index] += 2 * np.pi * np.rint(
                (anchor[index] - delta[index]) / (2 * np.pi)
            )
    return delta


def wilson_b(coords, definitions, step=1e-5):
    """Central Cartesian differences. Columns x1,y1,z1,x2,... .

    Distance rows: angstrom/angstrom. Angular rows: radian/angstrom.
    Halving step independently in diagnose_and_save checks convergence.
    """
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Finite-difference step must be positive and finite.")
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError("Expected finite Cartesian coordinates of shape (N,3).")
    b = np.empty((len(definitions), coords.size))
    for j in range(coords.size):
        plus, minus = coords.copy(), coords.copy()
        plus.flat[j] += step
        minus.flat[j] -= step
        b[:, j] = internal_difference(internal_values(plus, definitions),
                                      internal_values(minus, definitions), definitions)/(2*step)
    return b


def rigid_directions(coords, rank_rtol=1.0e-10):
    """Return an orthonormal basis for translations and finite rotations.

    The returned rank is six for a nonlinear reference and five for a linear
    reference. These are unweighted diagnostic directions, not an Eckart
    projection.
    """
    centered = coords - np.mean(coords, axis=0)
    columns = []
    for axis in np.eye(3):
        columns.append(np.tile(axis, (len(coords), 1)).ravel())
        columns.append(np.cross(axis, centered).ravel())
    rigid = np.column_stack(columns)
    u, singular, _ = np.linalg.svd(rigid, full_matrices=False)
    rank = int(np.sum(singular > rank_rtol * singular[0]))
    if rank not in {5, 6}:
        raise ValueError(f"Expected five or six rigid directions, found {rank}.")
    return u[:, :rank]


def vibrational_dof(coords):
    """Return 3N-5 for a linear geometry and 3N-6 otherwise."""
    coords = np.asarray(coords, float)
    return int(coords.size - rigid_directions(coords).shape[1])


def read_masses(log_text, natoms):
    """Read Gaussian masses (amu); prefer the higher-precision AtmWgt line."""
    weight_lines = re.findall(r"^\s*AtmWgt=([^\n]+)",log_text,re.MULTILINE)
    for line in reversed(weight_lines):
        values = [float(x.replace("D","E")) for x in line.split()]
        if len(values) == natoms and all(np.isfinite(x) and x>0 for x in values):
            return np.array(values)
    pattern = re.compile(r"Atom\s+(\d+) has atomic number\s+\d+ and mass\s+([-+\d.EeDd]+)")
    found = {}
    for atom, mass in pattern.findall(log_text):
        found[int(atom)] = float(mass.replace("D", "E"))
    if sorted(found) != list(range(1, natoms+1)) or any(not np.isfinite(m) or m <= 0 for m in found.values()):
        raise ValueError("Could not read one positive Gaussian mass for every atom.")
    return np.array([found[i] for i in range(1, natoms+1)])


def load_step1_system(coords, modes, log_text, directory):
    """Use the SAVED U basis, not a newly diagonalized/reordered SVD basis.

    Validate the input log, coordinate definitions, matrix dimensions,
    orthonormality, B derivatives and the previously saved internal tangents.
    """
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Step-1 results missing: {root}; run --check-internals-only first.")
    meta = json.loads((root/"diagnostics.json").read_text())
    digest = hashlib.sha256(log_text.encode()).hexdigest()
    if meta.get("source_text_sha256") != digest or meta.get("stage") != "local_B_diagnostics_only":
        raise ValueError("Step-1 results do not belong to this optimization/frequency output.")
    definitions = read_internals(log_text, len(coords), reference_coords=coords)
    if [x.label for x in definitions] != [x["label"] for x in meta["internals"]]:
        raise ValueError("Step-1 internal definitions disagree with log.")
    n = int(meta.get("n_modes", vibrational_dof(coords)))
    basis = np.loadtxt(root/"independent_basis_U.dat")
    b0 = np.loadtxt(root/"B_matrix.dat")
    if basis.shape != (len(definitions),n) or b0.shape != (len(definitions),coords.size):
        raise ValueError("Invalid saved step-1 matrix dimensions.")
    np.testing.assert_allclose(basis.T @ basis,np.eye(n),atol=1e-10,rtol=0)
    b_check = wilson_b(coords,definitions,meta["B_step_angstrom"])
    np.testing.assert_allclose(b0,b_check,atol=1e-9,rtol=0)
    np.testing.assert_allclose(np.loadtxt(root/"internal_modes.dat"),
        b0 @ modes.reshape(n,-1).T,atol=1e-9,rtol=0)
    length = meta["reference_length_angstrom"]
    weights = np.array([1/length if x.kind=="R" else 1.0 for x in definitions])
    jacobian = basis.T @ (weights[:,None]*b0)
    np.testing.assert_allclose(np.loadtxt(root/"B_independent.dat"),jacobian,atol=1e-9,rtol=0)
    return definitions,internal_values(coords,definitions),weights,basis,jacobian,meta


def internal_system(coords, definitions, reference_length=1.0, step=1e-5, rank_rtol=1e-7):
    """Return the fixed reference DLC basis and its local Jacobian."""
    s0 = internal_values(coords, definitions)
    b0 = wilson_b(coords, definitions, step/2)
    weights = np.array([1/reference_length if x.kind == "R" else 1.0 for x in definitions])
    scaled = weights[:, None]*b0
    u, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.sum(singular > rank_rtol*singular[0]))
    expected = vibrational_dof(coords)
    if rank != expected:
        raise ValueError(f"B rank {rank}, expected {expected}.")
    basis = u[:, :rank]
    return s0, weights, basis, basis.T @ scaled


def mass_metric_step(jacobian, residual, masses):
    """Minimum mass-norm Cartesian step satisfying J dx = residual.

    dx = M^-1 J.T (J M^-1 J.T)^-1 residual. This is a LOCAL
    back-transformation step and is orthogonal (in the mass metric) to the
    rigid nullspace of J. It is not a kinetic-energy Hamiltonian.
    """
    inv_mass_xyz = np.repeat(1/np.asarray(masses, float), 3)
    g = (jacobian*inv_mass_xyz[None, :]) @ jacobian.T
    try:
        dual = np.linalg.solve(g, residual)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Singular internal-coordinate metric during back-transformation.") from exc
    return inv_mass_xyz*(jacobian.T @ dual)


def mass_align(coords, reference, masses):
    """Remove arbitrary translation/rotation by mass-weighted Kabsch alignment."""
    w = np.asarray(masses, float)[:, None]
    c, r = np.sum(w*coords,axis=0)/np.sum(w), np.sum(w*reference,axis=0)/np.sum(w)
    p, q = coords-c, reference-r
    if rigid_directions(reference).shape[1] == 5:
        # A linear reference does not define a rotation about its own axis, so
        # a full Kabsch fit may flip or arbitrarily rotate the two degenerate
        # bend components. Align the molecular axis by the minimum rotation
        # and preserve the transverse orientation of the current geometry.
        distances = np.linalg.norm(
            reference[:, None, :] - reference[None, :, :], axis=2,
        )
        atom_a, atom_b = np.unravel_index(np.argmax(distances), distances.shape)
        current_axis = unit(coords[atom_b] - coords[atom_a])
        reference_axis = unit(reference[atom_b] - reference[atom_a])
        rotation = rotation_align_vector(current_axis, reference_axis)
        return p @ rotation.T + r
    left, _, right_t = np.linalg.svd((p*w).T @ q)
    correction = np.eye(3)
    correction[-1,-1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    return p @ rotation + r


def backtransform(target_u, reference, definitions, s0, weights, basis, masses,
                  initial=None, step=1e-5, tolerance=1e-11,
                  max_iterations=30, cart_step_limit=0.05, branch_reference=None):
    """Iteratively solve U.T W [s(R)-s(R0)] = target_u.

    U, W and s0 remain fixed at the reference. B(R) is recalculated at every
    iteration. A monotonic line search and a Cartesian step limit protect the
    local Newton iteration. Returned geometry is aligned to the reference.
    """
    target_u = np.asarray(target_u,float)
    current = np.array(reference if initial is None else initial, dtype=float, copy=True)
    if (not np.isfinite(current).all() or not np.isfinite(target_u).all()
            or target_u.shape != (basis.shape[1],)):
        raise ValueError("Invalid initial geometry or target internal coordinate.")
    if tolerance<=0 or max_iterations<1 or cart_step_limit<=0:
        raise ValueError("Invalid back-transformation convergence parameters.")
    def difference(values):
        if branch_reference is None:
            return internal_difference(values, s0, definitions)
        return continuous_internal_difference(values, s0, definitions, branch_reference)

    history = []
    for iteration in range(max_iterations+1):
        current_s = internal_values(current, definitions)
        current_u = basis.T @ (weights*difference(current_s))
        residual = target_u-current_u
        error = float(np.linalg.norm(residual,np.inf))
        history.append(error)
        if error < tolerance:
            return mass_align(current,reference,masses), iteration, error, history
        if iteration == max_iterations:
            break
        jacobian = basis.T @ (weights[:,None]*wilson_b(current,definitions,step/2))
        delta = mass_metric_step(jacobian,residual,masses).reshape(reference.shape)
        maximum = np.max(np.linalg.norm(delta,axis=1))
        if maximum > cart_step_limit:
            delta *= cart_step_limit/maximum
        accepted = False
        for factor in (1.0,.5,.25,.125,.0625,.03125):
            trial = current+factor*delta
            trial_s = internal_values(trial,definitions)
            trial_difference = difference(trial_s)
            if branch_reference is not None:
                angular = np.asarray([item.kind != "R" for item in definitions])
                if np.any(np.abs((trial_difference - branch_reference)[angular]) >= np.pi / 2):
                    continue
            trial_u = basis.T @ (weights*trial_difference)
            if np.linalg.norm(target_u-trial_u,np.inf) < error:
                current, accepted = mass_align(trial,reference,masses), True
                break
        if not accepted:
            raise RuntimeError(f"Back-transformation line search failed at error {error:.3e}.")
    raise RuntimeError(f"Back-transformation did not converge: final error {history[-1]:.3e}.")


def continue_backtransform(target_u, reference, definitions, s0, weights, basis,
                           masses, initial, initial_difference, rank_rtol=1e-10,
                           max_subdivisions=20, **options):
    """Continue a lifted internal path, bisecting unresolved local steps.

    There is no bound on accumulated torsional displacement. The pi/2 bound
    applies only to a local predictor/corrector step, making the choice among
    branches separated by 2*pi unambiguous. Subdivision exhaustion is reported
    as an error, never accepted as a geometry. Returns geometry, lifted
    difference, Newton iterations, residual history, accepted step count.
    """
    target_u = np.asarray(target_u, float)
    angular = np.asarray([item.kind != "R" for item in definitions])
    fd = float(options.get("step", 1e-5))
    tolerance = float(options.get("tolerance", 1e-11))
    anchor = np.asarray(initial_difference, float)
    actual = continuous_internal_difference(
        internal_values(initial, definitions), s0, definitions, anchor,
    )
    if not np.allclose(actual, anchor, atol=10 * tolerance, rtol=0):
        raise ValueError("Initial geometry and continuous branch state disagree.")

    def advance(goal, geometry, difference, depth):
        start = basis.T @ (weights * difference)
        try:
            b = wilson_b(geometry, definitions, fd / 2)
            j = basis.T @ (weights[:, None] * b)
            singular = np.linalg.svd(j, compute_uv=False)
            if singular[-1] <= rank_rtol * singular[0]:
                raise ValueError("Internal Jacobian lost rank during continuation.")
            predicted = b @ mass_metric_step(j, goal - start, masses)
            if np.any(np.abs(predicted[angular]) >= np.pi / 2):
                raise ValueError("Angular predictor requires a smaller local step.")
            result, iterations, _, history = backtransform(
                goal, reference, definitions, s0, weights, basis, masses,
                initial=geometry, branch_reference=difference, **options,
            )
            lifted = continuous_internal_difference(
                internal_values(result, definitions), s0, definitions, difference,
            )
            if np.any(np.abs((lifted - difference)[angular]) >= np.pi / 2):
                raise ValueError("Angular corrector requires a smaller local step.")
            final_j = basis.T @ (
                weights[:, None] * wilson_b(result, definitions, fd / 2)
            )
            final_singular = np.linalg.svd(final_j, compute_uv=False)
            if final_singular[-1] <= rank_rtol * final_singular[0]:
                raise ValueError("Internal Jacobian lost rank during continuation.")
            return result, lifted, iterations, history, 1
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            if depth >= max_subdivisions or np.max(np.abs(goal - start)) <= tolerance:
                raise RuntimeError(
                    f"Continuous internal path could not be resolved after {depth} "
                    f"subdivisions; target increment={np.max(np.abs(goal-start)):.3e}: {exc}"
                ) from exc
            midpoint = (start + goal) / 2
            middle, mid_diff, count1, hist1, steps1 = advance(
                midpoint, geometry, difference, depth + 1,
            )
            result, final_diff, count2, hist2, steps2 = advance(
                goal, middle, mid_diff, depth + 1,
            )
            return result, final_diff, count1 + count2, hist1 + hist2, steps1 + steps2

    return advance(target_u, np.asarray(initial, float), actual, 0)


def remove_rigid_velocity(coords, velocity, masses):
    """Remove instantaneous translation/rotation in the mass metric.

    Fixed-reference Kabsch alignment alone does not in general impose zero
    instantaneous angular momentum at a finite deformation. For an isolated
    internal path we use the horizontal (zero linear/angular momentum) velocity.
    """
    masses = np.asarray(masses,float)
    r = coords-np.average(coords,axis=0,weights=masses)
    v = velocity-np.average(velocity,axis=0,weights=masses)
    inertia = np.eye(3)*np.sum(masses*np.sum(r*r,axis=1))-(r*masses[:,None]).T @ r
    angular_momentum = np.sum(masses[:,None]*np.cross(r,v),axis=0)
    # A linear molecule has a singular inertia tensor about its molecular
    # axis. The Moore-Penrose solution removes the two physical rotations and
    # leaves the null rotation (which moves no atom) exactly harmless.
    omega = np.linalg.pinv(inertia, rcond=1.0e-12) @ angular_momentum
    return v-np.cross(omega,r)


def path_mass_and_save(coords, modes, log_text, internal_input, finite_input,
                       output_dir, derivative_step=1e-4, source_log=""):
    """Diagnostic of a constrained 1D internal path: mu=l.T G^-1 l.

    G=J M^-1 J.T is evaluated at each saved geometry. We independently verify
    this mass using central derivatives of nearby reconstructions, projecting
    out instantaneous rigid motion. NOT a full vibrational quantum Hamiltonian.
    """
    out, finite = Path(output_dir), Path(finite_input)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Use a new/empty path-mass directory: {out}")
    if not np.isfinite(derivative_step) or not 0 < derivative_step <= .01:
        raise ValueError("Path derivative step must be finite and in (0, 0.01] Angstrom.")
    coords,modes = np.asarray(coords,float),np.asarray(modes,float)
    defs,s0,w,basis,j0,meta = load_step1_system(coords,modes,log_text,internal_input)
    finite_meta = json.loads((finite/"diagnostics.json").read_text())
    if (finite_meta.get("stage") != "finite_geometric_comparison_only"
            or finite_meta.get("step1_source_hash") != meta["source_text_sha256"]):
        raise ValueError("Step-3 geometries must belong to this log and the finite-geometry stage.")
    masses = read_masses(log_text,len(coords))
    mass_xyz = np.repeat(masses,3)
    step = 2*meta["B_step_angstrom"]
    options = dict(step=step,tolerance=1e-12,max_iterations=50,cart_step_limit=.05)
    rows, summaries = [], []
    max_tangent_error, max_fd_change, max_B_change = 0.,0.,0.
    max_closure = 0.
    for mode in finite_meta["selected_modes"]:
        if not isinstance(mode,int) or not 1 <= mode <= len(modes):
            raise ValueError("Invalid saved mode number.")
        l = j0 @ modes[mode-1].ravel()
        t0 = mass_metric_step(j0,l,masses)
        mu0 = float(np.sum(mass_xyz*t0*t0))
        if mu0 <= 0:
            raise ValueError("Zero internal tangent cannot define a path mass.")
        mu_values = []
        for q in finite_meta["Q_angstrom"]:
            path = finite/"curvilinear"/f"vib{mode}"/f"{q:.2f}"/f"vib{mode}_{q:.2f}.xyz"
            geometry = np.loadtxt(path,skiprows=2,usecols=(1,2,3))
            if geometry.shape != coords.shape or not np.isfinite(geometry).all():
                raise ValueError(f"Invalid saved geometry: {path}")
            # Recover the branch by continuation, not from a wrapped XYZ angle.
            _, branch_reference, *_ = continue_backtransform(
                q*l, coords, defs, s0, w, basis, masses, initial=coords,
                initial_difference=np.zeros_like(s0), rank_rtol=meta["rank_rtol"], **options,
            )
            difference = continuous_internal_difference(
                internal_values(geometry,defs), s0, defs, branch_reference,
            )
            closure = float(np.max(np.abs(basis.T @ (w*difference)-q*l)))
            if closure > 1e-9:
                raise ValueError(f"Saved geometry does not satisfy the step-1 internal target: {path}")
            max_closure = max(max_closure,closure)
            j = basis.T @ (w[:,None]*wilson_b(geometry,defs,step/2))
            singular = np.linalg.svd(j,compute_uv=False)
            if singular[-1]/singular[0] <= meta["rank_rtol"]:
                raise ValueError("Rank-deficient path Jacobian.")
            v_metric = mass_metric_step(j,l,masses).reshape(coords.shape)
            mu = float(np.sum(masses[:,None]*v_metric**2))
            j_half = basis.T @ (w[:,None]*wilson_b(geometry,defs,step/4))
            v_half = mass_metric_step(j_half,l,masses)
            mu_half_B = float(np.sum(mass_xyz*v_half**2))
            b_change = abs(mu_half_B/mu-1)
            fd_masses, raw_masses, fd_errors = [], [], []
            for h in (derivative_step,derivative_step/2):
                plus,*_ = continue_backtransform(
                    (q+h)*l,coords,defs,s0,w,basis,masses,initial=geometry,
                    initial_difference=difference,rank_rtol=meta["rank_rtol"],**options)
                minus,*_ = continue_backtransform(
                    (q-h)*l,coords,defs,s0,w,basis,masses,initial=geometry,
                    initial_difference=difference,rank_rtol=meta["rank_rtol"],**options)
                v_raw = (plus-minus)/(2*h)
                v = remove_rigid_velocity(geometry,v_raw,masses)
                raw_masses.append(float(np.sum(masses[:,None]*v_raw*v_raw)))
                fd_masses.append(float(np.sum(masses[:,None]*v*v)))
                fd_errors.append(float(np.sqrt(np.sum(masses[:,None]*(v-v_metric)**2)/mu)))
            fd_change = abs(fd_masses[1]/fd_masses[0]-1)
            max_tangent_error = max(max_tangent_error,fd_errors[1])
            max_fd_change, max_B_change = max(max_fd_change,fd_change),max(max_B_change,b_change)
            if fd_errors[1] > 2e-6 or fd_change > 2e-6 or b_change > 1e-7:
                raise ValueError(f"Path-mass numerical convergence failed: mode {mode}, Q={q}; "
                                 f"tangent_error={fd_errors[1]:.6e}, FD_mass_change={fd_change:.6e}, "
                                 f"B_mass_change={b_change:.6e}. Reduce PATH_MASS_FD_STEP_ANG "
                                 "to test derivative convergence; do not relax acceptance thresholds.")
            if abs(q) < 1e-14 and abs(mu/mu0-1) > 1e-8:
                raise ValueError("Equilibrium mass disagrees with the step-2 projected tangent.")
            mu_values.append(mu)
            rows.append((mode,q,mu,mu0,100*(mu/mu0-1),fd_masses[0],fd_masses[1],
                         raw_masses[1],fd_errors[1],fd_change,b_change))
        summaries.append(dict(mode=mode,mu0_amu=mu0,min_mu_amu=min(mu_values),max_mu_amu=max(mu_values),
                              max_abs_relative_change_percent=max(abs(x/mu0-1)*100 for x in mu_values)))
    result = dict(stage="internal_path_mass_diagnostic_only",source_log=str(source_log),
        step1_source_hash=meta["source_text_sha256"],internal_input=str(internal_input),finite_input=str(finite_input),
        derivative_step_angstrom=derivative_step,reference_B_step_angstrom=step/2,
        max_saved_target_residual=max_closure,max_relative_fd_tangent_error=max_tangent_error,
        max_relative_fd_mass_change=max_fd_change,max_relative_B_mass_change=max_B_change,
        modes=summaries,hamiltonian_modified=False,energies_calculated=False,
        formula="mu(X)=l^T [J(R(X)) M^-1 J(R(X))^T]^-1 l",
        scope="constrained 1D internal curve; instantaneous rigid motion removed; no full quantum operator or mode-coupling validation")
    out.mkdir(parents=True,exist_ok=True)
    np.savetxt(out/"path_mass.dat",rows,fmt="%.12e",header=
        "mode Q(Ang) mu_internal(amu) mu0_projected(amu) change_percent "
        "mu_FD_h(amu) mu_FD_half_h(amu) mu_fixed_alignment_FD_half_h(amu) "
        "relative_tangent_error_half_h relative_FD_mass_change relative_B_mass_change")
    summary_rows = [(x['mode'],x['mu0_amu'],x['min_mu_amu'],x['max_mu_amu'],
                     100*(x['min_mu_amu']/x['mu0_amu']-1),
                     100*(x['max_mu_amu']/x['mu0_amu']-1),
                     x['max_abs_relative_change_percent']) for x in summaries]
    np.savetxt(out/'mass_summary.dat',summary_rows,fmt=['%d']+['%.12e']*6,header=
        'mode mu0_projected(amu) min_mu(amu) max_mu(amu) min_change_percent max_change_percent max_abs_change_percent')
    (out/"diagnostics.json").write_text(json.dumps(result,indent=2)+"\n")
    report = ("AnhDis: masa de la trayectoria INTERNA 1D (paso 4)\n"
              f"Modos comprobados: {len(summaries)}; puntos de masa comprobados: {len(rows)}\n")
    for item in summaries:
        report += (f"Modo {item['mode']}: mu(0)={item['mu0_amu']:.10f} amu; "
                   f"min={item['min_mu_amu']:.10f}; max={item['max_mu_amu']:.10f} amu\n"
                   f"Maxima desviacion respecto a mu(0): {item['max_abs_relative_change_percent']:.6f} %\n")
    report += (f"Error relativo maximo de tangente FD: {max_tangent_error:.3e}\n"
               f"Cambio relativo maximo al reducir h: {max_fd_change:.3e}\n"
               f"Cambio relativo maximo al reducir paso de B: {max_B_change:.3e}\n"
               "Comprobaciones NUMERICAS: PASS.\n"
               "Se elimina traslacion/rotacion instantanea; alineamiento fijo y movimiento interno no son identicos a Q finito.\n"
               "No se ha decidido una aproximacion de masa constante ni construido un operador cuantico.\n"
               "No se han modificado el Hamiltoniano, a6 ni a7.\n")
    (out/"report.txt").write_text(report)
    print(report)
    return result


def finite_geometries_and_save(coords, modes, atomic_numbers, log_text, output_dir,
                               internal_input, selected_modes, amplitudes,
                               continuation_step=.025, tolerance=1e-11,
                               max_iterations=30, cart_step_limit=.05, source_log=""):
    """Step 3: geometric comparison only, in the fixed saved step-1 chart.

    Continue outward from Q=0 independently for each sign. All substeps check
    closure, Jacobian rank and angular branch safety. This is a local chart,
    not a periodic rotor, energy minimization or a kinetic-energy model.
    """
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Use a new/empty finite-geometry directory: {out}")
    coords, modes = np.asarray(coords,float), np.asarray(modes,float)
    qs = np.asarray(amplitudes,float)
    selected = list(selected_modes)
    if (qs.ndim != 1 or not qs.size or not np.isfinite(qs).all()
            or len(set(f"{q:.2f}" for q in qs)) != len(qs)):
        raise ValueError("Q values must be finite and unique at two-decimal filename precision.")
    if (not selected or len(set(selected)) != len(selected)
            or any(not isinstance(i,(int,np.integer)) or not 1 <= i <= len(modes) for i in selected)):
        raise ValueError("Select distinct, valid integer mode numbers (starting at 1).")
    if not np.isfinite(continuation_step) or continuation_step <= 0:
        raise ValueError("Positive finite continuation step required.")
    element_symbols = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
        "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm "
        "Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu "
        "Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
    if len(atomic_numbers) != len(coords) or any(not 1 <= z < len(element_symbols) for z in atomic_numbers):
        raise ValueError("Physical atomic numbers 1...118 are required for XYZ output.")
    symbols = [element_symbols[z] for z in atomic_numbers]
    masses = read_masses(log_text,len(coords))
    defs,s0,w,u,j0,meta = load_step1_system(coords,modes,log_text,internal_input)
    step = 2*meta["B_step_angstrom"]
    options = dict(step=step,tolerance=tolerance,max_iterations=max_iterations,cart_step_limit=cart_step_limit)
    angular = np.array([x.kind != "R" for x in defs])
    branches = ("cartesian_original","cartesian_projected","curvilinear")
    geometries, internal_rows, summary_rows, continuation_rows = [], [], [], []
    max_closure, min_rank_ratio = 0., float("inf")
    for mode in selected:
        d = modes[mode-1]
        internal_tangent = j0 @ d.ravel()
        t = mass_metric_step(j0,internal_tangent,masses).reshape(coords.shape)
        reconstructed = {0.: coords.copy()}
        reconstructed_differences = {0.: np.zeros_like(s0)}
        for sign in (-1,1):
            previous_q, previous = 0., coords.copy()
            previous_difference = np.zeros_like(s0)
            for q in sorted((float(q) for q in qs if sign*q > 0),key=abs):
                nsteps = max(1,int(np.ceil(abs(q-previous_q)/continuation_step)))
                for subq in np.linspace(previous_q,q,nsteps+1)[1:]:
                    current,difference,iterations,_,_ = continue_backtransform(
                        internal_tangent*subq,coords,defs,s0,w,u,masses,
                        initial=previous,initial_difference=previous_difference,
                        rank_rtol=meta["rank_rtol"],**options)
                    closure = float(np.max(np.abs(u.T @ (w*difference)-internal_tangent*subq)))
                    jc = u.T @ (w[:,None]*wilson_b(current,defs,step/2))
                    singular = np.linalg.svd(jc,compute_uv=False)
                    ratio = float(singular[-1]/singular[0])
                    if closure > tolerance or ratio <= meta["rank_rtol"]:
                        raise ValueError(f"Finite reconstruction failed closure/rank test at mode {mode}, Q={subq}.")
                    max_closure, min_rank_ratio = max(max_closure,closure), min(min_rank_ratio,ratio)
                    continuation_rows.append((mode,subq,iterations,closure,ratio))
                    previous = current
                    previous_difference = difference
                reconstructed[q] = previous.copy()
                reconstructed_differences[q] = previous_difference.copy()
                previous_q = q
        for q in sorted(float(q) for q in qs):
            for branch_id,geometry in enumerate((coords+d*q,coords+t*q,reconstructed[q]),start=1):
                vals = internal_values(geometry,defs)
                # Report angular values continuously about their equilibrium branch.
                values_display = s0 + (
                    reconstructed_differences[q] if branch_id == 3
                    else internal_difference(vals,s0,defs)
                )
                values_display[angular] = np.rad2deg(values_display[angular])
                internal_rows.append([mode,q,branch_id,*values_display])
                geometries.append((mode,q,branches[branch_id-1],geometry))
            original,projected,curvi = coords+d*q,coords+t*q,reconstructed[q]
            aligned_original = mass_align(original,coords,masses)
            delta = curvi-projected
            # Unweighted 3N Euclidean norm is a chord, NOT an arc length.
            summary_rows.append((mode,q,np.linalg.norm(original-coords),np.linalg.norm(projected-coords),
                np.linalg.norm(curvi-coords),np.sqrt(np.sum(masses[:,None]*delta**2)/sum(masses)),
                np.sqrt(np.sum(masses[:,None]*(aligned_original-projected)**2)/sum(masses))))
    # Do not leave apparently complete outputs if any mode/point failed above.
    out.mkdir(parents=True,exist_ok=True)
    for mode,q,branch,geometry in geometries:
        folder = out/branch/f"vib{mode}"/f"{q:.2f}"
        folder.mkdir(parents=True,exist_ok=True)
        xyz = f"{len(coords)}\nAngstrom; {branch}; mode={mode}; Q={q:.12g} Angstrom parameter\n"
        xyz += "".join(f"{sym} {r[0]:.12f} {r[1]:.12f} {r[2]:.12f}\n" for sym,r in zip(symbols,geometry))
        (folder/f"vib{mode}_{q:.2f}.xyz").write_text(xyz)
    labels = " ".join(f"{x.label}({'Ang' if x.kind=='R' else 'deg'})" for x in defs)
    np.savetxt(out/"internal_values.dat",internal_rows,fmt="%.12e",header=
        "branch: 1=cartesian_original 2=cartesian_projected 3=curvilinear\nmode Q(Ang) branch "+labels)
    np.savetxt(out/"geometry_comparison.dat",summary_rows,fmt="%.12e",header=
        "mode Q(Ang) chord_original(Ang) chord_projected(Ang) chord_curvi(Ang) "
        "mass_RMSD_curvi_vs_projected(Ang) mass_RMSD_aligned_original_vs_projected(Ang)")
    np.savetxt(out/"continuation.dat",np.asarray(continuation_rows).reshape(-1,5),fmt="%.12e",header=
        "mode Q_substep(Ang) iterations independent_residual sigma_min_over_max_J")
    result = dict(stage="finite_geometric_comparison_only",source_log=str(source_log),
        step1_directory=str(internal_input),step1_source_hash=meta["source_text_sha256"],
        selected_modes=selected,Q_angstrom=sorted(qs.tolist()),continuation_step_angstrom=continuation_step,
        max_target_residual=max_closure,min_jacobian_singular_ratio=min_rank_ratio if continuation_rows else None,
        tolerance=tolerance,max_iterations=max_iterations,cart_step_limit_angstrom=cart_step_limit,
        xyz_count=len(geometries),masses_amu=masses.tolist(),energies_calculated=False,hamiltonian_modified=False,
        Q_definition="u(R)=U0.T W [s(R)-s0]=Q J0 d; Q is not Cartesian chord or arc length",
        original_vectors_modified=False,branches=list(branches))
    (out/"diagnostics.json").write_text(json.dumps(result,indent=2)+"\n")
    report = ("AnhDis: comparacion GEOMETRICA a amplitud finita (paso 3)\n"
        f"Modos: {selected}; Q (A): {sorted(qs.tolist())}\n"
        f"Geometrias XYZ: {len(geometries)}; tres trayectorias por punto.\n"
        f"Maximo residuo interno en subpasos: {max_closure:.3e}\n"
        "Reconstruccion geometrica y rango local: PASS.\n"
        "Las distancias primitivas NO se fijan artificialmente.\n"
        "Q es el parametro de la trayectoria, no su longitud cartesiana ni de arco.\n"
        "PASS no valida potenciales, masas efectivas finitas ni el Hamiltoniano.\n"
        "No se han calculado energias. Este diagnostico no ejecuta a2--a7.\n")
    (out/"report.txt").write_text(report)
    print(report)
    return result


def diagnose_backtransform_and_save(coords, modes, log_text, output_dir,
                                    amplitude=1e-3, tolerance=1e-11,
                                    max_iterations=30, cart_step_limit=.05,
                                    source_log="", internal_input=None):
    """Local +/-amplitude reconstruction test for every Gaussian mode."""
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Use a new/empty back-transform directory: {out}")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("Positive finite test amplitude required.")
    coords, modes = np.asarray(coords,float), np.asarray(modes,float)
    masses = read_masses(log_text,len(coords))
    if internal_input is None:
        raise ValueError("An existing step-1 result directory is required.")
    definitions,s0,weights,basis,jacobian0,step1_meta = load_step1_system(coords,modes,log_text,internal_input)
    # Use the saved step-1 numerical conventions, not a potentially changed config.
    step = 2*step1_meta["B_step_angstrom"]
    flat_modes = modes.reshape(len(modes),-1)
    independent_modes = jacobian0 @ flat_modes.T
    projected = np.column_stack([mass_metric_step(jacobian0,independent_modes[:,i],masses)
                                 for i in range(len(modes))]).T
    mass_xyz = np.repeat(masses,3)
    rows, max_eq, max_target, max_tangent = [], 0.0, 0.0, 0.0
    equilibrium, eq_iter, eq_error, _ = backtransform(np.zeros(jacobian0.shape[0]),coords,definitions,s0,
        weights,basis,masses,step=step,tolerance=tolerance,max_iterations=max_iterations,cart_step_limit=cart_step_limit)
    max_eq = float(np.max(np.abs(equilibrium-coords)))
    for i in range(len(modes)):
        target = independent_modes[:,i]*amplitude
        guess_plus = coords+projected[i].reshape(coords.shape)*amplitude
        guess_minus = coords-projected[i].reshape(coords.shape)*amplitude
        plus, ip, ep, _ = backtransform(target,coords,definitions,s0,weights,basis,masses,guess_plus,
            step,tolerance,max_iterations,cart_step_limit)
        minus, im, em, _ = backtransform(-target,coords,definitions,s0,weights,basis,masses,guess_minus,
            step,tolerance,max_iterations,cart_step_limit)
        central = (plus-minus)/(2*amplitude)
        tangent_error = np.sqrt(np.sum(mass_xyz*(central.ravel()-projected[i])**2)/
                                np.sum(mass_xyz*projected[i]**2))
        original_mu = np.sum(mass_xyz*flat_modes[i]**2)
        projected_mu = np.sum(mass_xyz*projected[i]**2)
        cosine = np.sum(mass_xyz*flat_modes[i]*projected[i])/np.sqrt(original_mu*projected_mu)
        target_error = max(ep,em)
        max_target, max_tangent = max(max_target,target_error),max(max_tangent,tangent_error)
        rows.append((i+1,original_mu,projected_mu,cosine,tangent_error,ip,im,ep,em))
    if max_eq > 1e-12 or max_target > tolerance or max_tangent > 2e-5:
        raise ValueError("Local back-transformation validation failed.")
    out.mkdir(parents=True,exist_ok=True)
    np.savetxt(out/"projected_tangents.dat",projected,fmt="%.12e",
               header="Rows vib1...; columns x1 y1 z1 ...; mass-metric inverse of the independent internal tangent")
    np.savetxt(out/"mode_reconstruction.dat",np.array(rows),fmt=["%d"]+["%.12e"]*4+["%d","%d","%.12e","%.12e"],
               header="mode original_mu(amu) projected_mu(amu) mass_cosine relative_mass_tangent_error iter_plus iter_minus residual_plus residual_minus")
    metadata={"stage":"local_internal_to_cartesian_validation_only","source_log":str(source_log),
              "amplitude_angstrom":amplitude,"tolerance":tolerance,"max_iterations":max_iterations,
              "cart_step_limit_angstrom":cart_step_limit,"equilibrium_iterations":eq_iter,
              "max_equilibrium_cartesian_error_angstrom":max_eq,"max_target_residual":max_target,
              "max_relative_mass_tangent_error":max_tangent,"all_modes_tested":len(modes),
              "finite_scan_generated":False,"hamiltonian_modified":False,
              "step1_directory":str(internal_input),"step1_source_hash":step1_meta["source_text_sha256"],
              "masses_amu":masses.tolist(),
              "minimum_original_projected_mass_cosine":float(min(x[3] for x in rows)),
              "maximum_relative_mass_change_on_rigid_projection":float(max(abs(x[2]/x[1]-1) for x in rows)),
              "metric_formula":"dx=M^-1 J^T (J M^-1 J^T)^-1 du"}
    (out/"diagnostics.json").write_text(json.dumps(metadata,indent=2)+"\n")
    report=("AnhDis: validacion LOCAL interna -> cartesiana (paso 2)\n"
            f"Modos comprobados: {len(modes)}; amplitud: +/-{amplitude:g} A\n"
            f"Error cartesiano en Q=0: {max_eq:.3e} A\n"
            f"Maximo residuo en coordenadas independientes: {max_target:.3e}\n"
            f"Maximo error relativo de tangente (metrica de masas): {max_tangent:.3e}\n"
            "Todas las comprobaciones: PASS.\n"
            "La inversa usa la metrica G=J M^-1 J^T y elimina la libertad rigida por alineamiento.\n"
            f"Solapamiento minimo original/proyectado: {metadata['minimum_original_projected_mass_cosine']:.10f}\n"
            "La tangente se valida contra la componente vibracional proyectada, NO contra el vector redondeado completo.\n"
            "Los vectores originales y los scans cartesianos no se han sustituido.\n"
            "No se han generado scans finitos ni modificado el Hamiltoniano.\n")
    (out/"report.txt").write_text(report)
    print(report); print(f"Back-transform diagnostics saved to {out}")
    return metadata


def diagnose_and_save(coords, modes, log_text, output_dir, step=1e-5,
                      rank_rtol=1e-7, reference_length=1.0, source_log=""):
    """Validate before writing; reject a nonempty output directory.

    SVD convention: z = W(s-s0), W_R=1/reference_length, W_A=W_D=1.
    u = U_r.T z and B_u = U_r.T W B form an independent LOCAL basis.
    This does not identify 15 physical primitives or define a global trajectory.
    """
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Use a new/empty diagnostic directory: {out}")
    if not np.isfinite(reference_length) or reference_length <= 0 or not 0 < rank_rtol < 1:
        raise ValueError("Invalid reference length or relative rank tolerance.")
    coords, modes = np.asarray(coords, float), np.asarray(modes, float)
    expected = vibrational_dof(coords)
    if modes.shape != (expected, len(coords), 3) or not np.isfinite(modes).all():
        raise ValueError(f"Expected {expected} complete finite vibrational modes.")
    definitions = read_internals(log_text, len(coords), reference_coords=coords)
    s0 = internal_values(coords, definitions)
    printed = np.array([x.printed_value if x.kind == "R" else np.deg2rad(x.printed_value) for x in definitions])
    errors = internal_difference(s0, printed, definitions)
    display_errors = np.array([e if x.kind == "R" else np.rad2deg(e) for e,x in zip(errors,definitions)])
    ordinary = np.asarray([item.kind != "L" for item in definitions])
    linear = ~ordinary
    if np.any(ordinary) and np.max(np.abs(display_errors[ordinary])) > 5e-4:
        raise ValueError("Internal table and final geometry disagree (or dihedral conventions differ).")
    if np.any(linear) and np.max(np.abs(display_errors[linear])) > 1.0e-1:
        raise ValueError("Gaussian linear-bend table and equilibrium geometry disagree.")
    b_h = wilson_b(coords, definitions, step)
    b = wilson_b(coords, definitions, step/2)
    weights = np.array([1/reference_length if x.kind == "R" else 1.0 for x in definitions])
    scaled = weights[:, None]*b
    u, singular, _ = np.linalg.svd(scaled, full_matrices=False)
    rank = int(np.sum(singular > rank_rtol*singular[0]))
    if rank != expected:
        raise ValueError(f"B rank {rank}, expected {expected}: insufficient/singular internals.")
    basis = u[:, :rank]
    independent_b = basis.T @ scaled
    internal_modes = b @ modes.reshape(expected, -1).T
    direct = np.column_stack([
        internal_difference(internal_values(coords+step*d, definitions),
                            internal_values(coords-step*d, definitions), definitions)/(2*step)
        for d in modes
    ])
    fd_error = float(np.max(np.abs(weights[:,None]*(b-b_h))))
    tangent_error = float(np.max(np.abs(weights[:,None]*(internal_modes-direct))))
    rigid_error = float(np.max(np.abs(scaled @ rigid_directions(coords))))
    span_error = float(np.max(np.abs(scaled-basis @ independent_b)))
    for name, value in [("finite differences", fd_error), ("mode tangents", tangent_error),
                        ("rigid motions", rigid_error), ("independent span", span_error)]:
        if value > 1e-7:
            raise ValueError(f"Failed {name} check: {value:.3e}")
    metadata = {
        "stage": "local_B_diagnostics_only", "source_log": str(source_log),
        "source_text_sha256": hashlib.sha256(log_text.encode()).hexdigest(),
        "n_atoms": len(coords), "n_modes": expected, "n_internals": len(definitions),
        "B_shape": list(b.shape), "rank": rank, "rank_rtol": rank_rtol,
        "reference_length_angstrom": reference_length,
        "B_step_angstrom": step/2, "comparison_step_angstrom": step,
        "max_scaled_B_step_difference": fd_error, "max_scaled_mode_derivative_error": tangent_error,
        "max_scaled_rigid_derivative": rigid_error, "max_scaled_span_error": span_error,
        "singular_values": singular.tolist(),
        "normal_vectors_modified": False, "finite_curvilinear_geometries_generated": False,
        "reference_is_linear": rigid_directions(coords).shape[1] == 5,
        "internals": [{"label": x.label, "kind": x.kind, "atoms_1based": [a+1 for a in x.atoms],
                       "component": x.component, "axis_reference": x.axis_reference,
                       "frame_axis": x.frame_axis, "frame_direction_1": x.frame_direction_1,
                       "value": float(v), "unit": "angstrom" if x.kind=="R" else "radian"}
                      for x,v in zip(definitions,s0)],
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out/"B_matrix.dat", b, fmt="%.12e", header="Rows: internal_coordinates.dat; columns x1 y1 z1 ...; R rows A/A; A,L,D rows rad/A")
    np.savetxt(out/"B_dimensionless_internals.dat", scaled, fmt="%.12e", header="Derivative of dimensionless internals wrt Cartesian angstrom; W_R=1/reference_length")
    np.savetxt(out/"independent_basis_U.dat", basis, fmt="%.12e", header="u = U.T @ W @ (s-s0); local SVD basis; NOT normal modes")
    np.savetxt(out/"B_independent.dat", independent_b, fmt="%.12e", header="U.T W B; independent LOCAL coordinate Jacobian")
    np.savetxt(out/"internal_modes.dat", internal_modes, fmt="%.12e", header="B @ d_i; columns vib1...; R rows A/A, A,L,D rows rad/A; printed Gaussian vectors unchanged")
    with (out/"internal_coordinates.dat").open("w") as f:
        f.write("# Row Label Equilibrium(A_or_deg) d_s/d_Q_mode1(A/A_or_deg/A)\n")
        for i,x in enumerate(definitions):
            factor = 1 if x.kind=="R" else 180/np.pi
            f.write(f"{i+1:2d} {x.label:22s} {factor*s0[i]: .10f} {factor*internal_modes[i,0]: .10f}\n")
    (out/"diagnostics.json").write_text(json.dumps(metadata, indent=2)+"\n")
    report = (
        "AnhDis: comprobacion LOCAL de coordenadas internas (paso 1)\n"
        f"Atomos: {len(coords)}; modos: {expected}; internas: {len(definitions)}\n"
        f"B: {b.shape[0]} x {b.shape[1]}; rango: {rank}\n"
        f"Diferencia B(h)-B(h/2), escalada: {fd_error:.3e}\n"
        f"Error B*d frente a derivada directa, escalado: {tangent_error:.3e}\n"
        f"Derivada de movimientos rigidos, escalada: {rigid_error:.3e}\n"
        f"Error de representacion del espacio por U, escalado: {span_error:.3e}\n"
        "Todas las comprobaciones numericas anteriores: PASS (umbral 1e-7).\n"
        "No se han modificado los vectores ni generado trayectorias curvilineas.\n"
        "B*d NO es una back-transformation ni una prueba del Hamiltoniano.\n"
        f"La SVD usa distancias divididas por {reference_length:g} A y angulos en radianes.\n"
        "La eleccion de escala/base debe conservarse en una futura reconstruccion.\n"
    )
    (out/"report.txt").write_text(report)
    print(report)
    print(f"Diagnostics saved to {out}")
    return metadata
