"""Configuration for the curvilinear AnhDis workflow (stages a1--a7).

The options that are normally changed for a new calculation are grouped at
the beginning of this file. Numerical tolerances, output names, conversion
factors, and the complete cluster-node table are grouped afterwards and
normally do not need to be edited.

All molecular Cartesian coordinates and path coordinates Q and s are in
angstrom. Atomic masses are in unified atomic mass units (u/amu), Gaussian
energies are in hartree, angles are evaluated internally in radians, and
frequencies are in cm^-1.
"""

from pathlib import Path


def make_q_grid(q_min, q_max, step):
    """Return an inclusive uniform Q grid with stable decimal rounding."""
    if step <= 0:
        raise ValueError("The Q-grid step must be positive.")
    n_steps = int(round((q_max - q_min) / step))
    if n_steps < 1 or abs(q_min + n_steps * step - q_max) > 1.0e-10:
        raise ValueError(
            "Q-grid limits must be separated by an integer number of steps."
        )
    return [round(q_min + i * step, 10) for i in range(n_steps + 1)]


# ============================================================================
# USER SETTINGS -- inspect these first for each molecule or calculation
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# 1. Input files, output directories, and selected vibrational modes
# ----------------------------------------------------------------------------

# Gaussian optimization + frequency output used by a1. The command-line
# option --log can override it without modifying this file.
GAUSSIAN_LOG = PROJECT_ROOT / "b3lyp_631Gdp.log"

# Gaussian single-point header used by a2. It must contain {checkpoint} in
# %chk, end with charge/multiplicity, and use Cartesian coordinates in angstrom.
GAUSSIAN_HEADER = PROJECT_ROOT / "gaussian_head.com"

# Main data tree produced by a1 and consumed by a2--a7.
SCAN_ROOT = PROJECT_ROOT / "curvilinear_scans"

# Empty means all 3N-6 Gaussian modes. Examples: [1] or [1, 3, 9].
SELECTED_MODES = []


# ----------------------------------------------------------------------------
# 2. Curvilinear scan sent to Gaussian (a1--a4)
# ----------------------------------------------------------------------------

# Default electronic-energy grid for every mode not listed below.
Q_ENERGY_ANG = make_q_grid(-0.4, 0.4, 0.1)

# Optional mode-specific electronic-energy grids. These replace the default
# grid only for the listed modes. Add or remove entries as needed.
MHP_EXTENDED_Q_ENERGY_ANG = make_q_grid(-1.8, 2.4, 0.1)
Q_ENERGY_ANG_BY_MODE = {
    #1: MHP_EXTENDED_Q_ENERGY_ANG,
    # 2: make_q_grid(-1.4, 1.4, 0.1),
    # 3: make_q_grid(-0.5, 0.5, 0.1),
}

# Denser, electronic-structure-free step used by a1 to follow the path and
# calculate mu(Q), s(Q), the Wilson B matrix, and the SVD diagnostics.
PATH_Q_STEP_ANG = 0.025


# ----------------------------------------------------------------------------
# 3. Electronic-structure energy selected from the Gaussian outputs (a4--a5)
# ----------------------------------------------------------------------------

# Allowed values: "SCF", "MP2", "CCSD", "CCSD(T)", or "AUTO".
ENERGY_SOURCE = "SCF"


# ----------------------------------------------------------------------------
# 4. Potential fitting options (a5)
# ----------------------------------------------------------------------------

# Default potential representation and optional mode-specific replacements.
# "polynomial" retains the derivative expansion used by the original
# workflow. "pchip" interpolates the ab initio points without global
# polynomial oscillations and adds quadratic confining tails outside the
# fitted interval. Example: FIT_MODEL_BY_MODE = {1: "pchip"}.
FIT_MODEL = "polynomial"
FIT_MODEL_BY_MODE = {
    1: "pchip",
        }

# Default polynomial degree and optional mode-specific replacements.
FIT_DEGREE = 4
FIT_DEGREE_BY_MODE = {
    #1: 6,
    #2: 6,
}

# Optional fitting intervals in the original path parameter Q (angstrom).
# They select existing ab initio points only; they do not regenerate or rerun
# geometries. A mode absent here uses all its available points.
FIT_Q_RANGE_ANG_BY_MODE = {
    #1: (-1.0, 2.4),
    # 2: (-1.0, 1.0),
}

# Optional outward-monotonic constraints. Use True only if the calculated
# potential rises away from equilibrium over the relevant domain. Periodic
# or genuinely multi-well potentials require a different representation.
FIT_MONOTONIC_OUTWARD_BY_MODE = {
    1: True,
    2: True,
}

# Cubic-term control inherited from the original AnhDis implementation.
# None disables this empirical bound for a specific mode.
FIT_MAX_CUBIC_FACTOR = 10.0
FIT_MAX_CUBIC_FACTOR_BY_MODE = {
    1: None,
    2: None,
    4: 1.0,
}

# Optional extra least-squares weights for polynomial fits. Each tuple is
# (Q_min_angstrom, Q_max_angstrom, multiplier). PCHIP modes interpolate the
# points and therefore do not use these weights.
FIT_EXTRA_WEIGHT_Q_BY_MODE = {}

# PCHIP is used only inside the selected ab initio interval. Outside it, a
# quadratic tail with curvature factor*k_harm confines the quantum calculation.
# It matches the endpoint slope when that slope points outward; otherwise the
# slope is clipped to zero and a5 reports it. Increase the factor only if a6
# reports density at a grid edge.
PCHIP_TAIL_CURVATURE_FACTOR = 1.0
PCHIP_TAIL_CURVATURE_FACTOR_BY_MODE = {}

# Half-range used only for the diagnostic full-potential plot. It is given in
# bohr, although a5 displays the horizontal axis in angstrom.
FULL_POTENTIAL_HALF_RANGE_BOHR = 7.0


# ----------------------------------------------------------------------------
# 5. One-dimensional quantum calculation (a6)
# ----------------------------------------------------------------------------

TEMP = 0.0001  # K  if 0K put 0.0001 or less
L = 7.0        # numerical-grid half-length in bohr
NP = 1400      # number of grid points from -L to +L


# ----------------------------------------------------------------------------
# 6. Nuclear-ensemble generation (a7)
# ----------------------------------------------------------------------------

# a7 creates NSAMPLES anharmonic and NSAMPLES harmonic geometries.
NSAMPLES = 100

# Fixed seed makes the sampled ensemble reproducible.
A7_RANDOM_SEED = 1

# The directory must be new or empty for each a7 execution.
ENSEMBLE_ROOT = PROJECT_ROOT / "ensemble_output"


# ----------------------------------------------------------------------------
# 7. Cluster and node selection (a3)
# ----------------------------------------------------------------------------

PBS_USER = None  # None uses the current $USER.
PBS_PPN = 2

# Numeric node identifiers; leading zeros are added automatically. This is
# the list normally edited when choosing the nodes available for submission.
ALLOWED_NODE_NUMS = [51, 52, 2, 1]
ALLOWED_NODES = [f"qcexnod{number:02d}" for number in ALLOWED_NODE_NUMS]

# Change these only when Gaussian or the cluster layout changes.
GAUSSIAN_PROFILE = "/soft/g16.a03/g16/bsd/g16.profile"
GAUSSIAN_COMMAND = "g16"
SCRATCH_ROOT = "/scr"
SUBMIT_COMMAND = "qsub"


# ============================================================================
# ADVANCED SETTINGS -- normally leave unchanged
# ============================================================================


# ----------------------------------------------------------------------------
# a1: internal coordinates, Wilson B matrix, and back-transformation
# ----------------------------------------------------------------------------

INTERNAL_FD_STEP_ANG = 1.0e-5
INTERNAL_RANK_RTOL = 1.0e-7
INTERNAL_REFERENCE_LENGTH_ANG = 1.0
BACKTRANS_TEST_AMPLITUDE_ANG = 1.0e-3
BACKTRANS_TOL = 1.0e-11
BACKTRANS_MAX_ITER = 50
BACKTRANS_CART_STEP_LIMIT_ANG = 0.05

# Independent numerical validation of mu(Q) at electronic-energy points.
VALIDATE_PATH_MASS = True
PATH_MASS_FD_STEP_ANG = 1.0e-4
PATH_MASS_TANGENT_RTOL = 2.0e-6


# ----------------------------------------------------------------------------
# a4--a5: output validation and numerical fitting controls
# ----------------------------------------------------------------------------

# All Q=0 geometries are identical. This tolerance detects mixed or stale
# outputs and unexpectedly inconsistent reference energies.
REFERENCE_ENERGY_TOL_HARTREE = 1.0e-8

FIT_ENFORCE_STATIONARY_EQUILIBRIUM = True

# Backward-compatible quartic option and the general quartic/sextic rule.
FIT_REQUIRE_POSITIVE_QUARTIC = True
FIT_REQUIRE_POSITIVE_HIGHEST_EVEN = True

FIT_WEIGHT_SCALE_ANG = 0.20
FIT_GRID_POINTS = 1000
FIT_PLOT_ENERGY_UNIT = "hartree"
FIT_SHAPE_CONSTRAINT_GRID_POINTS = 1001


# ----------------------------------------------------------------------------
# a6: Hamiltonian, thermal-state, and diagnostic tolerances
# ----------------------------------------------------------------------------

THERMAL_WEIGHT_CUTOFF = 1.0e-3
A6_MAX_EIGENSTATES = 200
A6_MAX_PLOTTED_STATES = 12
A6_HARMONIC_E0_RTOL = 1.0e-2
A6_EDGE_POINTS = 5
A6_EDGE_DENSITY_RATIO_TOL = 1.0e-6
A6_SCAN_TAIL_PROBABILITY_TOL = 1.0e-3
A6_NEGATIVE_MINIMUM_TOL_CM = 10.0

# The Fourier transform is zero-padded only to obtain a smoother momentum
# grid; it does not alter the coordinate-space Hamiltonian or eigenfunctions.
A6_MOMENTUM_FFT_PAD_FACTOR = 8
A6_MOMENTUM_EDGE_DENSITY_RATIO_TOL = 1.0e-8
A6_MOMENTUM_KINETIC_RTOL = 2.0e-2
A6_MOMENTUM_HARMONIC_VARIANCE_RTOL = 2.0e-2


# ----------------------------------------------------------------------------
# a7: sampling and internal-to-Cartesian reconstruction controls
# ----------------------------------------------------------------------------

A7_CONTINUATION_STEP_ANG = 0.05
A7_MAX_CONTINUATION_STEPS = 250
A7_MAX_ATTEMPTS_FACTOR = 20

# a7 samples only inside the validated a1 path and, for the anharmonic case,
# inside the ab initio interval used by a5. It stops if the omitted probability
# exceeds these limits unless --allow-truncation is explicitly supplied.
A7_MAX_TAIL_PROBABILITY_PER_MODE = 1.0e-3
A7_MAX_JOINT_TAIL_PROBABILITY = 1.0e-2

A7_TORSION_CHART_MARGIN_RAD = 0.20
A7_MIN_INTERATOMIC_DISTANCE_ANG = 0.50
A7_VALIDATE_SINGLE_MODE_PATHS = True
A7_PATH_RECONSTRUCTION_TOL_ANG = 1.0e-8
A7_VELOCITY_INTERNAL_RTOL = 1.0e-6
A7_HISTOGRAM_BINS = 30
A7_SAVE_INDIVIDUAL_XYZ = True


# ----------------------------------------------------------------------------
# Output filenames used by a4--a7
# ----------------------------------------------------------------------------

CALCULATION_STATUS_FILE = "calculation_status.tsv"

FIT_COEFF_FILE = "fit_coefficients.dat"
FIT_SUMMARY_FILE = "fit_summary.tsv"
FIT_PDF = "curvilinear_potential_fits.pdf"
FULL_POTENTIAL_PDF = "full_potentials.pdf"
METRIC_PDF = "curvilinear_path_metrics.pdf"

PROBABILITY_DIR = "probabilities"
MOMENTUM_DIR = "momenta"
WAVE_PDF = "wave_wigner.pdf"
EXCITED_STATES_PDF = "test_excited_states.pdf"
MOMENTUM_PDF = "momentum_distributions.pdf"
THERMAL_SUMMARY_FILE = "thermal_summary.dat"
A6_DIAGNOSTICS_FILE = "wavefunction_diagnostics.tsv"

A7_MODE_DIAGNOSTICS_FILE = "mode_sampling_diagnostics.tsv"
A7_ENSEMBLE_DIAGNOSTICS_FILE = "ensemble_geometry_diagnostics.tsv"
A7_PATH_VALIDATION_FILE = "path_reconstruction_validation.tsv"
A7_MANIFEST_FILE = "a7_manifest.json"


# ----------------------------------------------------------------------------
# Unit conversions
# ----------------------------------------------------------------------------

BOHR_TO_ANG = 0.529177210903
AMU_TO_EMASS = 1822.888486209
CM_TO_HARTREE = 4.556335252912e-6
HARTREE_TO_CM = 1.0 / CM_TO_HARTREE
KB_AU = 3.166811563e-6
ATOMIC_TIME_TO_FS = 0.024188843265857
BOHR_PER_ATOMIC_TIME_TO_ANG_PER_FS = BOHR_TO_ANG / ATOMIC_TIME_TO_FS
MOMENTUM_AU_TO_AMU_ANG_PER_FS = (
    BOHR_PER_ATOMIC_TIME_TO_ANG_PER_FS / AMU_TO_EMASS
)


# ----------------------------------------------------------------------------
# Complete cluster-node capacity table used internally by a3
# ----------------------------------------------------------------------------

NODE_CPUS = {
    "qcexnod62": 24, "qcexnod61": 24, "qcexnod60": 24,
    "qcexnod59": 24, "qcexnod58": 24, "qcexnod57": 24,
    "qcexnod56": 24, "qcexnod55": 24, "qcexnod54": 24,
    "qcexnod52": 3, "qcexnod51": 3, "qcexnod50": 10,
    "qcexnod01": 12, "qcexnod02": 12, "qcexnod03": 24,
    "qcexnod10": 6, "qcexnod11": 6, "qcexnod12": 6,
    "qcexnod13": 6, "qcexnod14": 6, "qcexnod15": 6,
    "qcexnod16": 6, "qcexnod17": 6, "qcexnod18": 6,
    "qcexnod19": 6, "qcexnod04": 16,
}

