"""Configuration for the curvilinear AnhDis workflow (stages a1--a7).

The options that are normally changed for a new calculation are grouped at
the beginning of this file. Numerical tolerances, output names, conversion
factors, and the complete cluster-node table are grouped afterwards and
normally do not need to be edited.

All molecular Cartesian coordinates and path coordinates Q and s are in
angstrom. Atomic masses are in unified atomic mass units (u/amu), electronic
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

# Program and output used only for the equilibrium geometry, atomic masses,
# harmonic frequencies, and normal modes read by a1. Allowed software values:
# "auto", "gaussian", or "orca". ORCA .out and .log files are both accepted.
# Command-line options --software and --log override these settings.
OPT_FREQ_SOFTWARE = 'orca'
OPT_FREQ_OUTPUT = PROJECT_ROOT / 'hoso.log'

# Backward-compatible alias for older scripts/configurations. New code reads
# OPT_FREQ_OUTPUT first; keep this alias unless an external script requires it.
GAUSSIAN_LOG = OPT_FREQ_OUTPUT

# Gaussian single-point header used by a2. It must contain {checkpoint} in
# %chk, end with charge/multiplicity, and use Cartesian coordinates in angstrom.
GAUSSIAN_HEADER = PROJECT_ROOT / 'gaussian_head.com'

# Electronic-structure program used for the a2--a5 scan energies. "auto"
# follows the program recorded by a1; an explicit value also permits mixed
# workflows (for example ORCA frequencies followed by Gaussian single points).
# Allowed values: "auto", "gaussian", or "orca".
SINGLE_POINT_SOFTWARE = 'orca'

# ORCA single-point template used by a2. It must contain {job_tag}, contain
# nprocs {nprocs} in a %pal block, and end with * xyz charge multiplicity.
ORCA_HEADER = PROJECT_ROOT / 'orca_head.inp'

# When True and the a1 source and a2 single-point program are the same, a2
# regenerates the active header directly from the optimization/frequency
# output. Set False only for a deliberately different single-point method or
# a custom Gaussian Gen/GenECP/ORCA block that cannot be inferred safely.
AUTO_GENERATE_SINGLE_POINT_HEADER = True

# Main data tree produced by a1 and consumed by a2--a7.
SCAN_ROOT = PROJECT_ROOT / 'curvilinear_scans'

# Empty means every vibrational mode: 3N-6 for a nonlinear molecule
# and 3N-5 for a linear molecule. Examples: [1] or [1, 3, 9].
SELECTED_MODES = []


# ----------------------------------------------------------------------------
# 2. Coordinate path and electronic-energy scan (a1--a4)
# ----------------------------------------------------------------------------

# Geometry model used for the electronic-energy paths and, downstream, for
# ensemble reconstruction.  "curvilinear" retains the iterative internal-
# coordinate back-transformation.  "cartesian" uses the conventional analytic
# rectilinear normal-mode path R(Q)=R0+Q*d_i; then mu(Q)=mu(0) and s=Q.
# The command-line option --coordinate-system overrides this value for a1.
A1_COORDINATE_SYSTEM = 'curvilinear'

# Default electronic-energy grid for every mode not listed below.
Q_ENERGY_ANG = make_q_grid(-0.4, 0.4, 0.10000000000000003)

# Optional mode-specific electronic-energy grids. These replace the default
# grid only for the listed modes. Add or remove entries as needed.
MHP_EXTENDED_Q_ENERGY_ANG = make_q_grid(-1.8, 2.4, 0.1)
Q_ENERGY_ANG_BY_MODE = {#1: [-2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
                        }

# Denser, electronic-structure-free step used by a1 to follow the path and
# calculate mu(Q), s(Q), the Wilson B matrix, and the SVD diagnostics.
PATH_Q_STEP_ANG = 0.025

# Independent mode processes used by a1. Use 1 on a login/master host; the
# ./anhdis interface can generate and submit a PBS job for larger values.
A1_WORKERS = 12


# ----------------------------------------------------------------------------
# 3. Electronic-structure energy selected from the outputs (a4--a5)
# ----------------------------------------------------------------------------

# Allowed values: "SCF", "MP2", "CCSD", "CCSD(T)", or "AUTO". AUTO reads
# the method recorded by a2: for example, an MP2 input selects the MP2 total
# energy rather than the preceding SCF reference energy.
ENERGY_SOURCE = 'SCF'


# ----------------------------------------------------------------------------
# 4. Potential fitting options (a5)
# ----------------------------------------------------------------------------

# Default potential representation and optional mode-specific replacements.
# "polynomial" retains the derivative expansion used by the original
# workflow. "pchip" interpolates the ab initio points without global
# polynomial oscillations and adds quadratic confining tails outside the
# fitted interval. Example: FIT_MODEL_BY_MODE = {1: "pchip"}.
FIT_MODEL = 'pchip'
FIT_MODEL_BY_MODE = {}

# Default polynomial degree and optional mode-specific replacements.
FIT_DEGREE = 4
FIT_DEGREE_BY_MODE = {}

# Optional fitting intervals in the original path parameter Q (angstrom).
# They select existing ab initio points only; they do not regenerate or rerun
# geometries. A mode absent here uses all its available points.
FIT_Q_RANGE_ANG_BY_MODE = {}

# Optional outward-monotonic constraints. Use True only if the calculated
# potential rises away from equilibrium over the relevant domain. Periodic
# or genuinely multi-well potentials require a different representation.
FIT_MONOTONIC_OUTWARD_BY_MODE = {1: True, 2: True}

# Cubic-term control inherited from the original AnhDis implementation.
# None disables this empirical bound for a specific mode.
FIT_MAX_CUBIC_FACTOR = 10.0
FIT_MAX_CUBIC_FACTOR_BY_MODE = {1: None, 2: None}

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
FULL_POTENTIAL_HALF_RANGE_BOHR = 5.0


# ----------------------------------------------------------------------------
# 5. One-dimensional quantum calculation (a6)
# ----------------------------------------------------------------------------

TEMP = 298.15  # K
L = 5.0        # numerical-grid half-length in bohr
NP = 1000      # number of grid points from -L to +L


# ----------------------------------------------------------------------------
# 6. Nuclear-ensemble generation (a7)
# ----------------------------------------------------------------------------

# a7 creates NSAMPLES anharmonic and NSAMPLES harmonic geometries.
NSAMPLES = 100

# Fixed seed makes the sampled ensemble reproducible.
A7_RANDOM_SEED = 1

# Independent geometry-reconstruction processes. Candidate coordinates and
# momenta remain drawn in the parent process, preserving seeded results.
A7_WORKERS = 12

# The directory must be new or empty for each a7 execution.
ENSEMBLE_ROOT = PROJECT_ROOT / 'ensemble_output'


# ----------------------------------------------------------------------------
# 7. Cluster and node selection (a3)
# ----------------------------------------------------------------------------

PBS_USER = None  # None uses the current $USER.
PBS_PPN = 2

# Numeric node identifiers; leading zeros are added automatically. This is
# the list normally edited when choosing the nodes available for submission.
ALLOWED_NODE_NUMS = [57, 58 , 59, 60]
ALLOWED_NODES = [f"qcexnod{number:02d}" for number in ALLOWED_NODE_NUMS]

# Compute node used by the dynamically generated parallel a1/a7 PBS script.
# "auto" lets PBS choose; a value such as "qcexnod62" requests that node.
ANHDIS_PARALLEL_NODE = 'qcexnod62'

# Change these only when the electronic-structure installation or cluster
# layout changes.
GAUSSIAN_PROFILE = "/soft/g16.a03/g16/bsd/g16.profile"
GAUSSIAN_COMMAND = "g16"

# ORCA_PROFILE may be empty when the environment inherited by PBS already
# contains ORCA and its MPI libraries. For parallel ORCA, ORCA_COMMAND should
# preferably be the absolute path to the ORCA driver; do not put mpirun here.
ORCA_PROFILE = ""
ORCA_COMMAND = "orca"
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
# Coordinate-model comparison: geometry screening and energy validation
# ----------------------------------------------------------------------------

# Points within this many analytic harmonic standard deviations are used for
# the thermally relevant summary.  These tolerances classify a mode for
# screening; they are not universal physical constants and remain visible in
# the validator output.
COORD_COMPARE_THERMAL_SIGMA_LIMIT = 3.0
COORD_COMPARE_GEOMETRY_RMSD_TOL_ANG = 0.01
COORD_COMPARE_RELATIVE_PATH_TOL_PERCENT = 2.0
COORD_COMPARE_ENERGY_WEIGHTED_RMSE_TOL_CM = 10.0
COORD_COMPARE_ENERGY_MAX_TOL_CM = 25.0

COORD_COMPARE_POINT_FILE = "coordinate_model_points.tsv"
COORD_COMPARE_SUMMARY_FILE = "coordinate_model_summary.tsv"
COORD_COMPARE_PDF = "coordinate_model_comparison.pdf"
COORD_COMPARE_REPORT = "coordinate_model_report.txt"


# ----------------------------------------------------------------------------
# Output filenames used by a4--a7
# ----------------------------------------------------------------------------

# Every direct or PBS execution of a1--a7 appends one wall-time record here.
EXECUTION_TIMING_FILE = PROJECT_ROOT / "anhdis_execution_times.tsv"

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
