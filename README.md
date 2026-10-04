# Cartesian/curvilinear AnhDis workflow: stages `a1`--`a7`

## Program description and theoretical basis

AnhDis constructs vibrational nuclear ensembles from a Gaussian or ORCA
optimization/frequency calculation plus one-dimensional electronic-energy
scans. It
supports rectilinear Cartesian normal-mode paths and nonlinear curvilinear
paths, polynomial or PCHIP potentials, finite-temperature quantum position
and momentum distributions, and reconstruction of molecular geometries,
velocities, and momenta.

For mode \(i\), the Cartesian path is generated analytically as

\[
\mathbf R_i^{\mathrm{cart}}(Q)=\mathbf R_0+Q\mathbf d_i,
\]

whereas the curvilinear path is defined through the displacement of the
independent internal coordinates,

\[
\Delta\boldsymbol\xi\!\left(\mathbf R_i^{\mathrm{curv}}(Q)\right)
   =Q\mathbf l_i.
\]

The curvilinear geometry is obtained iteratively while recalculating the
Wilson matrix. Its coordinate-dependent effective mass \(\mu_i(Q)\) is
converted to the constant-mass coordinate

\[
s_i(Q)=\int_0^Q\sqrt{\frac{\mu_i(q)}{\mu_i(0)}}\,dq.
\]

Single-point energies define
\(V_i(s)=E[\mathbf R_i(Q(s))]-E(\mathbf R_0)\), represented by either a
polynomial or a shape-preserving PCHIP interpolation. In atomic units, `a6`
solves the one-dimensional Hamiltonian

\[
\hat H_i=-\frac{1}{2\mu_i(0)}\frac{d^2}{ds_i^2}+V_i(s_i)
\]

by finite differences. The eigenstates give the thermal marginal

\[
\rho_i(s,T)=Z_i^{-1}\sum_v e^{-\beta\varepsilon_{iv}}
|\psi_{iv}(s)|^2,
\]

and the corresponding momentum marginal is obtained by Fourier-transforming
the same eigenfunctions. `a7` samples the positive separable approximation
\(F_i(s_i,p_i;T)=\rho_i(s_i,T)\rho_i(p_i,T)\). Thus positions and momenta are
sampled independently; AnhDis does not construct an anharmonic Wigner
function. The harmonic reference ensemble is instead sampled analytically
from the exact unbounded thermal Gaussian marginals determined by each
frequency and \(\mu_i(0)\).

## Interactive setup and launcher

Run the interactive helper from the project directory:

```bash
./anhdis
```

The setup wizard asks independently for the optimization/frequency program
and the single-point program (`auto`, `gaussian`, or `orca`), their output and
active template, Cartesian or curvilinear paths, selected modes, the
default Q scan, energy source, potential model, temperature, ensemble size,
random seed, a1/a7 workers, output directories, and cluster nodes.
By default, the active single-point template is generated directly from the
same optimization/frequency output, so the user does not have to construct
`gaussian_head.com` or `orca_head.inp` manually.
Entering `0` K stores `0.0001` K, the small positive temperature used as the
numerical zero-temperature limit by a6.

The helper edits the original assignments in `config.py`, creates
`config.py.anhdis.bak`, and leaves unrelated scientific and advanced settings
unchanged. It does not create a second configuration layer at the end of the
file. If it finds the marked override block written by an older interface, it
migrates those effective values into the original assignments and removes the
old block. This matters because Python otherwise uses the last assignment in a
file. After setup, run `./anhdis` again and select a stage. Submissions and
resubmissions always require explicit confirmation.
The **Configure limits and fits by mode** menu changes a geometry grid for a
mode or a group such as `1-30`, assigns a polynomial degree or PCHIP per mode,
sets a fitting-only interval, configures or disables the cubic-term bound,
removes overrides, and displays the result. It
updates the same dictionaries used by `a1` and `a5`; no second configuration
file is created.

In the workflow menu, selecting **Generate a1 scan geometries**, **Safely
extend an a1 scan**, or **Generate nuclear ensembles** opens a second prompt:
run in the current session or submit to a compute node. For a compute-node
execution, the interface asks for the worker count and the node name; `auto`
lets PBS choose. Those answers apply only to that submission and do not alter
`config.py`. Before every `a7` execution, it also checks the configured output
directory. If that directory is nonempty, it asks for a new one and proposes
the first available name, such as `ensemble_output_seed1_run2`.

Useful non-interactive forms are:

```bash
./anhdis --show
./anhdis --configure
./anhdis --configure-modes
./anhdis --check
./anhdis --timings
./anhdis --run a1
./anhdis --run a1-pbs
./anhdis --run a1-update-pbs
./anhdis --run a2
./anhdis --run a2-update
./anhdis --run submit
./anhdis --run a4-strict
./anhdis --run a5
./anhdis --run a6
./anhdis --run a7
./anhdis --run a7-pbs
```

The node and worker count may be overridden for one submission without
changing `config.py`:

```bash
./anhdis --run a1-pbs --node qcexnod56 --workers 12 --yes
./anhdis --run a7-pbs --node qcexnod62 --workers 12 --yes
```

An a7 output directory can also be selected for one launcher invocation:

```bash
./anhdis --run a7-pbs --node 62 --workers 12 \
  --output ensemble_output_seed1_run2 --yes
```

The launcher refuses a nonempty `--output` path before submitting the PBS job.

Without these two options, the launcher uses `ANHDIS_PARALLEL_NODE` and the
corresponding `A1_WORKERS` or `A7_WORKERS` value from `config.py`.

The original a1--a7 commands remain fully supported.

### Execution-time history

Every real execution of stages `a1`--`a7`, whether started directly, through
`./anhdis`, or inside a PBS job, reports its wall-clock duration on completion:

```text
[TIMING] a7: OK in 01:18:43.527
[TIMING] History: /path/to/project/anhdis_execution_times.tsv
```

The shared tab-separated history records the start and finish times, stage,
success/failure status, elapsed time, compute host, process ID, working
directory, complete command, and any final exception. Concurrent PBS jobs use
a file lock while appending. Inspect totals, averages, maxima, failures, and
the most recent executions with menu option **Show execution-time summary** or

```bash
./anhdis --timings
```

This is elapsed wall time, so the reported `a1`/`a7` duration includes their
worker processes but excludes time spent waiting in the PBS queue. Stage `a3`
measures input/script preparation and submission only; the Gaussian or ORCA
single-point jobs have their own scheduler runtimes and are not included in
the `a3` value. Help commands (`--help`) are not written to the history.

When normal `a1` is selected through the interface and `scan_manifest.json`
already exists, the launcher compares every configured grid with the
manifest. If one or more grids were extended, it automatically runs
`a1 --update --modes <changed modes>`; if nothing changed, it exits without
regenerating files. Likewise, normal interface action `a2` automatically
becomes `a2 --update --modes <changed modes>` after an a1 extension. Existing
input/output identities are preserved.

`./anhdis --check` reads `scan_manifest.json`, `input_manifest.json`, the
single-point outputs, `calculation_status.tsv`, and the `a5`--`a7` metadata. It
labels stages as `OK`, `MISSING`, `INCOMPLETE`, `STALE`, or `ERROR` and prints
the appropriate next action. In particular, a changed a1 manifest is reported
as a stale a2 manifest instead of leaving the user to diagnose a manifest
exception.

This package builds one-dimensional Cartesian or curvilinear vibrational
potentials and thermal nuclear ensembles. The curvilinear workflow uses
internal-coordinate trajectories, while the optional Cartesian workflow uses
analytic rectilinear normal-mode paths. The workflow
retains the `vibN/Q` directory structure, but the final geometries are obtained
by nonlinear internal-to-Cartesian back-transformation rather than by directly
adding Cartesian displacement vectors.

| Stage | Main purpose |
|---|---|
| `a1` | Selected Cartesian or curvilinear paths, effective masses `mu(Q)`, and constant-mass coordinate `s(Q)` |
| `a2` | Gaussian or ORCA single-point input generation |
| `a3` | PBS script generation and calculation submission or resubmission |
| `a4` | Calculation termination, energy, input, and identity checks |
| `a5` | Fitting of the one-dimensional potentials `V(s)` |
| `a6` | One-dimensional Hamiltonians and thermal position/momentum distributions |
| `a7` | Independent position/momentum sampling and reconstruction of Cartesian geometries, velocities, and momenta |

## 1. Requirements and model

Requirements:

- Python 3;
- NumPy, SciPy, and Matplotlib;
- a normally terminated Gaussian or ORCA optimization/frequency calculation;
- Gaussian and/or ORCA plus PBS/Torque access for the single-point calculations.

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The implementation supports both nonlinear and linear molecules. For Gaussian,
primitive coordinates `R`, `A`, and `D` are accepted, as are paired
two-component linear bends `L(i,j,k,l,-1/-2)` with either a real reference atom
or Gaussian's automatic transverse axes (`l=-1/-2`). For ORCA, `a1` reads the
final Cartesian coordinates, atomic masses, frequencies, normal modes, and
redundant `B`, `A`, and `D` coordinates. ORCA's zero-based mode columns are
converted to AnhDis vibrational modes numbered from 1 after excluding the five
or six rigid modes. A near-linear ORCA angle is converted automatically into a
two-component linear bend. The mode mass is obtained from the normalized ORCA
Cartesian vector as `sum_a m_a*|d_ia|^2`.

The rigid rank and number of vibrational modes are detected from the
equilibrium geometry: `3N-6` for a nonlinear molecule and `3N-5` for a linear
molecule. No molecule-specific atom or mode list is used. Dummy/ghost atoms,
connectivity changes, incomplete linear-bend pairs, and paths that cross a
torsional chart discontinuity at `+/-pi` are rejected rather than silently
approximated. The current ORCA reader requires the redundant-internal table
produced by a standard geometry optimization because those definitions are
also retained for downstream reconstruction diagnostics. The output must
contain one complete vibrational analysis and `ORCA TERMINATED NORMALLY`.

The parsed ORCA conventions follow the official
[vibrational-frequency documentation](https://www.faccts.de/docs/orca/6.0/manual/contents/typical/frequencies.html)
and its [zero-based internal numbering convention](https://www.faccts.de/docs/orca/6.0/manual/contents/faq.html).
Stages `a2`--`a5` also support ORCA: `a2` writes `.inp` files, `a3` runs the
ORCA driver in local scratch, and `a4/a5` verify `ORCA TERMINATED NORMALLY`
and parse the requested energy. Gaussian remains fully supported and old
Gaussian input manifests remain readable.

The statistical model uses independent vibrational modes and the positive,
separable approximation

```text
F_i(s_i,p_i;T) = rho_i(s_i,T) rho_i(p_i,T).
```

Position and momentum are therefore sampled independently. The code does not
construct an anharmonic Wigner function. The nonlinear final
back-transformation combines all sampled internal-coordinate targets, but it
does not introduce a multidimensional energetic coupling between modes.

## 2. Units

| Quantity | Unit |
|---|---|
| Cartesian coordinates and `.xyz` files | angstrom |
| Path parameter `Q` | angstrom |
| Constant-mass coordinate `s` stored by `a1/a5` | angstrom |
| Quantum grid and `s` sampling inside `a6/a7` | bohr |
| Atomic and effective masses | unified atomic mass unit (amu) |
| Internal angles | radians |
| Electronic energies and fitted potentials | hartree |
| Frequencies and reported fitting errors | cm^-1 |
| Modal momentum `p_s` | atomic unit of momentum |
| Cartesian atomic momentum and `.mom` files | atomic unit of momentum |
| Modal/Cartesian velocity and `.vel` files | angstrom/fs |

Each line of a `.vel` file is the Cartesian velocity
`(dx/dt, dy/dt, dz/dt)` of the corresponding atom in the associated `.xyz`
file. A `.mom` file contains Cartesian momenta, not velocities.

The constant-mass coordinate is defined by

```text
ds/dQ = sqrt(mu(Q)/mu(0)).
```

Consequently, the one-dimensional Hamiltonian uses the constant mass `mu(0)`.
Before constructing it, `a6` converts `s` from angstrom to bohr and `mu(0)`
from amu to electron masses.

## 3. Required project files

The working directory must contain:

```text
anhdis
config.py
internal_coordinates.py
workflow_utils.py
single_point_templates.py
a1_geometries.py
a2_gaussian_input_generator.py
a3_lanzador.py
a4_check_calculation.py
a5_fitting.py
a6_probdis_therm.py
a7_coord_therm.py
optimization_frequency.log  # Gaussian, or an ORCA .out/.log file
```

With `AUTO_GENERATE_SINGLE_POINT_HEADER = True`, `gaussian_head.com` or
`orca_head.inp` is created automatically from that optimization/frequency
output when `a2` starts. It therefore does not need to exist in a new project.

If executable permissions are lost when copying the files:

```bash
chmod +x a1_geometries.py a2_gaussian_input_generator.py a3_lanzador.py \
  a4_check_calculation.py a5_fitting.py a6_probdis_therm.py a7_coord_therm.py \
  validate_coordinate_models.py anhdis
```

## 4. Main configuration

The options most frequently changed by the user are placed near the beginning
of `config.py`.

| Option | Meaning |
|---|---|
| `OPT_FREQ_SOFTWARE` | `auto`, `gaussian`, or `orca` for the `a1` source |
| `OPT_FREQ_OUTPUT` | Gaussian `.log` or ORCA `.out`/`.log` read by `a1` |
| `GAUSSIAN_LOG` | Legacy alias for `OPT_FREQ_OUTPUT`; retained for old scripts |
| `GAUSSIAN_HEADER` | Single-point template used by `a2` |
| `SINGLE_POINT_SOFTWARE` | `auto`, `gaussian`, or `orca` for the `a2`--`a5` energy scans; `auto` follows the program recorded by `a1` |
| `ORCA_HEADER` | Geometry-free ORCA single-point template used by `a2` |
| `AUTO_GENERATE_SINGLE_POINT_HEADER` | Regenerate the active Gaussian/ORCA template from the a1 output before a2 |
| `SCAN_ROOT` | Common output directory used by `a1`--`a7` |
| `SELECTED_MODES` | Mode list; empty means all modes (`3N-6` nonlinear, `3N-5` linear) |
| `Q_ENERGY_ANG` | Default electronic-energy scan grid |
| `Q_ENERGY_ANG_BY_MODE` | Optional mode-specific scan grids |
| `ENERGY_SOURCE` | Energy parsed by `a4/a5`; `AUTO` infers `SCF`, `MP2`, `CCSD`, or `CCSD(T)` from the actual a2 method |
| `FIT_MODEL`, `FIT_MODEL_BY_MODE` | Default and mode-specific potential representation: `polynomial` or `pchip` |
| `FIT_DEGREE_BY_MODE` | Mode-specific polynomial degree |
| `FIT_Q_RANGE_ANG_BY_MODE` | Calculated-Q interval included in a fit |
| `FIT_MONOTONIC_OUTWARD_BY_MODE` | Optional outward-growth constraint |
| `FIT_MAX_CUBIC_FACTOR`, `FIT_MAX_CUBIC_FACTOR_BY_MODE` | Default and mode-specific cubic-term restriction; `None` disables it |
| `TEMP`, `L`, `NP` | Temperature and numerical grid used by `a6` |
| `NSAMPLES` | Total number of geometries in each ensemble |
| `A7_RANDOM_SEED` | Reproducible ensemble-sampling seed |
| `A1_WORKERS`, `A7_WORKERS` | Independent reconstruction processes for a1 and a7 |
| `ANHDIS_PARALLEL_NODE` | Fallback node when the user does not select one for the current submission |
| `ALLOWED_NODE_NUMS` | Cluster nodes available to `a3` |
| `EXECUTION_TIMING_FILE` | Shared TSV history for elapsed times of stages `a1`--`a7` |

Example: use the default scan for every mode except mode 1:

```python
Q_ENERGY_ANG = make_q_grid(-0.4, 0.4, 0.1)

Q_ENERGY_ANG_BY_MODE = {
    1: make_q_grid(-1.8, 2.4, 0.1),
}
```

A mode absent from `Q_ENERGY_ANG_BY_MODE` uses `Q_ENERGY_ANG`. The fitting
interval can be narrower than the calculated interval:

```python
FIT_Q_RANGE_ANG_BY_MODE = {
    1: (-1.0, 2.4),
}
```

This changes only the points used by `a5`; it does not delete geometries or
rerun the electronic-structure program.

### Automatically generated Gaussian template

With `AUTO_GENERATE_SINGLE_POINT_HEADER = True`, `a2` reads the route,
memory, charge, and multiplicity from the Gaussian optimization/frequency
log, removes `Opt/Freq`, adds `SP`, and writes `gaussian_head.com`.

Whether generated or supplied manually, `gaussian_head.com` must contain one
`{checkpoint}` placeholder and one
`{job_tag}` placeholder in its title section. It must end with the charge and
multiplicity and must not contain the molecular geometry:

```text
%chk={checkpoint}.chk
%nprocshared={nprocs}
%mem=2GB
#sp MP2/aug-cc-pVDZ

AnhDis curvilinear single-point calculation {job_tag}

0 1
```

With the recommended `ENERGY_SOURCE = "AUTO"`, `a2` identifies the method in
the generated Gaussian route and records the matching energy reader. Thus an
MP2 calculation uses its MP2 total energy rather than the preceding SCF
reference. An explicit `ENERGY_SOURCE` still overrides this inference.
Automatic generation deliberately refuses `Gen/GenECP` and
unsafe geometry/unit controls because their auxiliary sections cannot be
reconstructed reliably. In such a case set
`AUTO_GENERATE_SINGLE_POINT_HEADER = False` and provide a manual template.

### ORCA template

`orca_head.inp` is the ORCA equivalent. It contains the electronic-structure
method but no atom rows. It must contain `{job_tag}`, contain exactly one
`{nprocs}` in a `%pal` block, and end with charge and multiplicity:

```text
! B3LYP 6-311+G(3df) SP TightSCF

%pal
  nprocs {nprocs}
end

%maxcore 10000

# {job_tag}
* xyz 0 2
```

With automatic generation enabled, the interface and `a2` create this file
from the input echoed in the ORCA optimization/frequency output, remove
geometry/frequency tasks and their control blocks, add `SP`, replace `%pal`,
and preserve electronic-method blocks such as `%scf`, `%basis`, `%cpcm`,
`%mdci`, or `%casscf`. The charge and multiplicity are copied from the echoed
coordinate directive or the ORCA output summary. Inspect the resulting file
before submitting calculations. External `%moinp` and `%pointcharges`
dependencies are rejected automatically because they cannot be made portable
without the referenced files.

Automatic generation is valid only when a1 and a2 use the same program and
method. For a deliberately mixed workflow or a higher-level single-point
method, set `AUTO_GENERATE_SINGLE_POINT_HEADER = False` and use the appropriate
manual template; the code never attempts to translate method syntax between
Gaussian and ORCA.

For ORCA HF/DFT, automatic selection reads `FINAL SINGLE POINT ENERGY`; MP2,
CCSD, and CCSD(T) methods select their corresponding total-energy markers.
The selection is made once from `orca_head.inp` and stored in the a2 manifest,
so all scan points necessarily use the same energy definition.

## 5. Complete execution

Run the workflow from the project directory:

```bash
./a1_geometries.py
./a2_gaussian_input_generator.py
./a3_lanzador.py
./a3_lanzador.py --submit
./a4_check_calculation.py --strict
./a5_fitting.py
./a6_probdis_therm.py
./a7_coord_therm.py --output ensemble_output_seed1
```

The first `a3` command is intentionally a dry run. It generates the PBS files
for inspection without submitting them. Run `a5` only after all required
points pass `a4 --strict`.

## 6. Stage details

### 6.1. `a1`: Cartesian/curvilinear paths, masses, and `s(Q)`

Normal execution:

```bash
./a1_geometries.py
```

Alternative log/output or selected modes:

```bash
./a1_geometries.py --log optimization_frequency.log --output curvilinear_scans
./a1_geometries.py --software orca --log optimization_frequency.out
./a1_geometries.py --modes 1 2 3
```

The same selection can be stored in `config.py`:

```python
OPT_FREQ_SOFTWARE = "auto"  # or "gaussian" / "orca"
OPT_FREQ_OUTPUT = PROJECT_ROOT / "optimization_frequency.out"
GAUSSIAN_LOG = OPT_FREQ_OUTPUT  # compatibility alias
```

With `auto`, stable program and termination markers in the output determine
the parser. An explicit value is safer when filenames or archived outputs are
ambiguous; a mismatch is rejected rather than parsed as the wrong format.

The default remains the validated curvilinear model. It can be selected either
in `config.py` or explicitly:

```python
A1_COORDINATE_SYSTEM = "curvilinear"  # or "cartesian"
```

```bash
./a1_geometries.py --coordinate-system curvilinear --output curvilinear_scans
./a1_geometries.py --coordinate-system cartesian --output cartesian_scans
```

Use separate scan directories. A Cartesian scan uses
`R_i(Q)=R_0+Q*d_i`, where `d_i` is the printed Gaussian or ORCA normal-mode
vector.
It is constructed analytically, has constant `mu(Q)=mu(0)` and `s=Q`, and
therefore avoids the iterative back-transformations that dominate the cost of
large curvilinear a1 calculations. Stages a2--a7 read the selected model from
`scan_manifest.json`; no additional switch is needed downstream.

`a1` reads the optimized structure, masses, frequencies, and normal modes;
constructs the primitive internals and Wilson matrix; obtains an independent
internal basis by SVD; validates the local back-transformation; follows each
branch from equilibrium; recalculates the Wilson matrix, `mu(Q)`, and Jacobian
rank; integrates `s(Q)`; and stores the geometries required for the subsequent
single-point calculation.

A new run does not overwrite a nonempty `SCAN_ROOT`.

To extend an existing scan after modifying `Q_ENERGY_ANG_BY_MODE`:

```bash
./a1_geometries.py --modes 1 --update
```

`--update` only extends an existing path. It verifies the source calculation
and preserves all matching existing XYZ files.

The interface provides the safe update sequence without requiring the user to
remember `--update` or `--modes`:

```bash
./anhdis --configure-modes
./anhdis --run a1-pbs          # automatically becomes a1-update-pbs if needed
./anhdis --run a2              # automatically becomes a2-update if needed
./anhdis --run a3-dry
./anhdis --run submit
```

The PBS action creates its script at execution time using the current worker
count and selected node. The job inherits the submitted environment with
`#PBS -V` and starts a login shell, which checks and then uses the compute
node's `python3`:

```bash
/bin/bash -lc 'python3 --version && python3 -u ./a7_coord_therm.py --workers 12'
```

If `ANHDIS_PARALLEL_NODE = "auto"`, PBS selects a suitable node; otherwise the
configured `qcexnodNN` is requested explicitly. For an update, the interface
compares each configured Q grid with `scan_manifest.json` and passes only the
modes that were actually extended. It rejects a grid that would remove old
points, because shrinking a scan requires a new `SCAN_ROOT`.

Main outputs:

- `curvilinear_scans/reference_data.npz`;
- `curvilinear_scans/internal_coordinates.json`;
- `curvilinear_scans/scan_manifest.json`;
- `curvilinear_scans/equilibrium.xyz`;
- `curvilinear_scans/vibN/path_metric.dat`;
- `curvilinear_scans/vibN/path_geometries.npz`;
- `curvilinear_scans/vibN/Q/*.xyz`.

### 6.1.1. Decide whether Cartesian paths are sufficient

The standalone validator can screen existing curvilinear geometries against
the analytic Cartesian paths without any new electronic-structure calculation:

```bash
./validate_coordinate_models.py \
  --curvilinear-root curvilinear_scans \
  --output coordinate_model_validation
```

If Cartesian and curvilinear Gaussian scans already exist, provide both roots
to compare their relative potentials as well:

```bash
./validate_coordinate_models.py \
  --curvilinear-root curvilinear_scans \
  --cartesian-root cartesian_scans \
  --output coordinate_model_validation \
  --force
```

The Cartesian root may also be an **original AnhDis calculation directory**
containing the already generated two-column files `vibN/energies.dat`
(`Q` in angstrom and total energy in hartree). In that case no Cartesian a1
tree and no repeated electronic-structure calculation are required:

```bash
./validate_coordinate_models.py \
  --curvilinear-root curvilinear_scans \
  --cartesian-root /path/to/original_cartesian_anhdis \
  --output coordinate_model_validation \
  --force
```

For a new-format Cartesian tree, the two roots must come from the same
optimization/frequency log and share Q points; completed outputs are validated
through the a2 job identities. For a legacy `energies.dat` tree, the program
checks the table format and common Q values, but the user must confirm that the
Cartesian and curvilinear energies use the same electronic method. The
comparison removes overall translation and rotation by a proper mass-weighted
alignment and reports, for every mode:

- aligned mass-weighted geometry RMSD and maximum atomic difference;
- relative path difference within the configured thermal window;
- the Cartesian-minus-curvilinear potential difference at common Q points;
- thermal-weighted energy RMSE and maximum absolute energy difference;
- a screening recommendation under the explicit tolerances in `config.py`.

The geometry-only result is a cheap predictor, not proof of energetic
equivalence. A recommendation ending in `supported_by_energy` is the stronger
criterion because it uses the already calculated potentials. The complete
pointwise data, summary, explanatory report, and plots are written to
`coordinate_model_points.tsv`, `coordinate_model_summary.tsv`,
`coordinate_model_report.txt`, and `coordinate_model_comparison.pdf`.

### 6.2. `a2`: Gaussian or ORCA inputs

```bash
./a2_gaussian_input_generator.py
./a2_gaussian_input_generator.py --modes 1 2
```

After extending a path:

```bash
./a2_gaussian_input_generator.py --modes 1 --update
```

`--update` adds only the new points to the manifest and leaves identical
existing `.com` or `.inp` files unchanged. Normally the program is read from
`SINGLE_POINT_SOFTWARE`; it can also be selected explicitly:

```bash
./a2_gaussian_input_generator.py --software orca --header orca_head.inp
```

The `--header` option is normally unnecessary: with automatic generation,
`a2` recreates the correct active template directly from the a1 source output.

`--force` is intended only for changing inputs before their calculations have
been run:

```bash
./a2_gaussian_input_generator.py --force
```

Do not combine `--force` and `--update`. The program refuses to force an
input that already has associated result files (`.log/.out/.chk` for Gaussian
or `.log/.out/.gbw/.property.txt` for ORCA).

Each input receives a `job_tag` based on its mode, Q value, and XYZ hash.
Gaussian may wrap its title across lines, while ORCA echoes the identifying
comment in its input section. `a4/a5` remove whitespace before checking the
tag, so line wrapping does not alter the recorded identity.

### 6.3. `a3`: PBS generation, submission, and resubmission

#### Adapting the launcher to your cluster

The supplied launcher contains settings specific to the author's cluster.
Before the first submission, review the following:

| Location | Settings to adapt |
|---|---|
| `config.py` | `PBS_USER` (`None` uses the current user), `PBS_PPN` (cores per single-point calculation), `SCRATCH_ROOT`, and `SUBMIT_COMMAND` (the `qsub` executable or its absolute path). |
| `config.py` | `ALLOWED_NODES` and `NODE_CPUS` must contain matching node names. `ALLOWED_NODE_NUMS` generates names using the supplied `qcexnodNN` convention; change that construction for a different naming scheme. The interactive configuration in `anhdis` also uses this convention. |
| `config.py`, Gaussian | Set `GAUSSIAN_PROFILE` and `GAUSSIAN_COMMAND` for your installation. If modules are required, adapt the Gaussian branch of `pbs_script()` in `a3_lanzador.py`. |
| `a3_lanzador.py`, ORCA branch | Adapt `MODULEPATH`, the `module()` initialization and module commands, `ORCA_DIR`, `MPI_DIR`, `PATH`, `LD_LIBRARY_PATH`, and `OMPI_MCA_plm_rsh_args`. The supplied setup loads `orca/6.1.0` and uses OpenMPI 4.1.8 under `/soft`; these paths and settings are site-specific. |
| `a3_lanzador.py`, both PBS headers | Adapt resource syntax, queue/account, memory, walltime, and notification directives to local requirements. |

Changing `ORCA_PROFILE` or `ORCA_COMMAND` alone does not replace the ORCA
environment block embedded in a3. `ORCA_PROFILE`, when set, is sourced after
that block. Review the supplied `module pure` command against your site's
module system, and use MPI libraries compatible with your ORCA installation.

`PBS_PPN` is shared by Gaussian and ORCA. If it changes, regenerate the a2
inputs so that `%nprocshared` or `%pal nprocs` agrees with the PBS request.
a3 chooses among matching configured nodes, weighted by `NODE_CPUS`; it does
not check current load or reject nodes with fewer cores than `PBS_PPN`.
Ensure every selectable node can satisfy the request.

For ORCA, the parent directory `SCRATCH_ROOT/user` must already exist and be
writable on the compute node. The working directory used for returning
results is the absolute directory of each scan point; do not blindly replace
it with `$PBS_O_WORKDIR`, which may be the directory from which the whole
scan was submitted. Add staging commands if your inputs require external
files other than the `.inp` and `.xyz` files copied by this launcher.

This launcher targets PBS/Torque. Switching to Slurm requires adapting the
headers and job environment variables as well as the submission command;
changing `qsub` to `sbatch` alone is insufficient. For PBS execution of a1/a7,
the separate template is `write_parallel_pbs()` in `anhdis`: review its
resource requests and Python environment too.

Make persistent changes in the configuration or template: a3 regenerates
the per-point `.pbs` files on each invocation. Use the dry run below to
inspect the resulting scripts before submitting.

#### Preparing and submitting jobs

Generate or regenerate PBS files without submitting:

```bash
./a3_lanzador.py
./a3_lanzador.py --modes 1 2
```

Submit new calculations:

```bash
./a3_lanzador.py --submit
./a3_lanzador.py --modes 1 2 --submit
```

Without `--resubmit`, any point that already has a log is skipped. Newly
created points without a log are submitted.

Resubmit every point of selected modes:

```bash
./a3_lanzador.py --modes 1 2 --submit --resubmit
```

This resubmits successful calculations as well as failed or incomplete ones.
To resubmit one failed calculation without rerunning a complete mode:

```bash
./a3_lanzador.py --modes 1
qsub curvilinear_scans/vib1/0.60/run_vib1_0.60.pbs
```

Use the exact path reported in `calculation_status.tsv`.

Node selection can be made reproducible:

```bash
./a3_lanzador.py --submit --seed 20260907
```

Monitor jobs with:

```bash
qstat -u "$USER"
```

Gaussian PBS scripts export the Gaussian scratch directory. Do not add
`set -e`, `set -u`, `set -eo pipefail`, or `set -euo pipefail` before
loading `g16.profile`: that profile and its startup environment are not
compatible with those shell options on the target cluster.

ORCA PBS scripts copy each `.inp` and available `.xyz` files to a job-specific
local scratch directory, run the ORCA driver there, and copy all non-hidden
scratch files back. Scratch is removed only after ORCA returns exit code zero
and the copy succeeds; otherwise it is retained for inspection. a4 still
checks normal termination in the output. The updated launcher resolves
`ORCA_COMMAND = "orca"` to the full path in `ORCA_BIN`; an explicit custom
command remains supported. Do not put `mpirun` in `ORCA_COMMAND`:
ORCA starts its parallel modules itself from `%pal nprocs`. This follows the
official [ORCA execution syntax](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/running.html)
and [parallel-run guidance](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/parallel.html).

### 6.4. `a4`: calculation checks

```bash
./a4_check_calculation.py
./a4_check_calculation.py --strict
./a4_check_calculation.py --modes 1 2 --strict
```

`--strict` performs the same checks but exits with a nonzero status if any
point is not `OK`.

| Status | Meaning |
|---|---|
| `OK` | Termination, energy, input, and identity are valid |
| `NO_INPUT` | Point is absent from the `a2` manifest |
| `MISSING` | No log exists |
| `INCOMPLETE` | Normal termination is absent |
| `ERROR` | Gaussian or ORCA terminated with an error |
| `NO_ENERGY` | The selected `ENERGY_SOURCE` was not found |
| `INPUT_MISMATCH` | Input or XYZ no longer matches the manifest |
| `OUTPUT_MISMATCH` | Log does not contain the expected `job_tag` |

The report is written to
`curvilinear_scans/calculation_status.tsv`. Show only failed points with:

```bash
awk -F '\t' 'NR == 1 || $4 != "OK"' curvilinear_scans/calculation_status.tsv
```

### 6.5. `a5`: potential representation

```bash
./a5_fitting.py
```

By default, the potential is represented by derivatives at `s=0`:

```text
V(s) = sum_n k_n s^n/n!,    k_n = d^nV/ds^n at s=0.
```

For a quartic fit:

```text
V(s) = (1/2) k s^2 + (1/6) kc s^3 + (1/24) kq s^4.
```

Degrees 2 through 6 can be selected per mode. The harmonic reference is built
from the parsed harmonic frequency and `mu(0)`. Because `s=0` is the optimized
geometry and is constrained to be stationary, no additional potential shift is
used.

For polynomial modes, the cubic coefficient can be bounded globally or by
mode:

```python
FIT_MAX_CUBIC_FACTOR = 10.0
FIT_MAX_CUBIC_FACTOR_BY_MODE = {
    1: 5.0,    # tighter cubic bound for mode 1
    2: None,   # no cubic bound for mode 2
}
```

The convention is `|a3| <= factor*a2_guess`, where the fitted potential uses
`V(s)=a2*s^2+a3*s^3+...` and `kc=6*a3`. The factor must be positive and
finite; `None` disables the bound for that mode. This is available from
**Configure limits and fits by mode → Set or disable the cubic-term
restriction by mode**. It is ignored when that mode uses PCHIP.

An explicit `None` and a missing dictionary entry are different. For example,
`32: None` disables the cubic bound for mode 32, whereas removing mode 32 from
the dictionary restores the global `FIT_MAX_CUBIC_FACTOR`. The interactive
menu therefore calls removal **Restore global fit defaults**. The final `a5`
line for every mode prints the effective cubic setting, whether it came from
the global default or a mode override, and whether the bound was reached.

After every interactive per-mode edit, the launcher removes any stale
`config.py` bytecode cache, reloads the saved dictionaries, and prints a
verification message. Consequently, an immediate `a5` run sees a changed
polynomial degree even when the old and new values have the same text length.

For broad or asymmetric modes that a single global polynomial cannot describe
reliably, select shape-preserving piecewise cubic interpolation (PCHIP):

```python
FIT_MODEL = "polynomial"
FIT_MODEL_BY_MODE = {
    1: "pchip",
    # 2: "pchip",
}
```

A mode absent from `FIT_MODEL_BY_MODE` retains the default polynomial model.
PCHIP passes exactly through every selected *ab initio* point in the interval
defined by `FIT_Q_RANGE_ANG_BY_MODE`; therefore polynomial degree, polynomial
weights, cubic bounds, and polynomial monotonic constraints do not apply to a
PCHIP mode.

PCHIP itself is used only between the first and last selected points. The `a6`
grid is normally wider, so `a5` attaches a quadratic confining tail to each
endpoint. The outward endpoint slope is retained. If a fitted endpoint points
back into the scan, its tail slope is clipped to zero and reported as a
warning. Tail curvature is defined in relation to the Gaussian harmonic force
constant:

```python
PCHIP_TAIL_CURVATURE_FACTOR = 1.0
PCHIP_TAIL_CURVATURE_FACTOR_BY_MODE = {
    # 1: 2.0,
}
```

Thus `1.0` means `k_tail = k_harm`. Increase it only when `a6` reports
non-negligible density at a numerical-grid edge. This option changes only the
uncomputed region outside the scan; it does not alter the PCHIP values between
calculated points.

For PCHIP, `rmse_cm` and `max_error_cm` are zero up to numerical precision by
construction and are not quality criteria. Instead, inspect the potential PDF,
the physical smoothness of the electronic energies, endpoint-slope warnings,
`P_outside_scan`, and the `a6` edge-density diagnostics.

For diagnostic fits with missing calculations:

```bash
./a5_fitting.py --allow-partial
```

This option is not recommended for a final ensemble.

Main outputs:

- `fit_coefficients.dat`;
- `fit_summary.tsv`;
- `curvilinear_potential_fits.pdf`;
- `full_potentials.pdf`;
- `curvilinear_path_metrics.pdf`;
- `vibN/energies_qs.dat`;
- `vibN/fit_curve.dat`;
- `vibN/potential_for_quantum_step.npz`.

Before `a6`, inspect both potential PDFs and check `rmse_cm`,
`max_error_cm`, the curvature ratio, fitted minimum, outward behavior, and
`warnings` in `fit_summary.tsv`.

### 6.6. `a6`: Hamiltonians and thermal distributions

```bash
./a6_probdis_therm.py
./a6_probdis_therm.py --temperature 300.0
./a6_probdis_therm.py --modes 1 2
```

For each mode, `a6` constructs and diagonalizes the coordinate-space
Hamiltonian once:

```text
H = -(1/(2*mu(0))) d2/ds2 + V(s).
```

Here `V(s)` is either the selected derivative polynomial or PCHIP with its
confining tails. `a5` stores the complete representation in
`vibN/potential_for_quantum_step.npz`, so no manual conversion is required.

It retains the eigenstates required by `THERMAL_WEIGHT_CUTOFF`, applies their
Boltzmann weights, and constructs

```text
rho(s,T) = sum_v w_v |psi_v(s)|^2.
```

The momentum-space wavefunction of each retained state is then obtained by a
Fourier transform of the same coordinate-space eigenfunction:

```text
phi_v(p_s) = (2*pi)^(-1/2) integral psi_v(s) exp(-i*p_s*s) ds
rho(p_s,T) = sum_v w_v |phi_v(p_s)|^2.
```

There is no second momentum-space Hamiltonian diagonalization. A separate
harmonic reference Hamiltonian is solved only to provide the harmonic
comparison distributions. FFT padding refines the momentum grid but does not
modify the Hamiltonian or its eigenfunctions.

`rho(s,T)` is normalized over `ds` in bohr, and `rho(p_s,T)` is normalized
over `dp_s` in atomic momentum units.

Strict diagnostic mode:

```bash
./a6_probdis_therm.py --strict
```

It computes the same quantities but exits with an error if a domain or
convergence warning is present.

Main outputs:

- `wave_wigner.pdf`;
- `test_excited_states.pdf`;
- `momentum_distributions.pdf`;
- `thermal_summary.dat`;
- `wavefunction_diagnostics.tsv`;
- `probabilities/vibN.dat`;
- `momenta/vibN.dat`;
- `vibN/probability_for_ensemble.npz`;
- `a6_metadata.json`.

After the final fit, run `a6` for all modes. A final run restricted with
`--modes 1 2` leaves metadata for only those modes and cannot be used to build
a complete ensemble.

### 6.7. `a7`: CDF sampling and Cartesian ensemble

Always use a new output directory:

```bash
./a7_coord_therm.py --output ensemble_output_seed1
```

The sample count and seed can also be provided on the command line:

```bash
./a7_coord_therm.py --samples 1000 --seed 7 --output ensemble_1000_seed7
```

Curvilinear reconstruction can be parallelized across independent candidate
geometries:

```bash
./a7_coord_therm.py --workers 12 --output ensemble_1000_seed7
./anhdis --run a7-pbs
./anhdis --run a7-pbs --node qcexnod56 --workers 12
```

The parent process draws all position and momentum random numbers in the same
order as the sequential algorithm. Worker processes only perform the expensive
geometry reconstruction, and results are consumed in candidate order.
Consequently a fixed seed produces exactly the same accepted geometries,
momenta, velocities, diagnostics, and ordering with one or multiple workers.
The generated PBS script also limits BLAS/OpenMP libraries to one thread per
Python process to prevent oversubscription.

`a7` refuses to overwrite a nonempty directory. Preserve an earlier run by
renaming it or choose a different output name:

```bash
mv ensemble_output ensemble_output_previous
./a7_coord_therm.py
```

`NSAMPLES` is the total number of geometries in each ensemble:

- `g1` is always the optimized geometry, with all `Q_i=s_i=0`;
- the momentum and velocity associated with `g1` are sampled from the same
  distributions as the other ensemble members and are not forced to zero;
- `g2` through `gNSAMPLES` are position samples;
- position histograms and position KS tests exclude `g1`, because its
  position is fixed rather than sampled;
- momentum diagnostics include `g1`;
- the ensemble `.npz` marks it with `is_equilibrium_geometry`.

The output contains per-geometry `.xyz`, `.vel`, and `.mom` files;
multiframe geometry, velocity, and momentum files; harmonic-reference files;
position and momentum histograms; and the following diagnostics:

```text
<a7 output directory>/
  ensemble_anh_therm/   anharmonic geometry, velocity, and momentum files
  ensemble_har_therm/   harmonic geometry, velocity, and momentum files
  histograms_anh/       anharmonic sampling plots
  histograms_har/       harmonic sampling plots
  all_geometries_anh.xyz
  all_geometries_har.xyz
  all_velocities_anh.xyz
  all_velocities_har.xyz
  all_momenta_anh.xyz
  all_momenta_har.xyz
```

They are selected by this `OUTPUT_LAYOUT` definition in `a7_coord_therm.py`:

```python
OUTPUT_LAYOUT = {
    ANHARMONIC: ("ensemble_anh_therm", "histograms_anh", "all_geometries_anh.xyz"),
    HARMONIC: ("ensemble_har_therm", "histograms_har", "all_geometries_har.xyz"),
}
VECTOR_OUTPUT_NAMES = {
    ANHARMONIC: ("all_velocities_anh.xyz", "all_momenta_anh.xyz"),
    HARMONIC: ("all_velocities_har.xyz", "all_momenta_har.xyz"),
}
```

- `sampled_coordinates_thermal_*.npz`;
- `mode_sampling_diagnostics.tsv`;
- `ensemble_geometry_diagnostics.tsv`;
- `path_reconstruction_validation.tsv`;
- `a7_manifest.json`.

For each mode, `a7` performs two independent inverse-CDF samples. The first
produces `s_i` in bohr; it is converted to angstrom and the monotonic
`s_i(Q_i)` map from `a1` is inverted. The second produces the conjugate
momentum `p_s,i` in atomic units.

All sampled internal-coordinate targets are combined as

```text
xi_target = sum_i Q_i*l_i.
```

A single simultaneous nonlinear back-transformation produces each Cartesian
geometry. The Wilson matrix is recalculated at every iteration; Cartesian
normal-mode displacement vectors are not added.

For the harmonic ensemble, both `s_i` and `p_s,i` are drawn directly from the
exact unbounded thermal Gaussian marginals determined by the harmonic
frequency and `mu_i(0)`. Harmonic sampling therefore does not use or truncate
the a1 path, a5 fitting interval, or a6 numerical grid. The numerical harmonic
Hamiltonian retained in a6 is an independent energy/grid check.

Velocities use the independent-mode approximation:

```text
ds_i/dt = p_s,i/mu_i(0)
dQ_i/ds_i = sqrt(mu_i(0)/mu_i(Q_i)).
```

At each sampled geometry, `a7` recalculates the Wilson matrix and its
mass-metric pseudoinverse to obtain the local `dR/dQ_i` tangents. Modal
contributions are projected to Cartesian velocities. For a nonlinear
reference, instantaneous translation and rotation are removed. For a linear
reference, translation and its two finite rigid rotations are excluded by the
five-dimensional Wilson-matrix nullspace; the angular momentum conjugate to
the azimuth of a degenerate two-component bend is retained because it is
vibrational, not a spurious rigid rotation. The geometry diagnostics record the
projection residual, center-of-mass velocity, angular momentum, minimum
interatomic distance, Jacobian rank, and modal/Cartesian kinetic energies.

`a7` reads:

- `reference_data.npz` and `internal_coordinates.json` from `a1`;
- `vibN/path_geometries.npz` from `a1`;
- `fit_coefficients.dat` and `vibN/energies_qs.dat` from `a5`;
- `probabilities/vibN.dat`, `momenta/vibN.dat`, and optional
  `a6_metadata.json` from `a6`.

It does not require `potential_for_quantum_step.npz` or
`probability_for_ensemble.npz` to reconstruct the ensemble.
For either polynomial or PCHIP potentials, it independently verifies that the
potential columns saved by `a6` agree with the current `a5` text outputs before
sampling.

Generate only one ensemble type with:

```bash
./a7_coord_therm.py --ensemble thermal_anharmonic --output ensemble_anh
./a7_coord_therm.py --ensemble thermal_harmonic --output ensemble_har
```

`--allow-truncation` renormalizes a distribution if its probability outside
the validated path exceeds the configured limits. Use it only as an explicit
approximation, not to hide an insufficient scan.

`a7` never overwrites a nonempty ensemble directory. When `a7` is selected
from `./anhdis`, the launcher detects this condition before direct execution or
PBS submission and prompts for a new output directory. The selected path is a
one-run override and does not modify `ENSEMBLE_ROOT` in `config.py`. Reusing the
same random seed reproduces the same random draws; change the seed as well when
an independent statistical realization is required.

## 7. Extending one mode

Example for extending only mode 1:

1. Choose **Configure limits and fits by mode** in `./anhdis` (or run
   `./anhdis --configure-modes`) and change mode 1.
2. In the interface, select normal `a1` or `a1-pbs`. It detects the existing
   manifest and automatically executes the equivalent of:

   ```bash
   ./a1_geometries.py --modes 1 --update
   ```

3. Select normal `a2` in the interface. It automatically generates only the
   new Gaussian or ORCA inputs, equivalently:

   ```bash
   ./a2_gaussian_input_generator.py --modes 1 --update
   ```

4. Regenerate and inspect its PBS files:

   ```bash
   ./a3_lanzador.py --modes 1
   ```

5. Submit only points without existing logs:

   ```bash
   ./a3_lanzador.py --modes 1 --submit
   ```

6. Validate the extended mode:

   ```bash
   ./a4_check_calculation.py --modes 1 --strict
   ```

7. Recompute the final fit and distributions for all modes, then create a new
   ensemble directory:

   ```bash
   ./a5_fitting.py
   ./a6_probdis_therm.py
   ./a7_coord_therm.py --output ensemble_after_extension
   ```

Existing calculated points are preserved and are not resubmitted by this
procedure.

At any time, inspect the chain before continuing:

```bash
./anhdis --check
```

## 8. Which stages must be repeated after a change?

| Change | Repeat |
|---|---|
| New optimization/frequency output or source program | `a1`--`a7` in a new `SCAN_ROOT` |
| Extended `Q_ENERGY_ANG_BY_MODE` | `a1 --update`, `a2 --update`, `a3`, `a4`, `a5`, `a6`, `a7` |
| New Gaussian/ORCA method, program, or template | `a2`--`a7` with new, consistent electronic results; use a new `SCAN_ROOT` when changing program |
| New fit degree, interval, or constraint | `a5`, `a6`, `a7` |
| New `TEMP`, `L`, `NP`, or thermal cutoff | `a6`, `a7` |
| New `NSAMPLES`, seed, or ensemble name | `a7` only |
| New cluster nodes, scratch, Gaussian, or ORCA installation | `a3` to regenerate PBS files |

## 9. Checks before using an ensemble

1. `a4 --strict` must report every required point as `OK`.
2. Inspect `curvilinear_potential_fits.pdf` and `full_potentials.pdf`.
3. Inspect fitting errors, curvature, minima, outward behavior, and warnings in
   `fit_summary.tsv`.
4. Inspect `wavefunction_diagnostics.tsv`: position and momentum densities
   should vanish at their numerical boundaries, and the probability outside
   the calculated path should be acceptably small.
5. Inspect `mode_sampling_diagnostics.tsv`: omitted tails and position/momentum
   KS statistics.
6. Inspect `ensemble_geometry_diagnostics.tsv`: internal residuals, Jacobian
   rank, minimum distances, connectivity changes, rejections, center-of-mass
   velocity, and residual angular momentum.
7. Confirm that `g1` matches `curvilinear_scans/equilibrium.xyz`.

## 10. Common problems

### `Use a new or empty a7 output directory`

The requested directory already contains an ensemble. Choose a new name:

```bash
./a7_coord_therm.py --output ensemble_output_new
```

### `OUTPUT_MISMATCH`

The log does not contain the `job_tag` recorded by `a2`, or the log belongs
to a different input. Gaussian title wrapping and the ORCA echoed-input form
are handled. Do not replace a `.com` or `.inp` file without updating the
manifest and recalculating its log.

### Errors while loading `g16.profile`

Check that the generated PBS file contains no `set -e`, `set -u`, or
`pipefail` variant, and that it exports the Gaussian scratch directory before
loading the profile.

### Unphysical fitted potential outside the scan

Inspect the ab initio points first. Then extend the scan, change
`FIT_Q_RANGE_ANG_BY_MODE`, choose a different degree, or apply a
physically appropriate constraint. Do not force a periodic or multiwell
torsional potential to be outward-monotonic.

### Non-negligible `a6` density at a numerical boundary

Inspect `full_potentials.pdf` and increase `L/NP` while preserving grid
resolution. If extrapolation outside the ab initio points causes the problem,
correct the scan or potential representation first.

### `a7` rejects excessive probability outside the path

Extend the `a1` path and, for the anharmonic ensemble, the ab initio interval
used by `a5`. Use `--allow-truncation` only when that approximation is
intentional.

## 11. Tests

Run the local test suite with:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

The data produced by `tests/create_synthetic_outputs.py` are synthetic
software-test data and must not be used as molecular results.
