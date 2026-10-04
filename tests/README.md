# Curvilinear AnhDis workflow: stages `a1`--`a7`

This package builds one-dimensional curvilinear vibrational potentials and
thermal nuclear ensembles using internal-coordinate trajectories. The workflow
retains the `vibN/Q` directory structure, but the final geometries are obtained
by nonlinear internal-to-Cartesian back-transformation rather than by directly
adding Cartesian displacement vectors.

| Stage | Main purpose |
|---|---|
| `a1` | Curvilinear paths, Wilson matrix, SVD, effective masses `mu(Q)`, and constant-mass coordinate `s(Q)` |
| `a2` | Gaussian single-point input generation |
| `a3` | PBS script generation and calculation submission or resubmission |
| `a4` | Calculation termination, energy, input, and identity checks |
| `a5` | Fitting of the one-dimensional potentials `V(s)` |
| `a6` | One-dimensional Hamiltonians and thermal position/momentum distributions |
| `a7` | Independent position/momentum sampling and reconstruction of Cartesian geometries, velocities, and momenta |

## 1. Requirements and model

Requirements:

- Python 3;
- NumPy, SciPy, and Matplotlib;
- a normally terminated Gaussian optimization and frequency calculation;
- Gaussian and PBS/Torque access for the single-point calculations.

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

The current implementation supports nonlinear molecules described by Gaussian
primitive internal coordinates `R`, `A`, and `D`. It deliberately rejects
unsupported or unsafe cases such as linear molecules, dummy atoms, connectivity
changes, linear bends, and paths that cross a torsional chart discontinuity at
`+/-pi`.

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
| Gaussian energies and fitted potentials | hartree |
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
config.py
gaussian_head.com
internal_coordinates.py
workflow_utils.py
a1_geometries.py
a2_gaussian_input_generator.py
a3_lanzador.py
a4_check_calculation.py
a5_fitting.py
a6_probdis_therm.py
a7_coord_therm.py
optimization_frequency.log
```

If executable permissions are lost when copying the files:

```bash
chmod +x a1_geometries.py a2_gaussian_input_generator.py a3_lanzador.py \
  a4_check_calculation.py a5_fitting.py a6_probdis_therm.py a7_coord_therm.py
```

## 4. Main configuration

The options most frequently changed by the user are placed near the beginning
of `config.py`.

| Option | Meaning |
|---|---|
| `GAUSSIAN_LOG` | Optimization/frequency output read by `a1` |
| `GAUSSIAN_HEADER` | Single-point template used by `a2` |
| `SCAN_ROOT` | Common output directory used by `a1`--`a7` |
| `SELECTED_MODES` | Mode list; an empty list means all `3N-6` modes |
| `Q_ENERGY_ANG` | Default electronic-energy scan grid |
| `Q_ENERGY_ANG_BY_MODE` | Optional mode-specific scan grids |
| `ENERGY_SOURCE` | Energy parsed by `a4/a5`: `SCF`, `MP2`, `CCSD`, `CCSD(T)`, or `AUTO` |
| `FIT_MODEL`, `FIT_MODEL_BY_MODE` | Default and mode-specific potential representation: `polynomial` or `pchip` |
| `FIT_DEGREE_BY_MODE` | Mode-specific polynomial degree |
| `FIT_Q_RANGE_ANG_BY_MODE` | Calculated-Q interval included in a fit |
| `FIT_MONOTONIC_OUTWARD_BY_MODE` | Optional outward-growth constraint |
| `TEMP`, `L`, `NP` | Temperature and numerical grid used by `a6` |
| `NSAMPLES` | Total number of geometries in each ensemble |
| `A7_RANDOM_SEED` | Reproducible ensemble-sampling seed |
| `ALLOWED_NODE_NUMS` | Cluster nodes available to `a3` |

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
rerun Gaussian.

### Gaussian template

`gaussian_head.com` must contain one `{checkpoint}` placeholder and one
`{job_tag}` placeholder in its title section. It must end with the charge and
multiplicity and must not contain the molecular geometry:

```text
%chk={checkpoint}.chk
%nprocshared=2
%mem=2GB
#sp MP2/aug-cc-pVDZ

AnhDis curvilinear single-point calculation {job_tag}

0 1
```

The Gaussian route and `ENERGY_SOURCE` must describe the same electronic
structure method.

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

### 6.1. `a1`: curvilinear paths, masses, and `s(Q)`

Normal execution:

```bash
./a1_geometries.py
```

Alternative log/output or selected modes:

```bash
./a1_geometries.py --log optimization_frequency.log --output curvilinear_scans
./a1_geometries.py --modes 1 2 3
```

`a1` reads the optimized structure, masses, frequencies, and normal modes;
constructs the primitive internals and Wilson matrix; obtains an independent
internal basis by SVD; validates the local back-transformation; follows each
branch from equilibrium; recalculates the Wilson matrix, `mu(Q)`, and Jacobian
rank; integrates `s(Q)`; and stores the geometries required for Gaussian.

A new run does not overwrite a nonempty `SCAN_ROOT`.

To extend an existing scan after modifying `Q_ENERGY_ANG_BY_MODE`:

```bash
./a1_geometries.py --modes 1 --update
```

`--update` only extends an existing path. It verifies the source calculation
and preserves all matching existing XYZ files.

Main outputs:

- `curvilinear_scans/reference_data.npz`;
- `curvilinear_scans/internal_coordinates.json`;
- `curvilinear_scans/scan_manifest.json`;
- `curvilinear_scans/equilibrium.xyz`;
- `curvilinear_scans/vibN/path_metric.dat`;
- `curvilinear_scans/vibN/path_geometries.npz`;
- `curvilinear_scans/vibN/Q/*.xyz`.

### 6.2. `a2`: Gaussian inputs

```bash
./a2_gaussian_input_generator.py
./a2_gaussian_input_generator.py --modes 1 2
```

After extending a path:

```bash
./a2_gaussian_input_generator.py --modes 1 --update
```

`--update` adds only the new points to the manifest and leaves identical
existing `.com` files unchanged.

`--force` is intended only for changing inputs before their calculations have
been run:

```bash
./a2_gaussian_input_generator.py --force
```

Do not combine `--force` and `--update`. The program refuses to force an
input that already has an associated `.log`, `.out`, or `.chk`.

Each input receives a `job_tag` based on its mode, Q value, and XYZ hash.
Gaussian may wrap that title across lines; `a4/a5` remove whitespace before
checking it, so line wrapping does not alter the recorded identity.

### 6.3. `a3`: PBS generation, submission, and resubmission

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

The generated PBS scripts export the Gaussian scratch directory. Do not add
`set -e`, `set -u`, `set -eo pipefail`, or `set -euo pipefail` before
loading `g16.profile`: that profile and its startup environment are not
compatible with those shell options on the target cluster.

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
| `ERROR` | Gaussian terminated with an error |
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
from the Gaussian frequency and `mu(0)`. Because `s=0` is the optimized
geometry and is constrained to be stationary, no additional potential shift is
used.

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

Velocities use the independent-mode approximation:

```text
ds_i/dt = p_s,i/mu_i(0)
dQ_i/ds_i = sqrt(mu_i(0)/mu_i(Q_i)).
```

At each sampled geometry, `a7` recalculates the Wilson matrix and its
mass-metric pseudoinverse to obtain the local `dR/dQ_i` tangents. Modal
contributions are projected to Cartesian velocities, after which instantaneous
translation and rotation are removed. The geometry diagnostics record the
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

## 7. Extending one mode

Example for extending only mode 1:

1. Change mode 1 in `Q_ENERGY_ANG_BY_MODE`.
2. Extend its path:

   ```bash
   ./a1_geometries.py --modes 1 --update
   ```

3. Generate only the new Gaussian inputs:

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

## 8. Which stages must be repeated after a change?

| Change | Repeat |
|---|---|
| New optimization/frequency log | `a1`--`a7` in a new `SCAN_ROOT` |
| Extended `Q_ENERGY_ANG_BY_MODE` | `a1 --update`, `a2 --update`, `a3`, `a4`, `a5`, `a6`, `a7` |
| New Gaussian method or template | `a2`--`a7` with new, consistent electronic results |
| New fit degree, interval, or constraint | `a5`, `a6`, `a7` |
| New `TEMP`, `L`, `NP`, or thermal cutoff | `a6`, `a7` |
| New `NSAMPLES`, seed, or ensemble name | `a7` only |
| New cluster nodes, scratch, or Gaussian installation | `a3` to regenerate PBS files |

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
to a different input. Gaussian title wrapping is already handled. Do not
replace a `.com` file without updating the manifest and recalculating its log.

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
