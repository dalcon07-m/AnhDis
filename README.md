```text
  /$$$$$$            /$$       /$$$$$$$  /$$          
 /$$__  $$          | $$      | $$__  $$|__/          
| $$  \ $$ /$$$$$$$ | $$$$$$$ | $$  \ $$ /$$  /$$$$$$$
| $$$$$$$$| $$__  $$| $$__  $$| $$  | $$| $$ /$$_____/
| $$__  $$| $$  \ $$| $$  \ $$| $$  | $$| $$|  $$$$$$ 
| $$  | $$| $$  | $$| $$  | $$| $$  | $$| $$ \____  $$
| $$  | $$| $$  | $$| $$  | $$| $$$$$$$/| $$ /$$$$$$$/
|__/  |__/|__/  |__/|__/  |__/|_______/ |__/|_______/ 
```                                                      
___________________________________________________________
___________________________________________________________                                                      
ANHDIS - Anharmonic Vibrational Analysis Workflow
___________________________________________________________

OVERVIEW
--------
Anhdis is a Python-based workflow designed to extract vibrational information from Gaussian calculations,
build anharmonic potential energy surfaces, solve 1D vibrational Schrödinger equations, and generate
quantum nuclear sampling geometries at finite temperature.

The workflow connects electronic structure calculations with vibrational quantum analysis beyond the
harmonic approximation.

------------------------------------------------------------
WORKFLOW SUMMARY
------------------------------------------------------------

1. Extract optimized geometry and vibrational modes from Gaussian output
2. Generate displaced geometries along normal modes
3. Create Gaussian input files for single-point calculations
4. Submit calculations to a cluster (PBS system)
5. Check calculation status and completeness
6. Extract energies and fit anharmonic potentials
7. Solve vibrational Schrödinger equation (1D per mode)
8. Construct thermal vibrational probability distributions using
   Boltzmann populations of the vibrational eigenstates 
9. Generate thermally sampled nuclear geometries

------------------------------------------------------------
PROJECT FILES
------------------------------------------------------------

a1_geometries.py
- Extracts optimized geometry from Gaussian log
- Reads vibrational modes, frequencies, reduced masses
- Renormalizes vibrational vectors
- Generates displaced geometries along normal modes

a2_gaussian_input_generator.py
- Converts .xyz geometries into Gaussian .com input files
- Uses gaussian_head.com template
- Automatically scans directories

a3_lanzador.py
- Submits Gaussian jobs to PBS cluster
- Distributes jobs across nodes weighted by CPU availability
- Creates and submits run scripts

a4_check_calculation.py
- Checks Gaussian job status
- Classifies outputs:
  [OK] Normal termination
  [ERROR] Gaussian error
  [INCOMPLETE] missing or unfinished job

a5_fitting.py
- Extracts energies from Gaussian outputs
- Fits anharmonic potential energy surfaces:

  V(x) = a0 + a2 x^2 + a3 x^3 + a4 x^4

- Computes harmonic and anharmonic force constants
- Generates plots and fit coefficients

a6_probdis_therm.py
- Builds vibrational Hamiltonians:

  H = -(1/2μ)d²/dx² + V(x)

- Solves for ground-state wavefunctions
- Constructs the thermal probability distribution,

      P(x,T) = Σn wn(T)|ψn(x)|²,

  using Boltzmann populations and including excited states with
  wn(T) > 10^-3.
- Exports the thermal probability distributions.

a7_coord.py
- Samples nuclear geometries using probability distributions
- Builds cumulative distributions for each mode
- Performs Monte Carlo sampling
- Outputs XYZ geometries and histograms
- Performs KS statistical tests

------------------------------------------------------------
CONFIGURATION FILE (config.py)
------------------------------------------------------------

Main parameters:

Paths:
- Gaussian log file  (without .log)
- Gaussian input template
- periodic table data:
    ___________________________________________
     PERIODIC TABLE FILE FORMAT
    
    The file defined in `PERIODIC_TABLE` is a plain text file used to
    map atomic numbers to element symbols and atomic masses.
    
    Each line must follow this format:
    
        Z   Symbol   Name   Mass
    _______________________________________________
    

Vibrational settings:
- SELECTED_MODES: choose subset of modes
- SCALING_FACTORS: displacement amplitudes

    _____________________________________________
    - If `SELECTED_MODES = [ ]` (empty list):
      → all vibrational modes are processed
    
    - If specific modes are listed:
      → only those modes are processed throughout the workflow
    
    Example:
    
        SELECTED_MODES = ["vib1", "vib5", "vib10"]
    _____________________________________________
    
Numerical parameters:
- L: grid size (Bohr)
- NP: number of grid points
- NSAMPLES: number of sampled geometries
- TEMP: temperature in K
Unit conversions:
- Bohr ↔ Angstrom
- amu ↔ electron mass
- cm⁻¹ ↔ Hartree

Cluster configuration:
- NODE_CPUS: CPU per node
- ALLOWED_NODES: allowed PBS nodes
     ___________________________________
     IMPORTANT RULE:
     
     - Node identifiers are written using ONLY the numeric part
     - Do NOT include leading zeros
     
     Correct format:
         1, 2, 10, 11, 52
     
     Incorrect format:
         01, 02, 10, 11, 052
     
     Example:
     
         ALLOWED_NODE_NUMS = [1, 2, 51, 52]
     ________________________________
     


------------------------------------------------------------
OUTPUT FILES
------------------------------------------------------------

geometry_gaussian_ang.dat
geometry_gaussian_bohr.dat
    Optimized geometry from Gaussian

vibrational_data.txt
    Vibrational modes, frequencies, reduced masses

fit_coefficients.dat
    Anharmonic fit parameters per mode

energies.dat
    Energy scan per displacement, each one inside the folder vib$number

vib_plots.pdf
    Fitted potentials and harmonic comparison

full_potentials.pdf
    Full potential energy curves

wave_wigner.pdf
    Wavefunctions for harmonic and anharmonic systems

thermal_summary.dat
    Summary of the thermal treatment for each vibrational mode,
    including the number of excited states included in the Boltzmann
    expansion.

test_excited_states.pdf
    Diagnostic plots showing the vibrational wavefunctions of all
    excited states included in the thermal expansion for each mode.

probabilities/vib*.dat
    Probability distributions |ψ(x)|²

NX/
NX_har/
    Wigner-sampled geometries (anharmonic / harmonic)

histograms/
    Distribution validation plots + KS tests

all_geometries.xyz
    Full ensemble of sampled geometries



------------------------------------------------------------
MAIN EXECUTION ORDER
------------------------------------------------------------

python a1_geometries.py
python a2_gaussian_input_generator.py
python a3_lanzador.py
python a4_check_calculation.py
python a5_fitting.py
python a6_probdis_therm.py
python a7_coord_therm.py

------------------------------------------------------------
NOTES
------------------------------------------------------------

- First Gaussian optimization log must include vibrational analysis
- Grid size L must be large enough to contain wavefunctions
- Reduced masses may differ slightly due to numerical normalization
- PBS cluster environment required for job submission
- All coordinates are converted between Bohr and Angstrom as needed

------------------------------------------------------------
END OF FILE
------------------------------------------------------------
