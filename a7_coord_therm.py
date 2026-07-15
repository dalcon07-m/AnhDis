#!/usr/bin/env python3
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from config import GAUSSIAN_LOG, FIT_COEFF_FILE, NSAMPLES, BOHR_TO_ANG, L, NP, OPT_GEO, VIB_DAT

# --- Read geometry from .dat file ---

symbols = []
coords = []

with open(OPT_GEO) as f:
    for line in f:
        if line.strip() == "" or line.startswith("#"):
            continue
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

coords = np.array(coords)
natoms = len(coords)
nmodes = natoms*3 - 6

#------ Read a1 shift from FIT_COEFF_FILE -------

a1_list = []

with open(FIT_COEFF_FILE) as f:
    next(f)  # skip header
    for line in f:
        parts = line.split()
        a1 = float(parts[1])
        a1_list.append(a1)

a1_list = np.array(a1_list)
a1_list_bohr=a1_list / BOHR_TO_ANG
# --- Read Gaussian normal mode vectors from VIB_DAT ---

displacements = []
with open(VIB_DAT) as f:
    for line in f:
        if line.strip() == "" or line.startswith("#"):
            continue
        parts = line.split()
        cart_start = 4
        cart_end = cart_start + 3*natoms
        gauss_start = cart_end
        gauss_end = gauss_start + 3*natoms
        gauss_values = [float(x) for x in parts[gauss_start:gauss_end]]
        gauss_vec = np.array(gauss_values).reshape((natoms, 3))
        displacements.append(gauss_vec)

displacements = np.array(displacements)  # (nmodes x natoms x 3)



# -----------  Import distribution probabilites -------------------

xgrid_list, prob_list, prob_h_list, prob_therm_list  = [], [], [], []

for i in range(1, nmodes+1):
    filename = f"probabilities/vib{i}.dat"
    data = np.loadtxt(filename, comments="#")
    xgrid = data[:,0]
    prob_anh = data[:,1]
    prob_harm = data[:,2]
    prob_therm = data[:,3]


    prob_anh /= np.trapezoid(prob_anh, xgrid)   # normalizamos
    prob_harm /= np.trapezoid(prob_harm, xgrid) # normalizamos
    prob_therm /= np.trapezoid(prob_therm, xgrid)   # normalizamos


    xgrid_list.append(xgrid)
    prob_list.append(prob_anh)
    prob_h_list.append(prob_harm)
    prob_therm_list.append(prob_therm)


# -----------------------------
# --- Function to generate samples and save ---
# -----------------------------
def generate_samples(xgrid_list, prob_therm_list, out_xyz_dir="NX", hist_dir="histograms", all_xyz_file="all_geometries.xyz", apply_shift=False):
    """
    Generate Wigner-sampled geometries based on input probability distributions.
    Saves XYZ geometries and histograms of mode displacements.

    Parameters:
    -----------
    xgrid_list : list of np.array
        List of grids for each mode.
    prob_list : list of np.array
        List of probability distributions (anharmonic or harmonic) for each mode.
    out_xyz_dir : str
        Directory to save XYZ geometries.
    hist_dir : str
        Directory to save histograms.
    """
    os.makedirs(out_xyz_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    rng = np.random.default_rng(1)  # Random number generator
    samples = []
    q_samples_list = [[] for _ in range(nmodes)]

    cdf_list = []
    for p, xgrid in zip(prob_therm_list, xgrid_list):
        dx = xgrid[1] - xgrid[0]
        cdf = np.cumsum(p * dx)
        cdf /= cdf[-1]  # normalize to 1
        cdf_list.append(cdf)


    # --- Generate NSAMPLES geometries ---
    for _ in range(NSAMPLES):
        new_coords = coords.copy()
        for mode in range(nmodes):
            u = rng.random()
            q = np.interp(u, cdf_list[mode], xgrid_list[mode])
             # Apply the a1 shift for mode (in Bohr) only in anharmonic case
            if apply_shift:  
                q_shifted = q + a1_list_bohr[mode]
            else:
                q_shifted = q
            q_samples_list[mode].append(q_shifted)
            new_coords += displacements[mode] * q_shifted
        samples.append(new_coords)

    samples = [np.array(s) for s in samples]
    # ---- CONVERSION TO Å ----
    samples_ang = [s * BOHR_TO_ANG for s in samples]

    # --- Save XYZ geometries ---
    for i, geom in enumerate(samples_ang, 1):
        log_name = os.path.basename(GAUSSIAN_LOG) 
        filename = os.path.join(out_xyz_dir, f"g{i}_{log_name}.xyz")
        with open(filename, "w") as f:
            f.write(f"{natoms}\nSample {i}\n")
            for sym, xyz in zip(symbols, geom):
                f.write(f"{sym} {' '.join(f'{x:.6f}' for x in xyz)}\n")

    # --- Save all geometries in a single file ---
    with open(all_xyz_file, "w") as f:
        for i, geom in enumerate(samples_ang, 1):
            f.write(f"{natoms}\nSample {i}\n")
            for sym, xyz in zip(symbols, geom):
                f.write(f"{sym} {' '.join(f'{x:.6f}' for x in xyz)}\n")

    print(f"Generated {len(samples)} geometries in '{out_xyz_dir}' and '{all_xyz_file}'")

    # --- Generate histograms ---
    for mode in range(nmodes):
        q_samples = q_samples_list[mode]

        plt.figure()
        plt.hist(q_samples, bins=30, density=True, alpha=0.6, label='Sampled q')

        # Plot theoretical probability distribution
        xgrid = xgrid_list[mode]
        prob = prob_therm_list[mode]
        shift = a1_list_bohr[mode]
        dx = xgrid[1] - xgrid[0]
        plt.plot(xgrid+shift, prob, 'r-', lw=2, label='|ψ0(x)|²')

        # KS test
        cdf = cdf_list[mode]
        def cdf_theoretical(x):
            return np.interp(x, xgrid+shift, cdf)
        D, p_value = kstest(q_samples, cdf_theoretical)

        # Display KS results on plot
        plt.text(
            0.05, 0.95,
            f"KS D={D:.3e}\np={p_value:.3f}",
            transform=plt.gca().transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
        )

        plt.xlabel("Displacement q (Bohr)")
        plt.ylabel("Probability Density")
        plt.title(f"Mode {mode+1} displacement distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(hist_dir, f"mode{mode+1}_distribution.png"))
        plt.close()

    print(f"Histograms saved in '{hist_dir}'")

# ---------------------------------------------
# --- Generate both anharmonic and harmonic ---
# ---------------------------------------------
generate_samples(xgrid_list, prob_therm_list, out_xyz_dir="NX_therm", hist_dir="histograms",all_xyz_file="all_geometries.xyz",apply_shift=True)
generate_samples(xgrid_list, prob_h_list, out_xyz_dir="NX_har", hist_dir="histograms_har", all_xyz_file="all_geometries_har.xyz", apply_shift=False)
