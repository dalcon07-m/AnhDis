#!/usr/bin/env python3

import os, re
import numpy as np
from config import GAUSSIAN_LOG, PERIODIC_TABLE, SCALING_FACTORS, SELECTED_MODES, BOHR_TO_ANG, AMU_TO_EMASS, CM_TO_HARTREE

#_____________________________________________________________________________________________________#

#---- Function to scale the optimized geometry adding the displacements (in distance units) by -------#
#----                        an adimensional scale factor                                      -------# 

def scale_and_save(matrix, mode_index, scale_factor, base_coord, atoms_symbols, outdir):
    os.makedirs(outdir, exist_ok=True)
    scaled_matrix = np.array(base_coord) + np.array(matrix) * scale_factor   # all in A
    outfile = os.path.join(outdir, f'vib{mode_index}_{scale_factor:.2f}.xyz')
    with open(outfile, 'w') as f:
        f.write(f"{len(atoms_symbols)}\n")
        f.write("Angstrom\n")
        for sym, pos in zip(atoms_symbols, scaled_matrix):
            f.write(f"{sym} {' '.join(f'{x:.6f}' for x in pos)}\n")
#____________________________________________________________________________________________________


#------------------------- Read atomic symbols from periodic table ------------------------------------#
#-------------------------   and optimized geometry from Gaussian   -----------------------------------# 
atomic_data = {}
with open(PERIODIC_TABLE) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 4:
            Z = int(parts[0])
            symbol = parts[1]
            mass = float(parts[3])
            atomic_data[Z] = {
                "symbol": symbol,
                "mass": mass
            }

# --- Read last optimized geometry (A) ---
lines = open(f"{GAUSSIAN_LOG}.log").readlines()
# First, try to find Standard orientation
indices = [i for i, line in enumerate(lines) if 'Standard orientation' in line]
# If not found, fall back to Input orientation
if not indices:
    indices = [i for i, line in enumerate(lines) if 'Input orientation' in line]
# If neither is found, raise an error
if not indices:
    raise ValueError("No 'Standard orientation' or 'Input orientation' found in log.")

last_index = indices[-1]
coord_start = last_index + 5
atoms, coords = [], []
for line in lines[coord_start:]:
    if '---' in line:
        break
    vals = line.split()
    atoms.append(int(vals[1]))
    coords.append([float(vals[3]), float(vals[4]), float(vals[5])])
coords = np.array(coords)
coords_bohr= coords/BOHR_TO_ANG
symbols = [atomic_data[z]["symbol"] for z in atoms]
natoms = len(atoms)
nmodes = natoms * 3 - 6

# Print optimized geometry Gaussian
outfile = "geometry_gaussian_ang.dat"
outfile_bohr = "geometry_gaussian_bohr.dat"

with open(outfile, "w") as f:
    for sym, pos in zip(symbols, coords):
        f.write(f"{sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}\n")
with open(outfile_bohr, "w") as f:
    for sym, pos in zip(symbols, coords_bohr):
        f.write(f"{sym:2s}  {pos[0]:12.6f}  {pos[1]:12.6f}  {pos[2]:12.6f}\n")


print(f"Gaussian geometry saved to {outfile}")


#!! important: Because Gaussian prints the normal-mode displacements with limited numerical precision,
#the vectors read from the output are not strictly normalized. As a consequence, reduced masses computed 
#directly from these displacements may differ slightly from their exact values. We therefore renormalize
#the coordinates and evaluate the corresponding reduced masses. 
#________________________________________________________________________________________________________




# ----------------------- Read reduced masses and frequencies from Gaussian------------------------------#
frequencies=[]
gaussian_red_masses = []
with open(f"{GAUSSIAN_LOG}.log") as f:
    for line in f:
        if "Red. masses --" in line:
            gaussian_red_masses.extend([float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line.split('--')[1])])
        if "Frequencies --" in line:
            frequencies.extend([float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line.split('--')[1])])

frequencies = np.array(frequencies)
gaussian_red_masses = np.array(gaussian_red_masses)

# ________________________________________________________________________________________________________


# ----------------------- Read vibrational vectors for selected modes -------------------------

displacements = []
# read gaussian log 
with open(f"{GAUSSIAN_LOG}.log") as f:
    log_lines = f.readlines()

all_modes = []
i = 0

while i < len(log_lines):
    line = log_lines[i]
    if "Frequencies --" in line:
        parts = line.split()
        n_block_modes = len(parts) - 2   # 3 frequencies per block

        while i < len(log_lines) and "Atom  AN" not in log_lines[i]:
            i += 1
        i += 1  # first line of displacements

        block = []
        for _ in range(natoms):
            cols = log_lines[i].split()
            needed = 2 + 3 * n_block_modes
            if len(cols) < needed:
                raise ValueError(f"Incomplet displacements in {GAUSSIAN_LOG}.log:\n{log_lines[i]}")
            floats = [float(x) for x in cols[2:needed]]
            block.append(floats)
            i += 1

        block = np.array(block)  # (natoms × 3*n_block_modes)

        # Separate each mode
        for m in range(n_block_modes):
            vec = block[:, 3*m : 3*(m+1)]  # natoms × 3
            all_modes.append(vec)
    else:
        i += 1

displacements = all_modes
# --- security ---
if len(all_modes) != nmodes:
    print(f"Advertencia: se esperaban {nmodes} modos pero se leyeron {len(all_modes)}.")

#-------------------Recalculate the reduced masses from the displacements--------------------
recalc_red_masses = []

for mode_idx, mode_vec in enumerate(displacements):
    mu_mode = 0.0
    for atom_idx, Z in enumerate(atoms):  
        m = atomic_data[Z]["mass"]  #  amu
        vec = mode_vec[atom_idx]  # 3 components
        mu_mode += m * np.sum(vec**2)
    recalc_red_masses.append(mu_mode)

recalc_red_masses = np.array(recalc_red_masses)

#___________________________________________


#--------------     New displacements in cartesian units d/sqrt(mu)    -------------
cartesian_displacements= []
for mode_idx, mode_vec in enumerate(displacements):
    mu_mode = recalc_red_masses[mode_idx]  # masa reducida del modo
    cart_vec = mode_vec / np.sqrt(1)
#    cart_vec = mode_vec / np.sqrt(mu_mode)   #use the same as gaussian prints
    cartesian_displacements.append(cart_vec)
cartesian_displacements = np.array(cartesian_displacements)
#___________________________________________________________________________________


for mode in range(1, nmodes + 1):
    if SELECTED_MODES and f"vib{mode}" not in SELECTED_MODES and mode not in SELECTED_MODES:
        continue

    vibration = cartesian_displacements[mode - 1]
    mode_dir = f"vib{mode}"
    for scale in SCALING_FACTORS:
        scale_dir = os.path.join(mode_dir, f"{scale:.2f}")
        scale_and_save(vibration, mode, scale, coords, symbols, scale_dir)

#---------------------   Export all vibrational data ----------------------------

output_file = "vibrational_data.txt"

with open(output_file, "w") as f:
    # Header
    f.write("# Mode  Gaussian_RM  Recalc_RM  Frequency  Cart_modes (x1 y1 z1 ...)  Gaussian_modes (x1 y1 z1 ...)\n")
    
    for mode_idx in range(nmodes):
        mode_number = mode_idx + 1
        g_rm = gaussian_red_masses[mode_idx]
        r_rm = recalc_red_masses[mode_idx]
        freq = frequencies[mode_idx]
        
        cart_mode = cartesian_displacements[mode_idx].flatten()
        gauss_mode = displacements[mode_idx].flatten()
        
        # Format all floats to 6 decimals, join with spaces
        cart_str = " ".join(f"{x:.6e}" for x in cart_mode)
        gauss_str = " ".join(f"{x:.6e}" for x in gauss_mode)
        
        f.write(f"{mode_number:4d}  {g_rm:12.6f}  {r_rm:12.6f}  {freq:12.6f}  {cart_str}  {gauss_str}\n")

print(f"Vibrational data exported to {output_file}")


print("\nDisplaced geometries generated for single point calculation.")
