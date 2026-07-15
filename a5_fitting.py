#!/usr/bin/env python3

import os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from config import (
    PROJECT_ROOT as MAIN_DIR, SELECTED_MODES, SCALING_FACTORS,
    GAUSSIAN_LOG, ENERGIES_FILE, FIT_COEFF_FILE, VIB_PDF,
    BOHR_TO_ANG, CM_TO_HARTREE, AMU_TO_EMASS, L, NP
)

MAIN_DIR = os.getcwd()
scf_pattern = re.compile(r"SCF Done:\s+E\(.+?\)\s*=\s*([-\d\.]+)")

# ================== Read vibrational data ==================
filename = "vibrational_data.txt"

recalc_rm = []
freqs = []

with open(filename) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        recalc_rm.append(float(parts[2]))
        freqs.append(float(parts[3]))

recalc_rm = np.array(recalc_rm)
freqs = np.array(freqs)

# Harmonic force constant (Hartree/Å^2)
k_har_A = (freqs*CM_TO_HARTREE)**2 * AMU_TO_EMASS * recalc_rm / (BOHR_TO_ANG**2)

# ================== Potential functions ==================

def poly4(x, a0, a1, a2, a3, a4):
    xx = x - a1
    return a0 + a2*xx**2 + a3*xx**3 + a4*xx**4

def harmonic(x, a0, k):
    return a0 + 0.5*k*x**2

# ================== Prepare vibration folders ==================

vibs = sorted(
    [
        d for d in os.listdir(MAIN_DIR)
        if os.path.isdir(os.path.join(MAIN_DIR, d))
        and (not SELECTED_MODES or d in SELECTED_MODES or f"vib{d}" in SELECTED_MODES)
    ],
    key=lambda x: int(re.findall(r'\d+', x)[0]) if re.search(r'\d+', x) else 0
)

coef_lines = []

# ================== PDF 1: Data + Fit ==================

pdf_pages = PdfPages(VIB_PDF)
plots_per_page = 4
plot_count = 0
rm_idx = 0

FIGSIZE = (7.5, 10.5)

def new_figure():
    fig = plt.figure(figsize=FIGSIZE)
    fig.subplots_adjust(
        left=0.12,
        right=0.93,
        bottom=0.08,
        top=0.95,
        wspace=0.40,
        hspace=0.35
    )
    return fig

fig = new_figure()

# ================== PDF 2: Full potentials ==================

POTENTIAL_PDF = "full_potentials.pdf"
pdf_pot = PdfPages(POTENTIAL_PDF)
plot_count_pot = 0
fig_pot = new_figure()

# ===========================================================

for vib in vibs:

    vib_path = os.path.join(MAIN_DIR, vib)
    results = []

    for scale in sorted(os.listdir(vib_path)):
        if scale not in [f"{s:.2f}" for s in SCALING_FACTORS]:
            continue

        log_file = os.path.join(vib_path, scale, f"{vib}_{scale}.log")
        if not os.path.exists(log_file):
            continue

        with open(log_file) as f:
            energy = next(
                (float(m.group(1)) for line in f if (m := scf_pattern.search(line))),
                None
            )
            if energy is not None:
                results.append((float(scale), energy))

    if not results:
        continue

    results.sort(key=lambda x: x[0])
    np.savetxt(os.path.join(vib_path, ENERGIES_FILE), results, fmt='%.6f')

    x = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])

    weights = 1/(1+np.abs(x)) + 2.0/(1+x**2)
    sigma = 1/np.sqrt(np.maximum(weights, 1e-12))

    E0_guess = np.min(y)
    x0_guess = x[np.argmin(y)]
    a2_guess = max((np.max(y)-np.min(y)) /
                   ((np.max(x)-np.min(x))**2+1e-12), 1e-3)

    a3_guess = 0.0
    a4_guess = max(a2_guess*1e-2, 0.0)

    max_cubic = 10 * a2_guess

    lower = [-np.inf, x.min(), -np.inf, -max_cubic, 0.0]
    upper = [np.inf, x.max(), np.inf, max_cubic, np.inf]

    p0 = [E0_guess, x0_guess, a2_guess, a3_guess, a4_guess]

    popt, _ = curve_fit(
        poly4, x, y, p0=p0,
        sigma=sigma,
        bounds=(lower, upper),
        maxfev=20000
    )

    a0, a1, a2, a3, a4 = popt
    k, kc, kq = 2*a2, 6*a3, 24*a4

    omega = freqs[rm_idx]
    E0_mode = 0.5 * CM_TO_HARTREE * omega
    Emax_plot = a0 + 60*E0_mode

    red_mass = recalc_rm[rm_idx]
    k_har = k_har_A[rm_idx]
    rm_idx += 1

    coef_lines.append(
        f"{vib} {a1:.6f} {k:.6f} {kc:.6f} {kq:.6f} "
        f"{red_mass:.6f} {k_har:.6f}"
    )

    x_check = np.linspace(-L*BOHR_TO_ANG, L*BOHR_TO_ANG, NP)
    V_fit = poly4(x_check, *popt)
    V_harm = harmonic(x_check, a0, k_har)

    # ================== FULL POTENTIAL PLOT ==================

    plot_count_pot += 1
    ax_pot = fig_pot.add_subplot(2, 2, plot_count_pot)

    ax_pot.plot(x_check, V_fit, '-', label='4th-order Fit')
    ax_pot.plot(x_check, V_harm, '--', label='Harmonic Approx.')
    ax_pot.set_xlabel("Displacement (Å)")
    ax_pot.set_ylabel("Energy (Hartree)")
    ax_pot.set_title(vib)
    ax_pot.set_ylim(a0, Emax_plot)
    ax_pot.legend(fontsize=8)

    if plot_count_pot == plots_per_page:
        pdf_pot.savefig(fig_pot)
        plt.close(fig_pot)
        fig_pot = new_figure()
        plot_count_pot = 0

    # ================== DATA + FIT PLOT ==================

    plot_count += 1
    ax = fig.add_subplot(2, 2, plot_count)

    ax.plot(x, y, 'o', label='Data')
    xfit = np.linspace(min(x), max(x), 200)
    ax.plot(xfit, poly4(xfit, *popt), '-', label='Fit')
    ax.plot(xfit, harmonic(xfit, a0, k_har), '--', label='Harmonic Appr.')

    ax.set_xlabel("Displacement (Å)")
    ax.set_ylabel("Energy (Hartree)")
    ax.set_title(vib)
    ax.legend(fontsize=8)

    text = (
        f"k = {k:.6f} Hartree/A^2\n"
        f"kc = {kc:.6f} Hartree/A^3\n"
        f"kq = {kq:.6f} Hartree/A^4\n"
        f"k_harm = {k_har:.6f} Hartree/A^2"
    )

    ax.text(
        0.05, 0.95, text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=8,
        bbox=dict(facecolor='white', alpha=0.6)
    )

    if plot_count == plots_per_page:
        pdf_pages.savefig(fig)
        plt.close(fig)
        fig = new_figure()
        plot_count = 0

# ================== Close PDFs ==================

if plot_count > 0:
    pdf_pages.savefig(fig)
    plt.close(fig)

pdf_pages.close()

if plot_count_pot > 0:
    pdf_pot.savefig(fig_pot)
    plt.close(fig_pot)

pdf_pot.close()

# ================== Save coefficients ==================

with open(FIT_COEFF_FILE, 'w') as f:
    f.write("Vib a1(A) k(H/A^2) kc(H/A^3) kq(H/A^4) RedMass k_harm(H/A^2)\n")
    f.write("\n".join(coef_lines))

print("Full potential plots saved in:", POTENTIAL_PDF)
print("Data + fit plots saved in:", VIB_PDF)
print("Fit coefficients saved in:", FIT_COEFF_FILE)

