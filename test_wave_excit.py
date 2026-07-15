#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from config import FIT_COEFF_FILE, AMU_TO_EMASS, L, NP, BOHR_TO_ANG

#------------ Configuración de la Prueba ---------
TEST_WAVE_PDF = "test_excited_states.pdf"
N_EXCITED_STATES = 4  # Número de estados a graficar (v=0, 1, 2, 3)

#------------ Rutinas Cuánticas ---------

def build_hamiltonian(k, kc, kq, mr_emass, xgrid):
    n = xgrid.size
    dx = xgrid[1] - xgrid[0]
    # Potencial Anarmónico V(x) = 1/2 k x^2 + 1/6 kc x^3 + 1/24 kq x^4
    V = np.diag(0.5*k*xgrid**2 + (1/6)*kc*xgrid**3 + (1/24)*kq*xgrid**4)
    main = np.full(n, -2.0)
    off = np.ones(n-1)
    D2 = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / (dx*dx)
    T = -1/(2*mr_emass) * D2
    return T + V

def normalize_wave(psi, dx):
    norm = np.sqrt(np.trapezoid(np.abs(psi)**2, dx=dx))
    return psi / norm

#------------ Lectura de Coeficientes ---------

modes = []
with open(FIT_COEFF_FILE) as f:
    for line in f.readlines()[1:]:
        parts = line.split()
        if len(parts) >= 7:
            modes.append({
                "vib": parts[0],
                "a1_b": float(parts[1])/BOHR_TO_ANG,
                "k_b": float(parts[2])*(BOHR_TO_ANG**2),
                "kc_b": float(parts[3])*(BOHR_TO_ANG**3),
                "kq_b": float(parts[4])*(BOHR_TO_ANG**4),
                "mr_emass": float(parts[5])*AMU_TO_EMASS
            })

#======================================================
# Main loop para graficar estados excitados
#======================================================

xgrid = np.linspace(-L, L, NP)
pdf_pages = PdfPages(TEST_WAVE_PDF)

for m in modes:
    vib = m["vib"]
    
    # 1. Resolver el Hamiltoniano completo
    H = build_hamiltonian(m["k_b"], m["kc_b"], m["kq_b"], m["mr_emass"], xgrid)
    evals, evecs = np.linalg.eigh(H)
    
    # 2. Preparar figura para este modo (4 subplots)
    fig, axs = plt.figure(figsize=(10, 12)), plt.gcf().get_axes()
    fig.suptitle(f"Anharmonic Excited States - Mode: {vib}", fontsize=14)
    
    for v in range(N_EXCITED_STATES):
        # Graficar v=0, v=1, v=2, v=3
        ax = plt.subplot(N_EXCITED_STATES, 1, v+1)
        
        # Extraer y normalizar la función de onda del estado v
        psi_v = normalize_wave(evecs[:, v], xgrid[1]-xgrid[0])
        
        # Graficar (centrado en el equilibrio real a1_b)
        ax.plot(xgrid + m["a1_b"], psi_v, 'b-', lw=1.5)
        
        # Añadir línea de base en cero y etiquetas
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_ylabel(rf"$\psi_{{{v}}}(x)$")
        ax.set_title(f"State v={v} (Energy: {evals[v]:.6f} Hartree)", fontsize=10)
        
        if v == N_EXCITED_STATES - 1:
            ax.set_xlabel("Coordinate (bohr)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf_pages.savefig(fig)
    plt.close(fig)

pdf_pages.close()
print(f"Prueba completada. Funciones de onda excitadas guardadas en: {TEST_WAVE_PDF}")
