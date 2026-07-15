#!/usr/bin/env python3

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from config import FIT_COEFF_FILE, WAVE_PDF, AMU_TO_EMASS, L, NP, BOHR_TO_ANG, TEMP, KB_AU

#------------ Read the parameters from fit_coefficients.dat---------
os.makedirs("probabilities", exist_ok=True)
THERMAL_SUMMARY = "thermal_summary.dat"
TEST_WAVE_PDF = "test_excited_states.pdf"

modes = []
with open(FIT_COEFF_FILE) as f:
    for line in f.readlines()[1:]:
        parts = line.split()
        if len(parts)>=7:
            vib = parts[0]
            a1 = float(parts[1])
            k = float(parts[2])
            kc = float(parts[3])
            kq = float(parts[4])
            mr = float(parts[5])
            k_har=float(parts[6])
            #TO A.U
            a1_b = a1/BOHR_TO_ANG          # shift a1 in bohr
            mr_emass = mr * AMU_TO_EMASS   # m in me
            k_b = k * (BOHR_TO_ANG**2)     # k in Hartree/Bohr^2
            kc_b = kc * (BOHR_TO_ANG**3)   # kc in Hartree/Bohr^3
            kq_b = kq * (BOHR_TO_ANG**4)   # kq in Hartree/Bohr^4
            k_har_b = k_har * (BOHR_TO_ANG**2)   # k_har in Hartree/Bohr^2

            modes.append({"vib": vib, "a1_b": a1_b, "k_b": k_b, "kc_b": kc_b, "kq_b": kq_b, "mr_emass": mr_emass, "k_har_b": k_har_b})


#_______________________________________________________________________________
#===============================================================================
# Quantum routines
#===============================================================================


def build_hamiltonian(k,kc,kq,mr_emass,xgrid):
    n = xgrid.size
    dx = xgrid[1]-xgrid[0]
    V = np.diag(0.5*k*xgrid**2 + (1/6)*kc*xgrid**3 + (1/24)*kq*xgrid**4)
    #---Kinetic --------------------
    main = np.full(n,-2.0)
    off = np.ones(n-1)
    D2 = (np.diag(main)+np.diag(off,1)+np.diag(off,-1))/(dx*dx)
    T = -1/(2*mr_emass)*D2
    return T+V

def normalize_wave(psi,dx):
    return psi/np.sqrt(np.trapezoid(np.abs(psi)**2,dx=dx))

def compute_state_boltz(k_b,kc_b,kq_b,mr_emass,temp, L=L,Np=NP):
    xgrid = np.linspace(-L,L,Np)
    dx = xgrid[1]-xgrid[0]
    H = build_hamiltonian(k_b,kc_b,kq_b,mr_emass,xgrid)
    evals,evecs = np.linalg.eigh(H)
    #Fundamental states
    E0= evals[0]
    psi0 = normalize_wave(evecs[:,0],xgrid[1]-xgrid[0])
    
    #Temperature
    beta = 1.0 / (KB_AU * temp)
    unnorm_weights = np.exp(-beta * (evals - E0))
    idx_rel = np.where(unnorm_weights > 1e-3)[0] #label for the excites states with considerable weights
    evecs_rel = evecs[:, idx_rel]
    weights = unnorm_weights[idx_rel] / np.sum(unnorm_weights[idx_rel])   #normalized weights

    p_thermal = np.zeros_like(xgrid)
    for i, v_idx in enumerate(idx_rel):
        psi_v = normalize_wave(evecs[:, v_idx], dx)
        p_thermal += weights[i] * (np.abs(psi_v)**2)

    return xgrid, psi0, E0 , p_thermal, weights, idx_rel, evecs_rel

#========================================================
# --- PDF setup ---
#=======================================================

pdf_pages = PdfPages(WAVE_PDF)
fig = plt.figure(figsize=(11,8))
plot_count = 0
plots_per_page = 6

pdf_excited = PdfPages(TEST_WAVE_PDF)
summary_f = open(THERMAL_SUMMARY, "w")
summary_f.write(f"# Thermal Analysis at T = {TEMP} K\n")
summary_f.write("# Mode | N_states | Weights (v0, v1, ...)\n")

#======================================================
#  Main loop
#======================================================


for m in modes:
    vib,a1_b,k_b,kc_b,kq_b,mr_emass,k_har_b = m["vib"],m["a1_b"] ,m["k_b"],m["kc_b"],m["kq_b"],m["mr_emass"],m["k_har_b"]
    
    # Anharmonic
    xgrid, psi0, E0, p_thermal, weights, idx_rel, evecs_rel = compute_state_boltz(k_b,kc_b,kq_b,mr_emass,TEMP)
    psi_abs = np.abs(psi0)
    psi_abs2 = psi_abs**2
    psi_thermal_eff = np.sqrt(p_thermal)
    

    # Harmonic
    x_h, psi_h, E0_h, *_ = compute_state_boltz(k_har_b,0,0,mr_emass, TEMP)
    psi_h_abs = np.abs(psi_h)
    psi_h_abs2 = psi_h_abs**2
    
    # Theoretical harmonic energy: E0_theory = 1/2 * sqrt(k/mu)
    E0_theory = 0.5 * np.sqrt(k_har_b / mr_emass)

    # Warning if calculated energy differs significantly from theory
    if abs(E0_h - E0_theory) / E0_theory > 0.01:  # 1% threshold
        raise RuntimeError(
            f"ENERGY MISMATCH for mode {vib}!\n"
            f"  Theoretical E0 = {E0_theory:.6e} a.u., Calculated E0 = {E0_h:.6e} a.u.\n"
            f"  ==> Check the grid limits ( [-{L},{L}] bohr is too small! )"
        )


    # -------------------------- SAVE .dat ---------------
   
    #weight register
    summary_f.write(f"{vib:8s} | {len(weights):3d} | {', '.join([f'{w:.4f}' for w in weights])}\n")


    outfile = f"probabilities/{vib}.dat"
    header = (
        "# x (bohr)    |psi(x)|^2 anharmonic    |psi(x)|^2 harmonic  |psi(x)_therm|^2   \n"
        f"# mode = {vib}\n"
        f"# L = {L} bohr, Np = {NP}\n"
    )

    np.savetxt(
        outfile,
        np.column_stack([xgrid, psi_abs2, psi_h_abs2, p_thermal ]),
        header=header
    )

    fig_exc = plt.figure(figsize=(10, 12))
    fig_exc.suptitle(f"Anharmonic Excited States - Mode: {vib}", fontsize=14)
    
    n=len(idx_rel)
    for i, v_idx in enumerate(idx_rel):
        # Graficar v=0, v=1, v=2, v=3
        ax = plt.subplot(n, 1, i+1)
        
        # Extraer y normalizar la función de onda del estado v
        psi_v = normalize_wave(evecs_rel[:, v_idx], xgrid[1]-xgrid[0])
        
        # Graficar (centrado en el equilibrio real a1_b)
        ax.plot(xgrid + m["a1_b"], psi_v, 'b-', lw=1.5)
        
        # Añadir línea de base en cero y etiquetas
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
        ax.set_ylabel(rf"$\psi_{v_idx}(x)$")
        ax.set_title(f"State v={v_idx} ", fontsize=10)
        
        if i == n - 1:
            ax.set_xlabel("Coordinate (bohr)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf_excited.savefig(fig_exc)
    plt.close(fig_exc)


    #====================================================
    #                Plot |psi|, |psi|^2 
    #====================================================


    plot_count+=1
    plt.subplot(3,2,plot_count)
    plt.plot(xgrid+a1_b,psi_abs,'o',markersize=3,label='Anharmonic')
    plt.plot(x_h,psi_h_abs,'-',label='Harmonic')
    plt.plot(xgrid+a1_b,psi_thermal_eff,'o',markersize=1,label='Therm_Anhar_eff')

    plt.xlabel("x (bohr)")
    plt.ylabel("|Psi0(x)|")
    plt.title(f"{vib} Wavefunction")
    plt.legend(fontsize=8)
    if plot_count==plots_per_page:
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close(fig)
        fig = plt.figure(figsize=(11,8))
        plot_count=0
    # --- Plot |psi|^2 ---
    plot_count+=1
    plt.subplot(3,2,plot_count)
    plt.plot(xgrid+a1_b,psi_abs2,'-',label='|Psi0(x)|^2')
    plt.plot(xgrid,psi_h_abs2,'-',label='|Psi0(x)|^2 Harmonic')
    plt.plot(xgrid+a1_b,p_thermal,'-',label='|Psi(x)_therm|^2')
    plt.xlabel("x (bohr)")
    plt.ylabel("Probability")
    plt.title(f"{vib} Probability Distribution")
    plt.legend(fontsize=8)
    if plot_count==plots_per_page:
        plt.tight_layout()
        pdf_pages.savefig(fig)
        plt.close(fig)
        fig = plt.figure(figsize=(11,8))
        plot_count=0

if plot_count>0:
    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close(fig)
pdf_pages.close()
pdf_excited.close()
summary_f.close()
print(f"Prueba completada. Funciones de onda excitadas guardadas en: {TEST_WAVE_PDF}")





print(f"PDF generated: {WAVE_PDF}")
print("Probability distributions written to probabilities/*.dat")
