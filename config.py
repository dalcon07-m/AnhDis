#!/usr/bin/env python3
import os

# --------------------- Paths and input files---------------------
PROJECT_ROOT = os.getcwd()  #current directory
GAUSSIAN_LOG = os.path.join(PROJECT_ROOT, 'b3lyp_6311g2dp') 
GAUSSIAN_HEADER = os.path.join(PROJECT_ROOT, 'gaussian_head.com')
PERIODIC_TABLE = '/home/macias/.scripts/Periodic_Table_mass.dat'
TEMP = 298.15   #ambient temp.


# --------------------- Vibration Modes ---------------------
SELECTED_MODES = [ ]   # empty list = all modes, here you can select the modes of interest

#The scaling factors are in adimensional units (but it could be A o Bohr), all depends on the displacement units  
SCALING_FACTORS = [ -0.4, -0.3, -0.2, -0.1, 0.00, 0.1, 0.2, 0.3, 0.4]  

# --------------------- Fitting & Sampling ---------------------
#___________________________________________________

L = 5.0         # grid half-length for wavefunction (-L,L (bohr)), look for the energy wave function
NP = 1000        # grid points
NSAMPLES = 100  #for the number of geometries to be generated, as Newton-X
#___________________________________________________

#---------------------- Output files--------------------------
FIT_COEFF_FILE = "fit_coefficients.dat"
ENERGIES_FILE = "energies.dat"
VIB_PDF = "vib_plots.pdf"
WAVE_PDF = "wave_wigner.pdf"
VIB_DAT="vibrational_data.txt"
OPT_GEO="geometry_gaussian_bohr.dat"
#____________________________________________________________

#--------------------conversion units-------------------------
BOHR_TO_ANG = 0.52917720859
AMU_TO_EMASS = 1822.888515
CM_TO_HARTREE= 4.55633539E-6
KB_AU = 3.166811e-6
#___________________________________________________

#--------------------- Computational ---------------------
NODE_CPUS = {
    "qcexnod62": 24,  
    "qcexnod61": 24,  
    "qcexnod60": 24,  
    "qcexnod59": 24, 
    "qcexnod58": 24,  
    "qcexnod57": 24,  
    "qcexnod56": 24,  
    "qcexnod55": 24,  
    "qcexnod54": 24,  
    "qcexnod52":  3,  
    "qcexnod51":  3,  
    "qcexnod50": 10,  
    "qcexnod01": 12,  
    "qcexnod02": 12,
    "qcexnod03": 24,  
    "qcexnod10":  6,  
    "qcexnod11":  6,  
    "qcexnod12":  6,  
    "qcexnod13":  6,  
    "qcexnod14":  6,  
    "qcexnod15":  6,  
    "qcexnod16":  6,
    "qcexnod17":  6,
    "qcexnod18":  6,
    "qcexnod19":  6,
    "qcexnod04": 16
}
#selected nodes to send calculations

ALLOWED_NODE_NUMS = [51, 52, 2, 1]  #edit here

ALLOWED_NODES = [f"qcexnod{n:02d}" for n in ALLOWED_NODE_NUMS]

