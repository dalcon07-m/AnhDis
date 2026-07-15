#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# =========================
# Funciones geométricas
# =========================

def distance(a, b):
    return np.linalg.norm(a - b)  #euclidean distance between two points


def dihedral(p1, p2, p3, p4):
    """
    Dihedro p1–p2–p3–p4 en grados.
    p1---p4  3D vectors corresponding to C-O-O-H
    Devuelve valor en (-180, 180].
    """
    b1 = p2 - p1 # from p1 to p2
    b2 = p3 - p2 # from p2 to p3
    b3 = p4 - p3 # from p3 to p4

    n1 = np.cross(b1, b2)  # vector normal to plane b1,b2
    n2 = np.cross(b2, b3)  # vector normal to plane b2,b3
    #n1, n2 planes to form the dihedral
    
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    b2 /= np.linalg.norm(b2) #b2 is the central axis

    m1 = -np.cross(n1, b2) #m1 perpendicular to n1, b2, opposite to b1

    x = np.dot(n1, n2) #cos(theta) magnitude of the angle
    y = np.dot(m1, n2) #projection of the normal of second plane on m1 
    # y gives the sign, positive or negative it is cos() between m1 and n2, then sin() between n1,n2

    return np.degrees(np.arctan2(y, x)) #atan2(y,x) to give the angle depending the sign of x and y


# =========================
# Lectura XYZ
# =========================

def read_xyz(filename):
    geometries = []

    with open(filename) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        n_atoms = int(lines[i].strip())
        i += 2  # comentario

        geom = []
        for _ in range(n_atoms):
            s, x, y, z = lines[i].split()
            geom.append((s, np.array([float(x), float(y), float(z)])))
            i += 1

        geometries.append(geom)

    return geometries


# =========================
# Identificación C–O–O–H
# =========================

def identify_atoms(geom):
    symbols = [a[0] for a in geom]

    idx_C = symbols.index("C")
    idx_O = [i for i, s in enumerate(symbols) if s == "O"]
    idx_H = [i for i, s in enumerate(symbols) if s == "H"]

    min_dist = 1e9
    idx_H_oo = None
    idx_O_oo = None

    for iO in idx_O:
        for iH in idx_H:
            d = distance(geom[iO][1], geom[iH][1])
            if d < min_dist:
                min_dist = d
                idx_H_oo = iH
                idx_O_oo = iO

    idx_O_c = [i for i in idx_O if i != idx_O_oo][0]
    return idx_C, idx_O_c, idx_O_oo, idx_H_oo


# =========================
# Análisis principal
# =========================

xyz_file = "all_geometries.xyz"
geometries = read_xyz(xyz_file)

dihedrals = []
oh_distances = []

for geom in geometries:
    iC, iO1, iO2, iH = identify_atoms(geom)

    C = geom[iC][1]
    O1 = geom[iO1][1]
    O2 = geom[iO2][1]
    H = geom[iH][1]

    dihedrals.append(dihedral(C, O1, O2, H))
    oh_distances.append(distance(O2, H))

dihedrals = np.array(dihedrals)
#dihedrals = dihedrals * 1.5 
dihedrals = np.where(dihedrals < 0, dihedrals + 360, dihedrals)


oh_distances = np.array(oh_distances)


# =========================
# Histograma 2D (conteo)
# =========================

nbins_r = 60
nbins_a = 80

H, xedges, yedges = np.histogram2d(
    oh_distances,
    dihedrals,
    bins=(nbins_r, nbins_a),
    range=[[0.7, 1.3], [50, 320]]
)

# =========================
# Suavizado FUERTE
# =========================
# sigma más alto para eliminar picos
H_smooth = gaussian_filter(H, sigma=2.0)

# =========================
# Normalización comparable
# =========================
H_norm = H_smooth / np.sum(H_smooth)



X, Y = np.meshgrid(
    0.5 * (xedges[:-1] + xedges[1:]),
    0.5 * (yedges[:-1] + yedges[1:])
)


# =========================
# Plot 2D suavizado
# =========================

plt.figure(figsize=(7, 6))

cf = plt.contourf(
    X,
    Y,
    H_norm.T,
    levels=100,
    cmap="viridis"
)

plt.contour(
    X,
    Y,
    H_smooth.T,
    levels=12,
    colors="black",
    linewidths=0.6
)

plt.xlabel("Distancia O–H (Å)")
plt.ylabel("Ángulo dihedro C–O–O–H (°)")

plt.xlim(0.7, 1.3)
plt.ylim(50, 320)

cbar = plt.colorbar(cf)
cbar.set_label("Densidad P")

plt.tight_layout()
plt.savefig("CH3OOH_surface_2D_smooth.png", dpi=300)
plt.close()

