#!/usr/bin/env python3
import numpy as np
import glob

def read_xyz(fname):
    coords = []
    with open(fname) as f:
        lines = f.readlines()[2:]  # saltar n_atoms y comentario
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        x, y, z = map(float, parts[-3:])
        coords.append([x, y, z])
    return np.array(coords)

def kabsch(P, Q):
    # centra
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1, 1, d])
    U = V @ D @ Wt

    return Pc @ U, Qc

def rmsd(P, Q):
    return np.sqrt(np.mean(np.sum((P - Q)**2, axis=1)))

# ---- archivos ----
ref_file = "ref.xyz"
xyz_files = sorted([f for f in glob.glob("g*.xyz") if f != ref_file])

# ---- cargar referencia ----
ref = read_xyz(ref_file)
print("Referencia:", ref.shape)

for f in xyz_files[:3]:
    g = read_xyz(f)
    print(f, g.shape)


# ---- RMSD respecto a referencia ----
rmsd_ref = []

geoms = []

for f in xyz_files:
    G = read_xyz(f)
    G_aligned, ref_aligned = kabsch(G, ref)
    geoms.append(G_aligned)
    rmsd_ref.append(rmsd(G_aligned, ref_aligned))

rmsd_ref = np.array(rmsd_ref)

# ---- RMSD respecto a la geometría media ----
mean_geom = np.mean(geoms, axis=0)

rmsd_mean = np.array([
    rmsd(G, mean_geom) for G in geoms
])

# ---- resultados básicos ----
print("RMSD respecto a referencia:")
print("  media =", rmsd_ref.mean())
print("  std   =", rmsd_ref.std())

print("\nRMSD respecto a geometría media:")
print("  media =", rmsd_mean.mean())
print("  std   =", rmsd_mean.std())

