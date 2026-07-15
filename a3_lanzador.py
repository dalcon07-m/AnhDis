#!/usr/bin/env python3

import os
import random
from config import SELECTED_MODES, SCALING_FACTORS, NODE_CPUS, ALLOWED_NODES

USER = os.environ['USER']
MAIN_DIR = os.getcwd()

allowed_nodes_with_cpus = {node: NODE_CPUS[node] for node in ALLOWED_NODES if node in NODE_CPUS}


# Pesos proporcionales al número de CPUs
nodes = list(allowed_nodes_with_cpus.keys())
weights = list(allowed_nodes_with_cpus.values())

# Mostrar las probabilidades de selección
total_cpus = sum(weights)
print("Node selection probabilities:")
for node, w in zip(nodes, weights):
    print(f"  {node}: {w/total_cpus*100:.1f}%")

for vib in sorted(os.listdir(MAIN_DIR)): 
    vib_path = os.path.join(MAIN_DIR, vib)
    if not os.path.isdir(vib_path):
        continue

    if SELECTED_MODES and vib not in SELECTED_MODES and f"vib{vib}" not in SELECTED_MODES:
        continue
    
    for scale in sorted(os.listdir(vib_path)):
        if scale not in [f"{s:.2f}" for s in SCALING_FACTORS]:
            continue
        scale_dir = os.path.join(vib_path, scale)
        com_file = os.path.join(scale_dir, f"{vib}_{scale}.com")
        if not os.path.exists(com_file):
            print(f"Missing {com_file}, skipping...")
            continue
        selected_node = random.choices(nodes, weights=weights, k=1)[0]
        run_script = os.path.join(scale_dir, f"run_{vib}_{scale}.sh")
        content = f"""#!/bin/bash
#PBS -u {USER}
#PBS -N {vib}_{scale}
#PBS -l nodes={selected_node}:ppn=2
#PBS -S /bin/bash
FILE="{vib}_{scale}"
export ScrDir=/scr/{USER}/${{PBS_JOBID}}_$FILE
mkdir -p $ScrDir
Wdir="{scale_dir}"
. /soft/g16.a03/g16/bsd/g16.profile
cd $ScrDir
g16 < $Wdir/$FILE.com > $ScrDir/$FILE.log
cp *.log $Wdir
cp *.o $Wdir
cp *.e $Wdir
exit
"""
        with open(run_script, 'w') as f:
            f.write(content)
        os.system(f"qsub {run_script}")
        print(f"Submitted {vib}_{scale} → {selected_node}")
