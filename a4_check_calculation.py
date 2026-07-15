#!/usr/bin/env python3

import os
from config import SELECTED_MODES, SCALING_FACTORS

MAIN_DIR = os.getcwd()

for vib in sorted(os.listdir(MAIN_DIR)):
    vib_path = os.path.join(MAIN_DIR, vib)
    
    if not os.path.isdir(vib_path):
        continue

    if SELECTED_MODES and vib not in SELECTED_MODES and f"vib{vib}" not in SELECTED_MODES:
        continue

    for scale in sorted(os.listdir(vib_path)):
        if scale not in [f"{s:.2f}" for s in SCALING_FACTORS]:
            continue
        log_file = os.path.join(vib_path, scale, f"{vib}_{scale}.log")
        if not os.path.exists(log_file):
            print(f"[MISSING] {log_file}")
            continue
        with open(log_file, 'r', errors='ignore') as f:
            content = f.read()

        if "Normal termination of Gaussian" in content:
            status = "[OK]"
        elif "Error termination" in content:
            status = "[ERROR]"
        else:
            status = "[INCOMPLETE]"
        print(f"{status} {vib}_{scale}")

