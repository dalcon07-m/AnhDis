#!/usr/bin/env python3

import os
from config import GAUSSIAN_HEADER, SELECTED_MODES

def create_gaussian_input(head_file, xyz_file, out_file):
    with open(head_file) as f:
        head = f.read().rstrip()
    with open(xyz_file) as f:
        coords = "".join(f.readlines()[2:])  # skip header lines
    with open(out_file, 'w') as f:
        f.write(head + "\n" + coords + "\n")

# --- Walk folders ---
for root, dirs, files in os.walk("."):
    if not SELECTED_MODES or any(mode in root for mode in SELECTED_MODES):
        for file in files:
            if file.endswith(".xyz"):
                xyz_path = os.path.join(root, file)
                com_path = os.path.splitext(xyz_path)[0] + ".com"
                create_gaussian_input(GAUSSIAN_HEADER, xyz_path, com_path)
                print(f"Generated {com_path}")

