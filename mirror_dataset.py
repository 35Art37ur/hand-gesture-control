"""
mirror_dataset.py
-----------------
Erstellt gespiegelte Trainingsdaten für linke/rechte Hand.

Aufruf:
    python mirror_dataset.py --ordner wischen_links

Beispiel:
    Eingabe:
        trainingsdaten/
            wischen_links/
                sample_0.npy
                sample_1.npy

    Ausgabe:
        trainingsdaten/
            wischen_links/
            wischen_links_mirrored/
                sample_0_m.npy
                sample_1_m.npy
"""

import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(
    description="Erstellt gespiegelte Trainingsdaten."
)

parser.add_argument(
    "--ordner",
    required=True,
    help="Name des Gestenordners innerhalb von 'trainingsdaten'"
)

args = parser.parse_args()

input_folder = os.path.join("trainingsdaten", args.ordner)
output_folder = os.path.join("trainingsdaten", args.ordner + "_mirrored")

if not os.path.isdir(input_folder):
    raise FileNotFoundError(f"Ordner nicht gefunden: {input_folder}")

os.makedirs(output_folder, exist_ok=True)

dateien = sorted(
    f for f in os.listdir(input_folder)
    if f.endswith(".npy")
)

print(f"{len(dateien)} Dateien gefunden.")

for datei in dateien:

    pfad = os.path.join(input_folder, datei)

    # Shape: (Frames, 66)
    daten = np.load(pfad).astype(np.float32)

    gespiegelt = daten.copy()

    # ------------------------------------------------------
    # Absolute Wrist-Position spiegeln
    # Feature 0 = wrist_x
    # ------------------------------------------------------
    gespiegelt[:, 0] = 1.0 - gespiegelt[:, 0]

    # ------------------------------------------------------
    # Relative Landmark-x spiegeln
    # Ab Feature 3:
    # rel_x, rel_y, rel_z
    # rel_x liegt immer bei:
    # 3,6,9,12,...
    # ------------------------------------------------------
    gespiegelt[:, 3::3] *= -1

    basisname = os.path.splitext(datei)[0]
    neuer_name = basisname + "_m.npy"

    np.save(
        os.path.join(output_folder, neuer_name),
        gespiegelt
    )

    print(f"{datei} -> {neuer_name}")

print()
print("Fertig.")
print(f"Ausgabeordner: {output_folder}")