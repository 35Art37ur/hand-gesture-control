"""
mirror_dataset.py
-----------------
Erstellt erweiterte Trainingsdaten:
- Spiegelung der Handdaten
- Umkehrung der Frame-Reihenfolge
- Kombination aus beidem

Aufruf:
    python mirror_dataset.py --ordner wischen_links

Optionen:
    --no_mirror     Keine gespiegelten Daten erzeugen
    --no_reverse    Keine rückwärts abgespielten Daten erzeugen
"""

import argparse
import os
import numpy as np


parser = argparse.ArgumentParser(
    description="Erstellt gespiegelte und/oder rückwärts abgespielte Trainingsdaten."
)

parser.add_argument(
    "--ordner",
    required=True,
    help="Name des Gestenordners innerhalb von 'trainingsdaten'"
)

parser.add_argument(
    "--no_mirror",
    action="store_true",
    help="Keine gespiegelten Daten erzeugen"
)

parser.add_argument(
    "--no_reverse",
    action="store_true",
    help="Keine rückwärts abgespielten Daten erzeugen"
)

args = parser.parse_args()


input_folder = os.path.join("trainingsdaten", args.ordner)

if not os.path.isdir(input_folder):
    raise FileNotFoundError(f"Ordner nicht gefunden: {input_folder}")


# ------------------------------------------------------
# Ausgabeordner vorbereiten
# ------------------------------------------------------

output_folders = {}

if not args.no_mirror:
    output_folders["mirror"] = os.path.join(
        "trainingsdaten",
        args.ordner + "_mirrored"
    )

if not args.no_reverse:
    output_folders["reverse"] = os.path.join(
        "trainingsdaten",
        args.ordner + "_reversed"
    )

if not args.no_mirror and not args.no_reverse:
    output_folders["mir_rev"] = os.path.join(
        "trainingsdaten",
        args.ordner + "_mir_rev"
    )


for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)


# ------------------------------------------------------
# Dateien laden
# ------------------------------------------------------

dateien = sorted(
    f for f in os.listdir(input_folder)
    if f.endswith(".npy")
)

print(f"{len(dateien)} Dateien gefunden.")


# ------------------------------------------------------
# Spiegelungsfunktion
# ------------------------------------------------------

def mirror_data(daten):
    """
    Spiegelt Handdaten horizontal.
    Shape:
        (Frames, 66)
    """

    gespiegelt = daten.copy()

    # Absolute Wrist-Position
    # Feature 0 = wrist_x
    gespiegelt[:, 0] = 1.0 - gespiegelt[:, 0]

    # Relative Landmark-x Werte
    # rel_x befindet sich bei 3,6,9,...
    gespiegelt[:, 3::3] *= -1

    return gespiegelt


# ------------------------------------------------------
# Verarbeitung
# ------------------------------------------------------

for datei in dateien:

    pfad = os.path.join(input_folder, datei)

    daten = np.load(pfad).astype(np.float32)

    basisname = os.path.splitext(datei)[0]


    # ----------------------------------------------
    # Nur gespiegelt
    # ----------------------------------------------
    if "mirror" in output_folders:

        gespiegelt = mirror_data(daten)

        neuer_name = basisname + "_m.npy"

        np.save(
            os.path.join(output_folders["mirror"], neuer_name),
            gespiegelt
        )

        print(f"{datei} -> {neuer_name}")


    # ----------------------------------------------
    # Nur rückwärts
    # ----------------------------------------------
    if "reverse" in output_folders:

        rueckwaerts = daten[::-1].copy()

        neuer_name = basisname + "_r.npy"

        np.save(
            os.path.join(output_folders["reverse"], neuer_name),
            rueckwaerts
        )

        print(f"{datei} -> {neuer_name}")


    # ----------------------------------------------
    # Gespiegelt + rückwärts
    # ----------------------------------------------
    if "mir_rev" in output_folders:

        gespiegelt = mirror_data(daten)

        gespiegelt_rueckwaerts = gespiegelt[::-1].copy()

        neuer_name = basisname + "_mr.npy"

        np.save(
            os.path.join(output_folders["mir_rev"], neuer_name),
            gespiegelt_rueckwaerts
        )

        print(f"{datei} -> {neuer_name}")


print()
print("Fertig.")

for name, folder in output_folders.items():
    print(f"{name}: {folder}")