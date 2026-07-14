"""
pruefe_shapes.py
-----------------
Prueft, ob ALLE .npy-Dateien in allen Gesten-Ordnern dieselbe Form (Shape)
haben. Das ist Voraussetzung dafuer, dass train_model.py die Daten zu einem
einheitlichen NumPy-Array zusammenbauen kann (np.array(X)) -- bei
unterschiedlichen Shapes bricht das Training sonst mit einem eher
kryptischen Fehler ab ("could not broadcast input array" o.ae.).

Besonders relevant, wenn Daten aus mehreren Quellen zusammengefuehrt wurden
(z.B. record_gesture.py UND konvertiere_datensatz_newGesture.py),
da beide dieselbe Form (SEQUENZ_LAENGE, 67) erzeugen SOLLTEN, aber ein
Versehen (z.B. --seq-laenge vergessen/anders gesetzt) das durchbrechen kann.

Aufruf:
    python pruefe_shapes.py
    python pruefe_shapes.py --daten-ordner trainingsdaten_newGesture
"""

import argparse
import os
from collections import defaultdict

import numpy as np

parser = argparse.ArgumentParser(description="Prueft .npy-Shapes auf Konsistenz.")
parser.add_argument("--daten-ordner", type=str, default="trainingsdaten_newGesture",
                    help="Zu pruefender Datenordner (Standard: trainingsdaten_newGesture)")
args = parser.parse_args()

daten_ordner = args.daten_ordner

if not os.path.isdir(daten_ordner):
    raise RuntimeError(f"Ordner '{daten_ordner}' nicht gefunden.")

gesten_ordner = sorted([
    d for d in os.listdir(daten_ordner)
    if os.path.isdir(os.path.join(daten_ordner, d))
])

alle_shapes_global = defaultdict(int)  # Shape -> Anzahl Dateien insgesamt
fehlerhafte_dateien = []  # (pfad, shape) fuer Dateien, die von der Mehrheit abweichen
ergebnisse_pro_ordner = {}

print(f"Pruefe Shapes in: {daten_ordner}\n")

for geste in gesten_ordner:
    ordner_pfad = os.path.join(daten_ordner, geste)
    npy_dateien = [f for f in os.listdir(ordner_pfad) if f.endswith(".npy")]

    if not npy_dateien:
        print(f"  Hinweis: '{geste}' enthaelt keine .npy-Dateien, wird uebersprungen.")
        continue

    shapes_in_ordner = defaultdict(int)
    dateien_mit_shape = []

    for datei in npy_dateien:
        pfad = os.path.join(ordner_pfad, datei)
        try:
            arr = np.load(pfad)
            shape = arr.shape
        except Exception as e:
            print(f"  FEHLER beim Laden von '{pfad}': {e}")
            continue

        shapes_in_ordner[shape] += 1
        alle_shapes_global[shape] += 1
        dateien_mit_shape.append((pfad, shape))

    ergebnisse_pro_ordner[geste] = shapes_in_ordner

    if len(shapes_in_ordner) == 1:
        einzige_shape = list(shapes_in_ordner.keys())[0]
        print(f"  {geste:25s}: {len(npy_dateien):4d} Dateien, alle Shape {einzige_shape}")
    else:
        print(f"  {geste:25s}: {len(npy_dateien):4d} Dateien, ABWEICHENDE SHAPES gefunden:")
        for shape, anzahl in sorted(shapes_in_ordner.items(), key=lambda x: -x[1]):
            print(f"      {shape}: {anzahl} Datei(en)")
        # Innerhalb des Ordners: Dateien mit der (lokalen) Minderheits-Shape auflisten
        haeufigste_shape_hier = max(shapes_in_ordner.items(), key=lambda x: x[1])[0]
        for pfad, shape in dateien_mit_shape:
            if shape != haeufigste_shape_hier:
                fehlerhafte_dateien.append((pfad, shape))

print(f"\n{'=' * 60}")
print("GESAMTÜBERSICHT ALLER SHAPES IM DATENSATZ")
print(f"{'=' * 60}")
for shape, anzahl in sorted(alle_shapes_global.items(), key=lambda x: -x[1]):
    print(f"  {shape}: {anzahl} Dateien")

if len(alle_shapes_global) == 1:
    einzige_shape = list(alle_shapes_global.keys())[0]
    print(f"\nOK: Alle Dateien in ALLEN Ordnern haben exakt dieselbe Shape {einzige_shape}.")
    print("train_model.py sollte problemlos laufen.")
else:
    print(f"\nWARNUNG: Es gibt {len(alle_shapes_global)} unterschiedliche Shapes im Datensatz!")
    print("train_model.py wird vermutlich mit einem Fehler abbrechen (np.array(X)")
    print("kann keine unterschiedlich geformten Sequenzen zu einem Array zusammenbauen).")
    if fehlerhafte_dateien:
        print("\nAuffaellige Dateien (weichen von der haeufigsten Shape in ihrem Ordner ab):")
        for pfad, shape in fehlerhafte_dateien:
            print(f"  {pfad}  ->  {shape}")
    print("\nTipp: Diese Dateien loeschen (falls versehentlich mit falschen Parametern")
    print("aufgenommen) oder neu aufnehmen/konvertieren, dann dieses Skript erneut ausfuehren.")