"""
konvertiere_datensatz.py
--------------------------
Wandelt bestehende Aufnahmen aus trainingsdaten/ (absolute Wrist-Position +
relative Landmarks) in das neue positions-/groessenunabhaengige Format um
(siehe record_gesture_relativ.py) -- OHNE dass neu mit der Kamera
aufgenommen werden muss.

Funktioniert, weil die alten .npy-Dateien bereits exakt die Rohdaten
enthalten, die normalisiere_sequenz() braucht: Form (SEQUENZ_LAENGE, 66)
mit absoluter Wrist-Position in Spalte 0:3 und relativen Landmarks in
Spalte 3:66.

WICHTIG, was diese Konvertierung NICHT loest:
Sie macht die FEATURE-DARSTELLUNG positions-/groessenunabhaengig, erfindet
aber keine neue Vielfalt in den Daten. Wurden alle bisherigen Aufnahmen
frontal und aehnlich gross gemacht, bleibt das nach der Konvertierung so --
das Modell sieht weiterhin nur Beispiele aus einem engen Bereich.
Der Vorteil trotzdem: Die absolute Position ist als Information komplett
aus den Daten entfernt, das Modell KANN sich also nicht mehr (auch nicht
versehentlich) auf absolute Koordinaten verlassen. Fuer echte Robustheit
bei stark abweichenden Positionen/Groessen bleibt zusaetzliches Aufnehmen
mit bewusster Variation (record_gesture_relativ.py) trotzdem sinnvoll --
diese Konvertierung ist aber ein guter, kostenloser erster Schritt, um zu
testen, ob sich die Erkennung dadurch schon verbessert.

Aufruf:
    python konvertiere_datensatz.py
    python konvertiere_datensatz.py --quelle trainingsdaten --ziel trainingsdaten_relativ
"""

import argparse
import os

import numpy as np


def normalisiere_sequenz(sequenz):
    """MUSS exakt identisch zu record_gesture_relativ.py und
    live_inference_relativ.py sein! Macht die Wrist-Trajektorie relativ zum
    ersten Frame und normiert sie auf den groessten Bewegungsradius."""
    sequenz = np.array(sequenz, dtype=np.float32).copy()
    wrist_start = sequenz[0, 0:3].copy()
    delta = sequenz[:, 0:3] - wrist_start
    radius = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
    scale = max(float(radius.max()), 1e-6)
    sequenz[:, 0:3] = delta / scale
    return sequenz


parser = argparse.ArgumentParser(
    description="Konvertiert bestehende Trainingsdaten in das positions-/groessenunabhaengige Format."
)
parser.add_argument("--quelle", type=str, default="trainingsdaten",
                     help="Ordner mit den alten Aufnahmen (Standard: trainingsdaten)")
parser.add_argument("--ziel", type=str, default="trainingsdaten_relativ",
                     help="Zielordner fuer die konvertierten Aufnahmen (Standard: trainingsdaten_relativ)")
args = parser.parse_args()

if not os.path.isdir(args.quelle):
    raise RuntimeError(f"Quellordner '{args.quelle}' nicht gefunden.")

gesten_ordner = sorted([
    d for d in os.listdir(args.quelle)
    if os.path.isdir(os.path.join(args.quelle, d))
])

gesamt_konvertiert = 0
gesamt_uebersprungen = 0

print(f"Konvertiere von '{args.quelle}' nach '{args.ziel}'...\n")

for geste in gesten_ordner:
    quell_ordner = os.path.join(args.quelle, geste)
    npy_dateien = [f for f in os.listdir(quell_ordner) if f.endswith(".npy")]

    if not npy_dateien:
        print(f"  Ueberspringe '{geste}' (keine .npy-Dateien, vermutlich kein Gesten-Ordner, z.B. .idea)")
        continue

    ziel_ordner = os.path.join(args.ziel, geste)
    os.makedirs(ziel_ordner, exist_ok=True)

    konvertiert_hier = 0
    fehler_hier = 0

    for datei in npy_dateien:
        quell_pfad = os.path.join(quell_ordner, datei)
        ziel_pfad = os.path.join(ziel_ordner, datei)

        try:
            sequenz = np.load(quell_pfad)
            if sequenz.ndim != 2 or sequenz.shape[1] != 66:
                print(f"    Warnung: '{quell_pfad}' hat unerwartete Form {sequenz.shape}, wird uebersprungen.")
                fehler_hier += 1
                continue

            sequenz_konvertiert = normalisiere_sequenz(sequenz)
            np.save(ziel_pfad, sequenz_konvertiert)
            konvertiert_hier += 1
        except Exception as e:
            print(f"    Fehler bei '{quell_pfad}': {e}")
            fehler_hier += 1

    zusatz = f", {fehler_hier} uebersprungen" if fehler_hier else ""
    print(f"  {geste}: {konvertiert_hier} konvertiert{zusatz}")
    gesamt_konvertiert += konvertiert_hier
    gesamt_uebersprungen += fehler_hier

zusatz_gesamt = f", {gesamt_uebersprungen} uebersprungen." if gesamt_uebersprungen else "."
print(f"\nFertig. {gesamt_konvertiert} Dateien konvertiert{zusatz_gesamt}")
print(f"Neue Daten liegen in: {args.ziel}")
print("\nNaechster Schritt:")
print(f"  python train_model.py --daten-ordner {args.ziel}")
print("Danach zum Testen: python live_inference_relativ.py")
