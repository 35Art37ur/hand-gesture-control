"""
konvertiere_datensatz_newGesture.py
-------------------------------------
Erweitert bestehende Aufnahmen (aus trainingsdaten/ oder trainingsdaten_relativ/)
um das 67. Feature (Pinch-Distanz), damit die BESTEHENDEN 4 Gesten zusammen
mit den NEUEN Gesten (wischen_oben, wischen_unten, pinch_auf, pinch_zu) in
einem gemeinsamen Modell trainiert werden koennen -- OHNE die alten Gesten
neu aufnehmen zu muessen.

Funktioniert, weil die Pinch-Distanz (Daumenspitze <-> Zeigefingerspitze)
bereits aus den vorhandenen 63 relativen Landmark-Werten berechnet werden
kann -- die Rohdaten dafuer sind schon da.

Die Wrist-Normalisierung (Spalte 0:3) wird IMMER angewendet, egal ob die
Quelle die alten absoluten Werte (trainingsdaten/) oder die bereits
normalisierten Werte (trainingsdaten_relativ/) enthaelt. Das ist unbedenklich,
da die Normalisierung idempotent ist (zweifache Anwendung aendert nichts
mehr an bereits normalisierten Daten).

Aufruf:
    python konvertiere_datensatz_newGesture.py
    python konvertiere_datensatz_newGesture.py --quelle trainingsdaten_relativ --ziel trainingsdaten_newGesture
"""

import argparse
import os

import numpy as np


def normalisiere_sequenz(sequenz):
    """Identisch zu record_gesture_relativ.py / record_gesture_newGesture.py."""
    sequenz = np.array(sequenz, dtype=np.float32).copy()
    wrist_start = sequenz[0, 0:3].copy()
    delta = sequenz[:, 0:3] - wrist_start
    radius = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
    scale = max(float(radius.max()), 1e-6)
    sequenz[:, 0:3] = delta / scale
    return sequenz


def berechne_pinch_spalte(sequenz_66):
    """Berechnet die Pinch-Distanz (Daumenspitze <-> Zeigefingerspitze,
    normiert auf die Handgroesse) aus den bereits vorhandenen relativen
    Landmark-Spalten. Spaltenindizes basierend auf der Reihenfolge in
    extrahiere_rohdaten(): Landmark i liegt bei Spalte 3 + i*3.
      Landmark 4 (Daumenspitze):        Spalten 15:18
      Landmark 8 (Zeigefingerspitze):   Spalten 27:30
      Landmark 9 (Mittelfinger-MCP):    Spalten 30:33
    """
    rel_daumen = sequenz_66[:, 15:18]
    rel_zeige = sequenz_66[:, 27:30]
    rel_mittel_mcp = sequenz_66[:, 30:33]

    pinch_distanz = np.linalg.norm(rel_daumen - rel_zeige, axis=1)
    hand_groesse_referenz = np.maximum(np.linalg.norm(rel_mittel_mcp, axis=1), 1e-6)

    return (pinch_distanz / hand_groesse_referenz).astype(np.float32)


parser = argparse.ArgumentParser(
    description="Erweitert bestehende Gesten-Aufnahmen um das Pinch-Feature (67. Spalte)."
)
parser.add_argument("--quelle", type=str, default="trainingsdaten_relativ",
                     help="Ordner mit den bestehenden Aufnahmen (Standard: trainingsdaten_relativ)")
parser.add_argument("--ziel", type=str, default="trainingsdaten_newGesture",
                     help="Zielordner fuer die erweiterten Aufnahmen (Standard: trainingsdaten_newGesture)")
args = parser.parse_args()

if not os.path.isdir(args.quelle):
    raise RuntimeError(f"Quellordner '{args.quelle}' nicht gefunden.")

gesten_ordner = sorted([
    d for d in os.listdir(args.quelle)
    if os.path.isdir(os.path.join(args.quelle, d))
])

gesamt_konvertiert = 0
gesamt_uebersprungen = 0

print(f"Erweitere Daten von '{args.quelle}' nach '{args.ziel}' (+ Pinch-Feature)...\n")

for geste in gesten_ordner:
    quell_ordner = os.path.join(args.quelle, geste)
    npy_dateien = [f for f in os.listdir(quell_ordner) if f.endswith(".npy")]

    if not npy_dateien:
        print(f"  Ueberspringe '{geste}' (keine .npy-Dateien)")
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

            sequenz_normalisiert = normalisiere_sequenz(sequenz)
            pinch_spalte = berechne_pinch_spalte(sequenz_normalisiert)
            sequenz_erweitert = np.concatenate(
                [sequenz_normalisiert, pinch_spalte[:, None]], axis=1
            )  # (N, 66) + (N, 1) -> (N, 67)

            np.save(ziel_pfad, sequenz_erweitert)
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
print(f"Erweiterte Daten liegen in: {args.ziel}")
print("\nNaechste Schritte:")
print("  1. Neue Gesten aufnehmen, direkt in denselben Ordner:")
print(f"     python record_gesture_newGesture.py --geste wischen_oben --ausgabe-ordner {args.ziel}")
print(f"     python record_gesture_newGesture.py --geste wischen_unten --ausgabe-ordner {args.ziel}")
print(f"     python record_gesture_newGesture.py --geste pinch_auf --ausgabe-ordner {args.ziel}")
print(f"     python record_gesture_newGesture.py --geste pinch_zu --ausgabe-ordner {args.ziel}")
print(f"  2. Trainieren: python train_model.py --daten-ordner {args.ziel}")
print("  3. Testen: python live_inference_newGesture.py")
