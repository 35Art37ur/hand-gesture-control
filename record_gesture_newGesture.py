"""
record_gesture_newGesture.py
------------------------------
Wie record_gesture_relativ.py, aber mit einem zusaetzlichen 67. Feature-Wert
pro Frame fuer die neue Pinch/Zoom-Geste. Gedacht fuer die 4 neuen
Gesten-Klassen:

    wischen_oben, wischen_unten        (vertikales Wischen, wie das
                                         bestehende horizontale Wischen,
                                         nur um 90 Grad gedreht)
    pinch_auf, pinch_zu                (Finger auseinander = reinzoomen,
                                         Finger zusammen = rauszoomen)
    faust                              (Ruhe-/Pause-Geste: geballte Faust
                                         halten -> live_inference_newGesture.py
                                         fuehrt dabei bewusst KEINE Aktion aus)

NEUES FEATURE (Spalte 66, zusaetzlich zu den bisherigen 66 Werten):
Abstand zwischen Daumenspitze (Landmark 4) und Zeigefingerspitze
(Landmark 8), normiert auf die Handgroesse (Referenz: Abstand Handgelenk
zu Mittelfinger-Grundgelenk, Landmark 9). Die Normierung macht den Wert
unabhaengig davon, wie nah die Hand an der Kamera ist.

Die Wrist-Trajektorie (Spalte 0:3) wird wie in record_gesture_relativ.py
positions-/groessenunabhaengig normalisiert. Die 63 Handform-Werte
(Spalte 3:66) bleiben unveraendert. Neu ist nur Spalte 66 (Pinch-Feature).

WICHTIG: Dieses Skript ist bewusst NEU und ersetzt KEINES der bestehenden
Skripte (record_gesture.py, record_gesture_relativ.py bleiben unveraendert
erhalten). Falls sich die neuen Gesten nicht gut trainieren lassen, kannst
du einfach zu den bewaehrten Skripten zurueckkehren.

Aufruf (Beispiel):
    python record_gesture_newGesture.py --geste wischen_oben --samples 100
    python record_gesture_newGesture.py --geste pinch_auf --samples 100

Steuerung im Fenster:
    's' = eine Aufnahme (Sequenz) starten
    'q' = Aufnahme beenden / Programm verlassen
"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request

# --- KONFIGURATION ---
SEQUENZ_LAENGE = 20

parser = argparse.ArgumentParser(description="Nimmt Trainingsdaten fuer neue Gesten auf (vertikales Wischen, Pinch/Zoom).")
parser.add_argument("--geste", type=str, required=True,
                     help="Name der Geste, z.B. wischen_oben, wischen_unten, pinch_auf, pinch_zu")
parser.add_argument("--samples", type=int, default=100,
                     help="Anzahl der aufzunehmenden Sequenzen (Standard: 100)")
parser.add_argument("--seq-laenge", type=int, default=SEQUENZ_LAENGE,
                     help=f"Anzahl Frames pro Sequenz (Standard: {SEQUENZ_LAENGE})")
parser.add_argument("--ausgabe-ordner", type=str, default="trainingsdaten_newGesture",
                     help="Zielordner fuer die Aufnahmen (Standard: trainingsdaten_newGesture)")
args = parser.parse_args()

GESTE_NAME = args.geste
ANZAHL_SAMPLES = args.samples
SEQUENZ_LAENGE = args.seq_laenge
SPEICHER_ORDNER = os.path.join(args.ausgabe_ordner, GESTE_NAME)

os.makedirs(SPEICHER_ORDNER, exist_ok=True)

vorhandene_dateien = [f for f in os.listdir(SPEICHER_ORDNER) if f.endswith(".npy")]
start_index = len(vorhandene_dateien)
if start_index > 0:
    print(f"Es sind bereits {start_index} Samples fuer '{GESTE_NAME}' vorhanden. "
          f"Es wird ab sample_{start_index} weiter aufgenommen.")

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Lade aktuelles MediaPipe Task-Modell herunter...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)


def draw_landmarks_on_image(image, detection_result):
    if not detection_result.hand_landmarks:
        return image
    for hand_landmarks in detection_result.hand_landmarks:
        for lm in hand_landmarks:
            cx, cy = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
            cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)
    return image


def extrahiere_rohdaten_erweitert(hand_landmarks):
    """Wie extrahiere_rohdaten() aus record_gesture_relativ.py (3 absolute
    Wrist-Werte + 21*3 relative Landmarks = 66 Werte), ZUSAETZLICH mit einem
    67. Wert: Pinch-Distanz (Daumenspitze <-> Zeigefingerspitze), normiert
    auf die Handgroesse."""
    wrist = hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    frame_koordinaten = [wrist_x, wrist_y, wrist_z]
    for lm in hand_landmarks:
        rel_x = lm.x - wrist_x
        rel_y = lm.y - wrist_y
        rel_z = lm.z - wrist_z
        frame_koordinaten.extend([rel_x, rel_y, rel_z])

    # Pinch-Feature: Abstand Daumenspitze (Index 4) zu Zeigefingerspitze
    # (Index 8), normiert auf Handgroesse (Abstand Handgelenk zu
    # Mittelfinger-Grundgelenk, Index 9) fuer Kameraentfernungs-Unabhaengigkeit.
    daumen = hand_landmarks[4]
    zeigefinger = hand_landmarks[8]
    mittelfinger_mcp = hand_landmarks[9]

    rel_daumen = np.array([daumen.x - wrist_x, daumen.y - wrist_y, daumen.z - wrist_z])
    rel_zeige = np.array([zeigefinger.x - wrist_x, zeigefinger.y - wrist_y, zeigefinger.z - wrist_z])
    rel_mittel_mcp = np.array([mittelfinger_mcp.x - wrist_x, mittelfinger_mcp.y - wrist_y, mittelfinger_mcp.z - wrist_z])

    pinch_distanz = float(np.linalg.norm(rel_daumen - rel_zeige))
    hand_groesse_referenz = max(float(np.linalg.norm(rel_mittel_mcp)), 1e-6)
    pinch_feature = pinch_distanz / hand_groesse_referenz

    frame_koordinaten.append(pinch_feature)
    return frame_koordinaten


def normalisiere_sequenz(sequenz):
    """Identisch zu record_gesture_relativ.py: macht NUR die ersten 3 Spalten
    (Wrist-Trajektorie) positions-/groessenunabhaengig. Die restlichen Spalten
    (Handform + Pinch-Feature) bleiben unveraendert."""
    sequenz = np.array(sequenz, dtype=np.float32).copy()
    wrist_start = sequenz[0, 0:3].copy()
    delta = sequenz[:, 0:3] - wrist_start
    radius = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
    scale = max(float(radius.max()), 1e-6)
    sequenz[:, 0:3] = delta / scale
    return sequenz


cap = cv2.VideoCapture(0)
sample_zaehler = start_index
ziel = start_index + ANZAHL_SAMPLES

print(f"Bereit zur Aufnahme fuer: '{GESTE_NAME}' (mit Pinch-Feature)")
print(f"Ziel: {ANZAHL_SAMPLES} neue Samples (insgesamt dann {ziel})")
print("Druecke 's' auf der Tastatur, um eine Geste aufzunehmen. 'q' zum Beenden.")

with HandLandmarker.create_from_options(options) as landmarker:
    while sample_zaehler < ziel:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)

        frame = draw_landmarks_on_image(frame, result)

        cv2.putText(frame, f"Geste: {GESTE_NAME}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Aufgenommen: {sample_zaehler}/{ziel}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Druecke 's' fuer Start, 'q' zum Beenden", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Tipp: Position/Groesse im Bild bewusst variieren!", (10, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
        cv2.imshow("Data Recorder (neue Gesten)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            sequenz_daten = []
            frames_aufgenommen = 0
            print(f"Starte Aufnahme von Sample {sample_zaehler + 1}...")

            while frames_aufgenommen < SEQUENZ_LAENGE:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    hand_landmarks = result.hand_landmarks[0]
                    frame_koordinaten = extrahiere_rohdaten_erweitert(hand_landmarks)
                    sequenz_daten.append(frame_koordinaten)
                    frames_aufgenommen += 1
                    frame = draw_landmarks_on_image(frame, result)

                cv2.putText(frame, f"Recording Frame: {frames_aufgenommen}/{SEQUENZ_LAENGE}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Data Recorder (neue Gesten)", frame)
                cv2.waitKey(30)

            if len(sequenz_daten) == SEQUENZ_LAENGE:
                sequenz_normalisiert = normalisiere_sequenz(sequenz_daten)
                datei_name = os.path.join(SPEICHER_ORDNER, f"sample_{sample_zaehler}.npy")
                np.save(datei_name, sequenz_normalisiert)
                sample_zaehler += 1
                print(f"Gespeichert: {datei_name}")
                time.sleep(0.4)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Fertig. {sample_zaehler} Samples fuer '{GESTE_NAME}' vorhanden.")
