"""
record_gesture_relativ.py
--------------------------
Wie record_gesture.py, aber mit einer zusaetzlichen Normalisierung der
Wrist-Trajektorie, damit die Geste UNABHAENGIG von Position und Groesse
im Kamerabild erkannt werden kann.

PROBLEM, DAS HIERMIT GELOEST WIRD:
Bisher wurde pro Frame die ABSOLUTE Wrist-Position im Bild gespeichert
(wrist_x, wrist_y, wrist_z). Das hat zwei Nachteile:
  1. Eine Geste, die z.B. oben rechts statt mittig ausgefuehrt wird, liefert
     ganz andere absolute Koordinatenwerte, die das Modell so nie im
     Training gesehen hat -> schlechtere Erkennung an ungewohnten Positionen.
  2. Eine kleine Geste (wenig Bewegung) und eine grosse Geste (viel Bewegung,
     z.B. Wischen ueber das ganze Bild) erzeugen komplett unterschiedlich
     grosse Zahlenwerte, obwohl es dieselbe Bewegungsform ist.

LOESUNG:
Nach der Aufnahme einer kompletten Sequenz wird die Wrist-Position NICHT
mehr absolut gespeichert, sondern:
  a) relativ zum ERSTEN Frame der Sequenz (= Startpunkt der Geste)
     -> macht die Trajektorie unabhaengig von der Position im Bild
  b) normiert auf den groessten Bewegungsradius innerhalb der Sequenz
     -> macht die Trajektorie unabhaengig von der Groesse/Ausholweite

Die 63 relativen Landmark-Werte (Handform) bleiben unveraendert -- die
waren schon vorher positionsunabhaengig.

WICHTIG: Diese Normalisierung veraendert das Datenformat gegenueber den
alten Aufnahmen! Alte (trainingsdaten/) und neue (trainingsdaten_relativ/)
Daten duerfen NICHT gemischt werden. Nimm mit diesem Skript ALLE 4 Gesten
komplett neu auf, bevor du train_model0.py auf den neuen Ordner ansetzt.
Passendes Live-Skript: live_inference_relativ.py

Aufruf (Beispiel):
    python record_gesture_relativ.py --geste wischen_rechts --samples 100

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

parser = argparse.ArgumentParser(description="Nimmt Trainingsdaten mit positions-/groessenunabhaengiger Normalisierung auf.")
parser.add_argument("--geste", type=str, required=True,
                     help="Name der Geste, z.B. wischen_rechts, wischen_links, "
                          "kreis_uhrzeigersinn, kreis_gegen_uhrzeigersinn")
parser.add_argument("--samples", type=int, default=100,
                     help="Anzahl der aufzunehmenden Sequenzen (Standard: 100)")
parser.add_argument("--seq-laenge", type=int, default=SEQUENZ_LAENGE,
                     help=f"Anzahl Frames pro Sequenz (Standard: {SEQUENZ_LAENGE})")
parser.add_argument("--ausgabe-ordner", type=str, default="trainingsdaten_relativ",
                     help="Zielordner fuer die neuen, normalisierten Aufnahmen "
                          "(Standard: trainingsdaten_relativ, NICHT trainingsdaten!)")
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


def extrahiere_rohdaten(hand_landmarks):
    """Baut den rohen Feature-Vektor pro Frame -- IDENTISCH zur Logik in
    record_gesture.py: 3 absolute Wrist-Werte + 21*3 relative Landmark-Werte.
    Die Normalisierung passiert NICHT hier, sondern erst nach Abschluss der
    kompletten Sequenz in normalisiere_sequenz()."""
    wrist = hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    frame_koordinaten = [wrist_x, wrist_y, wrist_z]
    for lm in hand_landmarks:
        rel_x = lm.x - wrist_x
        rel_y = lm.y - wrist_y
        rel_z = lm.z - wrist_z
        frame_koordinaten.extend([rel_x, rel_y, rel_z])

    return frame_koordinaten


def normalisiere_sequenz(sequenz):
    """Macht die Wrist-Trajektorie positions- und groessenunabhaengig:
      1. Delta zum ERSTEN Frame der Sequenz (statt absoluter Position)
      2. Division durch den groessten Bewegungsradius in der Sequenz

    WICHTIG: Diese Funktion muss in record_gesture_relativ.py UND
    live_inference_relativ.py exakt identisch sein, sonst passen Training
    und Live-Erkennung nicht mehr zusammen!"""
    sequenz = np.array(sequenz, dtype=np.float32).copy()

    wrist_start = sequenz[0, 0:3].copy()
    delta = sequenz[:, 0:3] - wrist_start

    radius = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
    scale = max(float(radius.max()), 1e-6)  # Division durch 0 vermeiden

    sequenz[:, 0:3] = delta / scale
    return sequenz


cap = cv2.VideoCapture(0)
sample_zaehler = start_index
ziel = start_index + ANZAHL_SAMPLES

print(f"Bereit zur Aufnahme fuer: '{GESTE_NAME}' (normalisierte Version)")
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
        cv2.imshow("Data Recorder (relativ, positions-/groessenunabhaengig)", frame)

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
                    frame_koordinaten = extrahiere_rohdaten(hand_landmarks)
                    sequenz_daten.append(frame_koordinaten)
                    frames_aufgenommen += 1
                    frame = draw_landmarks_on_image(frame, result)

                cv2.putText(frame, f"Recording Frame: {frames_aufgenommen}/{SEQUENZ_LAENGE}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Data Recorder (relativ, positions-/groessenunabhaengig)", frame)
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
