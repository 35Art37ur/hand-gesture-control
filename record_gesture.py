"""
record_gesture.py
------------------
Nimmt Trainingsdaten fuer EINE dynamische Geste auf.

Aufruf (Beispiel):
    python record_gesture.py --geste wischen_rechts --samples 100

Steuerung im Fenster:
    's' = eine Aufnahme (Sequenz) starten
    'q' = Aufnahme beenden / Programm verlassen

WICHTIG: Fuehre dieses Skript fuer JEDE der 4 Gesten separat aus:
    python record_gesture.py --geste wischen_rechts
    python record_gesture.py --geste wischen_links
    python record_gesture.py --geste kreis_uhrzeigersinn


"""

import argparse
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import urllib.request

# --- KONFIGURATION ---
SEQUENZ_LAENGE = 20  # Wie viele Frames hat eine Geste?

parser = argparse.ArgumentParser(description="Nimmt Trainingsdaten fuer eine dynamische Handgeste auf.")
parser.add_argument("--geste", type=str, required=True,
                     help="Name der Geste, z.B. wischen_rechts, wischen_links, "
                          "kreis_uhrzeigersinn, kreis_gegen_uhrzeigersinn")
parser.add_argument("--samples", type=int, default=100,
                     help="Anzahl der aufzunehmenden Sequenzen (Standard: 100)")
parser.add_argument("--seq-laenge", type=int, default=SEQUENZ_LAENGE,
                     help=f"Anzahl Frames pro Sequenz (Standard: {SEQUENZ_LAENGE})")
args = parser.parse_args()

GESTE_NAME = args.geste
ANZAHL_SAMPLES = args.samples
SEQUENZ_LAENGE = args.seq_laenge
SPEICHER_ORDNER = os.path.join("trainingsdaten", GESTE_NAME)

os.makedirs(SPEICHER_ORDNER, exist_ok=True)

# Bereits vorhandene Samples zaehlen, damit wir nicht ueberschreiben,
# falls die Aufnahme in mehreren Sitzungen stattfindet
vorhandene_dateien = [f for f in os.listdir(SPEICHER_ORDNER) if f.endswith(".npy")]
start_index = len(vorhandene_dateien)
if start_index > 0:
    print(f"Es sind bereits {start_index} Samples fuer '{GESTE_NAME}' vorhanden. "
          f"Es wird ab sample_{start_index} weiter aufgenommen.")

# 1. Das .task Modell herunterladen, falls nicht vorhanden
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Lade aktuelles MediaPipe Task-Modell herunter...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

# 2. MediaPipe Tasks initialisieren
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


def extrahiere_features(hand_landmarks):
    """
    Baut einen Feature-Vektor pro Frame:
      - 3 Werte: absolute Wrist-Position (x, y, z) -> wichtig fuer Wisch-/Kreis-Trajektorie
      - 21 * 3 Werte: relative Position jedes Landmarks zur Wrist -> wichtig fuer Handform
    => 66 Werte pro Frame insgesamt
    """
    wrist = hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    frame_koordinaten = [wrist_x, wrist_y, wrist_z]  # absolute Wrist-Position (fuer Trajektorie)

    for lm in hand_landmarks:
        rel_x = lm.x - wrist_x
        rel_y = lm.y - wrist_y
        rel_z = lm.z - wrist_z
        frame_koordinaten.extend([rel_x, rel_y, rel_z])

    return frame_koordinaten


# Webcam starten
cap = cv2.VideoCapture(0)
sample_zaehler = start_index
ziel = start_index + ANZAHL_SAMPLES

print(f"Bereit zur Aufnahme fuer: '{GESTE_NAME}'")
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
        cv2.imshow("Data Recorder (Tasks API)", frame)

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
                    frame_koordinaten = extrahiere_features(hand_landmarks)
                    sequenz_daten.append(frame_koordinaten)
                    frames_aufgenommen += 1
                    frame = draw_landmarks_on_image(frame, result)
                # Wenn keine Hand erkannt wird, wird der Frame uebersprungen
                # (frames_aufgenommen erhoeht sich nicht) -> es wird einfach weiter versucht

                cv2.putText(frame, f"Recording Frame: {frames_aufgenommen}/{SEQUENZ_LAENGE}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Data Recorder (Tasks API)", frame)
                cv2.waitKey(30)

            if len(sequenz_daten) == SEQUENZ_LAENGE:
                datei_name = os.path.join(SPEICHER_ORDNER, f"sample_{sample_zaehler}.npy")
                np.save(datei_name, np.array(sequenz_daten, dtype=np.float32))
                sample_zaehler += 1
                print(f"Gespeichert: {datei_name}")
                time.sleep(0.4)

        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print(f"Fertig. {sample_zaehler} Samples fuer '{GESTE_NAME}' vorhanden.")
