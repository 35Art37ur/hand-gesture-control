"""
live_inference_relativ.py
---------------------------
Wie live_inference2.py (Rohpuffer + Subsampling gegen zu kurze Kreisgesten),
zusaetzlich mit derselben Positions-/Groessennormalisierung wie in
record_gesture_relativ.py -- NUR mit Modellen nutzbar, die auf Daten aus
record_gesture_relativ.py trainiert wurden!

Die Normalisierung passiert bei jeder Vorhersage frisch auf dem AKTUELLEN
Sliding-Window-Ausschnitt: der erste Frame im (subgesampelten) Fenster gilt
als Startpunkt, alle Wrist-Positionen werden relativ dazu berechnet und auf
den groessten Bewegungsradius im Fenster normiert. Dadurch ist es egal, wo
im Bild und wie gross die Geste ausgefuehrt wird.

Voraussetzung: train_model0.py wurde mit Daten aus trainingsdaten_relativ/
(aufgenommen mit record_gesture_relativ.py) ausgefuehrt.

Aufruf (nutzt automatisch das empfohlene Modell des neuesten Batches):
    python live_inference_relativ.py

Aufruf mit einer bestimmten Architektur / einem bestimmten Batch:
    python live_inference_relativ.py --model modelle/vergleich_003/cnn_gross

Steuerung:
    'q' = Beenden
"""

import argparse
import collections
import json
import os
import re
import time

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from tensorflow.keras.models import load_model

# --- KONFIGURATION ---
MODELLE_BASIS_ORDNER = "modelle"
MODEL_PATH = "hand_landmarker.task"
SEQUENZ_LAENGE = 20           # Erwartete Eingabegroesse des trainierten Modells
SUBSAMPLE_SCHRITT = 2         # Jedes 2. Frame aus dem Rohpuffer wird verwendet
ROH_PUFFER_LAENGE = SEQUENZ_LAENGE * SUBSAMPLE_SCHRITT  # 40 rohe Kamera-Frames
KONFIDENZ_SCHWELLE = 0.85
COOLDOWN_SEKUNDEN = 1.0
VORHERSAGE_INTERVALL = 1


def neuester_batch_ordner(basis_ordner):
    kandidaten = []
    for name in os.listdir(basis_ordner):
        pfad = os.path.join(basis_ordner, name)
        if os.path.isdir(pfad):
            match = re.fullmatch(r"vergleich_(\d+)", name)
            if match:
                kandidaten.append((int(match.group(1)), pfad))
    if not kandidaten:
        raise RuntimeError(
            f"Kein Vergleichs-Batch in '{basis_ordner}/vergleich_XXX' gefunden. "
            f"Bitte zuerst train_model0.py ausfuehren."
        )
    kandidaten.sort(key=lambda x: x[0])
    return kandidaten[-1][1]


def ermittele_modell_ordner(explizit_angegeben, basis_ordner):
    if explizit_angegeben:
        return explizit_angegeben
    batch_ordner = neuester_batch_ordner(basis_ordner)
    marker_datei = os.path.join(batch_ordner, "bestes_modell.txt")
    if os.path.exists(marker_datei):
        with open(marker_datei, "r") as f:
            beste_architektur = f.read().strip()
        return os.path.join(batch_ordner, beste_architektur)
    unterordner = sorted([
        d for d in os.listdir(batch_ordner)
        if os.path.isdir(os.path.join(batch_ordner, d))
    ])
    if not unterordner:
        raise RuntimeError(f"Kein Architektur-Unterordner in '{batch_ordner}' gefunden.")
    return os.path.join(batch_ordner, unterordner[0])


parser = argparse.ArgumentParser(description="Live-Erkennung mit positions-/groessenunabhaengiger Normalisierung.")
parser.add_argument("--model", type=str, default=None,
                     help="Pfad zu einem bestimmten Architektur-Ordner, z.B. "
                          "modelle/vergleich_003/cnn_gross. Ohne Angabe wird "
                          "automatisch das empfohlene Modell des neuesten "
                          "Vergleichs-Batches genutzt.")
args = parser.parse_args()

MODELL_ORDNER = ermittele_modell_ordner(args.model, MODELLE_BASIS_ORDNER)
MODELL_DATEI = os.path.join(MODELL_ORDNER, "gesture_model.h5")
LABEL_MAP_DATEI = os.path.join(MODELL_ORDNER, "label_map.json")

# ---------------------------------------------------------------------------
# 1. Modell + Label-Map laden
# ---------------------------------------------------------------------------
print(f"Lade Modell aus: {MODELL_ORDNER}")
model = load_model(MODELL_DATEI)
with open(LABEL_MAP_DATEI, "r") as f:
    label_map_raw = json.load(f)
label_map = {int(k): v for k, v in label_map_raw.items()}
print(f"Geladene Gesten: {label_map}")


# ---------------------------------------------------------------------------
# 2. Aktionen pro Geste definieren
# ---------------------------------------------------------------------------
def fuehre_aktion_aus(geste_name):
    if geste_name == "wischen_rechts":
        print(">>> Aktion: Weiter (rechts)")
        pyautogui.press("right")
    elif geste_name == "wischen_links":
        print(">>> Aktion: Zurueck (links)")
        pyautogui.press("left")
    elif geste_name == "kreis_uhrzeigersinn":
        print(">>> Aktion: Scroll runter")
        pyautogui.scroll(-300)
    elif geste_name == "kreis_gegen_uhrzeigersinn":
        print(">>> Aktion: Scroll hoch")
        pyautogui.scroll(300)
    else:
        print(f">>> Unbekannte Geste: {geste_name} (keine Aktion definiert)")


# ---------------------------------------------------------------------------
# 3. MediaPipe initialisieren
# ---------------------------------------------------------------------------
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
    """IDENTISCH zu record_gesture_relativ.py: absolute Wrist-Position +
    relative Landmarks. Die Normalisierung erfolgt separat in
    normalisiere_sequenz(), da sie den gesamten Fensterinhalt braucht."""
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
    """MUSS exakt identisch zu record_gesture_relativ.py sein! Macht die
    Wrist-Trajektorie relativ zum ersten Frame im Fenster und normiert sie
    auf den groessten Bewegungsradius -> positions- und groessenunabhaengig."""
    sequenz = np.array(sequenz, dtype=np.float32).copy()
    wrist_start = sequenz[0, 0:3].copy()
    delta = sequenz[:, 0:3] - wrist_start
    radius = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
    scale = max(float(radius.max()), 1e-6)
    sequenz[:, 0:3] = delta / scale
    return sequenz


def subsample_puffer(puffer, schritt, ziel_laenge):
    """Nimmt aus dem Rohpuffer jedes 'schritt'-te Frame -- deckt dadurch den
    Zeitraum von ROH_PUFFER_LAENGE echten Frames ab (siehe live_inference2.py
    fuer die Begruendung: hilft bei zu lang dauernden Kreisgesten)."""
    liste = list(puffer)
    subsampled = liste[::schritt]
    return subsampled[:ziel_laenge]


# ---------------------------------------------------------------------------
# 4. Sliding Window Rohpuffer + Normalisierung bei jeder Vorhersage
# ---------------------------------------------------------------------------
frame_buffer = collections.deque(maxlen=ROH_PUFFER_LAENGE)
letzte_aktion_zeit = 0.0
frame_zaehler = 0
letzte_gedruckte_geste = None

cap = cv2.VideoCapture(0)
print("Live-Erkennung gestartet (positions-/groessenunabhaengig). Druecke 'q' zum Beenden.")

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect(mp_image)

        aktuelle_geste_text = ""
        aktuelle_konfidenz = 0.0

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            frame_koordinaten = extrahiere_rohdaten(hand_landmarks)
            frame_buffer.append(frame_koordinaten)
            frame = draw_landmarks_on_image(frame, result)
        else:
            frame_buffer.clear()

        frame_zaehler += 1

        if len(frame_buffer) == ROH_PUFFER_LAENGE and frame_zaehler % VORHERSAGE_INTERVALL == 0:
            subsampled_frames = subsample_puffer(frame_buffer, SUBSAMPLE_SCHRITT, SEQUENZ_LAENGE)
            subsampled_normalisiert = normalisiere_sequenz(subsampled_frames)
            eingabe = np.expand_dims(subsampled_normalisiert, axis=0)
            vorhersage = model.predict(eingabe, verbose=0)[0]
            klassen_idx = int(np.argmax(vorhersage))
            konfidenz = float(vorhersage[klassen_idx])
            geste_name = label_map[klassen_idx]

            aktuelle_geste_text = f"{geste_name} ({konfidenz:.2f})"
            aktuelle_konfidenz = konfidenz

            if geste_name != letzte_gedruckte_geste:
                print(f"Erkannt: {geste_name}  (Konfidenz: {konfidenz:.2f})")
                letzte_gedruckte_geste = geste_name

            jetzt = time.time()
            if (konfidenz >= KONFIDENZ_SCHWELLE
                    and (jetzt - letzte_aktion_zeit) >= COOLDOWN_SEKUNDEN):
                fuehre_aktion_aus(geste_name)
                letzte_aktion_zeit = jetzt

        cv2.putText(frame, f"Rohpuffer: {len(frame_buffer)}/{ROH_PUFFER_LAENGE}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if aktuelle_geste_text:
            farbe = (0, 255, 0) if aktuelle_konfidenz >= KONFIDENZ_SCHWELLE else (0, 165, 255)
            cv2.putText(frame, aktuelle_geste_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, farbe, 2)
        cv2.putText(frame, "Druecke 'q' zum Beenden", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Live Gesture Recognition (relativ)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
