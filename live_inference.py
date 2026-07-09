"""
live_inference.py
------------------
Erkennt die trainierten dynamischen Gesten in Echtzeit ueber die Webcam
und loest je nach Geste eine Aktion aus (Tastendruck / Scroll via pyautogui).

Funktionsweise: Sobald 20 Frames mit erkannter Hand gesammelt wurden, wird
EINMALIG vorhergesagt und der Puffer danach komplett geleert. Das kann
problematisch sein, wenn der erste gespeicherte Frame zufaellig schon
mitten in einer Handbewegung liegt (z.B. weil die Hand genau zu diesem
Zeitpunkt hochgehalten wurde) -- siehe live_inference2.py fuer eine
Variante mit echtem Sliding Window, die dieses Problem behebt.

Voraussetzung: train_model0.py wurde bereits ausgefuehrt. Dieses erzeugt pro
Lauf einen Vergleichs-Batch 'modelle/vergleich_XXX/' mit mehreren trainierten
Architekturen (z.B. gru_klein, lstm_mittel, cnn_gross) sowie einer Datei
'bestes_modell.txt', die angibt, welche Architektur die hoechste
Test-Accuracy erreicht hat.

Aufruf (nutzt automatisch das EMPFOHLENE Modell aus dem NEUESTEN Batch):
    python live_inference.py

Aufruf mit einer bestimmten Architektur / einem bestimmten Batch:
    python live_inference.py --model modelle/vergleich_003/cnn_gross

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
SEQUENZ_LAENGE = 20
KONFIDENZ_SCHWELLE = 0.85     # Nur Aktionen ausloesen, wenn Modell sich sicher genug ist
COOLDOWN_SEKUNDEN = 1.0       # Mindestabstand zwischen zwei ausgeloesten Aktionen


def neuester_batch_ordner(basis_ordner):
    """Findet automatisch den Vergleichs-Batch mit der hoechsten Nummer
    (vergleich_001, vergleich_002, ...), also den zuletzt trainierten Batch."""
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
    return kandidaten[-1][1]  # Pfad mit der hoechsten Nummer


def ermittele_modell_ordner(explizit_angegeben, basis_ordner):
    """Falls --model gesetzt wurde, wird genau dieser Pfad genutzt. Sonst wird
    automatisch der neueste Batch geoeffnet und dort das per 'bestes_modell.txt'
    empfohlene (= Architektur mit hoechster Test-Accuracy) Modell verwendet."""
    if explizit_angegeben:
        return explizit_angegeben

    batch_ordner = neuester_batch_ordner(basis_ordner)
    marker_datei = os.path.join(batch_ordner, "bestes_modell.txt")

    if os.path.exists(marker_datei):
        with open(marker_datei, "r") as f:
            beste_architektur = f.read().strip()
        return os.path.join(batch_ordner, beste_architektur)

    # Fallback, falls kein Marker vorhanden ist (z.B. Batch von Hand angelegt):
    # einfach den ersten gefundenen Architektur-Unterordner nehmen.
    unterordner = sorted([
        d for d in os.listdir(batch_ordner)
        if os.path.isdir(os.path.join(batch_ordner, d))
    ])
    if not unterordner:
        raise RuntimeError(f"Kein Architektur-Unterordner in '{batch_ordner}' gefunden.")
    return os.path.join(batch_ordner, unterordner[0])


parser = argparse.ArgumentParser(description="Live-Erkennung dynamischer Handgesten.")
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
# JSON-Keys sind Strings -> zurueck zu int
label_map = {int(k): v for k, v in label_map_raw.items()}
print(f"Geladene Gesten: {label_map}")

# ---------------------------------------------------------------------------
# 2. Aktionen pro Geste definieren
# ---------------------------------------------------------------------------
# HIER ANPASSEN je nachdem, wie deine Ordner/Gesten genau heissen!
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


def extrahiere_features(hand_landmarks):
    """Muss EXAKT dieselbe Logik wie in record_gesture.py verwenden!"""
    wrist = hand_landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    frame_koordinaten = [wrist_x, wrist_y, wrist_z]
    for lm in hand_landmarks:
        rel_x = lm.x - wrist_x
        rel_y = lm.y - wrist_y
        rel_z = lm.z - wrist_z
        frame_koordinaten.extend([rel_x, rel_y, rel_z])

    return frame_koordinaten


# ---------------------------------------------------------------------------
# 4. Puffer fuer die letzten N Frames (Einmal-Aufnahme, kein Sliding Window)
# ---------------------------------------------------------------------------
frame_buffer = collections.deque(maxlen=SEQUENZ_LAENGE)
letzte_aktion_zeit = 0.0

cap = cv2.VideoCapture(0)
print("Live-Erkennung gestartet. Druecke 'q' zum Beenden.")

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
            frame_koordinaten = extrahiere_features(hand_landmarks)
            frame_buffer.append(frame_koordinaten)
            frame = draw_landmarks_on_image(frame, result)
        else:
            # Optional: Buffer leeren, wenn Hand verschwindet, damit keine
            # "kaputten" Sequenzen (Hand weg -> wieder da) erkannt werden.
            frame_buffer.clear()

        # Sobald der Buffer voll ist, Vorhersage machen
        if len(frame_buffer) == SEQUENZ_LAENGE:
            eingabe = np.expand_dims(np.array(frame_buffer, dtype=np.float32), axis=0)
            vorhersage = model.predict(eingabe, verbose=0)[0]
            klassen_idx = int(np.argmax(vorhersage))
            konfidenz = float(vorhersage[klassen_idx])
            geste_name = label_map[klassen_idx]

            aktuelle_geste_text = f"{geste_name} ({konfidenz:.2f})"
            aktuelle_konfidenz = konfidenz

            print(f"Erkannt: {geste_name}  (Konfidenz: {konfidenz:.2f})")

            jetzt = time.time()
            if (konfidenz >= KONFIDENZ_SCHWELLE
                    and (jetzt - letzte_aktion_zeit) >= COOLDOWN_SEKUNDEN):
                fuehre_aktion_aus(geste_name)
                letzte_aktion_zeit = jetzt
                frame_buffer.clear()  # Buffer leeren, damit dieselbe Geste nicht doppelt feuert

        # --- Anzeige ---
        cv2.putText(frame, f"Buffer: {len(frame_buffer)}/{SEQUENZ_LAENGE}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if aktuelle_geste_text:
            farbe = (0, 255, 0) if aktuelle_konfidenz >= KONFIDENZ_SCHWELLE else (0, 165, 255)
            cv2.putText(frame, aktuelle_geste_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, farbe, 2)
        cv2.putText(frame, "Druecke 'q' zum Beenden", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Live Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
