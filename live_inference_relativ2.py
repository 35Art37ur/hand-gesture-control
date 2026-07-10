"""
live_inference_relativ2.py
-----------------------------
Wie live_inference_relativ.py (Positions-/Groessennormalisierung, KEIN
Pinch-Feature, funktioniert mit Modellen aus trainingsdaten_relativ/ --
also den urspruenglichen 4 Gesten: wischen_rechts, wischen_links,
kreis_uhrzeigersinn, kreis_gegen_uhrzeigersinn).

NEU gegenueber live_inference_relativ.py -- zwei Verbesserungen aus
live_inference_newGesture.py, aber OHNE die neuen Gesten/das Pinch-Feature:

1. Aktivierungs-Check "Hand ueber Handgelenk": Es wird nur erkannt, wenn
   die Fingerspitzen im Bild ueber (kleineres y als) dem Handgelenk liegen.
   Haengt die Hand einfach herunter, wird NICHTS ausgewertet. Reine
   Geometrie-Regel, kein Training noetig.

2. Frame-Schema mit "Anlauf-Puffer": 44 rohe Kamera-Frames werden
   gesammelt, die ERSTEN 4 davon verworfen (Anlaufphase), von den
   verbleibenden 40 wird jedes zweite Frame genutzt -> exakt die 20
   Frames, die das Modell erwartet.

3. Cooldown auf 2 Sekunden erhoeht, damit genug Zeit bleibt, die Haende
   nach einer erkannten Geste wieder in Ruhe zu senken.

Voraussetzung: train_model.py wurde mit Daten aus trainingsdaten_relativ/
ausgefuehrt (66 Feature-Werte pro Frame, KEIN Pinch-Feature).

Aufruf (nutzt automatisch das empfohlene Modell des neuesten Batches):
    python live_inference_relativ2.py

Aufruf mit einer bestimmten Architektur / einem bestimmten Batch:
    python live_inference_relativ2.py --model modelle/vergleich_003/cnn_gross

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
SEQUENZ_LAENGE = 20           # Erwartete Eingabegroesse des trainierten Modells (fix)
VERWERFE_ERSTE_FRAMES = 4     # "Anlauf-Frames" nach Puffer-Beginn, die verworfen werden
SUBSAMPLE_SCHRITT = 2         # Von den verbleibenden Frames wird jedes 2. genutzt
# 44 roh -> 4 verwerfen -> 40 uebrig -> jedes 2. -> 20 (SEQUENZ_LAENGE)
ROH_PUFFER_LAENGE = VERWERFE_ERSTE_FRAMES + SEQUENZ_LAENGE * SUBSAMPLE_SCHRITT
KONFIDENZ_SCHWELLE = 0.85
COOLDOWN_SEKUNDEN = 2.0       # 2s Pause nach einer erkannten Geste
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
            f"Bitte zuerst train_model.py ausfuehren."
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


parser = argparse.ArgumentParser(description="Live-Erkennung (positions-/groessenunabhaengig, mit Ruheposition + Anlauf-Puffer).")
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
    """IDENTISCH zu record_gesture_relativ.py: absolute Wrist-Position (wird
    spaeter normalisiert) + relative Landmarks. KEIN Pinch-Feature -> 66 Werte."""
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
    """MUSS identisch zu record_gesture_relativ.py sein."""
    sequenz = np.array(sequenz, dtype=np.float32).copy()
    wrist_start = sequenz[0, 0:3].copy()
    delta = sequenz[:, 0:3] - wrist_start
    radius = np.sqrt(delta[:, 0] ** 2 + delta[:, 1] ** 2 + delta[:, 2] ** 2)
    scale = max(float(radius.max()), 1e-6)
    sequenz[:, 0:3] = delta / scale
    return sequenz


def subsample_puffer(puffer, verwerfe_erste, schritt, ziel_laenge):
    """Verwirft die ersten 'verwerfe_erste' Frames (Anlaufphase), nutzt von
    den restlichen jedes 'schritt'-te Frame -- ergibt exakt 'ziel_laenge'
    Frames fuer das Modell."""
    liste = list(puffer)[verwerfe_erste:]
    subsampled = liste[::schritt]
    return subsampled[:ziel_laenge]


def ist_hand_aktiv(hand_landmarks):
    """Prueft, ob die Hand oberhalb des Handgelenks gehalten wird (typische
    'ich fuehre gerade eine Geste aus'-Haltung) statt einfach herunter-
    zuhaengen. y=0 ist oben im Bild, y=1 unten -- die Hand gilt als aktiv,
    wenn die Fingerspitzen im Schnitt oberhalb (kleineres y) des
    Handgelenks liegen. Reine Geometrie-Regel, kein Training noetig."""
    wrist = hand_landmarks[0]
    fingerspitzen_indices = [4, 8, 12, 16, 20]
    durchschnitt_y = sum(hand_landmarks[i].y for i in fingerspitzen_indices) / len(fingerspitzen_indices)
    return durchschnitt_y < wrist.y


# ---------------------------------------------------------------------------
# 4. Sliding Window Rohpuffer + Normalisierung + Aktivierungs-Check
# ---------------------------------------------------------------------------
frame_buffer = collections.deque(maxlen=ROH_PUFFER_LAENGE)
letzte_aktion_zeit = 0.0
frame_zaehler = 0
letzte_gedruckte_geste = None

cap = cv2.VideoCapture(0)
print("Live-Erkennung gestartet (relativ, mit Ruheposition + Anlauf-Puffer). Druecke 'q' zum Beenden.")

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
        hand_in_ruheposition = False

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]
            if ist_hand_aktiv(hand_landmarks):
                frame_koordinaten = extrahiere_rohdaten(hand_landmarks)
                frame_buffer.append(frame_koordinaten)
            else:
                hand_in_ruheposition = True
                frame_buffer.clear()
            frame = draw_landmarks_on_image(frame, result)
        else:
            frame_buffer.clear()

        frame_zaehler += 1

        if len(frame_buffer) == ROH_PUFFER_LAENGE and frame_zaehler % VORHERSAGE_INTERVALL == 0:
            subsampled_frames = subsample_puffer(frame_buffer, VERWERFE_ERSTE_FRAMES, SUBSAMPLE_SCHRITT, SEQUENZ_LAENGE)
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
        if hand_in_ruheposition:
            cv2.putText(frame, "Ruheposition - keine Erkennung", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        elif aktuelle_geste_text:
            farbe = (0, 255, 0) if aktuelle_konfidenz >= KONFIDENZ_SCHWELLE else (0, 165, 255)
            cv2.putText(frame, aktuelle_geste_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, farbe, 2)
        cv2.putText(frame, "Druecke 'q' zum Beenden", (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Live Gesture Recognition (relativ, v2)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()