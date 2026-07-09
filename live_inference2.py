"""
live_inference2.py
-------------------
Erkennt die trainierten dynamischen Gesten in Echtzeit ueber die Webcam
und loest je nach Geste eine Aktion aus (Tastendruck / Scroll via pyautogui).

Funktionsweise (Sliding Window MIT Rohpuffer-Subsampling):
Es werden fortlaufend die letzten ROH_PUFFER_LAENGE (40) Frames gespeichert
(Rohpuffer). Fuer die Vorhersage wird daraus aber nur JEDES ZWEITE Frame
verwendet (Index 0, 2, 4, ..., 38), was genau SEQUENZ_LAENGE (20) Frames
ergibt -- exakt die Eingabegroesse, die das trainierte Modell erwartet.

Der Trick dahinter: Das ausgewertete 20-Frame-Fenster deckt dadurch den
ZEITRAUM von 40 echten Kamera-Frames ab, also die doppelte Zeitspanne wie
zuvor. Das hilft insbesondere bei Kreisgesten, die oft laenger als 20
Frames zum vollstaendigen Ausfuehren brauchen und sonst als (aehnlich
aussehende) Wischgesten fehlerkannt werden, da sie in einem reinen
20-Frame-Fenster meist nur unvollstaendig erfasst werden.

WICHTIG: Das trainierte Modell selbst wird NICHT veraendert und muss NICHT
neu trainiert werden -- es bekommt weiterhin exakt 20 Frames als Eingabe,
nur eben mit doppeltem zeitlichen Abstand zwischen den Frames.

Voraussetzung: train_model0.py wurde bereits ausgefuehrt. Dieses erzeugt pro
Lauf einen Vergleichs-Batch 'modelle/vergleich_XXX/' mit mehreren trainierten
Architekturen (z.B. gru_klein, lstm_mittel, cnn_gross) sowie einer Datei
'bestes_modell.txt', die angibt, welche Architektur die hoechste
Test-Accuracy erreicht hat.

Aufruf (nutzt automatisch das EMPFOHLENE Modell aus dem NEUESTEN Batch):
    python live_inference2.py

Aufruf mit einer bestimmten Architektur / einem bestimmten Batch:
    python live_inference2.py --model modelle/vergleich_003/cnn_gross

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
SEQUENZ_LAENGE = 20           # Erwartete Eingabegroesse des trainierten Modells (NICHT aendern,
                              # ausser das Modell wurde mit anderer Laenge neu trainiert!)
SUBSAMPLE_SCHRITT = 2         # Jedes 2. Frame aus dem Rohpuffer wird verwendet
ROH_PUFFER_LAENGE = SEQUENZ_LAENGE * SUBSAMPLE_SCHRITT  # 40 rohe Kamera-Frames
KONFIDENZ_SCHWELLE = 0.85     # Nur Aktionen ausloesen, wenn Modell sich sicher genug ist
COOLDOWN_SEKUNDEN = 1.0       # Mindestabstand zwischen zwei ausgeloesten Aktionen
VORHERSAGE_INTERVALL = 1      # Alle wie viele Frames neu vorhergesagt wird.
                              # 1 = jeder Frame (reaktionsschnellste Variante,
                              # aber hoehere CPU-Last). Bei Rucklern z.B. auf
                              # 3-5 erhoehen -> spart Rechenzeit, die Erkennung
                              # bleibt aber weiterhin ein echtes Sliding Window.


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
# 4. Sliding Window Rohpuffer fuer die letzten ROH_PUFFER_LAENGE (40) Frames
# ---------------------------------------------------------------------------
# deque mit maxlen sorgt automatisch dafuer, dass beim Anhaengen eines neuen
# Frames der AELTESTE Frame rausfaellt, sobald der Puffer voll ist -- der
# Puffer enthaelt also IMMER die letzten ROH_PUFFER_LAENGE echten Kamera-
# Frames. Fuer die eigentliche Vorhersage wird daraus weiter unten nur jedes
# SUBSAMPLE_SCHRITT-te Frame entnommen (siehe subsample_puffer()).
frame_buffer = collections.deque(maxlen=ROH_PUFFER_LAENGE)
letzte_aktion_zeit = 0.0
frame_zaehler = 0
letzte_gedruckte_geste = None  # fuer Konsolen-Ausgabe bei Geste-Wechsel


def subsample_puffer(puffer, schritt, ziel_laenge):
    """Nimmt aus dem Rohpuffer jedes 'schritt'-te Frame (0, schritt, 2*schritt, ...)
    und gibt genau 'ziel_laenge' Frames zurueck -- das ist die Eingabegroesse,
    die das trainierte Modell erwartet."""
    liste = list(puffer)
    subsampled = liste[::schritt]
    return subsampled[:ziel_laenge]

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
            # Hand ist komplett verschwunden -> Puffer leeren, damit keine
            # "kaputten" Sequenzen (Hand weg -> wieder da) erkannt werden.
            frame_buffer.clear()

        frame_zaehler += 1

        # Sobald der Rohpuffer voll ist (ROH_PUFFER_LAENGE echte Frames),
        # wird alle VORHERSAGE_INTERVALL Frames neu vorhergesagt. Fuer die
        # Vorhersage selbst wird nur jedes SUBSAMPLE_SCHRITT-te Frame genutzt
        # (siehe subsample_puffer) -> ergibt genau SEQUENZ_LAENGE Frames,
        # die aber den ZEITRAUM von ROH_PUFFER_LAENGE echten Frames abdecken.
        if len(frame_buffer) == ROH_PUFFER_LAENGE and frame_zaehler % VORHERSAGE_INTERVALL == 0:
            subsampled_frames = subsample_puffer(frame_buffer, SUBSAMPLE_SCHRITT, SEQUENZ_LAENGE)
            eingabe = np.expand_dims(np.array(subsampled_frames, dtype=np.float32), axis=0)
            vorhersage = model.predict(eingabe, verbose=0)[0]
            klassen_idx = int(np.argmax(vorhersage))
            konfidenz = float(vorhersage[klassen_idx])
            geste_name = label_map[klassen_idx]

            aktuelle_geste_text = f"{geste_name} ({konfidenz:.2f})"
            aktuelle_konfidenz = konfidenz

            # Konsolen-Ausgabe nur bei Wechsel der erkannten Geste (nicht bei
            # jedem einzelnen Frame), damit die Konsole nicht zuspammt.
            if geste_name != letzte_gedruckte_geste:
                print(f"Erkannt: {geste_name}  (Konfidenz: {konfidenz:.2f})")
                letzte_gedruckte_geste = geste_name

            jetzt = time.time()
            if (konfidenz >= KONFIDENZ_SCHWELLE
                    and (jetzt - letzte_aktion_zeit) >= COOLDOWN_SEKUNDEN):
                fuehre_aktion_aus(geste_name)
                letzte_aktion_zeit = jetzt
                # Puffer NICHT leeren: das Sliding Window laeuft einfach weiter.
                # Der Cooldown allein verhindert, dass dieselbe Geste sofort
                # nochmal (auf demselben Fenster) feuert.

        # --- Anzeige ---
        cv2.putText(frame, f"Rohpuffer: {len(frame_buffer)}/{ROH_PUFFER_LAENGE}", (10, 30),
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
