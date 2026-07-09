# Dynamische Handgesten-Steuerung (LSTM + MediaPipe)

Erweiterung des "AI-based real-time hand gesture-controlled virtual mouse"-Ansatzes
um **dynamische** Gesten (Wischen, Kreisen), erkannt über ein LSTM auf
Zeitreihen von Hand-Landmarks.

## Projektstruktur

```
gesture_project/
├── record_gesture.py     # Schritt 1: Trainingsdaten aufnehmen
├── train_model.py        # Schritt 2: LSTM trainieren
├── live_inference.py     # Schritt 3: Live-Erkennung + Mausaktionen
├── requirements.txt
├── hand_landmarker.task   # wird automatisch heruntergeladen
├── trainingsdaten/        # wird automatisch erzeugt
│   ├── wischen_rechts/
│   ├── wischen_links/
│   ├── kreis_uhrzeigersinn/
│   └── kreis_gegen_uhrzeigersinn/
├── gesture_model.h5       # Ergebnis von train_model.py
└── label_map.json         # Ergebnis von train_model.py
```

## 0. Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Trainingsdaten sammeln — `record_gesture.py`

Für **jede der 4 Gesten separat** ausführen. Das Skript legt automatisch
einen Unterordner unter `trainingsdaten/` an und speichert dort
`sample_0.npy`, `sample_1.npy`, ... (je eine Datei pro aufgenommener
Geste, Shape `(20, 66)`).

```bash
python record_gesture.py --geste wischen_rechts --samples 100
python record_gesture.py --geste wischen_links --samples 100
python record_gesture.py --geste kreis_uhrzeigersinn --samples 100
python record_gesture.py --geste kreis_gegen_uhrzeigersinn --samples 100
```

**Bedienung im Kamerafenster:**
- `s` drücken → 20 Frames werden aufgenommen (führe währenddessen die Geste aus)
- `q` drücken → Aufnahme für diese Geste beenden (auch vorzeitig möglich)

**Tipps für gute Trainingsdaten:**
- Führe die Geste **nicht immer identisch** aus — leicht unterschiedliches
  Tempo, unterschiedliche Position im Kamerabild, unterschiedliche
  Handgröße im Bild (näher/weiter von der Kamera).
- Nimm alle 4 Gesten unter **ähnlichen Lichtbedingungen** wie später bei
  der Nutzung auf.
- 100 Samples pro Geste ist ein guter Startwert (400 insgesamt). Bei
  schwacher Testgenauigkeit später: mehr Samples aufnehmen.
- Achte darauf, dass wirklich nur EINE Hand im Bild ist (`num_hands=1`).
- Du kannst das Skript mehrfach für dieselbe Geste aufrufen (z. B. an
  verschiedenen Tagen) — es hängt automatisch weitere Samples an,
  ohne vorhandene zu überschreiben.

**Warum 66 Werte pro Frame?**
3 Werte = absolute Handgelenk-Position (x, y, z) → erfasst die
*Bewegung/Trajektorie* der Hand (wichtig für Wischen/Kreisen).
21 × 3 Werte = alle Landmarks relativ zum Handgelenk → erfasst die
*Handform* (nicht zwingend nötig für diese 4 Gesten, aber nützlich,
falls ihr später weitere, formabhängige Gesten ergänzt).

## 2. Modell trainieren — `train_model.py`

Sobald alle 4 Ordner mit Daten gefüllt sind:

```bash
python train_model_old.py
```

Das Skript:
1. lädt alle `.npy`-Dateien aus `trainingsdaten/<geste>/`,
2. teilt sie in Train/Validation/Test (70/15/15),
3. trainiert ein LSTM (2 LSTM-Layer + Dropout + Dense-Head),
4. gibt Testgenauigkeit, Classification Report und Confusion Matrix aus,
5. speichert `gesture_model.h5`, `label_map.json` und `training_history.png`.

**Worauf achten:**
- Test-Accuracy sollte deutlich über der Zufallswahrscheinlichkeit
  liegen (bei 4 Klassen: > 25 %, realistisch solltet ihr > 90 %
  anstreben).
- Schau dir die **Confusion Matrix** an: welche Gesten werden
  verwechselt? Meistens sind es die beiden Kreisrichtungen
  untereinander oder die beiden Wischrichtungen untereinander.
  → Falls hohe Verwechslung: mehr/bessere Trainingsdaten für genau
  diese Gesten aufnehmen.
- `training_history.png` zeigt, ob das Modell over-/underfittet
  (z. B. Val-Loss steigt wieder → Overfitting).

## 3. Live-Erkennung + Aktionen — `live_inference.py`

```bash
python live_inference_old.py
```

Funktionsweise:
- Läuft kontinuierlich über die Webcam, sammelt einen **Sliding
  Window** der letzten 20 Frames (Feature-Vektor identisch zu Schritt 1).
- Sobald der Puffer voll ist, sagt das Modell die wahrscheinlichste
  Geste vorher.
- Nur wenn die **Konfidenz** über `KONFIDENZ_SCHWELLE` (Standard 0.85)
  liegt UND seit der letzten Aktion mindestens `COOLDOWN_SEKUNDEN`
  (Standard 1.0s) vergangen sind, wird die Aktion ausgelöst:

| Geste | Aktion |
|---|---|
| `wischen_rechts` | "Weiter" → `pyautogui.press("right")` |
| `wischen_links` | "Zurück" → `pyautogui.press("left")` |
| `kreis_uhrzeigersinn` | Scroll runter |
| `kreis_gegen_uhrzeigersinn` | Scroll hoch |

Passe die `fuehre_aktion_aus()`-Funktion an eure tatsächlichen
Zielaktionen an (z. B. andere Tasten, `pyautogui.hotkey(...)`, oder
Integration mit eurem bestehenden Virtual-Mouse-Code aus dem Paper).

## Typischer Workflow / Reihenfolge

```
1. record_gesture.py   (4x ausführen, einmal pro Geste)
        ↓
2. train_model.py      (einmal, danach bei Bedarf erneut nach mehr Daten)
        ↓
3. live_inference.py   (Testen/Nutzen; bei schlechter Erkennung zurück zu 1.)
```

## Nächste sinnvolle Schritte (Ausblick)

- **Integration mit dem statischen Gesten-Code aus dem Paper:** Beide
  Systeme können parallel laufen — statische Gesten (z. B. Klick) über
  die im Paper beschriebenen mathematischen Regeln, dynamische Gesten
  über dieses LSTM. Ein gemeinsamer State-Machine-Layer entscheidet,
  welches System gerade "das Sagen" hat.
- **Datenaugmentation:** leichtes Rauschen, Zeit-Skalierung (schneller/
  langsamer), Spiegelung, um die Robustheit zu erhöhen, ohne mehr
  Rohdaten aufnehmen zu müssen.
- **Alternative Architekturen:** GRU statt LSTM (weniger Parameter,
  oft ähnlich gut), oder 1D-CNN als Vergleich.
- **Cross-User-Test:** Testet mit Personen, deren Daten *nicht* im
  Trainingsset waren, um echte Generalisierung zu prüfen.
