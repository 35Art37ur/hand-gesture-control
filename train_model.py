"""
train_model.py
---------------
Laedt alle aufgenommenen .npy-Sequenzen aus 'trainingsdaten/<geste>/',
baut daraus Trainings-/Validierungs-/Testdaten und trainiert ein LSTM,
das die dynamischen Gesten klassifiziert.

Erwartete Ordnerstruktur (von record_gesture.py erzeugt):

    trainingsdaten/
        wischen_rechts/          sample_0.npy, sample_1.npy, ...
        wischen_links/           ...
        kreis_uhrzeigersinn/     ...
        kreis_gegen_uhrzeigersinn/ ...

Jede .npy-Datei hat die Form (SEQUENZ_LAENGE, 66):
    - 66 = 3 (absolute Wrist-Position) + 21*3 (relative Landmark-Position)

Aufruf:
    python train_model.py

Ergebnis:
    - gesture_model.h5      (trainiertes Keras-Modell)
    - label_map.json        (Mapping Index -> Gestenname, wird von live_inference.py gebraucht)
    - training_history.png  (Plot von Accuracy/Loss, zur Kontrolle)
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # kein Display noetig, nur Datei speichern
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

DATEN_ORDNER = "trainingsdaten"
MODELL_DATEI = "gesture_model.h5"
LABEL_MAP_DATEI = "label_map.json"

# ---------------------------------------------------------------------------
# 1. Daten laden
# ---------------------------------------------------------------------------

def lade_daten(daten_ordner):
    gesten = sorted([
        d for d in os.listdir(daten_ordner)
        if os.path.isdir(os.path.join(daten_ordner, d))
    ])
    if len(gesten) == 0:
        raise RuntimeError(
            f"Keine Gesten-Ordner in '{daten_ordner}' gefunden. "
            f"Bitte zuerst record_gesture.py fuer jede Geste ausfuehren."
        )

    label_map = {i: name for i, name in enumerate(gesten)}
    name_to_idx = {name: i for i, name in label_map.items()}

    X, y = [], []
    for geste in gesten:
        ordner = os.path.join(daten_ordner, geste)
        dateien = [f for f in os.listdir(ordner) if f.endswith(".npy")]
        print(f"  {geste}: {len(dateien)} Samples")
        for datei in dateien:
            pfad = os.path.join(ordner, datei)
            sequenz = np.load(pfad)
            X.append(sequenz)
            y.append(name_to_idx[geste])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, label_map


print("Lade Trainingsdaten...")
X, y, label_map = lade_daten(DATEN_ORDNER)
print(f"\nGesamt: {X.shape[0]} Samples, Sequenzlaenge {X.shape[1]}, Feature-Dim {X.shape[2]}")
print(f"Klassen: {label_map}")

anzahl_klassen = len(label_map)
y_kategorisch = to_categorical(y, num_classes=anzahl_klassen)

# ---------------------------------------------------------------------------
# 2. Train / Val / Test Split
# ---------------------------------------------------------------------------
# Erst Test abtrennen, dann von Rest nochmal Validation abtrennen.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_kategorisch, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42,
    stratify=np.argmax(y_train, axis=1)
)

print(f"\nTrain: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

# ---------------------------------------------------------------------------
# 3. Modell definieren
# ---------------------------------------------------------------------------
sequenz_laenge = X.shape[1]
feature_dim = X.shape[2]

model = Sequential([
    Masking(mask_value=0.0, input_shape=(sequenz_laenge, feature_dim)),
    LSTM(64, return_sequences=True, activation="tanh"),
    Dropout(0.3),
    LSTM(32, return_sequences=False, activation="tanh"),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(anzahl_klassen, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ModelCheckpoint(MODELL_DATEI, monitor="val_accuracy", save_best_only=True),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=16,
    callbacks=callbacks,
    verbose=2,
)

# ---------------------------------------------------------------------------
# 5. Evaluation auf Testdaten
# ---------------------------------------------------------------------------
print("\n--- Evaluation auf Testdaten ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test-Accuracy: {test_acc:.3f}  Test-Loss: {test_loss:.3f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
zielnamen = [label_map[i] for i in range(anzahl_klassen)]

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=zielnamen))

print("Confusion Matrix (Zeilen=wahr, Spalten=vorhergesagt):")
print(zielnamen)
print(confusion_matrix(y_true, y_pred))

# ---------------------------------------------------------------------------
# 6. Speichern: Modell + Label-Map + Trainingsverlauf
# ---------------------------------------------------------------------------
model.save(MODELL_DATEI)
with open(LABEL_MAP_DATEI, "w") as f:
    json.dump(label_map, f, indent=2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["accuracy"], label="train")
axes[0].plot(history.history["val_accuracy"], label="val")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoche")
axes[0].legend()

axes[1].plot(history.history["loss"], label="train")
axes[1].plot(history.history["val_loss"], label="val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoche")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_history.png")

print(f"\nFertig! Modell gespeichert als '{MODELL_DATEI}', Label-Map als '{LABEL_MAP_DATEI}'.")
print("Trainingsverlauf als 'training_history.png' gespeichert.")
