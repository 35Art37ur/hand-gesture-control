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
import mlflow
import mlflow.keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

DATEN_ORDNER = "trainingsdaten"
MODELL_DATEI = "gesture_model.h5"
LABEL_MAP_DATEI = "label_map.json"

# --- MLflow Konfiguration ---
# Standard: lokaler Ordner "mlruns" im Projektverzeichnis. Fuer ein Team mit
# zentralem Tracking-Server stattdessen z.B.:
#   mlflow.set_tracking_uri("http://<server-ip>:5000")
MLFLOW_EXPERIMENT_NAME = "gesture_recognition_lstm"
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class MlflowEpochLogger(Callback):
    """Loggt nach jeder Epoche alle Metriken (loss/accuracy, train+val) an MLflow,
    damit der Trainingsverlauf in der MLflow-UI als Graph erscheint."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            mlflow.log_metric(name, float(value), step=epoch)

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


with mlflow.start_run():

    print("Lade Trainingsdaten...")
    X, y, label_map = lade_daten(DATEN_ORDNER)
    print(f"\nGesamt: {X.shape[0]} Samples, Sequenzlaenge {X.shape[1]}, Feature-Dim {X.shape[2]}")
    print(f"Klassen: {label_map}")

    anzahl_klassen = len(label_map)
    y_kategorisch = to_categorical(y, num_classes=anzahl_klassen)

    # Basisinfos zum Datensatz loggen
    mlflow.log_param("anzahl_samples_gesamt", X.shape[0])
    mlflow.log_param("sequenz_laenge", X.shape[1])
    mlflow.log_param("feature_dim", X.shape[2])
    mlflow.log_param("anzahl_klassen", anzahl_klassen)
    mlflow.log_param("klassen", list(label_map.values()))

    # -----------------------------------------------------------------------
    # 2. Train / Val / Test Split
    # -----------------------------------------------------------------------
    # Erst Test abtrennen, dann von Rest nochmal Validation abtrennen.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_kategorisch, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )

    print(f"\nTrain: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
    mlflow.log_param("anzahl_train_samples", X_train.shape[0])
    mlflow.log_param("anzahl_val_samples", X_val.shape[0])
    mlflow.log_param("anzahl_test_samples", X_test.shape[0])

    # -----------------------------------------------------------------------
    # 3. Modell definieren
    # -----------------------------------------------------------------------
    sequenz_laenge = X.shape[1]
    feature_dim = X.shape[2]

    lstm_units_1 = 64
    lstm_units_2 = 32
    dense_units = 32
    dropout_rate = 0.3
    epochs = 150
    batch_size = 16

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(sequenz_laenge, feature_dim)),
        LSTM(lstm_units_1, return_sequences=True, activation="tanh"),
        Dropout(dropout_rate),
        LSTM(lstm_units_2, return_sequences=False, activation="tanh"),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
        Dense(anzahl_klassen, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Hyperparameter loggen
    mlflow.log_params({
        "lstm_units_1": lstm_units_1,
        "lstm_units_2": lstm_units_2,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "epochs_max": epochs,
        "batch_size": batch_size,
        "optimizer": "adam",
    })

    # -----------------------------------------------------------------------
    # 4. Training
    # -----------------------------------------------------------------------
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ModelCheckpoint(MODELL_DATEI, monitor="val_accuracy", save_best_only=True),
        MlflowEpochLogger(),  # loggt jede Epoche einzeln -> Graphen in der MLflow-UI
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    mlflow.log_param("tatsaechliche_epochen", len(history.history["loss"]))

    # -----------------------------------------------------------------------
    # 5. Evaluation auf Testdaten
    # -----------------------------------------------------------------------
    print("\n--- Evaluation auf Testdaten ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test-Accuracy: {test_acc:.3f}  Test-Loss: {test_loss:.3f}")

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    zielnamen = [label_map[i] for i in range(anzahl_klassen)]

    report_text = classification_report(y_true, y_pred, target_names=zielnamen)
    print("\nClassification Report:")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix (Zeilen=wahr, Spalten=vorhergesagt):")
    print(zielnamen)
    print(cm)

    # Classification Report als Text-Artefakt loggen
    with open("classification_report.txt", "w") as f:
        f.write(report_text)
    mlflow.log_artifact("classification_report.txt")

    # Confusion Matrix als Grafik loggen
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(zielnamen)))
    ax_cm.set_yticks(range(len(zielnamen)))
    ax_cm.set_xticklabels(zielnamen, rotation=45, ha="right")
    ax_cm.set_yticklabels(zielnamen)
    ax_cm.set_xlabel("Vorhergesagt")
    ax_cm.set_ylabel("Wahr")
    ax_cm.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig_cm.colorbar(im, ax=ax_cm)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close(fig_cm)

    # -----------------------------------------------------------------------
    # 6. Speichern: Modell + Label-Map + Trainingsverlauf
    # -----------------------------------------------------------------------
    model.save(MODELL_DATEI)
    with open(LABEL_MAP_DATEI, "w") as f:
        json.dump(label_map, f, indent=2)

    mlflow.log_artifact(LABEL_MAP_DATEI)
    # Modell zusaetzlich als MLflow-Modell loggen -> Versionierung/Vergleich in der UI
    mlflow.keras.log_model(model, artifact_path="model")

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
    mlflow.log_artifact("training_history.png")
    plt.close(fig)

    print(f"\nFertig! Modell gespeichert als '{MODELL_DATEI}', Label-Map als '{LABEL_MAP_DATEI}'.")
    print("Trainingsverlauf als 'training_history.png' gespeichert.")
    print("\nAlle Metriken, Parameter und Artefakte wurden zusaetzlich an MLflow geloggt.")
    print("Starte 'mlflow ui' im Projektordner und oeffne http://localhost:5000 im Browser,")
    print("um den Trainingsverlauf grafisch zu sehen und mit frueheren Laeufen zu vergleichen.")