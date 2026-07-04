"""
train_model.py
---------------
Laedt alle aufgenommenen .npy-Sequenzen aus 'trainingsdaten/<geste>/',
baut daraus Trainings-/Validierungs-/Testdaten und trainiert ein LSTM,
das die dynamischen Gesten klassifiziert.

Jeder Aufruf dieses Skripts erzeugt ein NEUES, fortlaufend nummeriertes
Modellverzeichnis, sodass fruehere Modelle nicht ueberschrieben werden:

    modelle/
        model_001/
            gesture_model.h5
            label_map.json
            training_history.png
            confusion_matrix.png
            classification_report.txt
        model_002/
            ...

Alle Laeufe landen ausserdem als eigene, klar benannte Runs im selben
MLflow-Experiment ("gesture_recognition_lstm"), sodass sie in der
MLflow-UI direkt nebeneinander verglichen werden koennen (Metriken,
Hyperparameter, Confusion Matrix, Modell-Artefakt).

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

Nach dem Training MLflow-UI oeffnen:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    -> http://localhost:5000
"""

import json
import os
import re

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
MODELLE_BASIS_ORDNER = "modelle"

# --- MLflow Konfiguration ---
# MLflow >= 3.x setzt den reinen Datei-Tracking-Modus ("./mlruns") in den
# Maintenance Mode und verlangt standardmaessig ein Datenbank-Backend.
# Wir nutzen daher eine lokale SQLite-Datei statt des reinen Dateiordners.
MLFLOW_EXPERIMENT_NAME = "gesture_recognition_lstm"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class MlflowEpochLogger(Callback):
    """Loggt nach jeder Epoche alle Metriken (loss/accuracy, train+val) an MLflow,
    damit der Trainingsverlauf in der MLflow-UI als Graph erscheint."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            mlflow.log_metric(name, float(value), step=epoch)


# ---------------------------------------------------------------------------
# 0. Naechste fortlaufende Modell-Nummer ermitteln
# ---------------------------------------------------------------------------

def naechstes_modell_verzeichnis(basis_ordner):
    """Findet die naechste freie, fortlaufende Nummer (model_001, model_002, ...)
    und erstellt das zugehoerige Verzeichnis."""
    os.makedirs(basis_ordner, exist_ok=True)

    vorhandene_nummern = []
    for name in os.listdir(basis_ordner):
        pfad = os.path.join(basis_ordner, name)
        if os.path.isdir(pfad):
            match = re.fullmatch(r"model_(\d+)", name)
            if match:
                vorhandene_nummern.append(int(match.group(1)))

    naechste_nummer = max(vorhandene_nummern, default=0) + 1
    ordner_name = f"model_{naechste_nummer:03d}"
    ziel_pfad = os.path.join(basis_ordner, ordner_name)
    os.makedirs(ziel_pfad, exist_ok=False)  # soll nie existieren, sonst Bug in der Nummerierung
    return ziel_pfad, ordner_name, naechste_nummer


# ---------------------------------------------------------------------------
# 1. Daten laden
# ---------------------------------------------------------------------------

def lade_daten(daten_ordner):
    alle_unterordner = sorted([
        d for d in os.listdir(daten_ordner)
        if os.path.isdir(os.path.join(daten_ordner, d))
    ])

    # Nur Ordner beruecksichtigen, die auch tatsaechlich .npy-Dateien enthalten.
    # Filtert z.B. IDE-Ordner wie ".idea" oder leere/versehentliche Ordner heraus.
    gesten = []
    for d in alle_unterordner:
        pfad = os.path.join(daten_ordner, d)
        hat_npy = any(f.endswith(".npy") for f in os.listdir(pfad))
        if hat_npy:
            gesten.append(d)
        else:
            print(f"  Hinweis: Ordner '{d}' enthaelt keine .npy-Dateien, wird ignoriert.")

    if len(gesten) == 0:
        raise RuntimeError(
            f"Keine Gesten-Ordner mit .npy-Dateien in '{daten_ordner}' gefunden. "
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


# ---------------------------------------------------------------------------
# Modellverzeichnis fuer diesen Lauf anlegen
# ---------------------------------------------------------------------------
ziel_ordner, ordner_name, modell_nummer = naechstes_modell_verzeichnis(MODELLE_BASIS_ORDNER)
MODELL_DATEI = os.path.join(ziel_ordner, "gesture_model.h5")
LABEL_MAP_DATEI = os.path.join(ziel_ordner, "label_map.json")
TRAINING_HISTORY_DATEI = os.path.join(ziel_ordner, "training_history.png")
CONFUSION_MATRIX_DATEI = os.path.join(ziel_ordner, "confusion_matrix.png")
CLASSIFICATION_REPORT_DATEI = os.path.join(ziel_ordner, "classification_report.txt")

print(f"Neues Modellverzeichnis: {ziel_ordner}\n")

# Der MLflow-Run-Name entspricht dem Modellordner, damit man in der UI sofort
# sieht, welcher Run zu welchem gespeicherten Modell auf der Festplatte gehoert.
with mlflow.start_run(run_name=ordner_name):

    mlflow.set_tag("modell_ordner", ziel_ordner)
    mlflow.log_param("modell_nummer", modell_nummer)

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
    alle_label_indices = list(range(anzahl_klassen))

    # labels=alle_label_indices sorgt dafuer, dass auch Klassen beruecksichtigt
    # werden, die im (kleinen) Test-Split zufaellig nicht vorkommen.
    report_text = classification_report(
        y_true, y_pred, labels=alle_label_indices, target_names=zielnamen, zero_division=0
    )
    print("\nClassification Report:")
    print(report_text)

    cm = confusion_matrix(y_true, y_pred, labels=alle_label_indices)
    print("Confusion Matrix (Zeilen=wahr, Spalten=vorhergesagt):")
    print(zielnamen)
    print(cm)

    # Classification Report als Text-Artefakt loggen
    with open(CLASSIFICATION_REPORT_DATEI, "w") as f:
        f.write(report_text)
    mlflow.log_artifact(CLASSIFICATION_REPORT_DATEI)

    # Confusion Matrix als Grafik loggen
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(zielnamen)))
    ax_cm.set_yticks(range(len(zielnamen)))
    ax_cm.set_xticklabels(zielnamen, rotation=45, ha="right")
    ax_cm.set_yticklabels(zielnamen)
    ax_cm.set_xlabel("Vorhergesagt")
    ax_cm.set_ylabel("Wahr")
    ax_cm.set_title(f"Confusion Matrix ({ordner_name})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig_cm.colorbar(im, ax=ax_cm)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_DATEI)
    mlflow.log_artifact(CONFUSION_MATRIX_DATEI)
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
    axes[0].set_title(f"Accuracy ({ordner_name})")
    axes[0].set_xlabel("Epoche")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title(f"Loss ({ordner_name})")
    axes[1].set_xlabel("Epoche")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(TRAINING_HISTORY_DATEI)
    mlflow.log_artifact(TRAINING_HISTORY_DATEI)
    plt.close(fig)

    print(f"\nFertig! Modell gespeichert unter: {ziel_ordner}")
    print(f"  - {MODELL_DATEI}")
    print(f"  - {LABEL_MAP_DATEI}")
    print(f"  - {TRAINING_HISTORY_DATEI}")
    print(f"  - {CONFUSION_MATRIX_DATEI}")
    print(f"  - {CLASSIFICATION_REPORT_DATEI}")
    print(f"\nMLflow-Run-Name: '{ordner_name}' (im Experiment '{MLFLOW_EXPERIMENT_NAME}')")
    print("Starte 'mlflow ui --backend-store-uri sqlite:///mlflow.db' und oeffne")
    print("http://localhost:5000, um alle bisherigen Laeufe nebeneinander zu vergleichen.")