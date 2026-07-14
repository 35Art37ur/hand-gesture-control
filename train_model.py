"""
train_model.py
---------------
Trainiert und vergleicht MEHRERE unterschiedliche Modell-Architekturen auf
denselben Trainingsdaten, damit man objektiv sehen kann, welche Architektur
fuer diese Gesten am besten funktioniert.

Jeder Aufruf erzeugt einen neuen, fortlaufend nummerierten "Vergleichs-Batch":

    modelle/
        vergleich_001/
            gru_klein/
                gesture_model.h5
                label_map.json
                training_history.png
                confusion_matrix.png
                classification_report.txt
            lstm_mittel/
                ... (gleiche Dateien)
            cnn_gross/
                ... (gleiche Dateien)
            vergleich_zusammenfassung.csv   <- Tabelle aller Architekturen
            vergleich_chart.png             <- Balkendiagramm Test-Accuracy
            bestes_modell.txt                <- Name der besten Architektur
                                                 (wird von live_inference*.py
                                                 automatisch genutzt)
        vergleich_002/
            ...

METHODIK
--------
Datensplit: Ein einziger, stratifizierter 70/15/15 Train/Val/Test-Split wird
EINMAL berechnet und dann fuer ALLE Architekturen wiederverwendet. Nur so ist
der Vergleich fair -- Unterschiede in der Test-Accuracy koennen dann nur an
der Architektur liegen, nicht an einem zufaellig guenstigeren/ungueenstigeren
Split. Cross-Validation wuerde die Trainingszeit pro Architektur um den
Faktor k vervielfachen; bei mehreren zu vergleichenden Architekturen und
einem inzwischen mittelgrossen Datensatz (mehrere hundert Samples nach
Spiegelung) ist ein fixer Split zusammen mit Early Stopping der bessere
Kompromiss aus Aussagekraft und Trainingszeit.

Architekturen im Vergleich (bewusst unterschiedliche GROESSE und ART):
  - gru_klein   (~20k Parameter):  rekurrent, wenig Parameter
  - lstm_mittel (~47k Parameter):  rekurrent, bisherige Baseline-Architektur
  - cnn_gross   (~155k Parameter): Convolution statt Rekurrenz

MLflow-Organisation: Jeder Batch ist ein "Parent-Run", jede Architektur
darin ein "Child-Run" (nested=True). So lassen sich in der MLflow-UI sowohl
einzelne Architekturen vergleichen als auch ganze Batches gegeneinander.

Aufruf:
    python train_model.py

Nach dem Training MLflow-UI oeffnen:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    -> http://localhost:5000
"""

import argparse
import json
import os
import re
import time

import matplotlib
matplotlib.use("Agg")  # kein Display noetig, nur Datei speichern
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GRU,
    Input,
    LSTM,
    Masking,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from augmentation_generator import AugmentedSequence

parser = argparse.ArgumentParser(description="Trainiert und vergleicht mehrere Gesten-Modell-Architekturen.")
parser.add_argument("--daten-ordner", type=str, default="trainingsdaten",
                     help="Ordner mit den Trainingsdaten, z.B. trainingsdaten "
                          "oder trainingsdaten_relativ (Standard: trainingsdaten)")
parser.add_argument(
    "--no-augmentation",
    action="store_true",
    help="Deaktiviert die Datenaugmentation"
)
args = parser.parse_args()

DATEN_ORDNER = args.daten_ordner
AUGMENTIERUNG_AKTIV = not args.no_augmentation
MODELLE_BASIS_ORDNER = "modelle"

# --- MLflow Konfiguration ---
# MLflow >= 3.x setzt den reinen Datei-Tracking-Modus ("./mlruns") in den
# Maintenance Mode und verlangt standardmaessig ein Datenbank-Backend.
MLFLOW_EXPERIMENT_NAME = "gesture_recognition_vergleich"
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


class MlflowEpochLogger(Callback):
    """Loggt nach jeder Epoche alle Metriken (loss/accuracy, train+val) an MLflow."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            mlflow.log_metric(name, float(value), step=epoch)


# ---------------------------------------------------------------------------
# 0. Naechstes Vergleichs-Batch-Verzeichnis
# ---------------------------------------------------------------------------

def naechstes_vergleich_verzeichnis(basis_ordner):
    """Findet die naechste freie, fortlaufende Batch-Nummer (vergleich_001,
    vergleich_002, ...) und erstellt das zugehoerige Verzeichnis."""
    os.makedirs(basis_ordner, exist_ok=True)

    vorhandene_nummern = []
    for name in os.listdir(basis_ordner):
        pfad = os.path.join(basis_ordner, name)
        if os.path.isdir(pfad):
            match = re.fullmatch(r"vergleich_(\d+)", name)
            if match:
                vorhandene_nummern.append(int(match.group(1)))

    naechste_nummer = max(vorhandene_nummern, default=0) + 1
    ordner_name = f"vergleich_{naechste_nummer:03d}"
    ziel_pfad = os.path.join(basis_ordner, ordner_name)
    os.makedirs(ziel_pfad, exist_ok=False)
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
# 2. Modell-Architekturen (drei unterschiedliche Typen UND Groessen)
# ---------------------------------------------------------------------------
# 1) GRU (klein):    Rekurrent wie LSTM, aber nur 3 statt 4 "Gates" pro Zelle
#                     -> weniger Parameter, oft schneller trainiert, kann bei
#                     kleineren Datensaetzen sogar besser generalisieren.
# 2) LSTM (mittel):   Die bisher verwendete, bereits validierte Architektur
#                     (~93% Testgenauigkeit in frueheren Laeufen) -- dient als
#                     Referenzpunkt/Baseline fuer den Vergleich.
# 3) 1D-CNN (gross):  Voellig andere Herangehensweise: lokale Bewegungsmuster
#                     werden per Convolution statt sequenziellem Gedaechtnis
#                     erkannt. Kein rekurrenter Zustand noetig -> vollstaendig
#                     parallelisierbar, meist deutlich schneller trainiert,
#                     und fuer kurze, fest laengige Bewegungssequenzen wie
#                     unsere 20-Frame-Gesten oft ueberraschend konkurrenzfaehig.
#
# Bewusst NICHT gewaehlt:
# - Bidirektionales LSTM: haette nur die bestehende LSTM-Idee vergroessert,
#   ohne eine wirklich neue Modellfamilie ins Rennen zu bringen.
# - Transformer/Attention: braucht erfahrungsgemaess deutlich mehr Trainings-
#   daten als die paar hundert Samples pro Geste, die hier vorliegen --
#   hohes Overfitting-Risiko und viel Tuning-Aufwand ohne klaren Mehrwert.
# - Reines Dense/MLP (Flatten): ignoriert die Zeitstruktur der Bewegung
#   komplett, waere v.a. als Sanity-Check interessant, nicht als ernsthafter
#   Kandidat fuer die beste Loesung.

def baue_modell_gru_klein(sequenz_laenge, feature_dim, anzahl_klassen):
    modell = Sequential([
        Input(shape=(sequenz_laenge, feature_dim)),
        Masking(mask_value=0.0), # was mit fehlenden werten gemacht wird. Gibt es bei unsnicht
        GRU(32, return_sequences=True, activation="tanh"), # Gru schichten sind der standard fuer Sequenzen 48 ist die Anzahl der Neuronen in der Schicht
        Dropout(0.35), #30% der Neuronen werden deactiviert
        GRU(16, return_sequences=False, activation="tanh"), #zweite GRU-schicht mit 24 layern
        Dropout(0.35),
        Dense(16, activation="relu"), # erste Vostaendig verbundene schicht
        Dense(anzahl_klassen, activation="softmax"), # Ausgabeschicht
    ], name="gru_klein")

    hyperparameter = {
        "architektur": "GRU",
        "layer_1_units": 32,
        "layer_2_units": 16,
        "dense_units": 16,
        "dropout_rate": 0.35,
    }
    return modell, hyperparameter

def baue_modell_gru_mittel(sequenz_laenge, feature_dim, anzahl_klassen):
    modell = Sequential([
        Input(shape=(sequenz_laenge, feature_dim)),
        Masking(mask_value=0.0), # was mit fehlenden werten gemacht wird. Gibt es bei unsnicht
        GRU(32, return_sequences=True, activation="tanh"), # Gru schichten sind der standard fuer Sequenzen 48 ist die Anzahl der Neuronen in der Schicht
        Dropout(0.45), #30% der Neuronen werden deactiviert
        GRU(16, return_sequences=False, activation="tanh"), #zweite GRU-schicht mit 24 layern
        Dropout(0.45),
        Dense(16, activation="relu"), # erste Vostaendig verbundene schicht
        Dense(anzahl_klassen, activation="softmax"), # Ausgabeschicht
    ], name="gru_klein")

    hyperparameter = {
        "architektur": "GRU",
        "layer_1_units": 32,
        "layer_2_units": 16,
        "dense_units": 16,
        "dropout_rate": 0.4,
    }
    return modell, hyperparameter

def baue_modell_gru_gross(sequenz_laenge, feature_dim, anzahl_klassen):
    modell = Sequential([
        Input(shape=(sequenz_laenge, feature_dim)),
        Masking(mask_value=0.0), # was mit fehlenden werten gemacht wird. Gibt es bei unsnicht
        GRU(32, return_sequences=True, activation="tanh"), # Gru schichten sind der standard fuer Sequenzen 48 ist die Anzahl der Neuronen in der Schicht
        Dropout(0.5), #30% der Neuronen werden deactiviert
        GRU(16, return_sequences=False, activation="tanh"), #zweite GRU-schicht mit 24 layern
        Dropout(0.5),
        Dense(16, activation="relu"), # erste Vostaendig verbundene schicht
        Dense(anzahl_klassen, activation="softmax"), # Ausgabeschicht
    ], name="gru_klein")

    hyperparameter = {
        "architektur": "GRU",
        "layer_1_units": 32,
        "layer_2_units": 16,
        "dense_units": 16,
        "dropout_rate": 0.5,
    }
    return modell, hyperparameter

def baue_modell_lstm(sequenz_laenge, feature_dim, anzahl_klassen):
    modell = Sequential([
        Input(shape=(sequenz_laenge, feature_dim)),
        Masking(mask_value=0.0),
        LSTM(128, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(128, return_sequences=False, activation="tanh"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(anzahl_klassen, activation="softmax"),
    ], name="lstm_mittel")

    hyperparameter = {
        "architektur": "LSTM",
        "layer_1_units": 128,
        "layer_2_units": 128,
        "dense_units": 64,
        "dropout_rate": 0.3,
    }
    return modell, hyperparameter


def baue_modell_cnn(sequenz_laenge, feature_dim, anzahl_klassen):
    # Hinweis: Conv1D unterstuetzt kein Keras-Masking, ist hier aber auch
    # nicht noetig, da alle aufgenommenen Sequenzen bereits exakt gleich
    # lang sind (kein Zero-Padding im Datensatz vorhanden).
    modell = Sequential([
        Input(shape=(sequenz_laenge, feature_dim)),
        Conv1D(filters=256, kernel_size=3, padding="same", activation="relu"),
        Conv1D(filters=512, kernel_size=3, padding="same", activation="relu"),
        GlobalAveragePooling1D(),
        Dropout(0.4),
        Dense(265, activation="relu"),
        Dense(anzahl_klassen, activation="softmax"),
    ], name="cnn_gross")

    hyperparameter = {
        "architektur": "1D-CNN",
        "conv1_filters": 265,
        "conv2_filters": 512,
        "kernel_size": 3,
        "dense_units": 265,
        "dropout_rate": 0.4,
    }
    return modell, hyperparameter


# Registry: neue Architekturen koennen hier einfach ergaenzt werden.
MODELL_VARIANTEN = {
    "GRU_augment_0": (baue_modell_gru_mittel, 0.0005, 16, 20),
    #"GRU_pat_30": (baue_modell_gru_mittel, 0.0005, 16, 30),
    #"GRU_pat_40": (baue_modell_gru_mittel, 0.0005, 16, 40),
    #"lstm": baue_modell_lstm,
    #"cnn": baue_modell_cnn,
}


# ---------------------------------------------------------------------------
# 3. Eine einzelne Architektur trainieren + evaluieren
# ---------------------------------------------------------------------------

def trainiere_und_evaluiere(architektur_name, builder_fn, learning_rate, batch_size, patience, X_train, y_train,
                             X_val, y_val, X_test, y_test, label_map, sub_ordner):
    os.makedirs(sub_ordner, exist_ok=True)
    modell_datei = os.path.join(sub_ordner, "gesture_model.h5")
    label_map_datei = os.path.join(sub_ordner, "label_map.json")
    history_datei = os.path.join(sub_ordner, "training_history.png")
    cm_datei = os.path.join(sub_ordner, "confusion_matrix.png")
    report_datei = os.path.join(sub_ordner, "classification_report.txt")

    anzahl_klassen = len(label_map)
    sequenz_laenge = X_train.shape[1]
    feature_dim = X_train.shape[2]

    print(f"\n{'=' * 70}\nTrainiere Architektur: {architektur_name}\n{'=' * 70}")

    modell, hyperparameter = builder_fn(sequenz_laenge, feature_dim, anzahl_klassen)
    modell.compile(optimizer=Adam(learning_rate=learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    mlflow.log_param("learning_rate", learning_rate)
    modell.summary()

    anzahl_parameter = modell.count_params()
    mlflow.set_tag("architektur", architektur_name)
    mlflow.log_params(hyperparameter)
    mlflow.log_param("anzahl_modell_parameter", anzahl_parameter)

    callbacks = [
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,),
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(modell_datei, monitor="val_accuracy", save_best_only=True),
        MlflowEpochLogger(),
    ]

    train_generator = AugmentedSequence(
    X_train,
    y_train,
    batch_size=batch_size,
    augment=False
    )

    val_generator = AugmentedSequence(
        X_val,
        y_val,
        batch_size=batch_size,
        augment=False,
        shuffle=False
    )

    startzeit = time.time()
    history = modell.fit(
        train_generator,
        validation_data=val_generator,
        epochs=300,
        callbacks=callbacks,
        verbose=2,
    )

    endzeit = time.time()
    trainingszeit = endzeit - startzeit
    mlflow.log_metric("trainingszeit_sekunden", trainingszeit)

    mlflow.log_param("tatsaechliche_epochen", len(history.history["loss"]))

    test_loss, test_acc = modell.evaluate(X_test, y_test, verbose=0)
    print(f"[{architektur_name}] Test-Accuracy: {test_acc:.3f}  "
          f"Test-Loss: {test_loss:.3f}  Parameter: {anzahl_parameter}")
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    y_pred = np.argmax(modell.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    zielnamen = [label_map[i] for i in range(anzahl_klassen)]
    alle_label_indices = list(range(anzahl_klassen))

    report_text = classification_report(
        y_true, y_pred, labels=alle_label_indices, target_names=zielnamen, zero_division=0
    )
    with open(report_datei, "w") as f:
        f.write(report_text)
    mlflow.log_artifact(report_datei)

    cm = confusion_matrix(y_true, y_pred, labels=alle_label_indices)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    im = ax_cm.imshow(cm, cmap="Blues")
    ax_cm.set_xticks(range(len(zielnamen)))
    ax_cm.set_yticks(range(len(zielnamen)))
    ax_cm.set_xticklabels(zielnamen, rotation=45, ha="right")
    ax_cm.set_yticklabels(zielnamen)
    ax_cm.set_xlabel("Vorhergesagt")
    ax_cm.set_ylabel("Wahr")
    ax_cm.set_title(f"Confusion Matrix ({architektur_name})")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig_cm.colorbar(im, ax=ax_cm)
    plt.tight_layout()
    plt.savefig(cm_datei)
    mlflow.log_artifact(cm_datei)
    plt.close(fig_cm)

    modell.save(modell_datei)
    with open(label_map_datei, "w") as f:
        json.dump(label_map, f, indent=2)
    mlflow.log_artifact(label_map_datei)
    mlflow.keras.log_model(modell, artifact_path="model")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title(f"Accuracy ({architektur_name})")
    axes[0].set_xlabel("Epoche")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title(f"Loss ({architektur_name})")
    axes[1].set_xlabel("Epoche")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(history_datei)
    mlflow.log_artifact(history_datei)
    plt.close(fig)

    return {
        "architektur": architektur_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "patience": patience,
        "anzahl_parameter": anzahl_parameter,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "epochen": len(history.history["loss"]),
        "trainingszeit": trainingszeit,
        "ordner": sub_ordner,
    }


# ---------------------------------------------------------------------------
# Hauptablauf
# ---------------------------------------------------------------------------
batch_ordner, batch_name, batch_nummer = naechstes_vergleich_verzeichnis(MODELLE_BASIS_ORDNER)
print(f"Neuer Vergleichs-Batch: {batch_ordner}\n")

print("Lade Trainingsdaten...")
X, y, label_map = lade_daten(DATEN_ORDNER)
anzahl_klassen = len(label_map)
print(f"\nGesamt: {X.shape[0]} Samples, Sequenzlaenge {X.shape[1]}, Feature-Dim {X.shape[2]}")
print(f"Klassen: {label_map}")

y_kategorisch = to_categorical(y, num_classes=anzahl_klassen)

# Ein einziger, stratifizierter Split fuer ALLE Architekturen (siehe Docstring
# oben fuer die Begruendung, warum 70/15/15 statt Cross-Validation).
X_train, X_test, y_train, y_test = train_test_split(
    X, y_kategorisch, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42,
    stratify=np.argmax(y_train, axis=1)
)
print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

ergebnisse = []

with mlflow.start_run(run_name=batch_name):
    mlflow.set_tag("batch_ordner", batch_ordner)
    mlflow.log_param("anzahl_samples_gesamt", X.shape[0])
    mlflow.log_param("anzahl_klassen", anzahl_klassen)
    mlflow.log_param("klassen", list(label_map.values()))
    mlflow.log_param("anzahl_train_samples", X_train.shape[0])
    mlflow.log_param("anzahl_val_samples", X_val.shape[0])
    mlflow.log_param("anzahl_test_samples", X_test.shape[0])
    mlflow.log_param("augmentation_aktiv", AUGMENTIERUNG_AKTIV)
    mlflow.log_param("architekturen_im_vergleich", list(MODELL_VARIANTEN.keys()))

    for architektur_name, (builder_fn, learning_rate, batch_size, patience) in MODELL_VARIANTEN.items():
        sub_ordner = os.path.join(batch_ordner, architektur_name)
        with mlflow.start_run(run_name=f"{batch_name}_{architektur_name}", nested=True):
            ergebnis = trainiere_und_evaluiere(
                architektur_name, builder_fn, learning_rate, batch_size, patience,
                X_train, y_train, X_val, y_val, X_test, y_test,
                label_map, sub_ordner
            )
            ergebnisse.append(ergebnis)

    # --- Vergleichs-Zusammenfassung ueber alle Architekturen ---
    ergebnisse_sortiert = sorted(ergebnisse, key=lambda r: r["test_accuracy"], reverse=True)
    bestes = ergebnisse_sortiert[0]

    zusammenfassung_zeilen = ["architektur,anzahl_parameter,test_accuracy,test_loss,epochen"]
    for r in ergebnisse_sortiert:
        zusammenfassung_zeilen.append(
            f"{r['architektur']},{r['anzahl_parameter']},"
            f"{r['test_accuracy']:.4f},{r['test_loss']:.4f},{r['epochen']}"
        )
    zusammenfassung_csv = os.path.join(batch_ordner, "vergleich_zusammenfassung.csv")
    with open(zusammenfassung_csv, "w") as f:
        f.write("\n".join(zusammenfassung_zeilen))
    mlflow.log_artifact(zusammenfassung_csv)

    # Balkendiagramm: Test-Accuracy je Architektur (bestes Modell hervorgehoben)
    fig_vgl, ax_vgl = plt.subplots(figsize=(7, 4))
    namen = [r["architektur"] for r in ergebnisse_sortiert]
    werte = [r["test_accuracy"] for r in ergebnisse_sortiert]
    farben = ["#2ca02c" if r is bestes else "#1f77b4" for r in ergebnisse_sortiert]
    balken = ax_vgl.bar(namen, werte, color=farben)
    ax_vgl.set_ylabel("Test-Accuracy")
    ax_vgl.set_ylim(0, 1.05)
    ax_vgl.set_title(f"Architektur-Vergleich ({batch_name})")
    for balken_einzeln, wert in zip(balken, werte):
        ax_vgl.text(balken_einzeln.get_x() + balken_einzeln.get_width() / 2, wert + 0.02,
                    f"{wert:.2f}", ha="center")
    plt.tight_layout()
    vergleich_chart_datei = os.path.join(batch_ordner, "vergleich_chart.png")
    plt.savefig(vergleich_chart_datei)
    mlflow.log_artifact(vergleich_chart_datei)
    plt.close(fig_vgl)

    mlflow.log_metric("bestes_modell_test_accuracy", bestes["test_accuracy"])
    mlflow.set_tag("bestes_modell", bestes["architektur"])

    # Marker-Datei: live_inference*.py liest das automatisch aus, um ohne
    # weitere Angabe das beste Modell dieses Batches zu verwenden.
    with open(os.path.join(batch_ordner, "bestes_modell.txt"), "w") as f:
        f.write(bestes["architektur"])

    print(f"\n{'=' * 70}")
    print("ZUSAMMENFASSUNG")
    print(f"{'=' * 70}")
    for r in ergebnisse_sortiert:
        markierung = "  <-- BESTES MODELL" if r is bestes else ""
        print(
            f"{r['architektur']:15s}  "
            f"Test-Acc: {r['test_accuracy']:.3f}  "
            f" LR: {r['learning_rate']:.0e}  "
            f"Patience: {r['patience']:>3}  "
            f"Parameter: {r['anzahl_parameter']:>7}  "
            f"Epochen: {r['epochen']:>3}  "
            f"Trainingszeit: {r['trainingszeit']:>7.2f}s"
            f"{markierung}"
        )
    print(f"\nAlle Modelle gespeichert unter: {batch_ordner}")
    print(f"Empfohlenes Modell fuer live_inference*.py: {bestes['architektur']}")
    print("\nStarte 'mlflow ui --backend-store-uri sqlite:///mlflow.db' und oeffne")
    print("http://localhost:5000, um alle Architekturen dieses und frueherer")
    print("Batches im Detail zu vergleichen (Parent-Run = Batch, Child-Runs = Architekturen).")
