"""
pca_tsne_visualisierung.py
-----------------------------
Untersucht (Data-Mining-Perspektive: explorative Datenanalyse, EDA), ob sich
die aufgenommenen Gesten-Klassen im Feature-Raum ueberhaupt trennen lassen --
unabhaengig von einem konkreten Modell. Fuenf ergaenzende Analysen:

  1. PCA (Principal Component Analysis): linear, schnell, zeigt die
     Hauptvarianzrichtungen der Rohdaten.
  2. t-SNE: nichtlinear, meist deutlich klarere/kompaktere Cluster, aber
     rechenintensiver. Nutzt eine PCA-Vorstufe gegen hohe Dimensionalitaet.
  3. Silhouette-Analyse: QUANTITATIVES Mass (nicht nur visuell) dafuer, wie
     gut jede Klasse von den anderen abgegrenzt ist -- direkt im vollen
     Feature-Raum berechnet, nicht nur in der 2D-Projektion.
  4. Klassen-Distanz-Heatmap: paarweiser Abstand der Klassen-Zentroiden --
     zeigt, welche zwei Gesten sich im Feature-Raum am aehnlichsten sind
     (= wahrscheinlichste Verwechslungskandidaten, vergleichbar mit der
     Confusion Matrix aus dem trainierten Modell).
  5. Random-Forest-Feature-Importance: beantwortet eine ANDERE Frage als
     1-4 -- nicht "wie gut trennbar sind die Klassen insgesamt", sondern
     "WELCHE konkreten Merkmale (Wrist-Trajektorie, einzelne Landmarken,
     Pinch-Feature) tragen am meisten zur Trennung bei". Die 20 Zeitschritte
     werden dabei pro Merkmalstyp aufsummiert, damit die Grafik lesbar
     bleibt (z.B. "Daumenspitze_X" statt "Spalte 847").

WICHTIG: Dies ersetzt NICHT den Modellvergleich in train_model.py (das ist
Hold-out-Validierung, um das beste Modell zu ermitteln). Diese
Visualisierung beantwortet eine andere, vorgelagerte Frage: "Enthalten die
Rohdaten ueberhaupt genug Information, um die Gesten grundsaetzlich zu
unterscheiden?"

Aufruf:
    analyse_dataset.py
    analyse_dataset.py --daten-ordner trainingsdaten_newGesture
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

# MediaPipe Standard-Handmodell (21 Landmarken), fuer lesbare Feature-Namen
LANDMARK_NAMEN = [
    "Wrist", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_TIP",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_TIP",
]


def basis_feature_namen(feature_dim):
    """Baut fuer jede der 'feature_dim' Spalten EINES Frames einen lesbaren
    Namen -- Reihenfolge muss exakt der in record_gesture(_relativ/_newGesture).py
    entsprechen: [0:3]=Wrist-Trajektorie, [3:66]=21 Landmarken x(X,Y,Z),
    [66]=Pinch-Feature (nur falls feature_dim==67)."""
    namen = ["Wrist_X (Trajektorie)", "Wrist_Y (Trajektorie)", "Wrist_Z (Trajektorie)"]
    for landmark_name in LANDMARK_NAMEN:
        for achse in ["X", "Y", "Z"]:
            namen.append(f"{landmark_name}_{achse} (relativ)")
    if feature_dim >= 67:
        namen.append("Pinch-Feature")
    return namen[:feature_dim]


def lade_daten(daten_ordner):
    """Identisch zur Ladefunktion in train_model.py."""
    alle_unterordner = sorted([
        d for d in os.listdir(daten_ordner)
        if os.path.isdir(os.path.join(daten_ordner, d))
    ])

    gesten = []
    for d in alle_unterordner:
        pfad = os.path.join(daten_ordner, d)
        hat_npy = any(f.endswith(".npy") for f in os.listdir(pfad))
        if hat_npy:
            gesten.append(d)
        else:
            print(f"  Hinweis: Ordner '{d}' enthaelt keine .npy-Dateien, wird ignoriert.")

    if len(gesten) == 0:
        raise RuntimeError(f"Keine Gesten-Ordner mit .npy-Dateien in '{daten_ordner}' gefunden.")

    label_map = {i: name for i, name in enumerate(gesten)}
    name_to_idx = {name: i for i, name in label_map.items()}

    X, y = [], []
    for geste in gesten:
        ordner = os.path.join(daten_ordner, geste)
        dateien = [f for f in os.listdir(ordner) if f.endswith(".npy")]
        print(f"  {geste}: {len(dateien)} Samples")
        for datei in dateien:
            sequenz = np.load(os.path.join(ordner, datei))
            X.append(sequenz)
            y.append(name_to_idx[geste])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y, label_map


parser = argparse.ArgumentParser(description="PCA/t-SNE-Visualisierung der Gesten-Klassentrennbarkeit.")
parser.add_argument("--daten-ordner", type=str, default="trainingsdaten_newGesture",
                    help="Zu analysierender Datenordner (Standard: trainingsdaten_newGesture)")
parser.add_argument("--ausgabe", type=str, default="pca_tsne_visualisierung.png",
                    help="Dateiname fuer die gespeicherte Grafik")
parser.add_argument("--pca-vorstufe-dim", type=int, default=50,
                    help="Anzahl Dimensionen der PCA-Vorstufe vor t-SNE (Standard: 50)")
args = parser.parse_args()

print(f"Lade Daten aus '{args.daten_ordner}'...")
X, y, label_map = lade_daten(args.daten_ordner)
anzahl_klassen = len(label_map)
n_samples = X.shape[0]
print(f"\nGesamt: {n_samples} Samples, Sequenzlaenge {X.shape[1]}, Feature-Dim {X.shape[2]}")
print(f"Klassen: {label_map}")

# --- Jede Sequenz zu einem einzigen Punkt zusammenfassen (flatten) ---
X_flach = X.reshape(n_samples, -1)  # (N, seq_laenge * feature_dim)
print(f"\nGeflachte Feature-Dimension pro Sample: {X_flach.shape[1]}")

# --- Ausreisser-Diagnose ---
# Ein einzelner extremer Wert kann PCA/t-SNE komplett dominieren (Skala wird
# von Ausreissern gestreckt, der Rest wirkt dadurch "auf einem Fleck"
# zusammengestaucht). Deshalb zuerst pruefen, ob es solche Werte gibt --
# typischer Kandidat: das Pinch-Feature, da es eine Division enthaelt und
# bei kurzzeitig fehlerhafter Handerkennung explodieren kann.
abs_max_pro_sample = np.abs(X_flach).max(axis=1)
schwellwert = np.percentile(abs_max_pro_sample, 99)
ausreisser_idx = np.where(abs_max_pro_sample > schwellwert * 3)[0]
if len(ausreisser_idx) > 0:
    print(f"\nWARNUNG: {len(ausreisser_idx)} Sample(s) mit auffaellig extremen Werten gefunden "
          f"(vermutlich kurzzeitige Fehlerkennung, z.B. beim Pinch-Feature):")
    for idx in ausreisser_idx[:10]:
        print(f"  Sample-Index {idx} (Klasse '{label_map[int(y[idx])]}'): "
              f"max. Absolutwert = {abs_max_pro_sample[idx]:.2f}")
    if len(ausreisser_idx) > 10:
        print(f"  ... und {len(ausreisser_idx) - 10} weitere.")
    print("  Diese Samples werden aus der Visualisierung ausgeschlossen (nicht geclippt), "
          "damit die Skalierung fuer den Rest der Daten unverzerrt bleibt.")
    print("  Tipp: Die zugrundeliegenden Aufnahmen ggf. auch pruefen/loeschen, sie koennten "
          "ebenso das Training stoeren.")
else:
    print("\nKeine auffaelligen Ausreisser gefunden.")

# Erkannte Ausreisser aus dem Datensatz entfernen, statt sie zu clippen --
# so bleibt die Skalierung/Verteilung der uebrigen (unauffaelligen) Samples
# unverzerrt, was insbesondere PCA zugutekommt (Clipping haette sonst
# kuenstliche "Waende" bei den Clip-Grenzen erzeugt).
maske_ok = np.ones(n_samples, dtype=bool)
maske_ok[ausreisser_idx] = False
X_flach = X_flach[maske_ok]
y = y[maske_ok]
n_samples = X_flach.shape[0]
print(f"\nVerbleibende Samples nach Ausschluss: {n_samples}")

# --- Skalierung: fuer PCA und t-SNE unterschiedlich, da beide Verfahren
# unterschiedlich auf schiefe Verteilungen reagieren.
# PCA: StandardScaler (Mittelwert/Std) -- gibt eine unverzerrte, direkt
#      interpretierbare Projektion der tatsaechlichen Varianzstruktur.
# t-SNE: RobustScaler (Median/Interquartilsabstand) -- t-SNE reagiert
#      empfindlich auf schiefe/langschwaenzige Verteilungen einzelner
#      Features; RobustScaler daempft das, ohne wie hartes Clipping
#      kuenstliche "Waende" in den Daten zu erzeugen.
X_skaliert_pca = StandardScaler().fit_transform(X_flach)
X_skaliert_tsne = RobustScaler().fit_transform(X_flach)


# --- 1. PCA (linear, schnell) ---
print("\nBerechne PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_skaliert_pca)
erklaerte_varianz = pca.explained_variance_ratio_
print(f"Erklaerte Varianz: PC1={erklaerte_varianz[0]*100:.1f}%, "
      f"PC2={erklaerte_varianz[1]*100:.1f}%, Summe={erklaerte_varianz.sum()*100:.1f}%")

# --- 2. t-SNE (nichtlinear), mit PCA-Vorstufe gegen hohe Dimensionalitaet ---
vorstufe_dim = min(args.pca_vorstufe_dim, n_samples - 1, X_skaliert_tsne.shape[1])
print(f"\nBerechne PCA-Vorstufe auf {vorstufe_dim} Dimensionen fuer t-SNE...")
X_vorstufe = PCA(n_components=vorstufe_dim, random_state=42).fit_transform(X_skaliert_tsne)

perplexity = min(30, max(5, n_samples // 10))
print(f"Berechne t-SNE (perplexity={perplexity})... das kann etwas dauern.")
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
X_tsne = tsne.fit_transform(X_vorstufe)

# --- Plot: PCA und t-SNE nebeneinander, farblich nach Geste ---
fig, (ax_pca, ax_tsne) = plt.subplots(1, 2, figsize=(18, 9))

cmap = plt.get_cmap("tab10" if anzahl_klassen <= 10 else "tab20")

for klassen_idx, geste_name in label_map.items():
    maske = y == klassen_idx
    farbe = cmap(klassen_idx % cmap.N)
    ax_pca.scatter(X_pca[maske, 0], X_pca[maske, 1], label=geste_name, color=farbe, s=20, alpha=0.7)
    ax_tsne.scatter(X_tsne[maske, 0], X_tsne[maske, 1], label=geste_name, color=farbe, s=20, alpha=0.7)

ax_pca.set_title(f"PCA (erklaerte Varianz: {erklaerte_varianz.sum()*100:.1f}%)")
ax_pca.set_xlabel("Hauptkomponente 1")
ax_pca.set_ylabel("Hauptkomponente 2")

ax_tsne.set_title(f"t-SNE (perplexity={perplexity})")
ax_tsne.set_xlabel("t-SNE Dimension 1")
ax_tsne.set_ylabel("t-SNE Dimension 2")

# Legenden-Zeilen abschaetzen (mehrere Klassen -> mehrzeilige Legende), damit
# genug Platz reserviert wird und nichts mit der x-Achsen-Beschriftung
# ueberlappt.
legende_spalten = min(anzahl_klassen, 4)
legende_zeilen = -(-anzahl_klassen // legende_spalten)  # aufgerundete Ganzzahldivision
unterer_rand = 0.08 + 0.05 * legende_zeilen

handles, labels = ax_pca.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=legende_spalten,
           bbox_to_anchor=(0.5, 0.0), fontsize=11)

plt.tight_layout(rect=[0, unterer_rand, 1, 1])
plt.savefig(args.ausgabe, dpi=150)
print(f"\nGespeichert: {args.ausgabe}")
print("\nInterpretationshilfe (PCA/t-SNE):")
print("- Deutlich getrennte, kompakte Farbcluster -> Klassen sind im Feature-Raum gut trennbar,")
print("  ein Modell sollte damit gut klassifizieren koennen.")
print("- Ueberlappende Cluster (v.a. zwischen zwei bestimmten Gesten) -> diese Gesten sind sich")
print("  in den Rohdaten aehnlich und werden vom Modell vermutlich haeufiger verwechselt")
print("  (Abgleich mit der Confusion Matrix lohnt sich hier).")

# ---------------------------------------------------------------------------
# 3. Silhouette-Analyse: quantitatives Mass fuer Klassentrennbarkeit
# ---------------------------------------------------------------------------
# Der Silhouette-Koeffizient vergleicht pro Sample den mittleren Abstand zu
# Punkten der EIGENEN Klasse mit dem mittleren Abstand zur naechstgelegenen
# ANDEREN Klasse. Wertebereich -1 bis 1:
#   > 0.5  deutliche, klare Trennung
#   0.25-0.5  moderate/schwache, aber vorhandene Trennung
#   < 0.25  kaum/keine Struktur erkennbar
#   < 0     Sample liegt naeher an einer anderen Klasse als an der eigenen
# Wichtig: wird im VOLLEN skalierten Feature-Raum berechnet (nicht nur in
# der 2D-Projektion) -- zeigt also mehr als PCA/t-SNE allein sichtbar machen.
print(f"\n{'=' * 60}\nSILHOUETTE-ANALYSE\n{'=' * 60}")
silhouette_werte = silhouette_samples(X_skaliert_pca, y)
gesamt_silhouette = silhouette_score(X_skaliert_pca, y)
print(f"Durchschnittlicher Silhouette-Score (gesamt): {gesamt_silhouette:.3f}")

fig_sil, ax_sil = plt.subplots(figsize=(10, 0.6 * n_samples / anzahl_klassen + 2))
y_pos = 0
klassen_mittelwerte = {}
for klassen_idx in sorted(label_map.keys()):
    geste_name = label_map[klassen_idx]
    werte_dieser_klasse = np.sort(silhouette_werte[y == klassen_idx])
    klassen_mittelwerte[geste_name] = werte_dieser_klasse.mean()
    farbe = cmap(klassen_idx % cmap.N)
    ax_sil.barh(range(y_pos, y_pos + len(werte_dieser_klasse)), werte_dieser_klasse,
                height=1.0, color=farbe, edgecolor="none")
    ax_sil.text(-0.05, y_pos + len(werte_dieser_klasse) / 2, geste_name,
                ha="right", va="center", fontsize=10)
    y_pos += len(werte_dieser_klasse) + 5  # kleiner Abstand zwischen Klassen

ax_sil.axvline(gesamt_silhouette, color="red", linestyle="--",
               label=f"Durchschnitt gesamt ({gesamt_silhouette:.3f})")
ax_sil.set_xlabel("Silhouette-Koeffizient")
ax_sil.set_yticks([])
ax_sil.set_title("Silhouette-Analyse pro Geste (hoeher = besser trennbar)")
ax_sil.legend(loc="lower right")
plt.tight_layout()
silhouette_datei = "silhouette_analyse.png"
plt.savefig(silhouette_datei, dpi=150)
print(f"Gespeichert: {silhouette_datei}")

print("\nDurchschnittlicher Silhouette-Score je Geste:")
for geste_name, wert in sorted(klassen_mittelwerte.items(), key=lambda x: x[1]):
    print(f"  {geste_name:30s}: {wert:.3f}")

# ---------------------------------------------------------------------------
# 4. Klassen-Distanz-Heatmap: welche Gesten sind sich am aehnlichsten?
# ---------------------------------------------------------------------------
print(f"\n{'=' * 60}\nKLASSEN-DISTANZ-HEATMAP\n{'=' * 60}")
zentroiden = np.array([
    X_skaliert_pca[y == klassen_idx].mean(axis=0)
    for klassen_idx in sorted(label_map.keys())
])
gesten_namen_sortiert = [label_map[i] for i in sorted(label_map.keys())]

distanz_matrix = np.zeros((anzahl_klassen, anzahl_klassen))
for i in range(anzahl_klassen):
    for j in range(anzahl_klassen):
        distanz_matrix[i, j] = np.linalg.norm(zentroiden[i] - zentroiden[j])

fig_dist, ax_dist = plt.subplots(figsize=(8, 7))
im = ax_dist.imshow(distanz_matrix, cmap="viridis_r")
ax_dist.set_xticks(range(anzahl_klassen))
ax_dist.set_yticks(range(anzahl_klassen))
ax_dist.set_xticklabels(gesten_namen_sortiert, rotation=45, ha="right")
ax_dist.set_yticklabels(gesten_namen_sortiert)
ax_dist.set_title("Paarweiser Abstand der Klassen-Zentroiden\n(dunkler = aehnlicher = eher verwechselbar)")
for i in range(anzahl_klassen):
    for j in range(anzahl_klassen):
        farbe_text = "white" if distanz_matrix[i, j] < distanz_matrix.max() / 2 else "black"
        ax_dist.text(j, i, f"{distanz_matrix[i, j]:.0f}", ha="center", va="center",
                     color=farbe_text, fontsize=8)
fig_dist.colorbar(im, ax=ax_dist, label="Euklidischer Abstand")
plt.tight_layout()
heatmap_datei = "klassen_distanz_heatmap.png"
plt.savefig(heatmap_datei, dpi=150)
print(f"Gespeichert: {heatmap_datei}")

# Naehestes Klassenpaar (ohne Diagonale) automatisch ausgeben
np.fill_diagonal(distanz_matrix, np.inf)
naechstes_paar_idx = np.unravel_index(np.argmin(distanz_matrix), distanz_matrix.shape)
print(f"\nAehnlichstes Klassenpaar (potenziell am ehesten verwechselbar):")
print(f"  '{gesten_namen_sortiert[naechstes_paar_idx[0]]}' <-> "
      f"'{gesten_namen_sortiert[naechstes_paar_idx[1]]}' "
      f"(Abstand: {distanz_matrix[naechstes_paar_idx]:.1f})")
print("  Tipp: Mit der Confusion Matrix des trainierten Modells abgleichen --")
print("  wird genau dieses Paar dort auch am haeufigsten verwechselt?")

# ---------------------------------------------------------------------------
# 5. Random-Forest-Feature-Importance: welche Merkmale tragen am meisten bei?
# ---------------------------------------------------------------------------
# Andere Fragestellung als 1-4: nicht "wie trennbar insgesamt", sondern
# "welche konkreten Merkmale tragen am meisten zur Trennung bei". Random
# Forest ist skaleninvariant (Baummodell), daher wird hier direkt auf den
# ungeskalierten, aber bereits ausreisserbereinigten Daten gearbeitet.
print(f"\n{'=' * 60}\nRANDOM-FOREST-FEATURE-IMPORTANCE\n{'=' * 60}")

feature_dim = X.shape[2]
sequenz_laenge = X.shape[1]
basis_namen = basis_feature_namen(feature_dim)

# Kurzer Sanity-Check: erreicht ein simpler, nicht-sequenzieller Random-
# Forest-Baseline ueberhaupt brauchbare Genauigkeit auf den geflatteten
# Daten? (Kein Ersatz fuer train_model.py, nur ein zusaetzlicher Hinweis
# auf vorhandene Klasseninformation in den Rohdaten.)
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_flach, y, test_size=0.2, random_state=42, stratify=y
)
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train_rf, y_train_rf)
rf_accuracy = accuracy_score(y_test_rf, rf.predict(X_test_rf))
print(f"Random-Forest-Baseline-Accuracy (Hold-out, geflattete Daten, "
      f"OHNE zeitliche Struktur): {rf_accuracy:.3f}")
print("(Nur ein Plausibilitaets-Check, kein Ersatz fuer den eigentlichen")
print(" Modellvergleich in train_model.py, der die zeitliche Reihenfolge nutzt.)")

# Fuer die eigentliche Feature-Importance auf ALLEN (bereinigten) Daten
# trainieren, um die stabilste Schaetzung zu bekommen.
rf_voll = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_voll.fit(X_flach, y)
importances = rf_voll.feature_importances_  # Laenge: sequenz_laenge * feature_dim

# Ueber alle 20 Frames pro Basis-Merkmal aufsummieren, damit die Grafik
# lesbar bleibt (z.B. "Daumenspitze_X" statt 20 einzelne Balken dafuer).
importance_pro_basis_feature = np.zeros(feature_dim)
for frame in range(sequenz_laenge):
    start = frame * feature_dim
    ende = start + feature_dim
    importance_pro_basis_feature += importances[start:ende]

reihenfolge = np.argsort(importance_pro_basis_feature)[::-1]
top_n = min(15, feature_dim)
top_namen = [basis_namen[i] for i in reihenfolge[:top_n]]
top_werte = importance_pro_basis_feature[reihenfolge[:top_n]]

fig_rf, ax_rf = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
farben_rf = [cmap(i % cmap.N) for i in range(top_n)]
ax_rf.barh(range(top_n), top_werte[::-1], color=farben_rf[::-1])
ax_rf.set_yticks(range(top_n))
ax_rf.set_yticklabels(top_namen[::-1])
ax_rf.set_xlabel("Aufsummierte Feature-Importance (ueber alle 20 Frames)")
ax_rf.set_title(f"Top {top_n} wichtigste Merkmale (Random Forest)")
plt.tight_layout()
rf_datei = "feature_importance_random_forest.png"
plt.savefig(rf_datei, dpi=150)
print(f"\nGespeichert: {rf_datei}")

print(f"\nTop {top_n} Merkmale nach Wichtigkeit:")
for name, wert in zip(top_namen, top_werte):
    print(f"  {name:30s}: {wert:.4f}")