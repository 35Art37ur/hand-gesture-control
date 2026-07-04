"""
visualize_dataset.py
--------------------

Zeigt gespeicherte Trainingsdaten als Animation an.

Beispiele:

python visualize_dataset.py --ordner wischen_links

python visualize_dataset.py --ordner wischen_links --anzahl 5 --start 20

python visualize_dataset.py --ordner wischen_links --anzahl 10 --start -1
    -> zeigt 10 zufällige Samples
"""

import argparse
import os
import random

import cv2
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("--ordner", required=True,
                    help="Ordner innerhalb von trainingsdaten")

parser.add_argument("--anzahl", type=int, default=5,
                    help="Wie viele Samples angezeigt werden sollen")

parser.add_argument("--start", type=int, default=0,
                    help="Startindex. Negativ = zufällige Samples")

args = parser.parse_args()


input_folder = os.path.join("trainingsdaten", args.ordner)

if not os.path.isdir(input_folder):
    raise FileNotFoundError(input_folder)

dateien = sorted(
    [f for f in os.listdir(input_folder) if f.endswith(".npy")]
)

if len(dateien) == 0:
    raise RuntimeError("Keine npy-Dateien gefunden.")

# ------------------------------------------------------------
# Welche Dateien anzeigen?
# ------------------------------------------------------------

if args.start < 0:
    indices = random.sample(
        range(len(dateien)),
        min(args.anzahl, len(dateien))
    )
else:
    ende = min(args.start + args.anzahl, len(dateien))
    indices = list(range(args.start, ende))

print("Anzuzeigende Samples:")
for i in indices:
    print(f"  {dateien[i]}")

# ------------------------------------------------------------

WIDTH = 800
HEIGHT = 800

SCALE = 500

for idx in indices:

    daten = np.load(os.path.join(input_folder, dateien[idx]))

    print(f"\nAnzeige: {dateien[idx]}")

    for frame_nr, frame in enumerate(daten):

        img = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

        wrist_x = frame[0]
        wrist_y = frame[1]

        punkte = []

        for lm in range(21):

            start = 3 + lm * 3

            rel_x = frame[start]
            rel_y = frame[start + 1]

            abs_x = wrist_x + rel_x
            abs_y = wrist_y + rel_y

            x = int(abs_x * SCALE + WIDTH // 2 - SCALE // 2)
            y = int(abs_y * SCALE + HEIGHT // 2 - SCALE // 2)

            punkte.append((x, y))

        # Handverbindungen (MediaPipe)
        verbindungen = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]

        for a, b in verbindungen:
            cv2.line(img, punkte[a], punkte[b], (0,180,0), 2)

        for i, p in enumerate(punkte):
            cv2.circle(img, p, 5, (0,0,255), -1)
            cv2.putText(img, str(i), (p[0]+5, p[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0,0,0),
                        1)

        cv2.putText(
            img,
            f"{dateien[idx]}",
            (20,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            2
        )

        cv2.putText(
            img,
            f"Frame {frame_nr+1}/{len(daten)}",
            (20,65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,0),
            2
        )

        cv2.imshow("Dataset Viewer", img)

        key = cv2.waitKey(80)

        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()

    print("Beliebige Taste -> nächstes Sample | q -> Ende")

    key = cv2.waitKey(0)

    if key == ord('q'):
        break

cv2.destroyAllWindows()