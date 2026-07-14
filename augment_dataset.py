"""
augment_dataset.py
------------------

Augmentiert einen einzelnen Gestendatensatz durch

- zufällige Translation
- zufälligen Zoom

Zum Testen kann eine .npy-Datei geladen werden, welche anschließend
vorher/nachher nebeneinander angezeigt wird.

Beispiel:

python augment_dataset.py trainingsdaten/wischen_links/sample_0.npy
"""

import argparse
import cv2
import numpy as np


# -------------------------------------------------------------
# Einstellungen
# -------------------------------------------------------------

WIDTH = 600
HEIGHT = 600
SCALE = 400


# -------------------------------------------------------------
# Augmentation
# -------------------------------------------------------------

def augment_sequence(sequence,
                     min_zoom=0.95,
                     max_zoom=1.05):

    augmented = sequence.copy()


    zoom = np.random.uniform(min_zoom, max_zoom)

    for lm in range(21):
        idx = 3 + lm * 3

        augmented[:, idx] *= zoom
        augmented[:, idx + 1] *= zoom
        augmented[:, idx + 2] *= zoom

    augmented[:, 0] = np.clip(augmented[:, 0], 0.0, 1.0)
    augmented[:, 1] = np.clip(augmented[:, 1], 0.0, 1.0)

    return augmented, zoom


# -------------------------------------------------------------
# Zeichnen
# -------------------------------------------------------------

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


def draw_frame(img, frame):

    wrist_x = frame[0]
    wrist_y = frame[1]

    points = []

    for lm in range(21):

        idx = 3 + lm * 3

        rel_x = frame[idx]
        rel_y = frame[idx + 1]

        abs_x = wrist_x + rel_x
        abs_y = wrist_y + rel_y

        x = int(abs_x * SCALE + WIDTH / 2 - SCALE / 2)
        y = int(abs_y * SCALE + HEIGHT / 2 - SCALE / 2)

        points.append((x, y))

    for a, b in CONNECTIONS:
        cv2.line(img, points[a], points[b], (0, 180, 0), 2)

    for p in points:
        cv2.circle(img, p, 5, (0, 0, 255), -1)


# -------------------------------------------------------------
# Anzeige
# -------------------------------------------------------------

def visualize(original, augmented, zoom):

    frames = original.shape[0]

    for i in range(frames):

        left = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        right = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255

        draw_frame(left, original[i])
        draw_frame(right, augmented[i])

        cv2.putText(left,
                    "Original",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0),
                    2)

        cv2.putText(right,
                    "Augmentiert",
                    (20,35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0),
                    2)
        cv2.putText(right,
            "Augmentiert",
            (20,35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,0),
            2)

        cv2.putText(right,
                    f"Zoom : {zoom:.3f}",
                    (20,75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2)

        combined = np.hstack((left, right))

        cv2.imshow("Augmentation", combined)

        key = cv2.waitKey(80)

        if key == ord("q"):
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datei",
        help="Pfad zu einer .npy-Datei"
    )

    args = parser.parse_args()

    sequence = np.load(args.datei)

    augmented, zoom = augment_sequence(sequence)

    visualize(sequence, augmented, zoom)


if __name__ == "__main__":
    main()