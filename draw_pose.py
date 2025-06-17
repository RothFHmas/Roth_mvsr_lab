##QUELLE: https://github.com/megapose6d/megapose6d/issues/52
# ANGEPASST

import cv2
import numpy as np
import json
from scipy.spatial.transform import Rotation

# Bildpfad
image_path = "megapose6d/local_data/examples/morobot/image_rgb.png"
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not read image from {image_path}")
    exit(1)

# Kameraparameter laden
with open("megapose6d/local_data/examples/morobot/camera_data.json", "r") as f:
    cam_data = json.load(f)
K = np.array(cam_data["K"]).reshape(3, 3)

# Objekt-Pose-Daten laden
with open("megapose6d/local_data/examples/morobot/outputs/object_data.json", "r") as f:
    obj_data = json.load(f)

# Boxmaße (m) pro Objektlabel – individuell angepasst
box_dims = {
    "1A_gray":   (0.01045, 0.04, 0.0505),  # half_width, half_depth, half_height
    "1A_yellow": (0.01045, 0.04, 0.0505),
    "1B_yellow": (0.008, 0.04, 0.0505),
    "3B_gray":   (0.04, 0.008, 0.06),
}

# Kanten der Box
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Unterseite
    (4, 5), (5, 6), (6, 7), (7, 4),  # Oberseite
    (0, 4), (1, 5), (2, 6), (3, 7)   # Seiten
]

for obj in obj_data:
    label = obj["label"]
    TWO = obj["TWO"]
    quaternion = np.array(TWO[0])
    translation = np.array(TWO[1])

    # Passende Boxmaße abrufen
    if label not in box_dims:
        print(f"[WARN] Keine Boxmaße für {label} definiert – überspringe.")
        continue
    half_width, half_depth, half_height = box_dims[label]

    box_3d = np.array([
        [-half_width, -half_depth, -half_height],
        [ half_width, -half_depth, -half_height],
        [ half_width,  half_depth, -half_height],
        [-half_width,  half_depth, -half_height],
        [-half_width, -half_depth,  half_height],
        [ half_width, -half_depth,  half_height],
        [ half_width,  half_depth,  half_height],
        [-half_width,  half_depth,  half_height],
    ])

    # Rotation & Transformation
    rot = Rotation.from_quat(quaternion)
    R = rot.as_matrix()
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = translation

    # Punkte transformieren
    box_transformed = (transform[:3, :3] @ box_3d.T).T + transform[:3, 3]

    # Projektion
    box_2d = []
    for pt in box_transformed:
        proj = K @ pt
        proj /= proj[2]
        box_2d.append(proj[:2])
    box_2d = np.array(box_2d, dtype=np.int32)

    # Box zeichnen
    for edge in edges:
        start = tuple(box_2d[edge[0]])
        end = tuple(box_2d[edge[1]])
        cv2.line(img, start, end, (0, 255, 0), 2)

    # Koordinatenachsen
    origin = transform[:3, 3]
    axes_length = 0.05
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Z, Y, X (BGR)

    origin_2d = K @ origin
    origin_2d = (int(origin_2d[0] / origin_2d[2]), int(origin_2d[1] / origin_2d[2]))

    for i, color in enumerate(colors):
        axis_end = origin + axes_length * transform[:3, i]
        axis_2d = K @ axis_end
        axis_2d = (int(axis_2d[0] / axis_2d[2]), int(axis_2d[1] / axis_2d[2]))
        cv2.line(img, origin_2d, axis_2d, color, 2)

    # Label
    label_pos = (box_2d[0][0], box_2d[0][1] - 10)
    cv2.putText(img, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Bild speichern & anzeigen
output_path = "megapose6d/local_data/examples/morobot/visualizations/all_poses.png"
cv2.imwrite(output_path, img)
print(f"Visualisierung gespeichert unter: {output_path}")

'''
cv2.imshow("Alle Objekte – Pose Visualisierung", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)

print("Visualisierung beendet.")
'''